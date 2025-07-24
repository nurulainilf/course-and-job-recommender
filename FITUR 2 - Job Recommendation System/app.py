# 1. --- Import Library ---
from flask import Flask, request, jsonify
import pandas as pd
import re
import string
import nltk
import os
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # Tetap dibutuhkan untuk cosine similarity

# 2. --- Flask App Initialization ---
app = Flask(__name__)

# --- 3. Konfigurasi Lokasi Model dan Data yang Telah Disimpan ---
MODEL_DIR = 'recommender_models' 

# Path dan DataFrame yang dibutuhkan
SBERT_EMBEDDINGS_PATH = os.path.join(MODEL_DIR, 'sbert_embeddings.pkl')
SBERT_MODEL_DIR_PATH = os.path.join(MODEL_DIR, 'sbert_model_dir') # Folder tempat S-BERT model disimpan
DF_RECOMMENDATION_PATH = os.path.join(MODEL_DIR, 'df_for_recommendation.pkl')

# --- 4. Inisialisasi Komponen Preprocessing (Harus sama persis dengan di model_training_jobs.py) ---
try:
    _ = nltk.data.find('corpora/stopwords')
    _ = nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Peringatan: NLTK resources (stopwords, punkt) tidak ditemukan. Pastikan sudah diunduh.")

factory_stemmer_api = StemmerFactory()
stemmer_api = factory_stemmer_api.create_stemmer()

factory_stopword_api = StopWordRemoverFactory()
stopword_remover_api = factory_stopword_api.create_stop_word_remover()

stop_words_en = set(stopwords.words('english'))
stop_words_id = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'atau', 'pada', 'ini', 'itu',
    'sebagai', 'jika', 'ada', 'karena', 'saat', 'oleh', 'juga', 'agar', 'dalam', 'tidak',
    'adalah', 'bahwa', 'maupun', 'bagi', 'dapat', 'telah', 'sudah', 'lebih', 'harus',
    'setiap', 'kita', 'kami', 'saya', 'anda', 'mereka', 'semua', 'bisa', 'masih',
    'akan', 'dan', 'pun', 'hingga', 'dengan', 'berikut', 'namun', 'sehingga',
    'yaitu', 'yakni', 'yakin', 'yang', 'yapp', 'yup', 'yah', 'yak', 'serta', 'punya',
    'apakah', 'bagaimana', 'kenapa', 'mengapa', 'dimana', 'kapan', 'siapa', 'berapa', 'mana'
])
stop_words_all_api = stop_words_en.union(stop_words_id)


# --- 5. Fungsi Preprocessing ---

def tahap1_clean_api(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tahap3_tokenisasi_api(text):
    return word_tokenize(text)

def tahap4_stopword_api(tokens):
    return [word for word in tokens if word not in stop_words_all_api and len(word) > 2]

def stemming_job_api(token_list):
    return [stemmer_api.stem(word) for word in token_list]

def cleaning_pipeline_api(text):
    if not isinstance(text, str):
        return ""
    text = tahap1_clean_api(text)
    text = text.lower()
    tokens = tahap3_tokenisasi_api(text)
    tokens = tahap4_stopword_api(tokens)
    stemmed = stemming_job_api(tokens)
    return ' '.join(stemmed)


# 6. --- Global Variables for Loaded Models ---
sbert_model_loaded = None
sbert_embeddings_loaded = None
df_for_recommendation_loaded = None

# 7. --- Model Loading Function with @app.before_first_request ---
#    This function will be registered and called by the 'app' object.
def load_models_and_data():
    global sbert_model_loaded, sbert_embeddings_loaded, df_for_recommendation_loaded

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        return # Skip if it's the reloader process (only run in the main process)

    print("Memuat model dan data untuk API...")
    try:
        sbert_model_loaded = SentenceTransformer(SBERT_MODEL_DIR_PATH)
        print("S-BERT Model berhasil dimuat.")

        with open(SBERT_EMBEDDINGS_PATH, 'rb') as f:
            sbert_embeddings_loaded = pickle.load(f)
        print("S-BERT Embeddings lowongan berhasil dimuat.")

        with open(DF_RECOMMENDATION_PATH, 'rb') as f:
            df_for_recommendation_loaded = pickle.load(f)
        print("DataFrame untuk rekomendasi berhasil dimuat.")

        print("Semua model dan data berhasil dimuat.")
    except FileNotFoundError as e:
        print(f"Error: File model tidak ditemukan ({e}). Pastikan Anda sudah menjalankan model_training_jobs.py dan semua file model ada di '{MODEL_DIR}'.")
        os._exit(1) 
    except Exception as e:
        print(f"Error tidak terduga saat memuat model: {e}")
        os._exit(1)


# --- 8. Fungsi Rekomendasi (Menggunakan S-BERT) ---
def get_job_recommendations_sbert_only(course_description, num_recommendations=5):
    processed_course_text = cleaning_pipeline_api(course_description)
    
    if not processed_course_text:
        print("Peringatan: Deskripsi kursus kosong setelah preprocessing. Tidak dapat memberikan rekomendasi.")
        return pd.DataFrame()

    # Pastikan model sudah dimuat (seharusnya selalu True karena dimuat di global scope startup)
    if sbert_model_loaded is None or sbert_embeddings_loaded is None:
        print("Error: S-BERT model atau embeddings belum dimuat di API. Ada masalah saat startup.")
        return pd.DataFrame()
    
    course_vector = sbert_model_loaded.encode([processed_course_text])[0]
    sim_scores = cosine_similarity(course_vector.reshape(1, -1), sbert_embeddings_loaded).flatten()
    
    if sim_scores.size == 0 or np.all(sim_scores == 0):
        print("Peringatan: Skor kemiripan semuanya nol atau kosong. Tidak ada rekomendasi yang kuat.")
        return pd.DataFrame()

    top_indices = sim_scores.argsort()[-num_recommendations:][::-1]

    top_indices = [idx for idx in top_indices if sim_scores[idx] > 0]
    
    if not top_indices:
        return pd.DataFrame()

    recommended_jobs = df_for_recommendation_loaded.iloc[top_indices].copy()
    recommended_jobs['Similarity_Score'] = sim_scores[top_indices]

    return recommended_jobs[['Job Title', 'Company', 'Job Description', 'Qualification', 'Benefit', 'Similarity_Score']]


# --- 9. Endpoint API ---
@app.route('/recommend_job_by_course', methods=['POST'])
def recommend_job_by_course_endpoint():
    data = request.get_json(silent=True) 

    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    course_desc = data.get('course_description')
    num_rec = data.get('num_recommendations', 5)

    if not course_desc:
        return jsonify({"error": "Missing 'course_description' in request body."}), 400
    
    recommended_df = get_job_recommendations_sbert_only(
        course_desc,
        num_rec
    )

    if recommended_df.empty:
        return jsonify({"recommendations": []}), 200 

    recommendations_list = recommended_df.to_dict(orient='records')
    
    return jsonify({"recommendations": recommendations_list})

@app.route('/', methods=['GET'])
def home_api():
    return """
    <h1>Sistem Rekomendasi Lowongan Kerja (Menggunakan S-BERT)</h1>
    <p>Gunakan endpoint <code>/recommend_job_by_course</code> dengan metode POST untuk mendapatkan rekomendasi.</p>
    <p>Contoh request body (JSON):</p>
    <pre><code>
    {
        "course_description": "3D Interaction Design in Virtual Reality",
        "num_recommendations": 3
    }
    </code></pre>
    """

# 10. --- Run Flask App ---
if __name__ == '__main__':
    load_models_and_data()
    print("Memulai server Flask...")
    # debug=True akan merestart server saat ada perubahan kode. Hapus saat deployment.
    # host='0.0.0.0' agar bisa diakses dari jaringan lokal
    app.run(debug=True, host='0.0.0.0', port=5000)