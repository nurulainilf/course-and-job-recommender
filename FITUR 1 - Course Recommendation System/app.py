from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Init Flask
app = Flask(__name__)

# Load model, data and embeddings
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentence_model')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'courses_data.json')
EMBEDDING_PATH = os.path.join(BASE_DIR, 'course_embeddings.pkl')

model = SentenceTransformer(MODEL_PATH)
df = pd.read_json(DATA_PATH, orient='records', encoding='utf-8-sig')
embeddings = pickle.load(open(EMBEDDING_PATH, 'rb'))

# Function to compute similarity
def get_top_similar_courses(input_embedding, base_df, base_embeddings, top_n=10):
    similarities = cosine_similarity(input_embedding, base_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    
    result_df = base_df.iloc[top_indices][['Title', 'Institution', 'Type', 'Level', 'Duration', 'Category', 'Subcategory', 'Rating', 'Description', 'Skills', 'Modules Name', 'Modules Description']].copy()
    result_df['Similarity'] = similarities[top_indices]
    
    return result_df

# Course recommendation API
@app.route('/recommend_course', methods=['POST'])
def recommend_courses():
    data = request.json
    input_skill = data.get('skill', [])
    input_category = data.get('category', [])

    if isinstance(input_skill, list):
        input_skill = ', '.join(input_skill)

    if isinstance(input_category, str):
        input_category = [input_category]

    input_text = f"{input_skill}. {' '.join(input_category)}".lower().strip()
    input_embedding = model.encode([input_text])
    
    # Filter by category if provided
    if input_category:
        mask = df['Category'].str.lower().apply(
            lambda cat: any(ic.lower() in cat for ic in input_category if isinstance(cat, str))
        )
        filtered_df = df[mask].copy()
        filtered_embeddings = embeddings[filtered_df.index]

        # Fallback to full data if no match
        if filtered_df.empty:
            filtered_df = df.copy()
            filtered_embeddings = embeddings
    else:
        filtered_df = df.copy()
        filtered_embeddings = embeddings

    # Get top-N recommendations
    top_n = int(data.get('top_n', 10))
    result_df = get_top_similar_courses(input_embedding, filtered_df, filtered_embeddings, top_n)

    return jsonify(result_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)