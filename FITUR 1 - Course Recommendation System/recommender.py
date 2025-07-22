import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentence_model')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'courses_data.json')
EMBEDDING_PATH = os.path.join(BASE_DIR, 'course_embeddings.pkl')

# Load model, data and embeddings once
model = SentenceTransformer(MODEL_PATH)
df = pd.read_json(DATA_PATH, orient='records', encoding='utf-8-sig')
embeddings = pickle.load(open(EMBEDDING_PATH, 'rb'))

def get_top_similar_courses(input_text, input_category=None, top_n=10):
    input_embedding = model.encode([input_text])

    if input_category:
        mask = df['Category'].str.lower().apply(
            lambda cat: any(ic.lower() in cat for ic in input_category if isinstance(cat, str))
        )
        filtered_df = df[mask].copy()
        filtered_embeddings = embeddings[filtered_df.index]

        if filtered_df.empty:
            filtered_df = df.copy()
            filtered_embeddings = embeddings
    else:
        filtered_df = df.copy()
        filtered_embeddings = embeddings

    similarities = cosine_similarity(input_embedding, filtered_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]

    result_df = filtered_df.iloc[top_indices][[
        'Title', 'Institution', 'Type', 'Level', 'Duration',
        'Category', 'Subcategory', 'Rating', 'Description',
        'Skills', 'Modules Name', 'Modules Description'
    ]].copy()
    result_df['Similarity'] = similarities[top_indices]

    return result_df