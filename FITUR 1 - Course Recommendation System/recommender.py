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

# Mapped category groupings
subcategory_mapping = {
    "Basic Science": "IT & Engineering",
    "Business Essentials": "Data & Product",
    "Cloud Computing": "IT & Engineering",
    "Computer Security and Networks": "IT & Engineering",
    "Data Analysis": "Data & Product",
    "Data Management": "Data & Product",
    "Design and Product": "Design & Creative",
    "Electrical Engineering": "IT & Engineering",
    "Machine Learning": "Data & Product",
    "Marketing": "Marketing & Social Media",
    "Math and Logic": "Data & Product",
    "Mechanical Engineering": "IT & Engineering",
    "Mobile and Web Development": "IT & Engineering",
    "Music and Art": "Design & Creative",
    "Networking": "IT & Engineering",
    "Probability and Statistics": "Data & Product",
    "Research Methods": "Data & Product",
    "Security": "IT & Engineering",
    "Software Development": "IT & Engineering",
    "Support and Operations": "IT & Engineering"
}

category_mapping = {
    "Computer Science": "IT & Engineering",
    "Data Science": "Data & Product",
    "Information Technology": "IT & Engineering",
    "Math and Logic": "Data & Product",
    "Physical Science and Engineering": "IT & Engineering",
    "Marketing": "Marketing & Social Media"
}

def map_category_group(cat, subcat):
    groups = set()
    if cat in category_mapping:
        groups.add(category_mapping[cat])
    if subcat in subcategory_mapping:
        groups.add(subcategory_mapping[subcat])
    return list(groups) if groups else None


def get_top_similar_courses(input_text, input_category=None, top_n=10):
        input_embedding = model.encode([input_text])
        if input_category:
            input_category_lower = [c.lower() for c in input_category]

            if 'MappedGroup' not in df.columns:
                df['MappedGroup'] = df.apply(
                    lambda row: map_category_group(row['Category'], row['Subcategory']), axis=1
                )

            mask = df['MappedGroup'].apply(
                lambda group_list: any(
                    group.lower() in input_category_lower
                    for group in group_list
                ) if isinstance(group_list, list) else False
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
            'Skills', 'Enrolled', 'Modules Name', 'Modules Description', 'Modules Duration'
        ]].copy()
        result_df['Similarity'] = similarities[top_indices]

        return result_df
