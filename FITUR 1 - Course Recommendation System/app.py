from flask import Flask, request, jsonify
from recommender import get_top_similar_courses

app = Flask(__name__)

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
    top_n = int(data.get('top_n', 10))

    result_df = get_top_similar_courses(input_text, input_category, top_n)
    return jsonify(result_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)