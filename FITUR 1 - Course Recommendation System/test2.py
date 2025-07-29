import requests
import json

test_cases = [
    {
        "skill": [
            "User Interface (UI)",
            "User Experience",
            "Figma",
            "Prototyping",
            "Mockups",
            "Visual Design",
            "Creativity"
        ],
        "category": ["Design & Creative"],
        "top_n": 10
    },
    {
        "skill": [
            "Project Management",
            "Agile",
            "Scrum",
            "Kanban",
            "Jira",
            "Team Management",
            "Project Scheduling",
            "Stakeholder Management",
            "Gantt Charts"
        ],
        "category": ["Data & Product", "IT & Engineering"],
        "top_n": 10
    },
    {
        "skill": [
            "Node.js",
            "Express.js",
            "RESTful APIs",
            "Middleware",
            "Routing",
            "Laravel (PHP)",
            "Website"
        ],
        "category": ["IT & Engineering"],
        "top_n": 10
    },
    {
        "skill": [
            "Pay-Per-Click Advertising",
            "Google Ads",
            "Ad Copywriting",
            "Traffic Source Analysis",
            "Bounce Rate",
            "Marketing Strategy",
            "Content Marketing"
        ],
        "category": ["Marketing & Social Media"],
        "top_n": 10
    },
    {
        "skill": [
            "Predictive Modeling",
            "Model Evaluation",
            "Pattern Recognition",
            "Python",
            "Pandas",
            "Scikit-learn",
            "Jupyter Notebook",
            "Data Wrangling",
            "Linear Regression",
            "Clustering"
        ],
        "category": ["Data & Product", "IT & Engineering"],
        "top_n": 10
    }
]


url = "http://127.0.0.1:5000/recommend_course"

all_outputs = []

for input_data in test_cases:
    response = requests.post(url, json=input_data)

    if response.status_code == 200:
        recommendations = response.json()
        all_outputs.append({
            "input": input_data,
            "recommendations": recommendations
        })
    else:
        print(f"Error with input: {input_data}")
        print(response.status_code, response.text)

# Simpan ke file JSON
filename = "test2_output.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_outputs, f, indent=4, ensure_ascii=False)

print("Done")