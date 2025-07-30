import requests
import json
import pandas as pd

test_cases = [
    {
        "skill": [
            "UI/UX",
            "User Research",
            "Figma",
            "Adobe XD",
            "Color Theory"
        ],
        "category": ["Design & Creative"],
        "top_n": 10
    },
    {
        "skill": [
            "Agile",
            "Scrum",
            "Kanban",
            "Sprint Retrospectives",
            "Waterfall Methodology",
            "Stakeholder Engagement",
            "Product Lifecycle Management"
        ],
        "category": ["Data & Product", "IT & Engineering"],
        "top_n": 10
    },
    {
        "skill": [
            "Javascript",
            "Angular",
            "React",
            "CSS",
            "Responsive Website Design"
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
            "Python",
            "Scikit-learn",
            "NumPy",
            "Unsupervised Learning",
            "Clustering"
        ],
        "category": ["Data & Product", "IT & Engineering"],
        "top_n": 10
    },
    {
        "skill": [
            "Network Protocols",
            "Firewalls",
            "VPN",
            "Intrusion Detection Systems (IDS)",
            "Wireshark"
        ],
        "category": ["IT & Engineering"],
        "top_n": 10
    },
    {
        "skill": [
            "Unity",
            "C#",
            "Unreal Engine",
            "Blueprints",
            "Frame Rate"
        ],
        "category": ["IT & Engineering", "Design & Creative"],
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
filename = "for_evaluation_output.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_outputs, f, indent=4, ensure_ascii=False)

rows = []

for case in all_outputs:
    input_skills = case["input"]["skill"]
    input_category = case["input"]["category"]
    recommendations = case["recommendations"]

    for rec in recommendations:
        rows.append({
            "Skills": ", ".join(input_skills),
            "Job Industry": ", ".join(input_category),
            "Course Title": rec.get("Title"),
            "Course Description": rec.get("Description"),
            "Course Category": rec.get("Category"),
            "Course Subcategory": rec.get("Subcategory"),
            "Skills Achieved": rec.get("Skills"),
            "Course Modules": "\n".join(
                f"{name}: {desc}" if desc else name
                for name, desc in zip(rec.get("Modules Name", []), rec.get("Modules Description", []))
            ) if isinstance(rec.get("Modules Name"), list) and isinstance(rec.get("Modules Description"), list) else rec.get("Modules Name"),
            "Similarity Score": rec.get("Similarity")
        })

df_result = pd.DataFrame(rows)
df_result.to_csv("evaluation/recommendation_results.csv", index=False)

print("Done")
