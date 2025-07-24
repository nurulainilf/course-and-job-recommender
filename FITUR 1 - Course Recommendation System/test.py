import requests
import json

test_cases = [
    {
        "skill": ["python", "machine learning", "data analysis"],
        "category": ["data science"],
        "top_n": 5
    },
    {
        "skill": ["user experience", "user interface", "figma"],
        "category": ["design"]
    },
    {
        "skill": ["excel", "power bi", "data visualization"],
        "category": ["business intelligence"]
    },
    {
        "skill": ["seo", "digital advertising", "google ads"],
        "category": ["marketing"]
    },
    {
        "skill": ["cloud computing", "aws", "devops"]
    },
    {
        "skill": ["product strategy", "agile", "user research", "roadmap", "a/b testing"],
        "category": ["product management", "business"]
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
filename = "PP_MLOps_Nurul Ainil Fitri_Pricillia Silfany_Output2.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_outputs, f, indent=4, ensure_ascii=False)

print("Done")