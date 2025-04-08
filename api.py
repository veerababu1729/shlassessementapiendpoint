from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os

app = Flask(__name__)
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Configure Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Load assessments dataset
df = pd.read_csv("assessments.csv")
if "full_text" not in df.columns:
    df["full_text"] = df.apply(
        lambda row: f"{row['name']} is a {row['test_type']} assessment with duration of {row['duration']}.", axis=1
    )

# Embed function
def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Missing query"}), 400

        query_vec = get_embedding(query)
        df["score"] = df["full_text"].apply(lambda x: cosine_similarity([query_vec], [get_embedding(x)])[0][0])
        top_df = df.sort_values("score", ascending=False).head(10)

        results = top_df[[
            "name", "url", "remote_testing", "adaptive_irt", "duration", "test_type", "score"
        ]].rename(columns={
            "name": "Assessment Name",
            "url": "URL",
            "remote_testing": "Remote Testing Support",
            "adaptive_irt": "Adaptive/IRT Support",
            "duration": "Duration",
            "test_type": "Test Type",
            "score": "Relevance Score"
        }).to_dict(orient="records")

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
