from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os

app = Flask(__name__)

# ✅ Health check endpoint for API status
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Configure Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Load assessments dataset
try:
    df = pd.read_csv("assessments.csv")
    if "full_text" not in df.columns:
        df["full_text"] = df.apply(
            lambda row: f"{row['name']} is a {row['test_type']} assessment with duration of {row['duration']}. This test is designed to assess {row['test_type']} skills.", 
            axis=1
        )
except Exception as e:
    print(f"Error loading data: {e}")

# Embed function
def get_embedding(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response["embedding"])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

# Check if an assessment is relevant based on keywords in query
def is_relevant_to_query(assessment_text, query_terms):
    assessment_text = assessment_text.lower()
    # Extract important words from query (remove common words)
    query_terms = [term.lower() for term in query_terms.split() 
                  if len(term) > 2 and term.lower() not in ['the', 'and', 'for', 'with']]
    
    # Count how many query terms appear in the assessment text
    matches = sum(1 for term in query_terms if term in assessment_text)
    
    # Consider relevant if at least one term matches or if the query is very short
    return matches > 0 or len(query_terms) == 0

# Function to filter relevant assessments by excluding obvious mismatches
def filter_irrelevant_tests(df, query):
    query = query.lower()
    
    # Special case handling for common language/tech queries
    filtered_df = df.copy()
    
    # Handle programming language specific queries
    if 'java ' in query or query == 'java' or 'java developer' in query:
        # For Java queries, prioritize Java tests and exclude non-relevant languages
        relevant_terms = ['java', 'javascript', 'sql', 'programming', 'coding', 'developer']
        filtered_df = filtered_df[filtered_df['full_text'].str.lower().apply(
            lambda text: any(term in text.lower() for term in relevant_terms))]
        
    elif 'python' in query:
        # For Python queries, prioritize Python tests
        relevant_terms = ['python', 'programming', 'data science', 'sql', 'coding']
        filtered_df = filtered_df[filtered_df['full_text'].str.lower().apply(
            lambda text: any(term in text.lower() for term in relevant_terms))]
    
    # Add more language/tech specific filters as needed
    
    # If filtering removed all options, revert to original dataset
    if len(filtered_df) == 0:
        return df
    
    return filtered_df

# Dynamic threshold based on overall score distribution
def get_dynamic_threshold(scores, default_min=0.60):
    if len(scores) < 5:
        return default_min
    
    # Use statistical approaches to find natural cutoffs
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # More aggressive threshold: mean + 0.25*std deviation 
    threshold = mean_score + 0.25 * std_score
    
    # Don't go below our minimum acceptable threshold
    return max(threshold, default_min)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Missing query"}), 400
        
        # First filter obviously irrelevant tests based on keyword matching
        filtered_df = filter_irrelevant_tests(df, query)
        
        query_vec = get_embedding(query)
        
        # Calculate scores with boost for relevant matches
        filtered_df = filtered_df.copy()
        filtered_df["score"] = 0.0
        
        for index, row in filtered_df.iterrows():
            try:
                doc_vec = get_embedding(row["full_text"])
                similarity_score = cosine_similarity([query_vec], [doc_vec])[0][0]
                
                # Give a boost to assessments that match keywords in query
                if is_relevant_to_query(row["full_text"], query):
                    similarity_score += 0.1  # Boost matching assessments
                
                filtered_df.at[index, "score"] = similarity_score
            except Exception as e:
                print(f"Error processing assessment {row.get('name', 'Unknown')}: {e}")
        
        # Sort by relevance score
        sorted_df = filtered_df.sort_values("score", ascending=False)
        
        # Get a dynamic threshold based on score distribution
        all_scores = sorted_df["score"].values
        dynamic_threshold = get_dynamic_threshold(all_scores)
        
        # Filter by relevance threshold to ensure results are relevant
        relevant_df = sorted_df[sorted_df["score"] >= dynamic_threshold]
        
        # Ensure we show between 1-10 results
        if len(relevant_df) == 0:
            top_df = sorted_df.head(1)
        elif len(relevant_df) > 10:
            top_df = relevant_df.head(10)
        else:
            top_df = relevant_df
        
        # Prepare results
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
        
        # Format the score as percentage in the response
        for result in results:
            result["Relevance Score"] = f"{result['Relevance Score']:.2%}"
        
        # Add count of results to response
        response = {
            "count": len(results),
            "results": results
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
