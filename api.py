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

# Load assessments dataset with the exact column names from your data
columns = ["name", "url", "remote_testing", "adaptive_irt", "duration", "test_type"]
data = [
    ["Verify Numerical Reasoning Test", "https://www.shl.com/solutions/products/verify-numerical-reasoning-test/", "Yes", "Yes", "18 minutes", "Cognitive"],
    ["Verify Verbal Reasoning Test", "https://www.shl.com/solutions/products/verify-verbal-reasoning-test/", "Yes", "Yes", "17 minutes", "Cognitive"],
    ["Verify Coding Pro", "https://www.shl.com/solutions/products/verify-coding-pro/", "Yes", "No", "60 minutes", "Technical"],
    ["OPQ Personality Assessment", "https://www.shl.com/solutions/products/opq/", "Yes", "No", "25 minutes", "Personality"],
    ["Java Programming Test", "https://www.shl.com/solutions/products/java-programming-test/", "Yes", "No", "30 minutes", "Technical"],
    ["Python Programming Test", "https://www.shl.com/solutions/products/python-programming-test/", "Yes", "No", "30 minutes", "Technical"],
    ["SQL Test", "https://www.shl.com/solutions/products/sql-test/", "Yes", "No", "30 minutes", "Technical"],
    ["JavaScript Test", "https://www.shl.com/solutions/products/javascript-test/", "Yes", "No", "30 minutes", "Technical"],
    ["Workplace Personality Assessment", "https://www.shl.com/solutions/products/workplace-personality/", "Yes", "No", "20 minutes", "Personality"],
    ["Business Simulation", "https://www.shl.com/solutions/products/business-simulation/", "Yes", "No", "40 minutes", "Simulation"],
    ["General Ability Test", "https://www.shl.com/solutions/products/general-ability/", "Yes", "Yes", "30 minutes", "Cognitive"],
    ["Teamwork Assessment", "https://www.shl.com/solutions/products/teamwork-assessment/", "Yes", "No", "15 minutes", "Behavioral"]
]
df = pd.DataFrame(data, columns=columns)

# Add descriptions for each assessment (simplified for this example)
descriptions = {
    "Python Programming Test": "Multi-choice test that measures the knowledge of Python programming, databases, modules and library. For developers.",
    "Java Programming Test": "Multi-choice test that measures the knowledge of Java programming, databases, frameworks and libraries. For developers.",
    "SQL Test": "Assessment that measures SQL querying and database knowledge. For data professionals and developers.",
    "JavaScript Test": "Assessment that evaluates JavaScript programming skills including DOM manipulation and frameworks.",
    "Verify Numerical Reasoning Test": "Assessment that measures numerical reasoning ability for workplace performance.",
    "Verify Verbal Reasoning Test": "Assessment that measures verbal reasoning ability for workplace performance.",
    "Verify Coding Pro": "Advanced coding assessment for professional developers across multiple languages.",
    "OPQ Personality Assessment": "Comprehensive workplace personality assessment for job fit and development.",
    "Workplace Personality Assessment": "Assessment that evaluates workplace behavior and personality traits.",
    "Business Simulation": "Interactive business scenario simulation for evaluating decision-making skills.",
    "General Ability Test": "Assessment that measures general mental ability across various cognitive domains.",
    "Teamwork Assessment": "The Technology Job Focused Assessment assesses key behavioral attributes required for success in fast-paced technology environments."
}
df["description"] = df["name"].map(descriptions)

# Add full_text column for embedding comparison
df["full_text"] = df.apply(
    lambda row: f"{row['name']} is a {row['test_type']} assessment with duration of {row['duration']}. {row['description']}", 
    axis=1
)

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
            
        # Format results to match the exact output shown in the screenshot
        results = []
        for _, row in top_df.iterrows():
            # Extract numerical duration value
            duration_value = ''.join(filter(str.isdigit, row["duration"]))
            
            # Format test_type as list (from the screenshot it appears test_type is an array)
            test_type_list = [row["test_type"]]
            if row["test_type"] == "Behavioral":
                # For demonstration - adding multiple test types for certain assessments 
                # as shown in your screenshot
                test_type_list = ["Competencies", "Personality & Behaviour"]
            
            result = {
                "url": row["url"],
                "adaptive_support": row["adaptive_irt"],
                "description": row["description"],
                "duration": int(duration_value) if duration_value else 0,
                "remote_support": row["remote_testing"],
                "test_type": test_type_list
            }
            results.append(result)
        
        # Construct response in the exact format shown in the screenshot
        response = {
            "recommended_assessments": results
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
