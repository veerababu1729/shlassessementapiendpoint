from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import re

app = Flask(__name__)

# Health check endpoint for API status
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Configure Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Load assessments dataset from CSV file
df = pd.read_csv("assessments.csv")

# Add descriptions for each assessment
descriptions = {
    "Python Programming Test": "Multi-choice test that measures the knowledge of Python programming, databases, modules and library. For Mid-Professional developers.",
    "Java Programming Test": "Multi-choice test that measures the knowledge of Java programming, databases, frameworks and libraries. For Entry-Level developers.",
    "SQL Test": "Assessment that measures SQL querying and database knowledge. For data professionals and developers.",
    "JavaScript Test": "Assessment that evaluates JavaScript programming skills including DOM manipulation and frameworks.",
    "Verify Numerical Reasoning Test": "Assessment that measures numerical reasoning ability for workplace performance.",
    "Verify Verbal Reasoning Test": "Assessment that measures verbal reasoning ability for workplace performance.",
    "Verify Coding Pro": "Advanced coding assessment for professional developers across multiple languages.",
    "OPQ Personality Assessment": "Comprehensive workplace personality assessment for job fit and development.",
    "Workplace Personality Assessment": "Assessment that evaluates workplace behavior and personality traits.",
    "Business Simulation": "Interactive business scenario simulation for evaluating decision-making skills.",
    "General Ability Test": "Assessment that measures general mental ability across various cognitive domains.",
    "Teamwork Assessment": "The Technology Job Focused Assessment assesses key behavioral attributes required for success in fast-paced, rapidly changing technology work environments."
}
df["description"] = df["name"].map(descriptions)

# Define test type mappings for API response format
test_type_mappings = {
    "Cognitive": ["Knowledge & Skills"],
    "Technical": ["Knowledge & Skills"],
    "Personality": ["Personality & Behaviour"],
    "Behavioral": ["Competencies", "Personality & Behaviour"],
    "Simulation": ["Competencies"]
}

# Define semantic concepts and their related terms for each assessment
semantic_concepts = {
    "Java Programming Test": {
        "primary": ["java", "java programming", "java developer", "java development"],
        "secondary": ["programming", "coding", "development", "backend", "enterprise", "spring", "hibernate"],
        "excluded": ["javascript", "js", "frontend", "python"]
    },
    "JavaScript Test": {
        "primary": ["javascript", "js", "frontend", "web development", "frontend developer"],
        "secondary": ["programming", "coding", "development", "web", "react", "angular", "vue"],
        "excluded": []
    },
    "Python Programming Test": {
        "primary": ["python", "python programming", "python developer", "data science"],
        "secondary": ["programming", "coding", "development", "data analysis", "machine learning", "AI"],
        "excluded": []
    },
    "SQL Test": {
        "primary": ["sql", "database", "data", "query"],
        "secondary": ["data management", "database design", "data engineering"],
        "excluded": []
    },
    "Teamwork Assessment": {
        "primary": ["team", "teamwork", "collaboration", "cooperate", "collaborate"],
        "secondary": ["interpersonal", "communication", "group work", "soft skills"],
        "excluded": []
    },
    "OPQ Personality Assessment": {
        "primary": ["personality", "character", "behavior", "traits"],
        "secondary": ["soft skills", "interpersonal", "work style", "culture fit"],
        "excluded": []
    },
    "Workplace Personality Assessment": {
        "primary": ["workplace", "personality", "behavior", "traits"],
        "secondary": ["soft skills", "interpersonal", "work style", "culture fit"],
        "excluded": []
    },
    "Business Simulation": {
        "primary": ["business", "simulation", "scenario", "strategy"],
        "secondary": ["decision making", "management", "leadership", "executive"],
        "excluded": []
    },
    "General Ability Test": {
        "primary": ["general ability", "aptitude", "intelligence", "cognitive"],
        "secondary": ["reasoning", "problem solving", "analytical thinking"],
        "excluded": []
    },
    "Verify Numerical Reasoning Test": {
        "primary": ["numerical", "math", "quantitative", "calculations"],
        "secondary": ["reasoning", "problem solving", "analysis", "data interpretation"],
        "excluded": []
    },
    "Verify Verbal Reasoning Test": {
        "primary": ["verbal", "language", "reading", "comprehension"],
        "secondary": ["reasoning", "communication", "analysis", "critical thinking"],
        "excluded": []
    },
    "Verify Coding Pro": {
        "primary": ["coding", "programming", "development", "technical skills"],
        "secondary": ["problem solving", "algorithms", "software engineering"],
        "excluded": []
    }
}

# Add full_text column for embedding comparison
df["full_text"] = df.apply(
    lambda row: f"{row['name']} is a {row['test_type']} assessment with duration of {row['duration']}. {row.get('description', '')}", 
    axis=1
)

# Create semantic concept mapping for each assessment
df["semantic_concepts"] = df["name"].map(semantic_concepts)

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
        # Return a zero vector as fallback
        return np.zeros(768)  # Adjust dimension based on your embedding model

# Extract key concepts from the query
def extract_key_concepts(query):
    # Normalize query text
    query = query.lower()
    
    # Remove common stop words and keep important terms
    words = re.findall(r'\b\w+\b', query)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'who', 'also', 'can'}
    important_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Look for compound concepts (e.g., "Java developer", "team work")
    compound_concepts = []
    for i in range(len(words) - 1):
        if words[i] not in stop_words or words[i+1] not in stop_words:
            compound_concepts.append(f"{words[i]} {words[i+1]}")
    
    # Add trigrams if they might be meaningful
    for i in range(len(words) - 2):
        if not all(word in stop_words for word in [words[i], words[i+1], words[i+2]]):
            compound_concepts.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    # Combine all extracted concepts
    all_concepts = important_terms + compound_concepts
    
    # Look for specific skills/domains typically assessed
    skill_domains = {
        "programming": ["coding", "programming", "developer", "software", "engineer"],
        "language_specific": ["java", "python", "javascript", "js", "sql", "html", "css"],
        "teamwork": ["team", "teamwork", "collaborate", "collaboration", "interpersonal"],
        "cognitive": ["reasoning", "cognitive", "thinking", "intelligence", "aptitude", "problem solving"],
        "personality": ["personality", "behavior", "traits", "character", "temperament"],
        "technical": ["technical", "coding", "programming", "development", "engineering"],
        "business": ["business", "management", "leadership", "strategy", "decision"]
    }
    
    # Identify which domains are present in the query
    identified_domains = {}
    for domain, terms in skill_domains.items():
        domain_matches = [term for term in terms if any(term in concept for concept in all_concepts)]
        if domain_matches:
            identified_domains[domain] = domain_matches
    
    return {
        "terms": important_terms,
        "compounds": compound_concepts,
        "all_concepts": all_concepts,
        "domains": identified_domains
    }

# Semantic scoring function
def calculate_semantic_relevance(assessment, extracted_concepts):
    semantic_data = assessment.get("semantic_concepts", {})
    if not semantic_data:
        return 0.0
    
    score = 0.0
    
    # Check for primary concept matches (highest weight)
    primary_matches = sum(1 for concept in extracted_concepts["all_concepts"] 
                         if any(primary_term in concept for primary_term in semantic_data.get("primary", [])))
    score += primary_matches * 3.0
    
    # Check for secondary concept matches (medium weight)
    secondary_matches = sum(1 for concept in extracted_concepts["all_concepts"] 
                           if any(secondary_term in concept for secondary_term in semantic_data.get("secondary", [])))
    score += secondary_matches * 1.5
    
    # Check for domain relevance
    for domain, terms in extracted_concepts["domains"].items():
        if domain == "language_specific":
            # Special handling for programming languages to ensure exact matches
            for term in terms:
                if term in semantic_data.get("primary", []):
                    score += 4.0  # Strong boost for exact language match
                elif term in semantic_data.get("excluded", []):
                    score -= 5.0  # Strong penalty for excluded languages
        elif domain in ["programming", "technical"] and assessment["test_type"] == "Technical":
            score += 2.0
        elif domain == "teamwork" and "Teamwork Assessment" in assessment["name"]:
            score += 3.0
        elif domain == "personality" and "Personality" in assessment["test_type"]:
            score += 3.0
        elif domain == "cognitive" and "Cognitive" in assessment["test_type"]:
            score += 3.0
        elif domain == "business" and "Business" in assessment["name"]:
            score += 3.0
    
    # Check for exclusions (negative weight)
    exclusion_matches = sum(1 for concept in extracted_concepts["all_concepts"] 
                           if any(excluded_term in concept for excluded_term in semantic_data.get("excluded", [])))
    score -= exclusion_matches * 4.0
    
    # Special case handling for combined skillsets
    if "java" in extracted_concepts["terms"] and any(team_term in " ".join(extracted_concepts["all_concepts"]) 
                                                  for team_term in ["team", "teamwork", "collaborate"]):
        if "Java Programming Test" in assessment["name"] or "Teamwork Assessment" in assessment["name"]:
            score += 2.5  # Boost both Java and Teamwork assessments
    
    # Duration constraints - if mentioned in query
    duration_terms = ["time", "minutes", "duration", "quick", "fast", "short", "long"]
    if any(term in " ".join(extracted_concepts["all_concepts"]) for term in duration_terms):
        # Prefer shorter tests if query mentions time constraints
        duration_mins = int(''.join(filter(str.isdigit, assessment["duration"])))
        if duration_mins <= 20:
            score += 1.0  # Boost for short tests
        elif duration_mins >= 45:
            score -= 1.0  # Penalty for long tests
    
    return max(score, 0.0)  # Ensure we don't return negative scores

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Missing query"}), 400
        
        # Extract semantic concepts from query
        extracted_concepts = extract_key_concepts(query)
        
        # Calculate vector embedding for query for comparison
        query_vec = get_embedding(query)
        
        # Create results dataframe with all relevant data
        results_df = df.copy()
        
        # Calculate semantic relevance score
        results_df["semantic_score"] = results_df.apply(
            lambda row: calculate_semantic_relevance({
                "name": row["name"],
                "test_type": row["test_type"],
                "semantic_concepts": row["semantic_concepts"],
                "duration": row["duration"]
            }, extracted_concepts), 
            axis=1
        )
        
        # Calculate embedding similarity score
        for index, row in results_df.iterrows():
            try:
                doc_vec = get_embedding(row["full_text"])
                similarity_score = cosine_similarity([query_vec], [doc_vec])[0][0]
                results_df.at[index, "embedding_score"] = similarity_score
            except Exception as e:
                print(f"Error processing assessment {row.get('name', 'Unknown')}: {e}")
                results_df.at[index, "embedding_score"] = 0.0
        
        # Combine scores (weighted approach)
        results_df["final_score"] = (0.7 * results_df["semantic_score"]) + (0.3 * results_df["embedding_score"])
        
        # Sort by combined score
        sorted_df = results_df.sort_values("final_score", ascending=False)
        
        # Determine how many results to return (between 1 and 10)
        max_results = 10
        min_results = 1
        score_threshold = 0.5
        relevant_df = sorted_df[sorted_df["final_score"] >= score_threshold]
        
        if len(relevant_df) < min_results:
            top_df = sorted_df.head(min_results)  # Show at least minimum results
        elif len(relevant_df) > max_results:
            top_df = relevant_df.head(max_results)  # Limit to max results
        else:
            top_df = relevant_df
            
        # Format results to match the exact format required by API documentation
        results = []
        for _, row in top_df.iterrows():
            # Extract numerical duration value
            duration_value = int(''.join(filter(str.isdigit, row["duration"])))
            
            # Map test_type to the required format
            test_type_list = test_type_mappings.get(row["test_type"], [row["test_type"]])
            
            result = {
                "url": row["url"],
                "adaptive_support": "Yes" if row["adaptive_irt"] == "Yes" else "No",
                "description": row["description"],
                "duration": duration_value,
                "remote_support": "Yes" if row["remote_testing"] == "Yes" else "No",
                "test_type": test_type_list
            }
            results.append(result)
        
        # Construct response in the exact format specified in the documentation
        response = {
            "recommended_assessments": results
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
