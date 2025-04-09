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

# Refine keywords to be more specific and avoid false positives
assessment_keywords = {
    "Java Programming Test": {
        "must_include": ["java"],
        "boost_terms": ["java developer", "java programming", "j2ee", "spring", "hibernate"],
        "excludes": ["javascript", "js", "python", "ruby", "php"]
    },
    "JavaScript Test": {
        "must_include": ["javascript", "js", "frontend"],
        "boost_terms": ["web developer", "frontend", "react", "angular", "vue", "node"],
        "excludes": ["java developer"]
    },
    "Python Programming Test": {
        "must_include": ["python"],
        "boost_terms": ["data science", "machine learning", "django", "flask", "pandas"],
        "excludes": ["java", "javascript", "ruby"]
    },
    "SQL Test": {
        "must_include": ["sql", "database"],
        "boost_terms": ["data", "database", "query", "postgresql", "mysql"],
        "excludes": []
    },
    "Teamwork Assessment": {
        "must_include": ["team", "teamwork", "collaboration", "cooperate"],
        "boost_terms": ["interpersonal", "communication", "group work", "soft skills"],
        "excludes": []
    },
    "OPQ Personality Assessment": {
        "must_include": ["personality", "character", "behavior"],
        "boost_terms": ["traits", "soft skills", "culture fit"],
        "excludes": []
    },
    "Workplace Personality Assessment": {
        "must_include": ["personality", "workplace"],
        "boost_terms": ["work style", "professional behavior"],
        "excludes": []
    },
    "Business Simulation": {
        "must_include": ["business", "simulation", "management"],
        "boost_terms": ["strategy", "leadership", "executive"],
        "excludes": []
    },
    "General Ability Test": {
        "must_include": ["general", "aptitude", "cognitive"],
        "boost_terms": ["intelligence", "reasoning", "problem solving"],
        "excludes": []
    },
    "Verify Numerical Reasoning Test": {
        "must_include": ["numerical", "math", "quantitative"],
        "boost_terms": ["calculations", "data interpretation"],
        "excludes": []
    },
    "Verify Verbal Reasoning Test": {
        "must_include": ["verbal", "language", "reading"],
        "boost_terms": ["comprehension", "communication"],
        "excludes": []
    },
    "Verify Coding Pro": {
        "must_include": ["coding", "programming", "developer"],
        "boost_terms": ["algorithm", "software engineer"],
        "excludes": []
    }
}

# Use Gemini to analyze the query
def analyze_query_with_gemini(query):
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Based on the following job description or query, identify the EXACT skills, roles, and qualities being sought.
        
        Job Description/Query: "{query}"
        
        Please respond in the following JSON format only:
        {{
            "primary_skills": ["skill1", "skill2"],  // Technical or specific skills (e.g. Java, Python, SQL)
            "soft_skills": ["skill1", "skill2"],     // Soft skills (e.g. teamwork, communication)
            "roles": ["role1", "role2"],             // Job roles (e.g. developer, analyst)
            "experience_level": "entry/mid/senior",  // Experience level mentioned
            "time_constraints": true/false           // Whether time or speed is mentioned
        }}
        
        Your response should ONLY include the JSON, nothing else.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract just the JSON part
        json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
        if json_match:
            try:
                import json
                analysis = json.loads(json_match.group(1))
                return analysis
            except json.JSONDecodeError:
                return None
        return None
    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        return None

# Improved relevance function using Gemini analysis and strict filtering
def filter_and_score_assessments(query, df):
    # Basic query processing
    query_lower = query.lower()
    
    # Get Gemini analysis
    gemini_analysis = analyze_query_with_gemini(query)
    
    # Extract all skills and roles from Gemini analysis
    all_terms = []
    if gemini_analysis:
        all_terms.extend(gemini_analysis.get('primary_skills', []))
        all_terms.extend(gemini_analysis.get('soft_skills', []))
        all_terms.extend(gemini_analysis.get('roles', []))
    
    # Fallback to basic keyword extraction if Gemini fails
    if not all_terms:
        words = query_lower.split()
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about'}
        all_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    results = []
    
    for index, row in df.iterrows():
        assessment_name = row['name']
        keywords = assessment_keywords.get(assessment_name, {})
        
        # Check if assessment has required keywords
        must_include = keywords.get('must_include', [])
        excludes = keywords.get('excludes', [])
        
        # Initialize score and relevance flag
        score = 0
        relevant = False
        
        # Check for must-include terms
        if must_include:
            for term in must_include:
                if term in query_lower:
                    relevant = True
                    score += 5
                    break
        
        # If not relevant by must-include, check Gemini analysis
        if not relevant and gemini_analysis:
            # Check primary skills
            for skill in gemini_analysis.get('primary_skills', []):
                if any(must_term in skill.lower() for must_term in must_include):
                    relevant = True
                    score += 5
                    break
            
            # Special case for Teamwork Assessment
            if assessment_name == "Teamwork Assessment" and any(
                team_term in skill.lower() for skill in gemini_analysis.get('soft_skills', [])
                for team_term in ['team', 'collaborate', 'interpersonal']
            ):
                relevant = True
                score += 5
        
        # Check for exclusions - disqualify if found
        for term in excludes:
            if term in query_lower:
                relevant = False
                break
                
        # Only continue if assessment is relevant
        if relevant:
            # Add scores for boost terms
            boost_terms = keywords.get('boost_terms', [])
            for term in boost_terms:
                if term in query_lower:
                    score += 2
            
            # Check experience level match (if provided by Gemini)
            if gemini_analysis and 'experience_level' in gemini_analysis:
                exp_level = gemini_analysis['experience_level'].lower()
                if "entry" in exp_level and "entry" in row.get('description', '').lower():
                    score += 1
                elif "mid" in exp_level and "mid" in row.get('description', '').lower():
                    score += 1
                elif "senior" in exp_level and "senior" in row.get('description', '').lower():
                    score += 1
            
            # Adjust score based on time constraints
            if gemini_analysis and gemini_analysis.get('time_constraints', False):
                duration_mins = int(''.join(filter(str.isdigit, row["duration"])))
                if duration_mins <= 20:
                    score += 1  # Boost for short tests
                elif duration_mins >= 45:
                    score -= 1  # Penalty for long tests
            
            # Calculate embedding similarity with Gemini (optional - can be removed for simplicity)
            # This step is omitted for simplicity
            
            # Format the assessment data
            duration_value = int(''.join(filter(str.isdigit, row["duration"])))
            test_type_list = test_type_mappings.get(row["test_type"], [row["test_type"]])
            
            results.append({
                "assessment": {
                    "url": row["url"],
                    "adaptive_support": "Yes" if row["adaptive_irt"] == "Yes" else "No",
                    "description": row["description"],
                    "duration": duration_value,
                    "remote_support": "Yes" if row["remote_testing"] == "Yes" else "No",
                    "test_type": test_type_list
                },
                "score": score
            })
    
    # Sort by score and return
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_results

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Missing query"}), 400
        
        # Get scored and filtered assessments
        assessment_results = filter_and_score_assessments(query, df)
        
        # Take only the relevant assessments (max 10)
        relevant_assessments = [result["assessment"] for result in assessment_results[:10]]
        
        # If no assessments were found to be relevant, fall back to basic matching
        if not relevant_assessments:
            # Get most relevant by test type
            if "java" in query.lower():
                java_test = df[df["name"] == "Java Programming Test"].iloc[0]
                test_type_list = test_type_mappings.get(java_test["test_type"], [java_test["test_type"]])
                relevant_assessments = [{
                    "url": java_test["url"],
                    "adaptive_support": "Yes" if java_test["adaptive_irt"] == "Yes" else "No",
                    "description": java_test["description"],
                    "duration": int(''.join(filter(str.isdigit, java_test["duration"]))),
                    "remote_support": "Yes" if java_test["remote_testing"] == "Yes" else "No",
                    "test_type": test_type_list
                }]
        
        # Ensure we have at least one result
        if not relevant_assessments:
            general_test = df[df["name"] == "General Ability Test"].iloc[0]
            test_type_list = test_type_mappings.get(general_test["test_type"], [general_test["test_type"]])
            relevant_assessments = [{
                "url": general_test["url"],
                "adaptive_support": "Yes" if general_test["adaptive_irt"] == "Yes" else "No",
                "description": general_test["description"],
                "duration": int(''.join(filter(str.isdigit, general_test["duration"]))),
                "remote_support": "Yes" if general_test["remote_testing"] == "Yes" else "No",
                "test_type": test_type_list
            }]
        
        # Construct response in the exact format specified in the documentation
        response = {
            "recommended_assessments": relevant_assessments
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
