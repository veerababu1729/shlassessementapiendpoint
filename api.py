from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
import os
import json
import re

app = Flask(__name__)

# Health check endpoint for API status
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Configure Gemini API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment variables")
    # Continue without API key, fallback method will be used

if api_key:
    genai.configure(api_key=api_key)

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

# Simple keyword-based fallback when Gemini fails
def keyword_based_fallback(query, assessments_data):
    print("Using keyword-based fallback matching")
    query = query.lower()
    scores = {}
    
    # Define role/skill keywords and their related assessments
    keyword_matches = {
        "python": ["Python Programming Test", "Verify Coding Pro"],
        "java": ["Java Programming Test", "Verify Coding Pro"],
        "javascript": ["JavaScript Test", "Verify Coding Pro"],
        "js": ["JavaScript Test", "Verify Coding Pro"],
        "sql": ["SQL Test"],
        "database": ["SQL Test"],
        "developer": ["Verify Coding Pro", "Python Programming Test", "Java Programming Test", "JavaScript Test"],
        "programming": ["Verify Coding Pro", "Python Programming Test", "Java Programming Test", "JavaScript Test"],
        "leadership": ["OPQ Personality Assessment", "Business Simulation"],
        "manager": ["OPQ Personality Assessment", "Business Simulation"],
        "teamwork": ["Teamwork Assessment", "OPQ Personality Assessment"],
        "analytical": ["Verify Numerical Reasoning Test", "General Ability Test"],
        "communication": ["Verify Verbal Reasoning Test", "OPQ Personality Assessment"],
        "personality": ["OPQ Personality Assessment", "Workplace Personality Assessment"],
    }
    
    # Score each assessment based on keyword matches
    for _, row in assessments_data.iterrows():
        score = 0
        name_desc = (row["name"] + " " + row["description"]).lower()
        
        # Direct keyword matching
        for keyword in query.split():
            keyword = keyword.strip(",.;:()[]{}\"'")
            if keyword in name_desc:
                score += 2
            
            # Check predefined keyword matches
            if keyword in keyword_matches:
                if row["name"] in keyword_matches[keyword]:
                    score += 3
        
        scores[row["name"]] = score
    
    # Return top 3 matches with positive scores, or General Ability Test as fallback
    relevant = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True) if v > 0][:3]
    if not relevant:
        relevant = ["General Ability Test"]
        if "General Ability Test" not in assessments_data["name"].values:
            relevant = [assessments_data["name"].iloc[0]]
    
    return relevant

# Function to use Gemini to match assessments with query
def match_assessments_with_gemini(query, assessments_data):
    if not api_key:
        return keyword_based_fallback(query, assessments_data)
    
    try:
        # List available models to help with debugging
        try:
            print("Available Gemini models:")
            for model in genai.list_models():
                if "gemini" in model.name.lower():
                    print(f"- {model.name}")
        except Exception as e:
            print(f"Could not list models: {e}")
        
        # Try to use gemini-1.5-pro first, fall back to other versions if needed
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
        except Exception as e1:
            try:
                print(f"Could not load gemini-1.5-pro: {e1}, trying gemini-pro")
                model = genai.GenerativeModel('gemini-pro')
            except Exception as e2:
                print(f"Could not load gemini-pro: {e2}, falling back to keyword matching")
                return keyword_based_fallback(query, assessments_data)
        
        # Create a context with all assessment information
        assessments_context = []
        for _, row in assessments_data.iterrows():
            assessment_info = {
                "name": row["name"],
                "description": row["description"],
                "test_type": row["test_type"],
                "duration": row["duration"]
            }
            assessments_context.append(assessment_info)
        
        # Create the prompt for Gemini
        prompt = f"""
        As an HR assessment recommendation system, I need to find the most relevant assessments for the following job description or query:

        Query: "{query}"

        Available assessments:
        {json.dumps(assessments_context, indent=2)}

        For this query, which assessments from the list would be most relevant? Consider:
        1. Technical skills mentioned in the query
        2. Soft skills or personality traits mentioned
        3. The specific job role or industry
        4. Required experience level if mentioned

        Identify only the assessment names that are truly relevant to the query. Do not include any assessment that doesn't directly relate to the skills or attributes mentioned in the query. For example, if the query mentions "Java developer", don't include Python assessments.

        Respond with a JSON object having this exact format:
        {{
          "relevant_assessments": ["Assessment Name 1", "Assessment Name 2", ...]
        }}

        Your response should only include the JSON object, nothing else.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse the JSON response
        try:
            # Find JSON content between curly braces
            json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                assessments = result.get("relevant_assessments", [])
                if assessments:
                    return assessments
            
            print(f"Could not extract JSON from Gemini response. Falling back to keyword matching.")
            print(f"Raw response: {response_text}")
            return keyword_based_fallback(query, assessments_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Response was: {response_text}")
            return keyword_based_fallback(query, assessments_data)
    except Exception as e:
        print(f"Error in Gemini matching: {e}")
        return keyword_based_fallback(query, assessments_data)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Missing query"}), 400
        
        # Get relevant assessment names using Gemini or fallback
        relevant_assessment_names = match_assessments_with_gemini(query, df)
        
        # Filter the DataFrame to include only relevant assessments
        if relevant_assessment_names:
            relevant_df = df[df["name"].isin(relevant_assessment_names)]
        else:
            # Fallback: If no matches, return a general assessment
            relevant_df = df[df["name"] == "General Ability Test"]
            if relevant_df.empty:
                relevant_df = df.head(1)  # Absolute fallback
        
        # Format the assessment data for the response
        recommended_assessments = []
        for _, row in relevant_df.iterrows():
            # Extract duration value, handling potential non-numeric characters
            try:
                duration_value = int(''.join(filter(str.isdigit, str(row["duration"]))))
            except:
                duration_value = 30  # Default duration if extraction fails
                
            test_type_list = test_type_mappings.get(row["test_type"], [row["test_type"]])
            
            assessment = {
                "url": row["url"],
                "adaptive_support": "Yes" if row["adaptive_irt"] == "Yes" else "No",
                "description": row["description"],
                "duration": duration_value,
                "remote_support": "Yes" if row["remote_testing"] == "Yes" else "No",
                "test_type": test_type_list
            }
            recommended_assessments.append(assessment)
        
        # Ensure we have at least one but no more than 10 assessments
        if not recommended_assessments:
            # Absolute fallback - use first assessment in dataset
            first_row = df.iloc[0]
            try:
                duration_value = int(''.join(filter(str.isdigit, str(first_row["duration"]))))
            except:
                duration_value = 30
                
            test_type_list = test_type_mappings.get(first_row["test_type"], [first_row["test_type"]])
            
            fallback_assessment = {
                "url": first_row["url"],
                "adaptive_support": "Yes" if first_row["adaptive_irt"] == "Yes" else "No",
                "description": first_row["description"],
                "duration": duration_value,
                "remote_support": "Yes" if first_row["remote_testing"] == "Yes" else "No",
                "test_type": test_type_list
            }
            recommended_assessments = [fallback_assessment]
        elif len(recommended_assessments) > 10:
            recommended_assessments = recommended_assessments[:10]
        
        # Construct response in the exact format specified in the documentation
        response = {
            "recommended_assessments": recommended_assessments
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in recommendation endpoint: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
