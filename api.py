from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
import os
import json

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

# Function to use Gemini to match assessments with query
def match_assessments_with_gemini(query, assessments_data):
    try:
        model = genai.GenerativeModel('gemini-pro')
        
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
            import re
            json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                return result.get("relevant_assessments", [])
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Response was: {response_text}")
            return []
    except Exception as e:
        print(f"Error in Gemini matching: {e}")
        return []

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Missing query"}), 400
        
        # Get relevant assessment names using Gemini
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
            duration_value = int(''.join(filter(str.isdigit, row["duration"])))
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
            duration_value = int(''.join(filter(str.isdigit, first_row["duration"])))
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
