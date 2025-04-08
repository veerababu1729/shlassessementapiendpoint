 📄 SHL Assessment Recommendation System – Solution Documentation

**Author:** [Veerababu Palepu]  
**Live UI Webapp:** [https://shlassesmentrecommendationsystem.streamlit.app/      
**API Endpoint:** [https://shlassessementapiendpoint.onrender.com/recommend](https://shlassessementapiendpoint.onrender.com/recommend)  
Documentation Link: https://docs.google.com/document/d/1sxx3PIxQii_J7cgBx6FcNcF7dYggR5JBwr4CbwBIKOs/edit?usp=sharing

---

 ✅ Objective

Create a system to recommend the most relevant SHL assessments based on a job description or query. Deliver both:
- **STEP 1:** An interactive web application.
- **STEP 2:** A REST API endpoint that returns JSON recommendations.

---

🧠 Approach Summary

STEP 1 – Web UI (Streamlit)**
  Access Here: **Live UI Webapp:** [https://shlassesmentrecommendationsystem.streamlit.app/  
- Built using **Streamlit** for a clean user interface.
- Users input a job description or role requirement.
- The system embeds both query and test descriptions using **Google Gemini Embedding API** (`models/embedding-001`).
- Computes **cosine similarity** to recommend the top 10 relevant SHL assessments.
- Displays results in a sortable, clickable table with URLs to the test pages.

STEP 2 – JSON API Endpoint (Render)**

- Created a **FastAPI-based endpoint** `/recommend`.
- Accepts a POST request with a query string.
- Returns a ranked list of the top 5 assessments in JSON.
- Deployed the backend using **Render.com** for public accessibility.

---

### 🛠️ Technologies Used

| Component         | Tool / Library                    |
|------------------|-----------------------------------|
| Embedding Model  | Google Gemini `embedding-001`     |
| Web App          | Streamlit                         |
| REST API         | FastAPI                           |
| Deployment       | Streamlit Cloud, Render           |
| Data Processing  | pandas, NumPy, scikit-learn       |
| Hosting          | GitHub (for version control)      |

---

### 📦 API Usage

**POST /recommend**  
**URL:** `https://shlassessementapiendpoint.onrender.com/recommend`

**Request:**
```json
{
  "query": "Looking for cognitive ability and reasoning assessments"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "name": "Logical Reasoning Test",
      "test_type": "Cognitive",
      "duration": "30 mins",
      "url": "https://example.com/logical-test"
    },
    ...
  ]
}
```



---

### 🔧 Here are 3 Easy Ways to Test API End Points:
Test the health endpoint in browser:👇

https://shlassessementapiendpoint.onrender.com/health

You should see:
{
  "status": "healthy"
}


#### **1. Use Python Script (Recommended)**
Run this locally in a `.py` file or Jupyter Notebook:
```python
import requests

url = "https://shlassessementapiendpoint.onrender.com/recommend"
payload = {"query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Results:\n", response.json())
```

---

#### **2. Use [Postman](https://www.postman.com/)**
- Method: `POST`
- URL: `https://shlassessementapiendpoint.onrender.com/recommend`
- Body → `raw` → `JSON`:
```json
{
  "query": "Looking to hire mid-level professionals skilled in Python, SQL, and JavaScript."
}
```

---

#### **3. Use `curl` in Terminal**
```bash
curl -X POST https://shlassessementapiendpoint.onrender.com/recommend \
-H "Content-Type: application/json" \
-d '{"query": "Looking to hire analysts with strong cognitive and personality skills."}'
```
