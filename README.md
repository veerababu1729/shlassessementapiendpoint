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

