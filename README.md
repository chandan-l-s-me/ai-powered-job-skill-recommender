
#  Resume-Based Job & Skill Recommender with LLM Matching

This project is a powerful **AI-driven job & skill recommendation system** that intelligently analyzes your resume and matches it to the most relevant job listings and suggest some On market demand skills that you have missed out.It uses a blend of **LLMs, NLP, translation, semantic similarity, fuzzy matching, and job market data** to guide career decisions with precision.

---

##  Features

-  **Resume Parsing** (PDF support)
-  **Technical Skill Extraction** using LLM (Mistral-7B)
-  **Job Matching** via:
  - Keyword-based filtering
  - Semantic skill similarity with `SentenceTransformer`
-  **Real-time Job Fetching** from [Arbeitnow Job Board API](https://www.arbeitnow.com/)
-  **Multi-criteria Ranking** (salary, location, work mode, company culture, etc.)
-  **LLM-Based Career Evaluation** using Azure OpenAI (`GPT-4o`, `LLaMA`)
-  **Skill Gap Analysis** & Recommendations
-  **Translation** of job descriptions using `DeepTranslator`
-  **Interactive Gradio UI** with two tabs:
  - Job Recommender
  - Resume Analysis by Job Title

---

##  File Structure

- `main.py` — Complete pipeline with job fetching, skill extraction, evaluation, and Gradio UI.
- `jobs.csv` — CSV file of job titles and key skills used for skill-based filtering and fuzzy match.
- `semantic_top5_jobs.txt` — Output file saving the top 5 matched jobs and their analysis.

---

##  Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`:

```text
torch
transformers
sentence-transformers
PyPDF2
rapidfuzz
fuzzywuzzy
pandas
gradio
deep-translator
beautifulsoup4
requests
huggingface_hub
ipywidgets
azure-ai-inference
```

---

##  Environment Variables

Set the following environment variables before running:

| Variable        | Description                          |
|-----------------|--------------------------------------|
| `HF_TOKEN`      | HuggingFace access token             |
| `GITHUB_TOKEN`  | Token used for Azure AI inference    |

---

##  How to Run

```bash
python main.py
```

It launches an interactive **Gradio UI** for:

1. Finding best-matched jobs based on your resume.
2. Analyzing your resume against a specific job title.

---

##  How It Works

1. **Resume Upload** → Extracts technical skills using LLM.
2. **Job Fetching** → Jobs are fetched from Arbeitnow API or `jobs.csv`.
3. **Keyword + Semantic Matching** → Finds top jobs based on fuzzy and semantic scores.
4. **Job Transformation** → Uses LLM to extract job insights (salary, work culture, etc.).
5. **Fit Prediction** → Final evaluation ranks jobs based on weighted criteria.
6. **Skill Gap Analysis** → Suggests technical skills to improve based on job demands.

---

##  Evaluation Metrics

Each job is scored based on:

- Long-term potential
- Work culture
- Salary fit
- Transport & location fit
- Facilities offered
- Final weighted score (used for ranking)

---

##  UI Preview

<!-- Replace this with actual screenshot if needed -->
![Gradio Interface Preview](https://via.placeholder.com/800x400?text=Gradio+App+Preview)

---

##  License

This project is open-sourced under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Sentence-Transformers](https://www.sbert.net/)
- [Arbeitnow Job API](https://www.arbeitnow.com/api/job-board-api)
- [Azure AI Inference](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
