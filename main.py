

import re
import torch
from rapidfuzz import fuzz
from huggingface_hub import login
import ast
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from rapidfuzz import fuzz
import time
from concurrent.futures import ThreadPoolExecutor
import ipywidgets as widgets
from IPython.display import display
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from pathlib import Path
import ast
from fuzzywuzzy import fuzz
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil
# Replace with your actual token
hf_token = os.getenv("HF_TOKEN")
token = os.getenv("GITHUB_TOKEN")

login(token=hf_token, new_session=False)
# Load model directly

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model_mistral= AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",device_map="auto")
KEYWORD_TOP_K = 20
SEMANTIC_TOP_K = 10
FUZZY_THRESHOLD=70
MAX_COMBINED_TEXT_LEN = 500
API_URL = "https://www.arbeitnow.com/api/job-board-api"
endpoint = "https://models.github.ai/inference"
model_gpt = "openai/gpt-4o"
translator = GoogleTranslator(source='auto', target='en')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# token = os.environ[token] # Removed this line
model1="meta/Llama-4-Maverick-17B-128E-Instruct-FP8"
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token), # Using the token variable directly
)
client1 = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token), # Using the token variable directly
)


df = pd.read_csv('jobs.csv')
print(df.columns)
df = df[['Job Title', 'Key Skills']]
df = df[df['Job Title'].str.len() <= 30]
def merge_skills(skills_series):
    all_skills = set()
    for skills in skills_series:
        # Split by comma and strip spaces
        split_skills = [s.strip() for s in skills.split(',')]
        all_skills.update(split_skills)
    # Return skills joined by comma
    return ','.join(sorted(all_skills))

# Group by Job Title and apply
merged = df.groupby('Job Title')['Key Skills'].apply(merge_skills).reset_index()

print(merged)
# Assume df and merged are your original and merged DataFrames

# Convert df['Job Title'] to a set for fast lookup
original_jobs = set(df['Job Title'].unique())

# Check if each Job Title in merged exists in original_jobs
merged['Job_Title_in_df'] = merged['Job Title'].apply(lambda x: x in original_jobs)

print(merged[['Job Title', 'Job_Title_in_df']])
df['Job Title'].unique()

# ---------------------------
# ✅ PDF Reader
# ---------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ---------------------------
# ✅ Prompt Builder
# ---------------------------
def build_resume_skill_prompt(resume_text):
    prompt = f"""
Extract only technical skills from the following resume.
Return the result as a clean Python list of skill names only.
Do not include soft skills, education, or interests.

Resume:
\"\"\"
{resume_text}
\"\"\"


Skills (Python list format):
"""
    return prompt



# ---------------------------
# ✅ LLM Generator Function
# ---------------------------
def generate_with_llm(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model_mistral.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# ✅ Skill List Extractor from Output
# ---------------------------
def extract_skills_from_output(output):
    """
    Extract the correct list of skills from a string containing a Python list.
    Safely skips early unrelated brackets like [link].
    """
    # Find ALL bracketed blocks
    matches = re.findall(r"\[(.*?)\]", output, re.DOTALL)

    if not matches:
        print("⚠ No list found in output.")
        print("Raw Output:\n", output)
        return []

    # Pick the longest match — likely to be the skills list
    best_match = max(matches, key=len)
    list_str = "[" + best_match.strip() + "]"

    try:
        # Fix any smart quotes (just in case)
        list_str = list_str.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        skills = ast.literal_eval(list_str)
        if isinstance(skills, list):
            return [skill.strip() for skill in skills if isinstance(skill, str)]
    except Exception as e:
        print("⚠ Failed to parse skills list:", e)
        print("Raw matched string:", list_str)
        return []

# ---------------------------
# ✅ Main Flow
# ---------------------------
def extract_skills_from_resume(pdf_path):
    resume_text = extract_text_from_pdf(pdf_path)
    prompt = build_resume_skill_prompt(resume_text)
    llm_output = generate_with_llm(prompt)
    #print(llm_output)
    skills = extract_skills_from_output(llm_output)
    print("✅ Extracted Skills:", skills)
    return skills


def extract_skills_from_bullets(raw_output):
    """
    Extract skills from lines starting with asterisk (*) in the given text.
    """
    skills = []
    lines = raw_output.splitlines()

    for line in lines:
        line = line.strip()
        if line.startswith("*"):
            skill = line.lstrip("*").strip()
            if skill:
                skills.append(skill)

    return skills
# ===== UTILS =====
def translate_text(text):
    try:
        return translator.translate(text)
    except:
        return text

def extract_skill_phrases(description_html):
    soup = BeautifulSoup(description_html, 'html.parser')
    lines = [li.get_text(strip=True) for li in soup.find_all('li')]
    lines += [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 20]

    if not lines:
        full_text = soup.get_text(separator=' ', strip=True)
        lines = [full_text[i:i+200] for i in range(0, len(full_text), 200)]

    unique_lines = list({line for line in lines if len(line.split()) >= 3})
    translated = [translate_text(line) for line in unique_lines]
    return translated

def semantic_score(job_lines, resume_skills):
    if not job_lines:
        return 0.0
    job_embeds = model.encode(job_lines, convert_to_tensor=True, device=device)
    resume_embeds = model.encode(resume_skills, convert_to_tensor=True, device=device)
    scores = util.cos_sim(resume_embeds, job_embeds)
    return scores.max(dim=1).values.mean().item()

def keyword_match_score(job, resume_skills):
    text = " ".join([
        job.get("title", ""),
        job.get("description", ""),
        " ".join(job.get("tags", []))
    ]).lower()
    return sum(skill.lower() in text for skill in resume_skills)

# ===== MAIN =====
def fetch_jobs():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ API Fetch Failed: {e}")
        return []

def find_top_jobs(jobs, resume_skills, keyword_top_k, semantic_top_k):
    # Step 1: Fast keyword filtering
    print("⚡ Scoring jobs by keyword match...")
    keyword_ranked = sorted(
        [(job, keyword_match_score(job, resume_skills)) for job in jobs],
        key=lambda x: x[1],
        reverse=True
    )[:keyword_top_k]
    # Step 2: Semantic scoring
    print("🧠 Doing semantic matching on top keyword matches...")
    final_ranked = []
    for job, kw_score in keyword_ranked:
        skill_lines = extract_skill_phrases(job.get("description", ""))
        sem_score = semantic_score(skill_lines, resume_skills)
        print(f"[DEBUG] {job['title']} | kw_score: {kw_score} | sem_score: {sem_score:.4f}")
        final_ranked.append((job, sem_score, skill_lines))

    return sorted(final_ranked, key=lambda x: x[1], reverse=True)[:semantic_top_k]

def save_results(jobs_ranked):
    if not jobs_ranked:
        print("⚠️ No good matches found.")
        return

    with open("semantic_top5_jobs.txt", "w", encoding="utf-8") as f:
        for i, (job, score, skill_lines) in enumerate(jobs_ranked, 1):
            f.write(f"Job {i}: {job['title']} at {job['company_name']} | Score: {score:.4f}\n")
            f.write(f"URL: {job['url']}\n")
            f.write("Matched Skill-Like Phrases:\n")
            for line in skill_lines:
                f.write(f"- {line}\n")
            f.write("\n" + "=" * 80 + "\n")

    print("\n✅ Top 5 jobs saved to 'semantic_top5_jobs.txt'.")

def predictionModel(job,job_content,Salary,Location,work_mode):
  # Dictionary with user inputs
  user_inputs = {
      "job_content": job_content,
      "Salary": Salary,
      "Location": Location,
      "work_mode": work_mode
  }

# Update Location with appended instruction
  user_inputs["Location"] += " — If the candidate is currently residing in the same location as the job, give more preference to remote work and companies offering relocation benefits."

  # Dictionary with static string keys and values
  job_analysis_queries = {
      "long_term": "Fetch the information about the Long term Prospects of the job. Include the company reputation and how easier it is to get promotion.",
      "environment": "Fetch the information about the company culture, the work culture, and work-life balance.",
      "Transport": "Using the Location Info, explain the transport situation.",
      "Facilities": "Fetch the facilities provided by the company."
  }
  preffered="\nConsider the following the prefferd outcomes for the corresponding attributes\n"+str(user_inputs)
  extract="\nfor each company and role extract the following attribute take the help of the description given next to it\n"+str(job_analysis_queries)
  classification1="Classify each company role to their original role for example if full stack developer in companyA and CompanyB fall under full stack the job given by the user should be ranked at the top."
  classification="\nthe following are classification \n Very important with highest priority:long_term,environment,Salary,fit ratio\nImportant with medium priority:Transport,Location,work_mode,Lowest priority:facility"
  Score="\nFor every attribute calculate the fit score for 10"
  prompt="Consder the following jobs"+job+preffered+extract+classification1+classification+Score+"\n Consider the above criteria assign appropriate weights for the corresponding attributes and calculate a final score and represent it in the output given below"
  output="\n the output format should be  in a table for every job in the decending order of recommendation with the columns Rank|Company name|role|longterm|Work Culture | salary|Transport|Facilities |Final Score \n role Faciities mention the perkes provided by the  company like dinner,gym,generous holiday \n mention the evaluation mectrics and weights used\n Explain the top 3 reccommeded jobsfor every attribute mention the jobs who could fit them but just missed out of top 3 under notables \n add a disclaimer "
  response = client.complete(
    messages=[
        SystemMessage("You are the best career advisor in the world"),
        UserMessage(prompt+output),
    ],
    temperature=0.1,
    top_p=0.5,
    model=model_gpt
  )
  return response.choices[0].message.content
def transformation(job):
  job_list = job.strip().split("\n================================================================================\n")
  print("Hello")

  job_string=""
  j=1
  for i in job_list:
    print("HI")
    response = client1.complete(
    messages=[
        SystemMessage("You have to convert stirng to dictionary "),
        UserMessage("""you are given a job description analyze that and give the output in the form of {
    "job_content": "Enter the job description",
    "long_term": "Fetch the information about the Long term Prospects of the job Include the company reputation and how easier it is to get prmotion",
    "Salary": "Enter the prefferrd salary range",
    "environment": "Fetch the information about the company culture the work culture and worklife balance",
    "Location": "Enter your current Location",
    "Transport": "Using the Location Info Explain the transport situation",
    "work_mode": "Enter your preffered work mode",
    "Facilities": "Fetch the facilities provided by the company"
  }dont add anything extra"""+i)],
    temperature=1,
    top_p=1,
    model=model1
    )
    job_string=job_string+response.choices[0].message.content
    print(j)
    j=j+1
  return job_string


def is_close_match(skill1, skill2, threshold=80):
    return fuzz.partial_ratio(skill1.lower(), skill2.lower()) >= threshold
# # ---------------------------

# # ---------------------------
# # ✅ Recommend Missing Skills
# # ---------------------------
def recommend_missing_skills(jd_skills, resume_skills, threshold=7, match_threshold=80):
    missing = []
    matched_weight = 0
    total_weight = 0

    for jd_skill, weight in jd_skills:
        if weight < threshold:
            continue  # skip low-importance skills
        total_weight += weight

        matched = any(is_close_match(jd_skill, r_skill, match_threshold) for r_skill in resume_skills)
        if matched:
            matched_weight += weight
        else:
            missing.append((jd_skill, weight))  # store skill with its weight

    # Sort missing skills by weight in descending order
    missing_sorted = [skill for skill, _ in sorted(missing, key=lambda x: x[1], reverse=True)]

    match_score = (matched_weight / total_weight) * 100 if total_weight > 0 else 0

    return missing_sorted, round(match_score, 2)




TOP_JOBS_FOR_PROMPT = 3
MAX_TOKENS = 3500
def fetch_jobs_2():
    # Return jobs from the merged DataFrame as dicts
    return merged.to_dict(orient='records')

def fuzzy_keyword_score(job, resume_terms, threshold=70):
    # Score based on partial fuzzy match of resume terms in job title
    title = job.get("Job Title", "").lower()
    return sum(fuzz.partial_ratio(term.lower(), title) >= threshold for term in resume_terms)

def job_title_search(query, resume_skills):
    print(f"\n🔍 Searching for: {query}")
    resume_terms = query.lower().split()

    print("📡 Fetching jobs...")
    jobs = fetch_jobs_2()
    if not jobs:
        print("No jobs fetched.")
        return [], 0

    print("⚡ Fuzzy keyword filtering...")
    fuzzy_ranked = sorted(
        [(job, fuzzy_keyword_score(job, resume_terms)) for job in jobs],
        key=lambda x: x[1],
        reverse=True
    )[:KEYWORD_TOP_K]

    if not fuzzy_ranked:
        print("No matching jobs found after fuzzy filtering.")
        return [], 0

    # Merge Key Skills of top N jobs for prompt
    seen_skills = set()
    merged_skills = []
    for job, score in fuzzy_ranked[:TOP_JOBS_FOR_PROMPT]:
        skills_str = job.get("Key Skills", "")
        for skill in skills_str.split(","):
            skill = skill.strip()
            if skill and skill.lower() not in seen_skills:
                seen_skills.add(skill.lower())
                merged_skills.append(skill)

    merged_text = ", ".join(merged_skills)

    # Truncate merged_text to avoid token overflow in prompt
    if len(merged_text) > MAX_TOKENS:
        merged_text = merged_text[:MAX_TOKENS]

    # Construct the prompt for the LLM
    prompt = f"""
You are given a job description. Extract only technical skills (if front end developer is there increase weights to react, css like that; do not directly include 'front end developer') mentioned and assign a score between 1 and 10 based on how important or central they are to the job.if the skills are similar like react.js and react keep only one

Return the result strictly in the following format (Python list of tuples):
[("skill1", score), ("skill2", score), ...]

Ignore soft skills, benefits, or company perks. Only return the skills with scores.

Job Description:
\"\"\"
{merged_text}
\"\"\"
"""

    try:
        response = client.complete(
            messages=[
                {"role": "system", "content": "You have to convert string to a Python list of (skill, score) tuples."},
                {"role": "user", "content": prompt}
            ],
            model=model_gpt,
            temperature=0.1,
            top_p=0.5
        )
    except Exception as e:
        print("❌ LLM completion failed:", e)
        return [], 0

    raw_output = response.choices[0].message.content.strip()
    jd_skills = []

    # Parse Python list of tuples from code block
    match = re.search(r"```python(.*?)```", raw_output, re.DOTALL)
    if match:
        try:
            skill_list = ast.literal_eval(match.group(1).strip())
            for skill, weight in skill_list:
                jd_skills.append((skill.strip(), int(weight)))
        except Exception as e:
            print("❌ Failed to parse skills:", e)
            return [], 0
    else:
        print("❌ No Python code block found in LLM output.")
        return [], 0

    # Now recommend missing skills compared to resume_skills (your existing function)
    recommendations, score = recommend_missing_skills(jd_skills, resume_skills)

    print("✅ Extracted Skills and Weights:")
    for skill, weight in jd_skills:
        print(f"{skill}: {weight}")

    return recommendations, score




# ================== EXAMPLE RUN ======================
# if __name__ == "__main__":

#     pdf_file_path = "6.pdf"  # Update path if needed
#     resume_skills=extract_skills_from_resume(pdf_file_path)
#     print("📡 Fetching jobs...")
#     job_content="front end developer"
#     Salary=20000
#     Location="berlin"
#     work_mode="remote"
#     jobs = fetch_jobs()
#     top5 = find_top_jobs(jobs, resume_skills, keyword_top_k=KEYWORD_TOP_K, semantic_top_k=SEMANTIC_TOP_K)
#     save_results(top5)
#     with open("semantic_top5_jobs.txt", "r", encoding="utf-8") as file:
#         job = file.read()
#     job_string = transformation(job)
#     predictionModel(job_string,job_content,Salary,Location,work_mode)
#     user_query = "Full stack developer"
#     start = time.time()
#     recommendations=job_title_search(user_query,resume_skills)
#     print("\n✅ Recommend learning these skills:", recommendations)
#     print(f"⏱ Done in {time.time() - start:.2f} seconds.")



def job_recommendation_pipeline(resume_pdf, salary, location, work_mode, job_query):
    try:
        # Step 1: Copy uploaded resume file
        shutil.copy(resume_pdf.name, "resume.pdf")
        pdf_path = "resume.pdf"

        # Step 2: Extract skills from resume
        resume_skills = extract_skills_from_resume(pdf_path)

        # Step 3: Fetch jobs and rank
        jobs = fetch_jobs()
        top_jobs = find_top_jobs(jobs, resume_skills, keyword_top_k=20, semantic_top_k=10)
        save_results(top_jobs)

        # Step 4: Read top jobs and convert to structured data
        with open("semantic_top5_jobs.txt", "r", encoding="utf-8") as file:
            job_text = file.read()
        job_string = transformation(job_text)

        # Step 5: Predict fit based on user preferences
        final_evaluation = predictionModel(job_string, job_query,salary, location, work_mode)

        return final_evaluation

    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def analyze_resume_for_title(resume_pdf, job_title):
  try:
        # Step 1: Copy uploaded resume file
        shutil.copy(resume_pdf.name, "resume.pdf")
        pdf_path = "resume.pdf"

        # Step 2: Extract skills from resume
        resume_skills = extract_skills_from_resume(pdf_path)
        recommendations=job_title_search(job_title,resume_skills)
        recommendation_list, score = recommendations
        return recommendation_list
  except Exception as e:
        return f"❌ Error: {str(e)}", ""

with gr.Blocks(title="Resume-Based Job Recommender") as demo:
    gr.Markdown("## 📄 Resume-Based Job Recommender with LLM Matching")

    # === Section 1: Job Recommender ===
    with gr.Tab("🔍 Job Recommendations"):
        with gr.Row():
            resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])

        with gr.Row():
            salary = gr.Textbox(label="Preferred Salary Range", placeholder="e.g. 100000")
            location = gr.Textbox(label="Current Location", placeholder="e.g. Bangalore")
            work_mode = gr.Radio(["Remote", "Hybrid", "Onsite"], label="Preferred Work Mode")

        job_query = gr.Textbox(label="Job Title to Search", placeholder="e.g. Data Scientist")

        submit_btn = gr.Button("🔍 Find Best Job Matches")

        status_md = gr.Markdown(value="", visible=False)
        output_1 = gr.Markdown(label="📊 Output")

        def wrapped_job_pipeline(resume_pdf, salary, location, work_mode, job_query):
            yield gr.update(value="⏳ Matching jobs, please wait...", visible=True), ""
            result = job_recommendation_pipeline(resume_pdf, salary, location, work_mode, job_query)
            yield gr.update(value="✅ Done!", visible=True), result


        submit_btn.click(fn=wrapped_job_pipeline,
                         inputs=[resume_pdf, salary, location, work_mode, job_query],
                         outputs=[status_md, output_1])

    # === Section 2: Resume Analyzer by Job Title ===
    with gr.Tab("🧠 Analyze Resume for Job Title"):
        with gr.Row():
            resume_pdf_2 = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
            job_title_input = gr.Textbox(label="Job Title", placeholder="e.g. Data Analyst")

        analyze_btn = gr.Button("Analyze Resume")
        status_md_2 = gr.Markdown(value="", visible=False)
        output_2 = gr.Markdown(label="📊 Output")

        def wrapped_resume_analyzer(resume_pdf_2, job_title_input):
            status_message = "🔍 Analyzing resume..."
            result = analyze_resume_for_title(resume_pdf_2, job_title_input)
            status_message = "✅ Done!"
            return status_message, result

        analyze_btn.click(fn=wrapped_resume_analyzer,
                          inputs=[resume_pdf_2, job_title_input],
                          outputs=[status_md_2, output_2])

demo.launch(share=True,debug=True)
