# AI-Powered Talent Sourcing using Ollama

This repository contains two Python scripts that form a single workflow to automate job description parsing and candidate evaluation using **Ollama** (local LLM). Each script uses an **AI prompt** to generate more accurate and contextual results.

---

## 1. JD_to_config.py
**Purpose:** Reads a Job Description (PDF) and extracts structured hiring requirements using Ollama.

**Output:**
- `config_google.json` — for Google/X-Ray search
- `config_recruiter.json` — for LinkedIn Recruiter filters
- `xray_boolean.txt` — Boolean search string

**Run:**
```bash
python JD_to_config.py --pdf JD.pdf --model llama3.1:latest
```

---

## 2. process_and_evaluate.py
**Purpose:** Reads a LinkedIn-style Excel, infers candidate details, gives each a score (0–100), and generates personalized messages for the top candidates.

**Output:** Updates the same Excel file by adding new columns like:
- Inferred experience, seniority, industry, summary
- Score, reason for score, personalized message

**Run:**
```bash
python process_and_evaluate.py
```

---

## Requirements
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally (model: `llama3.1:latest`)
- Packages:
  ```bash
  pip install pypdf pandas langchain-ollama langchain-core
  ```

---

## Notes
- Each script uses a **custom AI prompt** in Ollama for structured inference and evaluation.
- All data stays local — no cloud API calls.
- Adjust model names in the scripts if you prefer (`qwen2`, `mistral`, etc.).

---

## Quick Flow
1. Run `JD_to_config.py` to create structured config files from the JD.
2. Run `process_and_evaluate.py` to evaluate profiles and rank candidates.
3. Review the same Excel for results.

---
## 🎥 Demo Video

https://github.com/abhinav0000004/AI-Powered-Talent-Sourcing-Ollama-/blob/main/Demo%20Video.mp4


Built for quick, private, and iterative AI-powered talent sourcing.
