'''
jd_to_configs.py
----------------
This script reads a job description (JD) from a PDF file, uses a local Ollama LLM
to extract structured hiring requirements, and generates configuration files for
various platforms (e.g., Google, Recruiter).

Usage (Windows, inside your venv):
python jd_to_configs.py --pdf JD.pdf 

Outputs:
- config_google.json : File later used to build Google X-Ray search strings and evaluation
- config_recruiter.json : File for LinkedIn Recruiter filters
- xray_boolean.txt:   File containing the X-Ray boolean search string

'''
import os
import re
import json
import argparse
from typing import Any, Dict, List

from pypdf import PdfReader #Reads PDF files
from langchain_ollama import ChatOllama #Interface to Ollama local LLMs
from langchain_core.messages import SystemMessage, HumanMessage

def read_pdf_text(pdf_path: str, max_chars: int = 60000) -> str:
    #This function reads text from a PDF file and returns the first max_chars characters.
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find PDF at: {pdf_path}")
    r = PdfReader(pdf_path)
    text = "\n".join((p.extract_text() or "") for p in r.pages).strip()
    return text[:max_chars]


def coerce_json(raw: str) -> Dict[str, Any]:
    # This function attempts to parse JSON from a raw string, handling common formatting issues.
    s = raw.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(s)
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end+1])
    raise ValueError("Model did not return parseable JSON. First 500 chars:\n" + s[:500])

def norm_list(x):
    # Normalize input to a list of unique, non-empty strings.
    if not x: return []
    if isinstance(x, list):
        out, seen = [], set()
        for i in x:
            s = str(i).strip()
            if not s: continue
            k = s.lower()
            if k not in seen:
                seen.add(k); out.append(s)
        return out
    s = str(x).strip()
    return [s] if s else []

def build_google_config(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Build Google config from base configuration.
    return {
        "role": base_cfg.get("role", ""),
        "title_keywords": norm_list(base_cfg.get("title_keywords")),
        "must_skills": norm_list(base_cfg.get("must_skills")),
        "nice_skills": norm_list(base_cfg.get("nice_skills")),
        "min_years": int(base_cfg.get("min_years") or 0),
        "locations": norm_list(base_cfg.get("locations")),
        "domains": norm_list(base_cfg.get("domains")),          # kept in JSON
        "exclude_keywords": norm_list(base_cfg.get("exclude_keywords")),
        "languages": norm_list(base_cfg.get("languages")),      # kept in JSON
        "education": norm_list(base_cfg.get("education")),
        "seniority": (base_cfg.get("seniority") or "").title() or "Senior",
    }

def _sanitize(term: str) -> str:
    # Remove quotes, parentheses, and extra whitespace from a term.
    s = re.sub(r'["()]+', " ", term or "")
    return re.sub(r"\s+", " ", s).strip()

def _unique(vals: List[str]) -> List[str]:
    # Return a list of unique strings, preserving order and ignoring case.
    seen, out = set(), []
    for v in vals:
        k = v.lower()
        if v and k not in seen:
            seen.add(k); out.append(v)
    return out

def _or_block(vals: List[str]) -> str:
    # Create an OR block from a list of terms.
    vals = _unique([_sanitize(v) for v in vals if _sanitize(v)])
    return "(" + " OR ".join(f'"{v}"' for v in vals) + ")" if vals else ""

def _and_block(vals: List[str]) -> str:
    # Create an AND block from a list of terms.
    vals = _unique([_sanitize(v) for v in vals if _sanitize(v)])
    return "(" + " AND ".join(f'"{v}"' for v in vals) + ")" if vals else ""

def build_xray_boolean(cfg: Dict[str, Any]) -> str:
    # Build an X-Ray boolean search string from the Google config.
    site = "site:linkedin.com/in"

    role       = (cfg.get("role") or "").strip()
    seniority  = (cfg.get("seniority") or "").strip()
    titles     = cfg.get("title_keywords", []) or []
    musts      = cfg.get("must_skills", []) or []
    nice       = cfg.get("nice_skills", []) or []
    locs       = cfg.get("locations", []) or []
    domains    = cfg.get("domains", []) or []
    languages  = cfg.get("languages", []) or []
    education  = cfg.get("education", []) or []
    nots       = cfg.get("exclude_keywords", []) or []

    # Enrich title block with role and seniority-role combo if present
    title_candidates = list(titles)
    if role:
        title_candidates.append(role)
    if role and seniority:
        title_candidates.append(f"{seniority} {role}")

    parts: List[str] = [site]

    # Titles boosted in page title
    tblock = _or_block(title_candidates)
    if tblock:
        parts.append(f"intitle:{tblock}")

    # Must-have skills as strict AND
    mblock = _and_block(musts)
    if mblock:
        parts.append(mblock)

    # Optional OR blocks (kept separate for clarity)
    nblock = _or_block(nice)
    if nblock:
        parts.append(nblock)

    lblock = _or_block(locs)
    if lblock:
        parts.append(lblock)

    dblock = _or_block(domains)
    if dblock:
        parts.append(dblock)

    langblock = _or_block(languages)
    if langblock:
        parts.append(langblock)

    edu_block = _or_block(education)
    if edu_block:
        parts.append(edu_block)

    # Exclusions
    xblock = _or_block(nots)
    if xblock:
        parts.append(f"-{xblock}")

    q = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", q).strip()

def build_recruiter_config(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Build Recruiter config from base configuration.
    return {
        "filters": {
            "titles": norm_list(base_cfg.get("title_keywords")),
            "skills_must": norm_list(base_cfg.get("must_skills")),
            "skills_nice": norm_list(base_cfg.get("nice_skills")),
            "min_years_experience": int(base_cfg.get("min_years") or 0),
            "locations": norm_list(base_cfg.get("locations")),
            "domains": norm_list(base_cfg.get("domains")),
            "languages": norm_list(base_cfg.get("languages")),
            "education": norm_list(base_cfg.get("education")),
            "seniority": (base_cfg.get("seniority") or "").title() or "Senior",
            "exclude_keywords": norm_list(base_cfg.get("exclude_keywords")),
            "open_to_work_preferred": True,
            "recent_joiner_exclude_months": 3,
        },
        "notes": "Map these to LinkedIn Recruiter UI filters or a partner API payload.",
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate Google & Recruiter configs from a JD PDF using a local Ollama model.")
    ap.add_argument("--pdf", default="JD.pdf")
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    jd_text = read_pdf_text(args.pdf)

    system_msg = SystemMessage(content=(
    "You are an expert HR and talent intelligence assistant. "
    "Read the job description carefully and extract structured hiring requirements. "
    "Return ONLY valid, compact JSON — no text, no markdown, no comments.\n\n"
    "JSON Schema:\n"
    "role: string,\n"
    "title_keywords: string[],\n"
    "must_skills: string[],\n"
    "nice_skills: string[],\n"
    "min_years: integer,\n"
    "locations: string[],\n"
    "domains: string[],\n"
    "exclude_keywords: string[],\n"
    "languages: string[],\n"
    "education: string[],\n"
    "seniority: one of ['Junior','Mid','Senior','Lead','Principal']\n\n"
    "Guidelines:\n"
    "- If a field is not explicitly mentioned, infer it logically based on job scope, seniority cues, and context.\n"
    "- Infer 'min_years' conservatively when missing\n"
    "- Determine 'seniority' from cues like 'lead', 'architect', 'mentor', 'manage team' (higher) "
    "or 'assist', 'support', 'entry-level' (lower).\n"
    "- Extract 2–8 precise 'title_keywords' explicitly found or strongly implied (e.g., 'Data Scientist', 'ML Engineer').\n"
    "- Include only hard technical or domain skills in 'must_skills' (e.g., Python, PyTorch, SQL). "
    "Use 'nice_skills' for preferred or secondary tools.\n"
    "- Use only explicit or clearly implied information for locations, domains, languages, and education.\n"
    "- Exclude generic soft skills (communication, teamwork) and non-relevant text.\n"
    "- Always return valid JSON; do not explain your reasoning."
    ))
    human_msg  = HumanMessage(content=f"JD:\n{jd_text}\n\nReturn JSON only with the relevant keys described above. Use your knowledge to fill in any missing fields.")

    llm = ChatOllama(model=args.model)

    tries = 0
    base_cfg = None
    while base_cfg is None and tries < 2:
        tries += 1
        response = llm.invoke([system_msg, human_msg])
        try:
            base_cfg = coerce_json(response.content)
        except Exception:
            if tries >= 2: raise
            human_msg = HumanMessage(content=f"Your previous output was not valid JSON. Re-emit STRICT JSON only.\n\nJD:\n{jd_text}")

    google_cfg   = build_google_config(base_cfg)
    xray         = build_xray_boolean(google_cfg)
    recruiter_cfg= build_recruiter_config(base_cfg)

    with open(os.path.join(args.outdir, "config_google.json"), "w", encoding="utf-8") as f:
        json.dump(google_cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "config_recruiter.json"), "w", encoding="utf-8") as f:
        json.dump(recruiter_cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "xray_boolean.txt"), "w", encoding="utf-8") as f:
        f.write(xray)

    print("Wrote:\n  config_google.json\n  config_recruiter.json\n  xray_boolean.txt")

if __name__ == "__main__":
    main()