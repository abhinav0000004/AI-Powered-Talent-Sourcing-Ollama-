'''
rank_and_message.py
This script reads candidate data from a LinkedIn-style Excel file 
and uses a local Ollama LLM to extract key information such as years of experience, seniority level, industry background, 
and a short summary for each profile. It then evaluates and ranks candidates based on multiple weighted criteria 
from the job description (JD) and generates personalized outreach messages for the top 5 profiles.

Inputs:
Mock_LinkedIn_Data.xlsx : Excel file containing candidate information
config_google.json : JD configuration file with hiring criteria

Outputs:
Updated Mock_LinkedIn_Data.xlsx : Same Excel file enriched with new columns —
Inferred Years of Experience, Inferred Seniority, Inferred Industry Experience, Summary, Score, Reason for Score, and Personalized Message
'''

import json, re
import time
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

MODEL = "llama3.1:latest"
EXCEL = "Mock_LinkedIn_Data.xlsx"
JD_JSON = "config_google.json"

ADD_COLS = [
    "Inferred Years of Experience","Inferred Seniority","Inferred Industry Experience",
    "Summary","Score","Reason for Score","Personalized Message"
]

SYSTEM = """You are an expert HR / talent intelligence assistant.
Given a JD and a LinkedIn-like profile, infer:
- inferred_years_experience (float, e.g., 5.5)
- inferred_seniority (one of: Junior, Mid, Senior, Lead, Principal)
- inferred_industry (short label, e.g., Fintech, E-commerce, AI Research)
- summary (3–4 short bullets)

Then compute a STRICT weighted score (0–100) using these weights:
- Title/Role match: 15
- Must-skill coverage: 35
- Nice-skill coverage: 10
- Years vs min_years: 15
- Seniority alignment: 10
- Industry/Domain fit: 10
- Location/Language fit: 5

Rules:
- Be harsh: weak or missing must-skills should drag down the total.
- If a JD criterion is missing, auto-rebalance remaining weights to still sum to 100.
- Provide a per-criterion breakdown with each criterion's: {score: 0–100, weight: %, points: contribution}

Return ONLY valid minified JSON with EXACT keys:
{ "inferred_years_experience": float,
  "inferred_seniority": "Junior|Mid|Senior|Lead|Principal",
  "inferred_industry": string,
  "summary": string,
  "score": float,
  "reason_for_score": { "<criterion>": {"score": float, "weight": float, "points": float}, ... }
}
"""

def to_list(x):
    if isinstance(x, list): return [str(s).strip() for s in x if str(s).strip()]
    if isinstance(x, str):  return [s.strip() for s in re.split(r"[,\n;/]+", x) if s.strip()]
    return []

def infer_and_score(llm, jd, prof):
    prompt = f"""JD:\n{json.dumps(jd, ensure_ascii=False)}\n\nPROFILE:\n{json.dumps(prof, ensure_ascii=False)}"""
    try:
        r = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]).content.strip()
        r = r.replace("```json","").replace("```","").strip()
        data = json.loads(r)
        # minimal sanity defaults
        data.setdefault("inferred_years_experience","")
        data.setdefault("inferred_seniority","")
        data.setdefault("inferred_industry","")
        data.setdefault("summary","")
        data.setdefault("score",0)
        data.setdefault("reason_for_score",{})
        return data
    except Exception as e:
        return {
            "inferred_years_experience": "",
            "inferred_seniority": "",
            "inferred_industry": "",
            "summary": "LLM parse failed.",
            "score": 0,
            "reason_for_score": {"error":{"score":0,"weight":0,"points":0}}
        }

def main():
    df = pd.read_excel(EXCEL)
    jd = json.load(open(JD_JSON,"r",encoding="utf-8"))
    llm = ChatOllama(model=MODEL)

    # ensure new columns exist; won't touch other columns
    for c in ADD_COLS:
        if c not in df.columns: df[c] = ""

    for i, row in df.iterrows():
        prof = {
            "Full Name": row.get("Full Name",""),
            "Headline": row.get("Headline",""),
            "About": row.get("About",""),
            "Current Company": row.get("Current Company",""),
            "Current Role": row.get("Current Role",""),
            "Entire Work History": row.get("Entire Work History",""),
            "Skills": row.get("Skills",""),
            "Language": row.get("Language",""),
            "Location": row.get("Location",""),
        }
        out = infer_and_score(llm, jd, prof)

        df.at[i,"Inferred Years of Experience"]   = out["inferred_years_experience"]
        df.at[i,"Inferred Seniority"]             = out["inferred_seniority"]
        df.at[i,"Inferred Industry Experience"]   = out["inferred_industry"]
        df.at[i,"Summary"]                        = out["summary"]
        df.at[i,"Score"]                          = out["score"]
        df.at[i,"Reason for Score"]               = json.dumps(out["reason_for_score"], ensure_ascii=False)
        print(f"✅ {i+1}/{len(df)} — {prof.get('Full Name','Candidate')} → {out['score']}")
        time.sleep(5)  # wait 5 seconds before processing the next candidate

    # Top 5 personalized messages
    tmp = df.copy()
    tmp["__score__"] = pd.to_numeric(tmp["Score"], errors="coerce").fillna(0)
    top_idx = tmp.sort_values("__score__", ascending=False).head(5).index
    role = jd.get("role","an open position")
    for i in top_idx:
        name = df.at[i,"Full Name"] if "Full Name" in df.columns else "there"
        highlights = df.at[i,"Summary"]
        df.at[i,"Personalized Message"] = (
            f"Hi {name},\n\n"
            f"We came across your profile and were impressed by your background — especially {highlights}.\n"
            f"We’re currently hiring for {role}. Would you be open to a quick chat?\n"
            f"https://cal.com/your-team/intro-15\n\nBest,\nMerantix Hiring Team"
        )

    df = df.sort_values("Score", ascending=False)
    df.to_excel(EXCEL, index=False)
    print(f"\n✅ Updated: {EXCEL} (scores + top-5 messages)")

if __name__ == "__main__":
    main()
