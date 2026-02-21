"""
AI Driven Impact Analyzer - FastAPI Backend
Uses Groq (Llama 3) to intelligently analyze GitHub Pull Requests.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import re
import os
import json
from typing import Optional

app = FastAPI(title="AI Impact Analyzer", version="2.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Request / Response Models ---

class PRRequest(BaseModel):
    pr_url: str
    github_token: Optional[str] = None

class AnalysisResponse(BaseModel):
    changed_files: list[str]
    impacted_modules: list[str]
    suggested_tests: list[str]
    risk_level: str
    ai_explanation: str

# --- GitHub API Helper ---

def parse_pr_url(pr_url: str) -> tuple[str, str, str]:
    """Extract owner, repo, and PR number from a GitHub PR URL."""
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, pr_url)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Invalid GitHub PR URL. Expected format: https://github.com/{owner}/{repo}/pull/{number}"
        )
    return match.group(1), match.group(2), match.group(3)

async def fetch_pr_files(owner: str, repo: str, pr_number: str, token: Optional[str]) -> list[str]:
    """Call GitHub REST API to retrieve changed files for a given PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {"Accept": "application/vnd.github+json"}

    # Use token from request, fall back to environment variable
    resolved_token = token or os.getenv("GITHUB_TOKEN")
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers)

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="PR not found. Check the URL and ensure the repo is public.")
    if resp.status_code == 403:
        raise HTTPException(status_code=403, detail="GitHub API rate limit exceeded. Provide a GitHub token.")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"GitHub API error: {resp.text}")

    return [f["filename"] for f in resp.json()]

# --- Groq AI Analysis ---

async def analyze_with_groq(changed_files: list[str]) -> dict:
    """
    Send changed files to Groq (Llama 3) for intelligent impact analysis.
    Returns structured JSON with modules, tests, risk level, and explanation.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server.")

    # Build a detailed prompt for structured JSON output
    prompt = f"""You are an expert software engineering analyst. Analyze the following list of changed files from a GitHub Pull Request and return a structured impact analysis.

Changed files:
{json.dumps(changed_files, indent=2)}

Your task:
1. Identify which business modules are impacted based on the file paths and names
2. Suggest specific regression test cases that should be run
3. Calculate the risk level (LOW, MEDIUM, or HIGH) based on:
   - Number of files changed (>5 = higher risk)
   - Sensitivity of modules affected (auth, payment, database = higher risk)
   - Number of modules affected (more modules = higher risk)
4. Write a clear, concise explanation of the risk for an engineering team

Respond ONLY with a valid JSON object in this exact format, no extra text:
{{
  "impacted_modules": ["Module Name 1", "Module Name 2"],
  "suggested_tests": ["Test case 1", "Test case 2", "Test case 3"],
  "risk_level": "HIGH",
  "ai_explanation": "A clear 2-3 sentence explanation of the risk and what the team should watch out for before merging."
}}

Rules:
- risk_level must be exactly one of: LOW, MEDIUM, HIGH
- impacted_modules should be human-readable business module names
- suggested_tests should be specific and actionable, not generic
- ai_explanation should mention specific files or patterns you noticed
- Return valid JSON only, no markdown, no backticks"""

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",  # Fast and capable Groq model
        "messages": [
            {
                "role": "system",
                "content": "You are a senior software engineering analyst. You always respond with valid JSON only, no markdown formatting, no extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,       # Low temperature for consistent, structured output
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq API error: {resp.text}")

    # Extract the AI response text
    ai_text = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown backticks if model accidentally adds them
    ai_text = re.sub(r"^```json\s*", "", ai_text)
    ai_text = re.sub(r"^```\s*", "", ai_text)
    ai_text = re.sub(r"\s*```$", "", ai_text)

    try:
        result = json.loads(ai_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Please try again.")

    # Validate required fields exist
    for field in ["impacted_modules", "suggested_tests", "risk_level", "ai_explanation"]:
        if field not in result:
            raise HTTPException(status_code=500, detail=f"AI response missing field: {field}")

    # Normalize risk level to uppercase
    result["risk_level"] = result["risk_level"].upper()
    if result["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
        result["risk_level"] = "MEDIUM"  # Safe fallback

    return result

# --- Main Endpoint ---

@app.post("/analyze-pr", response_model=AnalysisResponse)
async def analyze_pr(request: PRRequest):
    """
    Analyze a GitHub Pull Request using Groq AI (Llama 3) and return:
    - Changed files (from GitHub API)
    - Impacted business modules (AI detected)
    - Suggested regression tests (AI generated)
    - Risk level: LOW / MEDIUM / HIGH (AI calculated)
    - Human-readable AI explanation
    """
    # Step 1: Parse and validate PR URL
    owner, repo, pr_number = parse_pr_url(request.pr_url)

    # Step 2: Fetch changed files from GitHub
    changed_files = await fetch_pr_files(owner, repo, pr_number, request.github_token)

    if not changed_files:
        raise HTTPException(status_code=422, detail="No changed files found in this PR.")

    # Step 3: Send to Groq AI for full analysis
    ai_result = await analyze_with_groq(changed_files)

    return AnalysisResponse(
        changed_files=changed_files,
        impacted_modules=ai_result["impacted_modules"],
        suggested_tests=ai_result["suggested_tests"],
        risk_level=ai_result["risk_level"],
        ai_explanation=ai_result["ai_explanation"],
    )

@app.get("/health")
def health():
    return {"status": "ok", "ai": "groq/llama-3.3-70b-versatile"}