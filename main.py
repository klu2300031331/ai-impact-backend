"""
AI Driven Impact Analyzer - FastAPI Backend
Uses Groq (Llama 3) for AI analysis with rule-based fallback.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import re
import os
import json
from typing import Optional

app = FastAPI(title="AI Impact Analyzer", version="2.1.0")

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
    analysis_mode: str  # "ai" or "fallback" — tells frontend which mode was used

# --- Rule-Based Fallback Logic ---

MODULE_RULES = {
    "payment":    "Payment Module",
    "auth":       "Authentication Module",
    "order":      "Order Management Module",
    "database":   "Database Layer",
    "db":         "Database Layer",
    "service":    "Service Layer",
    "controller": "Controller Layer",
}

TEST_SUGGESTIONS = {
    "Payment Module":          ["Test payment success flow", "Test refund logic", "Test payment gateway timeout handling"],
    "Authentication Module":   ["Test login with valid credentials", "Test token validation", "Test session expiry"],
    "Order Management Module": ["Test order creation flow", "Test order cancellation", "Test order status transitions"],
    "Database Layer":          ["Test read/write operations", "Test transaction rollback", "Test connection pooling"],
    "Service Layer":           ["Test service contract validation", "Test service failure handling"],
    "Controller Layer":        ["Test API endpoint responses", "Test request validation"],
    "Core Application Module": ["Run full regression suite", "Test integration touchpoints"],
}

def rule_based_analysis(changed_files: list[str]) -> dict:
    """
    Fallback: rule-based module detection and risk scoring.
    Used when Groq API is unavailable or rate limited.
    """
    def detect_module(filename: str) -> str:
        lower = filename.lower()
        for keyword, module in MODULE_RULES.items():
            if keyword in lower:
                return module
        return "Core Application Module"

    modules = list(dict.fromkeys(detect_module(f) for f in changed_files))

    tests = []
    seen = set()
    for module in modules:
        for test in TEST_SUGGESTIONS.get(module, []):
            if test not in seen:
                tests.append(test)
                seen.add(test)

    # Risk scoring
    risk = "LOW"
    reasons = []

    if len(changed_files) > 5:
        risk = "HIGH"
        reasons.append(f"{len(changed_files)} files modified — large changeset increases regression risk.")
    if "Database Layer" in modules:
        risk = "HIGH"
        reasons.append("Database Layer affected — schema or query changes carry elevated risk.")
    if risk != "HIGH" and len(modules) >= 2:
        risk = "MEDIUM"
        reasons.append(f"Changes span {len(modules)} modules, requiring cross-module validation.")
    if risk == "LOW":
        reasons.append("Minimal scope — single module with few files. Standard smoke tests should suffice.")

    explanation = (
        f"[Fallback Mode] Risk assessed as {risk}. "
        + " ".join(reasons)
        + f" Impacted modules: {', '.join(modules)}."
        + " Note: AI analysis was unavailable, using rule-based detection."
    )

    return {
        "impacted_modules": modules,
        "suggested_tests": tests,
        "risk_level": risk,
        "ai_explanation": explanation,
    }

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
    Raises an exception if Groq is unavailable — caller handles fallback.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not set")

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
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": "You are a senior software engineering analyst. You always respond with valid JSON only, no markdown formatting, no extra text."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

    if resp.status_code == 429:
        raise ValueError("Groq rate limit exceeded")
    if resp.status_code != 200:
        raise ValueError(f"Groq API returned {resp.status_code}")

    ai_text = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown backticks if model accidentally adds them
    ai_text = re.sub(r"^```json\s*", "", ai_text)
    ai_text = re.sub(r"^```\s*", "", ai_text)
    ai_text = re.sub(r"\s*```$", "", ai_text)

    result = json.loads(ai_text)  # Let caller catch JSONDecodeError

    for field in ["impacted_modules", "suggested_tests", "risk_level", "ai_explanation"]:
        if field not in result:
            raise ValueError(f"AI response missing field: {field}")

    result["risk_level"] = result["risk_level"].upper()
    if result["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
        result["risk_level"] = "MEDIUM"

    return result

# --- Main Endpoint ---

@app.post("/analyze-pr", response_model=AnalysisResponse)
async def analyze_pr(request: PRRequest):
    """
    Analyze a GitHub Pull Request.
    Tries Groq AI first — silently falls back to rule-based logic if unavailable.
    """
    owner, repo, pr_number = parse_pr_url(request.pr_url)
    changed_files = await fetch_pr_files(owner, repo, pr_number, request.github_token)

    if not changed_files:
        raise HTTPException(status_code=422, detail="No changed files found in this PR.")

    # Try Groq AI first
    try:
        result = await analyze_with_groq(changed_files)
        analysis_mode = "ai"
        print(f"✅ Groq AI analysis successful for PR with {len(changed_files)} files")

    except Exception as e:
        # Silently fall back to rule-based logic
        print(f"⚠️  Groq unavailable ({e}), falling back to rule-based analysis")
        result = rule_based_analysis(changed_files)
        analysis_mode = "fallback"

    return AnalysisResponse(
        changed_files=changed_files,
        impacted_modules=result["impacted_modules"],
        suggested_tests=result["suggested_tests"],
        risk_level=result["risk_level"],
        ai_explanation=result["ai_explanation"],
        analysis_mode=analysis_mode,
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai": "groq/llama-3.3-70b-versatile",
        "fallback": "rule-based"
    }