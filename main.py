"""
AI Driven Impact Analyzer - FastAPI Backend
Analyzes GitHub Pull Requests for impact assessment and risk scoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import re
from typing import Optional

app = FastAPI(title="AI Impact Analyzer", version="1.0.0")

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
    github_token: Optional[str] = None  # Optional: avoids GitHub rate limits

class AnalysisResponse(BaseModel):
    changed_files: list[str]
    impacted_modules: list[str]
    suggested_tests: list[str]
    risk_level: str
    ai_explanation: str

# --- Module Detection Logic ---

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
    "Payment Module":        ["Test payment success flow", "Test refund logic", "Test payment gateway timeout handling"],
    "Authentication Module": ["Test login with valid credentials", "Test token validation", "Test session expiry"],
    "Order Management Module": ["Test order creation flow", "Test order cancellation", "Test order status transitions"],
    "Database Layer":        ["Test read/write operations", "Test transaction rollback", "Test connection pooling"],
    "Service Layer":         ["Test service contract validation", "Test service failure handling"],
    "Controller Layer":      ["Test API endpoint responses", "Test request validation"],
    "Core Application Module": ["Run full regression suite", "Test integration touchpoints"],
}

def detect_module(filename: str) -> str:
    """Map a filename to its business module using keyword rules."""
    lower = filename.lower()
    for keyword, module in MODULE_RULES.items():
        if keyword in lower:
            return module
    return "Core Application Module"

def get_test_cases(modules: list[str]) -> list[str]:
    """Gather unique test cases for all affected modules."""
    tests = []
    seen = set()
    for module in modules:
        for test in TEST_SUGGESTIONS.get(module, []):
            if test not in seen:
                tests.append(test)
                seen.add(test)
    return tests

def calculate_risk(changed_files: list[str], modules: list[str]) -> tuple[str, str]:
    """
    Determine risk level and generate a human-readable explanation.
    Rules:
      - More than 5 files changed → HIGH
      - Database Layer affected  → HIGH
      - 2+ distinct modules      → MEDIUM
      - Otherwise                → LOW
    """
    reasons = []
    risk = "LOW"

    if len(changed_files) > 5:
        risk = "HIGH"
        reasons.append(f"{len(changed_files)} files were modified — large changesets increase regression probability.")

    if "Database Layer" in modules:
        risk = "HIGH"
        reasons.append("Database Layer changes detected — schema or query modifications carry elevated risk.")

    if risk != "HIGH" and len(modules) >= 2:
        risk = "MEDIUM"
        reasons.append(f"Changes span {len(modules)} modules ({', '.join(modules)}), requiring cross-module validation.")

    if risk == "LOW":
        reasons.append("Minimal scope: single module with few files changed. Standard smoke tests should suffice.")

    explanation = (
        f"Risk assessed as {risk}. "
        + " ".join(reasons)
        + f" Impacted modules: {', '.join(modules)}."
        + " Review suggested test cases before merging."
    )
    return risk, explanation

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
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers)

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="PR not found. Check the URL and ensure the repo is public.")
    if resp.status_code == 403:
        raise HTTPException(status_code=403, detail="GitHub API rate limit exceeded. Provide a GitHub token.")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"GitHub API error: {resp.text}")

    return [f["filename"] for f in resp.json()]

# --- Main Endpoint ---

@app.post("/analyze-pr", response_model=AnalysisResponse)
async def analyze_pr(request: PRRequest):
    """
    Analyze a GitHub Pull Request and return:
    - Changed files
    - Impacted business modules
    - Suggested regression tests
    - Risk level (LOW / MEDIUM / HIGH)
    - Human-readable explanation
    """
    owner, repo, pr_number = parse_pr_url(request.pr_url)
    changed_files = await fetch_pr_files(owner, repo, pr_number, request.github_token)

    if not changed_files:
        raise HTTPException(status_code=422, detail="No changed files found in this PR.")

    # Detect modules (deduplicated, ordered)
    modules = list(dict.fromkeys(detect_module(f) for f in changed_files))
    suggested_tests = get_test_cases(modules)
    risk_level, ai_explanation = calculate_risk(changed_files, modules)

    return AnalysisResponse(
        changed_files=changed_files,
        impacted_modules=modules,
        suggested_tests=suggested_tests,
        risk_level=risk_level,
        ai_explanation=ai_explanation,
    )

@app.get("/health")
def health():
    return {"status": "ok"}
