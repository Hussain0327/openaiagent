import os
import glob
import math
import re
import hashlib
import csv
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool

# =========================================================
# Pydantic models (structured output validated by the SDK)
# =========================================================

class Citation(BaseModel):
    source: str = Field(..., description="File path or doc id")
    span: str = Field(..., description="Short quoted passage used as evidence")
    score: float = Field(..., description="Relevance score in [0,1]")

class EvidenceRef(BaseModel):
    uri: str
    hash: Optional[str] = None

class ControlFinding(BaseModel):
    control_id: str
    status: str = Field(..., description="One of: Satisfied, Partial, Unsatisfied, Unknown")
    rationale: str
    citations: List[Citation]
    evidence: List[EvidenceRef] = []
    risk: Optional[str] = Field(None, description="High, Medium, Low")

class FindingsOutput(BaseModel):
    control_findings: List[ControlFinding]

# =========================================================
# Simple corpus loader & scorer (stdlib only)
# =========================================================

@dataclass
class Passage:
    source: str
    span: str
    score: float

_WORD = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text)]

def _load_corpus(dirpath: str = "corpus", exts: Tuple[str, ...] = ("txt", "md")) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for ext in exts:
        for fp in glob.glob(os.path.join(dirpath, "**", f"*.{ext}"), recursive=True):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    out.append((fp, f.read()))
            except Exception:
                continue
    return out

def _split_passages(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_chars]
        chunks.append(chunk)
        i += max(1, chunk_chars - overlap)
    return chunks

def _score(query_tokens: List[str], passage_text: str) -> float:
    # Lightweight BM25-ish count with log-like saturation
    ptoks = _tokenize(passage_text)
    if not ptoks:
        return 0.0
    hits = sum(ptoks.count(q) for q in query_tokens)
    if hits == 0:
        return 0.0
    return round(1.0 - math.exp(-hits / (len(ptoks) ** 0.5)), 4)

# =========================================================
# Tools
# =========================================================

# Control-specific synonyms to bias retrieval even if wording differs
_SYNONYMS = {
    "AC-2": ["account management", "access review", "user provisioning", "periodic review", "recertification"],
    "IA-2": ["multi-factor authentication", "mfa", "two-factor", "strong authentication", "authenticator"],
    "CM-6": ["configuration settings", "baseline configuration", "secure configuration", "hardening"],
    "AU-6": ["audit review", "audit analysis", "audit reporting", "log review"],
}

@function_tool
def policies_search(query: str, control_id: Optional[str] = None, k: int = 5) -> Dict[str, Any]:
    """
    Search local policy/standard documents for passages relevant to a control.
    Returns: { "passages": [{ "source", "span", "score" }, ...] } sorted by score desc.
    """
    ctrl = (control_id or "").upper().replace(" ", "")
    bias_terms = " ".join(_SYNONYMS.get(ctrl, []))
    qtoks = _tokenize(f"{query} {ctrl} {bias_terms}".strip())
    corpus = _load_corpus("corpus", exts=("txt", "md"))
    cands: List[Passage] = []
    for src, text in corpus:
        for span in _split_passages(text, chunk_chars=900, overlap=150):
            s = _score(qtoks, span)
            if s > 0:
                cands.append(Passage(source=src, span=span[:600], score=s))
    cands.sort(key=lambda p: p.score, reverse=True)
    top = cands[: max(1, k)]
    return {"passages": [p.__dict__ for p in top]}

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

@function_tool
def evidence_lookup(control_id: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Stub connector that would fetch evidence references (URIs) for a control.
    Replace with real integrations (Okta/Jira/AWS Config) later.
    """
    kws = "-".join(keywords or [])
    return {
        "evidence": [
            {"uri": f"s3://evidence/{control_id}/control_policy.json", "hash": "sha256:deadbeef"},
            {"uri": f"s3://evidence/{control_id}/ticket_export_{kws or 'default'}.csv"},
        ]
    }

# ---- PURE HELPER (call this in backend) ----
def evidence_csv_summary_py(control_id: str, dirpath: str = "evidence") -> Dict[str, Any]:
    """
    Scan local evidence/<CONTROL_ID>/*.csv for completed review dates.
    Expected columns include 'completed_at' (YYYY-MM-DD). Returns recency info.
    """
    ctrl = control_id.upper().replace(" ", "")
    folder = os.path.join(dirpath, ctrl)
    latest_dt = None
    files = []
    if os.path.isdir(folder):
        for fp in glob.glob(os.path.join(folder, "*.csv")):
            files.append({"uri": "file://" + fp, "hash": _sha256_file(fp)})
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for row in csv.DictReader(f):
                        ts = (row.get("completed_at") or "").strip()
                        if ts:
                            dt = datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            if (latest_dt is None) or (dt > latest_dt):
                                latest_dt = dt
            except Exception:
                continue
    now = datetime.now(timezone.utc)
    days_since = (now - latest_dt).days if latest_dt else None
    return {
        "latest_completed_at": latest_dt.isoformat() if latest_dt else None,
        "days_since": days_since,
        "files": files
    }

# ---- TOOL WRAPPER (LLM can call this) ----
@function_tool
def evidence_read_csv_summary(control_id: str, dirpath: str = "evidence") -> Dict[str, Any]:
    """LLM-invokable wrapper that delegates to the pure helper."""
    return evidence_csv_summary_py(control_id=control_id, dirpath=dirpath)

# =========================================================
# Agent
# =========================================================

INSTRUCTIONS = """
You are a compliance copilot that maps controls and drafts findings.

Operational rules:
- Always call `policies_search` at least once per request.
- Use `evidence_read_csv_summary` to assess recency of operational reviews where relevant.
- Return ONLY JSON matching FindingsOutput.
- Every claim MUST include ≥1 citation from `policies_search`.
- If passages are insufficient, set status to "Unknown" and specify what is missing.
- Status rubric:
  • Satisfied: Policy + standard support the requirement AND evidence shows latest completion ≤ 95 days ago.
  • Partial: Policy present but evidence is stale (96–180 days) OR incomplete scope.
  • Unsatisfied: Policy conflicts or indicates not done; or explicit failure.
  • Unknown: Insufficient policy/evidence to judge.
Be concise and specific in rationale; avoid policy restatement without analysis.
"""

agent = Agent(
    name="Compliance Copilot",
    instructions=INSTRUCTIONS,
    tools=[policies_search, evidence_lookup, evidence_read_csv_summary],  # tools available to the LLM
    output_type=FindingsOutput,               # Enforce strict JSON via Pydantic
    model=os.getenv("OPENAI_MODEL", "gpt-5"),
)

# =========================================================
# Runner helpers & CLI
# =========================================================

_CTRL_RE = re.compile(r"\b([A-Z]{2,3}-\d{1,2}(?:\s*\(\d+\))?)\b")

def _extract_control_id(text: str) -> Optional[str]:
    m = _CTRL_RE.search(text or "")
    if not m:
        return None
    return m.group(1).upper().replace(" ", "")

def _synthesize_query(user_query: str) -> Tuple[str, Optional[str]]:
    ctrl = _extract_control_id(user_query) or ""
    q = f"{ctrl} {user_query}".strip() if ctrl and ctrl not in user_query else user_query
    return q, (ctrl or None)

def _needs_retry(findings: FindingsOutput) -> bool:
    return any(len(f.citations) == 0 for f in findings.control_findings)

def _merge_evidence(f: ControlFinding, new_files: List[Dict[str, str]]):
    known = {(e.uri, e.hash) for e in f.evidence}
    for fi in new_files:
        tup = (fi.get("uri",""), fi.get("hash"))
        if tup not in known:
            f.evidence.append(EvidenceRef(uri=fi.get("uri",""), hash=fi.get("hash")))

def _apply_recency_rubric(f: ControlFinding, ctrl: str):
    """
    Post-run backend check: even if the model forgot to call the evidence tool,
    read local CSV evidence and adjust status/rationale consistently.
    """
    # Use the pure helper (NOT the tool wrapper)
    rec = evidence_csv_summary_py(control_id=ctrl)
    latest = rec.get("latest_completed_at")
    days = rec.get("days_since")
    files = rec.get("files", [])
    if files:
        _merge_evidence(f, files)

    # Only upgrade if we have at least one citation (policy/standard present)
    if len(f.citations) == 0:
        return

    # Decide status by recency thresholds
    if days is None:
        # no evidence dates found: keep model's status unless it was Satisfied
        if f.status == "Satisfied":
            f.status = "Unknown"
        if "Insufficient evidence" not in f.rationale:
            f.rationale = f"{f.rationale} Missing operational evidence (review completion dates)."
        return

    # ≤95 days → Satisfied; 96–180 → Partial; >180 → Unknown
    if days <= 95:
        f.status = "Satisfied"
        if not f.risk:
            f.risk = "Low"
        f.rationale = f"{f.rationale} Latest review completed {days} days ago ({latest})."
    elif days <= 180:
        f.status = "Partial"
        if not f.risk:
            f.risk = "Medium"
        f.rationale = f"{f.rationale} Evidence is stale: last review {days} days ago ({latest})."
    else:
        f.status = "Unknown"
        f.rationale = f"{f.rationale} No recent review: last completion {days} days ago ({latest}). Consider immediate remediation."

def run(query: str) -> FindingsOutput:
    """
    Expects user input like:
    "Map AC-2 (user account management). Are quarterly access reviews satisfied?"
    """
    # Optional env check
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it in your shell before running.")

    user_q, ctrl = _synthesize_query(query)

    # Attempt 1 (normal)
    result = Runner.run_sync(agent, input=user_q)
    if not result.final_output:
        raise RuntimeError("No final output from agent (attempt 1)")
    findings: FindingsOutput = result.final_output

    # Retry once if any finding is missing citations: force tool-first behavior
    if _needs_retry(findings):
        hint = (
            "Before answering, call the tool `policies_search` with "
            f"query='{user_q}' and control_id='{ctrl or ''}'. "
            "Do not return any finding without at least one citation."
        )
        result2 = Runner.run_sync(agent, input=hint)
        if result2.final_output:
            findings = result2.final_output

    # Backend guardrails: still empty citations → set Unknown + actionable ask
    for f in findings.control_findings:
        if len(f.citations) == 0:
            f.status = "Unknown"
            f.rationale = (
                "Insufficient evidence. Upload or reference a policy/SOP excerpt and supporting records. "
                "Expected citations: standard clause (e.g., NIST) and local policy/procedure."
            )

    # Backend recency rubric (local CSV evidence)
    ctrl_id = ctrl or (findings.control_findings[0].control_id if findings.control_findings else None)
    if ctrl_id:
        for f in findings.control_findings:
            _apply_recency_rubric(f, ctrl_id)

    return findings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compliance Copilot CLI")
    parser.add_argument("query", type=str, help="Compliance question (include a control id like AC-2, IA-2, etc.)")
    args = parser.parse_args()

    # Quick diagnostics for empty corpus issues
    files = _load_corpus()
    if len(files) == 0:
        print("WARNING: No files found in ./corpus (.txt/.md). Add policy texts for citations.\n", flush=True)

    out = run(args.query)
    print(out.model_dump_json(indent=2))
