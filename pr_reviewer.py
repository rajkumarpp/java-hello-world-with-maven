# pr_reviewer.py
import os
import json
import math
import requests
from typing import List, Tuple, Dict, Any, Optional

from llm_client import generate_review

GITHUB_API = os.getenv("GITHUB_API_URL", "https://api.github.com")
MAX_DIFF_CHARS = int(os.getenv("DIFF_MAX_CHARS", "120000"))  # per LLM call
MIN_FILE_CHUNK = int(os.getenv("MIN_FILE_CHUNK", "20000"))   # pack small files together

def _gh_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-pr-reviewer-bot"
    }

def get_pr_context_from_env() -> Tuple[str, str, int]:
    """
    Determines owner, repo, and PR number from typical GitHub Actions envs.
    Fallback: environment variable PR_NUMBER.
    """
    repo_full = os.getenv("GITHUB_REPOSITORY")  # e.g. "owner/repo"
    if not repo_full or "/" not in repo_full:
        raise RuntimeError("GITHUB_REPOSITORY is not set or invalid (expected 'owner/repo').")
    owner, repo = repo_full.split("/", 1)

    pr_number_env = os.getenv("PR_NUMBER")
    if pr_number_env:
        return owner, repo, int(pr_number_env)

    event_path = os.getenv("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        try:
            with open(event_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            pr_number = payload.get("pull_request", {}).get("number")
            if pr_number:
                return owner, repo, int(pr_number)
        except Exception:
            pass

    raise RuntimeError("Cannot determine PR number. Set PR_NUMBER env or run from a pull_request event.")

def fetch_pr_details(owner: str, repo: str, pr_number: int, gh_token: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    headers = _gh_headers(gh_token)

    pr_resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}",
        headers=headers, timeout=30
    )
    pr_resp.raise_for_status()
    pr = pr_resp.json()

    # Note: pagination if many files changed
    files: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    while True:
        fr = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files",
            headers=headers,
            params={"page": page, "per_page": per_page},
            timeout=30
        )
        fr.raise_for_status()
        batch = fr.json()
        files.extend(batch)
        if len(batch) < per_page:
            break
        page += 1

    return pr.get("title", ""), pr.get("body", "") or "", files

def build_unified_diffs(files: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Returns list of (filename, unified_diff_text) for files that include a 'patch'.
    Skips binary/large files where GitHub omits 'patch'.
    """
    diffs: List[Tuple[str, str]] = []
    for f in files:
        filename = f.get("filename")
        patch = f.get("patch")
        if not filename or not patch:
            # Skip binary or too-large diffs
            continue
        diff_text = f"--- a/{filename}\n+++ b/{filename}\n{patch}\n"
        diffs.append((filename, diff_text))
    return diffs

def chunk_diffs_by_size(diffs: List[Tuple[str, str]], max_chars: int, min_chunk: int) -> List[List[Tuple[str, str]]]:
    """
    Group per-file diffs into chunks, each not exceeding max_chars.
    Tries to pack small files together for better context.
    """
    # Sort by size ascending to pack small files first
    diffs_sorted = sorted(diffs, key=lambda x: len(x[1]))
    chunks: List[List[Tuple[str, str]]] = []
    current: List[Tuple[str, str]] = []
    current_size = 0

    for item in diffs_sorted:
        size = len(item[1])
        # If single file is huge, put it alone (hard cap)
        if size >= max_chars:
            if current:
                chunks.append(current)
                current = []
                current_size = 0
            chunks.append([item])
            continue

        if current_size + size <= max_chars or (not current and size < max_chars):
            current.append(item)
            current_size += size
        else:
            # finalize current and start new
            chunks.append(current)
            current = [item]
            current_size = size

    if current:
        chunks.append(current)

    # Merge tiny chunks if possible
    merged: List[List[Tuple[str, str]]] = []
    carry: List[Tuple[str, str]] = []
    carry_size = 0
    for ch in chunks:
        ch_size = sum(len(d) for _, d in ch)
        if ch_size < min_chunk:
            carry.extend(ch)
            carry_size += ch_size
            if carry_size >= min_chunk:
                merged.append(carry)
                carry = []
                carry_size = 0
        else:
            if carry:
                merged.append(carry)
                carry = []
                carry_size = 0
            merged.append(ch)
    if carry:
        merged.append(carry)
    return merged

def assemble_diff_text(chunk: List[Tuple[str, str]]) -> str:
    return "\n".join(diff for _, diff in chunk)

def format_review_comment(aggregated: Dict[str, Any]) -> str:
    provider = aggregated.get("_provider", "unknown")
    model = aggregated.get("_model", "unknown")

    comment = "### 🤖 AI PR Review (Automated)\n"
    comment += f"_Provider: **{provider}**, Model: **{model}**_\n\n"

    summary = aggregated.get("summary") or ""
    if summary.strip():
        comment += "**Summary**\n\n"
        comment += f"{summary.strip()}\n\n"

    suggestions: List[str] = aggregated.get("suggestions") or []
    if suggestions:
        comment += "**Suggestions**\n\n"
        for i, s in enumerate(suggestions, 1):
            comment += f"{i}. {s.strip()}\n"
        comment += "\n"

    inline = aggregated.get("inline_comments") or []
    if inline:
        comment += "<details><summary>Inline comments (suggested locations)</summary>\n\n"
        for c in inline:
            file = c.get("file", "")
            line = c.get("line", "")
            text = c.get("comment", "").strip()
            if text:
                comment += f"- `{file}`:{line} — {text}\n"
        comment += "\n</details>\n"

    comment += "\n> _Note_: Inline positions are suggestions only. The bot posts a single review comment to avoid noisy threads."
    return comment

def post_pr_review(owner: str, repo: str, pr_number: int, gh_token: str, body: str, event: str = "COMMENT") -> Dict[str, Any]:
    headers = _gh_headers(gh_token)
    payload = {"body": body, "event": event}
    resp = requests.post(
        f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
        headers=headers, json=payload, timeout=30
    )
    resp.raise_for_status()
    return resp.json()

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple chunk reviews into a single review:
    - Concatenate summaries with headings
    - Merge suggestions (deduplicate similar lines)
    - Merge inline comments
    """
    summaries: List[str] = []
    suggestions: List[str] = []
    inline_comments: List[Dict[str, Any]] = []
    provider, model = None, None

    def _norm(s: str) -> str:
        return " ".join((s or "").strip().split())

    seen_suggestions = set()

    for idx, r in enumerate(results, start=1):
        if not provider and r.get("_provider"):
            provider = r.get("_provider")
        if not model and r.get("_model"):
            model = r.get("_model")

        s = (r.get("summary") or "").strip()
        if s:
            summaries.append(f"**Chunk {idx}**:\n{s}")

        for sug in r.get("suggestions") or []:
            key = _norm(sug)
            if key and key not in seen_suggestions:
                suggestions.append(sug)
                seen_suggestions.add(key)

        for ic in r.get("inline_comments") or []:
            if isinstance(ic, dict) and ic.get("comment"):
                inline_comments.append(ic)

    final_summary = "\n\n".join(summaries) if summaries else "No significant issues detected in analyzed diffs."
    return {
        "summary": final_summary,
        "suggestions": suggestions,
        "inline_comments": inline_comments,
        "_provider": provider or "unknown",
        "_model": model or "unknown",
    }

def run():
    gh_token = os.getenv("GITHUB_TOKEN")
    if not gh_token:
        raise RuntimeError("GITHUB_TOKEN is required.")

    owner, repo, pr_number = get_pr_context_from_env()

    title, body, files = fetch_pr_details(owner, repo, pr_number, gh_token)
    per_file_diffs = build_unified_diffs(files)

    if not per_file_diffs:
        comment = "### 🤖 AI PR Review (Automated)\nNo textual diffs available (binary or very large files)."
        post_pr_review(owner, repo, pr_number, gh_token, comment, event="COMMENT")
        return

    # Chunk diffs and review each chunk
    chunks = chunk_diffs_by_size(per_file_diffs, MAX_DIFF_CHARS, MIN_FILE_CHUNK)

    # Optional hints to improve review quality (set via env):
    language_hint = os.getenv("LANGUAGE_HINT")  # e.g., "Python (FastAPI), TypeScript (React)"
    guidelines = os.getenv("REVIEW_GUIDELINES")  # short team rules or expectations

    all_results: List[Dict[str, Any]] = []
    for ch_index, ch in enumerate(chunks, start=1):
        diff_text = assemble_diff_text(ch)
        # Truncate just in case (safety)
        if len(diff_text) > MAX_DIFF_CHARS:
            diff_text = diff_text[:MAX_DIFF_CHARS] + "\n...TRUNCATED BY BOT..."

        try:
            result = generate_review(title, body, diff_text, language_hint, guidelines)
        except Exception as e:
            result = {
                "summary": f"Chunk {ch_index}: Failed to generate review due to error: {e}",
                "suggestions": [],
                "inline_comments": [],
                "_provider": os.getenv("LLM_PROVIDER", "groq"),
                "_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            }
        all_results.append(result)

    aggregated = aggregate_results(all_results)
    comment = format_review_comment(aggregated)
    post_pr_review(owner, repo, pr_number, gh_token, comment, event="COMMENT")

if __name__ == "__main__":
    run()
