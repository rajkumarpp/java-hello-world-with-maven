# llm_client.py
import os
import re
import json
import time
import requests
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_SYSTEM_PROMPT = (
    "You are a seasoned senior software engineer and code reviewer. "
    "Review the pull request changes for code quality, correctness, security, performance, "
    "maintainability, readability, and test coverage. "
    "Be specific and actionable. Prefer concise, structured feedback.\n\n"
    "Return STRICT JSON with keys:\n"
    "  - summary: string\n"
    "  - suggestions: array of strings (actionable, prioritized)\n"
    "  - inline_comments: array of objects with keys {file: string, line: number, comment: string}\n"
)

def build_prompt(
    pr_title: str,
    pr_body: str,
    diff_text: str,
    language_hint: Optional[str] = None,
    guidelines: Optional[str] = None,
) -> str:
    extras = []
    if language_hint:
        extras.append(f"Primary language/framework context: {language_hint}")
    if guidelines:
        extras.append(f"Team guidelines:\n{guidelines}")
    extras_block = "\n\n".join(extras).strip()
    if extras_block:
        extras_block = "\n\n" + extras_block

    return (
        f"PR Title: {pr_title}\n\n"
        f"PR Description:\n{(pr_body or '').strip()}\n"
        f"{extras_block}\n\n"
        "Unified Diff:\n"
        f"{diff_text}\n\n"
        "Return STRICT JSON only, no prose outside JSON."
    )

def _openai_compatible_chat(
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    retries: int = 3,
    timeout: int = 60,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                # Rate limited; exponential backoff
                wait = min(2 ** attempt, 10)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 5))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

def _huggingface_inference(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_new_tokens: int = 1200,
    retries: int = 3,
    timeout: int = 60,
) -> str:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    inputs = f"{system_prompt}\n\n{user_prompt}\n\nReturn STRICT JSON only."
    payload = {
        "inputs": inputs,
        "parameters": {"temperature": temperature, "max_new_tokens": max_new_tokens},
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp
