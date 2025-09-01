# file: llm_client.py
import os
import json
import requests
from typing import Dict

SYSTEM_PROMPT = (
    "You are a senior software engineer reviewing a pull request. "
    "Review for code quality, security issues, test coverage, and best practices. "
    "Return JSON with fields: summary, suggestions (list), and optionally inline_comments "
    "(list of {file, line, comment}). Be concise, specific, and actionable."
)

def _prompt_for_diff(pr_title: str, pr_body: str, diff_text: str) -> str:
    return f"""PR Title: {pr_title}

PR Description:
{pr_body}

Diff (unified format):
{diff_text}

Return strictly valid JSON with keys: summary, suggestions, inline_comments.
"""

def _post_openai_compatible(url: str, api_key: str, model: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Supports OpenAI-compatible chat APIs
    return data["choices"][0]["message"]["content"]

def _post_huggingface_inference(model_id: str, api_key: str, prompt: str) -> str:
    # Basic text-generation style for some HF hosted models.
    # Many chat models also accept prompts in chat-template form.
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": f"{SYSTEM_PROMPT}\n\n{prompt}\n\nReturn strictly valid JSON.",
        "parameters": {"max_new_tokens": 1200, "temperature": 0.2},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # HF returns a list[ { "generated_text": ... } ] or a dict for some models
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # Some chat models return text under different fields; adapt as needed:
    return json.dumps(data)

def generate_review(pr_title: str, pr_body: str, diff_text: str) -> Dict:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    prompt = _prompt_for_diff(pr_title, pr_body, diff_text)

    if provider == "groq":
        api_key = os.environ["GROQ_API_KEY"]
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        url = "https://api.groq.com/openai/v1/chat/completions"
        content = _post_openai_compatible(url, api_key, model, prompt)

    elif provider == "openrouter":
        api_key = os.environ["OPENROUTER_API_KEY"]
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        url = "https://openrouter.ai/api/v1/chat/completions"
        content = _post_openai_compatible(url, api_key, model, prompt)

    elif provider == "huggingface":
        api_key = os.environ["HF_API_KEY"]
        model_id = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        content = _post_huggingface_inference(model_id, api_key, prompt)

    elif provider == "local":
        # Example: Ollama’s OpenAI-compatible endpoint enabled via `OLLAMA_OPENAI_COMPAT=1`
        api_key = os.getenv("LOCAL_API_KEY", "not-needed")
        model = os.getenv("LOCAL_MODEL", "llama3.1")
        url = os.getenv("LOCAL_OPENAI_URL", "http://localhost:11434/v1/chat/completions")
        content = _post_openai_compatible(url, api_key, model, prompt)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    # Try to parse JSON; if the model returns extra text, attempt to extract JSON block.
    try:
        # Direct JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: extract nearest {...} block
        import re
        m = re.search(r"\{(?:[^{}]|(?R))*\}", content, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # Ultimate fallback
        return {"summary": content[:8000], "suggestions": [], "inline_comments": []}
