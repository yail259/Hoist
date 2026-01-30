"""
Reusable Ollama ask handler for Hoist programs.
Zero pip dependencies — uses only urllib.request.
"""

import json
import re
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"

# Channel → model routing.  Override via environment or direct assignment.
CHANNEL_MODELS = {
    "fast":    "gemma3:1b",
    "smart":   "deepseek-r1:7b",
    "default": "qwen3:4b",
}


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output (qwen3, deepseek-r1)."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def ollama_ask(prompt: str, channel: str | None = None) -> str:
    """
    Ask handler compatible with hoist.HoistRuntime / hoist.run().

    Args:
        prompt:  The prompt string from the Hoist program.
        channel: Optional channel name (e.g. "fast", "smart").

    Returns:
        The model's response as a plain string.
    """
    model = CHANNEL_MODELS.get(channel or "default", CHANNEL_MODELS["default"])

    # Prefix with /no_think for qwen3 models to suppress <think> wrapper
    actual_prompt = prompt
    if model.startswith("qwen3"):
        actual_prompt = "/no_think " + prompt

    payload = json.dumps({
        "model": model,
        "prompt": actual_prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return strip_think_tags(body.get("response", ""))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama request failed ({model}): {exc}\n"
            "Is Ollama running?  Start it with: ollama serve"
        ) from exc
