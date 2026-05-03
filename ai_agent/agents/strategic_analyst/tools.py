"""
Gemini helpers for Agent 1. Uses the official `google-genai` SDK (not deprecated `google.generativeai`).
"""

import asyncio
import os
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# Default model: `gemini-2.0-flash` hit free-tier limit 0 for some accounts; 2.5 Flash is current in docs.
# Override with GEMINI_MODEL in .env if your project prefers another id (e.g. gemini-1.5-flash).
def model_name() -> str:
    """Resolved model id (GEMINI_MODEL env, else gemini-2.5-flash)."""
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _require_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in the environment.")
    return key


async def generate_content_with_retry(aio_models, **kwargs):
    """
    Retry transient API failures with exponential backoff.

    Why:
    - 429: rate limit / quota pacing.
    - 503: temporary high demand (your current error).
    - 500/502/504: transient server / gateway issues.

    Backoff schedule (seconds): 2, 5, 12, 25
    """
    backoff_seconds = (2, 5, 12, 25)
    attempt = 0
    while True:
        try:
            return await aio_models.generate_content(**kwargs)
        except genai_errors.APIError as exc:
            retryable_codes = {429, 500, 502, 503, 504}
            if exc.code in retryable_codes and attempt < len(backoff_seconds):
                await asyncio.sleep(backoff_seconds[attempt])
                attempt += 1
                continue
            raise


async def search_with_gemini(prompt: str) -> str:
    """
    Call Gemini with Google Search grounding so the model can retrieve fresh web context.

    Uses GEMINI_API_KEY. Raises ValueError if the API returns no usable text.
    """
    api_key = _require_api_key()
    client = genai.Client(api_key=api_key)
    model = model_name()

    # Grounding: Google Search tool (replaces deprecated string tools= on the old SDK).
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )

    response = await generate_content_with_retry(
        client.aio.models,
        model=model,
        contents=prompt,
        config=config,
    )

    text = (response.text or "").strip()
    if not text:
        raise ValueError(f"Empty response from Gemini for prompt: {prompt!r}")

    return text


async def json_with_gemini(prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Non-grounded Gemini call that returns structured JSON.

    We use this when downstream steps must be consistent, e.g.:
    - get a canonical competitor list
    - then ask for strengths/weaknesses ONLY for that same list
    """
    api_key = _require_api_key()
    client = genai.Client(api_key=api_key)
    model = model_name()

    # Note: Gemini currently rejects "tools + response_mime_type=application/json" for some tool types.
    # So this helper does NOT use google_search grounding. If you need grounding, do it in a
    # separate call (e.g. search_with_gemini) and then parse/extract JSON with this function.
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=json_schema,
    )

    response = await generate_content_with_retry(
        client.aio.models,
        model=model,
        contents=prompt,
        config=config,
    )

    raw = (response.text or "").strip()
    if not raw:
        raise ValueError(f"Empty JSON response from Gemini for prompt: {prompt!r}")

    # The SDK guarantees JSON shape via schema; still parse defensively.
    import json as _json  # local import to keep module surface small

    data = _json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Gemini JSON response was not an object.")
    return data
