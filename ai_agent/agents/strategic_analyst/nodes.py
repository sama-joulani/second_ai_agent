"""
LangGraph node functions for the Strategic Analyst. Each node is async and returns a partial state.
"""

import asyncio
import json
import os
import re
from typing import Any

from google import genai
from google.genai import types

from state.schema import AgentState

from .tools import generate_content_with_retry, model_name, search_with_gemini


def _parse_json_object(raw: str) -> dict[str, Any]:
    """
    Parse a JSON object from the model output.

    If the model wraps JSON in markdown fences, strip them before json.loads.
    """
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


async def extract_product_info(state: AgentState) -> dict:
    """
    Node 1: derive product_category and target_audience from the raw idea using Gemini (JSON).
    """
    if state.get("error"):
        return {}

    user_input = state["user_input"]
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "error": "GEMINI_API_KEY is not set in the environment.",
            "current_agent": "strategic_analyst",
        }

    prompt = f"""You are a business analyst. Extract the product category and target audience
from this product idea.
Return ONLY valid JSON with exactly two keys: product_category, target_audience.
Keep both values short (under 8 words each).

Product idea: {user_input}"""

    # Structured JSON via response_json_schema (google-genai).
    extraction_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "product_category": {"type": "string"},
            "target_audience": {"type": "string"},
        },
        "required": ["product_category", "target_audience"],
    }

    client = genai.Client(api_key=api_key)
    model = model_name()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=extraction_schema,
    )

    try:
        response = await generate_content_with_retry(
            client.aio.models,
            model=model,
            contents=prompt,
            config=config,
        )
        raw_text = (response.text or "").strip()
        if not raw_text:
            return {
                "error": "Gemini returned an empty response while extracting product info.",
                "current_agent": "strategic_analyst",
            }

        data = _parse_json_object(raw_text)
        product_category = str(data.get("product_category", "")).strip()
        target_audience = str(data.get("target_audience", "")).strip()

        if not product_category or not target_audience:
            return {
                "error": "Gemini JSON was missing product_category or target_audience.",
                "current_agent": "strategic_analyst",
            }

        return {
            "product_category": product_category,
            "target_audience": target_audience,
            "current_agent": "strategic_analyst",
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return {
            "error": f"Failed to parse Gemini extraction JSON: {exc}",
            "current_agent": "strategic_analyst",
        }
    except Exception as exc:  # noqa: BLE001 — surface SDK/network failures without crashing the process
        return {
            "error": f"extract_product_info failed: {exc}",
            "current_agent": "strategic_analyst",
        }


async def search_competitors(state: AgentState) -> dict:
    """
    Node 2: three grounded Gemini searches run concurrently; results stored under competitors.
    """
    if state.get("error"):
        return {}

    product_category = (state.get("product_category") or "").strip()
    target_audience = (state.get("target_audience") or "").strip()

    # Prompts follow requirements.md; angles must match the expected labels exactly.
    angles = ["top competitors", "strengths and weaknesses", "user sentiment"]
    prompts = [
        (
            f"Who are the top competitors in the {product_category} market for {target_audience}? "
            f"List the main ones with a brief description of each."
        ),
        (f"What are the strengths and weaknesses of the leading {product_category} apps in 2025?"),
        (
            f"What do users complain about most in {product_category} apps? What do they love?"
        ),
    ]

    # Run all three searches together; exceptions or empty text become state["error"], not a crash.
    results_raw = await asyncio.gather(
        *[search_with_gemini(p) for p in prompts],
        return_exceptions=True,
    )

    competitors: list[dict[str, str]] = []
    error_parts: list[str] = []

    for angle, res in zip(angles, results_raw):
        # Normal failures from Gemini or the SDK surface as Exception subclasses.
        if isinstance(res, Exception):
            error_parts.append(f"{angle}: {res!s}")
            competitors.append({"angle": angle, "findings": ""})
            continue

        text = str(res).strip()
        if not text:
            error_parts.append(f"{angle}: empty response from Gemini")
            competitors.append({"angle": angle, "findings": ""})
        else:
            competitors.append({"angle": angle, "findings": text})

    if error_parts:
        return {
            "competitors": competitors,
            "current_agent": "builder",
            "error": "; ".join(error_parts),
        }

    return {
        "competitors": competitors,
        "current_agent": "builder",
    }
