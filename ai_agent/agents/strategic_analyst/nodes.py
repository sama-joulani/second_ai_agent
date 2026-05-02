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

from .tools import (
    generate_content_with_retry,
    json_with_gemini,
    model_name,
    search_with_gemini,
)


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
    Node 2: find a canonical competitor list, then analyze that SAME list.

    Why:
    - If we ask "top competitors" and "strengths/weaknesses" as separate open-ended prompts,
      the model can introduce new apps in the second answer.
    - We want strengths/weaknesses to be explicitly about the competitors we found.
    """
    if state.get("error"):
        return {}

    product_category = (state.get("product_category") or "").strip()
    target_audience = (state.get("target_audience") or "").strip()

    # 1) Canonical competitor list as structured JSON.
    # We do this in TWO steps:
    # - grounded web search (text) to gather real names
    # - non-grounded JSON extraction to force a consistent list shape
    top_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "competitors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                },
                "minItems": 3,
                "maxItems": 12,
            }
        },
        "required": ["competitors"],
    }

    try:
        grounded_top_text = await search_with_gemini(
            f"Find the top competitors in the {product_category} market for {target_audience}. "
            f"List 5-10 competitor app/product names with a 1-2 sentence description each."
        )
        top_data = await json_with_gemini(
            prompt=(
                "Extract a competitor list from the grounded research text below.\n"
                "Return ONLY JSON matching the provided schema.\n"
                "- Keep the competitor names exactly as written in the text when possible.\n"
                "- If the text contains fewer than 5 clear competitors, return as many as you can.\n\n"
                f"Grounded research text:\n{grounded_top_text}"
            ),
            json_schema=top_schema,
        )
        top_list = top_data.get("competitors") or []
        if not isinstance(top_list, list) or not top_list:
            raise ValueError("Top competitors JSON was empty.")
    except Exception as exc:
        return {
            "competitors": [
                {"angle": "top competitors", "findings": ""},
                {"angle": "strengths and weaknesses", "findings": ""},
                {"angle": "user sentiment", "findings": ""},
            ],
            "current_agent": "builder",
            "error": f"top competitors: {exc!s}",
        }

    # Extract names for consistent follow-up prompts.
    names: list[str] = []
    for item in top_list:
        if not isinstance(item, dict):
            continue
        n = str(item.get("name") or "").strip()
        if n:
            names.append(n)

    # Keep it readable in prompts and prevent huge context.
    names = names[:12]
    names_text = ", ".join(names) if names else "(unknown)"

    # 2) Analyze strengths/weaknesses and sentiment FOR THAT SAME LIST (run concurrently).
    prompts = {
        "strengths and weaknesses": (
            f"For the following competitors in the {product_category} space for {target_audience}: {names_text}\n\n"
            f"For EACH competitor, summarize key strengths and weaknesses.\n"
            f"Rules:\n"
            f"- Only discuss the competitors in the list above.\n"
            f"- If you are unsure about a competitor, say so instead of inventing details.\n"
        ),
        "user sentiment": (
            f"For the following competitors in the {product_category} space for {target_audience}: {names_text}\n\n"
            f"Summarize what users complain about and what they love.\n"
            f"Rules:\n"
            f"- Focus on themes that appear across these competitors.\n"
            f"- Do not introduce new competitor names outside the list.\n"
        ),
    }

    results_raw = await asyncio.gather(
        search_with_gemini(prompts["strengths and weaknesses"]),
        search_with_gemini(prompts["user sentiment"]),
        return_exceptions=True,
    )

    strengths_res, sentiment_res = results_raw

    error_parts: list[str] = []
    strengths_text = ""
    sentiment_text = ""

    if isinstance(strengths_res, Exception):
        error_parts.append(f"strengths and weaknesses: {strengths_res!s}")
    else:
        strengths_text = str(strengths_res).strip()
        if not strengths_text:
            error_parts.append("strengths and weaknesses: empty response from Gemini")

    if isinstance(sentiment_res, Exception):
        error_parts.append(f"user sentiment: {sentiment_res!s}")
    else:
        sentiment_text = str(sentiment_res).strip()
        if not sentiment_text:
            error_parts.append("user sentiment: empty response from Gemini")

    competitors_out = [
        {"angle": "top competitors", "findings": json.dumps(top_data, ensure_ascii=False, indent=2)},
        {"angle": "strengths and weaknesses", "findings": strengths_text},
        {"angle": "user sentiment", "findings": sentiment_text},
    ]

    out: dict[str, Any] = {
        "competitors": competitors_out,
        "current_agent": "builder",
    }
    if error_parts:
        out["error"] = "; ".join(error_parts)
    return out
