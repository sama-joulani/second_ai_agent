"""
Builder Agent nodes: turn raw competitor findings into a structured Markdown comparison report.

If QA rejected the previous draft, qa_feedback_for_builder drives a full rewrite.
"""

import json
import os
from typing import Any

from google import genai

from state.schema import AgentState

from agents.strategic_analyst.tools import generate_content_with_retry, model_name


def _competitors_blob(state: AgentState) -> str:
    """Serialize competitors for the prompt (safe, readable JSON)."""
    raw = state.get("competitors")
    if not raw:
        return "[]"
    try:
        return json.dumps(raw, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(raw)


async def build_comparison_report(state: AgentState) -> dict:
    """
    Read state['competitors'] and write state['comparison_report'] as Markdown.

    Hands off to QA via current_agent='qa'. If this is a rewrite, use qa_feedback_for_builder.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        msg = "GEMINI_API_KEY is not set in the environment."
        return {
            "error": f"{state.get('error')}; {msg}" if state.get("error") else msg,
            "current_agent": "qa",
        }

    user_input = state.get("user_input") or ""
    product_category = state.get("product_category") or ""
    target_audience = state.get("target_audience") or ""
    competitors_json = _competitors_blob(state)
    feedback = (state.get("qa_feedback_for_builder") or "").strip()
    prior = (state.get("comparison_report") or "").strip()
    revision_count = int(state.get("qa_revision_count") or 0)

    revision_block = ""
    if feedback:
        # QA asked for a full rewrite — cite the prior draft so the model can diff mentally.
        revision_block = f"""
This is **revision round {revision_count}**. QA rejected the previous draft and requires a full rewrite.

QA instructions (must follow):
{feedback}

Previous draft to improve (do not copy mistakes; fix all issues):
---
{prior if prior else "(no prior text)"}
---
"""

    prompt = f"""You are a senior product strategist. Using the research angles below, write a clear,
well-structured Markdown comparison report for the product idea.

Product idea: {user_input}
Product category: {product_category}
Target audience: {target_audience}

Research findings (JSON array of objects with angle and findings):
{competitors_json}

{revision_block}

Requirements:
- Use Markdown with headings (##, ###) and bullet lists where helpful.
- Include these sections in order:
  1) Overview
  2) Competitive landscape — separate **direct** vs **indirect** competitors using the
     `top competitors` JSON (relationship fields). Name each player and why they matter.
  3) Per-competitor strengths and weaknesses — use `per_competitor_sw` and the narrative
     `strengths and weaknesses` angle together; keep competitor names aligned with the landscape JSON.
  4) Comparison vs this product idea — synthesize `structured_comparison` (executive_summary and
     comparison_rows). Rows may use either a single `analysis` field per competitor (chunked extract)
     or separate how_they_compare / strengths_relative_to_idea / weaknesses_relative_to_idea /
     differentiation_angle fields; treat `analysis` as the full narrative when present.
  5) User sentiment themes (from the `user sentiment` angle)
  6) Strategic takeaways for this product idea
- If some findings are empty or clearly incomplete, state that limitation briefly.
- Ground the report in the supplied findings; do not claim specific metrics or facts that are not implied by them.

Output ONLY the Markdown document (no JSON wrapper, no preamble)."""

    client = genai.Client(api_key=api_key)
    model = model_name()

    try:
        response = await generate_content_with_retry(
            client.aio.models,
            model=model,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            err = "Builder Agent: Gemini returned an empty comparison report."
            return {
                "comparison_report": None,
                "current_agent": "qa",
                "error": (state.get("error") + "; " if state.get("error") else "") + err,
            }

        out: dict[str, Any] = {
            "comparison_report": text,
            "current_agent": "qa",
        }
        if state.get("error"):
            out["error"] = state["error"]
        return out
    except Exception as exc:  # noqa: BLE001 — keep the graph alive; record SDK failures
        err = f"build_comparison_report failed: {exc}"
        merged = (state.get("error") + "; " if state.get("error") else "") + err
        return {
            "comparison_report": None,
            "current_agent": "qa",
            "error": merged,
        }
