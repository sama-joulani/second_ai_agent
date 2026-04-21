"""
QA Agent nodes: review the Builder report, then either approve (END) or send feedback back to Builder.
"""

import json
import os
from typing import Any

from google import genai
from google.genai import types

from state.schema import AgentState

from agents.qa.revision_config import max_qa_revision_loops
from agents.strategic_analyst.tools import generate_content_with_retry, model_name


def _short_json(obj: Any, limit: int = 12000) -> str:
    """Compact JSON for prompts; truncate if huge to stay within context limits."""
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "\n... [truncated]"
    return s


def _qa_decision_schema() -> dict[str, Any]:
    """Structured output so we can route the graph without fragile string parsing."""
    return {
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "True if the comparison report is good enough to ship.",
            },
            "feedback_for_builder": {
                "type": "string",
                "description": "If approved is false, specific edits the Builder must apply. Empty if approved.",
            },
            "qa_markdown": {
                "type": "string",
                "description": "Markdown for the user: QA verdict, issues, and if approved the final polished deliverable.",
            },
        },
        "required": ["approved", "feedback_for_builder", "qa_markdown"],
    }


async def run_quality_review(state: AgentState) -> dict:
    """
    Review comparison_report. Either approve (qa_route=end) or request Builder rewrite (qa_route=builder).

    Loops at most MAX_QA_REVISION_LOOPS times (see revision_config).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        msg = "GEMINI_API_KEY is not set in the environment."
        return {
            "error": f"{state.get('error')}; {msg}" if state.get("error") else msg,
            "current_agent": "complete",
            "qa_route": "end",
        }

    user_input = state.get("user_input") or ""
    product_category = state.get("product_category") or ""
    target_audience = state.get("target_audience") or ""
    competitors_blob = _short_json(state.get("competitors"))
    report = (state.get("comparison_report") or "").strip()
    revision_count = int(state.get("qa_revision_count") or 0)
    max_loops = max_qa_revision_loops()

    # No report to review — cannot loop; finish with a diagnostic.
    if not report:
        note = (
            "QA could not review an empty comparison_report. "
            "Fix upstream research or Builder errors and re-run."
        )
        merged_err = f"{state.get('error')}; {note}" if state.get("error") else note
        return {
            "qa_report": f"## QA status\n\n{note}\n",
            "current_agent": "complete",
            "error": merged_err,
            "qa_route": "end",
            "qa_feedback_for_builder": None,
        }

    prompt = f"""You are a strict QA editor for competitive intelligence deliverables.

Decide if the Markdown comparison report is accurate vs the research JSON, clear, and complete enough.

Context:
- Product idea: {user_input}
- Category: {product_category}
- Audience: {target_audience}
- QA revision rounds completed so far (how many times QA already sent work back to Builder): {revision_count}
- Maximum such rounds allowed (hard cap): {max_loops}

Original research (JSON, angles + findings):
{competitors_blob}

Comparison report to review:
---
{report}
---

Rules:
- Set approved=true only if the report is strong enough to ship with at most minor issues.
- If approved=false, feedback_for_builder MUST list concrete, actionable edits (sections to add/fix, tone, missing caveats).
- qa_markdown should always include a short verdict and, if not approved, what you expect after revision.

Return structured JSON only (schema enforced by the API)."""

    client = genai.Client(api_key=api_key)
    model = model_name()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=_qa_decision_schema(),
    )

    try:
        response = await generate_content_with_retry(
            client.aio.models,
            model=model,
            contents=prompt,
            config=config,
        )
        raw = (response.text or "").strip()
        if not raw:
            err = "QA Agent: empty structured response."
            return {
                "qa_report": None,
                "current_agent": "complete",
                "error": (state.get("error") + "; " if state.get("error") else "") + err,
                "qa_route": "end",
            }

        data = json.loads(raw)
        approved = bool(data.get("approved"))
        feedback = str(data.get("feedback_for_builder") or "").strip()
        qa_markdown = str(data.get("qa_markdown") or "").strip()

        if not qa_markdown:
            qa_markdown = "## QA verdict\n\n(No markdown returned.)\n"

        out: dict[str, Any] = {
            "qa_report": qa_markdown,
            "current_agent": "qa",
        }
        if state.get("error"):
            out["error"] = state["error"]

        # Approved — done.
        if approved:
            out["qa_feedback_for_builder"] = None
            out["qa_route"] = "end"
            out["current_agent"] = "complete"
            return out

        # Not approved — maybe loop back to Builder.
        if revision_count < max_loops:
            out["qa_feedback_for_builder"] = feedback or (
                "Improve clarity, tighten claims to the research JSON, and fix any unsupported statements."
            )
            out["qa_revision_count"] = revision_count + 1
            out["qa_route"] = "builder"
            out["current_agent"] = "builder"
            return out

        # Max revisions exhausted — ship best-effort with explicit note.
        forced = (
            f"{qa_markdown}\n\n---\n\n"
            f"**Note:** Maximum revision loops ({max_loops}) reached. "
            f"This is the best-effort QA output without another Builder pass.\n"
        )
        out["qa_report"] = forced
        out["qa_feedback_for_builder"] = None
        out["qa_route"] = "end"
        out["current_agent"] = "complete"
        return out

    except Exception as exc:  # noqa: BLE001
        err = f"run_quality_review failed: {exc}"
        merged = (state.get("error") + "; " if state.get("error") else "") + err
        return {
            "qa_report": None,
            "current_agent": "complete",
            "error": merged,
            "qa_route": "end",
        }
