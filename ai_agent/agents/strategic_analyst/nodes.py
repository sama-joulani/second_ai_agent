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


def _coerce_relationship_direct_indirect(
    item: dict[str, Any],
) -> str:
    """
    Normalize direct vs indirect for one competitor row.

    Why:
    - Some Gemini JSON payloads use alternate keys or phrasing ("Direct", "indirect competitor").
    - The UI and downstream prompts expect exactly 'direct' or 'indirect'.
    """
    # First non-empty field wins (avoid joining unrelated keys like Type="app" with relationship).
    raw = ""
    for key in (
        "relationship",
        "Relationship",
        "competitor_type",
        "competitorType",
        "direct_or_indirect",
        "directOrIndirect",
        "type",
        "Type",
    ):
        if key not in item or item[key] is None:
            continue
        t = str(item[key]).strip()
        if t:
            raw = t
            break
    s = raw.lower().replace("_", " ").strip()

    if s in ("direct", "indirect"):
        return s
    if s.startswith("direct") or "direct competitor" in s or s == "d":
        return "direct"
    if s.startswith("indirect") or "indirect competitor" in s or s == "i":
        return "indirect"
    if "substitute" in s or "adjacent" in s:
        return "indirect"

    # Last resort: scan free-text fields from the same row (extraction often repeats the label there).
    blob = " ".join(
        str(item.get(k) or "")
        for k in ("relationship_rationale", "relationshipRationale", "description", "name")
    ).lower()
    if re.search(r"\bindirect\b", blob) or re.search(r"\bsubstitute\b", blob) or "adjacent" in blob:
        return "indirect"
    if re.search(r"\bdirect\b", blob) or "same category" in blob or "head-to-head" in blob:
        return "direct"

    return ""


def _normalize_competitor_landscape(top_data: dict[str, Any]) -> None:
    """
    Mutate top_data['competitors'] in place: canonical 'relationship' + 'relationship_rationale' keys.

    Keeps Builder/UI stable even when the model returns camelCase or slightly different wording.
    """
    lst = top_data.get("competitors")
    if not isinstance(lst, list):
        return
    for row in lst:
        if not isinstance(row, dict):
            continue
        rel = _coerce_relationship_direct_indirect(row)
        if rel:
            row["relationship"] = rel
        rationale = (
            row.get("relationship_rationale")
            or row.get("relationshipRationale")
            or row.get("rationale")
            or ""
        )
        row["relationship_rationale"] = str(rationale).strip()


# Gemini JSON schema limits: one array of rich objects cannot use a very large maxItems (400 error).
# We take competitors in chunks (same row shape) and merge in Python for a higher total cap.
_COMPETITOR_CHUNK_SIZE = 18
_COMPETITOR_MAX_TOTAL = 36

# Structured comparison: one big array of fat objects + high maxItems triggers Gemini 400
# ("too many states"). Use several tiny schemas (executive summary + slim row chunks).
_COMPARISON_ROWS_PER_SCHEMA = 8

_EXEC_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"executive_summary": {"type": "string"}},
    "required": ["executive_summary"],
}


def _comparison_rows_chunk_schema() -> dict[str, Any]:
    """At most _COMPARISON_ROWS_PER_SCHEMA rows; three short strings each (no long property names)."""
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "maxItems": _COMPARISON_ROWS_PER_SCHEMA,
                "items": {
                    "type": "object",
                    "properties": {
                        "competitor_name": {"type": "string"},
                        "relationship": {"type": "string"},
                        "analysis": {"type": "string"},
                    },
                    "required": ["competitor_name", "relationship", "analysis"],
                },
            }
        },
        "required": ["rows"],
    }


def _competitor_item_json_schema() -> dict[str, Any]:
    """Single competitor row shape for response_json_schema (reused across extract calls)."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "relationship": {"type": "string"},
            "relationship_rationale": {"type": "string"},
        },
        "required": ["name", "description", "relationship", "relationship_rationale"],
    }


async def _merge_more_competitors_from_text(
    *,
    top_data: dict[str, Any],
    grounded_top_text: str,
    user_idea: str,
) -> None:
    """
    Pull a second batch of competitors from the same grounded text (deduped by name).

    Runs when the merged list is still under _COMPETITOR_MAX_TOTAL so we capture names the first
    JSON call missed without raising Gemini schema maxItems above the safe per-request chunk size.
    """
    cur = top_data.get("competitors")
    if not isinstance(cur, list) or not cur:
        return
    # Second pass whenever we still have room — the first extract often stops below the chunk cap.
    if len(cur) >= _COMPETITOR_MAX_TOTAL:
        return

    seen = {str(x.get("name") or "").strip().lower() for x in cur if isinstance(x, dict)}
    more_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "more_competitors": {
                "type": "array",
                "items": _competitor_item_json_schema(),
                "maxItems": _COMPETITOR_CHUNK_SIZE,
            }
        },
        "required": ["more_competitors"],
    }
    exclude_preview = ", ".join(sorted(seen))[:6000]
    truncated = grounded_top_text[:75000]
    prompt = (
        "Extract ONLY additional competitors from the research text below.\n"
        "Do NOT repeat any name from the EXCLUDE list (case-insensitive match counts as duplicate).\n"
        "Return ONLY JSON matching the schema.\n"
        "relationship should be the word direct or indirect.\n\n"
        f"EXCLUDE (already captured): {exclude_preview}\n\n"
        f"Product idea: {user_idea}\n\n"
        f"Research text:\n{truncated}"
    )
    extra = await json_with_gemini(prompt=prompt, json_schema=more_schema)
    more_list = extra.get("more_competitors") or []
    if not isinstance(more_list, list):
        return
    for row in more_list:
        if not isinstance(row, dict):
            continue
        nm = str(row.get("name") or "").strip().lower()
        if not nm or nm in seen:
            continue
        seen.add(nm)
        cur.append(row)
        if len(cur) >= _COMPETITOR_MAX_TOTAL:
            break
    top_data["competitors"] = cur


async def _run_structured_comparison_batched(
    *,
    user_idea: str,
    names: list[str],
    names_text: str,
    top_data: dict[str, Any],
    strengths_for_compare: str,
    error_parts: list[str],
) -> dict[str, Any]:
    """
    Build structured_comparison without one oversized response_json_schema.

    Gemini rejects large nested schemas; we split into a tiny executive object plus
    repeated small chunks (slim rows: name, relationship, analysis).
    """
    out: dict[str, Any] = {"executive_summary": "", "comparison_rows": []}
    try:
        sdata = await json_with_gemini(
            prompt=(
                "Write one concise executive summary in plain prose comparing the product idea "
                "to the competitive set named below.\n"
                f"Product idea:\n{user_idea}\n\n"
                f"Competitors:\n{names_text}\n"
            ),
            json_schema=_EXEC_SUMMARY_SCHEMA,
        )
        out["executive_summary"] = str(sdata.get("executive_summary") or "").strip()
    except Exception as exc:  # noqa: BLE001
        error_parts.append(f"comparison executive_summary: {exc!s}")

    chunk_schema = _comparison_rows_chunk_schema()
    step = _COMPARISON_ROWS_PER_SCHEMA
    merged_rows: list[dict[str, Any]] = []

    for i in range(0, len(names), step):
        chunk_names = names[i : i + step]
        name_lo = {x.strip().lower() for x in chunk_names}
        sub_comps = [
            x
            for x in (top_data.get("competitors") or [])
            if isinstance(x, dict) and str(x.get("name") or "").strip().lower() in name_lo
        ]
        sub_json = json.dumps({"competitors": sub_comps}, ensure_ascii=False, indent=2)
        try:
            pdata = await json_with_gemini(
                prompt=(
                    "Compare the product idea to EACH competitor in the subset JSON.\n"
                    "Return ONLY JSON matching the schema.\n"
                    "- competitor_name must match the subset exactly.\n"
                    "- relationship must match each row's relationship in the subset.\n"
                    "- analysis: compact prose (you may use short bullets) covering vs the idea, "
                    "their edge, their gap, and differentiation for the idea.\n\n"
                    f"Product idea:\n{user_idea}\n\n"
                    f"Competitor subset:\n{sub_json}\n\n"
                    f"Strengths/weaknesses research (may be truncated):\n{strengths_for_compare}"
                ),
                json_schema=chunk_schema,
            )
        except Exception as exc:  # noqa: BLE001
            error_parts.append(f"structured_comparison rows {i}-{i + step}: {exc!s}")
            continue
        rows = pdata.get("rows") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            merged_rows.append(
                {
                    "competitor_name": str(row.get("competitor_name") or "").strip(),
                    "relationship": str(row.get("relationship") or "").strip(),
                    "analysis": str(row.get("analysis") or "").strip(),
                }
            )

    out["comparison_rows"] = merged_rows
    _align_comparison_rows_with_landscape(out, top_data)
    return out


_PER_SW_ROWS_PER_SCHEMA = 8


def _per_sw_chunk_schema() -> dict[str, Any]:
    """Small per_competitor array so Gemini does not reject the schema."""
    return {
        "type": "object",
        "properties": {
            "per_competitor": {
                "type": "array",
                "maxItems": _PER_SW_ROWS_PER_SCHEMA,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "strengths": {"type": "string"},
                        "weaknesses": {"type": "string"},
                    },
                    "required": ["name", "strengths", "weaknesses"],
                },
            }
        },
        "required": ["per_competitor"],
    }


async def _run_per_sw_batched(
    *,
    names: list[str],
    strengths_text: str,
    error_parts: list[str],
) -> dict[str, Any]:
    """Chunked extract of strengths/weaknesses so maxItems stays small on each schema."""
    if not strengths_text:
        return {"per_competitor": []}
    merged: list[dict[str, Any]] = []
    schema = _per_sw_chunk_schema()
    step = _PER_SW_ROWS_PER_SCHEMA
    for i in range(0, len(names), step):
        chunk = names[i : i + step]
        chunk_text = ", ".join(chunk)
        try:
            pdata = await json_with_gemini(
                prompt=(
                    "Extract structured strengths and weaknesses from the research text.\n"
                    "Return ONLY JSON matching the schema.\n"
                    f"Include one object per name in this list only: {chunk_text}\n\n"
                    f"Research text:\n{strengths_text}"
                ),
                json_schema=schema,
            )
        except Exception as exc:  # noqa: BLE001
            error_parts.append(f"per_competitor_sw rows {i}-{i + step}: {exc!s}")
            continue
        part = pdata.get("per_competitor") or []
        if isinstance(part, list):
            merged.extend([x for x in part if isinstance(x, dict)])
    return {"per_competitor": merged}


def _align_comparison_rows_with_landscape(
    comparison_json: dict[str, Any], top_data: dict[str, Any]
) -> None:
    """
    Overwrite comparison_rows[].relationship using the canonical landscape list.

    Why:
    - The second JSON model call sometimes mis-labels rows; the landscape step is the source of truth.
    - Keeps the Streamlit comparison table aligned with the top-competitors table.
    """
    rows_out = comparison_json.get("comparison_rows")
    if not isinstance(rows_out, list):
        return
    by_name: dict[str, str] = {}
    for c in top_data.get("competitors") or []:
        if not isinstance(c, dict):
            continue
        nm = str(c.get("name") or "").strip().lower()
        if not nm:
            continue
        rel = str(c.get("relationship") or "").strip().lower()
        if rel in ("direct", "indirect"):
            by_name[nm] = rel
    for r in rows_out:
        if not isinstance(r, dict):
            continue
        key = str(r.get("competitor_name") or "").strip().lower()
        if key in by_name:
            r["relationship"] = by_name[key]


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
    Node 2: find a canonical competitor list (direct + indirect), then analyze that SAME list.

    Why:
    - If we ask "top competitors" and "strengths/weaknesses" as separate open-ended prompts,
      the model can introduce new apps in the second answer.
    - We want strengths/weaknesses to be explicitly about the competitors we found.
    - We classify each row as direct vs indirect so the Builder and UI can compare positioning clearly.
    """
    if state.get("error"):
        return {}

    # Collects non-fatal issues (e.g. second competitor-chunk extract failed) for the final state.
    error_parts: list[str] = []

    product_category = (state.get("product_category") or "").strip()
    target_audience = (state.get("target_audience") or "").strip()
    user_idea = (state.get("user_input") or "").strip()

    # 1) Canonical competitor list as structured JSON.
    # We do this in TWO steps:
    # - grounded web search (text) to gather real names
    # - non-grounded JSON extraction to force a consistent list shape
    # Each row includes relationship so we show direct vs indirect competitors explicitly.
    # maxItems stays at _COMPETITOR_CHUNK_SIZE per request; see _merge_more_competitors_from_text for more rows.
    top_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "competitors": {
                "type": "array",
                "items": _competitor_item_json_schema(),
                "maxItems": _COMPETITOR_CHUNK_SIZE,
            }
        },
        "required": ["competitors"],
    }

    # Empty slots keep the graph and UI stable if the first step fails.
    failed_competitors_shell = [
        {"angle": "top competitors", "findings": ""},
        {"angle": "strengths and weaknesses", "findings": ""},
        {"angle": "per_competitor_sw", "findings": ""},
        {"angle": "structured_comparison", "findings": ""},
        {"angle": "user sentiment", "findings": ""},
    ]

    try:
        # Ask for a wide net: substitutes and adjacent players count as indirect competitors.
        grounded_top_text = await search_with_gemini(
            f"Product idea (for context): {user_idea}\n\n"
            f"Product category: {product_category}\n"
            f"Target audience: {target_audience}\n\n"
            "Identify as many plausible competitors as you can (aim for many distinct named "
            "products, apps, companies, or clear substitutes — the more names in your answer, the better).\n\n"
            "Definitions:\n"
            "- DIRECT: same solution category, overlapping users, comparable core job-to-be-done.\n"
            "- INDIRECT: substitutes, adjacent tools, manual/offline alternatives, or platforms that "
            "only partly satisfy the same underlying need.\n\n"
            "For each entry, give the name, one or two sentences on what they offer, and state whether "
            "they are direct or indirect relative to the product idea above."
        )
        top_data = await json_with_gemini(
            prompt=(
                "Extract a competitor list from the grounded research text below.\n"
                "Return ONLY JSON matching the provided schema.\n"
                "- Keep competitor names exactly as written in the text when possible.\n"
                "- relationship must be exactly 'direct' or 'indirect'.\n"
                "- relationship_rationale: one concise sentence tied to the product idea.\n"
                "- If the text has fewer than 5 clear competitors, return as many valid rows as you can.\n\n"
                f"Grounded research text:\n{grounded_top_text}"
            ),
            json_schema=top_schema,
        )
        top_list = top_data.get("competitors") or []
        if not isinstance(top_list, list) or not top_list:
            raise ValueError("Top competitors JSON was empty.")
        # Ensure relationship fields are present for the UI table (handles key/enum drift from the model).
        _normalize_competitor_landscape(top_data)
        # Second structured pass when the first hit the chunk cap — same grounded text, no duplicate names.
        try:
            await _merge_more_competitors_from_text(
                top_data=top_data,
                grounded_top_text=grounded_top_text,
                user_idea=user_idea,
            )
            _normalize_competitor_landscape(top_data)
        except Exception as exc:  # noqa: BLE001 — optional enrich; keep primary list
            error_parts.append(f"more competitors: {exc!s}")

        top_list = top_data.get("competitors") or []
        if isinstance(top_list, list) and len(top_list) > _COMPETITOR_MAX_TOTAL:
            top_data["competitors"] = top_list[:_COMPETITOR_MAX_TOTAL]
    except Exception as exc:
        return {
            "competitors": failed_competitors_shell,
            "current_agent": "builder",
            "error": f"top competitors: {exc!s}",
        }

    # Extract names for consistent follow-up prompts (after optional second merge + trim).
    top_list = top_data.get("competitors") or []
    names: list[str] = []
    for item in top_list:
        if not isinstance(item, dict):
            continue
        n = str(item.get("name") or "").strip()
        if n:
            names.append(n)

    # Cap how many names we push into grounded follow-up prompts (token budget).
    names = names[:_COMPETITOR_MAX_TOTAL]
    names_text = ", ".join(names) if names else "(unknown)"
    # Compact classification map for follow-up prompts (direct vs indirect per name).
    rel_lines: list[str] = []
    for item in top_list:
        if not isinstance(item, dict):
            continue
        n = str(item.get("name") or "").strip()
        if not n:
            continue
        rel = str(item.get("relationship") or "").strip().lower()
        rel_lines.append(f"- {n} ({rel}): {str(item.get('relationship_rationale') or '').strip()}")
    relationships_block = "\n".join(rel_lines) if rel_lines else "(not classified)"

    # 2) Analyze strengths/weaknesses and sentiment FOR THAT SAME LIST (run concurrently).
    prompts = {
        "strengths and weaknesses": (
            f"Product idea: {user_idea}\n"
            f"Category: {product_category}. Audience: {target_audience}.\n\n"
            f"Competitors (analyze ONLY these names): {names_text}\n\n"
            "Known direct vs indirect labels from prior research (do not rename competitors):\n"
            f"{relationships_block}\n\n"
            "For EACH competitor in the list, write a clear subsection with:\n"
            "- Strengths (what they do well)\n"
            "- Weaknesses (gaps, complaints, limitations)\n"
            "Rules:\n"
            "- Do not add competitors outside the list.\n"
            "- If you are unsure about one competitor, say so instead of inventing facts.\n"
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

    # 3–4) Structured per-SW + structured comparison via chunked JSON schemas (avoids Gemini 400).
    strengths_for_compare = strengths_text[:12000] if strengths_text else ""

    per_sw_json: dict[str, Any] = {"per_competitor": []}
    comparison_json: dict[str, Any] = {"executive_summary": "", "comparison_rows": []}
    extract_raw = await asyncio.gather(
        _run_per_sw_batched(
            names=names,
            strengths_text=strengths_text,
            error_parts=error_parts,
        ),
        _run_structured_comparison_batched(
            user_idea=user_idea,
            names=names,
            names_text=names_text,
            top_data=top_data,
            strengths_for_compare=strengths_for_compare,
            error_parts=error_parts,
        ),
        return_exceptions=True,
    )
    sw_ex, cmp_ex = extract_raw
    if isinstance(sw_ex, Exception):
        error_parts.append(f"per_competitor_sw: {sw_ex!s}")
    elif isinstance(sw_ex, dict):
        per_sw_json = sw_ex
    if isinstance(cmp_ex, Exception):
        error_parts.append(f"structured_comparison: {cmp_ex!s}")
    elif isinstance(cmp_ex, dict):
        comparison_json = cmp_ex

    competitors_out = [
        {"angle": "top competitors", "findings": json.dumps(top_data, ensure_ascii=False, indent=2)},
        {"angle": "strengths and weaknesses", "findings": strengths_text},
        {
            "angle": "per_competitor_sw",
            "findings": json.dumps(per_sw_json, ensure_ascii=False, indent=2),
        },
        {
            "angle": "structured_comparison",
            "findings": json.dumps(comparison_json, ensure_ascii=False, indent=2),
        },
        {"angle": "user sentiment", "findings": sentiment_text},
    ]

    out: dict[str, Any] = {
        "competitors": competitors_out,
        "current_agent": "builder",
    }
    if error_parts:
        out["error"] = "; ".join(error_parts)
    return out
