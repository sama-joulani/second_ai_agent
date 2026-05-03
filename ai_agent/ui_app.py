"""
Simple UI for running and viewing your AI agent pipeline.

Why Streamlit:
- Very small amount of code for a clean interface.
- No frontend build step.
- Great for quick internal tools.

How to run (you run these commands yourself):
  pip install -r requirements.txt
  streamlit run ui_app.py

Notes:
- This UI calls the SAME LangGraph pipeline used by main.py.
- It reads GEMINI_API_KEY (and optional GEMINI_MODEL) from .env.
- It stores results in SQLite via db/database.py (same as CLI).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from agents.graph import build_graph
from db.database import get_run, save_run
from state.schema import AgentState


def _initial_state(user_input: str) -> AgentState:
    """
    Build a complete AgentState dict.

    Keeping all keys present makes debugging easier and avoids KeyError surprises.
    """
    return {
        "user_input": user_input,
        "product_category": None,
        "target_audience": None,
        "competitors": None,
        "comparison_report": None,
        "qa_report": None,
        "qa_feedback_for_builder": None,
        "qa_revision_count": 0,
        "qa_route": None,
        "current_agent": "strategic_analyst",
        "error": None,
    }


def _competitor_row_relationship(row: dict) -> str:
    """
    Show direct vs indirect in tables.

    Uses the same key fallbacks as the analyst post-processor so older SQLite runs
    still render correctly. Avoid column name 'Type' in st.dataframe (Arrow/Streamlit
    edge cases with type-like names).
    """
    for key in (
        "relationship",
        "Relationship",
        "competitor_type",
        "competitorType",
        "direct_or_indirect",
        "directOrIndirect",
    ):
        val = row.get(key)
        if val is None or str(val).strip() == "":
            continue
        s = str(val).strip().lower().replace("_", " ")
        if s in ("direct", "indirect"):
            return s
        if s.startswith("direct") or "direct competitor" in s:
            return "direct"
        if s.startswith("indirect") or "indirect competitor" in s:
            return "indirect"
        if "substitute" in s or "adjacent" in s:
            return "indirect"
    blob = (
        f"{row.get('relationship_rationale', '')} "
        f"{row.get('relationshipRationale', '')} "
        f"{row.get('description', '')}"
    ).lower()
    if "indirect" in blob or "substitute" in blob or "adjacent" in blob:
        return "indirect"
    if "direct" in blob or "same category" in blob:
        return "direct"
    return "—"


def _comparison_row_relationship(row: dict) -> str:
    """Same idea for structured_comparison rows (different key: competitor_name)."""
    val = row.get("relationship") or row.get("Relationship")
    if val is None or str(val).strip() == "":
        return "—"
    s = str(val).strip().lower().replace("_", " ")
    if s in ("direct", "indirect"):
        return s
    if s.startswith("direct"):
        return "direct"
    if s.startswith("indirect") or "substitute" in s:
        return "indirect"
    return "—"


def _json(obj: Any) -> str:
    """Pretty JSON for display; never raise inside the UI."""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


async def _run_pipeline(user_input: str) -> Dict[str, Any]:
    """
    Run the 3-agent pipeline and persist to SQLite.

    Returns a dict that includes run_id plus all state fields.
    """
    graph = build_graph()
    state = _initial_state(user_input)
    final_state = await graph.ainvoke(state)
    run_id = await save_run(final_state)
    out = dict(final_state)
    out["run_id"] = run_id
    return out


def _coerce_stream_item(item: Any) -> Optional[Tuple[Optional[str], Dict[str, Any]]]:
    """
    Best-effort normalization for LangGraph streaming outputs.

    Different LangGraph versions / stream modes can yield:
    - a full state dict
    - (node_name, state_dict)
    - a dict keyed by node name -> partial update

    We normalize into (node_name, state_dict_or_partial_update).
    If we can't recognize the shape, return None (safe to ignore).
    """
    if isinstance(item, dict):
        # Could be full state OR {node_name: partial_update}
        if len(item) == 1:
            (only_key,) = item.keys()
            maybe_partial = item.get(only_key)
            if isinstance(maybe_partial, dict):
                return str(only_key), dict(maybe_partial)
        return None, dict(item)

    if isinstance(item, tuple) and len(item) == 2:
        node, payload = item
        if isinstance(payload, dict):
            return (str(node) if node is not None else None), dict(payload)

    return None


async def _run_pipeline_with_live_updates(
    user_input: str,
    on_state_update: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """
    Run the pipeline but surface progress to the UI while it runs.

    Why the UI previously looked "stuck":
    - Streamlit only re-renders when Python yields control back to it.
    - Our old code awaited graph.ainvoke(...) and only then wrote last_output.
    - So `current_agent` could not be shown mid-run.

    This implementation uses LangGraph streaming (when available) and calls
    on_state_update(...) after each step so Streamlit can update a status widget.
    """
    graph = build_graph()
    state: Dict[str, Any] = dict(_initial_state(user_input))

    # Prefer streaming for live updates. Fall back to ainvoke if streaming isn't available
    # in the installed LangGraph version.
    if hasattr(graph, "astream"):
        async for item in graph.astream(state):  # type: ignore[attr-defined]
            coerced = _coerce_stream_item(item)
            if not coerced:
                continue
            _node, delta_or_state = coerced

            # We don't know if this is a delta or full state for sure, but both are safe
            # to merge because state is a dict and keys overwrite with latest values.
            state.update(delta_or_state)
            on_state_update(state)
    else:
        final_state = await graph.ainvoke(state)
        state = dict(final_state)
        on_state_update(state)

    run_id = await save_run(state)  # type: ignore[arg-type]
    out = dict(state)
    out["run_id"] = run_id
    return out


def _render_competitors(competitors: Any) -> None:
    """
    Render the competitor findings list in a readable way.

    The strategic analyst stores a list of dicts like:
      {"angle": "...", "findings": "..."}
    """
    if not competitors:
        st.info("No competitor findings found.")
        return

    if not isinstance(competitors, list):
        st.code(_json(competitors))
        return

    for item in competitors:
        angle = (item or {}).get("angle", "unknown angle")
        findings = (item or {}).get("findings", "")
        with st.expander(angle, expanded=True):
            if not findings:
                st.warning("Empty findings for this angle.")
                continue

            # "top competitors" is stored as pretty JSON (for consistency between agent steps).
            # Render it as a readable list instead of raw JSON.
            if angle == "top competitors" and isinstance(findings, str):
                try:
                    data = json.loads(findings)
                    comp_list = (data or {}).get("competitors")
                    if isinstance(comp_list, list) and comp_list:
                        rows = []
                        for c in comp_list:
                            if not isinstance(c, dict):
                                continue
                            name = str(c.get("name") or "").strip()
                            desc = str(c.get("description") or "").strip()
                            rel = _competitor_row_relationship(c)
                            rationale = (
                                str(c.get("relationship_rationale") or "").strip()
                                or str(c.get("relationshipRationale") or "").strip()
                            )
                            if name or desc:
                                rows.append(
                                    {
                                        "Name": name or "—",
                                        # Not named "Type" — some Streamlit/Arrow paths treat that like metadata.
                                        "Direct or indirect": rel,
                                        "Description": desc or "—",
                                        "Why direct/indirect": rationale or "—",
                                    }
                                )

                        if rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                        else:
                            st.code(_json(data))
                        continue
                except Exception:
                    # If parsing fails, fall back to showing the raw text.
                    pass

            # Structured strengths/weaknesses per competitor (JSON from analyst extraction step).
            if angle == "per_competitor_sw" and isinstance(findings, str):
                try:
                    pdata = json.loads(findings)
                    plist = (pdata or {}).get("per_competitor")
                    if isinstance(plist, list) and plist:
                        sw_rows = []
                        for row in plist:
                            if not isinstance(row, dict):
                                continue
                            sw_rows.append(
                                {
                                    "Competitor": str(row.get("name") or "").strip() or "—",
                                    "Strengths": str(row.get("strengths") or "").strip() or "—",
                                    "Weaknesses": str(row.get("weaknesses") or "").strip() or "—",
                                }
                            )
                        if sw_rows:
                            st.dataframe(sw_rows, use_container_width=True, hide_index=True)
                            continue
                except Exception:
                    pass

            # Comparison matrix vs the user's product idea (executive summary + table).
            if angle == "structured_comparison" and isinstance(findings, str):
                try:
                    cdata = json.loads(findings)
                    summary = str((cdata or {}).get("executive_summary") or "").strip()
                    if summary:
                        st.markdown("**Executive summary**")
                        st.write(summary)
                    crows = (cdata or {}).get("comparison_rows")
                    if isinstance(crows, list) and crows:
                        cmp_table = []
                        for r in crows:
                            if not isinstance(r, dict):
                                continue
                            analysis = str(r.get("analysis") or "").strip()
                            htc = str(r.get("how_they_compare") or "").strip()
                            srel = str(r.get("strengths_relative_to_idea") or "").strip()
                            wrel = str(r.get("weaknesses_relative_to_idea") or "").strip()
                            diff = str(r.get("differentiation_angle") or "").strip()
                            # Chunked comparison uses a single `analysis` field per row (slim schema).
                            if analysis and not (htc or srel or wrel or diff):
                                cmp_table.append(
                                    {
                                        "Competitor": str(r.get("competitor_name") or "").strip() or "—",
                                        "Direct or indirect": _comparison_row_relationship(r),
                                        "Analysis vs your idea": analysis,
                                    }
                                )
                            else:
                                cmp_table.append(
                                    {
                                        "Competitor": str(r.get("competitor_name") or "").strip() or "—",
                                        "Direct or indirect": _comparison_row_relationship(r),
                                        "How they compare": htc or analysis or "—",
                                        "Their strengths vs your idea": srel or "—",
                                        "Their weaknesses vs your idea": wrel or "—",
                                        "Differentiation for your idea": diff or "—",
                                    }
                                )
                        if cmp_table:
                            st.markdown("**Comparison vs your product idea**")
                            st.dataframe(cmp_table, use_container_width=True, hide_index=True)
                            continue
                except Exception:
                    pass

            st.write(findings)


def _render_flow_diagram() -> None:
    """
    Render a Mermaid diagram of the pipeline flow.

    Streamlit does not render Mermaid natively, so we embed Mermaid via a small HTML component.
    """
    mermaid = r"""
flowchart LR
  A[User input] --> SA1[Strategic Analyst\nextract_product_info]
  SA1 -->|ok| SA2[Strategic Analyst\nsearch_competitors]
  SA1 -->|error| END((END))

  SA2 --> B[Builder\nbuild_comparison_report]
  B --> QA[QA\nrun_quality_review]

  QA -->|approved / qa_route=end| END
  QA -->|not approved / qa_route=builder| B
"""

    html = f"""
<div class="mermaid">{mermaid}</div>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{ startOnLoad: true, theme: "default" }});
</script>
"""
    components.html(html, height=260, scrolling=False)


def main() -> None:
    # Load .env once at app start so GEMINI_API_KEY is available.
    load_dotenv()

    st.set_page_config(page_title="Competitor Research Agent", layout="wide")
    st.title("Competitor Research Agent")
    st.caption("Strategic Analyst → Builder → QA (Gemini only)")

    # Session storage so we can keep the last output on screen across reruns.
    if "last_output" not in st.session_state:
        st.session_state["last_output"] = None

    with st.sidebar:
        st.subheader("Run controls")
        user_input = st.text_area(
            "Product idea",
            value="a budgeting app for college students",
            height=120,
            help="Describe the product idea in one sentence.",
        )

        run_clicked = st.button("Run agent", type="primary")

        st.divider()
        st.subheader("Load a past run")
        run_id = st.number_input("Run id", min_value=1, step=1, value=1)
        load_clicked = st.button("Load run")

        st.divider()
        show_flow = st.checkbox("Show flow diagram", value=True)

        st.divider()
        show_raw = st.checkbox("Show raw JSON", value=False)

    if run_clicked:
        if not user_input.strip():
            st.error("Please enter a product idea.")
        else:
            status = st.status("Running agent pipeline...", expanded=True)
            status.write("Starting…")

            def _on_update(s: Dict[str, Any]) -> None:
                # Keep this small: it runs many times and we want fast UI updates.
                current = s.get("current_agent") or "—"
                rev = s.get("qa_revision_count", "—")
                status.update(label=f"Running… (current_agent: {current})")
                status.write(f"- current_agent: {current}\n- qa_revision_count: {rev}")

            try:
                # Streamlit is sync; use asyncio.run to execute the async pipeline.
                # We also push incremental updates to the status widget during execution.
                st.session_state["last_output"] = asyncio.run(
                    _run_pipeline_with_live_updates(user_input.strip(), _on_update)
                )
                status.update(label="Done", state="complete")
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(f"Failed to run pipeline: {exc}")

    if load_clicked:
        with st.spinner("Loading from SQLite..."):
            try:
                st.session_state["last_output"] = asyncio.run(get_run(int(run_id)))
            except Exception as exc:
                st.error(f"Failed to load run {int(run_id)}: {exc}")

    output = st.session_state.get("last_output")
    if not output:
        st.info("Run the agent or load a previous run to see results.")
        return

    if show_flow:
        st.subheader("Pipeline flow")
        _render_flow_diagram()

    # Streamlit's st.metric uses a fairly large, fixed-looking font and can truncate long text.
    # We render Category/Audience with smaller, wrapping text so long values stay readable.
    st.markdown(
        """
<style>
  .agent-kv-label { font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.15rem; }
  .agent-kv-value { font-size: 0.95rem; line-height: 1.15rem; word-break: break-word; }
</style>
""",
        unsafe_allow_html=True,
    )

    # Top summary row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Run id", output.get("run_id", "—"))
    col2.markdown(
        f"""
<div class="agent-kv-label">Category</div>
<div class="agent-kv-value">{(output.get("product_category") or "—")}</div>
""",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
<div class="agent-kv-label">Audience</div>
<div class="agent-kv-value">{(output.get("target_audience") or "—")}</div>
""",
        unsafe_allow_html=True,
    )
    col4.metric("QA revisions used", output.get("qa_revision_count", "—"))
    col5.metric("Current agent", output.get("current_agent") or "—")

    if output.get("error"):
        st.warning(output["error"])

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Competitor research (grounded search)")
        _render_competitors(output.get("competitors"))

    with right:
        st.subheader("Builder report (Markdown)")
        report = (output.get("comparison_report") or "").strip()
        if report:
            st.markdown(report)
        else:
            st.info("No comparison_report generated.")

        st.subheader("QA final deliverable (Markdown)")
        qa = (output.get("qa_report") or "").strip()
        if qa:
            st.markdown(qa)
        else:
            st.info("No qa_report generated.")

    if show_raw:
        st.subheader("Raw output")
        st.code(_json(output), language="json")


if __name__ == "__main__":
    main()

