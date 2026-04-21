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
from typing import Any, Dict

import streamlit as st
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
            if findings:
                st.write(findings)
            else:
                st.warning("Empty findings for this angle.")


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
        show_raw = st.checkbox("Show raw JSON", value=False)

    if run_clicked:
        if not user_input.strip():
            st.error("Please enter a product idea.")
        else:
            with st.spinner("Running agent pipeline..."):
                # Streamlit is sync; use asyncio.run to execute the async pipeline.
                # This keeps the agent code clean and fully async internally.
                st.session_state["last_output"] = asyncio.run(_run_pipeline(user_input.strip()))

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

    # Top summary row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Run id", output.get("run_id", "—"))
    col2.metric("Category", output.get("product_category") or "—")
    col3.metric("Audience", output.get("target_audience") or "—")
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

