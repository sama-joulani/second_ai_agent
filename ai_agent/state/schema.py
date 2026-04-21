"""
Shared LangGraph TypedDict state for all agents.
"""

from typing import Optional, TypedDict


class AgentState(TypedDict):
    # Input from the user (required).
    user_input: str

    # Agent 1 outputs — filled by the strategic analyst in Phase 1.
    product_category: Optional[str]
    target_audience: Optional[str]
    competitors: Optional[list]

    # Agent 2 / 3 outputs.
    comparison_report: Optional[str]
    qa_report: Optional[str]

    # QA ↔ Builder revision loop (see agents/graph.py).
    # qa_feedback_for_builder: concrete instructions for the next Builder pass (set by QA when not approved).
    qa_feedback_for_builder: Optional[str]
    # How many times QA has already sent the report back for rewrite (0 = first QA pass).
    qa_revision_count: Optional[int]
    # qa_route: "builder" to loop back to Builder, "end" to finish (set by QA node).
    qa_route: Optional[str]

    # Tracks which agent is active and any failure message for routing / UX.
    current_agent: str
    error: Optional[str]
