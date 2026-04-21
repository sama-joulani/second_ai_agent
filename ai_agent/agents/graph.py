"""
Full pipeline: Strategic Analyst -> Builder <-> QA (revision loop) until approved or max loops.
"""

from langgraph.graph import END, START, StateGraph

from state.schema import AgentState

from agents.builder.nodes import build_comparison_report
from agents.qa.nodes import run_quality_review
from agents.strategic_analyst.nodes import extract_product_info, search_competitors


def _route_after_extract(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "search_competitors"


def _route_after_qa(state: AgentState) -> str:
    """
    After QA: either finish, or send the report back to the Builder with feedback.
    """
    if state.get("qa_route") == "builder":
        return "build_comparison_report"
    return "end"


def build_graph():
    """
    extract -> search -> build -> QA -> (optional) build again ... -> END

    QA sets qa_route to "builder" to request another Builder pass, or "end" when done.
    """
    graph = StateGraph(AgentState)

    graph.add_node("extract_product_info", extract_product_info)
    graph.add_node("search_competitors", search_competitors)
    graph.add_node("build_comparison_report", build_comparison_report)
    graph.add_node("run_quality_review", run_quality_review)

    graph.add_edge(START, "extract_product_info")
    graph.add_conditional_edges(
        "extract_product_info",
        _route_after_extract,
        {
            "end": END,
            "search_competitors": "search_competitors",
        },
    )
    graph.add_edge("search_competitors", "build_comparison_report")
    graph.add_edge("build_comparison_report", "run_quality_review")
    graph.add_conditional_edges(
        "run_quality_review",
        _route_after_qa,
        {
            "end": END,
            "build_comparison_report": "build_comparison_report",
        },
    )

    return graph.compile()
