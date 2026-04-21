"""
Run the competitor-research pipeline (Strategic Analyst, Builder, QA) locally.

Example:
  python main.py "a meal planning app for busy parents"
"""

import argparse
import asyncio
import json
import sys
from typing import Any

from dotenv import load_dotenv

from agents.graph import build_graph
from db.database import save_run
from state.schema import AgentState


def _build_initial_state(user_input: str) -> AgentState:
    # All TypedDict keys are present so downstream nodes can rely on .get().
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


def _state_to_jsonable(state: AgentState) -> dict[str, Any]:
    """Convert state to a plain dict suitable for json.dumps."""
    return dict(state)


async def _async_main(user_input: str) -> None:
    load_dotenv()

    graph = build_graph()
    initial = _build_initial_state(user_input)
    final_state = await graph.ainvoke(initial)

    run_id = await save_run(final_state)

    out = _state_to_jsonable(final_state)
    out["run_id"] = run_id
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run competitor research pipeline (Strategic Analyst, Builder, QA).",
    )
    parser.add_argument(
        "user_input",
        nargs="?",
        default="a meal planning app for busy parents",
        help="Product idea to analyze (optional; defaults to a sample string).",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_async_main(args.user_input))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
