"""
How many times QA may send the report back to the Builder before we stop looping.

Set MAX_QA_REVISION_LOOPS in the environment (default 3).
Each loop is one extra Builder pass after a QA rejection.
"""

import os


def max_qa_revision_loops() -> int:
    raw = os.getenv("MAX_QA_REVISION_LOOPS", "3")
    try:
        n = int(raw)
    except ValueError:
        return 3
    return max(0, min(n, 10))
