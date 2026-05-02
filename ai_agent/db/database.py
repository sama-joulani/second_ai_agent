"""
SQLite persistence for agent runs using aiosqlite.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from state.schema import AgentState

# Store the database beside this package under the project root.
_DB_PATH = Path(__file__).resolve().parent.parent / "runs.db"


async def _ensure_qa_report_column(conn: aiosqlite.Connection) -> None:
    """Older DB files may lack qa_report; add it without losing rows."""
    cursor = await conn.execute("PRAGMA table_info(runs);")
    rows = await cursor.fetchall()
    column_names = [row[1] for row in rows]
    if "qa_report" not in column_names:
        await conn.execute("ALTER TABLE runs ADD COLUMN qa_report TEXT;")
        await conn.commit()


async def _ensure_progress_columns(conn: aiosqlite.Connection) -> None:
    """
    Older DB files may lack progress fields used by the UI.

    We add them in-place so existing runs remain readable.
    """
    cursor = await conn.execute("PRAGMA table_info(runs);")
    rows = await cursor.fetchall()
    column_names = [row[1] for row in rows]

    if "current_agent" not in column_names:
        await conn.execute("ALTER TABLE runs ADD COLUMN current_agent TEXT;")
        await conn.commit()

    if "qa_revision_count" not in column_names:
        await conn.execute("ALTER TABLE runs ADD COLUMN qa_revision_count INTEGER;")
        await conn.commit()


async def _ensure_schema(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            user_input TEXT,
            product_category TEXT,
            target_audience TEXT,
            competitors TEXT,
            comparison_report TEXT,
            qa_report TEXT,
            status TEXT,
            current_agent TEXT,
            qa_revision_count INTEGER
        );
        """
    )
    await conn.commit()
    await _ensure_qa_report_column(conn)
    await _ensure_progress_columns(conn)


async def save_run(state: AgentState) -> int:
    """
    Persist one run. Serializes competitors as JSON. Returns the new row id.
    """
    status = "error" if state.get("error") else "completed"
    created_at = datetime.now(timezone.utc).isoformat()

    competitors_json = json.dumps(state.get("competitors") or [])

    async with aiosqlite.connect(_DB_PATH) as conn:
        await _ensure_schema(conn)
        cursor = await conn.execute(
            """
            INSERT INTO runs (
                created_at, user_input, product_category, target_audience,
                competitors, comparison_report, qa_report, status,
                current_agent, qa_revision_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                created_at,
                state.get("user_input"),
                state.get("product_category"),
                state.get("target_audience"),
                competitors_json,
                state.get("comparison_report"),
                state.get("qa_report"),
                status,
                state.get("current_agent"),
                int(state.get("qa_revision_count") or 0),
            ),
        )
        await conn.commit()
        return int(cursor.lastrowid)


async def get_run(run_id: int) -> dict[str, Any]:
    """Load a single run by primary key. Parses competitors JSON back into a list."""
    async with aiosqlite.connect(_DB_PATH) as conn:
        await _ensure_schema(conn)
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM runs WHERE id = ?;",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        raise KeyError(f"No run with id {run_id}")

    data = dict(row)
    # UI expects "run_id" (CLI adds it after save). DB stores the primary key as "id".
    # Keep both so callers can use whichever they prefer.
    data["run_id"] = int(data.get("id") or run_id)
    # Backward-compatible defaults for older runs (before progress columns existed).
    data["qa_revision_count"] = int(data.get("qa_revision_count") or 0)
    data["current_agent"] = data.get("current_agent") or "complete"
    raw = data.get("competitors")
    if isinstance(raw, str) and raw:
        try:
            data["competitors"] = json.loads(raw)
        except json.JSONDecodeError:
            data["competitors"] = []
    else:
        data["competitors"] = []
    return data
