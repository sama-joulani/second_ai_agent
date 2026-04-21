# Competitor Research Tool — Phase 1: Strategic Analyst Agent

> **Instructions for Cursor**: Build ONLY what is described in this document.
> Do not scaffold Agent 2 or Agent 3. Do not create placeholder files for future agents.
> When Phase 1 is complete and tested, a separate spec will be provided for Agent 2.

---

## What this system does

The user provides a product idea or business (e.g. "a meal planning app for busy parents").
The system researches competitors, summarizes their strengths and weaknesses, and produces
a clean comparison report.

**Agent 1 (this phase)**: Understands the product idea and searches for competitors.
**Agent 2 (next phase)**: Synthesizes findings into a structured comparison.
**Agent 3 (final phase)**: QA checks the report and produces the final output.

---

## Project Structure (Phase 1 only)

```
project/
├── state/
│   └── schema.py          # Shared LangGraph TypedDict state definition
├── agents/
│   └── strategic_analyst/
│       ├── __init__.py
│       ├── agent.py       # LangGraph graph definition for Agent 1
│       ├── nodes.py       # All node functions
│       └── tools.py       # Tool definitions
├── db/
│   └── database.py        # SQLite setup using aiosqlite
├── main.py                # Entry point to run Agent 1 standalone
├── requirements.txt
└── .env.example
```

---

## Shared State Schema — `state/schema.py`

Define this now. All 3 agents will share it. Only Agent 1 populates its fields in Phase 1.

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Input
    user_input: str                  # e.g. "a meal planning app for busy parents"

    # Agent 1 outputs
    product_category: Optional[str]  # e.g. "meal planning app"
    target_audience: Optional[str]   # e.g. "busy parents"
    competitors: Optional[list]      # List of competitor finding dicts (see Node 2)

    # Agent 2 outputs (empty in Phase 1)
    comparison_report: Optional[str]

    # Agent 3 outputs (empty in Phase 1)
    qa_report: Optional[str]

    # Control
    current_agent: str               # Which agent is currently active
    error: Optional[str]             # Any error message
```

---

## Agent 1 — Strategic Analyst

### Role
Receives a raw product idea, extracts the product category and target audience,
then uses Gemini with Google Search Grounding to find and summarize real competitors.
Writes all results to the shared state.

### When it runs
- Always first. Triggered by user input.
- Hands off to Agent 2 by setting `current_agent = "builder"` in state.

---

### Nodes — `agents/strategic_analyst/nodes.py`

All node functions must be `async`. Each receives `state: AgentState` and returns a partial state dict.

#### Node 1: `extract_product_info`

**Purpose**: Use an LLM call to extract the product category and target audience
from the raw user input.

**When called**: First node in the graph.

**What it does**:
- Calls the LLM with a prompt asking it to extract:
  - `product_category` — what type of product it is (e.g. "meal planning app")
  - `target_audience` — who it's for (e.g. "busy parents")
- Parses the LLM response as JSON
- Writes both fields to state

**Prompt template**:
```
You are a business analyst. Extract the product category and target audience
from this product idea.
Return ONLY valid JSON with exactly two keys: product_category, target_audience.
Keep both values short (under 8 words each).

Product idea: {user_input}
```

**Returns**: `{"product_category": str, "target_audience": str, "current_agent": "strategic_analyst"}`

---

#### Node 2: `search_competitors`

**Purpose**: Use Gemini with Google Search Grounding to find real competitors
and summarize each one.

**When called**: After `extract_product_info` succeeds.

**What it does**:
- Reads `state["product_category"]` and `state["target_audience"]`
- Builds 3 prompts:
  1. `"Who are the top competitors in the {product_category} market for {target_audience}? List the main ones with a brief description of each."`
  2. `"What are the strengths and weaknesses of the leading {product_category} apps in 2025?"`
  3. `"What do users complain about most in {product_category} apps? What do they love?"`
- Calls `search_with_gemini` for all 3 concurrently using `asyncio.gather`
- Stores results as a list of dicts:
  ```python
  [
    {"angle": "top competitors",        "findings": "<gemini response text>"},
    {"angle": "strengths and weaknesses","findings": "<gemini response text>"},
    {"angle": "user sentiment",         "findings": "<gemini response text>"},
  ]
  ```
- Writes to `state["competitors"]`

**Tool used**: `search_with_gemini` (see tools.py)

**Returns**: `{"competitors": <list of finding dicts>, "current_agent": "builder"}`

---

### Tools — `agents/strategic_analyst/tools.py`

#### Tool: `search_with_gemini`

```python
import os
import asyncio
import google.generativeai as genai

async def search_with_gemini(prompt: str) -> str:
    """
    Call Gemini with Google Search Grounding enabled.
    Gemini searches Google internally and returns a clean synthesized response.
    Returns the response as a plain string.
    Use GEMINI_API_KEY from environment.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Wrap sync SDK call in thread pool to stay async-compatible
    response = await asyncio.to_thread(
        model.generate_content,
        contents=prompt,
        tools="google_search_retrieval"
    )

    if not response.text:
        raise ValueError(f"Empty response from Gemini for prompt: {prompt}")

    return response.text
```

---

### Graph Definition — `agents/strategic_analyst/agent.py`

```
Nodes:
  - extract_product_info
  - search_competitors

Edges:
  START --> extract_product_info
  extract_product_info --> search_competitors
  search_competitors --> END

Conditional edge (error handling):
  After each node, if state["error"] is set → route to END
```

Use `StateGraph(AgentState)` from LangGraph.
Compile with `.compile()` — no checkpointer needed in Phase 1.

---

### Database — `db/database.py`

Use `aiosqlite`. Create one table:

```sql
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    user_input TEXT,
    product_category TEXT,
    target_audience TEXT,
    competitors TEXT,        -- JSON string
    comparison_report TEXT,  -- empty in Phase 1
    status TEXT              -- "completed" | "error"
);
```

Provide two async functions:
- `async def save_run(state: AgentState) -> int` — saves a completed run, returns row id
- `async def get_run(run_id: int) -> dict` — retrieves a run by id

---

### Entry Point — `main.py`

```python
# Runs Agent 1 standalone for testing.
# Accepts user_input from CLI argument or hardcoded test string.
# Prints the final state to stdout as formatted JSON.
# Saves the run to SQLite.
# Example: python main.py "a meal planning app for busy parents"
```

---

### Requirements — `requirements.txt`

```
langgraph>=0.2.0
langchain-openai>=0.1.0
google-generativeai>=0.8.0
aiosqlite>=0.19.0
python-dotenv>=1.0.0
```

---

### Environment — `.env.example`

```
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

---

## Testing Checklist (do before moving to Phase 2)

Run `python main.py "a task manager app for freelancers"` and verify:

- [ ] `product_category` and `target_audience` are both short, clean strings
- [ ] `competitors` list has exactly 3 items, each with a non-empty `findings` string
- [ ] Run is saved to SQLite and retrievable by id
- [ ] No crash if Gemini returns an empty response (should set `state["error"]` instead)

---

## What comes next (Phase 2 — Builder Agent)

After Phase 1 tests pass, a new file `AGENT2_SPEC.md` will be provided.
Agent 2 will:
- Read `state["competitors"]` from the blackboard
- Use an LLM to synthesize the 3 findings into a structured markdown comparison report
- Write `state["comparison_report"]`
- Hand off to Agent 3

**Do not build any of this yet.**