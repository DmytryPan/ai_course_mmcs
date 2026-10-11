# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A clone of the course repository «Агентный ИИ: теория и инженерия» (MMCS, autumn 2026, upstream `github.com/myurushkin/ai_course_mmcs`). It holds course materials by week (`week1/`, `week2/`, …: lecture slides, practice, assignments) plus the student's own work. All materials and communication are in Russian.

- `week*/`: course materials, maintained upstream. Don't rewrite them; only the student's code goes into the practice notebook (`# ЗАДАЧА N` cells).
- `ai_course_agents/`: the student's own uv project for the MVP service (Python 3.11, `google-adk`, `litellm`, `qdrant-client`, `fastembed`, src layout `src/ai_course_agents/`). The work plan and stage status are in `ai_course_agents/docs/PLAN.md`; update the status table after each stage.
- `ai_course_agents/practice/`: a **separate, standalone** uv project just for the notebook (`openai-agents`, `openai>=3`, `ipykernel`, dev `nbclient`). It is separate because `litellm>=1.84` needs `openai<3`. Don't merge the two environments, and don't make `practice` a uv workspace member: `uv init` adds it to the workspace by default, so use `--no-workspace`.

## Week 2: the data and the practice

- `week2/practice/poi.sqlite`: 8860 places in Rostov-on-Don from OpenStreetMap (ODbL), one `poi` table. Key fields: `kind` (Latin OSM value: `cafe`, `restaurant`, `pharmacy`, `outpost` = order pickup point …), `category`, `name`, `brand`, `street`/`housenumber`, `cuisine`, `is_24_7`/`internet`/`wheelchair` (1/0/NULL, where NULL means "no data", not "no"), `lat`/`lon`. The data is dirty on purpose: many places lack a name, hours or address.
- Rebuild the DB from the snapshot: `python make_poi_dataset.py` (in `week2/practice`); `--refresh` downloads it again from Overpass.
- `week2/practice/sdk_tasks.py` holds ready-made tools over the DB (`find_places`, `count_places`, `top_brands`, `address_point`, `nearest`, `details`, `run_query`), the `distance_m`/`table` helpers, and the `check_*` functions the notebook uses to check tasks. Tools log their calls into the global `CALLS`, and the checks read from it.
- The student's working copy of the notebook is `ai_course_agents/practice/week02_agents.ipynb` (kernel: `ai_course_agents/practice/.venv`). Its first cell adds `../week2/practice` to `sys.path` and loads `.env` (`DEEPSEEK_API_KEY`). The setup cell configures DeepSeek (`base_url=https://api.deepseek.com`, models `deepseek-flash` / `deepseek-v4-pro`). The user wants very detailed Russian comments in the notebook code. Check cells in a real kernel with `nbclient` (a dev dependency of `practice`) by running cells `0..N` via `uv run` from `ai_course_agents/practice`. Don't use `exec` of the cell code: the kernel's cwd and top-level `await` behave differently there. Kernel cwd in VS Code can be either the notebook folder or the workspace root, so the setup cell finds the repo root by walking up the tree.
- In the service, LiteLLM's model string is `deepseek/deepseek-flash` (it parses `reasoning_content` → `Part(thought=True)`; DeepSeek's reasoning is in English). DeepSeek limitations (they matter for the ADK MVP too): no `response_format: json_schema`, so an agent's `output_type` returns 400. The workaround is "final answer as a tool": `StopAtTools` plus a tool whose argument is a pydantic model and which returns JSON. `deepseek-flash` runs in thinking mode, where `tool_choice="required"` is not supported.
- `week02_agents.ipynb`: 11 tasks on the **OpenAI Agents SDK** (`openai-agents`, not ADK). Dependencies are in `week2/practice/requirements.txt`. Cell kinds: `# ДАНО` (given code), `# ЗАДАЧА N` (fill in the `"..."` placeholders), `# ПРОВЕРКА` (check). Inside the notebook, use `await Runner.run(...)`; `run_sync` fails because an event loop is already running. With a non-OpenAI provider you need `set_default_openai_api("chat_completions")`, your own `AsyncOpenAI(base_url=...)` client and tracing turned off.

## Assignment: an MVP on ADK

The base task is in `week2/assignment.md`. The teacher extended it into a working MVP web service on Python + Google ADK, graded on UX and architecture choices:
- frontend with a question form (not reading from a file);
- async multi-user backend with authorization;
- SSE streaming of the agent's steps (tool calls and results, text);
- conversation context (sessions);
- semantic search through a vector DB (use case: the user typed "пельмени" and the agent finds restaurants with a similar meaning).

Decisions already made: LLM is DeepSeek via LiteLLM (key in `ai_course_agents/.env`), embeddings come from a local multilingual model, frontend has no build step (served by the backend), scope is the MVP only (no `check.py` or A/B report).

## Commands

```bash
cd ai_course_agents
uv sync                          # install dependencies from uv.lock
uv add <pkg>                     # add a dependency (updates pyproject.toml and uv.lock)
uv run ai-course-agents          # entry point = ai_course_agents:main
uv run python scripts/smoke_llm.py [--mock]   # ADK → LiteLLM → DeepSeek check (--mock = no network)
uv run python -m <module>
```

No tests or linter are configured yet.

## Environment

- Windows + PowerShell/Git Bash. The default console code page garbles Cyrillic from SQLite and Python: run with `python -X utf8` or set `PYTHONIOENCODING=utf-8`.
- Model access is through an OpenAI-compatible API. The course convention is the env variables `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL`. Keys live only in `.env`.
- Course rule: every report has a section «что делал агент, что проверял я», and at the defense the student must explain any line of code.
