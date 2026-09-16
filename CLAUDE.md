# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Research study that uses LLMs to classify disinformation articles against the **DISARM framework** (a MITRE ATT&CK-style taxonomy of disinformation tactics and techniques). Given article text, the code prompts an LLM (either a locally-hosted open-weight model or an OpenAI GPT model) to identify which DISARM tactics/techniques the *author or publisher* employed.

## Setup and commands

There is no build system, linter, or test suite configured (no pytest/pyproject.toml/lint config exists yet). The workflow is:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # requests, openai, pandas, openpyxl
```

Run the interface's manual demo/smoke test (classifies a hardcoded sample article):

```bash
python scripts/DISARM_LLM_Interface.py
```

This executes `test_interface()` at the bottom of the file. Edit the `DISARM_LLM(...)` construction there to switch between a local model and GPT.

Run the agent console (web GUI) — configure a run, feed it pasted/uploaded articles or the dataset, and watch the agent work step by step:

```bash
.venv/bin/python scripts/gui/app.py   # http://localhost:5000
```

Environment/external dependencies:
- `OPENAI_API_KEY` must be set in the environment to use non-local models (`local_model=False`).
- `DIFFBOT_TOKEN` must be set to use web-search evidence gathering with a local model. If it is unset, `DISARM_LLM` prints a warning and runs with evidence gathering disabled.
- Local models expect a vLLM OpenAI-compatible server running at `http://localhost:8000`.

## Architecture

### `scripts/DISARM_LLM_Interface.py` — core classification engine
`DISARM_LLM` is the main class. It loads the DISARM framework definition from `.data/DISARM.json` (a STIX-like bundle: tactics are `x-mitre-tactic` objects, techniques are `attack-pattern` objects with MITRE-style `external_id`s such as `T0001.001`, where a `.` in the id denotes a sub-technique).

Two classification **modes** (`Mode` enum), each with its own system prompt pair (tactic-level and technique-level):
- `INVESTIGATE` — strict attribution rules: only classify a tactic/technique if the *author/publisher themselves* performed it, not if they merely reported/quoted another actor using it. Prefers false negatives.
- `RECOGNISE` — looser: tags any tactic/technique referenced in the article, regardless of who performed it.

Three classification **architectures**, all built from the same `identify_tactics()` / `identify_techniques()` primitives:
- `batch_clf(fast=False)` — hierarchical: identify tactics for the article, then run a separate technique-classification prompt per tactic (`identify_techniques_for_tactic`). `fast=True` uses the LLM to pick candidate tactics first instead of testing all of them.
- `select_all_clf()` — single prompt classifying against every technique at once; prompt is ordered `Techniques` before `Article` to take advantage of prefix caching across repeated calls with different articles.
- `single_clf()` — issues one prompt per individual technique across the whole framework (most expensive, most isolated).

Both local (vLLM `chat.completions`) and OpenAI (`responses.create`) code paths are handled inside `prompt_llm_response`/`vllm_response`/`chatGPT_response`; response schemas are enforced via `response_format` JSON schema (shape differs slightly between the vLLM and OpenAI APIs — see `identify_tactics`/`identify_techniques`). Responses are validated in `valid_classifications`, which also auto-corrects a model returning an invalid sub-technique by falling back to its parent technique id when `check_sub_techniques=False`.

Technique/tactic external IDs are tokenized (`.` → `d`) before being sent to the model as JSON schema `enum` values, and detokenized in the results, to avoid the id being split across multiple tokens.

For `INVESTIGATE` mode, `identify_techniques` looks up each technique's `external_id` in `.data/llm_additional_requirements.xlsx` (via `get_additional_llm_requirements` in `DISARM_DATA_MASTER.py`) to decide whether it needs web search (i.e. techniques marked `Internet/OSINT access`). Those techniques are passed to `gather_evidence`, an agentic research step that runs *before* classification — the classifier itself never gets live tool access, it just receives the gathered findings appended to each technique's description.

`gather_evidence` has two backends:
- **Hosted** (`gather_evidence_hosted`) — one `responses.create` call with OpenAI's built-in `web_search` tool.
- **Local** (`gather_evidence_local`) — the vLLM server has no tool-call parser enabled, so native `tools` come back as unparsed text in `message.content`. Instead the search loop is driven by structured output: each round the model returns `{"action": "search"|"answer", "queries": [...], "evidence": [...]}`, and the script runs any requested queries against **Diffbot's web search API** (`diffbot_web_search`, `https://llm.diffbot.com/api/v1/web_search` — *not* the Knowledge Graph search API) and feeds the results back as a user message. Capped at `max_search_rounds` (default 4) with up to 5 queries per round; on the final round `action` is narrowed to `["answer"]` to force a report. Already-run queries are deduplicated, and a failed search is reported back to the model rather than raised, so it can answer from what it already has.

Constructor flags: `web_search` (default `True`), `max_search_rounds`, `results_per_query`.

### `scripts/DISARM_DATA_MASTER.py` — DISARM framework/data helpers
Free functions for working with DISARM/MITRE-style JSON objects: `get_mitre_external_id`, `is_sub_tech`, `get_parent_extended_desc`, `get_additional_llm_requirements` (all read from `.data/llm_additional_requirements.xlsx`, keyed by external id in column A).

`DISARMDataMaster` (ported from the [upstream DISARM frameworks repo](https://github.com/DISARMFoundation/DISARMframeworks)) loads the incidents database from a set of `.xlsx` workbooks (`DISARM_FRAMEWORKS_MASTER.xlsx`, `DISARM_DATA_MASTER.xlsx`, `DISARM_COMMENTS_MASTER.xlsx` — not currently present in `.data/`) and builds pandas cross-tables linking incidents, techniques, counters, detections, actors, etc. This is separate from the DISARM.json framework definition used by the LLM interface, and is currently only used for incident-level lookups (`get_incident_ids`, `get_incident_urls`, `get_incident_techniques*`).

### `.data/` — datasets and framework definitions (gitignored except a few files)
- `DISARM.json` — the DISARM framework definition (tactics/techniques) consumed directly by `DISARM_LLM`.
- `llm_additional_requirements.xlsx` — per-technique metadata: extended parent-technique descriptions and additional tool requirements (e.g. "Internet/OSINT access").
- `euvsdisinfo.csv` — large (~130MB) source dataset of disinformation articles, presumably the input corpus for labeling/experiments.
- `DISARM_DATA_MASTER.xlsx` — incidents database workbook used by `DISARMDataMaster`.

### `scripts/agent_events.py` — observer plumbing
`EventBus` fans structured run events out to subscribers (each gets its own queue, and history is replayed on subscribe so a reconnecting client sees the whole run). `Cancelled` is raised inside a run when an observer asks it to stop.

`DISARM_LLM` takes optional `on_event(event_type, **payload)` and `should_stop()` callbacks. Both default to no-ops, so the CLI paths are unchanged. The engine emits `task`, `plan`, `tactic_started`/`tactic_finished`, `techniques_identified`, `sub_techniques_identified`, `evidence_started`/`evidence_reported`, `search_round`/`search_results`/`search_failed`, `llm_call_started`/`llm_call_finished` and `classification_complete`; `should_stop()` is polled at loop boundaries (per tactic, per search round, per technique). `Log.add_sink(fn)` additionally forwards every log line to an observer.

### `scripts/gui/` — the agent console
- `app.py` — Flask backend. One run at a time via `RunManager`; progress streams to the browser over SSE at `/api/events` (history replayed on connect, so the page can be opened or reloaded mid-run). `POST /api/run/articles` classifies pasted/uploaded articles, `POST /api/run/dataset` drives `label_dataset.run_labelling`, `POST /api/stop` cancels. It `chdir`s to the repo root on import, because the engine reads `.data/` by relative path, and reads `.env` for `OPENAI_API_KEY`/`DIFFBOT_TOKEN`.
- `catalog.py` — parses `.data/DISARM.json` into tactic/technique lookup tables so the GUI can show the name and description behind each bare external id.
- `static/` — vanilla HTML/CSS/JS single page, no build step. The activity feed nests events into `<details>` groups (run root > article > tactic, plus a subgroup for the taxonomy sub-technique stage) built in `openArticleGroup`/`openTacticGroup`/`openSubTechniqueGroup`; `feed()` appends to the innermost open group. `techniqueCard()` renders one technique with its DISARM name, description and gathered evidence, and is shared by the Techniques tab and the expandable Articles rows (which scope it to one article via each evidence item's `articleKey`).

### `scripts/collect/label_dataset.py`
`run_labelling(...)` labels articles from the balanced subset and appends each result to a JSONL cache, skipping anything already cached — so a run can be stopped and resumed. It takes the same classification config as `DISARM_LLM` plus `architecture` (see the `ARCHITECTURES` dict), `limit`, `cache_path`, and the `on_event`/`should_stop` callbacks. `main()` keeps the original CLI behaviour. `export_splits()` writes the labelled CSV (named after its cache file) and the stratified train/dev split into `.data/experiments/`, skipping the split rather than raising when the cache is too partial to stratify.

### `scripts/experiments/ZeDPEB.py`
Currently an empty stub. An experiment/benchmark (logs go to `scripts/experiments/.ZeDPEB_logs/`, gitignored except a `saved/` subfolder) intended to exercise the classification architectures above against the dataset. No established interface yet — check with the user before assuming its intended shape.
