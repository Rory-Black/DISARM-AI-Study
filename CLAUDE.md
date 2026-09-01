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

This executes `test_interface()` at the bottom of the file, which is currently the only way to exercise `DISARM_LLM` end-to-end. Edit the `DISARM_LLM(...)` construction there to switch between a local model and GPT.

Environment/external dependencies:
- `OPENAI_API_KEY` must be set in the environment to use non-local models (`local_model=False`).
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

For `INVESTIGATE` mode, `identify_techniques` looks up each technique's `external_id` in `.data/llm_additional_requirements.xlsx` (via `get_additional_llm_requirements` in `DISARM_DATA_MASTER.py`) to decide whether the model needs the `web_search` tool (e.g. for techniques requiring OSINT/internet access) — this is only wired up for the non-local (OpenAI) path.

### `scripts/DISARM_DATA_MASTER.py` — DISARM framework/data helpers
Free functions for working with DISARM/MITRE-style JSON objects: `get_mitre_external_id`, `is_sub_tech`, `get_parent_extended_desc`, `get_additional_llm_requirements` (all read from `.data/llm_additional_requirements.xlsx`, keyed by external id in column A).

`DISARMDataMaster` (ported from the [upstream DISARM frameworks repo](https://github.com/DISARMFoundation/DISARMframeworks)) loads the incidents database from a set of `.xlsx` workbooks (`DISARM_FRAMEWORKS_MASTER.xlsx`, `DISARM_DATA_MASTER.xlsx`, `DISARM_COMMENTS_MASTER.xlsx` — not currently present in `.data/`) and builds pandas cross-tables linking incidents, techniques, counters, detections, actors, etc. This is separate from the DISARM.json framework definition used by the LLM interface, and is currently only used for incident-level lookups (`get_incident_ids`, `get_incident_urls`, `get_incident_techniques*`).

### `.data/` — datasets and framework definitions (gitignored except a few files)
- `DISARM.json` — the DISARM framework definition (tactics/techniques) consumed directly by `DISARM_LLM`.
- `llm_additional_requirements.xlsx` — per-technique metadata: extended parent-technique descriptions and additional tool requirements (e.g. "Internet/OSINT access").
- `euvsdisinfo.csv` — large (~130MB) source dataset of disinformation articles, presumably the input corpus for labeling/experiments.
- `DISARM_DATA_MASTER.xlsx` — incidents database workbook used by `DISARMDataMaster`.

### `scripts/collect/label_dataset.py` and `scripts/experiments/ZeDPEB.py`
Both currently empty stub files. `ZeDPEB.py` is an experiment/benchmark (logs go to `scripts/experiments/.ZeDPEB_logs/`, gitignored except a `saved/` subfolder) intended to exercise the classification architectures above against the dataset; `label_dataset.py` is presumably for building/labeling the training or evaluation dataset from `.data/euvsdisinfo.csv`. Neither has an established interface yet — check with the user before assuming their intended shape.
