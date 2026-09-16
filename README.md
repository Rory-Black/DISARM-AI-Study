# DISARM-AI-Study

Research code that uses LLMs to classify disinformation articles against the
[DISARM framework](https://www.disarm.foundation/).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Environment (a `.env` file at the repository root is read by the GUI):

- `OPENAI_API_KEY` — required for hosted models (`local_model=False`).
- `DIFFBOT_TOKEN` — required for web-search evidence gathering with a local model.
- Local models expect a vLLM OpenAI-compatible server on `http://localhost:8000`.

## The agent console (GUI)

```bash
.venv/bin/python scripts/gui/app.py     # then open http://localhost:5000
```

A web console for running and watching the classifier. It exists to make the agent's
work legible: what it is doing right now, what it searched for, what evidence it came
back with, and which techniques it settled on.

**Configure** the classification mode (`INVESTIGATE` / `RECOGNISE`), the architecture
(`batch_clf`, `batch_clf_fast`, `taxonomy`, `select_all_clf`, `single_clf`), the model,
local vs hosted, sub-technique checking, and the web-search budget.

**Give it articles** three ways: paste text, drop in `.txt`/`.md`/`.json` files (each
file is one article; a JSON array is a list of articles), or run the dataset labeller
over the balanced euvsdisinfo subset.

**Watch it work** — the header shows the task in progress and progress through the
tactics; the tabs break the run down into:

| Tab | Shows |
| --- | --- |
| Activity | Chronological feed of every step, grouped by article and — under `batch_clf`/`taxonomy` — subgrouped by tactic. Each group collapses to a one-line summary when it finishes, so a 16-tactic run stays scannable |
| Techniques | Each identified technique with its DISARM name and description, its tactic, and any evidence gathered for it |
| Evidence | The research agent's findings per technique, with the source URLs it actually saw |
| Searches | Every web search round: the queries issued and the results returned |
| Articles | Per-article status, timings and results; expand a row for the same technique view as above, scoped to that one article |
| Framework | Searchable browser of all 16 tactics and 391 techniques from `.data/DISARM.json` |
| Logs | The raw engine log, as the terminal would show it |

The page can be opened or reloaded mid-run — the server replays the run so far. **Stop
run** cancels between steps; for dataset labelling, everything already written to the
cache is kept and the next run resumes from there.

Host and port come from `DISARM_GUI_HOST` / `DISARM_GUI_PORT`.

## Command line

Classify one hardcoded sample article:

```bash
python scripts/DISARM_LLM_Interface.py
```

Label the dataset (same work the GUI's Dataset tab does):

```bash
python scripts/collect/label_dataset.py
```
