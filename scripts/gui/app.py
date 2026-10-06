"""Web GUI for the DISARM classification agent.

Run it from the repository root:

    .venv/bin/python scripts/gui/app.py

then open http://localhost:5000. The page shows what the agent is doing while it
works: the task it is on, the web searches it runs, the evidence it gathers, and the
techniques it settles on - with each technique's DISARM name and description resolved
from .data/DISARM.json.
"""

import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "gui"))


def load_dotenv(path=REPO_ROOT / ".env"):
    """Minimal .env loader so OPENAI_API_KEY / DIFFBOT_TOKEN can live in the repo file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

# the engine reads paths relative to the repository root, so anchor the process there
os.chdir(REPO_ROOT)

from flask import Flask, Response, jsonify, request, send_from_directory

from agent_events import Cancelled, EventBus, truncate
from DISARM_LLM_Interface import DISARM_LLM, Log, Mode
from catalog import load_framework
import cache_store

sys.path.insert(0, str(SCRIPTS_DIR / "collect"))
import label_dataset

ARCHITECTURES = label_dataset.ARCHITECTURES

DEFAULT_CONFIG = {
    "model_name": "google/gemma-4-26B-A4B-it",
    "local_model": True,
    "mode": "INVESTIGATE",
    "architecture": "batch_clf",
    "check_sub_techniques": False,
    "web_search": True,
    "max_search_rounds": 4,
    "results_per_query": 5,
}


def parse_config(payload):
    config = dict(DEFAULT_CONFIG)
    for key in config:
        if key in payload and payload[key] is not None:
            config[key] = payload[key]
    config["local_model"] = bool(config["local_model"])
    config["check_sub_techniques"] = bool(config["check_sub_techniques"])
    config["web_search"] = bool(config["web_search"])
    config["max_search_rounds"] = int(config["max_search_rounds"])
    config["results_per_query"] = int(config["results_per_query"])
    if config["architecture"] not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {config['architecture']}")
    if config["mode"] not in Mode.__members__:
        raise ValueError(f"Unknown mode: {config['mode']}")
    if not str(config["model_name"]).strip():
        raise ValueError("A model name is required")
    return config


class RunManager:
    """Owns the single in-flight run and broadcasts its progress.

    Only one run at a time: the agent is talking to one LLM server and appending to one
    cache, and the whole point of the GUI is to watch one run closely.
    """

    def __init__(self):
        self.bus = EventBus()
        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.state = "idle"
        self.current = None
        self.started_at = None
        self.finished_at = None

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def status(self):
        return {
            "state": self.state,
            "running": self.is_running(),
            "run": self.current,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "env": {
                "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
                "diffbot_token": bool(os.environ.get("DIFFBOT_TOKEN")),
            },
        }

    def start(self, kind, config, target, **meta):
        with self.lock:
            if self.is_running():
                raise RuntimeError("A run is already in progress - stop it before starting another")

            self.bus.clear()
            self.stop_event.clear()
            self.state = "running"
            self.started_at = time.time()
            self.finished_at = None
            self.current = {"kind": kind, "config": config, **meta}

            self.bus.emit("run_started", kind=kind, config=config, **meta)

            self.thread = threading.Thread(target=self._run, args=(target,), daemon=True)
            self.thread.start()

    def _run(self, target):
        # route the engine's own log lines into the stream too, so the GUI's log pane
        # shows exactly what the terminal would have shown
        sink = lambda kind, message: self.bus.emit("log", level=kind, message=message)
        Log.add_sink(sink)
        try:
            target(self.emit, self.stop_event.is_set)
            self.state = "stopped" if self.stop_event.is_set() else "finished"
            self.bus.emit("run_finished", state=self.state)
        except Cancelled:
            self.state = "stopped"
            self.bus.emit("run_finished", state="stopped")
        except Exception as e:
            self.state = "failed"
            self.bus.emit("run_failed", error=str(e), traceback=traceback.format_exc())
            self.bus.emit("run_finished", state="failed")
        finally:
            Log.remove_sink(sink)
            self.finished_at = time.time()

    def emit(self, event_type, **payload):
        self.bus.emit(event_type, **payload)
        # checked here as well as in the engine's own loops so a stop lands promptly
        if self.stop_event.is_set():
            raise Cancelled("Run stopped by user")

    def stop(self):
        if self.is_running():
            self.state = "stopping"
            self.stop_event.set()
            self.bus.emit("run_stopping")
            return True
        return False


manager = RunManager()
framework = load_framework()

# static_url_path="" serves the page assets from the root, so index.html can reference
# them as ./style.css and ./app.js
app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "static"), static_url_path="")


# ---------------------------------------------------------------- run targets

def article_run(articles, config):
    """Classifies a list of {name, text} articles one after another."""

    def target(emit, should_stop):
        mode = Mode[config["mode"]]
        architecture = ARCHITECTURES[config["architecture"]]

        for index, article in enumerate(articles, start=1):
            if should_stop():
                emit("batch_stopped", completed=index - 1)
                return

            text = article.get("text", "")
            emit(
                "article_started",
                index=index,
                total=len(articles),
                article_id=article.get("name") or f"article-{index}",
                source="upload",
                chars=len(text),
                preview=truncate(text, 500),
            )

            llm = DISARM_LLM(
                article_content=text,
                model_name=config["model_name"],
                local_model=config["local_model"],
                mode=mode,
                check_sub_techniques=config["check_sub_techniques"],
                debug_log=False,
                web_search=config["web_search"],
                max_search_rounds=config["max_search_rounds"],
                results_per_query=config["results_per_query"],
                on_event=emit,
                should_stop=should_stop,
            )

            started = time.time()
            try:
                tactics, techniques = architecture(llm)
            except Cancelled:
                emit("batch_stopped", completed=index - 1)
                return
            except Exception as e:
                emit("article_failed", index=index, article_id=article.get("name"), error=str(e))
                continue

            emit(
                "article_finished",
                index=index,
                article_id=article.get("name") or f"article-{index}",
                techniques=techniques,
                tactics=list(tactics) if tactics else [],
                duration=round(time.time() - started, 2),
            )

    return target


def dataset_run(config, limit, language, export, cache_path, subset_path):
    """Runs the dataset labelling script over the balanced euvsdisinfo subset."""

    def target(emit, should_stop):
        label_dataset.run_labelling(
            model_name=config["model_name"],
            local_model=config["local_model"],
            mode=Mode[config["mode"]],
            architecture=config["architecture"],
            check_sub_techniques=config["check_sub_techniques"],
            web_search=config["web_search"],
            max_search_rounds=config["max_search_rounds"],
            results_per_query=config["results_per_query"],
            limit=limit,
            language=language,
            subset_path=subset_path,
            cache_path=cache_path,
            export=export,
            quiet=False,
            on_event=emit,
            should_stop=should_stop,
        )

    return target


# -------------------------------------------------------------------- routes

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/framework")
def api_framework():
    return jsonify(framework)


@app.get("/api/defaults")
def api_defaults():
    return jsonify(
        {
            "config": DEFAULT_CONFIG,
            "architectures": sorted(ARCHITECTURES),
            "modes": list(Mode.__members__),
            "paths": {
                "subset": label_dataset.SUBSET_PATH,
                "cache": label_dataset.CACHE_PATH,
            },
        }
    )


@app.get("/api/status")
def api_status():
    return jsonify(manager.status())


_subset_counts = {}


def subset_size(subset_path):
    """Rows in the balanced subset, counted once per file and remembered."""
    if not os.path.exists(subset_path):
        return None
    key = (subset_path, os.path.getmtime(subset_path))
    if key not in _subset_counts:
        try:
            import pandas as pd

            # count rows without loading a 50MB frame into memory
            _subset_counts[key] = int(
                sum(len(chunk) for chunk in pd.read_csv(subset_path, usecols=["article_id"], chunksize=20000))
            )
        except Exception:
            _subset_counts[key] = None
    return _subset_counts[key]


_subset_language_counts = {}


def subset_language_counts(subset_path):
    """Articles per language in the subset, counted once per file and remembered."""
    if not os.path.exists(subset_path):
        return {}
    key = (subset_path, os.path.getmtime(subset_path))
    if key not in _subset_language_counts:
        try:
            _subset_language_counts[key] = label_dataset.subset_languages(subset_path)
        except Exception:
            _subset_language_counts[key] = {}
    return _subset_language_counts[key]


@app.get("/api/dataset/summary")
def api_dataset_summary():
    """How far through the dataset the cache already is, so a run can be resumed knowingly.

    The per-language breakdown feeds the language filter: for each language the subset
    holds, how many are already labelled and how many a run would still have to do.
    """
    cache_path = request.args.get("cache_path", label_dataset.CACHE_PATH)
    subset_path = request.args.get("subset_path", label_dataset.SUBSET_PATH)
    entries = cache_store.load(cache_path)

    labelled_by_language = {}
    for entry in entries:
        name = entry.get("article_language") or entry.get("language")
        if name:
            labelled_by_language[name] = labelled_by_language.get(name, 0) + 1

    languages = [
        {
            "language": name,
            "total": total,
            "labelled": labelled_by_language.get(name, 0),
            "remaining": max(0, total - labelled_by_language.get(name, 0)),
        }
        for name, total in subset_language_counts(subset_path).items()
    ]

    return jsonify(
        {
            "cache_path": cache_path,
            "subset_path": subset_path,
            "subset_exists": os.path.exists(subset_path),
            "labelled": len(entries),
            "total": subset_size(subset_path),
            "languages": languages,
        }
    )


@app.get("/api/cache")
def api_cache():
    """The labelled cache: headline stats, facets and a page of matching rows.

    Article bodies are left out of the list - `/api/cache/entry` serves one on demand,
    so opening the tab never ships tens of MB of text to the browser.
    """
    cache_path = request.args.get("cache_path", label_dataset.CACHE_PATH)
    subset_path = request.args.get("subset_path", label_dataset.SUBSET_PATH)
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    sort = request.args.get("sort", "recent")

    entries = cache_store.load(cache_path)
    matching = [
        e
        for e in entries
        if cache_store.matches(
            e,
            query=request.args.get("q", "").strip(),
            technique=request.args.get("technique", "").strip(),
            language=request.args.get("language", "").strip(),
            publisher=request.args.get("publisher", "").strip(),
            label=request.args.get("label", "").strip(),
        )
    ]

    if sort == "techniques":
        matching = sorted(matching, key=lambda e: -len(e.get("disarm_techniques") or []))
    elif sort == "oldest":
        matching = list(matching)
    else:  # most recently labelled first - the cache is append-ordered
        matching = list(reversed(matching))

    start = (page - 1) * page_size
    page_entries = matching[start : start + page_size]

    return jsonify(
        {
            "cache_path": cache_path,
            "exists": os.path.exists(cache_path),
            "total": len(entries),
            "filtered": len(matching),
            "page": page,
            "page_size": page_size,
            "pages": max(1, -(-len(matching) // page_size)),
            "entries": [cache_store.light(e) for e in page_entries],
            "facets": cache_store.facets(entries),
            "stats": cache_store.stats(entries, subset_total=subset_size(subset_path)),
        }
    )


@app.get("/api/cache/entry")
def api_cache_entry():
    """One labelled row in full, including the article body."""
    cache_path = request.args.get("cache_path", label_dataset.CACHE_PATH)
    article_id = request.args.get("article_id", "")
    entry = cache_store.find(cache_store.load(cache_path), article_id)
    if entry is None:
        return jsonify({"error": f"No cached article with id {article_id}"}), 404

    full = dict(entry)
    full["text"] = cache_store.body(entry)
    full.pop("article_text", None)
    return jsonify(full)


@app.post("/api/run/articles")
def api_run_articles():
    payload = request.get_json(force=True) or {}
    articles = [a for a in payload.get("articles", []) if (a.get("text") or "").strip()]
    if not articles:
        return jsonify({"error": "No article text provided"}), 400

    try:
        config = parse_config(payload.get("config", {}))
        manager.start(
            "articles",
            config,
            article_run(articles, config),
            article_count=len(articles),
            article_names=[a.get("name") or f"article-{i}" for i, a in enumerate(articles, start=1)],
        )
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 409

    return jsonify(manager.status())


@app.post("/api/run/dataset")
def api_run_dataset():
    payload = request.get_json(force=True) or {}
    try:
        config = parse_config(payload.get("config", {}))
        limit = payload.get("limit")
        limit = int(limit) if limit else None
        language = (payload.get("language") or "").strip() or None
        cache_path = payload.get("cache_path") or label_dataset.CACHE_PATH
        subset_path = payload.get("subset_path") or label_dataset.SUBSET_PATH

        if language and language not in subset_language_counts(subset_path):
            raise ValueError(f"No articles in the subset have language '{language}'")

        manager.start(
            "dataset",
            config,
            dataset_run(config, limit, language, bool(payload.get("export", True)), cache_path, subset_path),
            limit=limit,
            language=language,
            cache_path=cache_path,
            subset_path=subset_path,
        )
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 409

    return jsonify(manager.status())


@app.post("/api/stop")
def api_stop():
    return jsonify({"stopping": manager.stop()})


@app.get("/api/events")
def api_events():
    """Server-sent events stream of everything the agent is doing.

    A newly connected browser is replayed the current run's history first, so opening
    or reloading the page mid-run shows the whole run rather than just what follows.
    """
    q = manager.bus.subscribe(replay=True)

    def stream():
        try:
            yield f"event: status\ndata: {json.dumps(manager.status())}\n\n"
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"  # keeps proxies from dropping an idle stream
                    continue
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            manager.bus.unsubscribe(q)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    host = os.environ.get("DISARM_GUI_HOST", "127.0.0.1")
    port = int(os.environ.get("DISARM_GUI_PORT", "5000"))
    print(f"DISARM agent GUI -> http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)
