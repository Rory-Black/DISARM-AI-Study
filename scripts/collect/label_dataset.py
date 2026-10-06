import os
import sys
import json
import time
import contextlib
from datetime import datetime, timezone

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display when driven from the GUI or a headless box
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DISARM_LLM_Interface import DISARM_LLM, Mode
from agent_events import Cancelled, truncate


DATASET_PATH = os.path.join(".data/euvsdisinfo.csv")
SUBSET_PATH = ".data/euvsdisinfo_bal_set.csv"
CACHE_PATH = os.path.join(".data/euvsdisinfo_cache.json")
EXPERIMENTS_DIR = ".data/experiments"

# the classification architectures a caller may pick between; each returns (tactics, techniques)
ARCHITECTURES = {
    "batch_clf": lambda llm: llm.batch_clf(),
    "batch_clf_fast": lambda llm: llm.batch_clf(fast=True),
    "taxonomy": lambda llm: llm.taxonomy(),
    "select_all_clf": lambda llm: (None, llm.select_all_clf()),
    "single_clf": lambda llm: (None, llm.single_clf()),
}


def ballance_dataset(df, required_balance, min_class_samples):
    # for each language, ballance the number of articles that are trustworthy and disinfo by a balance threshold
    languages = df["language"].unique()
    df_list = []
    for language in languages:
        language_df = df[df["language"] == language]
        disinfo_df = language_df[language_df["class"] == 1]
        trustworthy_df = language_df[language_df["class"] == 0]

        disinfo_sample_size = len(disinfo_df)
        trustworthy_sample_size = len(trustworthy_df)

        # balancee to match the balance threshold
        if len(disinfo_df) / len(language_df) > required_balance: 
            disinfo_sample_size = int(len(trustworthy_df) * required_balance / (1 - required_balance))
        elif len(trustworthy_df) / len(language_df) > required_balance:
            trustworthy_sample_size = int(len(disinfo_df) * required_balance / (1 - required_balance))

        # if sample size is less than minimum samples, remove the language from the dataset
        if min(disinfo_sample_size, trustworthy_sample_size) < min_class_samples:
            continue

        disinfo_df = disinfo_df.sample(n=disinfo_sample_size, random_state=42)
        trustworthy_df = trustworthy_df.sample(n=trustworthy_sample_size, random_state=42)

        df_list.append(pd.concat([disinfo_df, trustworthy_df]))
    df = pd.concat(df_list).reset_index(drop=True)
    return df
    

def create_euvsdisinfo_bal_set():
    """Selects a subset of the euvsdisinfo database that is ballanced between disinfo and trustworthy
    articles between each language and saves it to .data/euvsdisinfo_bal_set"""

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df = df.fillna("")
        df["text"] = df.apply(
            lambda x: x["article_title"] + " " + x["article_text"] if x["article_title"] != "" else x["article_text"],
            axis=1,
        )  # combine the title and text into a single column
        df["language"] = df["article_language"]
        df["stratify"] = df["class"] + df["language"]
        df["class"] = df["class"].apply(lambda x: 1 if x == "disinformation" else 0)
        df["label"] = df["class"]
        df["disarm_techniques"] = None

        df = ballance_dataset(df, required_balance=0.6, min_class_samples=2)

        # save the test set to .data/euvsdisinfo_bal_set
        df.to_csv(SUBSET_PATH, index=False)

        # draw a bar chart of the number of articles per language and class with plt
        df.groupby(["language", "class"]).size().unstack(fill_value=0).plot(kind="bar", figsize=(15, 6))
        plt.title("Distribution of Articles by Language and Class")
        plt.xlabel("Language")
        plt.ylabel("Number of Articles")
        plt.legend(["Trustworthy", "Disinformation"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(".data/euvsdisinfo_bal_set_distribution.png")

        # print the number of articles per language and class
        print(df.groupby(["language", "class"]).size().unstack(fill_value=0))
        print(f"Total number of articles: {len(df)}")
    else:
        raise Exception(f"{DATASET_PATH} does not exist. Please download the dataset first.")


def load_cache(path=CACHE_PATH):
    cache = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                try:
                    cache.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    print("Wrong formatting at line", i + 1)

    else:
        print("Cache file does not exist. Creating a new one.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            pass

    return cache


class _ArticleRecorder:
    """Captures the evidence and searches behind one article's labels.

    The classifier reports these as events rather than returning them, so without
    this they would only ever exist in a log. Recording them makes the cache
    self-describing: each row says not just which techniques were identified but what
    the agent looked up to decide, and which sources it saw.
    """

    def __init__(self, downstream=None):
        self.downstream = downstream
        self.evidence = []
        self.queries = []

    def __call__(self, event_type, **payload):
        if event_type == "evidence_reported":
            for item in payload.get("evidence") or []:
                self.evidence.append(
                    {
                        "external_id": item.get("external_id"),
                        "findings": item.get("findings", ""),
                        "sources": item.get("sources") or [],
                    }
                )
        elif event_type == "search_round":
            self.queries.extend(payload.get("queries") or [])

        if self.downstream is not None:
            self.downstream(event_type, **payload)


def row_language(row):
    """The article's language. `article_language` is the dataset's own column; the
    balanced subset copies it to `language`, so fall back to that."""
    for field in ("article_language", "language"):
        value = row.get(field) if hasattr(row, "get") else getattr(row, field, None)
        if value not in (None, ""):
            return value
    return None


def subset_languages(subset_path=SUBSET_PATH):
    """Article counts per language in the balanced subset, largest first."""
    if not os.path.exists(subset_path):
        return {}
    column = "article_language"
    counts = pd.read_csv(subset_path, usecols=[column])[column].value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def _json_safe(value):
    """pandas/numpy scalars aren't JSON serialisable on their own - fall back to str."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return str(value)


def export_splits(cache_path=CACHE_PATH, experiments_dir=EXPERIMENTS_DIR):
    """Turns the labelled cache into the labelled CSV plus the train/dev split.

    Returns the paths written. The stratified split needs at least two articles per
    class+language group, so it is skipped (rather than raising) on a partial cache.
    """
    df = pd.DataFrame(load_cache(cache_path))
    if df.empty:
        return []

    # name the labelled CSV after the cache it came from, so a run against an alternate
    # cache never overwrites the main .data/euvsdisinfo_labelled.csv
    stem = os.path.splitext(os.path.basename(cache_path))[0]
    if stem.endswith("_cache"):
        stem = stem[: -len("_cache")]
    written = []
    labelled_path = os.path.join(os.path.dirname(cache_path) or ".data", f"{stem}_labelled.csv")
    df.to_csv(labelled_path, index=False)
    written.append(labelled_path)

    # SPLITTING INTO TRAIN AND TEST DATA
    os.makedirs(experiments_dir, exist_ok=True)
    try:
        train_df, dev_df = train_test_split(df, test_size=0.1, stratify=df["stratify"], random_state=42)
    except (ValueError, KeyError) as e:
        logger.warning(f"Skipping train/dev split: {e}")
        return written

    columns = ["text", "disarm_techniques", "label", "language", "keywords", "debunk_date"]
    columns = [c for c in columns if c in train_df.columns]

    train_path = os.path.join(experiments_dir, "euvsdisinfo.csv")
    dev_path = os.path.join(experiments_dir, "euvsdisinfo_dev.csv")
    train_df[columns].to_csv(train_path, index=False)
    dev_df[columns].to_csv(dev_path, index=False)
    written += [train_path, dev_path]
    return written


def run_labelling(
    model_name="google/gemma-4-26B-A4B-it",
    local_model=True,
    mode=Mode.INVESTIGATE,
    architecture="batch_clf",
    check_sub_techniques=False,
    web_search=True,
    max_search_rounds=4,
    results_per_query=5,
    limit=None,
    language=None,
    subset_path=SUBSET_PATH,
    cache_path=CACHE_PATH,
    export=True,
    quiet=True,
    on_event=None,
    should_stop=None,
):
    """Labels articles from the balanced subset with DISARM techniques.

    Every article already present in the cache is skipped, so a run can be stopped and
    resumed. `language` restricts the run to one of the dataset's article languages
    (matched against `article_language`); `limit` then caps that filtered queue.
    `on_event(event_type, **payload)` receives per-article progress on top of the
    per-step events the classifier itself emits, and `should_stop()` is polled between
    articles so a caller can cancel cleanly without losing completed work.
    """

    def emit(event_type, **payload):
        if on_event is not None:
            on_event(event_type, **payload)

    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from {sorted(ARCHITECTURES)}")

    if not os.path.exists(subset_path):
        emit("dataset_preparing", subset_path=subset_path)
        create_euvsdisinfo_bal_set()

    df = pd.read_csv(subset_path)
    cache = load_cache(cache_path)
    done_ids = {c.get("article_id") for c in cache}

    pending = [(i, row) for i, row in df.iterrows() if row.article_id not in done_ids]
    if language:
        pending = [(i, row) for i, row in pending if row_language(row) == language]
    # the limit caps whatever is left after the language filter, not before it
    if limit:
        pending = pending[: int(limit)]

    emit(
        "dataset_started",
        total=len(df),
        already_labelled=len(done_ids),
        queued=len(pending),
        subset_path=subset_path,
        cache_path=cache_path,
        architecture=architecture,
        language=language,
    )

    skipped = len(done_ids)
    failures = 0
    completed = 0

    progress = tqdm(pending, total=len(pending)) if not quiet else pending
    for position, (i, row) in enumerate(progress, start=1):
        if should_stop is not None and should_stop():
            emit("dataset_stopped", completed=completed, failed=failures)
            break

        emit(
            "article_started",
            index=position,
            total=len(pending),
            article_id=row.article_id,
            publisher=getattr(row, "article_publisher", None),
            url=getattr(row, "article_url", None),
            language=getattr(row, "language", None),
            label=int(row["class"]) if "class" in row else None,
            chars=len(str(row.text)),
            preview=truncate(row.text, 500),
        )

        recorder = _ArticleRecorder(on_event)
        started = time.time()
        llm = DISARM_LLM(
            article_content=row.text,
            model_name=model_name,
            local_model=local_model,
            mode=mode,
            check_sub_techniques=check_sub_techniques,
            debug_log=False,
            web_search=web_search,
            max_search_rounds=max_search_rounds,
            results_per_query=results_per_query,
            on_event=recorder,
            should_stop=should_stop,
        )

        # the classifier prints a fair amount to stdout; suppress it for the CLI run
        # but leave it alone when a caller wants to capture it
        stdout_guard = open(os.devnull, "w") if quiet else None
        try:
            with contextlib.redirect_stdout(stdout_guard) if quiet else contextlib.nullcontext():
                tactics, techniques = ARCHITECTURES[architecture](llm)
        except Cancelled:
            emit("dataset_stopped", completed=completed, failed=failures)
            break
        except Exception as e:
            logger.error(f"Failed to label {row.article_id}: {e}")
            failures += 1
            emit("article_failed", index=position, article_id=row.article_id, error=str(e))
            continue
        finally:
            if stdout_guard is not None:
                stdout_guard.close()

        logger.debug(f"Successfully labelled {row.article_id} with techniques - {techniques}")

        data = {k: v for k, v in dict(row).items()}
        data["disarm_techniques"] = techniques
        data["disarm_tactics"] = list(tactics) if tactics else []
        data["disarm_evidence"] = recorder.evidence
        data["disarm_meta"] = {
            "model": model_name,
            "mode": mode.name,
            "architecture": architecture,
            "check_sub_techniques": check_sub_techniques,
            "web_search": web_search,
            "search_queries": recorder.queries,
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started, 2),
        }
        with open(cache_path, "a") as f:
            f.write(json.dumps(data, default=_json_safe) + "\n")

        completed += 1
        emit("article_finished", index=position, article_id=row.article_id, techniques=techniques)

    exports = []
    if export and completed:
        exports = export_splits(cache_path)

    emit("dataset_finished", labelled=completed, skipped=skipped, failed=failures, exports=exports)
    return {"labelled": completed, "skipped": skipped, "failed": failures, "exports": exports}


def main():
    """Uses the LLM interface to label the articles with DISARM techniques"""
    run_labelling(quiet=True)


if __name__ == "__main__":
    main()
