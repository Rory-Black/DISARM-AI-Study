"""Reads the labelled-dataset cache (JSONL) for the GUI's Database tab.

Each line of `.data/euvsdisinfo_cache.json` is one labelled article: the original
euvsdisinfo row plus what the agent concluded (`disarm_techniques`, and, for rows
written after evidence persistence was added, `disarm_evidence` and `disarm_meta`).

The file is parsed once and memoised against its path/mtime/size, because it grows
to tens of MB of article text and the Database tab hits it on every filter change.
"""

import json
import os
import threading
from collections import Counter

# article bodies are large and only needed when a row is opened, so the list view
# gets everything except these
HEAVY_FIELDS = ("article_text", "text")

_lock = threading.Lock()
_memo = {}  # path -> (signature, entries)


def _signature(path):
    stat = os.stat(path)
    return (stat.st_mtime_ns, stat.st_size)


def load(path):
    """Returns the cache as a list of dicts, reparsing only when the file changes."""
    if not os.path.exists(path):
        return []

    signature = _signature(path)
    with _lock:
        cached = _memo.get(path)
        if cached and cached[0] == signature:
            return cached[1]

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # a partially written final line is normal while a run is appending
                continue
            entry.setdefault("_line", line_no)
            entries.append(entry)

    with _lock:
        _memo[path] = (signature, entries)
    return entries


def body(entry):
    return entry.get("text") or entry.get("article_text") or ""


def light(entry, preview_chars=320):
    """The list-view projection: everything but the article body, plus a preview."""
    out = {k: v for k, v in entry.items() if k not in HEAVY_FIELDS}
    text = body(entry)
    out["chars"] = len(text)
    out["preview"] = text[:preview_chars].rstrip() + ("..." if len(text) > preview_chars else "")
    out["technique_count"] = len(entry.get("disarm_techniques") or [])
    out["evidence_count"] = len(entry.get("disarm_evidence") or [])
    return out


def matches(entry, query="", technique="", language="", publisher="", label=""):
    if technique and technique not in (entry.get("disarm_techniques") or []):
        return False
    if language and entry.get("language") != language:
        return False
    if publisher and entry.get("article_publisher") != publisher:
        return False
    if label != "" and label is not None and str(entry.get("label", entry.get("class"))) != str(label):
        return False
    if query:
        needle = query.lower()
        haystack = " ".join(
            str(entry.get(field) or "")
            for field in ("article_title", "keywords", "article_publisher", "article_domain", "article_url", "article_id")
        )
        # the article body is searched too - it is the reason most people search at all
        if needle not in haystack.lower() and needle not in body(entry).lower():
            return False
    return True


def facets(entries):
    """Distinct filterable values with their counts, most common first."""

    def ranked(counter):
        return [{"value": value, "count": count} for value, count in counter.most_common() if value not in (None, "")]

    return {
        "languages": ranked(Counter(e.get("language") for e in entries)),
        "publishers": ranked(Counter(e.get("article_publisher") for e in entries)),
        "labels": ranked(Counter(str(e.get("label", e.get("class"))) for e in entries)),
    }


def stats(entries, subset_total=None):
    """Headline numbers plus the technique-frequency distribution the chart plots."""
    technique_counts = Counter(t for e in entries for t in (e.get("disarm_techniques") or []))
    tagged = sum(1 for e in entries if e.get("disarm_techniques"))
    with_evidence = sum(1 for e in entries if e.get("disarm_evidence"))
    total_tags = sum(technique_counts.values())
    disinfo = sum(1 for e in entries if str(e.get("label", e.get("class"))) == "1")

    return {
        "labelled": len(entries),
        "subset_total": subset_total,
        "coverage": round(100 * len(entries) / subset_total, 1) if subset_total else None,
        "distinct_techniques": len(technique_counts),
        "total_tags": total_tags,
        "avg_techniques": round(total_tags / len(entries), 2) if entries else 0,
        "articles_with_techniques": tagged,
        "articles_without_techniques": len(entries) - tagged,
        "articles_with_evidence": with_evidence,
        "disinformation": disinfo,
        "trustworthy": len(entries) - disinfo,
        "technique_counts": [
            {"external_id": external_id, "count": count} for external_id, count in technique_counts.most_common()
        ],
        "languages": len({e.get("language") for e in entries if e.get("language")}),
        "publishers": len({e.get("article_publisher") for e in entries if e.get("article_publisher")}),
    }


def find(entries, article_id):
    for entry in entries:
        if entry.get("article_id") == article_id:
            return entry
    return None
