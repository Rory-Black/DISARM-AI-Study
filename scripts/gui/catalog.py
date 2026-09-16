"""Reads DISARM.json into lookup tables the GUI can serve to the browser.

The classifier only ever returns bare external ids (T0004, T0084.001, ...). The GUI
needs the human-readable name and description behind each of those ids, plus the
tactic each technique sits under, so results read as findings rather than as codes.
"""

import json
import os
from pathlib import Path

JSON_DATA = Path(__file__).resolve().parent.parent.parent / ".data" / "DISARM.json"


def _external_id(obj, source_names=("mitre-attack", "DISARM")):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") in source_names:
            return ref.get("external_id"), ref.get("url")
    return None, None


def load_framework(path=JSON_DATA):
    """Returns {"tactics": [...], "techniques": {external_id: {...}}, "meta": {...}}."""
    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    tactics = {}
    tactic_order = []
    for obj in bundle["objects"]:
        if obj.get("type") != "x-mitre-tactic":
            continue
        ext_id, url = _external_id(obj)
        shortname = obj.get("x_mitre_shortname")
        tactics[shortname] = {
            "external_id": ext_id,
            "shortname": shortname,
            "name": obj.get("name"),
            "description": obj.get("description"),
            "url": url,
            "technique_ids": [],
        }
        tactic_order.append(shortname)

    techniques = {}
    for obj in bundle["objects"]:
        if obj.get("type") != "attack-pattern":
            continue
        ext_id, url = _external_id(obj)
        if not ext_id:
            continue
        phases = [p.get("phase_name") for p in obj.get("kill_chain_phases", [])]
        is_sub = "." in ext_id
        techniques[ext_id] = {
            "external_id": ext_id,
            "name": obj.get("name"),
            "description": obj.get("description") or "",
            "url": url,
            "is_sub_technique": is_sub,
            "parent_id": ext_id.split(".")[0] if is_sub else None,
            "tactics": phases,
        }
        for phase in phases:
            if phase in tactics:
                tactics[phase]["technique_ids"].append(ext_id)

    # tactics keep the order they appear in the bundle, which is the DISARM kill-chain order
    ordered_tactics = [tactics[name] for name in tactic_order]

    return {
        "tactics": ordered_tactics,
        "techniques": techniques,
        "meta": {
            "tactic_count": len(ordered_tactics),
            "technique_count": sum(1 for t in techniques.values() if not t["is_sub_technique"]),
            "sub_technique_count": sum(1 for t in techniques.values() if t["is_sub_technique"]),
            "source": os.path.basename(str(path)),
        },
    }
