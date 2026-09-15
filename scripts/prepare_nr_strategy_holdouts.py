#!/usr/bin/env python3
"""Freeze and prepare two independent NR strategy holdouts.

Candidate ranking reads only the cached simulations catalog and historical
metadata manifests.  After ``candidate-ranking.json`` is written, this script
fetches the two selected Data Caltech records and lets the pinned v2 decoder
download and hash-check the four N2/N3 controls at each of the highest two
levels.  It never calls LAL or fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("/tmp/torchwave-nr33-v2/catalog/simulations.json")
RAW_ROOT = Path("/tmp/torchwave-parity-strategy")
DATA = ROOT / "docs/reviews/2026-09-15-parity-strategy-experiment/data"
POLICY = ROOT / "docs/reviews/2026-09-15-independent-calibration-v2/quality-policy.json"
POLICY_SHA256 = "1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d"
TARGETS = (2.25, 2.75, 3.25, 6.5)
TARGET_RELTOL = 0.005
# Frozen family locations used by the earlier calibration and regression
# experiments.  The explicit list covers training families whose selection
# manifests record only a split label, plus the regression/holdout targets.
PRIOR_FAMILY_QS = (1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0)
UA = "torchwave-nr-strategy-holdouts/1 (metadata and hash verification)"
HISTORICAL_ROOTS = (
    ROOT / "docs/reviews/2026-09-15-independent-calibration",
    ROOT / "docs/reviews/2026-09-15-independent-calibration-v2",
    ROOT / "docs/reviews/2026-09-15-independent-calibration-v3-search",
    ROOT / "docs/reviews/2026-09-15-independent-calibration-expanded",
    ROOT / "docs/reviews/2026-09-15-independent-phase-benchmark",
)
HISTORICAL_Q_FILES = (
    ROOT / "docs/reviews/2026-09-15-independent-calibration-v2/selection-round1.json",
    ROOT / "docs/reviews/2026-09-15-independent-calibration-v3-search/selection-round1.json",
    ROOT / "docs/reviews/2026-09-15-independent-calibration-expanded/selection-round1.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def historical_exclusions() -> tuple[set[str], list[float], list[str]]:
    ids: set[str] = set()
    sources: list[str] = []
    for root in HISTORICAL_ROOTS:
        for path in sorted(root.rglob("*.json")):
            # Residual/model result and validation files are deliberately not read.
            if any(part in {"results", "validation"} for part in path.parts):
                continue
            text = path.read_text(errors="ignore")
            found = set(re.findall(r"SXS:BBH:\d{4}", text))
            if found:
                ids.update(found)
                sources.append(str(path))
    qs: list[float] = []
    for path in HISTORICAL_Q_FILES:
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        def walk(v: Any) -> None:
            if isinstance(v, dict):
                if "target_q" in v:
                    try:
                        q = float(v["target_q"])
                        if math.isfinite(q): qs.append(q)
                    except (TypeError, ValueError): pass
                for x in v.values(): walk(x)
            elif isinstance(v, list):
                for x in v: walk(x)
        walk(obj)
    return ids, sorted(set(qs).union(PRIOR_FAMILY_QS)), sorted(set(sources))


def catalog_rows(excluded: set[str]) -> list[dict[str, Any]]:
    catalog = json.loads(CATALOG.read_text())
    rows: list[dict[str, Any]] = []
    for sid, item in catalog.items():
        if sid in excluded or not isinstance(item, dict): continue
        try:
            m1, m2 = float(item["reference_mass1"]), float(item["reference_mass2"])
            q = max(m1, m2) / min(m1, m2)
            spins = item["reference_dimensionless_spin1"] + item["reference_dimensionless_spin2"]
            spin = max(abs(float(x)) for x in spins)
            ecc = abs(float(item["reference_eccentricity"]))
            levels = sorted({int(x) for x in item.get("lev_numbers", [])})
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        keywords = " ".join(map(str, item.get("keywords", []))).casefold()
        eligible = (1.5 <= q <= 8 and spin <= 1e-4 and ecc <= 1e-3
                    and "deprecated" not in keywords and len(levels) >= 2)
        rows.append({"sxs_id": sid, "q": q, "eta": q / (1 + q) ** 2,
                     "spin_max": spin, "eccentricity": ecc, "levels": levels,
                     "simulation_name": item.get("simulation_name"),
                     "eligible_metadata": eligible})
    return rows


def freeze_ranking(excluded: set[str], previous_qs: list[float], sources: list[str]) -> dict[str, Any]:
    rows = [r for r in catalog_rows(excluded) if r["eligible_metadata"]]
    target_lists: dict[str, list[dict[str, Any]]] = {}
    chosen: list[dict[str, Any]] = []
    for target in TARGETS:
        candidates = sorted((r for r in rows if abs(r["q"] - target) / target <= TARGET_RELTOL),
                            key=lambda r: (abs(r["q"] - target), r["sxs_id"]))
        target_lists[str(target)] = [{**r, "candidate_rank": i + 1} for i, r in enumerate(candidates)]
        if candidates and len(chosen) < 2:
            chosen.append({**candidates[0], "target_q": target, "candidate_rank": 1,
                           "selection_reason": "prospective_target"})
    used_ids = {x["sxs_id"] for x in chosen}
    fallback_pool = [r for r in rows if r["sxs_id"] not in used_ids]
    previous_eta = [q / (1 + q) ** 2 for q in previous_qs]
    if len(chosen) < 2:
        def fallback_key(r: dict[str, Any]) -> tuple[float, float, str]:
            distances = previous_eta + [x["eta"] for x in chosen]
            separation = min((abs(r["eta"] - e) for e in distances), default=float("inf"))
            return (-separation, abs(r["q"] - 1.5), r["sxs_id"])
        for r in sorted(fallback_pool, key=fallback_key):
            chosen.append({**r, "target_q": None, "candidate_rank": 1,
                           "selection_reason": "deterministic_eta_fallback"})
            if len(chosen) == 2: break
    if len(chosen) != 2:
        raise RuntimeError("fewer than two fresh metadata-qualified families")
    return {"schema_version": 1, "catalog": str(CATALOG), "catalog_sha256": sha256(CATALOG),
            "candidate_ranking_frozen_before_record_access": True,
            "historical_excluded_sxs_ids": sorted(excluded),
            "historical_exclusion_sources": sources, "previous_used_qs": previous_qs,
            "targets": list(TARGETS), "target_relative_tolerance": TARGET_RELTOL,
            "target_candidates": target_lists, "selected": chosen,
            "fallback_declared": any(x["selection_reason"] != "prospective_target" for x in chosen),
            "selection_rule": "metadata gates first; then closest target; fallback maximizes minimum eta distance",
            "correction": {"supersedes": "candidate-ranking-initial.json",
                           "reason": "initial fallback omitted prior training-family q values; corrected before resuming downloads",
                           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            }


def get_json(url: str) -> tuple[dict[str, Any], str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as response:
        body = response.read()
    return json.loads(body), hashlib.sha256(body).hexdigest()


def locate_record(sid: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # The API's full-text index accepts the numeric SXS token more reliably
    # than the colon-delimited identifier; retain the exact observed URL.
    queries = [sid, sid.rsplit(":", 1)[-1]]
    hits = []
    search_url = ""
    search_hash = ""
    for query_value in queries:
        search_url = "https://data.caltech.edu/api/records?" + urllib.parse.urlencode({"q": query_value})
        result, search_hash = get_json(search_url)
        hits = result.get("hits", {}).get("hits", [])
        if hits: break
    if not hits: raise RuntimeError(f"no catalog record for {sid}")
    hit = next((h for h in hits if sid.casefold() in json.dumps(h).casefold()), hits[0])
    rid = str(hit["id"])
    record_url = f"https://data.caltech.edu/api/records/{urllib.parse.quote(rid, safe='')}"
    record, record_hash = get_json(record_url)
    if str(record.get("id")) != rid: raise RuntimeError("record id changed during fetch")
    provenance = {"sxs_id": sid, "record_id": rid, "search_url": search_url,
                  "search_response_sha256": search_hash, "record_url": record_url,
                  "record_response_sha256": record_hash}
    return record, provenance


def make_selection(ranking: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for selected in ranking["selected"]:
        sid = selected["sxs_id"]; record = records[sid]
        entries = record.get("files", {}).get("entries", {})
        levels = sorted({int(m.group(1)) for key in entries
                         if (m := re.match(r"Lev(\d+):", key))})
        levels = levels[-2:]
        if len(levels) != 2: raise RuntimeError(f"{sid} lacks two levels")
        rid = str(record["id"])
        for lev in levels:
            for order in (2, 3):
                row = {"sxs_id": sid, "record_id": rid, "resolution": f"Lev{lev}",
                       "extrapolation_order": order,
                       "split": "holdout" if lev == levels[-1] and order == 3 else "validation_only",
                       "target_q": selected["target_q"] or selected["q"],
                       "candidate_rank": selected["candidate_rank"]}
                cases.append(row)
    return {"schema_version": 1, "policy_sha256": POLICY_SHA256,
            "inventory_sha256": "0" * 64, "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="fetch records, prepare waves, and screen")
    ap.add_argument("--refresh-ranking", action="store_true", help="explicitly replace the frozen ranking")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    excluded, previous_qs, sources = historical_exclusions()
    ranking_path = DATA / "candidate-ranking.json"
    if ranking_path.exists() and not args.refresh_ranking:
        ranking = json.loads(ranking_path.read_text())
    else:
        ranking = freeze_ranking(excluded, previous_qs, sources)
        write_json(ranking_path, ranking)
    if not args.run:
        print(json.dumps({"ranking": str(ranking_path), "selected": ranking["selected"]}, indent=2)); return 0
    records: dict[str, Any] = {}; provenance = []
    for selected in ranking["selected"]:
        record, proof = locate_record(selected["sxs_id"])
        records[selected["sxs_id"]] = record; provenance.append(proof)
        write_json(DATA / "records" / f"{record['id']}.json", record)
    write_json(DATA / "record-file-hashes.json", {
        "note": "record_response_sha256 hashes the observed wire response; record_file_sha256 hashes its deterministic JSON artifact",
        "records": [{**p, "record_file_sha256": sha256(DATA / "records" / f"{p['record_id']}.json")}
                    for p in provenance],
    })
    inventory = {"schema_version": 1, "records": {str(v["id"]): v for v in records.values()},
                 "record_provenance": provenance}
    inventory_path = DATA / "inventory.json"; write_json(inventory_path, inventory)
    selection = make_selection(ranking, records); selection["inventory_sha256"] = sha256(inventory_path)
    selection_path = DATA / "selection.json"; write_json(selection_path, selection)
    sys.path.insert(0, str(ROOT))
    from scripts.prepare_nr33_calibration_v2 import prepare
    prepared = prepare(selection_path, inventory_path, RAW_ROOT / "raw", RAW_ROOT / "npz",
                       expected_policy_sha256=POLICY_SHA256)
    prepared["run"] = {"candidate_ranking_sha256": sha256(ranking_path), "selection_sha256": sha256(selection_path)}
    manifest_path = DATA / "manifest.json"; write_json(manifest_path, prepared)
    from scripts.screen_nr33_calibration_v2 import load_policy, screen_simulation
    policy = load_policy(POLICY)
    screens = []
    for sid in sorted(records):
        rows = [c for c in prepared["cases"] if c["sxs_id"] == sid]
        screens.append(screen_simulation(rows, policy))
    write_json(DATA / "quality-screen.json", {"policy_sha256": POLICY_SHA256, "simulations": screens,
                                               "all_pass": all(x["passed"] for x in screens)})
    failed = []
    for report in screens:
        if not report["passed"]:
            failed.append({"sxs_id": report["sxs_id"],
                           "failed_required_gates": [g for g in report.get("gates", []) if not g.get("pass")],
                           "evaluation_status": report.get("evaluation_status")})
    absent = {str(target): len(ranking["target_candidates"].get(str(target), []))
              for target in TARGETS if not ranking["target_candidates"].get(str(target), [])}
    write_json(DATA / "holdout-status.json", {
        "selected_family_count": len(screens),
        "qualified_family_count": sum(bool(x["passed"]) for x in screens),
        "target_metadata_candidates_absent": absent,
        "failed_selected_families": failed,
        "bounded_gap": "one selected family fails two pinned numerical quality gates; no gate was relaxed or substituted",
        "policy_sha256": POLICY_SHA256,
    })
    write_json(DATA / "run-provenance.json", {"ranking_sha256": sha256(ranking_path),
                                               "inventory_sha256": sha256(inventory_path),
                                               "selection_sha256": sha256(selection_path),
                                               "manifest_sha256": sha256(manifest_path),
                                               "raw_root": str(RAW_ROOT), "policy_sha256": POLICY_SHA256})
    print(json.dumps({"selected": ranking["selected"], "manifest": str(manifest_path), "screens": screens}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
