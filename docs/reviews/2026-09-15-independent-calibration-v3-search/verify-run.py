"""Verify the completed v3 artifacts without fitting or calling LAL."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from scripts.calibrate_nr33_v2 import check_inputs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


base = Path(__file__).resolve().parent
v2 = base.with_name("2026-09-15-independent-calibration-v2")
snapshot = read(base / "integrity-start.json")
plan = read(base / "search-plan.json")
selection = read(base / "selection-round1.json")
manifest = read(base / "nr33-manifest-round1.json")
new = read(base / "nr33-new-manifest-round1.json")
quality = read(base / "quality-screen-round1.json")
calibration = read(base / "nr33-calibration.json")
freeze = read(base / "nr33-calibration-freeze.json")
validation = read(base / "validation/nr33-validation.json")

assert sha(base / "search-plan.json") == selection["search_plan_sha256"]
assert sha(base / "candidate-inventory.json") == selection["inventory_sha256"]
assert sha(base / "selection-round1.json") == new["selection_manifest_sha256"]
assert sha(v2 / "quality-policy.json") == plan["quality_policy_sha256"] == selection["policy_sha256"]
assert sha(base / "nr33-manifest-round1.json") == quality["input_manifest_sha256"]
assert sha("scripts/prepare_nr33_calibration_v2.py") == new["run"]["script_sha256"]

integrity = {}
for key, entries in snapshot.items():
    if key.endswith("_sha256") and isinstance(entries, dict):
        changed = [p for p, expected in entries.items() if sha(p) != expected]
        assert not changed, changed
        integrity[key] = {"checked": len(entries), "changed": changed}
assert sha("LICENSE") == snapshot["license_sha256"]

old_rows = {r["id"]: r for r in read(v2 / "nr33-manifest-round1.json")["cases"]}
reused = [r for r in manifest["cases"] if r["sxs_id"] in plan["preserved_passing_sxs_ids"]]
assert len(reused) == 20 and all(r == old_rows[r["id"]] for r in reused)
assert manifest["cases"] == reused + new["cases"] and len(manifest["cases"]) == 28
for source in manifest["source_manifests"]:
    assert sha(source["path"]) == source["sha256"]

raw_checks = []
for row in new["cases"]:
    meta = row["metadata"]
    source = meta["source_identity"]
    stem = row["sxs_id"].replace(":", "_") + f"_{source['record_id']}_{row['resolution']}_N{row['extrapolation_order']}"
    for key, suffix in {"h5": ".h5", "sidecar": ".json", "metadata": "_metadata.json"}.items():
        path = Path(new["raw_dir"]) / (stem + suffix)
        data = path.read_bytes()
        expected = source["selected_files"][key]
        assert len(data) == expected["size"]
        assert hashlib.md5(data).hexdigest() == expected["checksum"].removeprefix("md5:")
        assert sha(path) == expected["sha256"]
        raw_checks.append({"path": str(path), "sha256": sha(path), "size_md5_sha256_pass": True})

waves = []
for row in manifest["cases"]:
    assert sha(row["path"]) == row["npz_sha256"]
    with np.load(row["path"]) as wave:
        t, h22, h33 = wave["time"], wave["h22"], wave["h33"]
        assert t.shape == h22.shape == h33.shape
        assert all(np.isfinite(x).all() for x in (t, h22, h33)) and np.all(np.diff(t) > 0)
        peak, end = float(t[np.argmax(np.abs(h22))]), float(t[-1])
        assert peak == row["metadata"]["peak_time"] and end == row["metadata"]["time_end"]
        waves.append({"id": row["id"], "npz_sha256": sha(row["path"]), "samples": len(t), "post_peak_M": end - peak})

assert quality["overall_pass"] and all(s["passed"] for s in quality["simulations"])
for sim in quality["simulations"]:
    measured = next(w for w in waves if w["id"] == sim["primary_case_id"])
    assert measured["post_peak_M"] == sim["h22_peak"]["duration_to_end_M"]
_, _, train, holdout = check_inputs(v2 / "quality-policy.json", base / "quality-screen-round1.json", base / "nr33-qualified-manifest.json")
assert {c.case_id for c in train} == {c["case_id"] for c in calibration["direct_train"]}
assert len(train) == 5 and len(holdout) == 2
digest = sha(base / "nr33-calibration.json")
assert digest == "ade377c62a57a5d9cff513e34f22b0fe43677177a0c1577a9a4af5ad3b8aa561"
assert digest == freeze["calibration_sha256"] == validation["calibration_before_sha256"] == validation["calibration_after_sha256"]
assert datetime.fromisoformat(freeze["fit_completed_at_utc"]) < datetime.fromisoformat(validation["run"]["started_utc"])
for path, expected in freeze["input_sha256"].items():
    assert sha(path) == expected

predictions = read(base / "validation/nr33-frozen-predictions.json")["predictions"]
targets = {r["case_id"]: r for r in read(base / "validation/nr33-heldout-targets.json")["targets"]}
metrics = {r["case_id"]: r for r in validation["cases"]}
for pred in predictions:
    target, metric = targets[pred["case_id"]], metrics[pred["case_id"]]
    assert pred["frequency_Mf"] == target["frequency_Mf"] == metric["frequency_Mf"]
    a, b = np.array(pred["prediction_amplitude_geometric"]), np.array(target["target_amplitude_geometric"])
    assert float(np.linalg.norm(a - b) / np.linalg.norm(b)) == metric["relative_amp_l2"]
assert {r["case_id"] for r in predictions} == {c.case_id for c in holdout}

artifacts = {str(p.relative_to(base)): sha(p) for p in base.rglob("*") if p.is_file() and p.suffix in {".json", ".png", ".py"} and p.name != "verification.json"}
report = {"status": "verified", "completed_utc": datetime.now(timezone.utc).isoformat(),
          "host": platform.node(), "cwd": str(Path.cwd()), "command": [sys.executable, *sys.argv],
          "audit_script_sha256": sha(__file__), "integrity": integrity, "license_unchanged": True,
          "reused_rows_identical": len(reused), "new_raw_file_checks": raw_checks,
          "waveform_checks": waves, "counts": quality["counts"], "calibration_sha256": digest,
          "fit_completed_before_lal_validation": True, "holdout_metrics_recomputed": True,
          "artifact_sha256": artifacts}
with (base / "verification.json").open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": report["status"], "raw_files": len(raw_checks), "waveforms": len(waves), "integrity": integrity}))
