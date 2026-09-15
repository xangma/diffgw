"""Audit external validation and the transparent inventory replay, without LAL."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from scripts.prepare_nr33_calibration_v2 import inventory_files


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


base = Path(__file__).resolve().parent
snapshot = read(base / "integrity-start.json")
protocol = read(base / "protocol.json")
selection = read(base / "selection-round1.json")
replay = read(base / "selection-replay.json")
original = read(base / "nr33-manifest-round1.json")
manifest = read(base / "nr33-manifest-replay.json")
inventory = read(base / "candidate-inventory-during-prepare.json")
quality = read(base / "quality-replay.json")
qualified = read(base / "qualified-replay.json")
freeze = read(base / "validation-freeze.json")
validation = read(base / "validation/nr33-external-validation.json")

integrity = {}
for key, entries in snapshot.items():
    if isinstance(entries, dict):
        changed = [p for p, expected in entries.items() if sha(p) != expected]
        assert not changed, changed
        integrity[key] = {"checked": len(entries), "changed": changed}
assert sha("LICENSE") == snapshot["license_sha256"]
for path, expected in freeze["input_sha256"].items():
    assert sha(path) == expected, path
assert selection["cases"] == replay["cases"] and len(replay["cases"]) == 24
assert selection["candidate_order"] == replay["candidate_order"]
assert len(replay["candidate_order"]) == 181
for row in replay["candidate_order"]:
    current = next(c for c in inventory["candidates"] if c["sxs_id"] == row["sxs_id"])
    assert all(current[k] == v for k, v in row.items())
assert sha(manifest["inventory"]) == manifest["inventory_sha256"] == replay["inventory_sha256"]
assert sha(manifest["selection_manifest"]) == manifest["selection_manifest_sha256"]
assert sha(base / "nr33-manifest-replay.json") == quality["input_manifest_sha256"]
assert sha(base / "qualified-replay.json") == quality["qualified_manifest"]["sha256"]
assert sha("scripts/prepare_nr33_calibration_v2.py") == manifest["run"]["script_sha256"]
assert sha("scripts/screen_nr33_calibration_v2.py") == quality["source_sha256"]["original_screen"]
assert sha("scripts/validate_nr33_amplitude_pilot.py") == validation["source_sha256"]["validator"]

raw_checks, wave_checks = [], []
for selected, before, row in zip(replay["cases"], original["cases"], manifest["cases"]):
    assert before["id"] == row["id"] and before["metadata"] == row["metadata"]
    assert before["npz_sha256"] == row["npz_sha256"] == sha(row["path"]) == sha(before["path"])
    source = row["metadata"]["source_identity"]
    files = inventory_files(inventory["records"][selected["record_id"]], selected)
    stem = row["sxs_id"].replace(":", "_") + f"_{source['record_id']}_{row['resolution']}_N{row['extrapolation_order']}"
    for key, suffix in {"h5": ".h5", "sidecar": ".json", "metadata": "_metadata.json"}.items():
        expected = source["selected_files"][key]
        assert all(expected[k] == files[key][k] for k in ("key", "size", "checksum"))
        assert expected["content"] == files[key]["links"]["content"]
        path = Path(manifest["raw_dir"]) / (stem + suffix)
        data = path.read_bytes()
        assert len(data) == expected["size"]
        assert hashlib.md5(data).hexdigest() == expected["checksum"].removeprefix("md5:")
        assert sha(path) == expected["sha256"]
        raw_checks.append({"path": str(path), "sha256": sha(path), "size_md5_sha256_pass": True})
    with np.load(row["path"]) as wave, np.load(before["path"]) as prior:
        assert wave.files == prior.files and all(np.array_equal(wave[k], prior[k]) for k in wave.files)
        t, h22, h33 = wave["time"], wave["h22"], wave["h33"]
        assert t.shape == h22.shape == h33.shape
        assert all(np.isfinite(x).all() for x in (t, h22, h33)) and np.all(np.diff(t) > 0)
        peak, end = float(t[np.argmax(np.abs(h22))]), float(t[-1])
        assert peak == row["metadata"]["peak_time"] and end == row["metadata"]["time_end"]
        wave_checks.append({"id": row["id"], "sha256": sha(row["path"]), "samples": len(t), "post_peak_M": end - peak})

assert quality["simulations"] == read(base / "quality-round1.json")["simulations"]
assert quality["missing_or_failed_targets"] == [1.2, 1.8, 2.5]
assert not quality["overall_pass"] and len(qualified["cases"]) == 5
assert [r["id"] for r in qualified["cases"]] == quality["selected_primary_case_ids"]
for row in qualified["cases"]:
    assert row["split"] == "holdout" and row["sxs_id"] not in protocol["excluded_sxs_ids"]
    sim = next(s for s in quality["simulations"] if s["primary_case_id"] == row["id"])
    assert sim["passed"] and all(g["pass"] for g in sim["gates"])

digest = sha(protocol["calibration_path"])
assert digest == protocol["calibration_sha256"] == validation["calibration_before_sha256"] == validation["calibration_after_sha256"]
assert datetime.fromisoformat(freeze["created_utc"]) < datetime.fromisoformat(validation["run"]["started_utc"])
predictions = read(base / "validation/nr33-external-predictions.json")["predictions"]
targets = {r["case_id"]: r for r in read(base / "validation/nr33-external-targets.json")["targets"]}
metrics = {r["case_id"]: r for r in validation["cases"]}
assert list(metrics) == quality["selected_primary_case_ids"]
results = []
for pred in predictions:
    target, metric = targets[pred["case_id"]], metrics[pred["case_id"]]
    assert pred["frequency_Mf"] == target["frequency_Mf"] == metric["frequency_Mf"]
    a, b = np.array(pred["prediction_amplitude_geometric"]), np.array(target["target_amplitude_geometric"])
    assert float(np.linalg.norm(a - b) / np.linalg.norm(b)) == metric["relative_amp_l2"]
    lal = metric["lal_validation"]
    assert lal["xhm_release_version"] == 122022 and lal["mode_argument"] == [3, -3]
    row = next(r for r in qualified["cases"] if r["id"] == pred["case_id"])
    results.append({"case_id": pred["case_id"], "target_q": row["target_q"], "actual_q": row["q"],
                    "region": "extrapolation" if row["target_q"] in protocol["extrapolation_targets"] else "interpolation",
                    "fit_vs_NR": metric["relative_amp_l2"],
                    "fit_vs_LAL": lal["prediction_vs_lal"]["relative_amp_l2"],
                    "NR_vs_LAL": lal["target_vs_lal"]["relative_amp_l2"]})

report = {"status": "verified_replay", "completed_utc": datetime.now(timezone.utc).isoformat(),
          "host": platform.node(), "cwd": str(Path.cwd()), "command": [sys.executable, *sys.argv],
          "audit_script_sha256": sha(__file__), "integrity": integrity, "license_unchanged": True,
          "original_full_inventory_recovered": False, "original_inventory_expected_sha256": selection["inventory_sha256"],
          "candidate_order_entries_unchanged": 181, "selection_rows_unchanged": 24,
          "replay_waveforms_and_metadata_identical": True, "raw_checks": raw_checks, "waveform_checks": wave_checks,
          "counts": quality["counts"], "missing_or_failed_targets": quality["missing_or_failed_targets"],
          "calibration_sha256": digest, "new_fitting_performed": False,
          "external_inputs_frozen_before_validation": True, "results": results}
with (base / "verification.json").open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": report["status"], "raw_files": len(raw_checks), "waveforms": len(wave_checks), "results": results}))
