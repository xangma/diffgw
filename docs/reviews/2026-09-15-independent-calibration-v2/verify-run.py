"""Audit this completed NR-only screening run; never fit or call LAL."""

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


base = Path(__file__).resolve().parent
snapshot = json.loads((base / "integrity-start.json").read_text())
manifest = json.loads((base / "nr33-manifest-round1.json").read_text())
quality = json.loads((base / "quality-screen-round1.json").read_text())
policy = json.loads((base / "quality-policy.json").read_text())
inventory = json.loads((base / "candidate-inventory.json").read_text())
selection = json.loads((base / "selection-round1.json").read_text())

assert sha(base / "quality-policy.json") == snapshot["quality_policy_sha256"]
assert sha(base / "candidate-inventory.json") == selection["inventory_sha256"]
assert sha(base / "selection-round1.json") == manifest["selection_manifest_sha256"]
assert sha(base / "nr33-manifest-round1.json") == quality["input_manifest_sha256"]
assert sha("scripts/prepare_nr33_calibration_v2.py") == manifest["run"]["script_sha256"]

integrity = {}
for key in ("old_artifact_sha256", "runtime_sha256"):
    changed = [path for path, expected in snapshot[key].items() if sha(path) != expected]
    assert not changed, changed
    integrity[key] = {"checked": len(snapshot[key]), "changed": changed}

waves = []
for row in manifest["cases"]:
    assert sha(row["path"]) == row["npz_sha256"]
    with np.load(row["path"]) as wave:
        t, h22, h33 = wave["time"], wave["h22"], wave["h33"]
        assert t.shape == h22.shape == h33.shape
        assert np.isfinite(t).all() and np.isfinite(h22).all() and np.isfinite(h33).all()
        assert np.all(np.diff(t) > 0)
        peak = float(t[np.argmax(np.abs(h22))])
        end = float(t[-1])
        assert peak == row["metadata"]["peak_time"]
        assert end == row["metadata"]["time_end"]
        waves.append({"id": row["id"], "samples": len(t), "npz_sha256": sha(row["path"]),
                      "peak_h22_M": peak, "end_M": end, "post_peak_M": end - peak})

failed = [s for s in quality["simulations"] if not s["passed"]]
assert {s["target_q"] for s in failed} == {6, 8}
for sim in failed:
    assert [g["name"] for g in sim["gates"] if not g["pass"]] == ["post_peak_duration"]
    assert len([c for c in inventory["candidates"] if c["target_q"] == sim["target_q"]]) == 1
    measured = next(w for w in waves if w["id"] == sim["primary_case_id"])
    assert measured["post_peak_M"] == sim["h22_peak"]["duration_to_end_M"]

assert quality["overall_pass"] is False and quality["qualified_manifest"] is None
try:
    check_inputs(base / "quality-policy.json", base / "quality-screen-round1.json",
                 base / "nr33-qualified-manifest.json")
except ValueError as error:
    guard = str(error)
    assert guard == "NR quality screen did not pass; no fit allowed"
else:
    raise AssertionError("fit admission unexpectedly succeeded")

for name in ("nr33-qualified-manifest.json", "nr33-calibration.json", "nr33-calibration-freeze.json"):
    assert not (base / name).exists(), name

sources = ["scripts/prepare_nr33_calibration_v2.py", "scripts/screen_nr33_calibration_v2.py",
           "scripts/calibrate_nr33_v2.py", "scripts/prepare_nr33_calibration_data.py",
           "scripts/fit_nr33_amplitude_pilot.py", "scripts/kerr_leaver_solver.py"]
report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
          "cwd": str(Path.cwd()), "command": [sys.executable, *sys.argv],
          "audit_script_sha256": sha(__file__), "numpy_version": np.__version__,
          "python_version": platform.python_version(), "status": "insufficient_quality_do_not_fit",
          "integrity": integrity, "input_sha256": {p.name: sha(p) for p in
              (base / "integrity-start.json", base / "quality-policy.json", base / "candidate-inventory.json",
               base / "selection-round1.json", base / "nr33-manifest-round1.json", base / "quality-screen-round1.json")},
          "source_sha256": {p: sha(p) for p in sources}, "waveform_checks": waves,
          "counts": quality["counts"], "failed_target_q": [s["target_q"] for s in failed],
          "fit_admission_rejection": guard, "calibration_created": False, "LAL_called": False,
          "data_location": "raw H5 and decoded NPZ remain outside repository under /tmp/torchwave-nr33-v2"}
with (base / "verification.json").open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": report["status"], "waveforms_verified": len(waves), "integrity": integrity}))
