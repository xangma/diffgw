"""Generate native strict NRTidal before/after evidence."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import lal
import lalsimulation
import torch
from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases, run_oracle_comparison
from torchwave.approximants import imrphenomd_nrtidal as nrt

CASES = [c for fn in (get_canonical_catalog_cases, get_canonical_holdout_cases)
         for c in fn() if "NRTidal" in c.get("torch_name", "")]

def run(label, d):
    nrt.NRTIDAL_V2_AMP_COEFFS["d"] = d
    rows = []
    for case in CASES:
        result = run_oracle_comparison(case, strict=True)
        rows.append({"scope": "holdout" if "holdout" in case["id"] else "catalog",
                     "case_id": case["id"], "input_sha256": hashlib.sha256(
                         json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                     "result": result})
    return {"label": label, "amplitude_denominator": d, "cases": rows}

root = Path(__file__).resolve().parents[5]
paths = [Path(__file__).resolve(), root / "tests/parity_harness.py",
         root / "src/torchwave/constants.py", root / "src/torchwave/waveforms.py"]
paths += [root / "src/torchwave/approximants" / name for name in (
    "imrphenomd_nrtidal.py", "nrtidal_published_fits.py", "imrphenomd.py",
    "phenomd_unrounded_fits.py", "kerr_qnm_interp.py", "kerr_qnm_generated.py")]
paths += sorted((root / "src/torchwave/detectors").glob("*.py"))

def hashes():
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}

out = Path(__file__).with_name("results.json")
before_hashes = hashes()
original = nrt.NRTIDAL_V2_AMP_COEFFS["d"]
try:
    data = {"command": f"PYTHONPATH=src:tests {sys.executable} {Path(__file__).relative_to(root)}",
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "lal": lal.__version__, "lalsimulation": lalsimulation.__version__},
            "source_sha256_before": before_hashes,
            "before": run("before", 13477.8073677), "after": run("after", 13477.8)}
finally:
    nrt.NRTIDAL_V2_AMP_COEFFS["d"] = original
data["source_sha256_after"] = hashes()
data["changed_sources"] = [name for name, digest in before_hashes.items()
                           if digest != data["source_sha256_after"][name]]
out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(out)
