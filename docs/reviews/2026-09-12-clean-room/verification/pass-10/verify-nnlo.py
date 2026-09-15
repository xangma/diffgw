"""Supplemental explicit NNLO-102/FinalSpinMod-0 public binary comparisons.

The frozen default-configuration inventories and acceptance gates are untouched.
This verifier supplies the paper-defined remnant option explicitly to the public
LAL API; it does not inspect implementation code or fit waveform residuals.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import lal
import lalsimulation as lalsim
import torchwave
# Resolve tests alongside the selected native source, including frozen snapshots.
sys.path.insert(0, str(Path(torchwave.__file__).resolve().parents[2] / "tests"))
import parity_harness as harness


parser = argparse.ArgumentParser()
parser.add_argument("--json-out", type=Path, required=True)
args = parser.parse_args()
choose_fd = lalsim.SimInspiralChooseFDWaveform


def choose_nnlo(*positional, **keywords):
    # Public ChooseFDWaveform argument 18 is its LAL parameter dictionary.
    # PrecVersion is already forwarded by the existing harness.
    extra = positional[18]
    lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(extra, 0)
    return choose_fd(*positional, **keywords)


cases = []
for case in harness.get_canonical_catalog_cases() + harness.get_canonical_holdout_cases():
    if case["torch_name"] in ("IMRPhenomXP", "IMRPhenomXPHM"):
        case = copy.deepcopy(case)
        case["id"] += "_NNLO102_FinalSpin0"
        case["params"]["prec_version"] = 102
        cases.append(case)
lalsim.SimInspiralChooseFDWaveform = choose_nnlo
try:
    results = [harness.run_oracle_comparison(case, strict=True) for case in cases]
finally:
    lalsim.SimInspiralChooseFDWaveform = choose_fd
report = dict(scope="supplemental_explicit_nnlo", native_module=torchwave.__file__,
              oracle_options={"PhenomXPrecVersion": 102, "PhenomXPFinalSpinMod": 0},
              oracle_default_final_spin_mod=lalsim.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(lal.CreateDict()),
              cases=cases, results=results, num_cases=len(results),
              num_passed=sum(r["status"] == "PASSED" for r in results),
              note="Separate diagnostic inventory; no frozen catalog/holdout acceptance claim.")
args.json_out.write_text(json.dumps(report, indent=2) + "\n")
print(f"Explicit NNLO: {report['num_passed']}/{report['num_cases']} passed; {args.json_out}")
raise SystemExit(0 if report["num_passed"] == report["num_cases"] else 1)
