"""Separate public-binary diagnostics with explicit options and mode controls.

Neither the frozen acceptance inventories nor their thresholds are modified.
This verifier is kept separate from clean-room implementers.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import lal
import lalsimulation as lalsim
import numpy as np
import torch
import torchwave

sys.path.insert(0, str(Path(torchwave.__file__).resolve().parents[2] / "tests"))
import parity_harness as harness


parser = argparse.ArgumentParser()
parser.add_argument("--json-out", type=Path, required=True)
args = parser.parse_args()
original_choose = lalsim.SimInspiralChooseFDWaveform
original_native = harness.get_fd_waveform
selected_case = None
captures = {}


def choose_explicit(*positional, **keywords):
    extra = positional[18]
    options = selected_case["oracle_options"]
    if "PhenomXPFinalSpinMod" in options:
        lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
            extra, options["PhenomXPFinalSpinMod"])
        assert lalsim.SimInspiralWaveformParamsLookupPhenomXPrecVersion(extra) == options["PhenomXPrecVersion"]
    if "modes" in options:
        modes = lalsim.SimInspiralCreateModeArray()
        for ell, emm in options["modes"]:
            lalsim.SimInspiralModeArrayActivateMode(modes, ell, emm)
        lalsim.SimInspiralWaveformParamsInsertModeArray(extra, modes)
    hp, hc = original_choose(*positional, **keywords)
    captures.setdefault(selected_case["id"], {})["lal"] = (
        hp.data.data.copy(), hc.data.data.copy())
    return hp, hc


def native_capture(*positional, **keywords):
    hp, hc = original_native(*positional, **keywords)
    captures.setdefault(selected_case["id"], {})["native"] = (
        hp.detach().cpu().numpy().copy(), hc.detach().cpu().numpy().copy())
    return hp, hc


cases, mode_groups = [], []
bases = harness.get_canonical_catalog_cases() + harness.get_canonical_holdout_cases()
for base in bases:
    if base["torch_name"] in ("IMRPhenomXP", "IMRPhenomXPHM"):
        for version, remnant in ((102, 0), (223, 3)):
            case = copy.deepcopy(base)
            case["id"] += f"_Prec{version}_FinalSpin{remnant}"
            case["params"]["prec_version"] = version
            case["oracle_options"] = dict(PhenomXPrecVersion=version, PhenomXPFinalSpinMod=remnant)
            cases.append(case)
    if base["torch_name"] in ("IMRPhenomHM", "IMRPhenomXHM"):
        modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
        if base["torch_name"] == "IMRPhenomHM":
            modes.append((4, 3))
        group = dict(base_id=base["id"], components=[])
        for active in [[mode] for mode in modes] + [modes]:
            case = copy.deepcopy(base)
            suffix = "AllExplicitModes" if len(active) > 1 else f"Mode{active[0][0]}{active[0][1]}"
            case["id"] += "_" + suffix
            case["params"]["modes"] = active
            case["oracle_options"] = dict(modes=active)
            cases.append(case)
            if len(active) == 1:
                group["components"].append(case["id"])
            else:
                group["total"] = case["id"]
        mode_groups.append(group)

lalsim.SimInspiralChooseFDWaveform = choose_explicit
harness.get_fd_waveform = native_capture
try:
    results = []
    for selected_case in cases:
        result = harness.run_oracle_comparison(selected_case, strict=True)
        results.append(result)
        print(selected_case["id"], result["status"], flush=True)
finally:
    lalsim.SimInspiralChooseFDWaveform = original_choose
    harness.get_fd_waveform = original_native

closures = []
for group in mode_groups:
    record = dict(base_id=group["base_id"], errors={})
    try:
        for side in ("lal", "native"):
            for index, polarization in enumerate(("plus", "cross")):
                total = captures[group["total"]][side][index]
                summed = sum(captures[key][side][index] for key in group["components"])
                record["errors"][f"{side}_{polarization}"] = float(
                    np.linalg.norm(summed - total) / np.linalg.norm(total))
        record["passed"] = all(np.isfinite(v) and v <= 1e-12 for v in record["errors"].values())
    except (KeyError, ValueError) as exc:
        record.update(passed=False, reason=str(exc))
    closures.append(record)

report = dict(
    scope="supplemental_explicit_components", native_module=torchwave.__file__,
    versions=dict(python=sys.version.split()[0], lal=lal.__version__,
                  lalsimulation=lalsim.__version__, torch=torch.__version__),
    public_lookup_defaults={name: getattr(lalsim, "SimInspiralWaveformParamsLookup" + name)(lal.CreateDict())
                            for name in ("PhenomXPrecVersion", "PhenomXPFinalSpinMod")},
    public_defaults_note="Lookup values do not establish the generation routine's resolved internal dispatch.",
    verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    cases=cases, results=results, mode_sum_controls=closures,
    num_cases=len(results), num_passed=sum(r["status"] == "PASSED" for r in results),
    note="Separate fixed diagnostics, including overlapping controls; not additions to frozen acceptance counts. No fitted alignment or coefficients.",
)
args.json_out.write_text(json.dumps(report, indent=2) + "\n")
print(f"Components: {report['num_passed']}/{report['num_cases']}; mode controls: {sum(c['passed'] for c in closures)}/{len(closures)}")
raise SystemExit(0 if report["num_passed"] == report["num_cases"] and all(c["passed"] for c in closures) else 1)
