"""Read-only review probes using TorchWave and public LAL binary APIs.

Run from the repository root with PYTHONPATH=src:tests. No coefficients are
estimated and neither implementation source nor binary internals are read.
"""
import copy
import json
import platform

import lal
import lalsimulation as ls
import numpy as np
import torch

from parity_harness import get_canonical_catalog_cases, run_oracle_comparison
from torchwave.approximants.teobresums import TEOBResumS
from torchwave.constants import MSUN_SEC


def lal_waveform(case, version=None, final_spin=None, convention=None):
    p, g = case["params"], case["grid"]
    extra = lal.CreateDict()
    for value, suffix in ((version, "PhenomXPrecVersion"),
                          (final_spin, "PhenomXPFinalSpinMod"),
                          (convention, "PhenomXPConvention")):
        if value is not None:
            getattr(ls, "SimInspiralWaveformParamsInsert" + suffix)(extra, value)
    hp, hc = ls.SimInspiralChooseFDWaveform(
        p["mass1"] * lal.MSUN_SI, p["mass2"] * lal.MSUN_SI,
        p["spin1x"], p["spin1y"], p["spin1z"],
        p["spin2x"], p["spin2y"], p["spin2z"],
        p["distance"] * 1e6 * lal.PC_SI, p["inclination"], p["phic"],
        0., 0., 0., g["df"], g["flow"], g["fhigh"], p["f_ref"], extra,
        ls.GetApproximantFromString(case["lal_name"]),
    )
    f = np.arange(len(hp.data.data)) * g["df"]
    mask = (f >= g["flow"]) & (f <= g["fhigh"])
    return np.stack([hp.data.data[mask], hc.data.data[mask]])


def relative_l2(value, reference):
    return (np.linalg.norm(value - reference, axis=-1)
            / np.linalg.norm(reference, axis=-1)).tolist()


def main():
    report = {"host": platform.node(), "lal_version": lal.__version__,
              "lalsimulation_version": ls.__version__, "binary_defaults": {}}
    for name in ("PhenomXPrecVersion", "PhenomXPFinalSpinMod", "PhenomXPConvention",
                 "PhenomXPHMPrecModes", "PhenomXPHMThresholdMband"):
        report["binary_defaults"][name] = getattr(
            ls, "SimInspiralWaveformParamsLookup" + name)(lal.CreateDict())
    cases = get_canonical_catalog_cases()
    report["precession"] = {}
    for name in ("IMRPhenomXP", "IMRPhenomXPHM"):
        case = next(c for c in cases if c["torch_name"] == name)
        baseline = lal_waveform(case)
        entry = {"lal_vs_lal_default_plus_cross_l2": {}, "paired_harness": {}}
        for version in (102, 223, 300):
            entry["lal_vs_lal_default_plus_cross_l2"][str(version)] = relative_l2(
                lal_waveform(case, version), baseline)
        for mod in (0, 3, 4):
            entry["lal_vs_lal_default_plus_cross_l2"]["final_spin_" + str(mod)] = relative_l2(
                lal_waveform(case, final_spin=mod), baseline)
        for version in (None, 102, 223):
            paired = copy.deepcopy(case)
            if version is not None:
                paired["params"]["prec_version"] = version
            result = run_oracle_comparison(paired, strict=True)
            entry["paired_harness"][str(version)] = {
                "status": result["status"],
                "plus_l2": result.get("plus", {}).get("complex_relative_l2"),
                "cross_l2": result.get("cross", {}).get("complex_relative_l2"),
                "reason": result.get("reason"),
            }
        report["precession"][name] = entry
    model = TEOBResumS()
    hp, hc = model(torch.arange(30., 100., dtype=torch.float64),
                   mass1=1.4, mass2=1.4, inclination=0., num_steps=150)
    report["teob"] = {"face_on_max_abs_hp_minus_hc": (hp - hc).abs().max().item(),
                      "face_on_hp_norm": hp.norm().item(),
                      "catalog_default_duration_seconds": float(1500 * 0.5 * 2.8 * MSUN_SEC),
                      "note": "Short probe tests polarization construction, not convergence."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
