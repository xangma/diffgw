"""Development-only frequency-crossing lag scan for one frozen v4 comparison."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_peak_anchor import fitted_peak, public_mode
from dev.native_rom.diagnose_q8_frequency_anchor import frequency_anchor
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from torchwave.approximants.eob import numerical_attachment as attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_public_frequency_lag_plan.json"
OUTPUT = EVIDENCE / "source_aligned_public_frequency_lag.json"


def _aligned_metric(nt, nm, rt, rm, mass_seconds, native, public, width_M):
    nt = (nt-native["time_s"])/mass_seconds
    rt = (rt-public["time_s"])/mass_seconds
    nm = nm*np.exp(-1j*native["phase_rad"])
    rm = rm*np.exp(-1j*public["phase_rad"])
    common = (rt >= nt[0]) & (rt <= nt[-1])
    if width_M is not None:
        common &= abs(rt) <= width_M
    t = rt[common]
    if len(t) < 3:
        raise ValueError("too few common frequency-chart samples")
    candidate = np.interp(t, nt, nm.real)+1j*np.interp(t, nt, nm.imag)
    reference = rm[common]
    return {"samples": int(len(t)), "support_M": [float(t[0]), float(t[-1])],
            "complex_relative_l2": float(np.linalg.norm(candidate-reference)
                                         /np.linalg.norm(reference))}


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-public-frequency-lag-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["level"] != "refined"
            or plan["targets_Momega22"] != [0.09, 0.1, 0.11, 0.12, 0.14, 0.16]
            or plan["local_half_width_M"] != 20.0
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen frequency-lag plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT/item["path"]) != item["sha256"]:
            raise ValueError(f"frozen {name} input changed")
    boundary_plan = json.loads((ROOT/plan["inputs"]["boundary_plan"]["path"]).read_text())
    boundary_result = json.loads((ROOT/plan["inputs"]["boundary_result"]["path"]).read_text())
    gap = json.loads((ROOT/plan["inputs"]["gap_result"]["path"]).read_text())
    source_plan = json.loads((ROOT/boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT/boundary_plan["inputs"]["ladder_plan"]["path"]).read_text())
    identity = driver._prepare_public(source_plan)
    level = next(item for item in ladder["levels"] if item["name"] == plan["level"])
    settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"], "atol": level["atol"],
                "rtol": level["rtol"]}
    m1, m2, chi1, chi2, omega_start = driver.CASES[plan["case"]]

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    started = time.monotonic()
    result = {"schema": "diffgw.seobnrv4.source-aligned-public-frequency-lag.v1",
              "status": "running", "release_qualified": False,
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
              "checks": {}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    diagnostic, series, stage, _, _ = _capture_level(
        plan["level"], settings, source_plan, identity, boundary_plan,
        _source_field(), hamiltonian, flux,
        attachment._full_series_nqc, attachment._eval_batch_scalar)
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
               omega_start=omega_start)
    run = {"time": np.asarray(series[0]), "mode": np.asarray(series[1]),
           "peak_time_M": float(stage["orbital_peak_time_M"]),
           "dt_geom": settings["coarse_output_step_M"]}
    nt, nm, rt, rm, mass_seconds = public_mode(row, run)
    native_peak = fitted_peak(nt, nm, "quadratic_power")
    public_peak = fitted_peak(rt, rm, "quadratic_power")
    crossings = []
    for target in plan["targets_Momega22"]:
        native = frequency_anchor(nt, nm, mass_seconds, target, native_peak["time_s"])
        public = frequency_anchor(rt, rm, mass_seconds, target, public_peak["time_s"])
        crossings.append({
            "target_Momega22": target, "native": native, "public": public,
            "native_minus_public_peak_lag_M": native["peak_lag_M"]-public["peak_lag_M"],
            "local": _aligned_metric(nt, nm, rt, rm, mass_seconds, native, public,
                                     plan["local_half_width_M"]),
            "whole": _aligned_metric(nt, nm, rt, rm, mass_seconds, native, public, None),
        })
    lag = np.array([item["native_minus_public_peak_lag_M"] for item in crossings])
    control = next(item for item in crossings if item["target_Momega22"] == 0.1)
    result.update({
        "status": "diagnostic_complete", "elapsed_s": time.monotonic()-started,
        "candidate": _series_summary(series), "public_binary": identity,
        "crossings": crossings,
        "lag_range_M": float(np.ptp(lag)),
        "checks": {
            "candidate_hash_replays": _series_summary(series)
                == boundary_result["levels"][plan["level"]]["candidate_torch"],
            "public_binary_replays": identity == diagnostic["public_binary"],
            "frequency_point_one_replays": abs(control["whole"]["complex_relative_l2"]
                - gap["frequency_anchor_complex_relative_l2"]) <= 1e-12,
            "all_crossings_pre_peak": all(
                item["native"]["peak_lag_M"] > 0 and item["public"]["peak_lag_M"] > 0
                for item in crossings),
            "finite_lags_and_metrics": bool(np.isfinite(lag).all() and all(
                np.isfinite(item["whole"]["complex_relative_l2"])
                and np.isfinite(item["local"]["complex_relative_l2"])
                for item in crossings)),
        },
    })
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"checks": result["checks"], "lag_range_M": result["lag_range_M"],
                      "rows": [{"target": item["target_Momega22"],
                                "lag_M": item["native_minus_public_peak_lag_M"],
                                "local_l2": item["local"]["complex_relative_l2"]}
                               for item in crossings]}), flush=True)
    if not result["diagnostic_checks_passed"]:
        raise ValueError("frozen frequency-lag replay failed")


if __name__ == "__main__":
    main()
