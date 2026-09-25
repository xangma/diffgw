"""Development-only, common-chart attribution of an aligned v4 public gap."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.compare_whole_waveforms import aligned_metrics
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, public_mode
from dev.native_rom.diagnose_q8_frequency_anchor import frequency_anchor
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_ode_chain_ringdown_inputs import _mode
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from torchwave.approximants.eob import numerical_attachment as attachment


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/source_aligned_public_gap_plan.json"
OUTPUT = ROOT / "dev/native_rom/evidence/2026-09-25/source_aligned_public_gap.json"


def _sha_array(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _metrics(candidate, reference, mask, total_error, total_reference):
    candidate, reference = candidate[mask], reference[mask]
    numerator = float(np.vdot(candidate-reference, candidate-reference).real)
    denominator = float(np.vdot(reference, reference).real)
    return {
        "samples": int(mask.sum()),
        "complex_relative_l2": float(np.sqrt(numerator/denominator)) if denominator else None,
        "error_squared": numerator,
        "reference_squared": denominator,
        "share_of_whole_error_squared": numerator/total_error,
        "share_of_whole_reference_squared": denominator/total_reference,
    }


def _common_chart(nt, nm, rt, rm, mass_seconds, native_anchor, public_anchor):
    nt = (nt-native_anchor[0])/mass_seconds
    rt = (rt-public_anchor[0])/mass_seconds
    nm = nm*np.exp(-1j*native_anchor[1])
    rm = rm*np.exp(-1j*public_anchor[1])
    common = (rt >= nt[0]) & (rt <= nt[-1])
    t = rt[common]
    candidate = np.interp(t, nt, nm.real)+1j*np.interp(t, nt, nm.imag)
    return t, candidate, rm[common], common


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-public-gap-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["level"] != "refined"
            or plan["pre_match_width_M"] != 50.0
            or plan["anchor_frequency_M"] != 0.1
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen public-gap plan changed")
    for key, item in plan["inputs"].items():
        if q.sha256(ROOT/item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {key}")
    boundary_plan = json.loads((ROOT/plan["inputs"]["boundary_plan"]["path"]).read_text())
    boundary_result = json.loads((ROOT/plan["inputs"]["boundary_result"]["path"]).read_text())
    source_plan = json.loads((ROOT/boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT/boundary_plan["inputs"]["ladder_plan"]["path"]).read_text())
    v2 = json.loads((ROOT/boundary_plan["inputs"]["amplitude_v2_result"]["path"]).read_text())
    identity = driver._prepare_public(source_plan)
    level = next(x for x in ladder["levels"] if x["name"] == plan["level"])
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
    result = {"schema": "diffgw.seobnrv4.source-aligned-public-gap.v1",
              "status": "running", "release_qualified": False,
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
              "checks": {}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    diagnostic, series, stage, inputs, controls = _capture_level(
        plan["level"], settings, source_plan, identity, boundary_plan,
        _source_field(), hamiltonian, flux,
        attachment._full_series_nqc, attachment._eval_batch_scalar)
    time_grid, candidate_mode = map(np.asarray, series)
    frozen = boundary_result["levels"][plan["level"]]
    candidate_replay = _series_summary(series) == frozen["candidate_torch"]
    boundary = float(stage["boundary_time_M"])
    ring_index = np.flatnonzero(time_grid >= boundary)
    chart_index = boundary_plan["fine_refinement_factor"]*(ring_index-int(stage["restart_index"]))
    original_ring = _mode(inputs, original_ringdown=attachment.merger_ringdown_22,
                          chart_index=chart_index)
    noop_mode = candidate_mode.copy()
    noop_mode[ring_index] = original_ring
    noop_replay = _series_summary((time_grid, noop_mode)) == v2["levels"][plan["level"]]["torch"]
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega_start)
    run = {"time": time_grid, "mode": candidate_mode,
           "peak_time_M": float(stage["orbital_peak_time_M"]),
           "dt_geom": settings["coarse_output_step_M"]}
    nt, nm, rt, rm, mass_seconds = public_mode(row, run)
    anchor_metric = aligned_metrics(nt, nm, rt, rm, mass_seconds)
    reported_metric = diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"]
    peak_replay = abs(anchor_metric["metrics"]["complex_relative_l2"]-reported_metric) <= 1e-12
    ni, ri = np.argmax(abs(nm)), np.argmax(abs(rm))
    native_peak_time_M = float(nt[ni]/mass_seconds+stage["orbital_peak_time_M"])
    t, candidate, reference, common = _common_chart(
        nt, nm, rt, rm, mass_seconds,
        (nt[ni], np.angle(nm[ni])), (rt[ri], np.angle(rm[ri])))
    full_error = float(np.vdot(candidate-reference, candidate-reference).real)
    full_reference = float(np.vdot(reference, reference).real)
    match = float(stage["match_time_M"])
    cut_early = match-plan["pre_match_width_M"]-native_peak_time_M
    cut_boundary = boundary-native_peak_time_M
    partition = {
        "early_pre_match": t < cut_early,
        "nqc_and_plunge": (t >= cut_early) & (t < cut_boundary),
        "ringdown": t >= cut_boundary,
    }
    if not np.all(sum(mask.astype(int) for mask in partition.values()) == 1):
        raise ValueError("public common-support windows do not partition")
    windows = {name: _metrics(candidate, reference, mask, full_error, full_reference)
               for name, mask in partition.items()}
    windows["late_20_M_before_public_peak"] = _metrics(
        candidate, reference, (t >= -20) & (t < 0), full_error, full_reference)
    windows["post_public_peak"] = _metrics(
        candidate, reference, t >= 0, full_error, full_reference)
    # A separately anchored frequency chart diagnoses sensitivity to the peak-time gauge.
    na = frequency_anchor(nt, nm, mass_seconds, plan["anchor_frequency_M"], nt[ni])
    ra = frequency_anchor(rt, rm, mass_seconds, plan["anchor_frequency_M"], rt[ri])
    ft, fc, fr, _ = _common_chart(
        nt, nm, rt, rm, mass_seconds,
        (na["time_s"], na["phase_rad"]), (ra["time_s"], ra["phase_rad"]))
    frequency_metric = float(np.linalg.norm(fc-fr)/np.linalg.norm(fr))
    peak_fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, "quadratic_power")
    fine_time = np.asarray(stage["fine_time"])
    fine_raw = np.asarray(stage["fine_base_mode"])
    fine_corrected = np.asarray(stage["fine_corrected_inspiral"])
    fine_mask = (fine_time >= match-plan["pre_match_width_M"]) & (fine_time < boundary)
    nqc_relative_change = float(np.linalg.norm(fine_corrected[fine_mask]-fine_raw[fine_mask])
                                / np.linalg.norm(fine_corrected[fine_mask]))
    result.update({
        "status": "diagnostic_complete", "elapsed_s": time.monotonic()-started,
        "identity": identity, "settings": settings,
        "candidate": frozen["candidate_torch"],
        "public_mode_sha256": _sha_array(rm),
        "public_mode_samples": len(rm),
        "common_peak_chart_samples": int(common.sum()),
        "common_peak_chart_M": [float(t[0]), float(t[-1])],
        "native_peak_time_M": native_peak_time_M,
        "public_peak_time_M": float(rt[ri]/mass_seconds),
        "native_match_relative_to_peak_M": match-native_peak_time_M,
        "native_boundary_relative_to_peak_M": cut_boundary,
        "whole_complex_relative_l2": float(np.sqrt(full_error/full_reference)),
        "frequency_anchor_complex_relative_l2": frequency_metric,
        "quadratic_power_peak_fit_complex_relative_l2": peak_fitted["metrics"]["complex_relative_l2"],
        "frequency_anchors": {"native": na, "public": ra},
        "fine_nqc_window": {"samples": int(fine_mask.sum()),
                            "raw_to_corrected_relative_l2": nqc_relative_change},
        "peak_chart_windows": windows,
        "checks": {
            "candidate_mode_hash_replays": candidate_replay,
            "v2_noop_mode_hash_replays": noop_replay,
            "public_peak_metric_replays": peak_replay,
            "peak_chart_whole_metric_replays": abs(np.sqrt(full_error/full_reference)-reported_metric) <= 1e-12,
            "partition_error_squared_replays": abs(sum(x["error_squared"] for x in list(windows.values())[:3])-full_error) <= 1e-12,
            "partition_reference_squared_replays": abs(sum(x["reference_squared"] for x in list(windows.values())[:3])-full_reference) <= 1e-12,
            "public_binary_identity_replays": identity == diagnostic["public_binary"],
        },
    })
    result["checks"] = {key: bool(value) for key, value in result["checks"].items()}
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"checks": result["checks"], "windows": windows,
                      "whole": result["whole_complex_relative_l2"],
                      "frequency_anchor": frequency_metric}), flush=True)
    if not result["diagnostic_checks_passed"]:
        raise ValueError("public-gap diagnostic replay failed")


if __name__ == "__main__":
    main()
