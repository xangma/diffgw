"""Dev-only fixed-chart phase/amplitude accounting of stored cap-start modes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_cap_phase_amplitude_attribution_plan.json"
OUTPUT = EVIDENCE / "source_aligned_cap_phase_amplitude_attribution.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.cap-phase-amplitude-plan.v1"
            or plan["release_qualified"] is not False
            or plan["script_sha256"] != sha_file(Path(__file__))):
        raise ValueError("frozen phase/amplitude plan changed")
    for name, item in plan["inputs"].items():
        if sha_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen phase/amplitude input changed: {name}")
    return plan


def complex_pairs(values: list) -> np.ndarray:
    pairs = np.asarray(values, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("expected stored complex pairs")
    return pairs[:, 0] + 1j*pairs[:, 1]


def main() -> None:
    plan = checked_plan()
    v2 = json.loads((ROOT / plan["inputs"]["stored_modes_result"]["path"]).read_text())
    v4 = json.loads((ROOT / plan["inputs"]["full_crop_v4_result"]["path"]).read_text())
    v5 = json.loads((ROOT / plan["inputs"]["full_crop_v5_result"]["path"]).read_text())
    time = np.asarray(v2["chart_time_M"], dtype=np.float64)
    source_pairs = np.asarray(v2["source_aligned_mode"], dtype=np.float64)
    native_pairs = np.asarray(v2["corrected_aligned_mode"], dtype=np.float64)
    source = complex_pairs(v2["source_aligned_mode"])
    native = complex_pairs(v2["corrected_aligned_mode"])
    n = len(time)

    checks = {
        "v2_complete": v2["status"] == "diagnostic_complete"
            and v2["diagnostic_checks_passed"] is True
            and v2["release_qualified"] is False,
        "v4_historical_failure_preserved": v4["status"] == "failed_validity"
            and sorted(k for k, value in v4["validity_checks"].items() if not value)
                == ["cap_frequency"],
        "v5_diagnostic_replay": v5["status"] == "diagnostic_complete"
            and v5["validity_passed"] is True
            and v5["release_qualified"] is False,
        "fixed_full_chart_hash": v2["chart_sha256"]
            == v4["fixed_chart"]["time_sha256"]
            == v5["fixed_chart"]["time_sha256"]
            == plan["fixed_full_chart_sha256"],
        "fixed_native_anchor": v2["native_peak_anchor"]
            == v4["levels"]["finest"]["native_peak_anchor"]
            == v5["levels"]["finest"]["native_peak_anchor"]
            == plan["native_anchor"],
        "fixed_source_anchor": v4["fixed_chart"]["source_anchor"]
            == v5["fixed_chart"]["source_anchor"]
            == plan["source_anchor"],
        "stored_chart_hash": sha_array(time) == plan["stored_chart_sha256"],
        "stored_source_hash": sha_array(source_pairs)
            == plan["stored_source_pairs_sha256"],
        "stored_native_hash": sha_array(native_pairs)
            == plan["stored_native_pairs_sha256"],
        "stored_lengths": n == len(source) == len(native)
            == plan["stored_samples"],
        "stored_time_support": n > 2 and time[0] == plan["stored_support_M"][0]
            and time[-1] == plan["stored_support_M"][1]
            and np.max(abs(np.diff(time)-plan["step_M"]))
                <= plan["gates"]["step_max_abs_M"],
        "finite_nonzero_modes": bool(np.isfinite(source).all()
            and np.isfinite(native).all()
            and np.all(abs(source) > 0) and np.all(abs(native) > 0)),
        "original_window_counts": v2["window_samples"]
            == plan["original_window_samples"],
    }
    if not all(checks.values()):
        raise ValueError(f"stored control failed: {[k for k, v in checks.items() if not v]}")

    source_amplitude = np.abs(source)
    native_amplitude = np.abs(native)
    # Native and source retain their independent frozen peak anchors; no fit.
    phase_residual = np.unwrap(np.angle(native*np.conj(source)))
    centered_rate = (phase_residual[2:]-phase_residual[:-2]) / (time[2:]-time[:-2])
    direct_energy = np.abs(native-source)**2
    amplitude_energy = (native_amplitude-source_amplitude)**2
    phase_energy = 4*native_amplitude*source_amplitude*np.sin(phase_residual/2)**2
    identity_residual = direct_energy-amplitude_energy-phase_energy
    norm = float(np.sum(direct_energy))
    checks["exact_energy_identity"] = bool(
        abs(float(np.sum(identity_residual))) / norm
        <= plan["gates"]["energy_identity_relative_max"])
    checks["finite_phase_and_rate"] = bool(np.isfinite(phase_residual).all()
        and np.isfinite(centered_rate).all())
    checks["phase_reconstruction"] = bool(
        np.linalg.norm(native-native_amplitude*np.exp(
            1j*(np.angle(source)+phase_residual))) / np.linalg.norm(native)
        <= plan["gates"]["phase_reconstruction_relative_l2"])

    early_stop = plan["early_stop_index"]
    windows = {
        "full_pre_boundary": np.arange(n),
        "early_pre_match_minus_50M": np.arange(early_stop),
        "nqc_local_50M": np.arange(early_stop,
                                   early_stop+plan["original_window_samples"]["nqc_local_50M"]),
        "first_50M_early": np.arange(plan["first_50M_samples"]),
        "last_50M_early": np.arange(early_stop-plan["last_50M_samples"], early_stop),
    }
    for quarter, (start, stop) in enumerate(zip(plan["early_quartile_edges"][:-1],
                                                  plan["early_quartile_edges"][1:]), 1):
        windows[f"early_quartile_{quarter}"] = np.arange(start, stop)
    checks["window_partition"] = bool(
        early_stop == plan["original_window_samples"]["early_pre_match_minus_50M"]
        and plan["early_quartile_edges"][0] == 0
        and plan["early_quartile_edges"][-1] == early_stop
        and n == early_stop+plan["original_window_samples"]["nqc_local_50M"]
            +plan["original_window_samples"]["boundary_margin_1M"])

    measurements = {}
    for name, indices in windows.items():
        source_energy = float(np.sum(source_amplitude[indices]**2))
        complex_energy = float(np.sum(direct_energy[indices]))
        amp_energy = float(np.sum(amplitude_energy[indices]))
        ph_energy = float(np.sum(phase_energy[indices]))
        interior = indices[(indices > 0) & (indices < n-1)]-1
        phase = phase_residual[indices]
        rates = centered_rate[interior]
        measurements[name] = {
            "samples": len(indices),
            "support_M": [float(time[indices[0]]), float(time[indices[-1]])],
            "complex_relative_l2": float(np.sqrt(complex_energy/source_energy)),
            "amplitude_relative_l2": float(np.sqrt(amp_energy/source_energy)),
            "phase_relative_l2": float(np.sqrt(ph_energy/source_energy)),
            "amplitude_fraction_of_error_energy": amp_energy/complex_energy,
            "phase_fraction_of_error_energy": ph_energy/complex_energy,
            "fraction_of_full_pre_boundary_error_energy": complex_energy/norm,
            "phase_residual_first_rad": float(phase[0]),
            "phase_residual_last_rad": float(phase[-1]),
            "phase_residual_span_rad": float(phase[-1]-phase[0]),
            "phase_residual_max_abs_rad": float(np.max(abs(phase))),
            "phase_rate_centered_rms_rad_per_M": float(np.sqrt(np.mean(rates**2))),
            "phase_rate_centered_max_abs_rad_per_M": float(np.max(abs(rates))),
        }
    for name in ("early_pre_match_minus_50M", "nqc_local_50M"):
        replay = v2["corrected_vs_source"][name]["complex_relative_l2"]
        checks[f"{name}_metric_replay"] = (
            abs(measurements[name]["complex_relative_l2"]-replay)
            <= plan["gates"]["stored_metric_max_abs"])
    checks["full_pre_boundary_metric_replay"] = (
        abs(measurements["full_pre_boundary"]["complex_relative_l2"]
            -np.linalg.norm(native-source)/np.linalg.norm(source))
        <= plan["gates"]["stored_metric_max_abs"])

    result = {
        "schema": "diffgw.seobnrv4.cap-phase-amplitude-attribution.v1",
        "status": "diagnostic_complete" if all(checks.values()) else "failed_validity",
        "release_qualified": False,
        "plan_sha256": sha_file(PLAN),
        "script_sha256": sha_file(Path(__file__)),
        "checks": {k: bool(v) for k, v in checks.items()},
        "row_tuple": plan["row_tuple"],
        "chart": {
            "full_chart_sha256": v2["chart_sha256"],
            "stored_pre_boundary_sha256": sha_array(time),
            "samples": n,
            "support_M": [float(time[0]), float(time[-1])],
            "step_M": plan["step_M"],
            "native_anchor": v2["native_peak_anchor"],
            "source_anchor": v4["fixed_chart"]["source_anchor"],
        },
        "stored_mode_hashes": {
            "source_pairs_sha256": sha_array(source_pairs),
            "native_pairs_sha256": sha_array(native_pairs),
        },
        "energy_identity_relative_residual":
            float(abs(np.sum(identity_residual))/norm),
        "measurements": measurements,
        "interpretation": (
            "Exact algebraic decomposition |h_native-h_source|^2 = "
            "(|h_native|-|h_source|)^2 + 4|h_native||h_source| "
            "sin^2(delta_phase/2) on the stored v2 pre-attachment chart. "
            "Both modes keep their independently frozen peak anchors; phase is "
            "unwrapped for descriptive endpoint and centered-secant rate summaries "
            "only. No pairwise fit or source-gap causality claim."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "measurements": {k: {"complex_relative_l2":
                          v["complex_relative_l2"], "phase_fraction_of_error_energy":
                          v["phase_fraction_of_error_energy"]}
                          for k, v in measurements.items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
