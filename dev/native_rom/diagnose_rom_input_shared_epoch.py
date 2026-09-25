"""Prospective, development-only ROM-input FFT epoch and shared-gauge check.

The default invocation runs only a cheap synthetic convention control. Physical
generator rows require the explicit ``--execute-physical`` flag after review.
No ROM is fitted and neither reserved ROM holdout is evaluated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_rom_input_stability import _support, deadline
from .eob.torch.conditioned_waveform_clean import conditioned_seobnrv4_fd
from .rom.conditioning import condition_modes_windowed


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/rom_input_shared_epoch_plan.json"


def _synthetic_control(plan: dict) -> dict:
    spec = plan["synthetic_control"]
    dt = spec["dt_seconds"]
    count = spec["retained_samples"]
    impulse = spec["impulse_index"]
    signal = np.zeros(count)
    signal[impulse] = 1.0
    epoch = -impulse * dt
    results = []
    for n_fft in spec["fft_samples"]:
        frequency = np.fft.rfftfreq(n_fft, dt)
        corrected = []
        for extra in (0, spec["extra_leading_zeros"]):
            row = np.pad(signal, (extra, 0))
            row_epoch = epoch - extra * dt
            padded = (np.pad(row, (n_fft - len(row), 0))
                      if n_fft >= len(row) else row[-n_fft:])
            sample_zero = row_epoch + (len(row) - n_fft) * dt
            spectrum = dt * np.fft.rfft(padded)
            physical = spectrum * np.exp(-2j * np.pi * frequency * sample_zero)
            corrected.append(physical)
            results.append({
                "fft_samples": n_fft,
                "extra_leading_zeros": extra,
                "fft_zero_epoch_seconds": sample_zero,
                "normalized_complex_error": float(np.max(abs(physical - dt)) / dt),
            })
        results.append({
            "fft_samples": n_fft,
            "leading_zero_invariance_error": float(
                np.max(abs(corrected[0] - corrected[1])) / dt
            ),
        })
    df = plan["delta_f_hz"]
    low, high = plan["comparison_band_hz"]
    frequency = np.arange(round(low / df), round(high / df) + 1) * df
    chart = frequency - plan["reference_frequency_hz"]
    base = np.exp(-0.00001 * chart**2 + 1j * (0.002 * chart + 0.00001 * chart**2))
    rotated = base * np.exp(1j * (0.6 + 2 * np.pi * 0.0007 * chart))
    width = plan["gauge_half_width_hz"]
    first = condition_modes_windowed(frequency, base, 100.0, half_width_hz=width)
    second = condition_modes_windowed(frequency, rotated, 100.0, half_width_hz=width)
    a = np.exp(first.log_amplitude + 1j * first.phase)
    b = np.exp(second.log_amplitude + 1j * second.phase)
    affine_error = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    numeric = [value for item in results for key, value in item.items()
               if key.endswith("error")]
    passed = (max(numeric) <= spec["maximum_normalized_complex_error"]
              and affine_error <= spec["maximum_affine_gauge_error"])
    return {"passed": passed, "impulse_controls": results,
            "affine_gauge_relative_l2": affine_error}


def _run(row: dict, level: dict, plan: dict, frequency: np.ndarray) -> dict:
    def generator(*args, **kwargs):
        from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive

        return reconstruct_seobnrv4_22_adaptive(
            *args, **kwargs, atol=plan["atol"], rtol=plan["rtol"]
        )

    start = time.monotonic()
    with deadline(plan["per_run_timeout_seconds"]), torch.no_grad():
        result = conditioned_seobnrv4_fd(
            row["mass1"], row["mass2"], chi1z=row["chi1"], chi2z=row["chi2"],
            f_low=row["f_low"], f_max=level, delta_f=plan["delta_f_hz"],
            distance=plan["distance_mpc"], inclination=plan["inclination_rad"],
            ringdown_duration=plan["ringdown_duration_M"],
            generator=generator, return_metadata=True,
        )
    full_frequency = result.frequencies.detach().cpu().numpy()
    indices = np.searchsorted(full_frequency, frequency)
    if not np.array_equal(full_frequency[indices], frequency):
        raise ValueError("FD frequency grid does not contain the frozen band")
    retained = int(result.fd_result.conditioning.h_plus.shape[-1])
    n_fft = int(result.sampling.n_time)
    dt = float(result.dt_seconds)
    epoch = float(result.physical_epoch)
    pad_crop_shift = (retained - n_fft) * dt
    fft_zero_epoch = epoch + pad_crop_shift
    pad_crop_rotation = np.exp(-2j * np.pi * frequency * pad_crop_shift)
    physical_epoch_rotation = np.exp(-2j * np.pi * frequency * epoch)
    spectra = {}
    for polarization, value in (("plus", result.h_plus), ("cross", result.h_cross)):
        raw = value.detach().cpu().numpy()[indices]
        pad_crop_only = raw * pad_crop_rotation
        corrected = pad_crop_only * physical_epoch_rotation
        if not all(np.isfinite(x).all() for x in (raw, pad_crop_only, corrected)):
            raise ValueError("nonfinite FD spectrum")
        spectra[polarization] = raw
        spectra[f"{polarization}_pad_crop_only"] = pad_crop_only
        spectra[f"{polarization}_physical_epoch"] = corrected
    return {
        "level_hz": level,
        "elapsed_seconds": time.monotonic() - start,
        "td_samples": int(result.td_time.numel()),
        "retained_samples": retained,
        "fft_samples": n_fft,
        "trailing_zero_samples": int(result.fd_result.conditioning.n_trailing_zeros),
        "dt_seconds": dt,
        "dt_geom": float(result.dt_geom),
        "physical_epoch_seconds": epoch,
        "pad_crop_shift_seconds": pad_crop_shift,
        "fft_zero_epoch_seconds": fft_zero_epoch,
        "omega_start": float(result.omega_start),
        **spectra,
    }


def _metric(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    norm = np.linalg.norm(second)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("zero or nonfinite comparison norm")
    phase_difference = np.unwrap(np.angle(first / second))
    return {
        "complex_relative_l2": float(np.linalg.norm(first - second) / norm),
        "amplitude_relative_l2": float(np.linalg.norm(abs(first) - abs(second)) / norm),
        "phase_rms_rad": float(np.sqrt(np.mean(phase_difference**2))),
    }


def _shared_gauge(run: dict, frequency: np.ndarray, plan: dict) -> dict[str, np.ndarray]:
    f_ref = plan["reference_frequency_hz"]
    plus = condition_modes_windowed(
        frequency, run["plus_physical_epoch"], f_ref,
        half_width_hz=plan["gauge_half_width_hz"],
    )
    affine = plus.removed_phase + plus.removed_phase_slope * (frequency - f_ref)
    rotation = np.exp(-1j * affine)
    return {
        "plus": run["plus_physical_epoch"] * rotation,
        "cross": run["cross_physical_epoch"] * rotation,
        "removed_plus_phase_rad": float(plus.removed_phase),
        "removed_plus_slope_rad_per_hz": float(plus.removed_phase_slope),
    }


def _pair(first: dict, second: dict, gauge_first: dict, gauge_second: dict,
          frequency: np.ndarray, plan: dict) -> dict:
    gates = plan["gates"]
    report = {
        "from_hz": first["level_hz"], "to_hz": second["level_hz"],
        "raw_ungauged": {}, "pad_crop_only_ungauged": {},
        "physical_epoch_ungauged": {},
        "shared_gauge": {}, "subbands": {},
    }
    for pol in ("plus", "cross"):
        report["raw_ungauged"][pol] = _metric(first[pol], second[pol])
        report["pad_crop_only_ungauged"][pol] = _metric(
            first[f"{pol}_pad_crop_only"], second[f"{pol}_pad_crop_only"]
        )
        report["physical_epoch_ungauged"][pol] = _metric(
            first[f"{pol}_physical_epoch"], second[f"{pol}_physical_epoch"]
        )
        result = _metric(gauge_first[pol], gauge_second[pol])
        report["shared_gauge"][pol] = result
        for low, high in plan["subbands_hz"]:
            selected = (frequency >= low) & (frequency <= high)
            label = f"{low:g}-{high:g}Hz"
            report["subbands"].setdefault(label, {})[pol] = _metric(
                gauge_first[pol][selected], gauge_second[pol][selected]
            )
    ratio_first = first["cross"] / first["plus"]
    ratio_second = second["cross"] / second["plus"]
    report["polarization_ratio_relative_l2"] = float(
        np.linalg.norm(ratio_first - ratio_second) / np.linalg.norm(ratio_second)
    )
    report["checks"] = {
        "whole_band": all(
            report["shared_gauge"][pol][key] <= gates[gate]
            for pol in ("plus", "cross")
            for key, gate in (
                ("complex_relative_l2", "maximum_adjacent_shared_gauge_complex_relative_l2"),
                ("amplitude_relative_l2", "maximum_adjacent_shared_gauge_amplitude_relative_l2"),
                ("phase_rms_rad", "maximum_adjacent_shared_gauge_phase_rms_rad"),
            )
        ),
        "all_subbands": all(
            metric["complex_relative_l2"] <= gates["maximum_subband_shared_gauge_complex_relative_l2"]
            for band in report["subbands"].values() for metric in band.values()
        ),
        "polarization_ratio": report["polarization_ratio_relative_l2"]
        <= gates["maximum_adjacent_polarization_ratio_relative_l2"],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--spectra-out", type=Path)
    parser.add_argument("--execute-physical", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.rom-input-shared-epoch-plan.v1"
            or plan["release_qualified"]
            or plan["reserved_holdouts_remain_unevaluated"]
            != ["off_grid_mixed_spin", "high_q_antialigned"]
            or len(plan["diagnostic_rows"]) != 2
            or any(row["name"] in plan["reserved_holdouts_remain_unevaluated"]
                   for row in plan["diagnostic_rows"])
            or len(plan["f_max_levels_hz"]) != 3
            or plan["gauge_half_width_hz"] != 2.0):
        parser.error("unexpected frozen development plan")
    prior = q.ROOT / "dev/native_rom/evidence/2026-09-25"
    conditioner = q.ROOT / "dev/native_rom/rom/conditioning.py"
    adapter = q.ROOT / "dev/native_rom/eob/torch/conditioned_waveform_clean.py"
    td_fd = q.ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py"
    source_hash = q.canonical_hash(q.source_snapshot())
    if (q.sha256(prior / "rom_input_windowed_gauge.json")
            != plan["prior_windowed_gauge_report_sha256"]
            or q.sha256(prior / "rom_input_windowed_gauge_spectra.npz")
            != plan["prior_windowed_gauge_spectra_sha256"]
            or source_hash != plan["source_snapshot_sha256"]
            or q.sha256(__file__) != plan["runner_sha256"]
            or q.sha256(conditioner) != plan["conditioner_sha256"]
            or q.sha256(adapter) != plan["fd_adapter_sha256"]
            or q.sha256(td_fd) != plan["td_fd_conditioner_sha256"]):
        parser.error("prior pilot, runner, conditioner, or source differs from the frozen plan")
    if args.execute_physical and args.spectra_out is None:
        parser.error("--execute-physical requires --spectra-out")
    df = plan["delta_f_hz"]
    low, high = plan["comparison_band_hz"]
    frequency = np.arange(round(low / df), round(high / df) + 1) * df
    if frequency[0] != low or frequency[-1] != high:
        parser.error("comparison band is not on the frozen grid")
    synthetic = _synthetic_control(plan)
    report = {
        "schema": "diffgw.seobnrv4.rom-input-shared-epoch.v1",
        "status": "synthetic_only", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "conditioner_sha256": q.sha256(conditioner),
        "fd_adapter_sha256": q.sha256(adapter),
        "td_fd_conditioner_sha256": q.sha256(td_fd),
        "source_snapshot_sha256": source_hash,
        "process": {"host": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid()},
        "synthetic_control": synthetic, "rows": [], "errors": [],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not synthetic["passed"] or not args.execute_physical:
        return 0 if synthetic["passed"] else 1
    spectra = {"frequency_hz": frequency}
    args.spectra_out.parent.mkdir(parents=True, exist_ok=True)
    report["status"] = "running"
    for row in plan["diagnostic_rows"]:
        item = {"name": row["name"], "levels": [], "pairs": [], "checks": {}}
        try:
            runs = []
            gauges = []
            for index, level in enumerate(plan["f_max_levels_hz"]):
                run = _run(row, level, plan, frequency)
                runs.append(run)
                gauges.append(_shared_gauge(run, frequency, plan))
                summary = {key: value for key, value in run.items()
                           if not isinstance(value, np.ndarray)}
                summary["shared_gauge_removed_plus_phase_rad"] = gauges[-1]["removed_plus_phase_rad"]
                summary["shared_gauge_removed_plus_slope_rad_per_hz"] = gauges[-1]["removed_plus_slope_rad_per_hz"]
                for pol in ("plus", "cross"):
                    for name in (pol, f"{pol}_pad_crop_only", f"{pol}_physical_epoch"):
                        signal = run[name]
                        spectra[f"{row['name']}_grid_{index}_{name}"] = signal
                        summary[f"{name}_sha256"] = hashlib.sha256(signal.tobytes()).hexdigest()
                    summary[f"{pol}_support"] = _support(
                        run[f"{pol}_physical_epoch"], plan["gates"]
                    )
                item["levels"].append(summary)
                np.savez(args.spectra_out, **spectra)
                args.json_out.write_text(json.dumps(report | {"rows": [*report["rows"], item]},
                                                    indent=2, sort_keys=True) + "\n")
                print(f"{row['name']} grid_{index}: {run['elapsed_seconds']:.1f}s", flush=True)
            for first, second, a, b in zip(runs, runs[1:], gauges, gauges[1:]):
                item["pairs"].append(_pair(first, second, a, b, frequency, plan))
            coarse, fine = item["pairs"]
            threshold = plan["gates"]["fine_to_coarse_shared_gauge_complex_error_ratio"]
            item["checks"] = {
                "pair_budgets": all(all(pair["checks"].values()) for pair in item["pairs"]),
                "support_and_unwrap": all(
                    all(level[f"{pol}_support"]["checks"].values())
                    for level in item["levels"] for pol in ("plus", "cross")
                ),
                "nonincreasing_fine_complex_error": all(
                    fine["shared_gauge"][pol]["complex_relative_l2"]
                    <= threshold * coarse["shared_gauge"][pol]["complex_relative_l2"] + 1e-12
                    for pol in ("plus", "cross")
                ),
            }
        except Exception as exc:
            report["errors"].append({"name": row["name"], "error": repr(exc)})
        report["rows"].append(item)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.spectra_out, **spectra)
    report["spectra_sha256"] = q.sha256(args.spectra_out)
    report["status"] = "measured" if not report["errors"] else "failed"
    report["all_checks_passed"] = (not report["errors"] and synthetic["passed"]
                                   and all(all(row["checks"].values()) for row in report["rows"]))
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
