"""Selected-case, detector-weighted SEOBNRv4/SXS diagnostic; not a release gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftfreq, next_fast_len
from scipy.optimize import minimize_scalar

from . import compare_sxs_2123 as c
from . import qualify_generator as q


def tapered_spectra(waves: dict, *, lower: float, dt: float, width: float) -> tuple:
    start = max(lower, *(float(wave["time"][0]) for wave in waves.values()))
    end = min(float(wave["time"][-1]) for wave in waves.values())
    grid = np.arange(math.ceil(start / dt), math.floor(end / dt) + 1) * dt
    if len(grid) < 1000 or grid[-1] - grid[0] <= 4 * width:
        raise ValueError("insufficient shared settled waveform support")
    rise = (1 + np.tanh((grid - grid[0] - width / 2) / (width / 10))) / 2
    fall = (1 + np.tanh((grid[-1] - width / 2 - grid) / (width / 10))) / 2
    taper = rise * fall
    count = next_fast_len(2 * len(grid))
    frequency = fftfreq(count, d=dt)
    positive = frequency > 0
    spectra = {}
    for name, wave in waves.items():
        mode = wave["mode"]
        sampled = np.interp(grid, wave["time"], mode.real) + 1j * np.interp(
            grid, wave["time"], mode.imag)
        if not np.isfinite(sampled).all():
            raise ValueError(f"nonfinite interpolated {name} mode")
        # h_22 uses exp(-i phi); conjugation selects its positive-frequency carrier.
        spectra[name] = fft(np.conj(sampled) * taper, n=count)[positive]
    return frequency[positive], spectra, [float(grid[0]), float(grid[-1])], count


def optimized_overlap(frequency: np.ndarray, candidate: np.ndarray,
                      reference: np.ndarray, psd: np.ndarray,
                      *, bounds: tuple, step: float) -> dict:
    if (len(frequency) < 8 or candidate.shape != frequency.shape
            or reference.shape != frequency.shape or psd.shape != frequency.shape
            or not np.isfinite(psd).all() or np.any(psd <= 0)):
        raise ValueError("invalid weighted spectrum")
    weights = 1 / psd
    # A common positive-frequency bin width and factor four cancel in the match.
    cross = np.conj(candidate) * reference * weights
    norms = np.sqrt(np.sum(abs(candidate)**2 * weights)
                    * np.sum(abs(reference)**2 * weights))
    if not np.isfinite(norms) or norms <= 0:
        raise ValueError("invalid weighted waveform norms")
    def score(lag):
        return float(abs(np.sum(cross * np.exp(2j * np.pi * frequency * lag))) / norms)
    lags = np.arange(bounds[0], bounds[1] + step / 2, step)
    scores = np.array([score(lag) for lag in lags])
    best = int(np.argmax(scores))
    lower = max(bounds[0], float(lags[best] - step))
    upper = min(bounds[1], float(lags[best] + step))
    if lower == upper:
        lag = lower
    else:
        lag = float(minimize_scalar(lambda x: -score(x), bounds=(lower, upper),
                                    method="bounded", options={"xatol": 1e-7}).x)
    overlap = score(lag)
    if not 0 <= overlap <= 1 + 1e-10:
        raise ValueError("weighted overlap outside [0,1]")
    return dict(overlap=min(overlap, 1.), best_lag_M=lag,
                time_shift_at_boundary=bool(best in (0, len(lags) - 1)),
                grid_max_overlap=float(scores[best]), bins=len(frequency))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--long-plan", type=Path, required=True)
    parser.add_argument("--long-result", type=Path, required=True)
    parser.add_argument("--array", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.weighted_plan.read_text())
    cases = json.loads(args.case_plan.read_text())
    long = json.loads(args.long_result.read_text())
    simulation = "SXS:BBH:2508"
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2508-weighted-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["simulation"] != simulation
            or plan["long_result_sha256"] != q.sha256(args.long_result)
            or long["long_plan_sha256"] != q.sha256(args.long_plan)
            or long["plan_sha256"] != q.sha256(args.case_plan)
            or long["array_sha256"] != q.sha256(args.array)
            or not long["sources_unchanged"]):
        raise ValueError("weighted-diagnostic prerequisites changed")
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if q.sha256(extraction_path) != json.loads(
            (args.long_result.parent / "sxs_five_case_physical.json").read_text()
            )["extraction_sha256"]:
        raise ValueError("NR extraction manifest changed")
    case = cases["cases"][simulation]
    high = max(case["levels"])
    low = min(case["levels"])
    high_metadata = args.archive_dir / simulation.replace(":", "_") / f"Lev{high}" / "metadata.json"
    if q.sha256(high_metadata) != plan["nr_high_metadata_sha256"]:
        raise ValueError("highest-resolution NR metadata changed")
    metadata = json.loads(high_metadata.read_text())
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    if (metadata["relaxation_time"] != plan["nr_relaxation_time_M"]
            or metadata["reference_time"] != metadata["relaxation_time"]
            or not math.isclose(orbital, plan["nr_reference_orbital_MOmega"], abs_tol=1e-12)):
        raise ValueError("NR settled-frequency definition changed")
    record = extraction["cases"][simulation]["extracted"][f"Lev{low}"]
    nr_low_path = args.decoded_dir / record["path"]
    if q.sha256(nr_low_path) != record["sha256"]:
        raise ValueError("lower-resolution NR waveform changed")
    with np.load(args.array, allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"], mode=source[f"{name}_mode"])
                 for name in ("native", "public", "nr")}
    waves["nr_low"] = c.anchor(c.load_mode(nr_low_path),
                               dt=plan["sample_step_M"])
    relaxed_start = metadata["relaxation_time"] - long["anchors"]["nr"]["anchor_time_M"]
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC
    binary = q.lal_binary_identity(plan["binary_versions"]["lal"],
                                   plan["binary_versions"]["lalsimulation"])
    pairs = {"native_vs_nr": "native", "public_vs_nr": "public",
             "low_vs_high_nr": "nr_low"}
    masses = plan["total_masses_msun"]
    def sweep(width, selected_masses):
        f_geom, spectra, support, fft_length = tapered_spectra(
            waves, lower=relaxed_start, dt=plan["sample_step_M"], width=width)
        rows = []
        for mass in selected_masses:
            mass_seconds = mass * float(MSUN_SEC)
            hz = f_geom / mass_seconds
            lower_hz = max(plan["minimum_frequency_Hz"], orbital / (math.pi * mass_seconds))
            band = (hz >= lower_hz) & (hz <= plan["maximum_frequency_Hz"])
            frequency = f_geom[band]
            if len(frequency) < 8:
                raise ValueError(f"insufficient detector band at {mass} solar masses")
            psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(value))
                            for value in hz[band]])
            results = {name: optimized_overlap(
                frequency, spectra[candidate][band], spectra["nr"][band], psd,
                bounds=tuple(plan["allowed_time_shift_M"]),
                step=plan["time_shift_grid_step_M"])
                for name, candidate in pairs.items()}
            rows.append(dict(total_mass_msun=mass, lower_frequency_Hz=lower_hz,
                             upper_frequency_Hz=plan["maximum_frequency_Hz"],
                             frequency_bins=len(frequency), results=results))
        return dict(taper_width_M=width, common_time_support_M=support,
                    fft_length=fft_length, rows=rows)
    primary = sweep(plan["taper_width_M"], masses)
    controls = [sweep(width, plan["taper_control_masses_msun"])
                for width in plan["taper_control_widths_M"]]
    result = dict(schema="diffgw.seobnrv4.sxs-2508-weighted.v1",
                  weighted_plan_sha256=q.sha256(args.weighted_plan),
                  long_result_sha256=q.sha256(args.long_result),
                  script_sha256=q.sha256(__file__), array_sha256=q.sha256(args.array),
                  extraction_sha256=q.sha256(extraction_path),
                  nr_low_sha256=q.sha256(nr_low_path),
                  nr_high_metadata_sha256=q.sha256(high_metadata),
                  lal_binary=binary, nr_relaxed_start_after_anchor_M=relaxed_start,
                  nr_reference_mode_Momega=2 * orbital, primary=primary,
                  taper_controls=controls, detector_noise_weighted=True,
                  dominant_mode_proxy=True, posthoc_selected_case=True,
                  release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                        sort_keys=True) + "\n")
    print(json.dumps({name: min(row["results"][name]["overlap"]
                                for row in primary["rows"]) for name in pairs}))


if __name__ == "__main__":
    main()
