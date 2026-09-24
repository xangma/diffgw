"""Post-hoc sensitivity of the failed 2569 weighted proxy to mode resampling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftfreq, next_fast_len

from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q
from .diagnose_frequency_anchor_polar import polar_interpolate


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/sxs_2569_resampling_plan.json"


def polar_spectra(waves, *, lower, dt, width):
    """Repeat the frozen taper and FFT, changing only time interpolation."""
    start = max(lower, *(float(wave["time"][0]) for wave in waves.values()))
    end = min(float(wave["time"][-1]) for wave in waves.values())
    grid = np.arange(math.ceil(start / dt), math.floor(end / dt) + 1) * dt
    if len(grid) < 1000 or grid[-1] - grid[0] <= 4 * width:
        raise ValueError("insufficient common waveform support")
    rise = (1 + np.tanh((grid - grid[0] - width / 2) / (width / 10))) / 2
    fall = (1 + np.tanh((grid[-1] - width / 2 - grid) / (width / 10))) / 2
    taper = rise * fall
    count = next_fast_len(2 * len(grid))
    frequency = fftfreq(count, d=dt)
    positive = frequency > 0
    spectra = {}
    for name, wave in waves.items():
        sampled = polar_interpolate(grid, wave["time"], wave["mode"])
        if not np.isfinite(sampled).all():
            raise ValueError(f"nonfinite interpolated {name} mode")
        spectra[name] = fft(np.conj(sampled) * taper, n=count)[positive]
    return frequency[positive], spectra, [float(grid[0]), float(grid[-1])], count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    paths = {key: Path(value["path"]) for key, value in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2569-resampling-plan.v1"
            or plan["case"] != "SXS:BBH:2569" or plan["mass_msun"] != 22
            or plan["release_qualified"] is not False
            or any(q.sha256(paths[key]) != value["sha256"]
                   for key, value in plan["inputs"].items())):
        parser.error("frozen inputs changed")
    prior = json.loads(paths["weighted_result"].read_text())
    weighted_plan = json.loads(paths["weighted_plan"].read_text())
    comparison = json.loads(paths["comparison"].read_text())
    case = prior["cases"][plan["case"]]
    selected = next(row for row in case["primary"]["rows"]
                    if row["total_mass_msun"] == plan["mass_msun"])
    if (prior["all_cases_succeeded"] is not True
            or prior["all_selected_cases_99_percent_passed"] is not False
            or selected["results"]["source_candidate_vs_nr"]["overlap"] >= 0.99
            or comparison["cases"][plan["case"]]["anchored_array"]["sha256"]
               != plan["inputs"]["anchored_array"]["sha256"]):
        parser.error("failed selected-case proxy or anchored array differs")
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    pin = weighted_plan["binary_versions"]
    q.lal_binary_identity(pin["lal"], pin["lalsimulation"])
    with np.load(paths["anchored_array"], allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"], mode=source[f"{name}_mode"])
                 for name in ("native", "nr_low", "nr_high")}
    settings = dict(lower=case["nr_relaxed_start_after_anchor_M"],
                    dt=weighted_plan["sample_step_M"],
                    width=weighted_plan["taper_width_M"])
    mass_seconds = plan["mass_msun"] * float(MSUN_SEC)
    report = dict(schema="diffgw.seobnrv4.sxs-2569-resampling.v1",
                  release_qualified=False, posthoc_selected_case=True,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: value["sha256"] for key, value in plan["inputs"].items()},
                  mass_msun=plan["mass_msun"], rows={})
    for method, sampler in (("complex", weighted.tapered_spectra),
                            ("amplitude_phase", polar_spectra)):
        frequency, spectra, support, count = sampler(waves, **settings)
        frequency_hz = frequency / mass_seconds
        band = ((frequency_hz >= selected["lower_frequency_Hz"])
                & (frequency_hz <= weighted_plan["maximum_frequency_Hz"]))
        psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                        for f in frequency_hz[band]])
        result = {key: weighted.optimized_overlap(
            frequency[band], spectra[candidate][band], spectra["nr_high"][band], psd,
            bounds=tuple(weighted_plan["allowed_time_shift_M"]),
            step=weighted_plan["time_shift_grid_step_M"])
            for key, candidate in (("source_candidate_vs_nr", "native"),
                                   ("low_vs_high_nr", "nr_low"))}
        report["rows"][method] = dict(common_support_M=support, fft_length=count,
                                       frequency_bins=int(band.sum()), results=result)
    for key in ("source_candidate_vs_nr", "low_vs_high_nr"):
        old = selected["results"][key]["overlap"]
        if abs(report["rows"]["complex"]["results"][key]["overlap"] - old) > 1e-12:
            raise ValueError(f"original {key} overlap did not replay")
    report["threshold"] = weighted_plan["selected_case_99_percent_threshold"]
    report["amplitude_phase_passed_threshold"] = (
        report["rows"]["amplitude_phase"]["results"]["source_candidate_vs_nr"][
            "overlap"] >= report["threshold"])
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps(dict(methods={name: row["results"]["source_candidate_vs_nr"][
        "overlap"] for name, row in report["rows"].items()},
        amplitude_phase_passed_threshold=report["amplitude_phase_passed_threshold"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
