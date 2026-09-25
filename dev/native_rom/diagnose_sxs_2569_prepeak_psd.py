"""Post-hoc 2569 pre-peak PSD attribution diagnostic; never a release gate."""

from __future__ import annotations

import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftfreq, next_fast_len

from . import compare_sxs_2123 as comparison_code
from . import compare_sxs_2123_public_control as public_code
from . import diagnose_sxs_cross_run_chirp as chirp_code
from . import measure_sxs_2508_weighted as weighted_code
from . import qualify_generator as q


ROOT = q.ROOT
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/sxs_2569_prepeak_psd_plan.json"
OUTPUT = ROOT / "dev/native_rom/evidence/2026-09-25/sxs_2569_prepeak_psd.json"


def _read_frozen(entry: dict) -> dict:
    path = ROOT / entry["path"]
    if q.sha256(path) != entry["sha256"]:
        raise ValueError(f"frozen input changed: {path}")
    return json.loads(path.read_text())


def _spectra(waves: dict, *, lower: float, end: float, dt: float,
             start_width: float, end_width: float) -> tuple:
    start = max(lower, *(float(w["time"][0]) for w in waves.values()))
    if end > min(float(w["time"][-1]) for w in waves.values()):
        raise ValueError("requested cutoff exceeds common support")
    grid = np.arange(math.ceil(start / dt), math.floor(end / dt) + 1) * dt
    if len(grid) < 1000 or grid[-1] - grid[0] <= 4 * max(start_width, end_width):
        raise ValueError("insufficient pre-peak settled support")
    # Exactly the original tanh factors, with only the end width varied.
    rise = (1 + np.tanh((grid - grid[0] - start_width / 2)
                        / (start_width / 10))) / 2
    fall = (1 + np.tanh((grid[-1] - end_width / 2 - grid)
                        / (end_width / 10))) / 2
    taper = rise * fall
    fft_length = next_fast_len(2 * len(grid))
    frequencies = fftfreq(fft_length, d=dt)
    positive = frequencies > 0
    spectra = {}
    for name, wave in waves.items():
        mode = wave["mode"]
        sampled = np.interp(grid, wave["time"], mode.real) + 1j * np.interp(
            grid, wave["time"], mode.imag)
        if not np.isfinite(sampled).all():
            raise ValueError(f"nonfinite sampled {name} waveform")
        spectra[name] = fft(np.conj(sampled) * taper, n=fft_length)[positive]
    return frequencies[positive], spectra, [float(grid[0]), float(grid[-1])], fft_length


def _residual(candidate: np.ndarray, reference: np.ndarray,
              frequencies: np.ndarray, psd: np.ndarray, match: dict) -> dict:
    """Exact radial/quadrature split of the complex, best-scaled residual."""
    weight = 1 / psd
    aligned = candidate * np.exp(-2j * np.pi * frequencies * match["best_lag_M"])
    alpha = np.sum(np.conj(aligned) * reference * weight) / np.sum(
        abs(aligned)**2 * weight)
    difference = reference - alpha * aligned
    norm = float(np.sum(abs(reference)**2 * weight))
    unit = np.ones_like(reference)
    nonzero = abs(reference) > 0
    unit[nonzero] = reference[nonzero] / abs(reference[nonzero])
    rotated = difference * np.conj(unit)
    radial = float(np.sum(rotated.real**2 * weight) / norm)
    quadrature = float(np.sum(rotated.imag**2 * weight) / norm)
    total = radial + quadrature
    expected = 1 - match["overlap"]**2
    if abs(total - expected) > 1e-10:
        raise ValueError("radial/quadrature residual does not close to match")
    return dict(radial_reference_power_fraction=radial,
                quadrature_reference_power_fraction=quadrature,
                total_reference_power_fraction=total,
                radial_fraction_of_residual=radial / total,
                quadrature_fraction_of_residual=quadrature / total,
                expected_one_minus_overlap_squared=expected,
                complex_scale_real=float(alpha.real),
                complex_scale_imag=float(alpha.imag))


def _measure(frequencies: np.ndarray, spectra: dict, *, lower_hz: float,
             mass_seconds: float, upper_hz: float, bounds: tuple,
             lag_step: float) -> dict:
    import lalsimulation as lalsim

    hz = frequencies / mass_seconds
    selected = (hz >= lower_hz) & (hz <= upper_hz)
    f = frequencies[selected]
    if len(f) < 8:
        raise ValueError("too few PSD bins")
    psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(value))
                    for value in hz[selected]])
    results = {}
    for label, candidate in (("native_vs_high_nr", "native"),
                             ("public_vs_high_nr", "public"),
                             ("low_vs_high_nr", "nr_low")):
        model = spectra[candidate][selected]
        reference = spectra["nr_high"][selected]
        match = weighted_code.optimized_overlap(
            f, model, reference, psd, bounds=bounds, step=lag_step)
        if match["time_shift_at_boundary"]:
            raise ValueError(f"lag optimization reached boundary: {label}")
        match["spectral_residual"] = _residual(model, reference, f, psd, match)
        results[label] = match
    return dict(frequency_bins=int(len(f)), lower_frequency_Hz=lower_hz,
                upper_frequency_Hz=upper_hz, results=results)


def main() -> None:
    started_utc = datetime.now(timezone.utc).isoformat()
    if OUTPUT.exists():
        raise ValueError(f"refusing to replace existing diagnostic: {OUTPUT}")
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2569-prepeak-psd-plan.v1"
            or plan["simulation"] != "SXS:BBH:2569"
            or plan["posthoc"] is not True or plan["release_qualified"] is not False
            or plan["total_mass_msun"] != 22
            or plan["primary_cutoff_mode_Momega"] != 0.09
            or plan["bracket_cutoff_mode_Momega"] != [0.08, 0.10]
            or plan["primary_end_taper_widths_M"] != [25, 50, 100]
            or plan["bracket_end_taper_width_M"] != 50
            or plan["start_taper_width_M"] != 50):
        raise ValueError("pre-peak diagnostic protocol changed")
    inputs = {name: _read_frozen(entry) for name, entry in plan["inputs"].items()}
    case_plan = inputs["case_plan"]
    comparison = inputs["comparison"]
    weighted_plan = inputs["weighted_plan"]
    weighted = inputs["weighted_result"]
    public_plan = inputs["public_plan"]
    public_prior = inputs["public_result"]
    chirp_plan = inputs["chirp_plan"]
    chirp_prior = inputs["chirp_result"]
    if (comparison["plan_sha256"] != plan["inputs"]["case_plan"]["sha256"]
            or weighted["plan_sha256"] != plan["inputs"]["weighted_plan"]["sha256"]
            or weighted["comparison_sha256"] != plan["inputs"]["comparison"]["sha256"]
            or public_prior["control_plan_sha256"] != plan["inputs"]["public_plan"]["sha256"]
            or chirp_prior["plan_sha256"] != plan["inputs"]["chirp_plan"]["sha256"]
            or weighted["cases"][plan["simulation"]]["selected_case_99_percent_passed"] is not False
            or chirp_prior["status"] != "diagnostic_complete"
            or weighted_plan["sample_step_M"] != 0.25
            or weighted_plan["taper_width_M"] != 50
            or weighted_plan["allowed_time_shift_M"] != [-20, 20]
            or weighted_plan["time_shift_grid_step_M"] != 0.25
            or weighted_plan["minimum_frequency_Hz"] != 10
            or weighted_plan["maximum_frequency_Hz"] != 3000
            or chirp_plan["smoothing_widths_M"] != [10, 20, 40]):
        raise ValueError("frozen comparison/PSD prerequisites changed")
    array_entry = plan["anchored_array"]
    array_path = Path(array_entry["path"])
    if q.sha256(array_path) != array_entry["sha256"]:
        raise ValueError("anchored waveform array changed")
    case = case_plan["cases"][plan["simulation"]]
    prior_case = comparison["cases"][plan["simulation"]]
    if prior_case["anchored_array"]["sha256"] != array_entry["sha256"]:
        raise ValueError("comparison used another anchored array")
    with np.load(array_path, allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"], mode=source[f"{name}_mode"])
                 for name in ("native", "nr_low", "nr_high")}
    q.lal_binary_identity(public_plan["binary_versions"]["lal"],
                          public_plan["binary_versions"]["lalsimulation"])
    dt = weighted_plan["sample_step_M"]
    public = comparison_code.anchor(
        public_code.public_mode(case["row"], dt, public_plan["public_call"]), dt=dt)
    for key, old in public_prior["anchors"]["public"].items():
        new = public[key]
        if isinstance(old, list):
            if not np.allclose(new, old, rtol=0, atol=1e-9):
                raise ValueError(f"public anchor changed: {key}")
        elif not math.isclose(new, old, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"public anchor changed: {key}")
    waves["public"] = public
    high_meta_entry = plan["nr_high_metadata"]
    metadata_path = Path(high_meta_entry["path"])
    if q.sha256(metadata_path) != high_meta_entry["sha256"]:
        raise ValueError("highest-resolution NR metadata changed")
    metadata = json.loads(metadata_path.read_text())
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    settled_start = (metadata["relaxation_time"]
                     - prior_case["anchor"]["Lev4"]["anchor_time_M"])
    if (not math.isclose(settled_start,
                         weighted["cases"][plan["simulation"]][
                             "nr_relaxed_start_after_anchor_M"], abs_tol=1e-9)
            or metadata["reference_time"] != metadata["relaxation_time"]):
        raise ValueError("settled support changed")
    from torchwave.constants import MSUN_SEC
    mass_seconds = plan["total_mass_msun"] * float(MSUN_SEC)
    lower_hz = max(weighted_plan["minimum_frequency_Hz"],
                   orbital / (math.pi * mass_seconds))
    bounds = tuple(weighted_plan["allowed_time_shift_M"])
    lag_step = weighted_plan["time_shift_grid_step_M"]
    upper_hz = weighted_plan["maximum_frequency_Hz"]

    # Full-waveform replay validates that the original PSD pipeline is unchanged.
    full_f, full_spectra, full_support, full_fft = weighted_code.tapered_spectra(
        waves, lower=settled_start, dt=dt, width=50)
    full = _measure(full_f, full_spectra, lower_hz=lower_hz,
                    mass_seconds=mass_seconds, upper_hz=upper_hz,
                    bounds=bounds, lag_step=lag_step)
    prior_full = next(row for row in weighted["cases"][plan["simulation"]]["primary"]["rows"]
                      if row["total_mass_msun"] == plan["total_mass_msun"])
    prior_public = next(row for row in public_prior["rows"]
                        if row["total_mass_msun"] == plan["total_mass_msun"])
    for name, old in (("native_vs_high_nr", prior_full["results"]["source_candidate_vs_nr"]),
                      ("low_vs_high_nr", prior_full["results"]["low_vs_high_nr"]),
                      ("public_vs_high_nr", prior_public["results"]["public_vs_nr"])):
        if abs(full["results"][name]["overlap"] - old["overlap"]) > 1e-10:
            raise ValueError(f"full-waveform replay differs: {name}")
    if (full_support != weighted["cases"][plan["simulation"]]["primary"]["common_time_support_M"]
            or full_fft != weighted["cases"][plan["simulation"]]["primary"]["fft_length"]):
        raise ValueError("full-waveform FFT support changed")

    # Fixed crossing chart uses high-resolution NR only, before inspecting results.
    chirp_times, _ = chirp_code._crossings(waves["nr_high"], chirp_plan, 20)
    frequencies = chirp_plan["mode_angular_frequencies_M_inv"]
    cutoffs = [0.08, 0.09, 0.10]
    cells = []
    for cutoff in cutoffs:
        crossing = float(chirp_times[frequencies.index(cutoff)])
        if crossing >= min(prior_case["anchor"]["Lev4"]["peak_after_anchor_M"],
                           prior_case["anchor"]["native"]["peak_after_anchor_M"],
                           public["peak_after_anchor_M"]):
            raise ValueError("cutoff is not pre-peak")
        widths = ([25, 50, 100] if cutoff == 0.09 else [50])
        for end_width in widths:
            f, spectra, support, fft_length = _spectra(
                waves, lower=settled_start, end=crossing, dt=dt,
                start_width=50, end_width=end_width)
            measurement = _measure(f, spectra, lower_hz=lower_hz,
                                   mass_seconds=mass_seconds, upper_hz=upper_hz,
                                   bounds=bounds, lag_step=lag_step)
            cells.append(dict(cutoff_mode_Momega=cutoff,
                              high_nr_crossing_time_after_anchor_M=crossing,
                              end_taper_width_M=end_width,
                              common_time_support_M=support,
                              fft_length=fft_length, **measurement))
    result = dict(schema="diffgw.seobnrv4.sxs-2569-prepeak-psd.v1",
                  status="diagnostic_complete", posthoc=True, release_qualified=False,
                  qualification_claim=False, dominant_mode_proxy=True,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  anchored_array_sha256=q.sha256(array_path),
                  nr_high_metadata_sha256=q.sha256(metadata_path),
                  total_mass_msun=plan["total_mass_msun"],
                  lower_frequency_Hz=lower_hz,
                  process=dict(host=platform.node(), cwd=str(Path.cwd()),
                               command="python -m dev.native_rom.diagnose_sxs_2569_prepeak_psd",
                               pid=os.getpid(),
                               log_path="/tmp/diffgw-sxs-2569-prepeak-psd.log",
                               expected_next_check="15 seconds after launch",
                               stop_command=f"kill {os.getpid()}",
                               started_utc=started_utc,
                               finished_utc=datetime.now(timezone.utc).isoformat()),
                  full_waveform_replay=dict(common_time_support_M=full_support,
                                            fft_length=full_fft, **full),
                  prepeak_cells=cells,
                  residual_interpretation="Exact Fourier-bin radial/quadrature split of the optimally time/phase/amplitude-aligned residual relative to the NR phasor. It does not uniquely identify time-domain amplitude, phase, NQC, or any physical cause; truncation/taper changes the frequency mixture.")
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    for cell in cells:
        print(cell["cutoff_mode_Momega"], cell["end_taper_width_M"],
              {name: value["overlap"] for name, value in cell["results"].items()},
              flush=True)


if __name__ == "__main__":
    main()
