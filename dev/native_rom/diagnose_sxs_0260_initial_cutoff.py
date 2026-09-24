"""Post-hoc initial-frequency and support sensitivity for frozen SXS 0260."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/sxs_0260_initial_cutoff_plan.json"
OUTPUT = q.ROOT / "dev/native_rom/evidence/2026-09-24/sxs_0260_initial_cutoff.json"


def main() -> None:
    if OUTPUT.exists():
        raise ValueError("output already exists")
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-0260-initial-cutoff-plan.v1"
            or plan["simulation"] != "SXS:BBH:0260"
            or plan["mass_msun"] != 22 or plan["release_qualified"] is not False
            or plan["weighted_function_sha256"] != q.sha256(weighted.__file__)):
        raise ValueError("unexpected frozen plan or scoring function")
    for key in ("comparison", "weighted", "weighted_plan", "array", "metadata"):
        if q.sha256(plan[f"{key}_path"]) != plan[f"{key}_sha256"]:
            raise ValueError(f"{key} changed after plan freeze")
    variants = [
        ("settled_cutoff_relaxed_support", "settled", "relaxed"),
        ("initial_cutoff_relaxed_support", "initial", "relaxed"),
        ("settled_cutoff_initial_support", "settled", "initial"),
        ("initial_cutoff_initial_support", "initial", "initial"),
    ]
    if [(x["name"], x["frequency"], x["support"]) for x in plan["variants"]] != variants:
        raise ValueError("variants changed")
    comparison = json.loads(Path(plan["comparison_path"]).read_text())
    original = json.loads(Path(plan["weighted_path"]).read_text())
    settings = json.loads(Path(plan["weighted_plan_path"]).read_text())
    metadata = json.loads(Path(plan["metadata_path"]).read_text())
    case = comparison["cases"][plan["simulation"]]
    baseline = original["cases"][plan["simulation"]]
    if (case["anchored_array"]["sha256"] != plan["array_sha256"]
            or baseline["array_sha256"] != plan["array_sha256"]
            or baseline["nr_high_metadata_sha256"] != plan["metadata_sha256"]):
        raise ValueError("selected waveforms differ from frozen results")
    with np.load(plan["array_path"], allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"], mode=source[f"{name}_mode"])
                 for name in ("native", "nr_low", "nr_high")}
    relaxed_start = (metadata["relaxation_time"]
                     - case["anchor"]["Lev5"]["anchor_time_M"])
    settled_orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    initial_orbital = float(metadata["initial_orbital_frequency"])
    if (metadata["reference_time"] != metadata["relaxation_time"]
            or not math.isclose(relaxed_start, baseline[
                "nr_relaxed_start_after_anchor_M"], abs_tol=1e-9)
            or not math.isclose(2 * settled_orbital, baseline[
                "nr_reference_mode_Momega"], abs_tol=1e-12)
            or initial_orbital <= 0):
        raise ValueError("NR metadata differs from frozen frequency definition")
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    binary = q.lal_binary_identity(settings["binary_versions"]["lal"],
                                   settings["binary_versions"]["lalsimulation"])
    mass_seconds = plan["mass_msun"] * float(MSUN_SEC)
    result = dict(schema="diffgw.seobnrv4.sxs-0260-initial-cutoff.v1",
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  lal_binary=binary, mass_msun=plan["mass_msun"],
                  initial_orbital_MOmega=initial_orbital,
                  settled_orbital_MOmega=settled_orbital,
                  relaxed_start_after_anchor_M=relaxed_start,
                  posthoc_selected_case=True, release_qualified=False,
                  variants={})
    for name, frequency_choice, support_choice in variants:
        lower = relaxed_start if support_choice == "relaxed" else min(
            float(wave["time"][0]) for wave in waves.values())
        frequency_geom, spectra, support, fft_length = weighted.tapered_spectra(
            waves, lower=lower, dt=settings["sample_step_M"],
            width=settings["taper_width_M"])
        cutoff_orbital = (settled_orbital if frequency_choice == "settled"
                          else initial_orbital)
        frequency_hz = frequency_geom / mass_seconds
        lower_hz = max(settings["minimum_frequency_Hz"],
                       cutoff_orbital / (math.pi * mass_seconds))
        band = ((frequency_hz >= lower_hz)
                & (frequency_hz <= settings["maximum_frequency_Hz"]))
        psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                        for f in frequency_hz[band]])
        pairs = {label: weighted.optimized_overlap(
            frequency_geom[band], spectra[candidate][band],
            spectra["nr_high"][band], psd,
            bounds=tuple(settings["allowed_time_shift_M"]),
            step=settings["time_shift_grid_step_M"])
            for label, candidate in (("native_vs_nr", "native"),
                                     ("low_vs_high_nr", "nr_low"))}
        result["variants"][name] = dict(
            common_time_support_M=support, fft_length=fft_length,
            lower_frequency_Hz=lower_hz, pairs=pairs)
    original_row = baseline["primary"]["rows"][plan["mass_msun"] - 10]
    first = result["variants"][variants[0][0]]
    if (original_row["total_mass_msun"] != plan["mass_msun"]
            or abs(first["pairs"]["native_vs_nr"]["overlap"]
                   - original_row["results"]["source_candidate_vs_nr"]["overlap"]) > 1e-12
            or abs(first["pairs"]["low_vs_high_nr"]["overlap"]
                   - original_row["results"]["low_vs_high_nr"]["overlap"]) > 1e-12):
        raise ValueError("did not reproduce frozen baseline")
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    for name, entry in result["variants"].items():
        print(name, entry["pairs"]["native_vs_nr"]["overlap"],
              entry["pairs"]["low_vs_high_nr"]["overlap"], flush=True)


if __name__ == "__main__":
    main()
