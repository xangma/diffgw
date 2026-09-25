"""Post-hoc, same-intrinsic-parameter NR control for the two failed q=3 rows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/sxs_cross_run_nr_plan.json"
OUTPUT = q.ROOT / "dev/native_rom/evidence/2026-09-25/sxs_cross_run_nr.json"


def _load_inputs(plan):
    inputs = {}
    for name, record in plan["inputs"].items():
        path = Path(record["path"])
        if q.sha256(path) != record["sha256"]:
            raise ValueError(f"frozen {name} changed")
        if name.endswith("_plan") or name.endswith("_comparison") or name.endswith("_weighted"):
            inputs[name] = json.loads(path.read_text())
        else:
            inputs[name] = path
    return inputs


def _case_inputs(label, plan, inputs):
    simulation = plan["cases"][label]
    case = inputs[label + "_plan"]["cases"][simulation]
    comparison = inputs[label + "_comparison"]["cases"][simulation]
    weighted_case = inputs[label + "_weighted"]["cases"][simulation]
    array = comparison["anchored_array"]
    if (array != plan["inputs"][label + "_array"]
            or weighted_case["array_sha256"] != array["sha256"]
            or weighted_case["selected_case_99_percent_passed"] is not False
            or abs(weighted_case["minimum_source_candidate_vs_nr"] - (
                0.9891417104357128 if label == "third" else 0.9880256014962968)) > 1e-12):
        raise ValueError(f"frozen {label} failed-gate evidence changed")
    with np.load(inputs[label + "_array"], allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{prefix}_time"],
                                mode=source[f"{prefix}_mode"])
                 for name, prefix in (("native", "native"), ("low", "nr_low"),
                                      ("high", "nr_high"))}
    if any(not np.all(np.diff(wave["time"]) > 0) for wave in waves.values()):
        raise ValueError(f"non-monotonic {label} NR time")
    return case, weighted_case, waves


def main() -> None:
    if OUTPUT.exists():
        raise ValueError("output already exists")
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-cross-run-nr-plan.v1"
            or plan["release_qualified"] is not False
            or plan["cases"] != {"third": "SXS:BBH:2569", "paper": "SXS:BBH:0260"}
            or plan["masses_msun"] != [22, 50, 100, 200]
            or plan["primary_mass_msun"] != 22
            or plan["primary_taper_width_M"] != 50
            or plan["taper_control_widths_M"] != [25, 100]
            or plan["sample_step_M"] != 0.25
            or plan["minimum_frequency_Hz"] != 10
            or plan["maximum_frequency_Hz"] != 3000
            or plan["allowed_time_shift_M"] != [-20, 20]
            or plan["time_shift_grid_step_M"] != 0.25):
        raise ValueError("frozen cross-run protocol changed")
    inputs = _load_inputs(plan)
    if inputs["scoring"] != Path(weighted.__file__).resolve():
        raise ValueError("scoring implementation changed")
    cases = {}
    waves = {}
    for label in ("third", "paper"):
        case, prior, pair = _case_inputs(label, plan, inputs)
        cases[label] = dict(simulation=plan["cases"][label],
                            row=case["row"],
                            orbital_M_inv=float(np.linalg.norm(
                                case["reference_orbital_frequency"])),
                            settled_start_after_anchor_M=prior[
                                "nr_relaxed_start_after_anchor_M"],
                            prior_native_minimum_overlap=prior[
                                "minimum_source_candidate_vs_nr"])
        waves.update({label + "_" + name: wave for name, wave in pair.items()})
    for key in ("chi1", "chi2"):
        if abs(cases["third"]["row"][key] - cases["paper"]["row"][key]) > 0.001:
            raise ValueError("cross-run spins differ too much for this control")
    if abs(cases["third"]["row"]["mass1"] / cases["third"]["row"]["mass2"]
           - cases["paper"]["row"]["mass1"] / cases["paper"]["row"]["mass2"]) > 0.001:
        raise ValueError("cross-run mass ratios differ too much for this control")
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    result = dict(schema="diffgw.seobnrv4.sxs-cross-run-nr.v1",
                  status="running", release_qualified=False, posthoc=True,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  lal_binary=q.lal_binary_identity("7.7.1", "6.2.1"),
                  cases=cases, comparisons={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lower = max(case["settled_start_after_anchor_M"] for case in cases.values())
    pairs = {"third_high_vs_paper_high": ("third_high", "paper_high"),
             "third_native_vs_third_high": ("third_native", "third_high"),
             "paper_native_vs_paper_high": ("paper_native", "paper_high"),
             "third_native_vs_paper_high": ("third_native", "paper_high"),
             "paper_native_vs_third_high": ("paper_native", "third_high"),
             "third_native_vs_paper_native": ("third_native", "paper_native"),
             "third_low_vs_paper_low": ("third_low", "paper_low"),
             "third_low_vs_third_high": ("third_low", "third_high"),
             "paper_low_vs_paper_high": ("paper_low", "paper_high")}
    for width in [plan["primary_taper_width_M"], *plan["taper_control_widths_M"]]:
        frequency_M, spectra, support, fft_length = weighted.tapered_spectra(
            waves, lower=lower, dt=plan["sample_step_M"], width=width)
        selected_masses = (plan["masses_msun"] if width == plan["primary_taper_width_M"]
                           else [plan["primary_mass_msun"]])
        entries = {}
        for mass in selected_masses:
            hz = frequency_M / (mass * float(MSUN_SEC))
            lower_hz = max(plan["minimum_frequency_Hz"], *(
                case["orbital_M_inv"] / (math.pi * mass * float(MSUN_SEC))
                for case in cases.values()))
            band = (hz >= lower_hz) & (hz <= plan["maximum_frequency_Hz"])
            if np.count_nonzero(band) < 8:
                raise ValueError("too few cross-run frequency bins")
            psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                            for f in hz[band]])
            measured = {name: weighted.optimized_overlap(
                frequency_M[band], spectra[candidate][band], spectra[reference][band],
                psd, bounds=tuple(plan["allowed_time_shift_M"]),
                step=plan["time_shift_grid_step_M"])
                for name, (candidate, reference) in pairs.items()}
            entries[str(mass)] = dict(lower_frequency_Hz=lower_hz,
                                     frequency_bins=int(np.count_nonzero(band)),
                                     pairs=measured)
        result["comparisons"][str(width)] = dict(common_support_M=support,
                                                  fft_length=fft_length,
                                                  masses=entries)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for width, entry in result["comparisons"].items():
        row = entry["masses"]["22"]["pairs"]
        print(width, row["third_high_vs_paper_high"]["overlap"], flush=True)


if __name__ == "__main__":
    main()
