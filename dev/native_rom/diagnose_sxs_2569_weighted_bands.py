"""Post-hoc frequency attribution of the failed SXS:BBH:2569 holdout row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def decompose(frequency_geom, frequency_hz, candidate, reference, psd, edges,
              global_lag_M, *, bounds, step):
    """Partition the globally fitted, PSD-weighted complex residual."""
    weight = 1 / psd
    shifted = candidate * np.exp(-2j * np.pi * frequency_geom * global_lag_M)
    alpha = np.sum(np.conj(shifted) * reference * weight) / np.sum(
        abs(shifted) ** 2 * weight)
    reference_power = np.sum(abs(reference) ** 2 * weight)
    residual = abs(reference - alpha * shifted) ** 2 * weight
    total_residual = float(np.sum(residual))
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        band = (frequency_hz >= lower) & (frequency_hz < upper)
        if not np.any(band):
            raise ValueError(f"empty band {lower}:{upper}")
        rows.append(dict(
            lower_Hz=lower, upper_Hz=upper, bins=int(np.sum(band)),
            reference_power_fraction=float(np.sum(abs(reference[band]) ** 2
                                                  * weight[band]) / reference_power),
            global_fit_residual_fraction=float(np.sum(residual[band]) / total_residual),
            band_refit=weighted.optimized_overlap(
                frequency_geom[band], candidate[band], reference[band], psd[band],
                bounds=bounds, step=step),
        ))
    return dict(global_complex_scale_real=float(alpha.real),
                global_complex_scale_imag=float(alpha.imag),
                total_residual_over_reference_power=total_residual / reference_power,
                bands=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--weighted-result", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for key in ("comparison", "weighted_plan", "weighted_result"):
        if q.sha256(getattr(args, key)) != plan[f"{key}_sha256"]:
            raise ValueError(f"frozen {key} changed")
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2569-weighted-bands-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["simulation"] != "SXS:BBH:2569"
            or plan["mass_msun"] != 22
            or plan["frequency_edges_Hz"][0] != 0
            or plan["frequency_edges_Hz"][-1] != 3001):
        raise ValueError("invalid post-hoc plan")
    comparison = json.loads(args.comparison.read_text())
    previous = json.loads(args.weighted_result.read_text())
    weighted_plan = json.loads(args.weighted_plan.read_text())
    if (previous["plan_sha256"] != q.sha256(args.weighted_plan)
            or weighted_plan["selected_case_99_percent_threshold"] != .99):
        raise ValueError("weighted holdout linkage changed")
    simulation = plan["simulation"]
    record = comparison["cases"][simulation]["anchored_array"]
    path = Path(record["path"])
    if q.sha256(path) != record["sha256"]:
        raise ValueError("anchored waveform changed")
    with np.load(path, allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"],
                            mode=source[f"{name}_mode"])
                 for name in ("native", "nr_low", "nr_high")}
    old = previous["cases"][simulation]
    row = old["primary"]["rows"][plan["mass_msun"] - 10]
    if row["total_mass_msun"] != plan["mass_msun"]:
        raise ValueError("frozen mass row changed")
    from torchwave.constants import MSUN_SEC
    import lalsimulation as lalsim

    f_geom, spectra, support, fft_length = weighted.tapered_spectra(
        waves, lower=old["nr_relaxed_start_after_anchor_M"],
        dt=weighted_plan["sample_step_M"], width=weighted_plan["taper_width_M"])
    hz = f_geom / (plan["mass_msun"] * float(MSUN_SEC))
    full = (hz >= row["lower_frequency_Hz"]) & (hz <= row["upper_frequency_Hz"])
    hz, f_geom = hz[full], f_geom[full]
    psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f)) for f in hz])
    if (support != old["primary"]["common_time_support_M"]
            or fft_length != old["primary"]["fft_length"]
            or len(hz) != row["frequency_bins"]):
        raise ValueError("frozen spectrum changed")
    edges = [row["lower_frequency_Hz"]] + [x for x in plan["frequency_edges_Hz"]
                                            if row["lower_frequency_Hz"] < x <= 3001]
    result = dict(schema="diffgw.seobnrv4.sxs-2569-weighted-bands.v1",
                  qualification_claim=False, posthoc=True,
                  plan_sha256=q.sha256(args.plan), script_sha256=q.sha256(__file__),
                  anchored_array_sha256=record["sha256"],
                  lal_binary=q.lal_binary_identity("7.7.1", "6.2.1"),
                  simulation=simulation, mass_msun=plan["mass_msun"],
                  frequency_edges_Hz=edges, pairs={})
    for label, candidate in (("native_vs_nr_high", "native"),
                             ("nr_low_vs_nr_high", "nr_low")):
        measured = weighted.optimized_overlap(
            f_geom, spectra[candidate][full], spectra["nr_high"][full], psd,
            bounds=tuple(weighted_plan["allowed_time_shift_M"]),
            step=weighted_plan["time_shift_grid_step_M"])
        old_measured = row["results"]["source_candidate_vs_nr" if candidate == "native"
                                      else "low_vs_high_nr"]
        if abs(measured["overlap"] - old_measured["overlap"]) > 1e-10:
            raise ValueError("frozen full-band overlap changed")
        result["pairs"][label] = dict(full_band=measured, **decompose(
            f_geom, hz, spectra[candidate][full], spectra["nr_high"][full], psd,
            edges, measured["best_lag_M"],
            bounds=tuple(weighted_plan["allowed_time_shift_M"]),
            step=weighted_plan["time_shift_grid_step_M"]))
        decomposition = result["pairs"][label]
        if (abs(sum(b["reference_power_fraction"] for b in decomposition["bands"]) - 1) > 1e-10
                or abs(sum(b["global_fit_residual_fraction"] for b in decomposition["bands"]) - 1) > 1e-10
                or abs(decomposition["total_residual_over_reference_power"]
                       - (1 - measured["overlap"] ** 2)) > 1e-10):
            raise ValueError("weighted residual decomposition is inconsistent")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps({name: [(row["lower_Hz"], row["upper_Hz"],
                              row["global_fit_residual_fraction"])
                             for row in value["bands"]]
                      for name, value in result["pairs"].items()}))


if __name__ == "__main__":
    main()
