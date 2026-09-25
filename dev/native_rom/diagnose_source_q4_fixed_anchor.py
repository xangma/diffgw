"""Dev-only q4 grid comparison with one frozen peak chart per model."""

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, fitted_peak, public_mode

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_fixed_anchor_plan.json"
OUTPUT = EVIDENCE / "source_q4_fixed_anchor.json"


def fixed_anchor_metrics(candidate_time, candidate_mode, reference_time,
                         reference_mode, mass_seconds, candidate_anchor,
                         reference_anchor):
    """Compare without refitting time or phase on either input grid."""
    ct = np.asarray(candidate_time) - candidate_anchor["time_s"]
    rt = np.asarray(reference_time) - reference_anchor["time_s"]
    cm = np.asarray(candidate_mode) * np.exp(-1j * candidate_anchor["phase_rad"])
    rm = np.asarray(reference_mode) * np.exp(-1j * reference_anchor["phase_rad"])
    common = (rt >= ct[0]) & (rt <= ct[-1])
    t = rt[common]
    if t.size < 3:
        raise ValueError("too few common fixed-anchor samples")
    candidate = np.interp(t, ct, cm.real) + 1j * np.interp(t, ct, cm.imag)
    reference = rm[common]
    windows = {}
    for name, mask in (("inspiral", t < -20 * mass_seconds),
                       ("late_inspiral", (t >= -20 * mass_seconds) & (t < 0)),
                       ("post_peak", t >= 0)):
        if mask.sum() >= 3:
            windows[name] = dict(
                time_M=[float(t[mask][0] / mass_seconds),
                        float(t[mask][-1] / mass_seconds)],
                metrics=q.mode_metrics(candidate[mask], reference[mask]),
            )
    return dict(common_support_M=[float(t[0] / mass_seconds),
                                  float(t[-1] / mass_seconds)],
                metrics=q.mode_metrics(candidate, reference), windows=windows)


def main():
    plan = json.loads(PLAN.read_text())
    prior_path = EVIDENCE / "source_q4_peak_grid_refinement.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-fixed-anchor-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.125, 0.0625, 0.03125]
            or plan["anchor_grid_M"] != 0.03125
            or q.sha256(prior_path) != plan["peak_grid_refinement_sha256"]):
        raise ValueError("frozen fixed-anchor inputs changed")
    prior = json.loads(prior_path.read_text())
    historical = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/"
                             "combined_domain_numerics.json").read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if prior["status"] != "diagnostic_complete" or identity != prior["public_binary"]:
        raise ValueError("pinned q4 grid evidence or public binary changed")
    row = prior["row"]
    settings = q.Settings(**historical["settings"])
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-q4-fixed-anchor.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, anchors={}, grids={}, adjacent={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    runs = {}
    # The finest grid defines both models' coordinate charts; all coarse
    # comparisons below reuse these time/phase origins without optimization.
    for step in reversed(plan["output_steps_M"]):
        started = time.monotonic()
        level = dict(name=f"fixed_anchor_{step:g}", dt_geom=step,
                     atol=plan["atol"], rtol=plan["rtol"])
        run = q.native_run(
            row, level, settings,
            diagnostic_spin_truncation="tabulated",
            diagnostic_mode_21_higher_spin="rho_hybrid",
            diagnostic_mode_31_higher_spin="lal_v4_rho",
            diagnostic_mode_32_spin_square="lal_v4_zero",
            diagnostic_mode_33_higher_spin="lal_v4_rho",
            diagnostic_mode_43_higher_spin="lal_v4_rho",
            diagnostic_hamiltonian_log_derivative="lal_v4",
            diagnostic_hamiltonian_calibration="lal_v4_rounded",
            diagnostic_qnm_source="independent_kerr",
            diagnostic_hermite_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=165.0,
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
        )
        summary = q.summarize_run(run)
        if summary["array_sha256"] != prior["grids"][str(step)]["native"]["array_sha256"]:
            raise ValueError(f"native {step:g} M waveform changed")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        baseline = prior["grids"][str(step)]["fitted_public"]["metrics"]
        if abs(fitted["metrics"]["complex_relative_l2"]
               - baseline["complex_relative_l2"]) > 1e-12:
            raise ValueError(f"public {step:g} M waveform changed")
        if step == plan["anchor_grid_M"]:
            result["anchors"] = dict(
                native=fitted_peak(nt, nm, plan["peak_fit"]),
                public=fitted_peak(rt, rm, plan["peak_fit"]),
            )
        anchor = result["anchors"]
        fixed = fixed_anchor_metrics(nt, nm, rt, rm, mass_seconds,
                                    anchor["native"], anchor["public"])
        result["grids"][str(step)] = dict(
            native=summary, fixed_public=fixed,
            independently_fitted_l2=fitted["metrics"]["complex_relative_l2"],
            native_peak_drift_M=(fitted["native_peak"]["time_s"]
                                 - anchor["native"]["time_s"]) / mass_seconds,
            public_peak_drift_M=(fitted["reference_peak"]["time_s"]
                                 - anchor["public"]["time_s"]) / mass_seconds,
            elapsed_s=time.monotonic() - started,
        )
        runs[step] = (nt, nm, rt, rm)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"grid={step:g} fixed_l2={fixed['metrics']['complex_relative_l2']:.9g}", flush=True)
    for coarse, fine in zip(plan["output_steps_M"], plan["output_steps_M"][1:]):
        cn, cm, cp, cpm = runs[coarse]
        fn, fm, fp, fpm = runs[fine]
        anchor = result["anchors"]
        result["adjacent"][f"{coarse:g}_to_{fine:g}"] = dict(
            # Evaluate on the coarse samples so exact nested native outputs
            # have no complex-interpolation floor.
            native=fixed_anchor_metrics(fn, fm, cn, cm, mass_seconds,
                                        anchor["native"], anchor["native"]),
            public=fixed_anchor_metrics(fp, fpm, cp, cpm, mass_seconds,
                                        anchor["public"], anchor["public"]),
        )
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
