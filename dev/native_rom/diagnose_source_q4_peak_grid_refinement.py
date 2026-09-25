"""Dev-only q4 source-backed peak-grid refinement to 0.03125 M."""

import json
import os
import platform
import time
from pathlib import Path

import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, public_mode

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_peak_grid_refinement_plan.json"
OUTPUT = EVIDENCE / "source_q4_peak_grid_refinement.json"


def main():
    plan = json.loads(PLAN.read_text())
    prior_path = EVIDENCE / "source_q4_grid_self_control.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-peak-grid-refinement-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.125, 0.0625, 0.03125]
            or q.sha256(prior_path) != plan["grid_self_control_sha256"]):
        raise ValueError("frozen peak-grid refinement inputs changed")
    prior = json.loads(prior_path.read_text())
    historical = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/combined_domain_numerics.json").read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if prior["status"] != "diagnostic_complete" or identity != prior["public_binary"]:
        raise ValueError("pinned grid control or public binary changed")
    row = prior["row"]
    settings = q.Settings(**historical["settings"])
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-q4-peak-grid-refinement.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={}, adjacent={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    previous_run = None
    previous_public = None
    previous_step = None
    for step in plan["output_steps_M"]:
        started = time.monotonic()
        level = dict(name=f"grid_{step:g}", dt_geom=step,
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
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        if step == 0.125:
            frozen = prior["grids"]["0.125"]
            if (q.summarize_run(run)["array_sha256"] != frozen["native"]["array_sha256"]
                    or abs(fitted["metrics"]["complex_relative_l2"]
                           - frozen["fitted_public"]["metrics"]["complex_relative_l2"])
                        > 1e-12):
                raise ValueError("0.125 M baseline changed")
        result["grids"][str(step)] = dict(native=q.summarize_run(run),
                                           fitted_public=fitted,
                                           elapsed_s=time.monotonic()-started)
        if previous_run is not None:
            previous_time = previous_run["time"]*mass_seconds
            current_time = run["time"]*mass_seconds
            result["adjacent"][f"{previous_step:g}_to_{step:g}"] = dict(
                native_nested=q.compare_runs(previous_run, run),
                native_fitted=fitted_metrics(previous_time, previous_run["mode"],
                                               current_time, run["mode"],
                                               mass_seconds, plan["peak_fit"]),
                public_fitted=fitted_metrics(previous_public[0], previous_public[1],
                                               rt, rm, mass_seconds,
                                               plan["peak_fit"]),
            )
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(step, fitted["metrics"]["complex_relative_l2"], flush=True)
        previous_run = run
        previous_public = (rt, rm)
        previous_step = step
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
