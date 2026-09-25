"""Dev-only q4 native and public self-comparison across output grids."""

import json
import os
import platform
import time
from pathlib import Path

import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, public_mode
from torchwave.constants import MSUN_SEC

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_grid_self_control_plan.json"
OUTPUT = EVIDENCE / "source_q4_grid_self_control.json"


def main():
    plan = json.loads(PLAN.read_text())
    cross_path = EVIDENCE / "source_q4_numerical_crosscheck.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-grid-self-control-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.5, 0.125]
            or q.sha256(cross_path) != plan["numerical_crosscheck_sha256"]):
        raise ValueError("frozen grid self-control inputs changed")
    cross = json.loads(cross_path.read_text())
    high = json.loads((EVIDENCE / "source_q4_historical_numerics.json").read_text())
    historical = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/combined_domain_numerics.json").read_text())
    if cross["status"] != "diagnostic_complete" or high["status"] != "diagnostic_complete":
        raise ValueError("prerequisite grid evidence incomplete")
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if identity != cross["public_binary"] or identity != high["public_binary"]:
        raise ValueError("pinned public binary changed")
    torch.set_num_threads(1)
    row = high["row"]
    settings = q.Settings(**historical["settings"])
    result = dict(schema="diffgw.seobnrv4.source-q4-grid-self-control.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    runs = {}
    public = {}
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
        metric = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        expected = (cross["interventions"]["coarse_grid_full_numerics"]
                    if step == 0.5 else high)
        if (q.summarize_run(run)["array_sha256"] != expected["native"]["array_sha256"]
                or abs(metric["metrics"]["complex_relative_l2"]
                       - expected["fitted_public"]["metrics"]["complex_relative_l2"])
                    > 1e-12):
            raise ValueError(f"grid {step:g} failed its frozen replay")
        runs[step] = run
        public[step] = (rt, rm)
        result["grids"][str(step)] = dict(native=q.summarize_run(run),
                                           fitted_public=metric,
                                           elapsed_s=time.monotonic()-started)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"grid={step:g}", flush=True)
    coarse, fine = plan["output_steps_M"]
    mass_seconds = (row["mass1"]+row["mass2"])*float(MSUN_SEC)
    result["self_comparisons"] = dict(
        native_nested=q.compare_runs(runs[coarse], runs[fine]),
        native_fitted=fitted_metrics(runs[coarse]["time"]*mass_seconds,
                                      runs[coarse]["mode"],
                                      runs[fine]["time"]*mass_seconds,
                                      runs[fine]["mode"], mass_seconds,
                                      plan["peak_fit"]),
        public_fitted=fitted_metrics(public[coarse][0], public[coarse][1],
                                      public[fine][0], public[fine][1],
                                      mass_seconds, plan["peak_fit"]),
    )
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v.get("metrics", v)["complex_relative_l2"]
                      for k, v in result["self_comparisons"].items()}), flush=True)


if __name__ == "__main__":
    main()
