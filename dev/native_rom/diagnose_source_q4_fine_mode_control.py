"""Dev-only q4 high-cadence source, public, and native mode control."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, public_mode
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_mode import _public as direct_wheel_mode
from dev.native_rom.diagnose_source_wheel_mode import _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_fine_mode_control_plan.json"
OUTPUT = EVIDENCE / "source_q4_fine_mode_control.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    prior_path = EVIDENCE / "source_q4_peak_grid_refinement.json"
    source_plan_path = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"
    adapter_path = ROOT / "dev/native_rom/reference/full_v4_source_mode_adapter.c"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-fine-mode-control-plan.v2"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.125, 0.0625]
            or q.sha256(prior_path) != plan["peak_grid_refinement_sha256"]
            or q.sha256(source_plan_path) != plan["source_flux_plan_sha256"]
            or q.sha256(adapter_path) != plan["source_adapter_sha256"]):
        raise ValueError("frozen q4 source comparison inputs changed")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        raise ValueError("peak-grid refinement incomplete")
    source_plan = json.loads(source_plan_path.read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    identity = _prepare_public(settings_plan)
    if identity != prior["public_binary"]:
        raise ValueError("pinned public binary changed")
    historical = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/"
                             "combined_domain_numerics.json").read_text())
    settings = q.Settings(**historical["settings"])
    row = prior["row"]
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-q4-fine-mode-control.v2",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={}, adjacent_source={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    previous_source = None
    previous_step = None
    with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-fine-mode-") as temp:
        private = Path(temp) / "source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        library = Path(temp) / "source_q4_fine_mode.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), adapter_path, library,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        function = ctypes.CDLL(str(library)).diffgw_v4_full_source_mode
        function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        for step in plan["output_steps_M"]:
            started = time.monotonic()
            level = dict(name=f"source_fine_{step:g}", dt_geom=step,
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
            native = q.summarize_run(run)
            if native["array_sha256"] != prior["grids"][str(step)]["native"]["array_sha256"]:
                raise ValueError(f"native {step:g} M waveform changed")
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            epoch, delta_t, source_mode = _source(function, row, step, plan["capacity"])
            wheel_epoch, wheel_dt, wheel_mode, _ = direct_wheel_mode(row, step)
            if abs(delta_t/mass_seconds-step) > 1e-10:
                raise ValueError("source output cadence changed")
            if abs(wheel_dt/mass_seconds-step) > 1e-10:
                raise ValueError("direct public mode cadence changed")
            st = epoch + np.arange(source_mode.size)*delta_t
            wt = wheel_epoch + np.arange(wheel_mode.size)*wheel_dt
            method = plan["peak_fit"]
            comparisons = dict(
                native_vs_source=fitted_metrics(nt, nm, st, source_mode,
                                                mass_seconds, method),
                source_vs_public=fitted_metrics(st, source_mode, rt, rm,
                                                mass_seconds, method),
                native_vs_public=fitted_metrics(nt, nm, rt, rm,
                                                mass_seconds, method),
                source_vs_direct_wheel=fitted_metrics(
                    st, source_mode, wt, wheel_mode, mass_seconds, method),
                direct_wheel_vs_public=fitted_metrics(
                    wt, wheel_mode, rt, rm, mass_seconds, method),
            )
            baseline = prior["grids"][str(step)]["fitted_public"]["metrics"]
            if abs(comparisons["native_vs_public"]["metrics"]["complex_relative_l2"]
                   - baseline["complex_relative_l2"]) > 1e-12:
                raise ValueError("native/public baseline differs")
            result["grids"][str(step)] = dict(
                native=native,
                source=dict(samples=int(source_mode.size), epoch_s=epoch,
                            delta_t_s=delta_t,
                            mode_sha256=hashlib.sha256(source_mode.tobytes()).hexdigest()),
                direct_wheel=dict(samples=int(wheel_mode.size), epoch_s=wheel_epoch,
                                  delta_t_s=wheel_dt,
                                  mode_sha256=hashlib.sha256(wheel_mode.tobytes()).hexdigest()),
                comparisons=comparisons, elapsed_s=time.monotonic()-started)
            if previous_source is not None:
                result["adjacent_source"][f"{previous_step:g}_to_{step:g}"] = \
                    fitted_metrics(previous_source[0], previous_source[1],
                                   st, source_mode, mass_seconds, method)
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(step, {name: value["metrics"]["complex_relative_l2"]
                         for name, value in comparisons.items()}, flush=True)
            previous_source = (st, source_mode)
            previous_step = step
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
