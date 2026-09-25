"""Dev-only finest q4 pinned-source, public, and native chart comparison."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import lal
import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, fitted_peak, public_mode
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_q4_fixed_anchor import fixed_anchor_metrics
from dev.native_rom.diagnose_source_wheel_mode import _public as direct_wheel_mode
from dev.native_rom.diagnose_source_wheel_mode import _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_finest_source_chart_plan.json"
OUTPUT = EVIDENCE / "source_q4_finest_source_chart.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    prior_path = EVIDENCE / "source_q4_fine_mode_control.json"
    fixed_path = EVIDENCE / "source_q4_fixed_anchor.json"
    source_plan_path = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"
    adapter_path = ROOT / "dev/native_rom/reference/full_v4_source_mode_adapter.c"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-finest-source-chart-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.0625, 0.03125]
            or plan["anchor_grid_M"] != 0.03125
            or q.sha256(prior_path) != plan["prior_source_control_sha256"]
            or q.sha256(fixed_path) != plan["fixed_anchor_control_sha256"]
            or q.sha256(source_plan_path) != plan["source_flux_plan_sha256"]
            or q.sha256(adapter_path) != plan["source_adapter_sha256"]):
        raise ValueError("frozen finest source chart plan changed")
    prior = json.loads(prior_path.read_text())
    fixed = json.loads(fixed_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    identity = _prepare_public(settings_plan)
    if (prior["status"] != "diagnostic_complete"
            or fixed["status"] != "diagnostic_complete"
            or prior["public_binary"] != identity
            or fixed["public_binary"] != identity):
        raise ValueError("pinned source or public control changed")
    historical = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/"
                             "combined_domain_numerics.json").read_text())
    settings = q.Settings(**historical["settings"])
    row = prior["row"]
    if row != fixed["row"]:
        raise ValueError("q4 source and fixed-anchor rows differ")
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-q4-finest-source-chart.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={}, anchors={}, adjacent={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    runs = {}
    with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-finest-") as temp:
        private = Path(temp)/"source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        library = Path(temp)/"source_q4_finest.dylib"
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
        for step in reversed(plan["output_steps_M"]):
            started = time.monotonic()
            epoch, delta_t, source_mode = _source(function, row, step, plan["capacity"])
            wheel_epoch, wheel_dt, wheel_mode, _ = direct_wheel_mode(row, step)
            if (abs(delta_t/mass_seconds-step) > 1e-10
                    or abs(wheel_dt/mass_seconds-step) > 1e-10):
                raise ValueError("source or wheel output cadence changed")
            source_time = epoch+np.arange(source_mode.size)*delta_t
            wheel_time = wheel_epoch+np.arange(wheel_mode.size)*wheel_dt
            if step == 0.0625:
                old = prior["grids"][str(step)]
                if (hashlib.sha256(source_mode.tobytes()).hexdigest()
                        != old["source"]["mode_sha256"]
                        or hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                        != old["direct_wheel"]["mode_sha256"]):
                    raise ValueError("previous pinned source/wheel q4 mode changed")
            if step == plan["anchor_grid_M"]:
                result["anchors"] = dict(
                    source=fitted_peak(source_time, source_mode, plan["peak_fit"]),
                    public=fitted_peak(wheel_time, wheel_mode, plan["peak_fit"]),
                    native=fixed["anchors"]["native"])
            anchors = result["anchors"]
            fitted = fitted_metrics(source_time, source_mode, wheel_time,
                                    wheel_mode, mass_seconds, plan["peak_fit"])
            grid = dict(
                source=dict(samples=int(source_mode.size), mode_sha256=hashlib.sha256(
                    source_mode.tobytes()).hexdigest()),
                direct_wheel=dict(samples=int(wheel_mode.size), mode_sha256=hashlib.sha256(
                    wheel_mode.tobytes()).hexdigest()),
                source_vs_public_fitted=fitted,
                source_vs_public_fixed=fixed_anchor_metrics(
                    source_time, source_mode, wheel_time, wheel_mode,
                    mass_seconds, anchors["source"], anchors["public"]),
                elapsed_s=time.monotonic()-started)
            if step == plan["anchor_grid_M"]:
                level = dict(name="source_finest_0.03125", dt_geom=step,
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
                    diagnostic_warm_start_Momega=0.03162277660168379)
                summary = q.summarize_run(run)
                if summary["array_sha256"] != fixed["grids"][str(step)]["native"]["array_sha256"]:
                    raise ValueError("native finest waveform changed")
                native_time, native_mode, public_time, public_mode_array, ms = public_mode(row, run)
                if abs(ms-mass_seconds) > 1e-15:
                    raise ValueError("public mass conversion changed")
                baseline = fixed["grids"][str(step)]["independently_fitted_l2"]
                native_public = fitted_metrics(native_time, native_mode,
                    public_time, public_mode_array, mass_seconds, plan["peak_fit"])
                if abs(native_public["metrics"]["complex_relative_l2"]-baseline) > 1e-12:
                    raise ValueError("native/public finest baseline changed")
                grid["native"] = summary
                grid["native_vs_source_fitted"] = fitted_metrics(
                    native_time, native_mode, source_time, source_mode,
                    mass_seconds, plan["peak_fit"])
                grid["native_vs_source_fixed"] = fixed_anchor_metrics(
                    native_time, native_mode, source_time, source_mode,
                    mass_seconds, anchors["native"], anchors["source"])
                grid["source_vs_polarization_public_fitted"] = fitted_metrics(
                    source_time, source_mode, public_time, public_mode_array,
                    mass_seconds, plan["peak_fit"])
                grid["wheel_vs_polarization_public_fitted"] = fitted_metrics(
                    wheel_time, wheel_mode, public_time, public_mode_array,
                    mass_seconds, plan["peak_fit"])
            result["grids"][str(step)] = grid
            runs[step] = (source_time, source_mode, wheel_time, wheel_mode)
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            print(f"grid={step:g} source/public={fitted['metrics']['complex_relative_l2']:.9g}",
                  flush=True)
    coarse, fine = plan["output_steps_M"]
    coarse_source_time, coarse_source_mode, coarse_public_time, coarse_public_mode = runs[coarse]
    fine_source_time, fine_source_mode, fine_public_time, fine_public_mode = runs[fine]
    anchors = result["anchors"]
    result["adjacent"] = dict(
        source=fixed_anchor_metrics(fine_source_time, fine_source_mode,
            coarse_source_time, coarse_source_mode, mass_seconds,
            anchors["source"], anchors["source"]),
        public=fixed_anchor_metrics(fine_public_time, fine_public_mode,
            coarse_public_time, coarse_public_mode, mass_seconds,
            anchors["public"], anchors["public"]))
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
