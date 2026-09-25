"""Dev-only full-mode control against the separately compiled pinned source."""

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
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, public_mode
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_spliced_anchor import replay
from dev.native_rom.diagnose_source_wheel_mode import _source
from dev.native_rom.diagnose_source_wheel_dynamics import _source as _source_dynamics

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_full_mode_control_plan.json"
OUTPUT = EVIDENCE / "source_q4_full_mode_control.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    baseline_path = EVIDENCE / "source_q4_route_attribution.json"
    source_plan_path = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"
    adapter_path = ROOT / "dev/native_rom/reference/full_v4_source_mode_adapter.c"
    dynamics_adapter_path = ROOT / "dev/native_rom/reference/full_v4_source_dynamics_adapter.c"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-full-mode-control-plan.v1"
            or plan["release_qualified"] is not False
            or q.sha256(baseline_path) != plan["baseline_evidence_sha256"]
            or q.sha256(source_plan_path) != plan["source_flux_plan_sha256"]
            or q.sha256(adapter_path) != plan["source_adapter_sha256"]
            or q.sha256(dynamics_adapter_path) != plan["source_dynamics_adapter_sha256"]):
        raise ValueError("frozen source control inputs changed")
    baseline = json.loads(baseline_path.read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    source_plan = json.loads(source_plan_path.read_text())
    m1, m2, chi1, chi2, omega = settings_plan["case"]
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega)
    n = settings_plan["numerical_settings"]
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if identity != baseline["public_binary"]:
        raise ValueError("public binary changed")
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-q4-full-mode-control.v1",
                  status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
                  source_flux_plan_sha256=q.sha256(source_plan_path),
                  source_adapter_sha256=q.sha256(adapter_path),
                  public_binary=identity, host=platform.node(), cwd=os.getcwd(),
                  pid=os.getpid(), row=row)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    run = replay(row, n, return_dynamics=True)
    native = q.summarize_run({key: value for key, value in run.items()
                              if key != "coarse_dynamics"})
    if native["array_sha256"] != baseline["previous_route"]["array_sha256"]:
        raise ValueError("native q4 baseline changed")
    nt, nm, rt, rm, mass_seconds = public_mode(row, run)
    with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-full-mode-") as temp:
        private = Path(temp) / "source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        library = Path(temp) / "source_q4_full_mode.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), adapter_path, library,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        dynamics_library = Path(temp) / "source_q4_dynamics.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), dynamics_adapter_path,
                 dynamics_library, source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        function = ctypes.CDLL(str(library)).diffgw_v4_full_source_mode
        function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        epoch, step, source_mode = _source(function, row, plan["output_step_M"],
                                           plan["capacity"])
        dynamics_function = ctypes.CDLL(str(dynamics_library)).diffgw_v4_full_source_dynamics
        dynamics_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
        dynamics_function.restype = ctypes.c_int
        source_dynamics = _source_dynamics(dynamics_function, row,
                                           plan["output_step_M"], plan["capacity"])
    if abs(step/mass_seconds-plan["output_step_M"]) > 1e-10:
        raise ValueError("source output step changed")
    st = epoch + np.arange(source_mode.size)*step
    method = plan["peak_fit"]
    comparisons = dict(
        native_vs_source=fitted_metrics(nt, nm, st, source_mode, mass_seconds, method),
        source_vs_public=fitted_metrics(st, source_mode, rt, rm, mass_seconds, method),
        native_vs_public=fitted_metrics(nt, nm, rt, rm, mass_seconds, method),
    )
    coarse = run["coarse_dynamics"]
    source_radius = source_dynamics[1]
    if (not np.all(np.diff(source_radius) < 0)
            or not np.all(np.diff(coarse["state"][:, 0]) < 0)):
        raise ValueError("orbital radius is not monotone on the coarse prefix")
    native_start_radius = float(coarse["state"][0, 0])
    if not source_radius[-1] < native_start_radius < source_radius[0]:
        raise ValueError("source trajectory does not span native initial radius")
    def source_at_radius(radius):
        return np.array([np.interp(radius, source_radius[::-1], field[::-1])
                         for field in source_dynamics])
    source_start = source_at_radius(native_start_radius)
    source_omega = np.gradient(source_dynamics[2], source_dynamics[0], edge_order=2)
    native_omega = np.gradient(coarse["state"][:, 1], coarse["time"], edge_order=2)
    trajectory_points = []
    for time_M in plan["native_trajectory_times_M"]:
        index = round(time_M/plan["output_step_M"])
        radius = float(coarse["state"][index, 0])
        aligned_source = source_at_radius(radius)
        source_frequency = np.interp(radius, source_radius[::-1], source_omega[::-1])
        trajectory_points.append(dict(
            native_time_M=time_M, native_radius=radius,
            source_time_at_radius_M=float(aligned_source[0]),
            native_minus_source_pr_star=float(coarse["state"][index, 2]
                                              - aligned_source[3]),
            native_minus_source_pphi=float(coarse["state"][index, 3]
                                           - aligned_source[4]),
            native_minus_source_Momega=float(native_omega[index]-source_frequency),
            orbital_phase_span_difference_rad=float(
                coarse["state"][index, 1] - coarse["state"][0, 1]
                - (aligned_source[2]-source_start[2])),
            elapsed_time_difference_M=float(time_M -
                (aligned_source[0]-source_start[0]))))
    result.update(status="diagnostic_complete", native=native,
                  source=dict(samples=int(source_mode.size), epoch_s=epoch,
                              delta_t_s=step,
                              mode_sha256=hashlib.sha256(source_mode.tobytes()).hexdigest()),
                  comparisons=comparisons,
                  trajectory_alignment=dict(
                      native_initial_radius=native_start_radius,
                      source_initial_radius=float(source_radius[0]),
                      source_time_at_native_initial_radius_M=float(source_start[0])),
                  trajectory_points=trajectory_points,
                  elapsed_s=time.monotonic()-started)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: value["metrics"]["complex_relative_l2"]
                      for name, value in comparisons.items()}), flush=True)


if __name__ == "__main__":
    main()
