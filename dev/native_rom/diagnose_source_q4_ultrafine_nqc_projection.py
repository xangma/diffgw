"""Dev-only pre-attachment source/wheel NQC coefficient counterfactual."""

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

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, fitted_peak
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_q4_fixed_anchor import fixed_anchor_metrics
from dev.native_rom.diagnose_source_wheel_mode import _public, _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_ultrafine_nqc_projection_plan.json"
OUTPUT = EVIDENCE / "source_q4_ultrafine_nqc_projection.json"


def _nqc_factor(low, query_time_M, coefficients):
    time_M, radius, phase, pr_star, _ = low
    omega = np.gradient(phase, time_M, edge_order=2)
    if (query_time_M[0] < time_M[0] or query_time_M[-1] > time_M[-1]
            or not np.all(np.diff(query_time_M) > 0)):
        raise ValueError("NQC query is outside the low-rate trajectory")
    radius = np.interp(query_time_M, time_M, radius)
    pr_star = np.interp(query_time_M, time_M, pr_star)
    omega = np.interp(query_time_M, time_M, omega)
    ratio = pr_star/(radius*omega)
    c = coefficients
    amplitude = 1+ratio**2*(c[0]+c[1]/radius+c[2]/radius**1.5)
    phase = c[6]*ratio+c[7]*ratio*pr_star**2
    return amplitude*np.exp(1j*phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    chart_path = EVIDENCE / "source_q4_ultrafine_chart.json"
    dynamics_path = EVIDENCE / "source_q4_ultrafine_dynamics.json"
    source_plan_path = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"
    adapter_path = ROOT / "dev/native_rom/reference/full_v4_source_mode_adapter.c"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-ultrafine-nqc-projection-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["pre_peak_cut_M"] != 20.0
            or plan["capacity"] != 50000
            or q.sha256(chart_path) != plan["ultrafine_chart_sha256"]
            or q.sha256(dynamics_path) != plan["dynamics_nqc_sha256"]
            or q.sha256(source_plan_path) != plan["source_flux_plan_sha256"]
            or q.sha256(adapter_path) != plan["source_mode_adapter_sha256"]):
        raise ValueError("frozen NQC projection inputs changed")
    chart = json.loads(chart_path.read_text())
    dynamics = json.loads(dynamics_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    identity = _prepare_public(settings_plan)
    if (chart["status"] != "diagnostic_complete"
            or dynamics["status"] != "diagnostic_complete"
            or chart["public_binary"] != identity
            or dynamics["public_binary"] != identity
            or chart["row"] != dynamics["row"]):
        raise ValueError("pinned q4 chart, dynamics, or public binary changed")
    row = chart["row"]
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    result = dict(schema="diffgw.seobnrv4.source-q4-ultrafine-nqc-projection.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-nqc-projection-") as temp:
        private = Path(temp)/"source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        library = Path(temp)/"source_q4_mode.dylib"
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
            source_epoch, source_dt, source_mode = _source(
                function, row, step, plan["capacity"])
            wheel_epoch, wheel_dt, wheel_mode, low = _public(row, step)
            control = chart["grids"][str(step)]
            nqc = dynamics["grids"][str(step)]["nqc"]
            if (hashlib.sha256(source_mode.tobytes()).hexdigest()
                    != control["source"]["mode_sha256"]
                    or hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                    != control["direct_wheel"]["mode_sha256"]
                    or hashlib.sha256(low.tobytes()).hexdigest()
                    != dynamics["grids"][str(step)]["wheel"]["array_sha256"]
                    or abs(source_dt/mass_seconds-step) > 1e-10
                    or abs(wheel_dt/mass_seconds-step) > 1e-10):
                raise ValueError("source, wheel, or low dynamics changed")
            source_time = source_epoch+np.arange(source_mode.size)*source_dt
            wheel_time = wheel_epoch+np.arange(wheel_mode.size)*wheel_dt
            full = fitted_metrics(source_time, source_mode, wheel_time,
                                  wheel_mode, mass_seconds, plan["peak_fit"])
            if (abs(full["metrics"]["complex_relative_l2"]
                    - control["source_vs_public_fitted"]["metrics"]["complex_relative_l2"])
                    > 1e-12):
                raise ValueError("source/wheel complete-mode baseline changed")
            source_peak = fitted_peak(source_time, source_mode, plan["peak_fit"])
            wheel_peak = fitted_peak(wheel_time, wheel_mode, plan["peak_fit"])
            source_mask = source_time < source_peak["time_s"]-plan["pre_peak_cut_M"]*mass_seconds
            wheel_mask = wheel_time < wheel_peak["time_s"]-plan["pre_peak_cut_M"]*mass_seconds
            orbital_omega = np.gradient(low[2], low[0], edge_order=2)
            orbital_peak_M = float(low[0, np.argmax(orbital_omega)])
            query_M = orbital_peak_M+(wheel_time[wheel_mask]-wheel_peak["time_s"])/mass_seconds
            source_factor = _nqc_factor(low, query_M, np.asarray(nqc["source"]))
            wheel_factor = _nqc_factor(low, query_M, np.asarray(nqc["wheel"]))
            predicted = wheel_mode[wheel_mask]*source_factor/wheel_factor
            peak_factor_ratio = (
                _nqc_factor(low, np.array([orbital_peak_M]), np.asarray(nqc["source"]))
                / _nqc_factor(low, np.array([orbital_peak_M]), np.asarray(nqc["wheel"]))
            )[0]
            predicted_peak = dict(wheel_peak)
            predicted_peak["phase_rad"] += float(np.angle(peak_factor_ratio))
            reference_time, reference_mode = source_time[source_mask], source_mode[source_mask]
            candidate_time, candidate_mode = wheel_time[wheel_mask], wheel_mode[wheel_mask]
            baseline = fixed_anchor_metrics(candidate_time, candidate_mode,
                reference_time, reference_mode, mass_seconds, wheel_peak, source_peak)
            projection = fixed_anchor_metrics(candidate_time, predicted,
                reference_time, reference_mode, mass_seconds, wheel_peak, source_peak)
            phase_adjusted_projection = fixed_anchor_metrics(candidate_time, predicted,
                reference_time, reference_mode, mass_seconds, predicted_peak, source_peak)
            result["grids"][str(step)] = dict(
                source_samples=int(source_mask.sum()), wheel_samples=int(wheel_mask.sum()),
                orbital_peak_M=orbital_peak_M,
                pre_attachment_baseline=baseline,
                coefficient_only_projection=projection,
                coefficient_and_peak_phase_projection=phase_adjusted_projection,
                coefficient_peak_phase_shift_rad=float(np.angle(peak_factor_ratio)),
                factor_ratio_amplitude_range=[float(np.min(abs(source_factor/wheel_factor))),
                                              float(np.max(abs(source_factor/wheel_factor)))],
                factor_ratio_phase_range_rad=[float(np.min(np.angle(source_factor/wheel_factor))),
                                              float(np.max(np.angle(source_factor/wheel_factor)))],
                elapsed_s=time.monotonic()-started)
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            print(f"grid={step:g} pre-baseline="
                  f"{baseline['metrics']['complex_relative_l2']:.9g} "
                  f"NQC-projection={projection['metrics']['complex_relative_l2']:.9g} "
                  f"peak-adjusted={phase_adjusted_projection['metrics']['complex_relative_l2']:.9g}",
                  flush=True)
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
