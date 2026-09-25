"""Dev-only causal NQC override in the pinned full v4 source generator."""

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
from dev.native_rom.diagnose_source_nqc_inputs import INCLUDE
from dev.native_rom.diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from dev.native_rom.diagnose_source_q4_fixed_anchor import fixed_anchor_metrics
from dev.native_rom.diagnose_source_wheel_mode import _public, _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_ultrafine_nqc_override_plan.json"
OUTPUT = EVIDENCE / "source_q4_ultrafine_nqc_override.json"


def _pre_peak(candidate, reference, mass_seconds, cut_M, fit):
    ct, cm = candidate
    rt, rm = reference
    cp = fitted_peak(ct, cm, fit)
    rp = fitted_peak(rt, rm, fit)
    cmask = ct < cp["time_s"]-cut_M*mass_seconds
    rmask = rt < rp["time_s"]-cut_M*mass_seconds
    return fixed_anchor_metrics(ct[cmask], cm[cmask], rt[rmask], rm[rmask],
                                mass_seconds, cp, rp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    paths = {
        "chart": EVIDENCE / "source_q4_ultrafine_chart.json",
        "dynamics": EVIDENCE / "source_q4_ultrafine_dynamics.json",
        "source_plan": ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json",
        "adapter": ROOT / "dev/native_rom/reference/full_v4_source_large_nqc_override_adapter.c",
    }
    hash_keys = {"chart": "ultrafine_chart_sha256",
                 "dynamics": "dynamics_nqc_sha256",
                 "source_plan": "source_flux_plan_sha256",
                 "adapter": "adapter_sha256"}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-ultrafine-nqc-override-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["capacity"] != 50000 or plan["pre_peak_cut_M"] != 20.0
            or any(q.sha256(path) != plan[hash_keys[name]] for name, path in paths.items())):
        raise ValueError("frozen causal NQC override inputs changed")
    chart = json.loads(paths["chart"].read_text())
    dynamics = json.loads(paths["dynamics"].read_text())
    source_plan = json.loads(paths["source_plan"].read_text())
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
    result = dict(schema="diffgw.seobnrv4.source-q4-ultrafine-nqc-override.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  instrumentation_sha256=hashlib.sha256(OVERRIDE.encode()).hexdigest(),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-nqc-override-") as temp:
        private = Path(temp)/"source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        source_file = private/"lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
        original = source_file.read_text()
        if original.count(INCLUDE) != 1:
            raise ValueError("pinned source NQC include anchor changed")
        source_file.write_text(original.replace(INCLUDE, INCLUDE+OVERRIDE))
        library = Path(temp)/"source_q4_nqc_override.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), paths["adapter"], library,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        lib = ctypes.CDLL(str(library))
        baseline_function = lib.diffgw_v4_full_source_mode
        baseline_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        baseline_function.restype = ctypes.c_int
        override_function = lib.diffgw_v4_full_source_mode_with_nqc
        override_function.argtypes = baseline_function.argtypes + [
            ctypes.POINTER(ctypes.c_double)]
        override_function.restype = ctypes.c_int
        for step in reversed(plan["output_steps_M"]):
            started = time.monotonic()
            source_epoch, source_dt, source_mode = _source(
                baseline_function, row, step, plan["capacity"])
            wheel_epoch, wheel_dt, wheel_mode, _ = _public(row, step)
            control = chart["grids"][str(step)]
            nqc = dynamics["grids"][str(step)]["nqc"]
            if (hashlib.sha256(source_mode.tobytes()).hexdigest()
                    != control["source"]["mode_sha256"]
                    or hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                    != control["direct_wheel"]["mode_sha256"]
                    or abs(source_dt/mass_seconds-step) > 1e-10
                    or abs(wheel_dt/mass_seconds-step) > 1e-10):
                raise ValueError("source or wheel baseline changed")
            source_active = np.asarray(nqc["source"], dtype=np.float64)[[0, 1, 2, 6, 7]].copy()
            wheel_active = np.asarray(nqc["wheel"], dtype=np.float64)[[0, 1, 2, 6, 7]].copy()
            noop = override_source(override_function, row, step,
                                   plan["capacity"], source_active)
            modified = override_source(override_function, row, step,
                                       plan["capacity"], wheel_active)
            if (noop[0] != source_epoch or noop[1] != source_dt
                    or not np.array_equal(noop[2], source_mode)
                    or modified[0] != source_epoch or modified[1] != source_dt
                    or len(modified[2]) != len(source_mode)):
                raise ValueError("NQC override control or source time axis changed")
            source = (source_epoch+np.arange(len(source_mode))*source_dt, source_mode)
            intervention = (modified[0]+np.arange(len(modified[2]))*modified[1], modified[2])
            wheel = (wheel_epoch+np.arange(len(wheel_mode))*wheel_dt, wheel_mode)
            fitted_baseline = fitted_metrics(*source, *wheel, mass_seconds, plan["peak_fit"])
            fitted_modified = fitted_metrics(*intervention, *wheel,
                                              mass_seconds, plan["peak_fit"])
            if (abs(fitted_baseline["metrics"]["complex_relative_l2"]
                    - control["source_vs_public_fitted"]["metrics"]["complex_relative_l2"])
                    > 1e-12):
                raise ValueError("source/wheel fitted baseline changed")
            pre_baseline = _pre_peak(source, wheel, mass_seconds,
                                     plan["pre_peak_cut_M"], plan["peak_fit"])
            pre_modified = _pre_peak(intervention, wheel, mass_seconds,
                                     plan["pre_peak_cut_M"], plan["peak_fit"])
            result["grids"][str(step)] = dict(
                source_active=source_active.tolist(), wheel_active=wheel_active.tolist(),
                source_mode_sha256=hashlib.sha256(source_mode.tobytes()).hexdigest(),
                noop_mode_sha256=hashlib.sha256(noop[2].tobytes()).hexdigest(),
                modified_mode_sha256=hashlib.sha256(modified[2].tobytes()).hexdigest(),
                fitted_baseline=fitted_baseline, fitted_modified=fitted_modified,
                pre_attachment_baseline=pre_baseline,
                pre_attachment_modified=pre_modified,
                same_source_mode_change=q.mode_metrics(modified[2], source_mode),
                elapsed_s=time.monotonic()-started)
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            print(f"grid={step:g} whole="
                  f"{fitted_baseline['metrics']['complex_relative_l2']:.9g}->"
                  f"{fitted_modified['metrics']['complex_relative_l2']:.9g} "
                  f"pre={pre_baseline['metrics']['complex_relative_l2']:.9g}->"
                  f"{pre_modified['metrics']['complex_relative_l2']:.9g}", flush=True)
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
