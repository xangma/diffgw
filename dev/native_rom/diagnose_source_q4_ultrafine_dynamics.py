"""Dev-only pinned-source/wheel low-dynamics q4 cadence attribution."""

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

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_mode import _public_nqc, _source_nqc
from dev.native_rom.diagnose_source_wheel_dynamics import _public, _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_ultrafine_dynamics_plan.json"
OUTPUT = EVIDENCE / "source_q4_ultrafine_dynamics.json"
FIELDS = ("r", "phi", "pr_star", "pphi")


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _compare(candidate, reference, fractions):
    ct, rt = candidate[0], reference[0]
    if (not np.isfinite(candidate).all() or not np.isfinite(reference).all()
            or not np.all(np.diff(ct) > 0) or not np.all(np.diff(rt) > 0)
            or abs(ct[0]) > 1e-10 or abs(rt[0]) > 1e-10):
        raise ValueError("invalid low-rate dynamics")
    common = (rt >= ct[0]) & (rt <= ct[-1])
    t = rt[common]
    if t.size < 10:
        raise ValueError("insufficient shared low-rate dynamics")
    diffs = np.stack([np.interp(t, ct, candidate[i])-reference[i, common]
                      for i in range(1, 5)])
    probe_indices = [round(f*(t.size-1)) for f in fractions]
    return dict(common_samples=int(t.size), common_support_M=[float(t[0]), float(t[-1])],
                max_abs={name: float(np.max(np.abs(diffs[i])))
                         for i, name in enumerate(FIELDS)},
                rms={name: float(np.sqrt(np.mean(diffs[i]**2)))
                     for i, name in enumerate(FIELDS)},
                probes=[dict(time_M=float(t[j]),
                             candidate_minus_reference={name: float(diffs[i, j])
                                                        for i, name in enumerate(FIELDS)})
                        for j in probe_indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    prior_path = EVIDENCE / "source_q4_ultrafine_chart.json"
    source_plan_path = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"
    adapter_path = ROOT / "dev/native_rom/reference/full_v4_source_dynamics_large_adapter.c"
    mode_adapter_path = ROOT / "dev/native_rom/reference/full_v4_source_mode_adapter.c"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-ultrafine-dynamics-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["capacity"] != 150000
            or q.sha256(prior_path) != plan["prior_chart_sha256"]
            or q.sha256(source_plan_path) != plan["source_flux_plan_sha256"]
            or q.sha256(adapter_path) != plan["source_dynamics_adapter_sha256"]
            or q.sha256(mode_adapter_path) != plan["source_mode_adapter_sha256"]):
        raise ValueError("frozen dynamics plan changed")
    prior = json.loads(prior_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    identity = _prepare_public(settings_plan)
    if prior["status"] != "diagnostic_complete" or prior["public_binary"] != identity:
        raise ValueError("pinned prior chart or public binary changed")
    row = prior["row"]
    result = dict(schema="diffgw.seobnrv4.source-q4-ultrafine-dynamics.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={}, adjacent={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    runs = {}
    with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-dynamics-") as temp:
        private = Path(temp)/"source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        library = Path(temp)/"source_q4_dynamics.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), adapter_path, library,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        function = ctypes.CDLL(str(library)).diffgw_v4_full_source_dynamics
        function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        mode_library = Path(temp)/"source_q4_nqc.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), mode_adapter_path,
                 mode_library, source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        nqc_function = ctypes.CDLL(str(mode_library)).diffgw_v4_full_source_nqc
        nqc_function.argtypes = [ctypes.c_double]*6 + [ctypes.POINTER(ctypes.c_double)]
        nqc_function.restype = ctypes.c_int
        for step in reversed(plan["output_steps_M"]):
            started = time.monotonic()
            source = _source(function, row, step, plan["capacity"])
            wheel = _public(row, step)
            source_nqc = _source_nqc(nqc_function, row, step)
            wheel_nqc = _public_nqc(row, step)
            result["grids"][str(step)] = dict(
                source=dict(samples=int(source.shape[1]), array_sha256=_digest(source)),
                wheel=dict(samples=int(wheel.shape[1]), array_sha256=_digest(wheel)),
                source_minus_wheel=_compare(source, wheel, plan["probe_fractions"]),
                nqc=dict(source=source_nqc.tolist(), wheel=wheel_nqc.tolist(),
                         source_minus_wheel=(source_nqc-wheel_nqc).tolist()),
                elapsed_s=time.monotonic()-started)
            runs[step] = source, wheel, source_nqc, wheel_nqc
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            print(f"grid={step:g} source/wheel phase max="
                  f"{result['grids'][str(step)]['source_minus_wheel']['max_abs']['phi']:.9g}",
                  flush=True)
    coarse, fine = plan["output_steps_M"]
    for name, index in (("source", 0), ("wheel", 1)):
        result["adjacent"][name] = _compare(runs[fine][index], runs[coarse][index],
                                             plan["probe_fractions"])
    result["adjacent"]["nqc"] = {
        name: (runs[fine][index]-runs[coarse][index]).tolist()
        for name, index in (("source", 2), ("wheel", 3))
    }
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
