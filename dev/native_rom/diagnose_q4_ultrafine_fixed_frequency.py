"""Dev-only q4 native/source comparison on a fixed inspiral-frequency chart."""

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
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, fitted_peak, public_mode
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_nqc_inputs import INCLUDE
from dev.native_rom.diagnose_source_nqc_whole_mode_override import override_source
from dev.native_rom.diagnose_source_q4_ultrafine_chart import _native
from dev.native_rom.diagnose_source_q4_ultrafine_local_nqc import (
    _anchored_metrics,
    _chart,
    _frequency_anchor,
    _mode_function,
)
from dev.native_rom.diagnose_source_wheel_mode import _public, _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "q4_ultrafine_fixed_frequency_plan.json"
OUTPUT = EVIDENCE / "q4_ultrafine_fixed_frequency.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    paths = {
        "source_plan": ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json",
        "prior_chart": EVIDENCE / "source_q4_ultrafine_chart.json",
        "local_fit": EVIDENCE / "source_q4_ultrafine_local_nqc.json",
        "wheel_override": EVIDENCE / "source_q4_ultrafine_nqc_override.json",
        "adapter": ROOT / "dev/native_rom/reference/full_v4_source_large_nqc_override_adapter.c",
    }
    if (plan["schema"] != "diffgw.seobnrv4.q4-ultrafine-fixed-frequency-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_step_M"] != 0.015625
            or plan["inspiral_anchor_frequency_M_inv"] != 0.1
            or plan["peak_fit"] != "quadratic_power"
            or plan["capacity"] != 50000
            or plan["atol"] != 1e-14 or plan["rtol"] != 1e-13
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        raise ValueError("frozen q4 frequency-chart control changed")
    source_plan = json.loads(paths["source_plan"].read_text())
    prior = json.loads(paths["prior_chart"].read_text())
    local = json.loads(paths["local_fit"].read_text())
    override = json.loads(paths["wheel_override"].read_text())
    if any(item["status"] != "diagnostic_complete" for item in (prior, local, override)):
        raise ValueError("pinned prior q4 diagnostic is incomplete")
    step = plan["output_step_M"]
    key = str(step)
    row = prior["row"]
    if row != local["row"] or row != override["row"]:
        raise ValueError("pinned q4 row changed")
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    historical = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/"
                             "combined_domain_numerics.json").read_text())
    settings = q.Settings(**historical["settings"])
    result = dict(schema="diffgw.seobnrv4.q4-ultrafine-fixed-frequency.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, output_step_M=step, mode_sha256={}, anchors={},
                  comparisons={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="diffgw-q4-fixed-frequency-") as temp:
        private = Path(temp) / "source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
        original = source_file.read_text()
        from dev.native_rom.diagnose_source_nqc_whole_mode_override import OVERRIDE
        if original.count(INCLUDE) != 1:
            raise ValueError("pinned source NQC include anchor changed")
        source_file.write_text(original.replace(INCLUDE, INCLUDE + OVERRIDE))
        library_path = Path(temp) / "source_override.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), paths["adapter"], library_path,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"], require_wheel_flux=False)
        library = ctypes.CDLL(str(library_path))
        baseline = _mode_function(library, "diffgw_v4_full_source_mode")
        modified = _mode_function(library, "diffgw_v4_full_source_mode_with_nqc")
        modified.argtypes += [ctypes.POINTER(ctypes.c_double)]
        source = _source(baseline, row, step, plan["capacity"])
        noop = override_source(modified, row, step, plan["capacity"],
                               np.asarray(override["grids"][key]["source_active"]))
        source_local = override_source(modified, row, step, plan["capacity"],
            np.asarray(local["grids"][key]["local_fit"]["coefficients"]))
        source_wheel_coefficients = override_source(modified, row, step,
            plan["capacity"], np.asarray(override["grids"][key]["wheel_active"]))
        wheel = _public(row, step)[:3]
        run = _native(row, step, plan, settings)
        native = q.summarize_run(run)
        native_time, native_mode, _, _, actual_mass_seconds = public_mode(row, run)
        if abs(actual_mass_seconds - mass_seconds) > 1e-15:
            raise ValueError("native mass conversion changed")
        if (not np.array_equal(noop[2], source[2])
                or native["array_sha256"] != prior["grids"][key]["native"]["array_sha256"]
                or hashlib.sha256(source[2].tobytes()).hexdigest()
                    != prior["grids"][key]["source"]["mode_sha256"]
                or hashlib.sha256(wheel[2].tobytes()).hexdigest()
                    != prior["grids"][key]["direct_wheel"]["mode_sha256"]
                or hashlib.sha256(source_local[2].tobytes()).hexdigest()
                    != local["grids"][key]["local_mode_sha256"]
                or hashlib.sha256(source_wheel_coefficients[2].tobytes()).hexdigest()
                    != override["grids"][key]["modified_mode_sha256"]):
            raise ValueError("frozen q4 source/wheel/native replay changed")
        charts = {"native": (native_time, native_mode),
                  "source": _chart(source), "source_local": _chart(source_local),
                  "source_wheel_coefficients": _chart(source_wheel_coefficients),
                  "wheel": _chart(wheel)}
        for name, chart in charts.items():
            result["mode_sha256"][name] = hashlib.sha256(chart[1].tobytes()).hexdigest()
            result["anchors"][name] = _frequency_anchor(chart, mass_seconds,
                plan["inspiral_anchor_frequency_M_inv"])
            result["anchors"][name]["peak"] = fitted_peak(*chart, plan["peak_fit"])
        for candidate_name, reference_name in (
            ("source", "wheel"),
            ("source_local", "wheel"),
            ("source_wheel_coefficients", "wheel"),
            ("native", "source"),
            ("native", "source_local"),
            ("native", "source_wheel_coefficients"),
            ("native", "wheel"),
        ):
            candidate, reference = charts[candidate_name], charts[reference_name]
            comparison = _anchored_metrics(candidate, reference, mass_seconds,
                result["anchors"][candidate_name], result["anchors"][reference_name])
            comparison["peak_fitted_complex_relative_l2"] = fitted_metrics(
                *candidate, *reference, mass_seconds, plan["peak_fit"]
            )["metrics"]["complex_relative_l2"]
            comparison["peak_lag_from_frequency_anchor_M"] = (
                result["anchors"][candidate_name]["time_before_peak_M"]
                - result["anchors"][reference_name]["time_before_peak_M"])
            result["comparisons"][candidate_name + "_vs_" + reference_name] = comparison
    result["elapsed_s"] = time.monotonic() - started
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, comparison in result["comparisons"].items():
        print(name, comparison["metrics"]["complex_relative_l2"], flush=True)


if __name__ == "__main__":
    main()
