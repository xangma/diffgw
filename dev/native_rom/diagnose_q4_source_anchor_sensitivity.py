"""Measure q4 source/wheel NQC-control sensitivity to inspiral anchor choice."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import lal
import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_peak_anchor import fitted_metrics
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_nqc_inputs import INCLUDE
from dev.native_rom.diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from dev.native_rom.diagnose_source_q4_ultrafine_local_nqc import (
    _anchored_metrics, _chart, _frequency_anchor, _mode_function)
from dev.native_rom.diagnose_source_wheel_mode import _public, _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "q4_source_anchor_sensitivity_plan.json"
OUTPUT = EVIDENCE / "q4_source_anchor_sensitivity.json"


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
        "fixed_frequency": EVIDENCE / "q4_ultrafine_fixed_frequency.json",
        "adapter": ROOT / "dev/native_rom/reference/full_v4_source_large_nqc_override_adapter.c",
    }
    if (plan["schema"] != "diffgw.seobnrv4.q4-source-anchor-sensitivity-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["frequencies_M_inv"] != [0.1, 0.11, 0.12]
            or plan["capacity"] != 50000
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        raise ValueError("frozen q4 source-anchor control changed")
    source_plan = json.loads(paths["source_plan"].read_text())
    prior = json.loads(paths["prior_chart"].read_text())
    local = json.loads(paths["local_fit"].read_text())
    override = json.loads(paths["wheel_override"].read_text())
    fixed = json.loads(paths["fixed_frequency"].read_text())
    if any(item["status"] != "diagnostic_complete"
           for item in (prior, local, override, fixed)):
        raise ValueError("pinned q4 source evidence is incomplete")
    row = prior["row"]
    if row != local["row"] or row != override["row"] or row != fixed["row"]:
        raise ValueError("pinned q4 row changed")
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    result = dict(schema="diffgw.seobnrv4.q4-source-anchor-sensitivity.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with tempfile.TemporaryDirectory(prefix="diffgw-q4-source-anchors-") as temp:
        private = Path(temp) / "source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
        original = source_file.read_text()
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
        for step in plan["output_steps_M"]:
            key = str(step)
            source = _source(baseline, row, step, plan["capacity"])
            source_local = override_source(modified, row, step, plan["capacity"],
                np.asarray(local["grids"][key]["local_fit"]["coefficients"]))
            source_wheel_coefficients = override_source(modified, row, step,
                plan["capacity"], np.asarray(override["grids"][key]["wheel_active"]))
            wheel = _public(row, step)[:3]
            charts = {"source": _chart(source), "source_local": _chart(source_local),
                      "source_wheel_coefficients": _chart(source_wheel_coefficients),
                      "wheel": _chart(wheel)}
            expected = {
                "source": prior["grids"][key]["source"]["mode_sha256"],
                "source_local": local["grids"][key]["local_mode_sha256"],
                "source_wheel_coefficients": override["grids"][key]["modified_mode_sha256"],
                "wheel": prior["grids"][key]["direct_wheel"]["mode_sha256"],
            }
            if any(hashlib.sha256(chart[1].tobytes()).hexdigest() != expected[name]
                   for name, chart in charts.items()):
                raise ValueError("pinned q4 source/wheel mode changed")
            grid = dict(mode_sha256=expected, frequency_charts={}, peak_fitted={})
            for name in ("source", "source_local", "source_wheel_coefficients"):
                grid["peak_fitted"][name] = fitted_metrics(
                    *charts[name], *charts["wheel"], mass_seconds, "quadratic_power"
                )["metrics"]["complex_relative_l2"]
            for target in plan["frequencies_M_inv"]:
                anchors = {}
                for name, chart in charts.items():
                    try:
                        anchors[name] = _frequency_anchor(chart, mass_seconds, target)
                    except ValueError as exc:
                        raise ValueError(f"frequency anchor failed: {step=} {target=} {name=}") from exc
                comparisons = {}
                for name in ("source", "source_local", "source_wheel_coefficients"):
                    comparison = _anchored_metrics(
                        charts[name], charts["wheel"], mass_seconds,
                        anchors[name], anchors["wheel"])
                    comparisons[name] = comparison["metrics"]["complex_relative_l2"]
                grid["frequency_charts"][str(target)] = dict(
                    anchors=anchors, complex_relative_l2=comparisons)
            if step == 0.015625:
                for name, item in (("source", "source_vs_wheel"),
                                   ("source_local", "source_local_vs_wheel"),
                                   ("source_wheel_coefficients",
                                    "source_wheel_coefficients_vs_wheel")):
                    saved = fixed["comparisons"][item]["metrics"]["complex_relative_l2"]
                    actual = grid["frequency_charts"]["0.1"]["complex_relative_l2"][name]
                    if abs(actual - saved) > 1e-12:
                        raise ValueError("pinned q4 0.1 frequency chart changed")
            result["grids"][key] = grid
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for step, grid in result["grids"].items():
        for target, item in grid["frequency_charts"].items():
            print(step, target, item["complex_relative_l2"], flush=True)


if __name__ == "__main__":
    main()
