"""Dev-only JAX 35-mode flux check against saved pinned-source boundary rows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jaxwave.approximants.eob.seobnrv4_flux import (  # noqa: E402
    SEOBNRv4PublishedFlux,
)

from . import qualify_generator as q  # noqa: E402
from .eob.jax.source_calibration_hamiltonian import (  # noqa: E402
    SourceCalibratedHistoricalHamiltonian,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_flux_boundary_modes_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / value["path"] for name, value in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.jax-flux-boundary-modes-plan.v1"
            or plan["release_qualified"] is not False
            or plan["expected_rows"] != 4 or plan["expected_points"] != 12
            or plan["expected_modes"] != 35
            or any(q.sha256(paths[name]) != value["sha256"]
                   for name, value in plan["inputs"].items())):
        parser.error("Frozen plan or inputs changed")
    source = json.loads(paths["source_modes"].read_text())
    if (source["status"] != "diagnostic_complete"
            or source["comparison_gates_passed"] is not True
            or source["release_qualified"] is not False
            or len(source["rows"]) != plan["expected_rows"]
            or sum(len(row["points"]) for row in source["rows"])
            != plan["expected_points"]):
        parser.error("Pinned source boundary evidence changed")

    report = {
        "schema": "diffgw.seobnrv4.jax-flux-boundary-modes.v1",
        "status": "running", "release_qualified": False,
        "scope": dict(q.SCOPE, development_only=True),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "flux_choices": source["flux_choices"],
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "rows": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        flux = SEOBNRv4PublishedFlux(**source["flux_choices"])
        if len(flux.mode_indices) != plan["expected_modes"]:
            raise ValueError("JAX flux mode count changed")
        max_component_source = max_component_torch = 0.0
        max_total_source = max_total_torch = 0.0
        max_sum = 0.0
        for source_row in source["rows"]:
            physical = source_row["physical"]
            m1, m2 = physical["mass1"], physical["mass2"]
            x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
            nu = x1 * x2
            ham = SourceCalibratedHistoricalHamiltonian(
                nu, physical["chi1"], physical["chi2"], delta=x1-x2)
            row = {"name": source_row["name"], "physical": physical, "points": []}
            report["rows"].append(row)
            for source_point in source_row["points"]:
                radius, pphi, omega = (source_point[key]
                                       for key in ("r", "pphi", "omega"))
                state = jnp.asarray([radius, 0., 0., pphi], dtype=jnp.float64)
                inputs = flux.from_hamiltonian(state, ham, omega=jnp.asarray(omega))
                signed = flux._signed_amplitudes(state, **inputs)
                components = {
                    mode: float((mode[1]*omega)**2 * signed[mode]**2
                                / (8*math.pi*nu))
                    for mode in flux.mode_indices
                }
                jax_total = float(flux.energy_flux(state, **inputs) / nu)
                source_total = source_point["source_total_over_nu"]
                torch_total = source_point["native_total_over_nu"]
                source_components = {(entry["ell"], entry["m"]): entry
                                     for entry in source_point["components"]}
                if (len(source_components) != plan["expected_modes"]
                        or set(components) != set(source_components)
                        or not math.isfinite(jax_total) or jax_total <= 0
                        or not all(math.isfinite(value)
                                   for value in components.values())):
                    raise ValueError("Invalid JAX or saved source mode data")
                component_source = max(abs(value-source_components[mode]["source_over_nu"])
                                       / source_total for mode, value in components.items())
                component_torch = max(abs(value-source_components[mode]["native_over_nu"])
                                      / source_total for mode, value in components.items())
                total_source = abs(jax_total-source_total) / source_total
                total_torch = abs(jax_total-torch_total) / source_total
                mode_sum = abs(sum(components.values())/jax_total-1.)
                max_component_source = max(max_component_source, component_source)
                max_component_torch = max(max_component_torch, component_torch)
                max_total_source = max(max_total_source, total_source)
                max_total_torch = max(max_total_torch, total_torch)
                max_sum = max(max_sum, mode_sum)
                row["points"].append({
                    "index": source_point["index"], "r": radius, "pphi": pphi,
                    "omega": omega, "jax_total_over_nu": jax_total,
                    "jax_minus_source_total_relative": (jax_total-source_total)/source_total,
                    "jax_minus_torch_total_relative_to_source":
                    (jax_total-torch_total)/source_total,
                    "max_abs_component_difference_source_relative_to_total":
                    component_source,
                    "max_abs_component_difference_torch_relative_to_total":
                    component_torch,
                    "jax_mode_sum_relative": sum(components.values())/jax_total-1.,
                    "components": [
                        {"ell": mode[0], "m": mode[1], "jax_over_nu": value,
                         "jax_minus_source_relative_to_source_total":
                         (value-source_components[mode]["source_over_nu"])/source_total,
                         "jax_minus_torch_relative_to_source_total":
                         (value-source_components[mode]["native_over_nu"])/source_total}
                        for mode, value in components.items()
                    ],
                })
            q.write_evidence(args.json_out, report)
        report["max_abs_component_source_relative_to_total"] = max_component_source
        report["max_abs_component_torch_relative_to_total"] = max_component_torch
        report["max_abs_total_source_relative"] = max_total_source
        report["max_abs_total_torch_relative_to_source"] = max_total_torch
        report["max_abs_mode_sum_relative"] = max_sum
        limits = plan["limits"]
        report["comparison_gates_passed"] = (
            max_component_source <= limits["component_source_relative_to_total"]
            and max_component_torch <= limits["component_torch_relative_to_total"]
            and max_total_source <= limits["total_source_relative"]
            and max_total_torch <= limits["total_torch_relative_to_source"]
            and max_sum <= limits["mode_sum_relative"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    q.write_evidence(args.json_out, report)
    print(json.dumps({key: report.get(key) for key in (
        "status", "comparison_gates_passed", "max_abs_component_source_relative_to_total",
        "max_abs_component_torch_relative_to_total", "max_abs_total_source_relative",
        "max_abs_total_torch_relative_to_source", "errors")}), flush=True)
    return 0 if (report["status"] == "diagnostic_complete"
                 and report["comparison_gates_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
