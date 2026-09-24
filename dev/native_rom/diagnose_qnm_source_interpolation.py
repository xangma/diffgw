"""Audit historical QNM table interpolation without importing it into native code."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from . import qualify_generator as q
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/qnm_source_interpolation_plan.json"


def _first_row(source: str, name: str) -> np.ndarray:
    match = re.search(
        rf"const double {name}\[8\]\[107\]\s*=\s*\{{\s*\{{([^}}]+)\}}\s*,",
        source,
    )
    if match is None:
        raise ValueError(f"Missing pinned source table: {name}")
    return np.fromstring(match.group(1), sep=",")


def _native(spins: np.ndarray) -> np.ndarray:
    frequency, damping = signed_qnm_220(torch.tensor(spins, dtype=torch.float64))
    return np.stack((frequency.detach().numpy(), damping.detach().numpy()), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-source-interpolation-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_revision"] != "53acc0c2cd7ea2d3c83fc71a7bd88035c2861d6c"
            or plan["source_sha256"] != "99430b73bd8f67aa72bfbe49b3a86051908d4bae83324dc5c5d6410b304cde95"
            or plan["spin_witnesses"] != [0.8941271984269911, 0.9994,
                                         0.999540793009, 0.9995469946384197,
                                         0.9996]
            or plan["source_knot_count"] != 107
            or plan["source_spline_to_binary_absolute_tolerance"] != 1e-12):
        parser.error("Frozen source-interpolation audit plan changed")
    source_path = args.source_root / plan["source_relative_path"]
    if q.sha256(source_path) != plan["source_sha256"]:
        parser.error("Pinned historical QNM source hash mismatch")
    source = source_path.read_text()
    match = re.search(r"static const double afinallist\[107\]\s*=\s*\{([^}]+)\};", source)
    if match is None or "gsl_spline_alloc(gsl_interp_cspline, 107)" not in source:
        parser.error("Pinned spin knots or spline rule not found")
    spins = np.fromstring(match.group(1), sep=",")
    table = np.stack((_first_row(source, "reomegaqnm22"),
                      _first_row(source, "imomegaqnm22")), axis=-1)
    if len(spins) != plan["source_knot_count"] or table.shape != (len(spins), 2):
        parser.error("Unexpected historical fundamental QNM table size")
    if not np.all(np.diff(spins) > 0):
        parser.error("Historical spin knots are not increasing")
    q.lal_binary_identity(plan["lal_version"], plan["lalsimulation_version"])
    spline = CubicSpline(spins, table, axis=0, bc_type="natural")
    native_knots = _native(spins)
    knot_errors = abs(native_knots - table)
    knot_worst = {}
    positive_knot_worst = {}
    for component, name in enumerate(("frequency", "damping")):
        index = int(np.argmax(knot_errors[:, component]))
        knot_worst[name] = {
            "spin": float(spins[index]),
            "source_printed": float(table[index, component]),
            "native_solved": float(native_knots[index, component]),
            "absolute_error": float(knot_errors[index, component]),
        }
        positive = np.flatnonzero(spins >= 0)
        index = int(positive[np.argmax(knot_errors[positive, component])])
        positive_knot_worst[name] = {
            "spin": float(spins[index]),
            "source_printed": float(table[index, component]),
            "native_solved": float(native_knots[index, component]),
            "absolute_error": float(knot_errors[index, component]),
        }
    native_knot_spline = CubicSpline(spins, native_knots, axis=0, bc_type="natural")
    witnesses = []
    for spin in plan["spin_witnesses"]:
        vector = lal.CreateCOMPLEX16Vector(1)
        status = lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(vector, 1., spin, 2, 2, 1)
        if status != 0:
            raise RuntimeError(f"Pinned binary QNM failed at spin {spin}")
        binary_value = vector.data[0] * lal.MTSUN_SI
        binary = np.array((binary_value.real, binary_value.imag))
        historical = np.asarray(spline(spin))
        solved_knot_interpolation = np.asarray(native_knot_spline(spin))
        native = _native(np.array([spin]))[0]
        witnesses.append({
            "spin": spin,
            "historical_spline": historical.tolist(),
            "binary": binary.tolist(),
            "native": native.tolist(),
            "solved_knot_interpolation": solved_knot_interpolation.tolist(),
            "spline_minus_binary_abs": abs(historical - binary).tolist(),
            "native_minus_historical_abs": abs(native - historical).tolist(),
            "interpolation_error_signed": (solved_knot_interpolation - native).tolist(),
            "printed_knot_effect_signed": (historical - solved_knot_interpolation).tolist(),
        })
    max_spline_binary = max(max(row["spline_minus_binary_abs"]) for row in witnesses)
    report = {
        "schema": "diffgw.seobnrv4.qnm-source-interpolation.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "scope": dict(q.SCOPE,
            description="Historical 220 fundamental QNM table/rule audit, not generator validation.",
            known_limitations=[
                "Pinned LAL source table entries are third-party numerical data, not native model inputs.",
                "Replaying the historical table establishes provenance of wheel output, not physical accuracy.",
                "Only selected witness spins and the printed source knots are tested.",
                "No full waveform or ROM is qualified by this component audit.",
            ]),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "source_revision": plan["source_revision"],
        "source_relative_path": plan["source_relative_path"],
        "source_sha256": q.sha256(source_path),
        "historical_rule": "GSL gsl_interp_cspline (natural cubic) over printed spin/QNM tables",
        "knot_count": len(spins), "spin_domain": [float(spins[0]), float(spins[-1])],
        "worst_printed_knot_minus_native": knot_worst,
        "worst_nonnegative_printed_knot_minus_native": positive_knot_worst,
        "witnesses": witnesses,
        "max_source_spline_minus_binary_abs": max_spline_binary,
        "gates": {"source_spline_reproduces_pinned_binary":
                  max_spline_binary < plan["source_spline_to_binary_absolute_tolerance"]},
    }
    q.write_evidence(args.json_out, report)
    if not all(report["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
