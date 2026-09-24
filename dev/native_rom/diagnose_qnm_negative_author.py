"""Compare sparse counterrotating QNM roots with hashed Cook author data."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import lal
import lalsimulation as lalsim
import torch

from . import qualify_generator as q
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/qnm_negative_author_plan.json"


def _author_row(archive: Path, dataset: str, index: int) -> tuple[float, float, float]:
    command = ["h5dump", "-m", "%.17g", "-d", dataset, "-s", f"{index},0",
               "-c", "1,3", str(archive)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values = {int(column): float(value) for row, column, value in re.findall(
        r"\((\d+),(\d+)\):\s*([^\n,]+)", result.stdout)
        if int(row) == index}
    if set(values) != {0, 1, 2}:
        raise ValueError(f"Unexpected author dataset row format at {index}")
    return values[0], values[1], -values[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-archive", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-negative-author-plan.v1"
            or plan["release_qualified"] is not False
            or plan["author_archive_sha256"] != "d5853acabc1a07b66b4fae437c02e1435a9d2b3561bd7e4752b9721c9cd48923"
            or plan["author_dataset"] != "/n00/m-02/{2,-2,0}"
            or plan["posthoc_row_indices"] != [990, 995, 1002, 1005, 1006]
            or plan["expected_positive_spin"] != [0.99, 0.995, 0.999, 0.9995, 0.999625]
            or plan["pinned_qnm_spin_absolute_upper_bound"] != 0.9996
            or plan["native_author_absolute_tolerance"] != 1e-9):
        parser.error("Frozen negative-spin author plan changed")
    if q.sha256(args.author_archive) != plan["author_archive_sha256"]:
        parser.error("Cook author archive SHA-256 mismatch")
    identity = q.lal_binary_identity(plan["lal_version"], plan["lalsimulation_version"])
    rows = []
    for index, expected_spin in zip(plan["posthoc_row_indices"],
                                    plan["expected_positive_spin"], strict=True):
        spin, author_frequency, author_damping = _author_row(
            args.author_archive, plan["author_dataset"], index)
        if abs(spin - expected_spin) > 1e-14 or author_damping <= 0:
            raise ValueError(f"Unexpected Cook author spin or damping at row {index}")
        native_frequency, native_damping = (
            float(value) for value in signed_qnm_220(
                torch.tensor(-spin, dtype=torch.float64)))
        item = {
            "author_row_index": index, "signed_spin": -spin,
            "author": [author_frequency, author_damping],
            "native": [native_frequency, native_damping],
            "native_minus_author_abs": [abs(native_frequency - author_frequency),
                                        abs(native_damping - author_damping)],
            "wheel_in_domain": spin <= plan["pinned_qnm_spin_absolute_upper_bound"],
        }
        if item["wheel_in_domain"]:
            vector = lal.CreateCOMPLEX16Vector(1)
            status = lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(
                vector, 1., -spin, 2, 2, 1)
            if status != 0:
                raise RuntimeError(f"Pinned wheel QNM failed at signed spin {-spin}")
            value = vector.data[0] * lal.MTSUN_SI
            wheel = [float(value.real), float(value.imag)]
            item["wheel"] = wheel
            item["wheel_minus_author_abs"] = [abs(wheel[0] - author_frequency),
                                              abs(wheel[1] - author_damping)]
        rows.append(item)
    max_native_author = max(max(row["native_minus_author_abs"]) for row in rows)
    max_wheel_author = max(max(row["wheel_minus_author_abs"]) for row in rows
                           if "wheel_minus_author_abs" in row)
    report = {
        "schema": "diffgw.seobnrv4.qnm-negative-author.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "scope": dict(q.SCOPE,
                      description="Sparse independent counterrotating author-data check only.",
                      known_limitations=[
                          "Rows were selected post hoc after inspecting the negative-spin source discrepancy.",
                          "Five author rows do not bound the full spin domain or validate waveform ringdown.",
                          "The Cook archive is a third-party validation source, not a native model asset.",
                      ]),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "author_archive_sha256": q.sha256(args.author_archive),
        "author_dataset": plan["author_dataset"], "binary": identity,
        "rows": rows, "max_native_minus_author_abs": max_native_author,
        "max_wheel_minus_author_abs": max_wheel_author,
        "gates": {"native_matches_sparse_cook_author_rows":
                  max_native_author < plan["native_author_absolute_tolerance"]},
    }
    q.write_evidence(args.json_out, report)
    if not all(report["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
