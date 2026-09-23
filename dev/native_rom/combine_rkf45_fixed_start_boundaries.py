"""Verify and combine the six frozen Fehlberg fixed-start row reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import qualify_generator as q


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "rkf45_fixed_start_boundary_plan.json"
COMPONENTS = (
    "rkf45_fixed_start_boundary_campaign.json",
    "rkf45_fixed_start_mass_ratio_four_nonspinning.json",
    "rkf45_fixed_start_strong_aligned_spin.json",
    "rkf45_fixed_start_strong_antialigned_spin.json",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    current = q.source_snapshot()
    current[str((EVIDENCE.parent.parent / "diagnose_rkf45_output_boundaries.py").relative_to(q.ROOT))] = q.sha256(
        EVIDENCE.parent.parent / "diagnose_rkf45_output_boundaries.py")
    rows = {}
    components = []
    manifests = set()
    heads = set()
    for name in COMPONENTS:
        path = EVIDENCE / name
        report = json.loads(path.read_text())
        if report["plan_sha256"] != q.sha256(PLAN):
            raise ValueError(f"plan mismatch in {name}")
        if report["source_sha256_before"] != current:
            raise ValueError(f"source mismatch in {name}")
        after = report.get("source_sha256_after")
        if after is not None and after != current:
            raise ValueError(f"source changed during {name}")
        if name != COMPONENTS[0] and report["status"] != "diagnostic_numerical_gates_passed":
            raise ValueError(f"incomplete component {name}")
        if name == COMPONENTS[0] and report["status"] != "running":
            raise ValueError("expected the deliberately stopped serial report")
        manifests.add(report["source_manifest_sha256"])
        heads.add(report["git_head"])
        for row_name, row in report["rows"].items():
            if row_name in rows or row_name not in plan["rows"]:
                raise ValueError(f"duplicate or unplanned row {row_name}")
            if row["inputs"] != plan["rows"][row_name] or len(row["runs"]) != 5:
                raise ValueError(f"incomplete row {row_name}")
            rows[row_name] = row
        components.append(dict(path=name, sha256=q.sha256(path), status=report["status"],
                               process=report["process"], rows=list(report["rows"])))
    if len(manifests) != 1 or len(heads) != 1 or set(rows) != set(plan["rows"]):
        raise ValueError("components do not form one frozen six-row campaign")
    all_passed = all(row["numerical_checks_passed"] for row in rows.values())
    grid_identity = all(
        row["checks"]["output_grid"][pair]["complex_relative_l2"] == 0
        for row in rows.values() for pair in ("coarse_to_medium", "medium_to_fine"))
    report = dict(
        schema="diffgw.seobnrv4.rkf45-fixed-start-combined.v1",
        scope=dict(**q.SCOPE, development_only=True,
                   numerical_generator_qualified=False, independent_physics=False,
                   exploratory_boundaries=True),
        status="diagnostic_numerical_gates_passed" if all_passed else "failed",
        plan_sha256=q.sha256(PLAN), git_head=heads.pop(),
        source_manifest_sha256=manifests.pop(), source_sha256_verified=current,
        script_sha256=q.sha256(__file__), components=components,
        rows={name: rows[name] for name in plan["rows"]},
        all_numerical_checks_passed=all_passed,
        nested_grid_waveforms_identical=grid_identity,
        grid_control_independent_convergence=False,
        interpretation=("Fixed first trial step makes nested grids reuse the same accepted "
                        "trajectory and fine output sampling. Zero grid differences are an "
                        "identity control, not an independent accuracy or physics check."))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], rows=list(report["rows"]),
                          nested_grid_waveforms_identical=grid_identity,
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if all_passed and grid_identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
