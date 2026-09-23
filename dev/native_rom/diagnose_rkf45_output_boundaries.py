"""Run the frozen development-only Fehlberg-output boundary experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from . import qualify_generator as q

import torch
import torchwave


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN_NAMES = ("rkf45_output_boundary_plan.json",
              "rkf45_fixed_start_boundary_plan.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--plan", choices=PLAN_NAMES, default=PLAN_NAMES[0])
    parser.add_argument("--row", choices=json.loads((EVIDENCE / PLAN_NAMES[0]).read_text())["rows"])
    args = parser.parse_args(argv)
    plan_path = EVIDENCE / args.plan
    torch.set_num_threads(1)
    module_path = Path(torchwave.__file__).resolve()
    if not module_path.is_relative_to(q.ROOT / "src"):
        parser.error(f"torchwave loaded outside this checkout: {module_path}")
    plan = json.loads(plan_path.read_text())
    rows = plan["rows"]
    q.validate_rows(list(rows.values()))
    expected = dict(output_resampler="one_fifth_order_fehlberg_substep_from_preceding_accepted_state",
                    fine_max_step_M=0.125,
                    local_nqc_phase_derivatives=True,
                    local_matching_second_derivatives=True,
                    local_second_window_M=0.5,
                    local_second_degree=6,
                    fixed_restart_time_M=165.0)
    if args.plan == PLAN_NAMES[1]:
        expected["coarse_initial_step_M"] = 0.5
    if plan["candidate"] != expected:
        parser.error("candidate differs from frozen Fehlberg output plan")
    config = plan["settings"]
    settings = q.Settings(atol_levels=tuple(config["atol_levels"]),
                          rtol_levels=tuple(config["rtol_levels"]),
                          dt_levels_M=tuple(config["dt_levels_M"]),
                          max_run_seconds=config["max_run_seconds"],
                          relative_l2_limit=config["relative_l2_limit"],
                          event_time_limit_M=config["event_time_limit_M"])
    q.validate_tolerance_levels(settings.atol_levels, settings.rtol_levels)
    before = q.source_snapshot()
    before[str(Path(__file__).resolve().relative_to(q.ROOT))] = q.sha256(__file__)
    report = dict(
        schema="diffgw.seobnrv4.rkf45-output-boundary-campaign.v1",
        scope=dict(**q.SCOPE, development_only=True,
                   numerical_generator_qualified=False, independent_physics=False,
                   exploratory_boundaries=True),
        plan_sha256=q.sha256(plan_path), plan_name=args.plan, source_sha256_before=before,
        source_manifest_sha256=q.canonical_hash(before),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid(),
                     torchwave_module=str(module_path), pythonpath=os.environ.get("PYTHONPATH")),
        settings=config, selected_rows=list(rows if args.row is None else [args.row]),
        rows={}, status="running")
    q.write_evidence(args.json_out, report)

    def runner(row, level, runner_settings):
        return q.native_run(
            row, level, runner_settings,
            diagnostic_rkf45_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=165.0,
            diagnostic_fine_max_step_M=0.125,
            diagnostic_coarse_initial_step_M=expected.get("coarse_initial_step_M"))

    for name in report["selected_rows"]:
        report["rows"][name] = q.qualify_row(
            rows[name], settings, runner=runner,
            progress=lambda level, name=name: print(f"row={name} run={level}", flush=True))
        q.write_evidence(args.json_out, report)
        checks = report["rows"][name]["checks"]
        print(json.dumps(dict(row=name,
                              numerical_checks_passed=report["rows"][name]["numerical_checks_passed"],
                              gates={k: v["gates"] for k, v in checks.items()},
                              errors=report["rows"][name]["errors"])), flush=True)
    after = q.source_snapshot()
    after[str(Path(__file__).resolve().relative_to(q.ROOT))] = q.sha256(__file__)
    report["source_sha256_after"] = after
    report["sources_unchanged"] = before == after
    report["all_numerical_checks_passed"] = (report["sources_unchanged"]
        and all(row["numerical_checks_passed"] for row in report["rows"].values()))
    report["status"] = ("diagnostic_numerical_gates_passed" if report["all_numerical_checks_passed"]
                        else "failed")
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["all_numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
