"""Run the predeclared local second-derivative rule through both v4 gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch

from . import qualify_generator as q
from .diagnose_nqc_second_derivatives import CAMPAIGNS


DECLARATION = "dev/native_rom/evidence/2026-09-23/nqc_second_derivative_candidate.md"


def snapshot():
    result = q.source_snapshot()
    for path in (__file__,):
        source = Path(path).resolve()
        result[str(source.relative_to(q.ROOT))] = q.sha256(source)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    frozen = {name: json.loads(Path(path).read_text())
              for name, path in CAMPAIGNS.items()}
    settings = q.Settings(atol_levels=(1e-12, 1e-13, 1e-14),
                          rtol_levels=(1e-11, 1e-12, 1e-13))
    for name, campaign in frozen.items():
        if (campaign["settings"]["atol_levels"] != list(settings.atol_levels)
                or campaign["settings"]["rtol_levels"] != list(settings.rtol_levels)
                or campaign["settings"]["dt_levels_M"] != list(settings.dt_levels_M)
                or campaign["numerical_choice"]["fixed_restart_time_M"] != 165.
                or not campaign["numerical_choice"]["hermite_resampling"]
                or not campaign["numerical_choice"]["local_nqc_phase_derivatives"]):
            parser.error(f"{name} campaign does not match the predeclaration")
    before = snapshot()
    report = dict(
        schema="diffgw.seobnrv4.nqc-second-derivative-campaign.v1",
        scope=dict(**q.SCOPE, development_only=True,
                   numerical_generator_qualified=False, independent_physics=False,
                   no_coefficient_fitting=True),
        declaration_sha256=q.sha256(DECLARATION),
        dense_diagnostic_sha256=q.sha256(
            "dev/native_rom/evidence/2026-09-23/nqc_second_derivatives_two_rows.json"),
        frozen_campaigns={name: dict(path=path, sha256=q.sha256(path))
                          for name, path in CAMPAIGNS.items()},
        numerical_choice=dict(hermite_resampling=True,
                              local_nqc_phase_matrix=True,
                              local_second_window_M=0.5,
                              local_second_degree=6,
                              fixed_restart_time_M=165.),
        settings=dict(atol_levels=settings.atol_levels,
                      rtol_levels=settings.rtol_levels,
                      dt_levels_M=settings.dt_levels_M,
                      relative_l2_limit=settings.relative_l2_limit,
                      event_time_limit_M=settings.event_time_limit_M),
        source_sha256_before=before, source_manifest_sha256=q.canonical_hash(before),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        rows={}, status="running")
    q.write_evidence(args.json_out, report)
    try:
        report["lal_binary"] = q.lal_binary_identity("7.7.1", "6.2.1")
        for name, campaign in frozen.items():
            def runner(row, level, runner_settings):
                return q.native_run(
                    row, level, runner_settings,
                    diagnostic_hermite_resampling=True,
                    diagnostic_local_nqc_derivatives=True,
                    diagnostic_matching_second_derivatives=True,
                    diagnostic_restart_time_M=165.)

            report["rows"][name] = q.qualify_row(
                campaign["row"], settings, runner=runner, oracle=q.compare_lal,
                progress=lambda level: print(f"row={name} run={level}", flush=True))
            q.write_evidence(args.json_out, report)
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    after = snapshot()
    report["source_sha256_after"] = after
    report["sources_unchanged"] = before == after
    passed = ("error" not in report and report["sources_unchanged"]
              and len(report["rows"]) == len(frozen)
              and all(row["numerical_checks_passed"] for row in report["rows"].values()))
    report["status"] = "diagnostic_numerical_gates_passed" if passed else "failed"
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], evidence=str(args.json_out.resolve()))),
          flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
