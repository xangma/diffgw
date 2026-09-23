"""Development-only v4 convergence campaign with opt-in Hermite sampling.

The declared qualification gates are reused unchanged. Passing them here would
not promote this provisional physics model or change the default sampler.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch

from dev.native_rom.qualify_generator import (
    Settings, canonical_hash, compare_lal, lal_binary_identity, native_run,
    qualify_row, sha256, source_snapshot, write_evidence,
)


ROW = dict(omega_start=0.04, mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0)


def snapshot():
    sources = source_snapshot()
    source = Path(__file__).resolve()
    sources[str(source.relative_to(source.parents[2]))] = sha256(source)
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = Settings(atol_levels=(1e-12, 1e-13, 1e-14),
                        rtol_levels=(1e-11, 1e-12, 1e-13))
    before = snapshot()
    report = dict(
        schema="diffgw.seobnrv4.hermite_campaign_diagnostic.v1",
        scope=dict(physics_validated=False, production_asset=False,
                   production_promotion_allowed=False,
                   training_generator_qualified=False,
                   development_only=True, no_coefficient_fitting=True),
        row=ROW, settings=dict(atol_levels=settings.atol_levels,
                               rtol_levels=settings.rtol_levels,
                               dt_levels_M=settings.dt_levels_M),
        numerical_choice=dict(hermite_resampling=True,
                              default_restart_rule=True,
                              default_initial_step_rule=True),
        source_sha256_before=before, source_manifest_sha256=canonical_hash(before),
        status="running", process=dict(argv=sys.argv, cwd=os.getcwd(),
                                       host=socket.gethostname(), pid=os.getpid()),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions=dict(torch=torch.__version__),
    )
    write_evidence(args.json_out, report)
    try:
        report["lal_binary"] = lal_binary_identity("7.7.1", "6.2.1")

        def runner(row, level, runner_settings):
            return native_run(row, level, runner_settings,
                              diagnostic_hermite_resampling=True)

        report["result"] = qualify_row(
            ROW, settings, runner=runner, oracle=compare_lal,
            progress=lambda name: print(f"run={name}", flush=True),
        )
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    after = snapshot()
    report["source_sha256_after"] = after
    report["sources_unchanged"] = before == after
    passed = ("error" not in report and report["sources_unchanged"]
              and report["result"]["numerical_checks_passed"])
    report["status"] = "diagnostic_numerical_gates_passed" if passed else "failed"
    write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"],
                      "evidence": str(args.json_out.resolve())}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
