"""Check native signed-spin 220 interpolation against author-seeded direct roots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import h5py
import jax
import numpy as np
import torch

from . import qualify_generator as q

jax.config.update("jax_enable_x64", True)
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220  # noqa: E402
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220  # noqa: E402
from torchwave.qnm.solver import solve  # noqa: E402


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/qnm_finite_domain_solver_plan.json"


def _author_seed(data: np.ndarray, magnitude: float) -> complex:
    spin = data[:, 0]
    if not spin[0] <= magnitude <= spin[-1]:
        raise ValueError("declared spin outside Cook author seed grid")
    return complex(np.interp(magnitude, spin, data[:, 1]),
                   np.interp(magnitude, spin, data[:, 2]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-archive", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    cases = plan["cases"]
    spins = np.asarray([case["signed_spin"] for case in cases], dtype=np.float64)
    if (plan["schema"] != "diffgw.seobnrv4.qnm-finite-domain-solver-plan.v1"
            or plan["component_id"] != "V4-QNM-INTERPOLATION"
            or plan["signed_spin_domain"] != [-0.9999, 0.9999]
            or plan["selection"]["log_grid_count"] != 25
            or len(cases) != 113
            or np.sum(spins >= 0) != 57 or np.sum(spins < 0) != 56
            or not np.all(np.isfinite(spins))
            or not np.all(np.abs(spins) <= 0.9999)
            or len(set(spins.tolist())) != len(spins)
            or plan["strict_solver"] is not True
            or plan["native_minus_solver_absolute_tolerance"] != 1e-9
            or plan["torch_minus_jax_absolute_tolerance"] != 1e-12
            or plan["solver_estimated_frequency_error_max"] != 1e-10
            or plan["no_continuous_bound_claim"] is not True
            or plan["release_qualified"] is not False
            or plan["production_change"] is not False):
        parser.error("frozen finite-domain plan differs")
    for name, digest in {**plan["source_sha256"],
                         **plan["prior_evidence_sha256"]}.items():
        if q.sha256(q.ROOT / name) != digest:
            parser.error(f"pinned source or prior evidence changed: {name}")
    if q.sha256(args.author_archive) != plan["author_archive_sha256"]:
        parser.error("pinned Cook author archive changed")
    author = {}
    with h5py.File(args.author_archive) as archive:
        for branch, path in plan["author_datasets"].items():
            data = archive[path][:, :3]
            if (data.ndim != 2 or data.shape[1] != 3
                    or not np.isfinite(data).all()
                    or data[0, 0] != 0
                    or data[-1, 0] < 0.9999
                    or not np.all(np.diff(data[:, 0]) > 0)
                    or not np.all(data[:, 1] > 0)
                    or not np.all(data[:, 2] < 0)):
                raise ValueError(f"unexpected Cook author seed dataset: {path}")
            author[branch] = data
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.qnm-finite-domain-solver.v1",
        status="running", release_qualified=False, production_change=False,
        scope=dict(q.SCOPE,
                   description="Finite signed-spin Kerr-equation consistency check with author-seeded direct roots.",
                   known_limitations=plan["limitations"]),
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        source_sha256=plan["source_sha256"],
        prior_evidence_sha256=plan["prior_evidence_sha256"],
        author_archive_sha256=plan["author_archive_sha256"],
        author_datasets=plan["author_datasets"],
        branch_seed_source="Cook author archive linear frequency interpolation; native values are not solver seeds",
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv),
        rows=[], errors=[], gates={})
    q.write_evidence(args.json_out, report)
    started = time.monotonic()
    try:
        with torch.no_grad():
            torch_values = np.stack([np.asarray(value.cpu()) for value in
                signed_qnm_220(torch.tensor(spins, dtype=torch.float64))], axis=-1)
        jax_values = np.stack([np.asarray(value) for value in
            evaluate_kerr_qnm_220(jax.numpy.asarray(spins, dtype=jax.numpy.float64))], axis=-1)
        if (torch_values.shape != (len(cases), 2)
                or jax_values.shape != (len(cases), 2)
                or not np.isfinite(torch_values).all()
                or not np.isfinite(jax_values).all()
                or not np.all(torch_values > 0)
                or not np.all(jax_values > 0)):
            raise ValueError("native QNM shape, finiteness, or sign invalid")
        for index, case in enumerate(cases):
            signed_spin = float(spins[index])
            branch = "positive" if signed_spin >= 0 else "negative"
            magnitude = abs(signed_spin)
            seed = _author_seed(author[branch], magnitude)
            result = solve(magnitude, 2, 2 if branch == "positive" else -2, 0,
                           omega_seed=seed, strict=True)
            if not result.converged:
                raise RuntimeError(f"direct Kerr solve failed at signed spin {signed_spin}")
            solved = np.array([result.omega.real, -result.omega.imag], dtype=np.float64)
            estimated = float(result.metadata["estimated_frequency_error"])
            if (not np.isfinite(solved).all() or not np.all(solved > 0)
                    or not np.isfinite(estimated)):
                raise ValueError(f"direct Kerr solve invalid at signed spin {signed_spin}")
            report["rows"].append(dict(
                signed_spin=signed_spin, categories=case["categories"],
                torch=torch_values[index].tolist(), jax=jax_values[index].tolist(),
                direct_solver=solved.tolist(),
                torch_minus_solver_abs=np.abs(torch_values[index]-solved).tolist(),
                jax_minus_solver_abs=np.abs(jax_values[index]-solved).tolist(),
                torch_minus_jax_abs=np.abs(torch_values[index]-jax_values[index]).tolist(),
                solver_depth=result.cf_depth,
                solver_estimated_frequency_error=estimated,
                solver_branch_origin=result.metadata["branch_origin"]))
            if (index+1) % 10 == 0:
                q.write_evidence(args.json_out, report)
                print(f"{index+1}/{len(cases)} roots complete", flush=True)
        worst_solver = max(
            (float(error), row["signed_spin"], backend, component)
            for row in report["rows"] for backend in ("torch", "jax")
            for component, error in enumerate(row[backend+"_minus_solver_abs"]))
        worst_parity = max(
            (float(error), row["signed_spin"], component)
            for row in report["rows"]
            for component, error in enumerate(row["torch_minus_jax_abs"]))
        worst_estimated = max(float(row["solver_estimated_frequency_error"])
                              for row in report["rows"])
        report["summary"] = dict(
            row_count=len(report["rows"]),
            max_native_minus_solver_abs=worst_solver[0],
            max_native_minus_solver_location=dict(signed_spin=worst_solver[1],
                backend=worst_solver[2], component=("frequency", "damping")[worst_solver[3]]),
            max_torch_minus_jax_abs=worst_parity[0],
            max_torch_minus_jax_location=dict(signed_spin=worst_parity[1],
                component=("frequency", "damping")[worst_parity[2]]),
            max_solver_estimated_frequency_error=worst_estimated,
            elapsed_seconds=time.monotonic()-started)
        report["gates"] = dict(
            all_strict_solver_converged=(len(report["rows"]) == len(cases)),
            native_matches_direct_solver_at_declared_points=(
                worst_solver[0] < plan["native_minus_solver_absolute_tolerance"]),
            torch_jax_agree_at_declared_points=(
                worst_parity[0] < plan["torch_minus_jax_absolute_tolerance"]),
            solver_estimate_below_predeclared_limit=(
                worst_estimated < plan["solver_estimated_frequency_error_max"]))
        report["status"] = "diagnostic_complete" if all(report["gates"].values()) else "diagnostic_failed_gates"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], rows=len(report["rows"]),
                          summary=report.get("summary"), errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
