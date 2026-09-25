"""Check native 220 QNM interpolation against all supported Cook author rows.

The 106 MB author archive is supplied externally and never copied into a
release package or used to generate native coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import jax
import numpy as np
import torch

from . import qualify_generator as q

jax.config.update("jax_enable_x64", True)
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220  # noqa: E402
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220  # noqa: E402


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/qnm_cook_domain_plan.json"


def _author_rows(archive: h5py.File, path: str, spin_max: float) -> tuple[np.ndarray, np.ndarray]:
    data = archive[path][:, :3]
    if (data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all()
            or not np.all(np.diff(data[:, 0]) > 0)
            or data[0, 0] != 0 or not np.all(data[:, 1] > 0)
            or not np.all(data[:, 2] < 0)):
        raise ValueError(f"Unexpected Cook 220 dataset layout: {path}")
    selected = data[data[:, 0] <= spin_max]
    if len(selected) < 1000:
        raise ValueError(f"Cook supported-domain sample unexpectedly sparse: {path}")
    return selected[:, 0], selected[:, 1:3] * np.array([1.0, -1.0])


def _worst(error: np.ndarray, spin: np.ndarray, author: np.ndarray,
           native: np.ndarray) -> dict:
    row, component = np.unravel_index(int(np.argmax(error)), error.shape)
    return {
        "author_row_index": int(row),
        "signed_spin": float(spin[row]),
        "component": ("frequency", "damping")[component],
        "author": float(author[row, component]),
        "native": float(native[row, component]),
        "absolute_error": float(error[row, component]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-archive", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-cook-domain-plan.v1"
            or plan["release_qualified"] is not False
            or plan["author_archive_sha256"] != "d5853acabc1a07b66b4fae437c02e1435a9d2b3561bd7e4752b9721c9cd48923"
            or plan["author_datasets"] != {
                "corotating": "/n00/m+02/{2,2,0}",
                "counterrotating": "/n00/m-02/{2,-2,0}",
            }
            or plan["native_signed_spin_max"] != 0.9999
            or plan["native_author_absolute_tolerance"] != 1e-9
            or plan["torch_jax_absolute_tolerance"] != 1e-12):
        parser.error("Frozen Cook domain plan changed")
    archive_hash = q.sha256(args.author_archive)
    if archive_hash != plan["author_archive_sha256"]:
        parser.error("Cook author archive SHA-256 mismatch")
    with h5py.File(args.author_archive) as archive:
        rows = {
            key: _author_rows(archive, path, plan["native_signed_spin_max"])
            for key, path in plan["author_datasets"].items()
        }
    spin = np.concatenate((rows["corotating"][0], -rows["counterrotating"][0]))
    author = np.concatenate((rows["corotating"][1], rows["counterrotating"][1]))
    with torch.no_grad():
        torch_values = np.stack([
            value.cpu().numpy() for value in signed_qnm_220(torch.tensor(spin, dtype=torch.float64))
        ], axis=-1)
    jax_values = np.stack([
        np.asarray(value) for value in evaluate_kerr_qnm_220(jax.numpy.asarray(spin))
    ], axis=-1)
    if (torch_values.shape != author.shape or jax_values.shape != author.shape
            or not np.isfinite(torch_values).all() or not np.isfinite(jax_values).all()):
        raise ValueError("Native QNM output shape or finiteness invalid")
    split = len(rows["corotating"][0])
    per_branch = {}
    for key, sample_slice in (("corotating", slice(None, split)),
                              ("counterrotating", slice(split, None))):
        branch_spin = spin[sample_slice]
        branch_author = author[sample_slice]
        branch_torch = torch_values[sample_slice]
        branch_jax = jax_values[sample_slice]
        per_branch[key] = {
            "row_count": len(branch_spin),
            "signed_spin_range": [float(branch_spin.min()), float(branch_spin.max())],
            "worst_torch_minus_author": _worst(abs(branch_torch - branch_author),
                                                branch_spin, branch_author, branch_torch),
            "worst_jax_minus_author": _worst(abs(branch_jax - branch_author),
                                              branch_spin, branch_author, branch_jax),
            "max_torch_jax_abs": float(np.max(abs(branch_torch - branch_jax))),
        }
    max_torch_author = max(branch["worst_torch_minus_author"]["absolute_error"]
                           for branch in per_branch.values())
    max_jax_author = max(branch["worst_jax_minus_author"]["absolute_error"]
                         for branch in per_branch.values())
    max_torch_jax = max(branch["max_torch_jax_abs"] for branch in per_branch.values())
    report = {
        "schema": "diffgw.seobnrv4.qnm-cook-domain.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "scope": dict(q.SCOPE,
                      description="Independent QNM component validation across every Cook author row in the native signed-spin domain.",
                      known_limitations=[
                          "Cook data are an external validation oracle, not native model inputs or an asset to distribute.",
                          "Table rows do not prove a continuous-domain bound between rows.",
                          "QNM validation alone does not qualify attachment, whole waveforms, or the ROM.",
                      ]),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "author_archive_sha256": archive_hash,
        "author_datasets": plan["author_datasets"],
        "branch_results": per_branch, "total_row_count": len(spin),
        "max_torch_minus_author_abs": max_torch_author,
        "max_jax_minus_author_abs": max_jax_author,
        "max_torch_jax_abs": max_torch_jax,
        "gates": {
            "torch_matches_cook_supported_rows": max_torch_author < plan["native_author_absolute_tolerance"],
            "jax_matches_cook_supported_rows": max_jax_author < plan["native_author_absolute_tolerance"],
            "torch_jax_agree": max_torch_jax < plan["torch_jax_absolute_tolerance"],
        },
    }
    q.write_evidence(args.json_out, report)
    if not all(report["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
