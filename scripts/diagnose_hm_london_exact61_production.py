#!/usr/bin/env python3
"""Verify the production Torch-native IMRPhenomHM route.

This report runs the unchanged public binary parity harness through the
production dispatcher, then exercises the public mode, batching, dtype, and
autograd contracts.  The four independent rows are copied from the fixed
``diagnose_hm_qnm_leaver_clean.CASES`` inventory.  LAL is used only by the
external comparison harness; it is not imported by the production module.

The N=61 coefficient grid remains a sampling hypothesis inferred from the
published London intervals.  This report records the fixed runtime payload
hash and does not claim historical coefficient provenance or fit coefficients
to waveform residuals.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = (
    ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification"
    / "hm-london-exact61-production"
)
N61_HASH = "afcb21b634efacbc1ed0b44f24fda5df5ab08e236f3b7cbfdc7e71d3452d6525"
PARITY_GATE = {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _finite_nonzero(value: torch.Tensor) -> tuple[bool, bool]:
    finite = bool(torch.isfinite(value).all().item())
    nonzero = bool(torch.any(value != 0.0).item())
    return finite, nonzero


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    projections = result.get("projections", {})
    return {
        "status": result.get("status"),
        "case_id": result.get("case_id"),
        "strict": result.get("strict"),
        "plus_complex_relative_l2": result.get("plus", {}).get("complex_relative_l2"),
        "cross_complex_relative_l2": result.get("cross", {}).get("complex_relative_l2"),
        "plus_flat_mismatch": result.get("plus", {}).get("flat_mismatch"),
        "cross_flat_mismatch": result.get("cross", {}).get("flat_mismatch"),
        "plus_psd_mismatches": result.get("psd_mismatches_plus", {}),
        "cross_psd_mismatches": result.get("psd_mismatches_cross", {}),
        "max_projection_complex_relative_l2": max(
            (row.get("complex_relative_l2", float("nan")) for row in projections.values()),
            default=None,
        ),
        "max_projection_flat_mismatch": max(
            (row.get("flat_mismatch", float("nan")) for row in projections.values()),
            default=None,
        ),
        "max_projection_psd_mismatch": max(
            (
                mismatch
                for row in projections.values()
                for mismatch in row.get("psd_mismatches", {}).values()
            ),
            default=None,
        ),
    }


def _canonical_hm_cases(parity: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    catalog = parity.get_canonical_catalog_cases()
    holdout = parity.get_canonical_holdout_cases()
    catalog_ok, catalog_errors = parity.validate_case_inventory(catalog, "catalog")
    holdout_ok, holdout_errors = parity.validate_case_inventory(holdout, "holdout")
    if not catalog_ok or not holdout_ok:
        raise RuntimeError(
            f"frozen inventory changed: catalog={catalog_errors}, holdout={holdout_errors}"
        )
    catalog_hm = [case for case in catalog if case.get("torch_name") == "IMRPhenomHM"]
    holdout_hm = [case for case in holdout if case.get("torch_name") == "IMRPhenomHM"]
    if len(catalog_hm) != 1 or len(holdout_hm) != 1:
        raise RuntimeError(
            f"expected one IMRPhenomHM row per inventory; catalog={len(catalog_hm)}, "
            f"holdout={len(holdout_hm)}"
        )
    return catalog_hm[0], holdout_hm[0], {
        "catalog_validated": True,
        "holdout_validated": True,
        "catalog_case_count": len(catalog),
        "holdout_case_count": len(holdout),
        "selected_catalog_id": catalog_hm[0]["id"],
        "selected_holdout_id": holdout_hm[0]["id"],
    }


def _independent_cases(helper: Any) -> list[dict[str, Any]]:
    cases = []
    for index, raw in enumerate(helper.CASES, start=1):
        m1, m2, chi1, chi2, distance, inclination, phic, f_ref = (float(x) for x in raw)
        cases.append(
            {
                "id": f"hm_london_exact61_independent_{index}",
                "torch_name": "IMRPhenomHM",
                "lal_name": "IMRPhenomHM",
                "params": {
                    "mass1": m1,
                    "mass2": m2,
                    "chi1": chi1,
                    "chi2": chi2,
                    "f_ref": f_ref,
                    "distance": distance,
                    "inclination": inclination,
                    "phic": phic,
                    "tc": 0.0,
                },
                "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
                "expected_gate": dict(PARITY_GATE),
                "verification_scope": "independent_case_raw",
                "oracle_type": "lal",
                "oracle_api": "SimInspiralChooseFDWaveform",
                "source_case": "diagnose_hm_qnm_leaver_clean.CASES",
            }
        )
    return cases


def _grad_record(value: torch.Tensor) -> dict[str, Any]:
    if value.grad is None:
        return {"present": False, "finite": False, "nonzero": False}
    finite, nonzero = _finite_nonzero(value.grad)
    return {
        "present": True,
        "finite": finite,
        "nonzero": nonzero,
        "values": value.grad.detach().cpu().tolist(),
    }


def _runtime_controls() -> dict[str, Any]:
    """Exercise production API shape, projection, mode, dtype, and gradients."""
    from torchwave.approximants.imrphenom_hm import IMRPhenomHM
    from torchwave.waveforms import get_fd_waveform

    model = IMRPhenomHM()
    f = torch.linspace(25.0, 280.0, 48, dtype=torch.float64)
    base = {
        "mass1": 36.0,
        "mass2": 19.0,
        "chi1": 0.24,
        "chi2": -0.18,
        "distance": 420.0,
        "inclination": 0.83,
        "phic": 0.27,
        "tc": 0.0012,
        "f_ref": 35.0,
    }

    with torch.no_grad():
        hp_infer, hc_infer = get_fd_waveform(
            "IMRPhenomHM", sample_frequencies=f, dtype=torch.float64, **base
        )
    infer_finite = bool(torch.isfinite(hp_infer).all() and torch.isfinite(hc_infer).all())

    hp_sample, hc_sample = model(f=None, sample_frequencies=f, **base)
    sample_error = max(
        float(torch.max(torch.abs(hp_sample - hp_infer)).detach()),
        float(torch.max(torch.abs(hc_sample - hc_infer)).detach()),
    )

    subset = ((2, 2), (3, 3))
    _, _, returned_modes = model(f=f, modes=subset, return_modes=True, **base)
    _, _, returned_unprojected = model(
        f=f, modes=subset, return_unprojected_modes=True, **base
    )
    subset_shapes = {
        str(mode): list(value[0].shape) for mode, value in returned_modes.items()
    }
    unprojected_shapes = {
        str(mode): list(value.shape) for mode, value in returned_unprojected.items()
    }

    # A scalar mass pair must broadcast over all nonmass parameters, including
    # a per-row reference frequency.  The loss is phase-sensitive.
    chi1 = torch.tensor([0.22, -0.31], dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor([-0.11, 0.17], dtype=torch.float64, requires_grad=True)
    distance = torch.tensor([120.0, 190.0], dtype=torch.float64, requires_grad=True)
    inclination = torch.tensor([0.4, 1.1], dtype=torch.float64, requires_grad=True)
    phic = torch.tensor([0.15, -0.35], dtype=torch.float64, requires_grad=True)
    tc = torch.tensor([0.0003, -0.0004], dtype=torch.float64, requires_grad=True)
    f_ref = torch.tensor([35.0, 55.0], dtype=torch.float64, requires_grad=True)
    hp_batch, hc_batch = get_fd_waveform(
        "IMRPhenomHM",
        sample_frequencies=f,
        mass1=35.0,
        mass2=21.0,
        chi1=chi1,
        chi2=chi2,
        distance=distance,
        inclination=inclination,
        phic=phic,
        tc=tc,
        f_ref=f_ref,
        dtype=torch.float64,
    )
    loss = (
        0.37 * hp_batch.real
        + 0.19 * hp_batch.imag
        - 0.23 * hc_batch.real
        + 0.11 * hc_batch.imag
    ).sum()
    loss.backward()
    batch_grads = {
        name: _grad_record(value)
        for name, value in {
            "chi1": chi1,
            "chi2": chi2,
            "distance": distance,
            "inclination": inclination,
            "phic": phic,
            "tc": tc,
            "f_ref": f_ref,
        }.items()
    }

    # Explicit leading frequency axes are accepted alongside the broadcast
    # parameter batch and retain the final sample axis.
    f_batch = f.unsqueeze(0).expand(2, -1)
    hp_explicit, hc_explicit = model(
        f=f_batch,
        mass1=35.0,
        mass2=21.0,
        chi1=chi1.detach(),
        chi2=chi2.detach(),
        distance=distance.detach(),
        inclination=inclination.detach(),
        phic=phic.detach(),
        tc=tc.detach(),
        f_ref=f_ref.detach(),
    )

    # Body exchange and odd-mode behavior are checked on unprojected modes.
    exchange_a = model(
        f=f, mass1=45.0, mass2=18.0, chi1=0.35, chi2=-0.25,
        return_unprojected_modes=True,
    )[2]
    exchange_b = model(
        f=f, mass1=18.0, mass2=45.0, chi1=-0.25, chi2=0.35,
        return_unprojected_modes=True,
    )[2]
    exchange_errors = {}
    for ell, m in model.SUPPORTED_MODES:
        expected = (-1.0) ** m
        exchange_errors[str((ell, m))] = {
            "magnitude": float(torch.linalg.norm(exchange_a[(ell, m)].abs() - exchange_b[(ell, m)].abs()).detach()),
            "parity": float(torch.linalg.norm(exchange_b[(ell, m)] - expected * exchange_a[(ell, m)]).detach()),
        }
    _, _, equal_modes = model(
        f=f, mass1=30.0, mass2=30.0, chi1=0.2, chi2=0.2,
        modes=((2, 1), (3, 3), (4, 3)), return_unprojected_modes=True,
    )
    _, _, unequal_spin_modes = model(
        f=f, mass1=30.0, mass2=30.0, chi1=0.5, chi2=-0.5,
        modes=((2, 1), (4, 3)), return_unprojected_modes=True,
    )

    f32 = f.to(torch.float32)
    hp32, hc32 = model(
        f=f32, mass1=36.0, mass2=19.0, chi1=0.24, chi2=-0.18,
        distance=420.0, inclination=0.83, phic=0.27, tc=0.0, f_ref=35.0,
    )

    source_path = Path(__file__).resolve().parents[1] / "src" / "torchwave" / "approximants" / "imrphenom_hm.py"
    runtime_text = source_path.read_text()
    forbidden_runtime_tokens = {
        token: (token in runtime_text.lower())
        for token in ("import lal", "import lalsimulation", "from scripts", "read_text()")
    }
    batch_finite = bool(torch.isfinite(hp_batch).all() and torch.isfinite(hc_batch).all())
    explicit_finite = bool(torch.isfinite(hp_explicit).all() and torch.isfinite(hc_explicit).all())
    return {
        "inference_first_then_autograd": {
            "inference_used_no_grad": True,
            "inference_shape": list(hp_infer.shape),
            "inference_dtype": str(hp_infer.dtype),
            "inference_finite": infer_finite,
            "sample_frequencies_with_f_none_max_abs_error": sample_error,
        },
        "mode_api": {
            "requested_modes": [list(mode) for mode in subset],
            "return_modes_keys": [list(mode) for mode in returned_modes],
            "return_unprojected_modes_keys": [list(mode) for mode in returned_unprojected],
            "projected_shapes": subset_shapes,
            "unprojected_shapes": unprojected_shapes,
        },
        "batched_parameters": {
            "shape": list(hp_batch.shape),
            "dtype": str(hp_batch.dtype),
            "finite": batch_finite,
            "loss": float(loss.detach()),
            "gradients": batch_grads,
            "explicit_frequency_batch_shape": list(hp_explicit.shape),
            "explicit_frequency_batch_finite": explicit_finite,
        },
        "body_exchange": {"mode_errors": exchange_errors},
        "odd_mode_contract": {
            "equal_mass_equal_spin_max_abs": {
                str(mode): float(equal_modes[mode].abs().max().detach())
                for mode in equal_modes
            },
            "equal_mass_unequal_spin_max_abs": {
                str(mode): float(unequal_spin_modes[mode].abs().max().detach())
                for mode in unequal_spin_modes
            },
        },
        "dtype": {
            "float32_plus": str(hp32.dtype),
            "float32_cross": str(hc32.dtype),
            "float32_finite": bool(torch.isfinite(hp32).all() and torch.isfinite(hc32).all()),
        },
        "runtime_import_audit": {
            "forbidden_tokens_present": forbidden_runtime_tokens,
            "production_native_torch_route": not any(forbidden_runtime_tokens.values()),
        },
    }


def _source_paths() -> dict[str, Path]:
    return {
        "scripts/diagnose_hm_london_exact61_production.py": SCRIPT_PATH,
        "scripts/diagnose_hm_qnm_leaver_clean.py": ROOT / "scripts" / "diagnose_hm_qnm_leaver_clean.py",
        "tests/parity_harness.py": ROOT / "tests" / "parity_harness.py",
        "tests/test_phenomhm_reconstruction.py": ROOT / "tests" / "test_phenomhm_reconstruction.py",
        "src/torchwave/approximants/imrphenom_hm.py": ROOT / "src" / "torchwave" / "approximants" / "imrphenom_hm.py",
        "src/torchwave/approximants/imrphenom_hm_qnm_n61.py": ROOT / "src" / "torchwave" / "approximants" / "imrphenom_hm_qnm_n61.py",
        "src/torchwave/approximants/hm_assembly_clean.py": ROOT / "src" / "torchwave" / "approximants" / "hm_assembly_clean.py",
        "src/torchwave/approximants/imrphenom_hm_phase_clean.py": ROOT / "src" / "torchwave" / "approximants" / "imrphenom_hm_phase_clean.py",
        "src/torchwave/approximants/kerr_qnm_d_sampling_clean.py": ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_d_sampling_clean.py",
        "src/torchwave/waveforms.py": ROOT / "src" / "torchwave" / "waveforms.py",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _source_paths()
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import diagnose_hm_qnm_leaver_clean as helper
    import parity_harness as parity
    from torchwave.approximants import imrphenom_hm_qnm_n61 as n61

    catalog_case, holdout_case, inventory = _canonical_hm_cases(parity)
    independent_cases = _independent_cases(helper)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "provenance": {
            "paper": "London et al., arXiv:1708.00404, Eq. (14)",
            "paper_url": "https://arxiv.org/abs/1708.00404",
            "route": "production IMRPhenomHM through torchwave.waveforms.get_fd_waveform",
            "projection": "native Torch positive-frequency harmonic projection",
            "hm_qnm_source": "independent N=61 coefficient artifact evaluated in Torch",
            "n61_grid_hypothesis": n61.N61_GRID_DEFINITION,
            "n61_provenance": "sampling hypothesis inferred from published intervals; inherited coefficient provenance is not certified",
            "eq14_carrier_contract": "HM QNM data enter Eq. (14) only through R=fRD_HM/fRD_lm and T=fdamp_lm/fdamp_HM; the D carrier retains its absolute fRD_D/fdamp_D and C1 join at 0.5*fRD_D",
            "d_carrier_qnm_source": "independent evaluate_d_qnm_220(final_spin), scaled by total_mass/final_mass",
            "coefficient_selection_from_waveform_residuals": False,
            "thresholds_changed": False,
            "external_oracle": "public LAL binary API through tests/parity_harness.py only",
            "runtime_external_imports": False,
        },
        "artifact": {
            "format": n61.N61_ARTIFACT_FORMAT,
            "count": n61.N61_COUNT,
            "fit_strategy": n61.N61_FIT_STRATEGY,
            "payload_sha256": n61.N61_COEFFICIENT_PAYLOAD_SHA256,
            "expected_payload_sha256": N61_HASH,
            "payload_hash_matches_fixed_expected": n61.N61_COEFFICIENT_PAYLOAD_SHA256 == N61_HASH,
        },
        "execution": {
            "host": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "command": [sys.executable, str(SCRIPT_PATH), *(argv if argv is not None else sys.argv[1:])],
            "pid": os.getpid(),
            "python": platform.python_version(),
            "log_path": None,
            "stop_command": f"kill {os.getpid()}",
        },
        "source_hashes_before": source_hashes_before,
        "inventory": inventory,
        "canonical_strict": [],
        "independent_strict": [],
        "summaries": [],
        "runtime_controls": None,
    }
    _write_json(output_dir / "progress.json", report)

    for scope, case in (("catalog", catalog_case), ("holdout", holdout_case)):
        print(f"running production strict HM {scope}: {case['id']}", flush=True)
        result = parity.run_oracle_comparison(copy.deepcopy(case), strict=True)
        report["canonical_strict"].append(result)
        report["summaries"].append({"scope": scope, **_summary(result)})
        _write_json(output_dir / "progress.json", report)

    for index, case in enumerate(independent_cases, start=1):
        print(f"running production independent HM {index}/{len(independent_cases)}: {case['id']}", flush=True)
        result = parity.run_oracle_comparison(copy.deepcopy(case), strict=True)
        report["independent_strict"].append(result)
        report["summaries"].append({"scope": "independent", **_summary(result)})
        _write_json(output_dir / "progress.json", report)

    report["runtime_controls"] = _runtime_controls()
    source_hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
    report["source_hashes_after"] = source_hashes_after
    report["source_hashes_changed_during_run"] = {
        name: [source_hashes_before[name], source_hashes_after[name]]
        for name in source_hashes_before
        if source_hashes_before[name] != source_hashes_after[name]
    }
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete_production_verification"
    _write_json(output_dir / "report.json", report)
    _write_json(output_dir / "progress.json", report)
    print(output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
