#!/usr/bin/env python3
"""Compare clean TEOB radial acceleration conventions on a frozen NQC input.

This bounded diagnostic reads the reviewed ``nu=0.24`` continuation cache and
does not integrate a trajectory or rewrite the cache.  It evaluates the
existing exact conservative chain rule together with the opt-in strict
leading-``p_r*`` and frozen-mobility branches from
``teob_radial_acceleration_clean``.  Each branch is passed through the clean
auxiliary-grid NQC assembler so the original and auxiliary corrected
amplitude branches can be compared on identical physical input arrays.

The cache's historical mode is reported separately as a generic
``teob_published_driver.teob22_mode``/``FactorizedModes`` carrier.  The NQC
comparison uses a freshly evaluated ``TEOBRadiationClean.mode`` carrier.  The
latter owns the TEOB tail gauge and therefore receives exactly one internal
tail factor; this script applies no external tail factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.teob_merger_nqc_clean import (
    PUBLISHED_2017_NQC_PROVENANCE,
    assemble_teob_merger_nqc,
    phase_from_mode,
    published_2017_nqc_targets,
)
from torchwave.approximants.eob.teob_published_driver import teob22_mode
from torchwave.approximants.eob.teob_radial_acceleration_clean import (
    radial_acceleration_approximations,
)
from torchwave.approximants.eob.teob_radiation_clean import TEOBRadiationClean
from torchwave.approximants.eob.teob_trajectory import (
    conservative_trajectory_quantities,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-radial-acceleration-clean"
DEFAULT_OUTPUT = VERIFY / "diagnostic.json"
DEFAULT_CACHE = (
    ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-merger-nqc-clean"
    / "physical-input-output.npz"
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    """Return the own file and direct clean dependencies included in hashes."""

    return {
        "module": ROOT / "src/torchwave/approximants/eob/teob_radial_acceleration_clean.py",
        "test": ROOT / "tests/test_teob_radial_acceleration_clean.py",
        "script": ROOT / "scripts/diagnose_teob_radial_acceleration_clean.py",
        "trajectory": ROOT / "src/torchwave/approximants/eob/teob_trajectory.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "nqc_basis": ROOT / "src/torchwave/approximants/eob/teob_nqc_clean.py",
        "generic_carrier": ROOT / "src/torchwave/approximants/eob/teob_published_driver.py",
    }


def _as_tensor_arrays(data: np.lib.npyio.NpzFile) -> dict[str, torch.Tensor]:
    required = {
        "time",
        "state",
        "mode",
        "amplitude",
        "phase",
        "orbital_frequency",
        "radial_acceleration",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError("frozen NQC cache is missing arrays: " + ", ".join(missing))
    # Copy before closing the NPZ handle and make the read-only input boundary
    # explicit.  The diagnostic never calls np.savez on this file.
    return {
        key: torch.from_numpy(np.array(data[key], copy=True))
        for key in data.files
    }


def _float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _vector(value: object) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1)]
    return [float(item) for item in value]  # type: ignore[union-attr]


def _stats(value: torch.Tensor) -> dict[str, float]:
    values = value.detach().cpu().reshape(-1).to(torch.float64)
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    return {
        "min": _float(values.min()),
        "max": _float(values.max()),
        "mean": _float(values.mean()),
        "max_abs": _float(values.abs().max()),
        "rms": _float(torch.sqrt(torch.mean(values.square()))),
    }


def _difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = value - reference
    denominator = reference.abs().clamp_min(torch.as_tensor(1.0e-30, dtype=reference.dtype))
    result = _stats(difference)
    result["max_relative"] = _float((difference.abs() / denominator).max())
    return result


def _shape_dtype(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}


def _hamiltonian() -> TEOBResumSHamiltonian:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    return TEOBResumSHamiltonian(
        torch.tensor(0.24, dtype=dtype), zero, zero, zero, zero,
    )


def _fresh_physical_mode(
    ham: TEOBResumSHamiltonian,
    state: torch.Tensor,
    orbital_frequency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the direct clean TEOB mode and its lifted signed phase."""

    radius, orbital_phase, pr_star, p_phi = state.unbind(-1)
    h_eff, h_real, h_circ, a_prime = ham.clean_radiation_inputs(
        radius, pr_star, p_phi,
    )
    radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")
    # TEOBRadiationClean.mode contains the one TEOB tail-gauge factor.  Keep
    # this call direct so the carrier provider and its tail count stay clear.
    mode = radiation.mode(
        2,
        2,
        radius,
        orbital_frequency,
        orbital_phase,
        pr_star,
        p_phi,
        h_eff,
        h_real,
        h_circ,
        a_prime,
        ham.m1,
        ham.m2,
    )
    amplitude = mode.abs()
    phase = phase_from_mode(mode, orbital_phase)
    return mode, amplitude, phase


def _nqc_record(
    name: str,
    nqc,
    acceleration: torch.Tensor,
    exact_acceleration: torch.Tensor,
    exact_result,
) -> dict[str, object]:
    original_corrected = nqc.original.corrected_amplitude
    auxiliary_corrected = nqc.auxiliary.corrected_amplitude
    exact_original = exact_result.original.corrected_amplitude
    exact_auxiliary = exact_result.auxiliary.corrected_amplitude
    return {
        "acceleration": _stats(acceleration),
        "acceleration_difference_vs_exact_chain_rule": _difference(
            acceleration, exact_acceleration,
        ),
        "target_provenance": nqc.target_provenance,
        "coefficients": {
            "amplitude": _vector(nqc.coefficients.amplitude),
            "phase": _vector(nqc.coefficients.phase),
        },
        "matching": {
            "cut_start_index": nqc.cut_start_index,
            "cut_last_index": nqc.cut_last_index,
            "auxiliary_samples": int(nqc.auxiliary.time.numel()),
            "auxiliary_spacing": _float(nqc.auxiliary_spacing),
            "t_peak_grid": _float(nqc.t_peak_grid),
            "t_nqc": _float(nqc.t_nqc),
            "matching_index": nqc.matching_index,
            "amplitude_condition": _float(nqc.amplitude_condition),
            "phase_condition": _float(nqc.phase_condition),
            "amplitude_residual": _vector(nqc.amplitude_residual),
            "phase_residual": _vector(nqc.phase_residual),
        },
        "amplitude_branches": {
            "original_corrected": _stats(original_corrected),
            "auxiliary_corrected": _stats(auxiliary_corrected),
            "original_difference_vs_exact_branch": _difference(
                original_corrected, exact_original,
            ),
            "auxiliary_difference_vs_exact_branch": _difference(
                auxiliary_corrected, exact_auxiliary,
            ),
            "auxiliary_negative_samples": int(nqc.auxiliary_negative_amplitude.sum()),
        },
        "phase_branches": {
            "original_corrected": _stats(nqc.original.corrected_phase),
            "auxiliary_corrected": _stats(nqc.auxiliary.corrected_phase),
        },
        "mode_provider": "TEOBRadiationClean.mode",
        "tail_application_count": 1,
        "external_tail_factor": False,
        "branch_name": name,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE,
        help=f"read-only frozen NQC NPZ path (default: {DEFAULT_CACHE})",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    torch.set_default_dtype(torch.float64)
    started_perf = time.perf_counter()
    source_paths = _source_paths()
    source_before = {key: _hash(path) for key, path in source_paths.items()}
    cache_before = _hash(args.cache)
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    report: dict[str, object] = {
        "status": "running",
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": command,
        "pid": os.getpid(),
        "python": sys.executable,
        "device": "cpu",
        "started_unix": time.time(),
        "input_cache": {
            "path": str(args.cache),
            "sha256_before": cache_before,
            "read_only": True,
        },
        "source_sha256_before": source_before,
        "derivation": {
            "primary_sources": [
                {
                    "citation": "Damour & Nagar, arXiv:1406.6913v1, Eqs. (96)--(97)",
                    "url": "https://arxiv.org/pdf/1406.6913v1",
                    "role": "NQC radial basis and conservative ddot-r quantity",
                },
                {
                    "citation": "Damour, Nagar & Bernuzzi, arXiv:1212.4357v2, Appendix A",
                    "url": "https://arxiv.org/pdf/1212.4357v2",
                    "role": "Hamilton-equation chain rule and conservative branch",
                },
                {
                    "citation": "Nagar et al., arXiv:1806.01772v2, Sec. II, Eqs. (4)--(11)",
                    "url": "https://arxiv.org/html/1806.01772v2",
                    "role": "zero-spin TEOB effective Hamiltonian and radial flow",
                },
            ],
            "exact_conservative_chain_rule": (
                "a=(partial_r rdot)*rdot+(partial_prstar rdot)*prstar_dot; "
                "all derivatives and flow factors use the supplied current (r,prstar,pphi), "
                "with dp_phi/dt=0"
            ),
            "strict_leading_p": (
                "rdot=prstar*C; a=C0*prstar_dot0, where C0=xi0/(H0*Heff0) "
                "and prstar_dot0=-xi0*(partial_r Heff)_0/H0; "
                "A, xi, Heff, H, and partial_r Heff are all evaluated at prstar=0 "
                "at the same (r,pphi)"
            ),
            "frozen_mobility": (
                "a=C_current*prstar_dot_current with C held fixed in the time derivative; "
                "C_current=xi*(1+2*z3*A*prstar^2/r^2)/(H*Heff), and both C and "
                "prstar_dot use the supplied current prstar; the prstar=0 limit is "
                "C0=xi0/(H0*Heff0)"
            ),
            "mixed_diagnostics": {
                "mixed_current_p0_force": "C_current*prstar_dot0",
                "mixed_p0_current_force": "C0*prstar_dot_current",
                "selection": "diagnostic only; no runtime branch is selected",
            },
        },
    }

    try:
        with np.load(args.cache, allow_pickle=False) as loaded:
            arrays = _as_tensor_arrays(loaded)
            cache_shapes = {
                key: _shape_dtype(value) for key, value in arrays.items()
            }
        time_grid = arrays["time"]
        state = arrays["state"]
        orbital_frequency = arrays["orbital_frequency"]
        if time_grid.dtype != torch.float64 or state.dtype != torch.float64:
            raise ValueError("the reviewed frozen cache must use float64 arrays")
        if time_grid.ndim != 1 or state.shape != (time_grid.numel(), 4):
            raise ValueError("frozen cache time/state shapes are inconsistent")
        if orbital_frequency.shape != time_grid.shape:
            raise ValueError("frozen cache frequency shape is inconsistent")
        if not bool(torch.isfinite(time_grid).all() & torch.isfinite(state).all()
                    & torch.isfinite(orbital_frequency).all()):
            raise ValueError("frozen cache contains nonfinite trajectory values")

        ham = _hamiltonian()
        with torch.no_grad():
            exact = conservative_trajectory_quantities(ham, state).radial_acceleration
            terms = radial_acceleration_approximations(
                ham, state[:, 0], state[:, 2], state[:, 3],
            )
            physical_mode, physical_amplitude, physical_phase = _fresh_physical_mode(
                ham, state, orbital_frequency,
            )
            # This call is retained only to identify/reproduce the historical
            # generic carrier.  It is never passed to the NQC assembler.
            generic_mode = teob22_mode(state, ham, omega=orbital_frequency)
            generic_phase = phase_from_mode(generic_mode, state[:, 1])

        report["input_cache"] = {
            "path": str(args.cache),
            "sha256_before": cache_before,
            "read_only": True,
            "arrays": cache_shapes,
            "samples": int(time_grid.numel()),
            "time_start": _float(time_grid[0]),
            "time_end": _float(time_grid[-1]),
            "cached_mode_provider": (
                "teob_published_driver.teob22_mode -> "
                "generic FactorizedModes.mode"
            ),
            "cached_mode_used_for_nqc": False,
            "generic_carrier_reproduction": {
                "mode_difference_vs_cache": _stats(generic_mode - arrays["mode"]),
                "amplitude_difference_vs_cache": _stats(
                    generic_mode.abs() - arrays["amplitude"],
                ),
                "phase_difference_vs_cache": _stats(generic_phase - arrays["phase"]),
            },
        }
        report["physical_carrier"] = {
            "provider": "TEOBRadiationClean.mode",
            "mode": "(2,2)",
            "residual_phase_policy": "zero-unpublished",
            "tail_application_count": 1,
            "tail_application_location": "inside TEOBRadiationClean.mode",
            "external_tail_factor": False,
            "phase_extraction": (
                "2*orbital_phase - unwrap(angle(mode*exp(2j*orbital_phase)))"
            ),
            "mode_samples": int(physical_mode.numel()),
            "amplitude": _stats(physical_amplitude),
            "phase": _stats(physical_phase),
            "physical_vs_generic_carrier": {
                "complex_mode": _stats(physical_mode - generic_mode),
                "amplitude": _difference(physical_amplitude, generic_mode.abs()),
                "phase": _stats(physical_phase - generic_phase),
            },
            "physical_vs_historical_cache": {
                "complex_mode": _stats(physical_mode - arrays["mode"]),
                "amplitude": _difference(physical_amplitude, arrays["amplitude"]),
                "phase": _stats(physical_phase - arrays["phase"]),
            },
            "parity_claim": False,
            "parity_note": (
                "The historical cache carrier and the freshly evaluated clean "
                "TEOB carrier are reported separately; these differences are "
                "diagnostic evidence, not a public waveform parity result."
            ),
        }

        report["acceleration"] = {
            "cache_radial_acceleration": _stats(arrays["radial_acceleration"]),
            "fresh_exact_chain_rule": _stats(exact),
            "fresh_exact_difference_vs_cache": _difference(
                exact, arrays["radial_acceleration"],
            ),
            "strict_p0": _stats(terms.strict_p0),
            "frozen_mobility": _stats(terms.frozen_mobility),
            "mixed_current_p0_force": _stats(terms.current_mobility_p0_force),
            "mixed_p0_current_force": _stats(terms.p0_mobility_current_force),
            "strict_difference_vs_exact": _difference(terms.strict_p0, exact),
            "frozen_difference_vs_exact": _difference(terms.frozen_mobility, exact),
        }

        nu = torch.tensor(0.24, dtype=time_grid.dtype)
        targets = published_2017_nqc_targets(nu, torch.tensor(0.0, dtype=nu.dtype))
        acceleration_branches = {
            "exact_conservative_chain_rule": exact,
            "strict_leading_p": terms.strict_p0,
            "frozen_mobility": terms.frozen_mobility,
            "mixed_current_mobility_p0_force": terms.current_mobility_p0_force,
            "mixed_p0_mobility_current_force": terms.p0_mobility_current_force,
        }
        nqc_results = {}
        for name, acceleration in acceleration_branches.items():
            nqc_results[name] = assemble_teob_merger_nqc(
                time_grid,
                state,
                amplitude=physical_amplitude,
                phase=physical_phase,
                orbital_frequency=orbital_frequency,
                radial_acceleration=acceleration,
                nu=nu,
                fixed_step=torch.tensor(0.1, dtype=time_grid.dtype),
                targets=targets,
                target_provenance=(
                    PUBLISHED_2017_NQC_PROVENANCE
                    + "; acceleration branch=" + name
                ),
            )
        exact_result = nqc_results["exact_conservative_chain_rule"]
        report["nqc"] = {
            "input_mode_provider": "TEOBRadiationClean.mode",
            "target_provider": "published_2017_nqc_targets",
            "targets": _vector(torch.stack(tuple(targets))),
            "fixed_step": 0.1,
            "branches": {
                name: _nqc_record(
                    name,
                    nqc,
                    acceleration_branches[name],
                    exact,
                    exact_result,
                )
                for name, nqc in nqc_results.items()
            },
        }
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - diagnostic failure is reported
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}

    source_after = {key: _hash(path) for key, path in source_paths.items()}
    cache_after = _hash(args.cache)
    report["source_sha256_after"] = source_after
    report["source_hashes_unchanged"] = source_before == source_after
    report["input_cache"]["sha256_after"] = cache_after  # type: ignore[index]
    report["input_cache"]["unchanged"] = cache_before == cache_after  # type: ignore[index]
    report["elapsed_seconds"] = time.perf_counter() - started_perf

    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
