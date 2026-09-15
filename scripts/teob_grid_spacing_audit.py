#!/usr/bin/env python3
"""Audit recovered sampled-NQC spacing against the nominal 0.1M policy.

This uses only the four frozen ``continuous.npz`` native caches.  It does not
evaluate a waveform, oracle, FFT, or public/source implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.teob_merger_nqc_clean import (
    PUBLISHED_2017_NQC_PROVENANCE,
    assemble_teob_merger_nqc,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reviews/2026-09-15-teob-fd-attribution"
REPORT = ROOT / "docs/reviews/2026-09-15-teob-grid-nqc/spacing-audit.json"
CASES = {
    "catalog": (1.4, 1.4),
    "bbh_reference": (30.0, 20.0),
    "equal_bbh": (20.0, 20.0),
    "unequal_bns": (1.6, 1.2),
}


def _float(x: torch.Tensor | float) -> float:
    return float(torch.as_tensor(x).detach().cpu().item())


def _coefficients(result) -> np.ndarray:
    return np.concatenate((
        result.coefficients.amplitude.detach().cpu().numpy().reshape(-1),
        result.coefficients.phase.detach().cpu().numpy().reshape(-1),
    ))


def _run(data: np.lib.npyio.NpzFile, masses: tuple[float, float], step: float):
    dtype = torch.float64
    tensor = lambda name: torch.as_tensor(data[name], dtype=dtype)
    m1, m2 = masses
    nu = torch.tensor(m1 * m2 / (m1 + m2) ** 2, dtype=dtype)
    targets = torch.stack(tuple(torch.as_tensor(data[name], dtype=dtype) for name in (
        "nqc_target_amplitude", "nqc_target_amplitude_derivative",
        "nqc_target_frequency", "nqc_target_frequency_derivative",
    )))
    with torch.no_grad():
        result = assemble_teob_merger_nqc(
            tensor("nqc_original_time"), tensor("nqc_original_state"),
            amplitude=tensor("nqc_original_amplitude"),
            phase=tensor("nqc_original_phase"),
            orbital_frequency=tensor("nqc_original_orbital_frequency"),
            radial_acceleration=tensor("nqc_original_radial_acceleration"),
            nu=nu, fixed_step=torch.tensor(step, dtype=dtype), targets=targets,
            target_provenance=PUBLISHED_2017_NQC_PROVENANCE,
            matching_method="grid",
        )
    return result


def main() -> None:
    rows = {}
    for case, masses in CASES.items():
        path = OUT / case / "continuous.npz"
        controls_path = OUT / case / "matching-controls.json"
        data = np.load(path, allow_pickle=False)
        controls = json.loads(controls_path.read_text())
        grid_reference = controls["variants"]["grid"]
        cached_aux = np.asarray(data["nqc_auxiliary_time"], dtype=np.float64)
        diffs = np.diff(cached_aux)
        recovered = float(diffs[0])
        nominal = _run(data, masses, 0.1)
        recovered_result = _run(data, masses, recovered)
        nominal_aux = nominal.aux_time.detach().cpu().numpy()
        recovered_aux = recovered_result.aux_time.detach().cpu().numpy()
        # continuous.npz stores the continuous-policy coefficients.  The
        # frozen grid control is the relevant runtime reference for this audit.
        grid_reference_coeff = np.asarray(grid_reference["nqc_coefficients"], dtype=np.float64).reshape(-1)
        nominal_coeff = _coefficients(nominal)
        recovered_coeff = _coefficients(recovered_result)

        def grid_stats(grid: np.ndarray) -> dict:
            delta = grid - cached_aux
            return {
                "shape": list(grid.shape),
                "exact_equal": bool(np.array_equal(grid, cached_aux)),
                "allclose_1e-12": bool(np.allclose(grid, cached_aux, rtol=0.0, atol=1e-12)),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "nonzero_delta_count": int(np.count_nonzero(delta)),
            }

        rows[case] = {
            "cache": str(path.relative_to(ROOT)),
            "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frozen_grid_reference": {
                "path": str(controls_path.relative_to(ROOT)),
                "nqc_coefficients": grid_reference_coeff.tolist(),
                "t_peak_grid_M": float(grid_reference["peak_M"]),
                "nqc_t_nqc_M": float(grid_reference["peak_M"] - 1.0),
            },
            "nominal_fixed_step_M": 0.1,
            "recovered_fixed_step_M": recovered,
            "cached_auxiliary_spacing": {
                "first": float(diffs[0]), "min": float(diffs.min()),
                "max": float(diffs.max()), "max_abs_from_0.1": float(np.max(np.abs(diffs - 0.1))),
            },
            # continuous.npz records the continuous branch's auxiliary
            # bracketing index; it is not the grid branch's solve index.
            "continuous_cache_bracketing_index": int(np.asarray(data["nqc_matching_index"]).item()),
            "cached_continuous_event_match_time_M": float(np.asarray(data["event_match_time"]).item()),
            "nominal": {
                "auxiliary_grid": grid_stats(nominal_aux),
                "matching_index": int(nominal.matching_index),
                "t_peak_grid_M": _float(nominal.t_peak_grid),
                "t_nqc_M": _float(nominal.t_nqc),
                "selected_knot_time_M": _float(nominal.aux_time[nominal.matching_index]),
                "coefficients": nominal_coeff.tolist(),
                "max_abs_coeff_delta_vs_frozen_grid_reference": float(np.max(np.abs(nominal_coeff - grid_reference_coeff))),
            },
            "recovered": {
                "auxiliary_grid": grid_stats(recovered_aux),
                "matching_index": int(recovered_result.matching_index),
                "t_peak_grid_M": _float(recovered_result.t_peak_grid),
                "t_nqc_M": _float(recovered_result.t_nqc),
                "selected_knot_time_M": _float(recovered_result.aux_time[recovered_result.matching_index]),
                "coefficients": recovered_coeff.tolist(),
                "max_abs_coeff_delta_vs_frozen_grid_reference": float(np.max(np.abs(recovered_coeff - grid_reference_coeff))),
            },
            "nominal_vs_recovered": {
                "max_abs_auxiliary_time_delta": float(np.max(np.abs(nominal_aux - recovered_aux))),
                "max_abs_coefficient_delta": float(np.max(np.abs(nominal_coeff - recovered_coeff))),
                "t_nqc_delta_M": _float(nominal.t_nqc - recovered_result.t_nqc),
                "matching_index_equal": bool(nominal.matching_index == recovered_result.matching_index),
            },
        }
        data.close()

    report = {
        "status": "complete",
        "nominal_policy": {
            "fixed_step_M": 0.1,
            "source_records": [
                "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md:28",
                "docs/reviews/2026-09-13-parity-review/verification/teob-merger-nqc-clean/diagnostic.json:nqc.auxiliary_spacing",
                "scripts/teob_fd_matching_controls.py:34",
            ],
            "runtime_rule": "assemble_teob_merger_nqc uses min(0.5M, fixed_step); these cases use fixed_step=0.1M.",
        },
        "scope": "Cached native NQC arrays only; no waveform, FFT, oracle, or source/reference evaluation.",
        "cases": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({case: {
        "recovered_step": row["recovered_fixed_step_M"],
        "nominal_recovered_coeff_delta": row["nominal_vs_recovered"]["max_abs_coefficient_delta"],
        "nominal_vs_grid_reference": row["nominal"]["max_abs_coeff_delta_vs_frozen_grid_reference"],
        "recovered_vs_grid_reference": row["recovered"]["max_abs_coeff_delta_vs_frozen_grid_reference"],
        "matching_index_equal": row["nominal_vs_recovered"]["matching_index_equal"],
    } for case, row in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
