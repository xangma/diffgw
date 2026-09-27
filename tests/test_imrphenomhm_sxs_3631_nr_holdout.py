"""Frozen, data-required SXS:BBH:3631v3.0 physical HM holdout.

Run with DIFFGW_REQUIRE_HM_SXS_3631_NR=1 and DIFFGW_HM_SXS_3631_NR_DIR pointing to the
external directory prepared by scripts/prepare_imrphenomhm_sxs_3631_nr_holdout.py.
The normal source-only suite skips this external-data test, but a requested
strict run fails on missing data, checksum, dependency, or physical gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_imrphenomhm_sxs_3631_nr_holdout import (
    ANGLES,
    FREQUENCIES,
    MODES,
    verify_sources,
)
from diffgw_reference.imrphenomhm_projection import evaluate as project_coefficient


MODE_GATES = {
    (2, 2): (0.97, 0.90, 1.10, 0.995),
    (3, 3): (0.97, 0.90, 1.10, 0.995),
    (2, 1): (0.94, 0.80, 1.20, 0.99),
    (4, 4): (0.94, 0.80, 1.20, 0.99),
    (3, 2): (0.85, 0.75, 1.25, 0.97),
    (4, 3): (0.85, 0.75, 1.25, 0.97),
}
POLARIZATION_GATES = (0.95, 0.85, 1.15, 0.99)
ALIGNMENT_BAND = (FREQUENCIES >= 35) & (FREQUENCIES <= 85)
TIME_GRID = -0.1 + np.arange(int(np.floor(0.2 * 8192)) + 1) / 8192


def _strict_data_dir() -> Path:
    if os.environ.get("DIFFGW_REQUIRE_HM_SXS_3631_NR") != "1":
        pytest.skip("external SXS holdout requires DIFFGW_REQUIRE_HM_SXS_3631_NR=1")
    value = os.environ.get("DIFFGW_HM_SXS_3631_NR_DIR")
    if not value:
        pytest.fail("strict SXS holdout requires DIFFGW_HM_SXS_3631_NR_DIR")
    data_dir = Path(value).resolve()
    if not data_dir.is_dir():
        pytest.fail(f"strict SXS holdout directory missing: {data_dir}")
    return data_dir


def _align_22(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    freq = FREQUENCIES[ALIGNMENT_BAND]
    product = reference[ALIGNMENT_BAND] * np.conj(candidate[ALIGNMENT_BAND])
    correlations = np.exp(2j * np.pi * TIME_GRID[:, None] * freq[None, :]) @ product
    index = int(np.argmax(np.abs(correlations)))
    return float(TIME_GRID[index]), float(-np.angle(correlations[index]) / 2)


def _apply_alignment(values: np.ndarray, m: int, alignment: tuple[float, float]) -> np.ndarray:
    dt, phase = alignment
    return values * np.exp(-2j * np.pi * FREQUENCIES * dt - 1j * m * phase)


def _stat(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    assert reference.shape == candidate.shape == FREQUENCIES.shape
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate))
    ref_norm = np.linalg.norm(reference)
    candidate_norm = np.linalg.norm(candidate)
    assert ref_norm > 0 and candidate_norm > 0
    match = np.vdot(reference, candidate).real / (ref_norm * candidate_norm)
    return float(match), float(candidate_norm / ref_norm)


def _native_modes(backend: str) -> dict[tuple[int, int], np.ndarray]:
    kwargs = dict(mass1=64.0, mass2=16.0, chi1=0.0, chi2=0.0,
                  distance=500.0, inclination=0.0, phi0=0.0,
                  phic=0.0, tc=0.0, f_ref=35.0,
                  return_unprojected_modes=True)
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM

        modes = IMRPhenomHM()(
            f=torch.as_tensor(FREQUENCIES, dtype=torch.float64, device="cpu"), **kwargs,
        )[2]
        return {mode: np.asarray(values.detach().cpu().numpy()) for mode, values in modes.items()}

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_hm import IMRPhenomHM

    with jax.default_device(jax.devices("cpu")[0]):
        modes = IMRPhenomHM.evaluate(
            f=jnp.asarray(FREQUENCIES, dtype=jnp.float64), **kwargs,
        )[2]
    return {mode: np.asarray(values) for mode, values in modes.items()}


def _project(modes: dict[tuple[int, int], np.ndarray], angle: tuple[float, float]):
    plus = np.zeros_like(FREQUENCIES, dtype=np.complex128)
    cross = np.zeros_like(FREQUENCIES, dtype=np.complex128)
    for ell, m in MODES:
        cp, cc = project_coefficient(ell, m, *angle)
        plus += cp * modes[ell, m]
        cross += cc * modes[ell, m]
    return plus, cross


def _row(name: str, reference: np.ndarray, candidate: np.ndarray,
         numerical: np.ndarray, gates: tuple[float, float, float, float]) -> dict:
    match, ratio = _stat(reference, candidate)
    resolution_match, resolution_ratio = _stat(reference, numerical)
    minimum, low, high, minimum_resolution = gates
    return {
        "name": name, "match": match, "amplitude_ratio": ratio,
        "lev2_lev3_match": resolution_match,
        "lev2_lev3_amplitude_ratio": resolution_ratio,
        "pass": (resolution_match >= minimum_resolution
                 and match >= minimum and low <= ratio <= high),
        "gates": {"match_min": minimum, "ratio_min": low, "ratio_max": high,
                  "resolution_match_min": minimum_resolution},
    }


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_sxs_3631_six_modes_and_polarizations(backend: str):
    data_dir = _strict_data_dir()
    source_md5 = verify_sources(data_dir)
    fixture_path = data_dir / "hm_sxs_3631_reference.npz"
    if not fixture_path.is_file():
        pytest.fail(f"strict SXS holdout fixture missing: {fixture_path}; run prepare script")
    with np.load(fixture_path) as fixture:
        if not np.array_equal(fixture["frequencies"], FREQUENCIES):
            pytest.fail("SXS fixture frequency grid disagrees with frozen protocol")
        nr3 = {mode: np.array(fixture[f"Lev3_mode_{mode[0]}_{mode[1]}"])
               for mode in MODES}
        nr2 = {mode: np.array(fixture[f"Lev2_mode_{mode[0]}_{mode[1]}"])
               for mode in MODES}
        nr_projection = {
            (angle_i, polarization): np.array(fixture[f"Lev3_{polarization}_{angle_i}"])
            for angle_i in range(len(ANGLES))
            for polarization in ("plus", "cross")
        }

    native = _native_modes(backend)
    assert tuple(native) == MODES
    model_alignment = _align_22(nr3[2, 2], native[2, 2])
    resolution_alignment = _align_22(nr3[2, 2], nr2[2, 2])
    aligned_native = {mode: _apply_alignment(native[mode], mode[1], model_alignment)
                      for mode in MODES}
    aligned_nr2 = {mode: _apply_alignment(nr2[mode], mode[1], resolution_alignment)
                   for mode in MODES}
    rows = [
        _row(f"mode_{ell}_{m}", nr3[ell, m], aligned_native[ell, m],
             aligned_nr2[ell, m], MODE_GATES[ell, m])
        for ell, m in MODES
    ]
    for angle_i, angle in enumerate(ANGLES):
        model_plus, model_cross = _project(aligned_native, angle)
        nr2_plus, nr2_cross = _project(aligned_nr2, angle)
        for polarization, measured, numerical in (
            ("plus", model_plus, nr2_plus), ("cross", model_cross, nr2_cross)
        ):
            rows.append(_row(
                f"{polarization}_{angle_i}",
                nr_projection[angle_i, polarization], measured,
                numerical, POLARIZATION_GATES,
            ))

    receipt = {
        "case": "SXS:BBH:3631v3.0/Lev3+Lev2/Strain_N2",
        "backend": backend, "source_md5": source_md5,
        "model_alignment": {"time_seconds": model_alignment[0], "orbital_phase_rad": model_alignment[1]},
        "resolution_alignment": {"time_seconds": resolution_alignment[0],
                                 "orbital_phase_rad": resolution_alignment[1]},
        "rows": rows,
    }
    (data_dir / f"hm_sxs_3631_{backend}_metrics.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    failed = [row["name"] for row in rows if not row["pass"]]
    assert not failed, f"frozen SXS holdout gates failed: {failed}"
