"""Precommitted, data-required SXS:BBH:3619 HM projected-strain holdout.

Set DIFFGW_REQUIRE_HM_SXS_3619_PROJECTION=1 and
DIFFGW_HM_SXS_3619_PROJECTION_DIR to enforce the physical gate. A normal
source-only test run skips this external-data comparison.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pytest

from diffgw_reference.imrphenomhm_projection import evaluate as project_coefficient
from scripts.prepare_imrphenomhm_sxs_3619_projection import (
    ANGLES,
    FREQUENCIES,
    verify_sources,
)


MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
FFT_LENGTH = 8192
TIME_INDICES = np.arange(-819, 820, dtype=int)
PHASE_SEEDS = (17, 23, 41, 67)
PROJECTION_GATE = (0.980, 0.85, 1.15)
RESOLUTION_GATE = 0.995


def _strict_data_dir() -> Path:
    if os.environ.get("DIFFGW_REQUIRE_HM_SXS_3619_PROJECTION") != "1":
        pytest.skip("external SXS projected holdout requires its strict flag")
    value = os.environ.get("DIFFGW_HM_SXS_3619_PROJECTION_DIR")
    if not value:
        pytest.fail("strict SXS projection requires DIFFGW_HM_SXS_3619_PROJECTION_DIR")
    data_dir = Path(value).resolve()
    if not data_dir.is_dir():
        pytest.fail(f"strict SXS projection directory missing: {data_dir}")
    return data_dir


def _psd() -> np.ndarray:
    import pycbc.psd

    psd = np.asarray(pycbc.psd.aLIGOZeroDetHighPower(1026, 1.0, 30.0))[
        FREQUENCIES.astype(int)
    ].astype(np.float64)
    if psd.shape != FREQUENCIES.shape or not np.all(np.isfinite(psd)) or np.any(psd <= 0):
        raise RuntimeError("design PSD has absent/nonpositive comparison bins")
    return psd


def _norm_squared(values: np.ndarray, psd: np.ndarray) -> float:
    value = float(4.0 * np.sum(np.abs(values) ** 2 / psd))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("nonfinite or zero PSD-weighted norm")
    return value


def _max_time_match(reference: np.ndarray, candidate: np.ndarray,
                    psd: np.ndarray) -> tuple[float, float, float]:
    """Return optimized real overlap, norm ratio, and arrival-time shift."""
    if not (reference.shape == candidate.shape == psd.shape == FREQUENCIES.shape):
        raise RuntimeError("projected spectrum has an unexpected grid")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise RuntimeError("nonfinite projected spectrum")
    ref_norm = _norm_squared(reference, psd)
    candidate_norm = _norm_squared(candidate, psd)
    cross = np.zeros(FFT_LENGTH // 2 + 1, dtype=np.complex128)
    cross[FREQUENCIES.astype(int)] = np.conj(reference) * candidate / psd
    # irfft uses +2*pi*i*f*k/N; the arrival factor is -2*pi*i*f*dt.
    correlation = 2.0 * FFT_LENGTH * np.fft.irfft(cross, n=FFT_LENGTH)
    scores = correlation[(-TIME_INDICES) % FFT_LENGTH] / np.sqrt(ref_norm * candidate_norm)
    index = int(np.argmax(scores))  # earliest grid time wins an exact tie
    return (float(scores[index]), float(np.sqrt(candidate_norm / ref_norm)),
            float(TIME_INDICES[index] / FFT_LENGTH))


def _native_modes(backend: str) -> dict[tuple[int, int], np.ndarray]:
    kwargs = dict(mass1=250 / 3, mass2=50 / 3, chi1=0.0, chi2=0.0,
                  distance=500.0, inclination=0.0, phi0=0.0,
                  phic=0.0, tc=0.0, f_ref=30.0,
                  return_unprojected_modes=True)
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM

        values = IMRPhenomHM()(
            f=torch.as_tensor(FREQUENCIES, dtype=torch.float64, device="cpu"), **kwargs,
        )[2]
        return {mode: np.asarray(v.detach().cpu().numpy()) for mode, v in values.items()}

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_hm import IMRPhenomHM

    with jax.default_device(jax.devices("cpu")[0]):
        values = IMRPhenomHM.evaluate(
            f=jnp.asarray(FREQUENCIES, dtype=jnp.float64), **kwargs,
        )[2]
    return {mode: np.asarray(v) for mode, v in values.items()}


def _template_basis(modes: dict[tuple[int, int], np.ndarray], inclination: float):
    if tuple(modes) != MODES:
        raise RuntimeError(f"unexpected native mode set: {tuple(modes)}")
    plus, cross = [], []
    for ell, m in MODES:
        cp, cc = project_coefficient(ell, m, inclination, 0.0)
        plus.append(cp * modes[ell, m])
        cross.append(cc * modes[ell, m])
    return np.stack(plus), np.stack(cross)


def _optimized_projection(reference: np.ndarray, basis: tuple[np.ndarray, np.ndarray],
                          psd: np.ndarray) -> dict:
    from scipy.optimize import differential_evolution

    plus, cross = basis
    m_values = np.array([mode[1] for mode in MODES], dtype=np.float64)

    def score(phases: np.ndarray) -> float:
        orbital, polarization = phases
        rotation = np.exp(-1j * m_values * orbital)[:, None]
        candidate = (np.cos(2 * polarization) * np.sum(rotation * plus, axis=0)
                     + np.sin(2 * polarization) * np.sum(rotation * cross, axis=0))
        return _max_time_match(reference, candidate, psd)[0]

    optimized = []
    for seed in PHASE_SEEDS:
        result = differential_evolution(
            lambda x: -score(x), bounds=((0.0, 2 * np.pi), (0.0, np.pi)),
            seed=seed, popsize=12, maxiter=120, tol=1e-9, polish=True,
            workers=1,
        )
        orbital, polarization = (float(v) for v in result.x)
        rotation = np.exp(-1j * m_values * orbital)[:, None]
        candidate = (np.cos(2 * polarization) * np.sum(rotation * plus, axis=0)
                     + np.sin(2 * polarization) * np.sum(rotation * cross, axis=0))
        match, ratio, shift = _max_time_match(reference, candidate, psd)
        optimized.append(dict(seed=seed, match=match, amplitude_ratio=ratio,
                              time_seconds=shift, orbital_phase_rad=orbital,
                              polarization_rad=polarization,
                              scipy_success=bool(result.success)))
    spread = max(item["match"] for item in optimized) - min(item["match"] for item in optimized)
    if spread > 1e-5:
        raise RuntimeError(f"four-seed projection optimization did not converge: {spread}")
    best = max(optimized, key=lambda item: item["match"])
    return dict(best=best, seeds=optimized, spread=spread)


def _resolution_match(reference: np.ndarray, plus: np.ndarray,
                      cross: np.ndarray, psd: np.ndarray) -> dict:
    candidates = []
    for j in range(1024):
        psi = j * np.pi / 1024
        candidate = np.cos(2 * psi) * plus + np.sin(2 * psi) * cross
        match, ratio, time = _max_time_match(reference, candidate, psd)
        candidates.append((match, ratio, time, float(psi)))
    match, ratio, time, psi = max(candidates, key=lambda row: row[0])
    return dict(match=match, amplitude_ratio=ratio, time_seconds=time,
                polarization_rad=psi)


def test_psd_time_maximization_recovers_known_shift():
    """A known arrival shift fixes the FFT correlation sign and scale."""
    reference = np.exp(-((FREQUENCIES - 160.0) / 90.0) ** 2)
    shift = 100 / FFT_LENGTH
    candidate = reference * np.exp(2j * np.pi * FREQUENCIES * shift)
    match, ratio, recovered = _max_time_match(reference, candidate,
                                              np.ones_like(FREQUENCIES))
    assert np.isclose(match, 1.0, rtol=0, atol=1e-12)
    assert np.isclose(ratio, 1.0, rtol=0, atol=1e-12)
    assert recovered == shift


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_sxs_3619_projected_strain(backend: str):
    data_dir = _strict_data_dir()
    source_hashes = verify_sources(data_dir)
    fixture_path = data_dir / "hm_sxs_3619_projection_reference.npz"
    if not fixture_path.is_file():
        pytest.fail(f"strict SXS projection fixture missing: {fixture_path}")
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    preparation_path = data_dir / "hm_sxs_3619_preparation.json"
    if not preparation_path.is_file():
        pytest.fail(f"strict SXS preparation metadata missing: {preparation_path}")
    preparation = json.loads(preparation_path.read_text())
    if (preparation["fixture_sha256"] != fixture_sha256
            or preparation["source_hashes"] != source_hashes):
        pytest.fail("SXS fixture or raw source differs from preparation metadata")
    with np.load(fixture_path) as fixture:
        if not np.array_equal(fixture["frequencies"], FREQUENCIES):
            pytest.fail("SXS fixture frequency grid disagrees with frozen protocol")
        nr = {
            (level, angle_i, polarization): np.array(fixture[f"{level}_{polarization}_{angle_i}"])
            for level in ("Lev3", "Lev2") for angle_i in range(len(ANGLES))
            for polarization in ("plus", "cross")
        }
    psd = _psd()
    psd_sha256 = hashlib.sha256(np.ascontiguousarray(psd, dtype="<f8").tobytes()).hexdigest()
    modes = _native_modes(backend)
    rows = []
    for angle_i, (inclination, source_polarization) in enumerate(ANGLES):
        nr3_plus, nr3_cross = nr["Lev3", angle_i, "plus"], nr["Lev3", angle_i, "cross"]
        nr2_plus, nr2_cross = nr["Lev2", angle_i, "plus"], nr["Lev2", angle_i, "cross"]
        reference = (np.cos(2 * source_polarization) * nr3_plus
                     + np.sin(2 * source_polarization) * nr3_cross)
        resolution = _resolution_match(reference, nr2_plus, nr2_cross, psd)
        optimized = _optimized_projection(reference, _template_basis(modes, inclination), psd)
        best = optimized["best"]
        rows.append(dict(inclination=inclination, source_polarization=source_polarization,
                         resolution=resolution, model=optimized,
                         passed=(resolution["match"] >= RESOLUTION_GATE
                                 and best["match"] >= PROJECTION_GATE[0]
                                 and PROJECTION_GATE[1] <= best["amplitude_ratio"]
                                 <= PROJECTION_GATE[2])))
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pycbc", "torch", "jax")}
    receipt = dict(case="SXS:BBH:3619v3.0/Lev3+Lev2/Strain_N2",
                   backend=backend, source_hashes=source_hashes,
                   fixture_sha256=fixture_sha256, psd_sha256=psd_sha256,
                   preparation_versions=preparation["versions"],
                   evaluation_versions=versions, rows=rows)
    (data_dir / f"hm_sxs_3619_{backend}_projection_metrics.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))
    assert all(row["passed"] for row in rows), "frozen SXS projected-strain gates failed"
