"""Run the frozen SXS:BBH:2158v3.0 physical HM qualification.

The binding protocol is
docs/validation/imrphenomhm_sxs_2158_prospective_protocol_20260927.md.
Raw/derived SXS data must remain outside the repository.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

import numpy as np


RECORD = "3gb3n-e5t37"
BASE = f"https://data.caltech.edu/api/records/{RECORD}/files"
FILES = {
    "Lev4:Strain_N2.h5": (2990318, "2c2ccd0a8505ae5d223283dbe39734df"),
    "Lev4:Strain_N2.json": (2232, "3b6e8c5656052f726771a95c0a9b4408"),
    "Lev4:metadata.json": (3347, "6f75989b5a2bb444a4ea612636994da2"),
    "Lev3:Strain_N2.h5": (3280093, "42bb7d97a1531a76d13b3cef0aa82d5c"),
    "Lev3:Strain_N2.json": (2231, "6f1c761e840c02ba4723e37048e35174"),
    "Lev3:metadata.json": (3345, "955f6b41c590f543656f88f34bdc291d"),
}
ANGLES = ((0.7, 0.0), (1.2, 0.4))
FREQUENCIES = np.arange(30.0, 1025.0)
NR_MODES = tuple((ell, m) for ell in range(2, 6) for m in range(-ell, ell + 1))
GM_SUN = 1.3271244e20
C = 299792458.0
PARSEC = 3.0856775814913673e16
TOTAL_MASS = 100.0
DISTANCE_MPC = 500.0
SAMPLE_RATE = 8192.0
REFERENCE_TIME_M = 1384.0
REFERENCE_Q = 3.0001323450539252
REFERENCE_CHI1 = 0.49997718539303
REFERENCE_CHI2 = 0.5000516952582452
START_RAMP_M = 500.0
END_RAMP_M = 100.0
TEMPLATE_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
FFT_LENGTH = 8192
TIME_INDICES = np.arange(-819, 820, dtype=int)
NR_PHASE_GRID = 2048
OPTIMIZER_SEEDS = (17, 23, 41, 67)
RESOLUTION_MATCH_FLOOR = 0.995
RESOLUTION_NORM_BOUNDS = (0.95, 1.05)
MODEL_MATCH_FLOOR = 0.980
MODEL_NORM_BOUNDS = (0.85, 1.15)


def source_path(data_dir: Path, key: str) -> Path:
    level, name = key.split(":", 1)
    return data_dir / level / name


def verify_sources(data_dir: Path) -> dict[str, dict[str, str]]:
    """Verify the official identities before calling the SXS decoder."""
    verified = {}
    for key, (size, md5) in FILES.items():
        path = source_path(data_dir, key)
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"missing/wrong-size official SXS file: {path}")
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != md5:  # noqa: S324
            raise RuntimeError(f"official SXS checksum mismatch: {path}")
        verified[key] = {"md5": md5, "sha256": hashlib.sha256(content).hexdigest()}
    return verified


def download_sources(data_dir: Path) -> dict[str, dict[str, str]]:
    for key, (size, md5) in FILES.items():
        path = source_path(data_dir, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == size:
            if hashlib.md5(path.read_bytes()).hexdigest() == md5:  # noqa: S324
                continue
        path.unlink(missing_ok=True)
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--silent",
             "--show-error", "--output", str(path), f"{BASE}/{quote(key, safe='')}/content"],
            check=True,
        )
    return verify_sources(data_dir)


def _check_sidecars(data_dir: Path) -> dict[str, object]:
    modifications = {}
    for level in ("Lev4", "Lev3"):
        sidecar = json.loads(source_path(data_dir, f"{level}:Strain_N2.json").read_text())
        info = sidecar["data_info"]
        for key, value in dict(data_type="h", spin_weight=-2, m_is_scaled_out=True,
                               r_is_scaled_out=True, ell_min=2, ell_max=8).items():
            if info.get(key) != value:
                raise RuntimeError(f"{level} sidecar {key} disagrees with protocol")
        metadata = json.loads(source_path(data_dir, f"{level}:metadata.json").read_text())
        if abs(metadata["reference_mass1"] / metadata["reference_mass2"] - REFERENCE_Q) > 0.01:
            raise RuntimeError(f"{level} metadata mass ratio disagrees with case")
        for index, expected in ((1, REFERENCE_CHI1), (2, REFERENCE_CHI2)):
            spin = np.asarray(metadata[f"reference_dimensionless_spin{index}"], dtype=float)
            if (spin.shape != (3,) or not np.all(np.isfinite(spin))
                    or np.linalg.norm(spin[:2]) >= 0.01
                    or abs(spin[2] - expected) > 5e-4):
                raise RuntimeError(f"{level} metadata spin{index} disagrees with case")
        if level == "Lev4" and not np.isclose(metadata["reference_time"], REFERENCE_TIME_M):
            raise RuntimeError("Lev4 reference time disagrees with case")
        modifications[level] = sidecar.get("modifications")
    return modifications


@lru_cache(maxsize=None)
def _harmonic(ell: int, m: int, inclination: float) -> complex:
    """Spin -2 Y_lm(iota,0), extending the project's scalar Wigner formula."""
    import sympy as sp
    from sympy.physics.wigner import wigner_d_small

    angle = sp.Float(str(inclination), 40)
    value = (sp.Integer(-1) ** m
             * sp.sqrt(sp.Rational(2 * ell + 1, 4) / sp.pi)
             * wigner_d_small(ell, angle)[ell - m, ell - 2])
    return complex(value.evalf(40))


def build_reference(data_dir: Path) -> dict[str, np.ndarray]:
    """Decode two resolutions and apply the frozen full-NR conditioning."""
    from scipy.interpolate import CubicSpline
    import sxs

    verify_sources(data_dir)
    _check_sidecars(data_dir)
    waves = {
        level: sxs.load(source_path(data_dir, f"{level}:Strain_N2.h5"), download=False)
        for level in ("Lev4", "Lev3")
    }
    for level, wave in waves.items():
        if getattr(wave, "data_type", None) != "h" or getattr(wave, "spin_weight", None) != -2:
            raise RuntimeError(f"{level} decoded conventions disagree with sidecar")
        frame = np.asarray(wave.frame)
        if (getattr(wave, "frame_type", None) != "inertial"
                or frame.shape != (1, 4)
                or not np.allclose(frame, ((1.0, 0.0, 0.0, 0.0),), rtol=0, atol=1e-12)):
            raise RuntimeError(f"{level} is not in the expected inertial identity frame")
        for ell, m in NR_MODES:
            try:
                wave.index(ell, m)
            except Exception as exc:
                raise RuntimeError(f"{level} missing mode {(ell, m)}") from exc

    m_seconds = TOTAL_MASS * GM_SUN / C**3
    distance_m = DISTANCE_MPC * 1e6 * PARSEC
    amplitude_scale = TOTAL_MASS * GM_SUN / (C**2 * distance_m)
    reference_times = [
        float(json.loads(source_path(data_dir, f"{level}:metadata.json").read_text())["reference_time"])
        for level in ("Lev4", "Lev3")
    ]
    start_m = max(*reference_times, *(float(wave.t[0]) for wave in waves.values()))
    end_m = min(float(wave.t[-1]) for wave in waves.values())
    if end_m - start_m <= START_RAMP_M + END_RAMP_M:
        raise RuntimeError("common interval cannot accommodate frozen taper")
    grid_m = np.arange(start_m * m_seconds, end_m * m_seconds, 1 / SAMPLE_RATE) / m_seconds
    if grid_m.size < 2:
        raise RuntimeError("empty uniform NR grid")

    interpolated = {}
    for level, wave in waves.items():
        per_mode = {}
        for ell, m in NR_MODES:
            raw = np.asarray(wave.data[:, wave.index(ell, m)], dtype=np.complex128)
            if not np.all(np.isfinite(raw)):
                raise RuntimeError(f"{level} nonfinite {(ell, m)}")
            per_mode[ell, m] = (CubicSpline(wave.t, raw.real)(grid_m)
                                + 1j * CubicSpline(wave.t, raw.imag)(grid_m))
        interpolated[level] = per_mode

    peak_i = int(np.argmax(np.abs(interpolated["Lev4"][2, 2])))
    time_s = (grid_m - grid_m[peak_i]) * m_seconds
    start_ramp = np.sin(0.5 * np.pi * np.clip((grid_m - start_m) / START_RAMP_M, 0, 1)) ** 2
    end_ramp = np.sin(0.5 * np.pi * np.clip((end_m - grid_m) / END_RAMP_M, 0, 1)) ** 2
    window = start_ramp * end_ramp
    result = {
        "frequencies": FREQUENCIES,
        "time_shift_origin_m": np.array(grid_m[peak_i]),
        "common_start_m": np.array(start_m),
        "common_end_m": np.array(end_m),
        "uniform_samples": np.array(grid_m.size),
        "m_seconds": np.array(m_seconds),
        "amplitude_scale": np.array(amplitude_scale),
    }
    dt = 1 / SAMPLE_RATE
    for level in ("Lev4", "Lev3"):
        channels = []
        for inclination, _source_polarization in ANGLES:
            strain = np.zeros_like(time_s, dtype=np.complex128)
            for ell, m in NR_MODES:
                strain += interpolated[level][ell, m] * _harmonic(ell, m, inclination)
            channels.extend((window * strain.real, -window * strain.imag))
        channel_array = np.stack(channels, axis=1)
        spectrum = np.empty((FREQUENCIES.size, channel_array.shape[1]), dtype=np.complex128)
        for first in range(0, FREQUENCIES.size, 64):
            last = min(first + 64, FREQUENCIES.size)
            kernel = np.exp(-2j * np.pi * FREQUENCIES[first:last, None] * time_s[None, :])
            spectrum[first:last] = amplitude_scale * dt * (kernel @ channel_array)
        for i in range(len(ANGLES)):
            result[f"{level}_plus_{i}"] = spectrum[:, 2 * i]
            result[f"{level}_cross_{i}"] = spectrum[:, 2 * i + 1]
    # A linear Fourier basis permits the common Lev3 azimuth alignment and
    # reports the full-minus-six-mode NR diagnostic without reconditioning.
    # Both zero-phase sums must reproduce the direct Riemann spectra.
    nfft = int(np.ceil(grid_m.size / SAMPLE_RATE) * SAMPLE_RATE)
    bins = (FREQUENCIES * (nfft / SAMPLE_RATE)).astype(int)
    phase = np.exp(-2j * np.pi * FREQUENCIES * time_s[0])
    basis = {}
    for level in ("Lev4", "Lev3"):
        level_basis = np.empty((len(ANGLES), 2, len(NR_MODES), len(FREQUENCIES)),
                               dtype=np.complex128)
        for j, (ell, m) in enumerate(NR_MODES):
            mode = interpolated[level][ell, m]
            for i, (inclination, _) in enumerate(ANGLES):
                projected = mode * _harmonic(ell, m, inclination)
                level_basis[i, 0, j] = (amplitude_scale * dt * phase
                                         * np.fft.rfft(window * projected.real, n=nfft)[bins])
                level_basis[i, 1, j] = (-amplitude_scale * dt * phase
                                         * np.fft.rfft(window * projected.imag, n=nfft)[bins])
        basis[level] = level_basis
    discrepancies = []
    for level in ("Lev4", "Lev3"):
        for i in range(len(ANGLES)):
            for channel_i, name in enumerate(("plus", "cross")):
                direct = result[f"{level}_{name}_{i}"]
                discrepancy = (
                    np.max(np.abs(np.sum(basis[level][i, channel_i], axis=0) - direct))
                    / np.max(np.abs(direct))
                )
                discrepancies.append(float(discrepancy))
    if max(discrepancies) > 1e-10:
        raise RuntimeError(f"orbital basis does not reproduce direct NR: {discrepancies}")
    result["resolution_basis"] = basis["Lev3"]
    result["lev4_mode_basis"] = basis["Lev4"]
    result["resolution_basis_modes"] = np.asarray(NR_MODES, dtype=np.int64)
    result["basis_relative_discrepancies"] = np.asarray(discrepancies)
    return result


def psd_comparison_bins() -> np.ndarray:
    """Build the precommitted design PSD and reject absent comparison bins."""
    import pycbc.psd

    values = np.asarray(pycbc.psd.aLIGOZeroDetHighPower(1026, 1.0, 30.0))[
        FREQUENCIES.astype(int)
    ].astype(np.float64)
    if (values.shape != FREQUENCIES.shape or not np.all(np.isfinite(values))
            or np.any(values <= 0)):
        raise RuntimeError("nonfinite or nonpositive design PSD comparison bin")
    return values


def norm_squared(values: np.ndarray, psd: np.ndarray) -> float:
    """Sum channel norms, allowing one detector or a multi-view network."""
    value = float(4 * np.sum(np.abs(values) ** 2 / psd))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("nonfinite or nonpositive PSD-weighted norm")
    return value


def direct_match(reference: np.ndarray, candidate: np.ndarray,
                 psd: np.ndarray, shift: float) -> float:
    """Independent direct inner-product check at a chosen arrival shift."""
    if reference.shape != candidate.shape or reference.shape[-1] != FREQUENCIES.size:
        raise RuntimeError("direct match has an unexpected spectrum shape")
    numerator = 4 * np.real(np.sum(
        np.conj(reference) * candidate
        * np.exp(-2j * np.pi * FREQUENCIES * shift) / psd
    ))
    return float(numerator / np.sqrt(norm_squared(reference, psd)
                                     * norm_squared(candidate, psd)))


def max_time_match(reference: np.ndarray, candidate: np.ndarray,
                   psd: np.ndarray) -> dict[str, float]:
    """Optimize the shared discrete shift using a zero-padded real inverse FFT."""
    if reference.shape != candidate.shape or reference.shape[-1] != FREQUENCIES.size:
        raise RuntimeError("FFT match has an unexpected spectrum shape")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise RuntimeError("FFT match has a nonfinite spectrum")
    ref_power = norm_squared(reference, psd)
    candidate_power = norm_squared(candidate, psd)
    weighted = np.conj(reference) * candidate / psd
    if weighted.ndim > 1:
        weighted = np.sum(weighted, axis=tuple(range(weighted.ndim - 1)))
    cross = np.zeros(FFT_LENGTH // 2 + 1, dtype=np.complex128)
    cross[FREQUENCIES.astype(int)] = weighted
    # irfft uses positive exp(+2*pi*i*f*k/N); physical arrival uses its negative.
    correlation = 2 * FFT_LENGTH * np.fft.irfft(cross, n=FFT_LENGTH)
    scores = correlation[(-TIME_INDICES) % FFT_LENGTH] / np.sqrt(
        ref_power * candidate_power
    )
    winner = int(np.argmax(scores))
    shift = float(TIME_INDICES[winner] / SAMPLE_RATE)
    match = float(scores[winner])
    direct = direct_match(reference, candidate, psd, shift)
    error = abs(match - direct)
    if error > 1e-12:
        raise RuntimeError(f"FFT/direct inner products disagree: {error}")
    return {
        "match": match,
        "norm_ratio": float(np.sqrt(candidate_power / ref_power)),
        "reference_norm": float(np.sqrt(ref_power)),
        "candidate_norm": float(np.sqrt(candidate_power)),
        "shift_seconds": shift,
        "direct_recheck_error": error,
    }


def synthetic_fft_check() -> dict[str, float]:
    """A nontrivial two-channel exact shift checks FFT sign and normalization."""
    envelope = np.exp(-((FREQUENCIES - 160) / 90) ** 2)
    reference = np.stack((envelope, 0.4j * envelope))
    imposed = 100 / SAMPLE_RATE
    candidate = reference * np.exp(2j * np.pi * FREQUENCIES * imposed)
    result = max_time_match(reference, candidate, np.ones_like(FREQUENCIES))
    if abs(result["match"] - 1) > 1e-12 or result["shift_seconds"] != imposed:
        raise RuntimeError("synthetic FFT shift/sign check failed")
    return result


def rotated_nr_basis(basis: np.ndarray, orbital_phase: float) -> np.ndarray:
    """Rotate all Lev3 modes by one physical orbital phase before projection."""
    if basis.shape != (len(ANGLES), 2, len(NR_MODES), len(FREQUENCIES)):
        raise RuntimeError("NR mode basis has unexpected shape")
    m_values = np.asarray([m for _, m in NR_MODES])
    cosine = np.cos(m_values * orbital_phase)[None, :, None]
    sine = np.sin(m_values * orbital_phase)[None, :, None]
    plus = np.sum(cosine * basis[:, 0] - sine * basis[:, 1], axis=1)
    cross = np.sum(sine * basis[:, 0] + cosine * basis[:, 1], axis=1)
    return np.stack((plus, cross), axis=1)


def align_resolutions(reference: np.ndarray, basis: np.ndarray,
                      psd: np.ndarray) -> dict[str, object]:
    """Select one common orbital phase and arrival shift for all NR checks."""
    from scipy.optimize import minimize_scalar

    if reference.shape != (len(ANGLES), 2, len(FREQUENCIES)):
        raise RuntimeError("NR two-view reference has unexpected shape")

    def evaluate(phase: float) -> dict[str, float]:
        return max_time_match(reference, rotated_nr_basis(basis, phase), psd)

    spacing = 2 * np.pi / NR_PHASE_GRID
    phases = spacing * np.arange(NR_PHASE_GRID)
    grid = np.array([evaluate(float(phase))["match"] for phase in phases])
    local = np.where((grid >= np.roll(grid, 1)) & (grid >= np.roll(grid, -1)))[0]
    selected = sorted(local, key=lambda j: (-grid[j], j))[:8]
    if not selected:
        raise RuntimeError("NR phase grid has no periodic local maximum")
    candidates = [{"source": "grid", "phase_rad": float(phases[int(np.argmax(grid))]),
                   "match": float(np.max(grid))}]
    for j in selected:
        result = minimize_scalar(
            lambda phase: -evaluate(float(phase % (2 * np.pi)))["match"],
            bounds=((j - 1) * spacing, (j + 1) * spacing), method="bounded",
            options={"xatol": 1e-10, "maxiter": 500},
        )
        phase = float(result.x % (2 * np.pi))
        candidates.append({"source": f"local_{int(j)}", "phase_rad": phase,
                           "match": evaluate(phase)["match"],
                           "scipy_success": bool(result.success)})
    best = max(candidates, key=lambda row: row["match"])
    phase = float(best["phase_rad"])
    combined = evaluate(phase)
    shift = combined["shift_seconds"]
    rotated = rotated_nr_basis(basis, phase)
    views = []
    for index, (inclination, source_psi) in enumerate(ANGLES):
        ref, candidate = reference[index], rotated[index]
        match = direct_match(ref, candidate, psd, shift)
        ratio = float(np.sqrt(norm_squared(candidate, psd) / norm_squared(ref, psd)))
        views.append({"inclination_rad": inclination, "source_psi_rad": source_psi,
                      "match": match, "norm_ratio": ratio,
                      "reference_norm": float(np.sqrt(norm_squared(ref, psd))),
                      "candidate_norm": float(np.sqrt(norm_squared(candidate, psd))),
                      "passed": (match >= RESOLUTION_MATCH_FLOOR
                                 and RESOLUTION_NORM_BOUNDS[0] <= ratio
                                 <= RESOLUTION_NORM_BOUNDS[1])})
    return {"orbital_phase_rad": phase, "shift_seconds": shift,
            "combined": combined, "views": views,
            "periodic_local_maxima": int(len(local)), "candidates": candidates}


def native_modes(backend: str) -> dict[tuple[int, int], np.ndarray]:
    """Read native six-mode carriers with the exact frozen physical inputs."""
    kwargs = dict(mass1=TOTAL_MASS * REFERENCE_Q / (1 + REFERENCE_Q),
                  mass2=TOTAL_MASS / (1 + REFERENCE_Q),
                  chi1=REFERENCE_CHI1, chi2=REFERENCE_CHI2, distance=DISTANCE_MPC,
                  inclination=0.0, phi0=0.0, phic=0.0, tc=0.0,
                  f_ref=30.0, return_unprojected_modes=True)
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM

        values = IMRPhenomHM()(
            f=torch.as_tensor(FREQUENCIES, dtype=torch.float64, device="cpu"), **kwargs,
        )[2]
        modes = {mode: np.asarray(value.detach().cpu().numpy())
                 for mode, value in values.items()}
    elif backend == "jax":
        import jax
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_hm import IMRPhenomHM

        jax.config.update("jax_enable_x64", True)
        with jax.default_device(jax.devices("cpu")[0]):
            values = IMRPhenomHM.evaluate(
                f=jnp.asarray(FREQUENCIES, dtype=jnp.float64), **kwargs,
            )[2]
        modes = {mode: np.asarray(value) for mode, value in values.items()}
    else:
        raise ValueError(f"unsupported native backend: {backend}")
    if tuple(modes) != TEMPLATE_MODES:
        raise RuntimeError(f"unexpected native HM mode set: {tuple(modes)}")
    for mode, value in modes.items():
        if value.shape != FREQUENCIES.shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"nonfinite or wrong-shape native mode {mode}")
    return modes


def template_basis(modes: dict[tuple[int, int], np.ndarray],
                   inclination: float) -> tuple[np.ndarray, np.ndarray]:
    from diffgw_reference.imrphenomhm_projection import evaluate

    plus, cross = [], []
    for ell, m in TEMPLATE_MODES:
        cp, cc = evaluate(ell, m, inclination, 0.0)
        plus.append(cp * modes[ell, m])
        cross.append(cc * modes[ell, m])
    return np.stack(plus), np.stack(cross)


def model_detector_spectrum(basis: tuple[np.ndarray, np.ndarray],
                            orbital_phase: float, psi: float) -> np.ndarray:
    plus, cross = basis
    phases = np.exp(-1j * np.array([m for _, m in TEMPLATE_MODES]) * orbital_phase)
    return (np.cos(2 * psi) * np.sum(phases[:, None] * plus, axis=0)
            + np.sin(2 * psi) * np.sum(phases[:, None] * cross, axis=0))


def optimize_model(reference: np.ndarray, basis: tuple[np.ndarray, np.ndarray],
                   psd: np.ndarray) -> dict[str, object]:
    from scipy.optimize import differential_evolution

    def score(angles: np.ndarray) -> float:
        spectrum = model_detector_spectrum(basis, float(angles[0]), float(angles[1]))
        return max_time_match(reference, spectrum, psd)["match"]

    trials = []
    for seed in OPTIMIZER_SEEDS:
        result = differential_evolution(
            lambda angles: -score(angles), bounds=((0, 2 * np.pi), (0, np.pi)),
            seed=seed, popsize=12, maxiter=120, tol=1e-9, polish=True, workers=1,
        )
        orbital, psi = (float(value) for value in result.x)
        spectrum = model_detector_spectrum(basis, orbital, psi)
        trials.append({"seed": seed, "orbital_phase_rad": orbital,
                       "polarization_rad": psi, **max_time_match(reference, spectrum, psd),
                       "scipy_success": bool(result.success)})
    spread = max(row["match"] for row in trials) - min(row["match"] for row in trials)
    return {"best": max(trials, key=lambda row: row["match"]),
            "seeds": trials, "match_spread": spread, "feasible": bool(spread <= 1e-5)}


def omitted_mode_diagnostics(reference: np.ndarray, basis: np.ndarray,
                             psd: np.ndarray) -> dict[str, object]:
    """Describe full NR power outside the six template mode pairs, without retuning."""
    included = {(ell, sign * m) for ell, m in TEMPLATE_MODES for sign in (-1, 1)}
    supported = np.array([mode in included for mode in NR_MODES])
    full = np.sum(basis, axis=2)
    six_mode = np.sum(basis[:, :, supported], axis=2)
    omitted = full - six_mode
    if np.max(np.abs(full - reference)) / np.max(np.abs(reference)) > 1e-10:
        raise RuntimeError("Lev4 mode basis does not reconstruct the full NR reference")
    per_view = []
    for i in range(len(ANGLES)):
        ref_power = norm_squared(reference[i], psd)
        per_view.append({
            "retained_mode_match_no_alignment": direct_match(reference[i], six_mode[i], psd, 0),
            "retained_mode_norm_ratio": float(np.sqrt(norm_squared(six_mode[i], psd) / ref_power)),
            "omitted_residual_norm_fraction": float(np.sqrt(norm_squared(omitted[i], psd) / ref_power)),
        })
    mode_norms = {
        f"{ell},{m}": float(np.sqrt(norm_squared(basis[:, :, j], psd)
                                    / norm_squared(reference, psd)))
        for j, (ell, m) in enumerate(NR_MODES)
    }
    return {"omitted_modes": [list(mode) for mode in NR_MODES if mode not in included],
            "views": per_view, "individual_mode_network_norm_fraction": mode_norms}


def version_record() -> dict[str, str]:
    names = ("numpy", "scipy", "pycbc", "torch", "jax", "jaxlib", "sympy")
    return {"python": sys.version.split()[0],
            **{name: importlib.metadata.version(name) for name in names}}


def evaluate_fixture(data_dir: Path, source_revision: str) -> dict[str, object]:
    """Score both frontends under the frozen gates; always retain red results."""
    fixture_path = data_dir / "hm_sxs_2158_reference.npz"
    preparation_path = data_dir / "hm_sxs_2158_preparation.json"
    preparation = json.loads(preparation_path.read_text())
    source_hashes = verify_sources(data_dir)
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (source_hashes != preparation["source_hashes"]
            or hashlib.sha256(fixture_path.read_bytes()).hexdigest()
            != preparation["fixture_sha256"]
            or script_sha256 != preparation["script_sha256"]):
        raise RuntimeError("raw files, derived fixture, or preparation script differ")
    with np.load(fixture_path) as fixture:
        if (not np.array_equal(fixture["frequencies"], FREQUENCIES)
                or not np.array_equal(fixture["resolution_basis_modes"], NR_MODES)
                or float(np.max(fixture["basis_relative_discrepancies"])) > 1e-10):
            raise RuntimeError("prepared fixture violates frozen grid/mode/reconstruction")
        nr = np.stack([
            np.stack([np.array(fixture[f"{level}_{name}_{i}"])
                      for name in ("plus", "cross")])
            for level in ("Lev4", "Lev3") for i in range(len(ANGLES))
        ]).reshape(2, len(ANGLES), 2, FREQUENCIES.size)
        resolution_basis = np.array(fixture["resolution_basis"])
        lev4_basis = np.array(fixture["lev4_mode_basis"])
        conditioning = {name: float(fixture[name]) for name in (
            "time_shift_origin_m", "common_start_m", "common_end_m",
            "uniform_samples", "m_seconds", "amplitude_scale")}
        reconstruction_errors = fixture["basis_relative_discrepancies"].tolist()
    psd = psd_comparison_bins()
    psd_sha256 = hashlib.sha256(np.ascontiguousarray(psd, dtype="<f8").tobytes()).hexdigest()
    synthetic = synthetic_fft_check()
    resolution = align_resolutions(nr[0], resolution_basis, psd)
    omitted = omitted_mode_diagnostics(nr[0], lev4_basis, psd)
    backend_results = {}
    for backend in ("torch", "jax"):
        modes = native_modes(backend)
        rows = []
        for i, (inclination, source_psi) in enumerate(ANGLES):
            reference = (np.cos(2 * source_psi) * nr[0, i, 0]
                         + np.sin(2 * source_psi) * nr[0, i, 1])
            optimization = optimize_model(reference, template_basis(modes, inclination), psd)
            best = optimization["best"]
            rows.append({"inclination_rad": inclination, "source_psi_rad": source_psi,
                         "optimization": optimization,
                         "passed": bool(optimization["feasible"]
                                        and best["match"] >= MODEL_MATCH_FLOOR
                                        and MODEL_NORM_BOUNDS[0] <= best["norm_ratio"]
                                        <= MODEL_NORM_BOUNDS[1])})
        backend_results[backend] = rows
    resolution_pass = all(row["passed"] for row in resolution["views"])
    model_pass = all(row["passed"] for rows in backend_results.values() for row in rows)
    result = {
        "case": "SXS:BBH:2158v3.0/Lev4+Lev3/Strain_N2",
        "protocol": "docs/validation/imrphenomhm_sxs_2158_prospective_protocol_20260927.md",
        "source_revision": source_revision,
        "source_hashes": source_hashes,
        "metadata_catalog_sha256": "132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6",
        "fixture_sha256": preparation["fixture_sha256"],
        "script_sha256": script_sha256,
        "preparation_versions": preparation["versions"],
        "evaluation_versions": version_record(),
        "sidecar_modifications": preparation["sidecar_modifications"],
        "conditioning": conditioning,
        "basis_relative_discrepancies": reconstruction_errors,
        "psd_sha256": psd_sha256,
        "psd_bins": {"start_hz": 30, "end_hz": 1024, "count": len(psd)},
        "synthetic_fft_check": synthetic,
        "nr_modes": [list(mode) for mode in NR_MODES],
        "native_modes": [list(mode) for mode in TEMPLATE_MODES],
        "resolution": resolution,
        "omitted_mode_diagnostics": omitted,
        "model": backend_results,
        "gates": {"resolution_match_floor": RESOLUTION_MATCH_FLOOR,
                  "resolution_norm_bounds": RESOLUTION_NORM_BOUNDS,
                  "model_match_floor": MODEL_MATCH_FLOOR,
                  "model_norm_bounds": MODEL_NORM_BOUNDS},
        "resolution_pass": resolution_pass,
        "model_pass": model_pass,
        "qualified": bool(resolution_pass and model_pass),
    }
    output = data_dir / "hm_sxs_2158_metrics.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--source-revision", default="unknown")
    args = parser.parse_args()
    if sum((args.verify_only, args.prepare_only, args.evaluate_only)) > 1:
        parser.error("choose at most one of --verify-only/--prepare-only/--evaluate-only")
    data_dir = args.data_dir.resolve()
    source_hashes = (verify_sources(data_dir) if args.evaluate_only
                     else download_sources(data_dir))
    print(json.dumps({"source_hashes": source_hashes}, indent=2), flush=True)
    if args.verify_only:
        return
    if not args.evaluate_only:
        modifications = _check_sidecars(data_dir)
        print(json.dumps({"sidecar_modifications": modifications}, indent=2), flush=True)
        reference = build_reference(data_dir)
        output = data_dir / "hm_sxs_2158_reference.npz"
        np.savez_compressed(output, **reference)
        preparation = {
            "source_hashes": source_hashes,
            "sidecar_modifications": modifications,
            "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {
                "python": sys.version.split()[0],
                **{name: importlib.metadata.version(name)
                   for name in ("numpy", "scipy", "sxs", "sympy")},
            },
        }
        (data_dir / "hm_sxs_2158_preparation.json").write_text(
            json.dumps(preparation, indent=2) + "\n"
        )
        print(f"prepared {output}", flush=True)
    if not args.prepare_only:
        metrics = evaluate_fixture(data_dir, args.source_revision)
        print(json.dumps({"qualified": metrics["qualified"],
                          "resolution_pass": metrics["resolution_pass"],
                          "model_pass": metrics["model_pass"],
                          "metrics": str(data_dir / "hm_sxs_2158_metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
