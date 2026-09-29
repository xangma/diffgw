"""Prepare the precommitted SXS:BBH:3630v3.0 projected HM NR holdout.

The binding protocol is
docs/validation/imrphenomhm_sxs_3630_projection_protocol_20260927.md.
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

import numpy as np


RECORD = "28pv5-4wj69"
BASE = f"https://data.caltech.edu/api/records/{RECORD}/files"
FILES = {
    "Lev3:Strain_N2.h5": (3040680, "f415773285072330601c0c39d4d3f0b5"),
    "Lev3:Strain_N2.json": (2227, "67aeb7928db5fdab25bb008465bdbeee"),
    "Lev3:metadata.json": (3364, "6629f45f5a1fd26c0e2b284c18ae57ea"),
    "Lev2:Strain_N2.h5": (3276474, "e29f4cafff41290630179327824ac7ba"),
    "Lev2:Strain_N2.json": (2227, "c136be72e617904f2f1895321406cc5d"),
    "Lev2:metadata.json": (3367, "a4846f6ab9c7a265476eed5245e40b4b"),
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
REFERENCE_TIME_M = 778.0
START_RAMP_M = 500.0
END_RAMP_M = 100.0


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
             "--show-error", "--output", str(path), f"{BASE}/{key}/content"],
            check=True,
        )
    return verify_sources(data_dir)


def _check_sidecars(data_dir: Path) -> dict[str, object]:
    modifications = {}
    for level in ("Lev3", "Lev2"):
        sidecar = json.loads(source_path(data_dir, f"{level}:Strain_N2.json").read_text())
        info = sidecar["data_info"]
        for key, value in dict(data_type="h", spin_weight=-2, m_is_scaled_out=True,
                               r_is_scaled_out=True, ell_min=2, ell_max=8).items():
            if info.get(key) != value:
                raise RuntimeError(f"{level} sidecar {key} disagrees with protocol")
        metadata = json.loads(source_path(data_dir, f"{level}:metadata.json").read_text())
        if abs(metadata["reference_mass1"] / metadata["reference_mass2"] - 6) > 0.01:
            raise RuntimeError(f"{level} metadata mass ratio disagrees with case")
        if (np.linalg.norm(metadata["reference_dimensionless_spin1"]) > 0.001
                or np.linalg.norm(metadata["reference_dimensionless_spin2"]) > 0.001):
            raise RuntimeError(f"{level} metadata spins disagree with case")
        if level == "Lev3" and not np.isclose(metadata["reference_time"], REFERENCE_TIME_M):
            raise RuntimeError("Lev3 reference time disagrees with case")
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
        for level in ("Lev3", "Lev2")
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
    start_m = max(REFERENCE_TIME_M, *(float(wave.t[0]) for wave in waves.values()))
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

    peak_i = int(np.argmax(np.abs(interpolated["Lev3"][2, 2])))
    time_s = (grid_m - grid_m[peak_i]) * m_seconds
    start_ramp = np.sin(0.5 * np.pi * np.clip((grid_m - start_m) / START_RAMP_M, 0, 1)) ** 2
    end_ramp = np.sin(0.5 * np.pi * np.clip((end_m - grid_m) / END_RAMP_M, 0, 1)) ** 2
    window = start_ramp * end_ramp
    result = {"frequencies": FREQUENCIES, "time_shift_origin_m": np.array(grid_m[peak_i])}
    dt = 1 / SAMPLE_RATE
    for level in ("Lev3", "Lev2"):
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
    # Linear Fourier basis for applying one common orbital azimuth to Lev2.
    # Its zero-phase sum must reproduce the independently integrated channels.
    nfft = int(np.ceil(grid_m.size / SAMPLE_RATE) * SAMPLE_RATE)
    bins = (FREQUENCIES * (nfft / SAMPLE_RATE)).astype(int)
    phase = np.exp(-2j * np.pi * FREQUENCIES * time_s[0])
    basis = np.empty((len(ANGLES), 2, len(NR_MODES), len(FREQUENCIES)),
                     dtype=np.complex128)
    for j, (ell, m) in enumerate(NR_MODES):
        mode = interpolated["Lev2"][ell, m]
        for i, (inclination, _) in enumerate(ANGLES):
            projected = mode * _harmonic(ell, m, inclination)
            basis[i, 0, j] = (amplitude_scale * dt * phase
                              * np.fft.rfft(window * projected.real, n=nfft)[bins])
            basis[i, 1, j] = (-amplitude_scale * dt * phase
                              * np.fft.rfft(window * projected.imag, n=nfft)[bins])
    discrepancies = []
    for i in range(len(ANGLES)):
        for channel_i, name in enumerate(("plus", "cross")):
            direct = result[f"Lev2_{name}_{i}"]
            discrepancy = (np.max(np.abs(np.sum(basis[i, channel_i], axis=0) - direct))
                           / np.max(np.abs(direct)))
            discrepancies.append(float(discrepancy))
    if max(discrepancies) > 1e-10:
        raise RuntimeError(f"orbital basis does not reproduce direct NR: {discrepancies}")
    result["resolution_basis"] = basis
    result["resolution_basis_modes"] = np.asarray(NR_MODES, dtype=np.int64)
    result["basis_relative_discrepancies"] = np.asarray(discrepancies)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    source_hashes = download_sources(data_dir)
    print(json.dumps({"source_hashes": source_hashes}, indent=2), flush=True)
    if not args.verify_only:
        modifications = _check_sidecars(data_dir)
        print(json.dumps({"sidecar_modifications": modifications}, indent=2), flush=True)
        reference = build_reference(data_dir)
        output = data_dir / "hm_sxs_3630_projection_reference.npz"
        np.savez_compressed(output, **reference)
        preparation = {
            "source_hashes": source_hashes,
            "sidecar_modifications": modifications,
            "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "versions": {
                "python": sys.version.split()[0],
                **{name: importlib.metadata.version(name)
                   for name in ("numpy", "scipy", "sxs", "sympy")},
            },
        }
        (data_dir / "hm_sxs_3630_preparation.json").write_text(
            json.dumps(preparation, indent=2) + "\n"
        )
        print(f"prepared {output}", flush=True)


if __name__ == "__main__":
    main()
