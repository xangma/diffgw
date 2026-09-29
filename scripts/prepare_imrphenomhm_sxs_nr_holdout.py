"""Prepare the frozen, independent SXS:BBH:2265v3.0 HM NR holdout.

Run from any directory with --data-dir outside the repository. This module
contains data identities and conditioning, but no model coefficients or LAL.
The binding protocol is docs/validation/imrphenomhm_sxs_2265_protocol_20260927.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


RECORD = "258wv-a5t45"
BASE = f"https://data.caltech.edu/api/records/{RECORD}/files"
FILES = {
    "Lev3:Strain_N2.h5": (5894558, "2a7336507c63dbea3b3980cd6d3a783a"),
    "Lev3:Strain_N2.json": (2229, "76283d83e102361309030d4a4f6bd9fb"),
    "Lev3:metadata.json": (3304, "ea2133e45d1075aa4f53068a3cc0e437"),
    "Lev2:Strain_N2.h5": (6246326, "6c98c77914cd7cd30f558b35f6fd7d7c"),
    "Lev2:Strain_N2.json": (2227, "268bf2bd33cb16c7b4b10134da0694ad"),
    "Lev2:metadata.json": (3309, "97a683f1e7f738d81895e09ee6d3236f"),
}
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
ANGLES = ((0.7, 0.3), (1.2, -0.2))
FREQUENCIES = np.arange(30.0, 351.0, 1.0)
GM_SUN = 1.3271244e20
C = 299792458.0
PARSEC = 3.0856775814913673e16
TOTAL_MASS = 80.0
DISTANCE_MPC = 500.0
SAMPLE_RATE = 8192.0
REFERENCE_TIME_M = 1077.5
START_RAMP_M = 500.0
END_RAMP_M = 100.0


def source_path(data_dir: Path, key: str) -> Path:
    level, name = key.split(":", 1)
    return data_dir / level / name


def verify_sources(data_dir: Path) -> dict[str, str]:
    """Require all six official files with the frozen byte counts and MD5s."""
    verified = {}
    for key, (size, md5) in FILES.items():
        path = source_path(data_dir, key)
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"missing/wrong-size official SXS file: {path}")
        # MD5 is the published file identity, not a security digest.
        digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324
        if digest != md5:
            raise RuntimeError(f"official SXS checksum mismatch: {path}")
        verified[key] = digest
    return verified


def download_sources(data_dir: Path) -> dict[str, str]:
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


def _check_sidecars(data_dir: Path) -> None:
    for level in ("Lev3", "Lev2"):
        sidecar = json.loads(source_path(data_dir, f"{level}:Strain_N2.json").read_text())
        info = sidecar["data_info"]
        expected = dict(data_type="h", spin_weight=-2, m_is_scaled_out=True,
                        r_is_scaled_out=True, ell_min=2, ell_max=8)
        for key, value in expected.items():
            if info.get(key) != value:
                raise RuntimeError(f"{level} sidecar {key} disagrees with frozen protocol")
        metadata = json.loads(source_path(data_dir, f"{level}:metadata.json").read_text())
        if level == "Lev3":
            for key, value in (("reference_mass1", 0.7500097310012441),
                               ("reference_mass2", 0.24999389181355253),
                               ("reference_time", REFERENCE_TIME_M)):
                if not np.isclose(metadata[key], value, rtol=1e-7, atol=0):
                    raise RuntimeError(f"{level} metadata {key} disagrees with case")
        if abs(metadata["reference_mass1"] / metadata["reference_mass2"] - 3) > 0.01:
            raise RuntimeError(f"{level} metadata mass ratio disagrees with case")


def _harmonic(ell: int, m: int, inclination: float, physical_azimuth: float) -> complex:
    from diffgw_reference.imrphenomhm_projection import AZIMUTH, IOTA, harmonic

    expression = harmonic(ell, m).subs({IOTA: inclination, AZIMUTH: physical_azimuth})
    return complex(expression.evalf(40))


def build_reference(data_dir: Path) -> dict[str, np.ndarray]:
    """Decode both resolutions and apply exactly the precommitted conditioning."""
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
        for ell, m in MODES:
            for mm in (m, -m):
                try:
                    wave.index(ell, mm)
                except Exception as exc:
                    raise RuntimeError(f"{level} missing mode {(ell, mm)}") from exc

    m_seconds = TOTAL_MASS * GM_SUN / C**3
    distance_m = DISTANCE_MPC * 1e6 * PARSEC
    amplitude_scale = TOTAL_MASS * GM_SUN / (C**2 * distance_m)
    start_m = max(REFERENCE_TIME_M, *(float(wave.t[0]) for wave in waves.values()))
    end_m = min(float(wave.t[-1]) for wave in waves.values())
    if (end_m - start_m) <= START_RAMP_M + END_RAMP_M:
        raise RuntimeError("common interval cannot accommodate frozen taper")
    grid_m = np.arange(start_m * m_seconds, end_m * m_seconds, 1 / SAMPLE_RATE) / m_seconds
    if grid_m.size < 2:
        raise RuntimeError("empty uniform NR grid")

    interpolated = {}
    for level, wave in waves.items():
        per_mode = {}
        for ell, m in MODES:
            for mm in (m, -m):
                raw = np.asarray(wave.data[:, wave.index(ell, mm)], dtype=np.complex128)
                if not np.all(np.isfinite(raw)):
                    raise RuntimeError(f"{level} nonfinite {(ell, mm)}")
                per_mode[ell, mm] = (CubicSpline(wave.t, raw.real)(grid_m)
                                     + 1j * CubicSpline(wave.t, raw.imag)(grid_m))
        interpolated[level] = per_mode

    peak_i = int(np.argmax(np.abs(interpolated["Lev3"][2, 2])))
    time_s = (grid_m - grid_m[peak_i]) * m_seconds
    start_ramp = np.sin(0.5 * np.pi * np.clip((grid_m - start_m) / START_RAMP_M, 0, 1)) ** 2
    end_ramp = np.sin(0.5 * np.pi * np.clip((end_m - grid_m) / END_RAMP_M, 0, 1)) ** 2
    window = start_ramp * end_ramp
    dt = 1 / SAMPLE_RATE
    kernel = np.exp(-2j * np.pi * FREQUENCIES[:, None] * time_s[None, :])
    result = {"frequencies": FREQUENCIES, "time_shift_origin_m": np.array(grid_m[peak_i])}
    for level in ("Lev3", "Lev2"):
        for ell, m in MODES:
            result[f"{level}_mode_{ell}_{m}"] = (
                amplitude_scale * dt * (kernel @ (window * np.conj(interpolated[level][ell, m])))
            )
        for angle_i, (inclination, azimuth) in enumerate(ANGLES):
            strain = np.zeros_like(time_s, dtype=np.complex128)
            for ell, m in MODES:
                # Native phi0 is minus the physical SXS azimuth.
                strain += (interpolated[level][ell, m] * _harmonic(ell, m, inclination, -azimuth)
                           + interpolated[level][ell, -m] * _harmonic(ell, -m, inclination, -azimuth))
            result[f"{level}_plus_{angle_i}"] = amplitude_scale * dt * (kernel @ (window * strain.real))
            result[f"{level}_cross_{angle_i}"] = amplitude_scale * dt * (kernel @ (-window * strain.imag))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    checksums = download_sources(data_dir)
    print(json.dumps({"source_md5": checksums}, indent=2), flush=True)
    if not args.verify_only:
        reference = build_reference(data_dir)
        output = data_dir / "hm_sxs_2265_reference.npz"
        np.savez_compressed(output, **reference)
        print(f"prepared {output}", flush=True)


if __name__ == "__main__":
    main()
