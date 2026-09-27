"""Post-result, non-gating orbital-alignment diagnostic for SXS:BBH:3619.

Run --prepare with the SXS decoder environment, then --score with PyCBC.
The precommitted resolution gate in the holdout test remains authoritative.
All generated arrays and metrics stay outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prepare_imrphenomhm_sxs_3619_projection import (
    ANGLES,
    C,
    DISTANCE_MPC,
    END_RAMP_M,
    FREQUENCIES,
    GM_SUN,
    NR_MODES,
    PARSEC,
    REFERENCE_TIME_M,
    SAMPLE_RATE,
    START_RAMP_M,
    TOTAL_MASS,
    _harmonic,
    verify_sources,
)


def prepare(data_dir: Path) -> None:
    from scipy.interpolate import CubicSpline
    import sxs

    verify_sources(data_dir)
    waves = {
        level: sxs.load(data_dir / level / "Strain_N2.h5", download=False)
        for level in ("Lev3", "Lev2")
    }
    m_seconds = TOTAL_MASS * GM_SUN / C**3
    amplitude = TOTAL_MASS * GM_SUN / (C**2 * DISTANCE_MPC * 1e6 * PARSEC)
    start_m = max(REFERENCE_TIME_M, *(float(w.t[0]) for w in waves.values()))
    end_m = min(float(w.t[-1]) for w in waves.values())
    grid_m = np.arange(start_m * m_seconds, end_m * m_seconds,
                       1 / SAMPLE_RATE) / m_seconds
    peak = waves["Lev3"]
    h22 = peak.data[:, peak.index(2, 2)]
    interpolated_h22 = (CubicSpline(peak.t, h22.real)(grid_m)
                        + 1j * CubicSpline(peak.t, h22.imag)(grid_m))
    origin_m = grid_m[np.argmax(np.abs(interpolated_h22))]
    window = (np.sin(0.5 * np.pi * np.clip((grid_m - start_m) / START_RAMP_M, 0, 1)) ** 2
              * np.sin(0.5 * np.pi * np.clip((end_m - grid_m) / END_RAMP_M, 0, 1)) ** 2)
    nfft = int(np.ceil(grid_m.size / SAMPLE_RATE) * SAMPLE_RATE)
    if nfft < grid_m.size:
        raise RuntimeError("diagnostic FFT is too short")
    bins = (FREQUENCIES * (nfft / SAMPLE_RATE)).astype(int)
    phase = np.exp(-2j * np.pi * FREQUENCIES * (grid_m[0] - origin_m) * m_seconds)
    basis = np.empty((len(ANGLES), 2, len(NR_MODES), len(FREQUENCIES)),
                     dtype=np.complex128)
    w = waves["Lev2"]
    for j, (ell, m) in enumerate(NR_MODES):
        raw = np.asarray(w.data[:, w.index(ell, m)])
        mode = (CubicSpline(w.t, raw.real)(grid_m)
                + 1j * CubicSpline(w.t, raw.imag)(grid_m))
        for i, (inclination, _) in enumerate(ANGLES):
            projected = mode * _harmonic(ell, m, inclination)
            basis[i, 0, j] = (amplitude / SAMPLE_RATE * phase
                              * np.fft.rfft(window * projected.real, n=nfft)[bins])
            basis[i, 1, j] = (-amplitude / SAMPLE_RATE * phase
                              * np.fft.rfft(window * projected.imag, n=nfft)[bins])
    output = data_dir / "hm_sxs_3619_resolution_orbital_basis.npz"
    np.savez_compressed(output, basis=basis, modes=np.array(NR_MODES),
                        frequencies=FREQUENCIES, origin_m=origin_m)
    print(f"prepared exploratory basis {output}; sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


def score(data_dir: Path) -> None:
    from scipy.optimize import differential_evolution
    from tests.test_imrphenomhm_sxs_3619_projection import _max_time_match, _psd

    source_hashes = verify_sources(data_dir)
    path = data_dir / "hm_sxs_3619_resolution_orbital_basis.npz"
    with np.load(path) as z:
        if not np.array_equal(z["modes"], NR_MODES) or not np.array_equal(z["frequencies"], FREQUENCIES):
            raise RuntimeError("diagnostic basis disagrees with frozen modes/grid")
        basis = z["basis"]
    with np.load(data_dir / "hm_sxs_3619_projection_reference.npz") as z:
        nr = {key: z[key] for key in z.files}
    psd = _psd()
    m = np.array([mode[1] for mode in NR_MODES])[:, None]
    rows = []
    for i, (inclination, source_psi) in enumerate(ANGLES):
        plus, cross = basis[i]
        discrepancy = max(np.max(np.abs(np.sum(plus, axis=0) - nr[f"Lev2_plus_{i}"])),
                          np.max(np.abs(np.sum(cross, axis=0) - nr[f"Lev2_cross_{i}"])))
        scale = max(np.max(np.abs(nr[f"Lev2_plus_{i}"])),
                    np.max(np.abs(nr[f"Lev2_cross_{i}"])))
        if discrepancy / scale > 1e-10:
            raise RuntimeError(f"FFT diagnostic basis does not reproduce frozen fixture: {discrepancy / scale}")
        reference = (np.cos(2 * source_psi) * nr[f"Lev3_plus_{i}"]
                     + np.sin(2 * source_psi) * nr[f"Lev3_cross_{i}"])

        def candidate(phi: float, psi: float) -> np.ndarray:
            c, s = np.cos(m * phi), np.sin(m * phi)
            rotated_plus = np.sum(c * plus - s * cross, axis=0)
            rotated_cross = np.sum(s * plus + c * cross, axis=0)
            return np.cos(2 * psi) * rotated_plus + np.sin(2 * psi) * rotated_cross

        optima = []
        for seed in (17, 23, 41, 67):
            result = differential_evolution(
                lambda x: -_max_time_match(reference, candidate(*x), psd)[0],
                bounds=((0.0, 2 * np.pi), (0.0, np.pi)), seed=seed,
                popsize=12, maxiter=120, tol=1e-9, polish=True, workers=1,
            )
            match, ratio, shift = _max_time_match(reference, candidate(*result.x), psd)
            optima.append(dict(seed=seed, match=match, amplitude_ratio=ratio,
                               time_seconds=shift, orbital_phase_rad=float(result.x[0]),
                               polarization_rad=float(result.x[1]), success=bool(result.success)))
        best = max(optima, key=lambda row: row["match"])
        rows.append(dict(inclination=inclination, source_polarization=source_psi,
                         relative_basis_discrepancy=discrepancy / scale,
                         best=best, spread=max(x["match"] for x in optima)
                         - min(x["match"] for x in optima), seeds=optima))
    receipt = dict(status="post-result non-gating diagnostic",
                   basis_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   source_hashes=source_hashes, rows=rows)
    output = data_dir / "hm_sxs_3619_resolution_orbital_diagnostic.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare", action="store_true")
    stage.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.data_dir.resolve())
    else:
        score(args.data_dir.resolve())


if __name__ == "__main__":
    main()
