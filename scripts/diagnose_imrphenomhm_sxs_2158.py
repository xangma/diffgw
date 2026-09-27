"""Reproduce the non-gating, fixed-alignment SXS:BBH:2158 residual audit.

Reuse the frozen runner and fixture without optimizing or rewriting either.
Only the public LAL binary API is used; no oracle-fitted corrections are made.
Run as ``python -m scripts.diagnose_imrphenomhm_sxs_2158`` with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from scripts import run_imrphenomhm_sxs_2158 as frozen


METRICS_SHA256 = "a1497fbbda34d7cb73d14dfde0113b4dd0c4acc513abfbe2ded4bb4716c79526"
BAND_EDGES = (30, 60, 100, 200, 400, 1025)  # Diagnostic bins, not physics joins or gates.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual_budget(reference, candidate, psd):
    """Exact amplitude/phase split of weighted squared complex residual.

    The phase term is 2 |r||c| (1-cos(delta)); angle(0)=0 is harmless
    because its weight vanishes. No phase unwrapping or fitted offsets.
    All powers are normalized by the reference power on this same grid.
    """
    reference, candidate, psd = map(np.asarray, (reference, candidate, psd))
    if (reference.shape != candidate.shape or reference.shape != psd.shape
            or reference.ndim != 1 or not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(psd))
            or np.any(psd <= 0)):
        raise ValueError("finite equal-length vectors and positive PSD required")
    a, b = np.abs(reference), np.abs(candidate)
    n = np.sum(a * a / psd)
    if n <= 0:
        raise ValueError("nonzero reference required")
    candidate_power = np.sum(b * b / psd)
    amplitude = np.sum((a - b) ** 2 / psd) / n
    phase = np.sum(4 * a * b * np.sin(np.angle(candidate * reference.conj()) / 2) ** 2 / psd) / n
    residual = np.sum(np.abs(candidate - reference) ** 2 / psd) / n
    np.testing.assert_allclose(amplitude + phase, residual, rtol=1e-12, atol=1e-14)
    return {
        "match": float(np.sum((reference.conj() * candidate).real / psd)
                       / np.sqrt(n * candidate_power)) if candidate_power > 0 else None,
        "norm_ratio": float(np.sqrt(candidate_power / n)),
        "residual_power": float(residual),
        "amplitude_residual_power": float(amplitude),
        "phase_residual_power": float(phase),
    }


def frequency_budget(reference, candidate, psd):
    total = frozen.norm_squared(reference, psd)
    rows = []
    for low, high in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        mask = (frozen.FREQUENCIES >= low) & (frozen.FREQUENCIES < high)
        fraction = frozen.norm_squared(reference[mask], psd[mask]) / total
        budget = residual_budget(reference[mask], candidate[mask], psd[mask])
        rows.append({"low_hz_inclusive": low, "high_hz_exclusive": high,
                     "reference_power_fraction": fraction,
                     **{k: v * fraction for k, v in budget.items() if k.endswith("power")}})
    np.testing.assert_allclose(sum(r["reference_power_fraction"] for r in rows), 1, atol=1e-14)
    np.testing.assert_allclose(sum(r["residual_power"] for r in rows),
                               residual_budget(reference, candidate, psd)["residual_power"],
                               rtol=1e-12, atol=1e-14)
    return rows


def lal_modes():
    import lal
    import lalsimulation as ls

    if version("lalsuite") != "7.26.15":
        raise RuntimeError("pinned LALSuite 7.26.15 required")
    q = frozen.REFERENCE_Q
    m2 = frozen.TOTAL_MASS / (1 + q)
    head = ls.SimInspiralChooseFDModes(
        q * m2 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0, 0, frozen.REFERENCE_CHI1, 0, 0, frozen.REFERENCE_CHI2,
        1.0, 30.0, 2048.0, 30.0, 0.0,
        frozen.DISTANCE_MPC * 1e6 * lal.PC_SI, 0.0, lal.CreateDict(), ls.IMRPhenomHM,
    )
    result = {}
    for ell, m in frozen.TEMPLATE_MODES:
        series = ls.SphHarmFrequencySeriesGetMode(head, ell, m)
        index = (-frozen.FREQUENCIES - series.f0) / series.deltaF
        np.testing.assert_array_equal(index, np.rint(index))
        if np.any(index < 0) or np.any(index >= len(series.data.data)):
            raise ValueError("LAL frequency grid does not cover diagnostic")
        # Fixed adapter from the maintained six-mode binary test, not fitted here.
        result[ell, m] = (-1j) ** m * np.conj(series.data.data[index.astype(int)])
    return result


def diagnose(data_dir, metrics_path):
    if sha256(metrics_path) != METRICS_SHA256:
        raise ValueError("frozen metrics hash mismatch")
    metrics = json.loads(metrics_path.read_text())
    fixture = data_dir / "hm_sxs_2158_reference.npz"
    if (sha256(fixture) != metrics["fixture_sha256"]
            or sha256(frozen.__file__) != metrics["script_sha256"]
            or frozen.verify_sources(data_dir) != metrics["source_hashes"]):
        raise ValueError("frozen input identity mismatch")
    psd = frozen.psd_comparison_bins()
    if hashlib.sha256(np.asarray(psd, dtype="<f8").tobytes()).hexdigest() != metrics["psd_sha256"]:
        raise ValueError("frozen PSD mismatch")
    native = {backend: frozen.native_modes(backend) for backend in ("torch", "jax")}
    binary = lal_modes()
    binary_rows = {}
    for mode, ref in binary.items():
        support = np.abs(ref) >= 0.01 * np.max(np.abs(ref))
        error = native["torch"][mode] - ref
        binary_rows[str(mode)] = {
            "relative_complex_l2": float(np.linalg.norm(error) / np.linalg.norm(ref)),
            "max_supported_relative_error": float(np.max(np.abs(error[support] / ref[support]))),
            "torch_jax_relative_l2": float(np.linalg.norm(native["torch"][mode] - native["jax"][mode])
                                           / np.linalg.norm(ref)),
        }
    views = []
    with np.load(fixture) as data:
        np.testing.assert_array_equal(data["frequencies"], frozen.FREQUENCIES)
        labels = [tuple(mode) for mode in data["resolution_basis_modes"]]
        for view, (iota, source_psi) in enumerate(frozen.ANGLES):
            best = metrics["model"]["torch"][view]["optimization"]["best"]
            shift = np.exp(-2j * np.pi * frozen.FREQUENCIES * best["shift_seconds"])
            orbit = np.exp(-1j * np.array([m for _, m in frozen.TEMPLATE_MODES]) * best["orbital_phase_rad"])
            psi = best["polarization_rad"]
            plus, cross = frozen.template_basis(native["torch"], iota)
            model_pairs = orbit[:, None] * (np.cos(2 * psi) * plus + np.sin(2 * psi) * cross) * shift
            candidate = model_pairs.sum(axis=0)
            binary_candidate = frozen.model_detector_spectrum(
                frozen.template_basis(binary, iota), best["orbital_phase_rad"], psi) * shift
            reference = (np.cos(2 * source_psi) * data[f"Lev4_plus_{view}"]
                         + np.sin(2 * source_psi) * data[f"Lev4_cross_{view}"])
            basis = data["lev4_mode_basis"][view]
            nr_pairs = []
            for ell, m in frozen.TEMPLATE_MODES:
                selected = [labels.index((ell, m)), labels.index((ell, -m))]
                nr_pairs.append(np.cos(2 * source_psi) * basis[0, selected].sum(axis=0)
                                + np.sin(2 * source_psi) * basis[1, selected].sum(axis=0))
            six = np.sum(nr_pairs, axis=0)
            budget = residual_budget(reference, candidate, psd)
            np.testing.assert_allclose([budget["match"], budget["norm_ratio"]],
                                       [best["match"], best["norm_ratio"]], atol=1e-12, rtol=0)
            views.append({
                "inclination_rad": iota, "source_psi_rad": source_psi,
                "fixed_alignment": {k: best[k] for k in ("orbital_phase_rad", "polarization_rad", "shift_seconds")},
                "model_full_nr": budget,
                "binary_full_nr": residual_budget(reference, binary_candidate, psd),
                "six_nr_full_nr": residual_budget(reference, six, psd),
                "model_six_nr": residual_budget(six, candidate, psd),
                "frequency_bands": frequency_budget(reference, candidate, psd),
                "projected_mode_pairs": {
                    str(mode): {**residual_budget(nr, model, psd),
                                "nr_power_fraction_of_full": frozen.norm_squared(nr, psd)
                                / frozen.norm_squared(reference, psd),
                                "frequency_bands": frequency_budget(nr, model, psd)}
                    for mode, nr, model in zip(frozen.TEMPLATE_MODES, nr_pairs, model_pairs)
                },
            })
    return {"case": "SXS:BBH:2158v3.0", "scope": "post-result diagnostic; no new qualification gates",
            "metrics_sha256": METRICS_SHA256, "fixture_sha256": metrics["fixture_sha256"],
            "frozen_runner_sha256": metrics["script_sha256"], "diagnostic_sha256": sha256(__file__),
            "psd_sha256": metrics["psd_sha256"],
            "versions": {name: version(name) for name in ("numpy", "scipy", "torch", "jax", "jaxlib", "sympy", "pycbc", "lalsuite")},
            "binary_modes": binary_rows, "views": views}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metrics = Path(__file__).resolve().parents[1] / "docs/validation/imrphenomhm_sxs_2158_metrics_20260927.json"
    result = diagnose(args.data_dir, metrics)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "matches": [v["model_full_nr"]["match"] for v in result["views"]]}))


if __name__ == "__main__":
    main()
