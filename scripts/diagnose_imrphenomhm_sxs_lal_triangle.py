"""Non-gating, clean-room HM comparison on the frozen 2265 and 3631 fixtures.

Only the public LALSimulation binary API is used. Raw/derived SXS arrays and
the JSON output stay outside Git. The precommitted NR tests and gates are not
altered by this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from scripts import prepare_imrphenomhm_sxs_3631_nr_holdout as prep3631
from scripts import prepare_imrphenomhm_sxs_nr_holdout as prep2265
from diffgw_reference.imrphenomhm_projection import evaluate as project_coefficient


MODES = prep2265.MODES
EXPECTED_FIXTURE_SHA256 = {
    "2265": "d7c53e4450e59ddf21711fff481d4ba152d48be4693050450489bed04c62e7ba",
    "3631": "f51ce5cb5613ca5bb1d61e67876f81fc3ef3a4e7a2f1feeae965505e3a8e8bed",
}
CASES = {
    "2265": dict(prep=prep2265, masses=(60.0, 20.0), f_ref=30.0, early_end=80.0,
                 fixture="hm_sxs_2265_reference.npz"),
    "3631": dict(prep=prep3631, masses=(64.0, 16.0), f_ref=35.0, early_end=85.0,
                 fixture="hm_sxs_3631_reference.npz"),
}
FROZEN_MODE_GATES = {
    (2, 2): (0.97, 0.90, 1.10, 0.995),
    (3, 3): (0.97, 0.90, 1.10, 0.995),
    (2, 1): (0.94, 0.80, 1.20, 0.99),
    (4, 4): (0.94, 0.80, 1.20, 0.99),
    (3, 2): (0.85, 0.75, 1.25, 0.97),
    (4, 3): (0.85, 0.75, 1.25, 0.97),
}
FROZEN_POLARIZATION_GATES = (0.95, 0.85, 1.15, 0.99)
TIME_GRID = -0.1 + np.arange(int(np.floor(0.2 * 8192)) + 1) / 8192
STRONG_FRACTION = 1e-3  # Invariant phase only where both factors are resolved.


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_case(case: str, data_dir: Path) -> dict:
    cfg = CASES[case]
    cfg["prep"].verify_sources(data_dir)
    path = data_dir / cfg["fixture"]
    digest = _sha256(path)
    if digest != EXPECTED_FIXTURE_SHA256[case]:
        raise ValueError(f"{case}: fixture SHA256 mismatch: {digest}")
    with np.load(path) as file:
        f = np.asarray(file["frequencies"], dtype=np.float64)
        if not np.array_equal(f, cfg["prep"].FREQUENCIES):
            raise ValueError(f"{case}: frozen frequency grid mismatch")
        nr3 = {mode: np.array(file[f"Lev3_mode_{mode[0]}_{mode[1]}"])
               for mode in MODES}
        nr2 = {mode: np.array(file[f"Lev2_mode_{mode[0]}_{mode[1]}"])
               for mode in MODES}
        projections = {
            (i, pol): np.array(file[f"Lev3_{pol}_{i}"])
            for i in range(len(cfg["prep"].ANGLES)) for pol in ("plus", "cross")
        }
    return dict(f=f, nr3=nr3, nr2=nr2, projections=projections,
                fixture_sha256=digest)


def _native(backend: str, case: str, f: np.ndarray, *, f_ref: float | None,
            phic: float = 0.0) -> dict:
    m1, m2 = CASES[case]["masses"]
    kwargs = dict(mass1=m1, mass2=m2, chi1=0.0, chi2=0.0,
                  distance=500.0, inclination=0.0, phi0=0.0, phic=phic,
                  tc=0.0, f_ref=f_ref, return_unprojected_modes=True)
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM

        raw = IMRPhenomHM()(f=torch.as_tensor(f, dtype=torch.float64, device="cpu"),
                            **kwargs)[2]
        return {mode: np.asarray(value.detach().cpu().numpy())
                for mode, value in raw.items()}
    import jax
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_hm import IMRPhenomHM

    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices("cpu")[0]):
        raw = IMRPhenomHM.evaluate(f=jnp.asarray(f, dtype=jnp.float64),
                                   **kwargs)[2]
    return {mode: np.asarray(value) for mode, value in raw.items()}


def _lal(case: str, f: np.ndarray, *, f_ref: float,
         phi_ref: float = 0.0) -> dict:
    import lal
    import lalsimulation as ls

    if importlib.metadata.version("lalsuite") != "7.26.15":
        raise RuntimeError("this diagnostic requires pinned LALSuite 7.26.15")
    m1, m2 = CASES[case]["masses"]
    head = ls.SimInspiralChooseFDModes(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, float(f[0]), 512.0, f_ref, phi_ref,
        500.0 * 1e6 * lal.PC_SI, 0.0, lal.CreateDict(), ls.IMRPhenomHM,
    )
    labels = set()
    node = head
    while node is not None:
        labels.add((node.l, node.m))
        node = node.next
    if labels != set(MODES) | {(ell, -m) for ell, m in MODES}:
        raise ValueError(f"unexpected public LAL mode labels: {labels}")
    result = {}
    for ell, m in MODES:
        series = ls.SphHarmFrequencySeriesGetMode(head, ell, m)
        neg = (-f - series.f0) / series.deltaF
        pos = (f - series.f0) / series.deltaF
        if not (np.array_equal(neg, np.rint(neg))
                and np.array_equal(pos, np.rint(pos))):
            raise ValueError("LAL series does not contain the frozen integer grid")
        values = np.asarray(series.data.data)
        if np.any(values[pos.astype(int)] != 0):
            raise ValueError("LAL positive-m positive-frequency branch is nonzero")
        # This fixed public-API adapter was frozen in the maintained binary test.
        result[ell, m] = (-1j) ** m * np.conj(values[neg.astype(int)])
    return result


def _align(f: np.ndarray, reference: np.ndarray,
           candidate: np.ndarray, early_end: float) -> tuple[float, float]:
    mask = (f >= f[0]) & (f <= early_end)
    product = reference[mask] * np.conj(candidate[mask])
    correlations = np.exp(2j * np.pi * TIME_GRID[:, None] * f[None, mask]) @ product
    index = int(np.argmax(np.abs(correlations)))
    return float(TIME_GRID[index]), float(-np.angle(correlations[index]) / 2)


def _aligned(f: np.ndarray, modes: dict, alignment: tuple[float, float]) -> dict:
    dt, phase = alignment
    return {mode: value * np.exp(-2j * np.pi * f * dt - 1j * mode[1] * phase)
            for mode, value in modes.items()}


def _stat(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict:
    a, b = reference[mask], candidate[mask]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not (np.isfinite(na) and np.isfinite(nb) and na > 0 and nb > 0):
        raise ValueError("nonfinite or zero observable norm")
    return {"match": float(np.vdot(a, b).real / (na * nb)),
            "amplitude_ratio": float(nb / na),
            "relative_complex_l2": float(np.linalg.norm(b - a) / na)}


def _frozen_gate(row: dict, gates: tuple[float, float, float, float]) -> bool:
    minimum, low, high, minimum_resolution = gates
    return bool(row["resolution_full"]["match"] >= minimum_resolution
                and row["full"]["match"] >= minimum
                and low <= row["full"]["amplitude_ratio"] <= high)


def _project(modes: dict, angle: tuple[float, float], f: np.ndarray) -> dict:
    plus = np.zeros_like(f, dtype=np.complex128)
    cross = np.zeros_like(f, dtype=np.complex128)
    for mode in MODES:
        cp, cc = project_coefficient(*mode, *angle)
        plus += cp * modes[mode]
        cross += cc * modes[mode]
    return {"plus": plus, "cross": cross}


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.abs(x)


def _carrier_at_mapped(h22: np.ndarray, f: np.ndarray,
                       mapped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate unwrapped 22 phase, as in the original SXS receipts."""
    phase = np.interp(mapped, f, np.unwrap(np.angle(h22)))
    amplitude = np.interp(mapped, f, np.abs(h22))
    return np.exp(1j * phase), amplitude


def _invariant(modes: dict, f: np.ndarray, mode: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """T_lm=u_lm(f)^2 conj[u_22(2f/m)]^m, interpolating 22 phase."""
    m = mode[1]
    mapped = 2 * f / m
    h22 = modes[2, 2]
    carrier_unit, carrier_amplitude = _carrier_at_mapped(h22, f, mapped)
    h = modes[mode]
    valid = ((mapped >= f[0]) & (mapped <= f[-1])
             & (np.abs(h) >= STRONG_FRACTION * np.max(np.abs(h)))
             & (carrier_amplitude >= STRONG_FRACTION * np.max(np.abs(h22))))
    result = np.ones_like(h, dtype=np.complex128)
    result[valid] = _unit(h[valid]) ** 2 * np.conj(carrier_unit[valid]) ** m
    return result, valid


def _circular(values: np.ndarray) -> dict:
    z = np.mean(values)
    return {"mean_phase_rad": float(np.angle(z)),
            "resultant_length": float(abs(z)),
            "rms_phase_rad": float(np.sqrt(np.mean(np.angle(values) ** 2)))}


def _invariant_residuals(nr: dict, model: dict, f: np.ndarray,
                         early_end: float, theta: float) -> dict:
    rows = {}
    for mode in MODES[1:]:
        t_nr, a = _invariant(nr, f, mode)
        t_model, b = _invariant(model, f, mode)
        valid = a & b
        for band, selection in (("early", f <= early_end), ("high", f > early_end)):
            mask = valid & selection
            if not np.any(mask):
                rows[f"{mode[0]}{mode[1]}_{band}"] = {"bins": 0}
                continue
            before = t_model[mask] * np.conj(t_nr[mask])
            after = before * np.exp(1j * (2 - mode[1]) * theta)
            rows[f"{mode[0]}{mode[1]}_{band}"] = {
                "bins": int(np.count_nonzero(mask)),
                "before": _circular(before), "after_common_factor": _circular(after),
            }
    return rows


def _fit_common_factor(nr: dict, model: dict, f: np.ndarray,
                       early_end: float) -> dict:
    pairs = []
    for mode in MODES[1:]:
        if mode[1] == 2:  # Exactly invariant under a common frame factor.
            continue
        t_nr, a = _invariant(nr, f, mode)
        t_model, b = _invariant(model, f, mode)
        mask = a & b & (f <= early_end)
        if not np.any(mask):
            raise ValueError(f"no early invariant bins for {mode}")
        pairs.append((mode[1], t_model[mask] * np.conj(t_nr[mask])))

    def objective(theta: float) -> float:
        return float(np.mean([
            np.mean(abs(value * np.exp(1j * (2 - m) * theta) - 1) ** 2)
            for m, value in pairs
        ]))

    # Survey all periodic basins, then refine each bracket. This is a single
    # unit-complex factor shared by every mode and every source frequency.
    grid = np.linspace(-np.pi, np.pi, 2049)
    scores = np.array([objective(theta) for theta in grid])
    candidates = []
    for i in range(1, len(grid) - 1):
        if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]:
            fit = minimize_scalar(objective, bounds=(grid[i - 1], grid[i + 1]),
                                  method="bounded", options={"xatol": 1e-12})
            candidates.append((float(fit.fun), float(fit.x)))
    candidates.append((float(scores[0]), float(grid[0])))
    loss, theta = min(candidates)
    return {"phase_rad": theta, "real": float(np.cos(theta)),
            "imag": float(np.sin(theta)), "mean_chord_loss": loss,
            "fit_case": "2265", "fit_band_hz": [float(f[0]), early_end],
            "fit_modes": [f"{ell}{m}" for ell, m in MODES[1:] if m != 2]}


def _phase_setting(base: dict, changed: dict, f: np.ndarray,
                   early_end: float, *, orbital_phase_change: bool,
                   orbital_phase_sign: int = 1) -> dict:
    """Check amplitude, measured-ratio identity, and orbital covariance."""
    rows = {}
    for mode in MODES:
        a, b = base[mode], changed[mode]
        strong = np.abs(a) >= STRONG_FRACTION * np.max(np.abs(a))
        amplitude = float(np.max(abs(abs(b[strong]) / abs(a[strong]) - 1)))
        full_ratio = _unit(b[strong] / a[strong])
        mean_ratio = _unit(np.mean(full_ratio))
        setting_row = {
            "max_strong_amplitude_fraction_change": amplitude,
            "mean_mode_phase_shift_rad": float(np.angle(mean_ratio)),
            "max_mode_phase_scatter_chord": float(np.max(abs(full_ratio / mean_ratio - 1))),
        }
        if orbital_phase_change:
            # Native +m*phic follows own assembly. The public LAL mode API
            # output is observed to ignore phi_ref in this black-box call.
            # Neither expected ratio uses the invariant identity.
            expected_phase = orbital_phase_sign * mode[1] * 0.2
            expected_mode_ratio = np.exp(1j * expected_phase)
            setting_row["expected_mode_ratio_phase_rad"] = float(expected_phase)
            setting_row["max_expected_mode_ratio_error"] = float(np.max(
                abs(full_ratio - expected_mode_ratio)))
        if mode == (2, 2):
            rows["22"] = setting_row
            continue
        t_a, va = _invariant(base, f, mode)
        t_b, vb = _invariant(changed, f, mode)
        mask = va & vb & (f <= early_end)
        shift = t_b[mask] * np.conj(t_a[mask])
        # The predicted identity comes from inserting u'=u*r into T, with r
        # measured solely from the two public waveform evaluations.
        m = mode[1]
        mapped = 2 * f / m
        r_mode = _unit(b[mask] / a[mask])
        carrier_a, _ = _carrier_at_mapped(base[2, 2], f, mapped)
        carrier_b, _ = _carrier_at_mapped(changed[2, 2], f, mapped)
        r22 = carrier_b[mask] / carrier_a[mask]
        predicted = r_mode ** 2 * np.conj(r22) ** m
        row = setting_row | {
            "bins": int(np.count_nonzero(mask)),
            "early_invariant_shift": _circular(shift),
            "max_measured_ratio_identity_error": float(np.max(abs(shift - predicted))),
        }
        if orbital_phase_change:
            # London et al.'s m-to-2 orbital-frequency relation makes T
            # invariant for a native orbital azimuth shift. For LAL's mode
            # API, the observed no-response also leaves T unchanged.
            row["expected_invariant_shift_rad"] = 0.0
            row["max_orbital_covariance_error"] = float(np.max(abs(shift - 1)))
        rows[f"{mode[0]}{m}"] = row
    return rows


def _reference_free(base: dict, case: str, f: np.ndarray) -> tuple[dict, dict]:
    """Move the public LAL output using the independently measured native gauge.

    The native model assembles a finite reference contribution proportional to
    (m/2-1); evaluating that same public model with f_ref=None determines its
    unit-modulus multiplier. No NR data or fitted phase enters this map.
    """
    free = {backend: _native(backend, case, f, f_ref=None)
            for backend in ("torch", "jax")}
    gauge = {}
    lal_free = {}
    for mode in MODES:
        native_ratio = free["torch"][mode] / base["torch"][mode]
        unit_ratio = _unit(native_ratio)
        lal_free[mode] = base["lal"][mode] * unit_ratio
        gauge[f"{mode[0]}{mode[1]}"] = {
            "max_native_amplitude_fraction_change": float(
                np.max(abs(abs(native_ratio) - 1))),
            "mean_phase_shift_rad": float(np.angle(np.mean(unit_ratio))),
            "max_phase_scatter_chord": float(np.max(abs(
                unit_ratio / _unit(np.mean(unit_ratio)) - 1))),
        }
    free["lal_gauge_mapped"] = lal_free
    return free, gauge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-2265", type=Path, required=True)
    parser.add_argument("--data-3631", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_dirs = {"2265": args.data_2265, "3631": args.data_3631}
    arrays = {case: _load_case(case, data_dirs[case]) for case in CASES}
    report = {"kind": "non_gating_clean_room_hm_sxs_lal_triangle",
              "versions": {name: importlib.metadata.version(name)
                           for name in ("lalsuite", "numpy", "scipy", "torch", "jax", "jaxlib")},
              "fixed_lal_adapter": "(-1j)**m * conjugate(public_positive_m_series_at_negative_f)",
              "invariant": "T_lm=u_lm(f)^2 conjugate(u_22(2f/m))^m",
              "strong_fraction": STRONG_FRACTION, "cases": {}}
    if report["versions"]["lalsuite"] != "7.26.15":
        raise RuntimeError("pinned LALSuite 7.26.15 required")
    for case, cfg in CASES.items():
        arr = arrays[case]
        f = arr["f"]
        mask_full = np.ones_like(f, dtype=bool)
        bands = {"full": mask_full, "early": f <= cfg["early_end"],
                 "high": f > cfg["early_end"]}
        base = {backend: _native(backend, case, f, f_ref=cfg["f_ref"])
                for backend in ("torch", "jax")}
        base["lal"] = _lal(case, f, f_ref=cfg["f_ref"])
        for backend in base:
            if tuple(base[backend]) != MODES:
                raise ValueError(f"{case} {backend}: wrong mode order")
        resolution_alignment = _align(f, arr["nr3"][2, 2], arr["nr2"][2, 2],
                                      cfg["early_end"])
        aligned_nr2 = _aligned(f, arr["nr2"], resolution_alignment)
        free, gauge = _reference_free(base, case, f)
        result = {"fixture_sha256": arr["fixture_sha256"],
                  "source_md5": cfg["prep"].verify_sources(data_dirs[case]),
                  "f_ref_hz": cfg["f_ref"], "frequency_hz": [float(f[0]), float(f[-1])],
                  "early_end_hz": cfg["early_end"],
                  "resolution_alignment": resolution_alignment,
                  "frontends": {}, "pairwise_raw": {}, "phase_settings": {},
                  "reference_free_gauge": gauge, "reference_free_pairwise": {}}
        for backend, modes in base.items():
            alignment = _align(f, arr["nr3"][2, 2], modes[2, 2], cfg["early_end"])
            aligned = _aligned(f, modes, alignment)
            rows = {}
            for mode in MODES:
                name = f"mode_{mode[0]}_{mode[1]}"
                rows[name] = {band: _stat(arr["nr3"][mode], aligned[mode], mask)
                              for band, mask in bands.items()}
                rows[name]["resolution_full"] = _stat(arr["nr3"][mode],
                                                         aligned_nr2[mode], mask_full)
                rows[name]["frozen_gate_pass"] = _frozen_gate(
                    rows[name], FROZEN_MODE_GATES[mode])
            for i, angle in enumerate(cfg["prep"].ANGLES):
                projected = _project(aligned, angle, f)
                nr2_projected = _project(aligned_nr2, angle, f)
                for pol in ("plus", "cross"):
                    name = f"{pol}_{i}"
                    rows[name] = {band: _stat(arr["projections"][i, pol],
                                              projected[pol], mask)
                                  for band, mask in bands.items()}
                    rows[name]["resolution_full"] = _stat(
                        arr["projections"][i, pol], nr2_projected[pol], mask_full)
                    rows[name]["frozen_gate_pass"] = _frozen_gate(
                        rows[name], FROZEN_POLARIZATION_GATES)
            result["frontends"][backend] = {"alignment": alignment,
                                               "nr_rows": rows}
            changed_phase = (_native(backend, case, f, f_ref=cfg["f_ref"], phic=0.2)
                             if backend != "lal" else
                             _lal(case, f, f_ref=cfg["f_ref"], phi_ref=0.2))
            changed_frequency = (_native(backend, case, f, f_ref=cfg["f_ref"] + 5)
                                 if backend != "lal" else
                                 _lal(case, f, f_ref=cfg["f_ref"] + 5))
            result["phase_settings"][backend] = {
                "phase_plus_0p2_rad": _phase_setting(modes, changed_phase, f,
                                                       cfg["early_end"],
                                                       orbital_phase_change=True,
                                                       orbital_phase_sign=(
                                                           0 if backend == "lal" else 1)),
                "f_ref_plus_5_hz": _phase_setting(modes, changed_frequency, f,
                                                    cfg["early_end"],
                                                    orbital_phase_change=False),
            }
        for left, right in (("torch", "jax"), ("torch", "lal"), ("jax", "lal")):
            result["pairwise_raw"][f"{left}_{right}"] = {
                f"{ell}{m}": _stat(base[right][ell, m], base[left][ell, m], mask_full)
                for ell, m in MODES
            }
        for left, right in (("torch", "jax"), ("torch", "lal_gauge_mapped")):
            result["reference_free_pairwise"][f"{left}_{right}"] = {
                f"{ell}{m}": _stat(free[right][ell, m], free[left][ell, m], mask_full)
                for ell, m in MODES
            }
        report["cases"][case] = result
        arrays[case]["base"] = base
        arrays[case]["free"] = free
    finite_fit = _fit_common_factor(arrays["2265"]["nr3"],
                                    arrays["2265"]["base"]["torch"],
                                    arrays["2265"]["f"], CASES["2265"]["early_end"])
    free_fit = _fit_common_factor(arrays["2265"]["nr3"],
                                  arrays["2265"]["free"]["torch"],
                                  arrays["2265"]["f"], CASES["2265"]["early_end"])
    report["finite_reference_frame_factor"] = finite_fit
    report["reference_free_common_frame_factor"] = free_fit
    report["lal_reference_free_map"] = (
        "Finite-reference LAL modes multiplied by the unit-modulus ratio "
        "of public Torch f_ref=None / finite-f_ref modes, without NR input")
    for case, cfg in CASES.items():
        arr = arrays[case]
        for backend, modes in arr["base"].items():
            report["cases"][case]["frontends"][backend]["finite_ref_invariants"] = (
                _invariant_residuals(arr["nr3"], modes, arr["f"],
                                     cfg["early_end"], finite_fit["phase_rad"])
            )
        report["cases"][case]["reference_free_invariants"] = {
            backend: _invariant_residuals(arr["nr3"], modes, arr["f"],
                                          cfg["early_end"], free_fit["phase_rad"])
            for backend, modes in arr["free"].items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output),
                      "finite_reference_factor": finite_fit,
                      "reference_free_factor": free_fit,
                      "fixture_sha256": {case: arr["fixture_sha256"]
                                         for case, arr in arrays.items()}}, indent=2))


if __name__ == "__main__":
    main()
