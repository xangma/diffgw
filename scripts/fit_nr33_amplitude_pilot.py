#!/usr/bin/env python3
"""Clean-room, bounded pilot fit of the XHM (3,3) ringdown amplitude.

This script deliberately fits only Eq. (42) of the XHM paper.  It consumes
raw, time-domain NR h33 data and independently supplied QNM metadata; it does
not read model coefficients or waveform outputs from another implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SIGMA_33 = 1.3  # XHM v2, Table 8
DEFAULT_FMAX = 0.3  # Mf, Eq. (41)
DEFAULT_F_RD_FACTOR = 0.95  # amplitude, (3,3), Table 2
PILOT_TRAIN_SUFFIXES = frozenset(("0008", "0169", "0182", "0056", "0063"))
PILOT_HOLDOUT_SUFFIXES = frozenset(("0030", "0166"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FFTControls:
    """Recorded controls for the raw-NR time-to-frequency conversion."""

    pad_factor: int = 4
    start_taper_duration: float = 500.0
    end_taper_duration: float = 20.0
    fmax: float = DEFAULT_FMAX
    sample_dt: float = 0.1

    def __post_init__(self) -> None:
        if self.pad_factor < 1 or int(self.pad_factor) != self.pad_factor:
            raise ValueError("pad_factor must be a positive integer")
        if self.start_taper_duration < 0 or self.end_taper_duration < 0:
            raise ValueError("taper durations must be non-negative")
        if not np.isfinite(self.fmax) or self.fmax <= 0:
            raise ValueError("fmax must be positive and finite")
        if not np.isfinite(self.sample_dt) or self.sample_dt <= 0:
            raise ValueError("sample_dt must be positive and finite")


@dataclass
class Case:
    case_id: str
    path: str
    q: float
    metadata: Dict[str, Any]


def _scalar(value: Any) -> float:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError("QNM metadata must be scalar")
    return float(value.reshape(-1)[0])


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in metadata and metadata[name] is not None:
            return _scalar(metadata[name])
    raise ValueError("missing required metadata; tried " + ", ".join(names))


def _spin_magnitude(metadata: Mapping[str, Any]) -> float:
    """Read scalar remnant spin, accepting the catalog's vector alias."""
    for name in ("final_spin_magnitude", "finalspin_magnitude"):
        if name in metadata and metadata[name] is not None:
            return abs(_scalar(metadata[name]))
    for name in ("final_spin", "finalspin", "remnant_spin", "chi_final"):
        if name not in metadata or metadata[name] is None:
            continue
        value = np.asarray(metadata[name], dtype=float)
        if value.size == 1:
            return abs(float(value.reshape(-1)[0]))
        if value.size == 3:
            return float(np.linalg.norm(value.reshape(-1)))
        raise ValueError("remnant spin must be scalar or a three-vector")
    raise ValueError("missing required metadata; tried final_spin_magnitude, final_spin")


def independent_qnm33(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute QNM frequencies from remnant metadata with the clean-room solver.

    Frequencies returned by the solver use the remnant mass.  Division by the
    supplied remnant/initial mass fraction converts them to the ``Mf`` units
    used by Eq. (42).  Explicit QNM values remain supported for auditability.
    """
    try:
        f_ring = _metadata_value(metadata, "f_ring_33", "f_ring")
        f_damp = _metadata_value(metadata, "f_damp_33", "f_damp")
        return {"f_ring_33": f_ring, "f_damp_33": f_damp,
                "qnm_source": "supplied remnant metadata"}
    except ValueError:
        pass
    spin = _spin_magnitude(metadata)
    mass_fraction = _metadata_value(metadata, "final_mass_fraction", "finalmassfraction",
                                    "remnant_mass_fraction")
    if not 0 <= abs(spin) < 1 or mass_fraction <= 0:
        raise ValueError("remnant spin must satisfy |chi|<1 and mass fraction must be positive")
    try:
        from scripts.kerr_leaver_solver import solve
        import scripts.kerr_leaver_solver as solver_module
    except ModuleNotFoundError:  # direct ``python scripts/...py`` invocation
        from kerr_leaver_solver import solve
        import kerr_leaver_solver as solver_module
    root = solve(spin, 3, 3, 0, strict=True)
    omega = complex(root.omega)
    if not root.converged or omega.real <= 0 or omega.imag >= 0:
        raise ValueError("independent (3,3,0) QNM solver did not converge")
    helper_hash = hashlib.sha256(Path(solver_module.__file__).read_bytes()).hexdigest()
    scale = 2 * np.pi * mass_fraction
    return {"f_ring_33": float(omega.real / scale),
            "f_damp_33": float(-omega.imag / scale),
            "qnm_source": "independent scripts/kerr_leaver_solver.py",
            "qnm_solver": {"omega_remnant": [float(omega.real), float(omega.imag)],
                           "radial_residual": [float(root.radial_residual.real),
                                                float(root.radial_residual.imag)],
                           "cf_depth": int(root.cf_depth), "cf_error": float(root.cf_error),
                           "angular_error": float(root.angular_error),
                           "helper_sha256": helper_hash}}


def read_manifest(path: Path) -> List[Case]:
    """Read a JSON list or ``{"cases": [...]}`` manifest."""
    raw = json.loads(path.read_text())
    entries = raw.get("cases", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("manifest must be a list or an object with a cases list")
    out: List[Case] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("each manifest case must be an object")
        npz_path = Path(entry["path"])
        if not npz_path.is_absolute():
            npz_path = path.parent / npz_path
        metadata = dict(entry.get("metadata", {}))
        for key in ("f_ring_33", "f_damp_33", "f_ring", "f_damp",
                    "final_spin", "finalspin", "final_spin_magnitude", "finalspin_magnitude",
                    "remnant_spin", "chi_final", "final_mass_fraction", "finalmassfraction",
                    "remnant_mass_fraction", "relaxation_time", "peak_time", "sxs_id", "split",
                    "npz_sha256"):
            if key in entry:
                metadata[key] = entry[key]
        if "q" not in entry:
            raise ValueError("case %d has no mass ratio q" % i)
        q = float(entry["q"])
        if not np.isfinite(q) or q <= 1:
            raise ValueError("case %d must have finite q>1 (unequal mass)" % i)
        out.append(Case(str(entry.get("id", "case-%02d" % i)), str(npz_path), q, metadata))
    if len(out) < 4 or len({c.q for c in out}) < 4:
        raise ValueError("at least four distinct q cases are needed for a split")
    if len({c.case_id for c in out}) != len(out):
        raise ValueError("case ids must be unique")
    return out


def load_case(case: Case) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load required ``time`` and ``h33`` arrays and scalar NPZ metadata."""
    expected_hash = case.metadata.get("npz_sha256")
    if expected_hash:
        actual_hash = _sha256_file(Path(case.path))
        if actual_hash.lower() != str(expected_hash).lower():
            raise ValueError("NPZ SHA256 mismatch for " + case.case_id)
    with np.load(case.path, allow_pickle=False) as data:
        if "time" not in data or "h33" not in data:
            raise ValueError("%s must contain time and h33 arrays" % case.path)
        time = np.asarray(data["time"], dtype=float).reshape(-1)
        h33 = np.asarray(data["h33"] , dtype=complex).reshape(-1)
        metadata = dict(case.metadata)
        allowed_metadata = {"f_ring_33", "f_damp_33", "f_ring", "f_damp",
                            "final_spin", "finalspin", "final_spin_magnitude", "finalspin_magnitude",
                            "remnant_spin", "chi_final",
                            "final_mass_fraction", "finalmassfraction", "remnant_mass_fraction",
                            "relaxation_time", "peak_time"}
        for key in data.files:
            if key in allowed_metadata and np.asarray(data[key]).size == 1:
                metadata.setdefault(key, _scalar(data[key]))
    if time.size != h33.size or time.size < 32:
        raise ValueError("time and h33 must have the same length (at least 32)")
    return time, h33, metadata


def _endpoint_window(time: np.ndarray, start: float, end: float) -> np.ndarray:
    """Raised-cosine taper at both endpoints, with a flat central interval."""
    if start < time[0] or end > time[-1] or start >= end:
        raise ValueError("endpoint taper leaves no flat portion of the NR record")
    w = np.ones(time.size)
    left = time < start
    right = time > end
    w[left] = 0.5 * (1 - np.cos(np.pi * (time[left] - time[0]) / (start - time[0])))
    w[right] = 0.5 * (1 + np.cos(np.pi * (time[right] - end) / (time[-1] - end)))
    return w


def frequency_amplitude(time: np.ndarray, h33: np.ndarray,
                        controls: FFTControls, relaxation_time: float = None,
                        peak_time: float = None, return_metadata: bool = False):
    """Return positive-frequency ``(f, |h~|)`` after recorded preprocessing.

    SXS h_lm convention has phase exp(-i m phi), placing positive physical
    frequency power at negative numpy frequencies.  We conjugate h33 before a
    complex FFT and retain f>0. Frequencies are cycles/M and the transform is
    ``dt * FFT(windowed, n=pad_factor*n)``.
    """
    order = np.argsort(time)
    t, h = np.asarray(time)[order], np.asarray(h33)[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, h = t[unique], h[unique]
    if t.size < 32 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(h)):
        raise ValueError("time/h33 must be finite and contain at least 32 unique samples")
    relaxation = t[0] if relaxation_time is None else float(relaxation_time)
    keep = t >= relaxation
    t, h = t[keep], h[keep]
    if t.size < 32:
        raise ValueError("relaxation_time leaves fewer than 32 NR samples")
    native_dt = float(np.median(np.diff(t)))
    if native_dt <= 0:
        raise ValueError("time must increase")
    dt = controls.sample_dt
    n_uniform = int(np.floor((t[-1] - t[0]) / dt)) + 1
    tu = t[0] + np.arange(n_uniform, dtype=float) * dt
    if tu[-1] > t[-1] + 1e-12:
        raise ValueError("uniform grid extrapolates beyond the retained NR record")
    try:
        from scipy.interpolate import CubicSpline
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the pilot fitter requires scipy>=1.8") from exc
    hu = (CubicSpline(t, h.real, extrapolate=False)(tu) +
          1j * CubicSpline(t, h.imag, extrapolate=False)(tu))
    nfft = int(controls.pad_factor) * hu.size
    if nfft < hu.size or controls.pad_factor < 1:
        raise ValueError("pad_factor must be a positive integer")
    # The band is a frequency-domain ringdown region of the full NR record.
    # Truncating at a peak would change the spectrum being fitted. Endpoint
    # durations are explicit so the merger is not suppressed by a whole-record
    # Tukey window. A supplied peak must lie in the flat portion.
    start = relaxation + controls.start_taper_duration
    end = t[-1] - controls.end_taper_duration
    if peak_time is not None and not (start <= peak_time <= end):
        raise ValueError("peak_time is outside the flat FFT window")
    window = _endpoint_window(tu, start, end)
    spectrum = np.fft.fft(np.conj(hu * window), n=nfft) * dt
    freq = np.fft.fftfreq(nfft, dt)
    positive = (freq > 0) & (freq <= controls.fmax)
    if positive.sum() < 8:
        raise ValueError("FFT has too few positive-frequency samples below fmax")
    output = (freq[positive], np.abs(spectrum[positive]))
    if return_metadata:
        return output + ({"native_dt": native_dt, "sample_dt": dt,
                          "n_samples": int(hu.size), "nfft": int(nfft),
                          "retained_time": [float(tu[0]), float(tu[-1])],
                          "taper_flat_window": [float(start), float(end)],
                          "frequency_convention": "conjugate h33; numpy FFT f>0"},)
    return output


def ringdown_shape(f: np.ndarray, f_ring: float, f_damp: float,
                   lam: float, sigma: float = SIGMA_33) -> np.ndarray:
    """Eq. (42), excluding the fitted positive coefficient ``|a_lambda|``."""
    f = np.asarray(f, dtype=float)
    width = f_damp * sigma
    if np.any(f <= 0) or f_ring <= 0 or f_damp <= 0 or width <= 0:
        raise ValueError("frequencies and damping must be positive")
    return (f ** (-1 / 12) * f_damp * sigma /
            ((f - f_ring) ** 2 + width ** 2) *
            np.exp(-(f - f_ring) * lam / width))


def fit_direct(f: np.ndarray, amplitude: np.ndarray, f_ring: float,
               f_damp: float, sigma: float = SIGMA_33) -> Dict[str, float]:
    """Fit ``log(|a_lambda|)`` and ``lambda`` for one raw-NR case."""
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - depends on optional dev dep
        raise RuntimeError("the pilot fitter requires scipy>=1.8") from exc
    f = np.asarray(f, dtype=float)
    y = np.asarray(amplitude, dtype=float)
    good = np.isfinite(f) & np.isfinite(y) & (y > 0)
    if good.sum() < 8:
        raise ValueError("at least eight positive amplitude samples are required")
    f, y = f[good], y[good]
    shape0 = ringdown_shape(f, f_ring, f_damp, 0.0, sigma)
    log_a0 = float(np.log(np.median(y) / np.median(shape0)))
    # Normalize the residual by the case's RMS amplitude, making the L2
    # objective independent of arbitrary NR distance/amplitude units.
    scale = float(np.sqrt(np.mean(y ** 2)))
    def residual(theta: np.ndarray) -> np.ndarray:
        model = np.exp(theta[0]) * ringdown_shape(f, f_ring, f_damp, theta[1], sigma)
        return (model - y) / scale
    result = least_squares(residual, [log_a0, 0.0], bounds=([-30, -10], [30, 10]),
                           loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=2000)
    if not result.success:
        raise RuntimeError("direct Eq.42 fit failed: " + result.message)
    model = np.exp(result.x[0]) * ringdown_shape(f, f_ring, f_damp, result.x[1], sigma)
    return {"log_abs_a_lambda": float(result.x[0]), "lambda": float(result.x[1]),
            "relative_rms": float(np.sqrt(np.mean(((model - y) / scale) ** 2))),
            "n_frequency_samples": int(y.size), "optimizer_status": int(result.status)}


def _sxs_suffix(case: Case) -> str:
    source_id = str(case.metadata.get("sxs_id", case.case_id))
    match = re.search(r"(?:SXS:BBH:)?(\d{4})(?:/|$)", source_id)
    if match:
        return match.group(1)
    digits = "".join(ch for ch in source_id if ch.isdigit())
    return digits[-4:]


def split_by_q(cases: Sequence[Case], holdout_stride: int = 3) -> Tuple[List[Case], List[Case]]:
    """Freeze a deterministic q split before reading/fitting any amplitudes."""
    if holdout_stride < 2:
        raise ValueError("holdout_stride must be at least 2")
    fit_cases = [c for c in cases if c.metadata.get("split") != "validation_only"
                 and "/N2" not in c.case_id]
    if len(fit_cases) < 4:
        raise ValueError("at least four non-validation cases are needed for a split")

    suffix = {c.case_id: _sxs_suffix(c) for c in fit_cases}
    if (set(suffix.values()) == PILOT_TRAIN_SUFFIXES | PILOT_HOLDOUT_SUFFIXES):
        train = [c for c in fit_cases if suffix[c.case_id] in PILOT_TRAIN_SUFFIXES]
        holdout = [c for c in fit_cases if suffix[c.case_id] in PILOT_HOLDOUT_SUFFIXES]
        return train, holdout
    unique_q = sorted({c.q for c in fit_cases})
    # Zero-based index 2,5,... gives q~3,6 for the seven-case pilot catalog.
    holdout_q = {q for i, q in enumerate(unique_q) if i % holdout_stride == 2}
    if not holdout_q:
        holdout_q = {unique_q[-1]}
    train = [c for c in fit_cases if c.q not in holdout_q]
    holdout = [c for c in fit_cases if c.q in holdout_q]
    if len(train) < 3 or not holdout:
        raise ValueError("q split needs at least three training cases and one holdout case")
    return train, holdout


def fit_parameter_space(direct: Sequence[Mapping[str, float]], degree: int = 2) -> Dict[str, Any]:
    """Fit a small polynomial in eta, separately for the two direct coefficients."""
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the pilot fitter requires scipy>=1.8") from exc
    if len(direct) < degree + 1:
        degree = len(direct) - 1
    eta = np.array([d["eta"] for d in direct], dtype=float)
    x = 4 * eta - 1  # bounded, centred coordinate on the nonspinning line
    design = np.column_stack([x ** k for k in range(degree + 1)])
    result: Dict[str, Any] = {"degree": int(degree), "coordinate": "x=4*eta-1"}
    delta = np.array([d["delta"] for d in direct], dtype=float)
    if np.any(delta <= 0):
        raise ValueError("odd-m amplitude parameterization requires q>1")
    for name in ("log_abs_a_lambda", "lambda"):
        y = np.array([d[name] for d in direct], dtype=float)
        if name == "log_abs_a_lambda":
            # Nonspinning odd-m modes vanish at equal mass. This fixed,
            # two-parameter-free normalization is imposed before fitting.
            y = y - np.log(delta)
        fit = least_squares(lambda c: design.dot(c) - y, np.polyfit(x, y, degree)[::-1],
                            loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=2000)
        if not fit.success:
            raise RuntimeError("parameter-space fit failed: " + fit.message)
        suffix = "_over_delta_coefficients" if name == "log_abs_a_lambda" else "_coefficients"
        result[name + suffix] = [float(v) for v in fit.x]
        result[name + "_train_rms"] = float(np.sqrt(np.mean((design.dot(fit.x) - y) ** 2)))
    return result


def _case_direct(case: Case, controls: FFTControls) -> Dict[str, Any]:
    time, h33, metadata = load_case(case)
    qnm = independent_qnm33(metadata)
    f_ring, f_damp = qnm["f_ring_33"], qnm["f_damp_33"]
    relaxation = metadata.get("relaxation_time", None)
    peak_time = metadata.get("peak_time", None)
    f, amp, fft_metadata = frequency_amplitude(
        time, h33, controls,
        None if relaxation is None else float(relaxation),
        None if peak_time is None else float(peak_time), return_metadata=True)
    f_rd = DEFAULT_F_RD_FACTOR * f_ring
    mask = (f >= f_rd) & (f <= controls.fmax)
    if mask.sum() < 8:
        raise ValueError("ringdown band [0.95*f_ring, 0.3] has too few FFT samples")
    # Eq. (42) is A_RD/A0^22.  We retain this normalization in the direct fit.
    eta = case.q / (1 + case.q) ** 2
    # Eq. (42) uses the full leading (2,2) amplitude A_0^22(f), whose
    # frequency dependence is given in Eq. (26), not just its prefactor.
    a0_22 = np.pi * np.sqrt(2 * eta / 3) * (np.pi * f[mask]) ** (-7 / 6)
    fit = fit_direct(f[mask], amp[mask] / a0_22, f_ring, f_damp)
    delta = (case.q - 1) / (case.q + 1)
    fit.update({"case_id": case.case_id, "q": case.q, "eta": eta, "delta": delta,
                "f_ring_33": f_ring, "f_damp_33": f_damp, "f_rd": f_rd,
                "qnm": qnm, "fft": fft_metadata,
                "input_npz_sha256": _sha256_file(Path(case.path))})
    return fit


def fit_manifest(manifest: Path, controls: FFTControls = FFTControls()) -> Dict[str, Any]:
    """Fit training cases and return a frozen calibration artifact.

    Holdout waveforms are intentionally not loaded here.  A downstream
    evaluator must consume this artifact and then read holdout spectra.
    """
    cases = read_manifest(manifest)
    train_cases, holdout_cases = split_by_q(cases)
    train = [_case_direct(c, controls) for c in train_cases]
    fit = fit_parameter_space(train)
    helper_path = Path(__file__).with_name("kerr_leaver_solver.py").resolve()
    result = {"model": "XHM-v2 Eq.42 (3,3) ringdown amplitude only",
              "sigma_33": SIGMA_33, "f_rd_rule": "0.95*f_ring_33",
              "controls": asdict(controls), "train_q": sorted(c.q for c in train_cases),
              "holdout_q": sorted(c.q for c in holdout_cases), "direct_train": train,
              "parameter_fit": fit,
              "provenance": {"manifest_path": str(manifest.resolve()),
                             "manifest_sha256": _sha256_file(manifest),
                             "fitter_script_sha256": _sha256_file(Path(__file__).resolve()),
                             "qnm_helper_path": str(helper_path),
                             "qnm_helper_sha256": _sha256_file(helper_path)},
              "split": {"rule": ("fixed published pilot IDs: 0008,0169,0182,0056,0063 train; "
                                  "0030,0166 hold out" if
                                  set(_sxs_suffix(c) for c in cases
                                      if c.metadata.get("split") != "validation_only"
                                      and "/N2" not in c.case_id) == PILOT_TRAIN_SUFFIXES | PILOT_HOLDOUT_SUFFIXES
                                  else "sorted unique q; zero-based indices 2,5,... hold out"),
                        "train_case_ids": [c.case_id for c in train_cases],
                        "holdout_case_ids": [c.case_id for c in holdout_cases],
                        "excluded_case_ids": [c.case_id for c in cases
                                              if c not in train_cases and c not in holdout_cases]},
              "limitations": ["ringdown subcomponent only", "no full-XHM or LAL parity claim"]}
    return result


def main(argv: Iterable[str] = ()) -> int:
    run_started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad-factor", type=int, default=4)
    parser.add_argument("--start-taper-duration", type=float, default=500.0)
    parser.add_argument("--end-taper-duration", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    args = parser.parse_args(list(argv) if argv else None)
    controls = FFTControls(args.pad_factor, args.start_taper_duration,
                           args.end_taper_duration, args.fmax, args.sample_dt)
    # Persist the split and all numerical controls before opening any waveform
    # arrays. This makes the training/holdout decision auditable and immutable
    # for the ensuing fit.
    cases = read_manifest(args.manifest)
    train_cases, holdout_cases = split_by_q(cases)
    split_path = args.output.with_name(args.output.stem + "-split.json")
    split_record = {"rule": "fixed SXS IDs; N2 validation_only excluded",
                    "train_case_ids": [c.case_id for c in train_cases],
                    "holdout_case_ids": [c.case_id for c in holdout_cases],
                    "excluded_case_ids": [c.case_id for c in cases
                                          if c not in train_cases and c not in holdout_cases],
                    "train_q": sorted(c.q for c in train_cases),
                    "holdout_q": sorted(c.q for c in holdout_cases)}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_record, indent=2, sort_keys=True) + "\n")
    freeze_path = args.output.with_name(args.output.stem + "-freeze.json")
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:  # pragma: no cover
        scipy_version = "unavailable"
    freeze = {"status": "inputs_frozen_before_fit", "manifest_path": str(args.manifest.resolve()),
              "manifest_sha256": _sha256_file(args.manifest), "split_path": str(split_path.resolve()),
              "split_sha256": _sha256_file(split_path), "controls": asdict(controls),
              "fitter_script_sha256": _sha256_file(Path(__file__).resolve()),
              "run_started_at_utc": run_started_at,
              "inputs_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": [str(value) for value in sys.argv],
              "host": socket.gethostname(), "platform": platform.platform(),
              "python_executable": sys.executable, "python_version": platform.python_version(),
              "numpy_version": np.__version__, "scipy_version": scipy_version}
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    result = fit_manifest(manifest=args.manifest, controls=controls)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    artifact_hash = _sha256_file(args.output)
    sidecar = args.output.with_name(args.output.name + ".sha256")
    sidecar.write_text(f"{artifact_hash}  {args.output.name}\n")
    freeze.update({"status": "fit_complete", "calibration_path": str(args.output.resolve()),
                   "calibration_sha256": artifact_hash,
                   "calibration_sidecar": str(sidecar.resolve()),
                   "fit_completed_at_utc": datetime.now(timezone.utc).isoformat()})
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
