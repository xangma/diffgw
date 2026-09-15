#!/usr/bin/env python3
"""Complex NR (3,3) data extraction for the phase benchmark.

The amplitude pilot intentionally throws away FFT phase.  This module keeps
that phase while retaining its sampling, conjugation, taper, and padding
conventions.  It is data preparation only: no QNM, calibration, fit, or LAL
code is called here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from scripts.fit_nr33_amplitude_pilot import FFTControls, _endpoint_window


@dataclass
class ComplexCase:
    """A complex mode and its shared time coordinate loaded from an NPZ."""

    path: str
    mode_name: str
    time: np.ndarray
    mode: np.ndarray
    h22: Optional[np.ndarray]
    metadata: dict[str, Any]

    @property
    def h33(self) -> np.ndarray:
        """Return the stored h33 mode when this case was loaded as h33."""
        if self.mode_name != "h33":
            raise AttributeError("case was loaded with mode=%r" % self.mode_name)
        return self.mode


@dataclass
class ComplexSpectrum:
    """Positive-frequency complex spectrum and its phase reference."""

    frequency_Mf: np.ndarray
    spectrum: np.ndarray
    spectrum_raw: np.ndarray
    metadata: dict[str, Any]

    @property
    def frequency(self) -> np.ndarray:
        return self.frequency_Mf

    @property
    def spectrum_conjugated_h33(self) -> np.ndarray:
        """Epoch-corrected spectrum of conjugated stored h33."""
        return self.spectrum

    @property
    def raw_spectrum(self) -> np.ndarray:
        """Backward-friendly alias for the uncorrected FFT spectrum."""
        return self.spectrum_raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("metadata values must be scalar")
    return float(array.reshape(-1)[0])


def load_complex_case(
    path: str | Path,
    mode: str = "h33",
    *,
    expected_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ComplexCase:
    """Load a complex mode without discarding its real/imaginary components.

    NPZ files must contain ``time`` and the requested mode.  ``h22`` is loaded
    when present so its peak can define the phase origin.  Optional scalar NPZ
    metadata and the explicit ``metadata`` mapping are retained for callers.
    """
    path = Path(path)
    if mode not in {"h22", "h33"}:
        raise ValueError("mode must be h22 or h33")
    if expected_sha256 is not None:
        actual = _sha256_file(path)
        if actual.lower() != str(expected_sha256).lower():
            raise ValueError("NPZ SHA256 mismatch for %s" % path)
    with np.load(path, allow_pickle=False) as data:
        if "time" not in data or mode not in data:
            raise ValueError("%s must contain time and %s arrays" % (path, mode))
        time = np.asarray(data["time"], dtype=float).reshape(-1)
        mode_data = np.asarray(data[mode], dtype=complex).reshape(-1)
        h22 = (np.asarray(data["h22"], dtype=complex).reshape(-1)
               if "h22" in data else None)
        meta = dict(metadata or {})
        # Keep only scalar sidecar-like values; object metadata is forbidden by
        # allow_pickle=False and therefore cannot silently enter the cache.
        for key in data.files:
            if key in {"time", "h22", "h33"}:
                continue
            if np.asarray(data[key]).size == 1:
                try:
                    meta.setdefault(key, _scalar(data[key]))
                except (TypeError, ValueError):
                    pass
    if time.size != mode_data.size or time.size < 32:
        raise ValueError("time and mode must have the same length (at least 32)")
    if h22 is not None and h22.size != time.size:
        raise ValueError("h22 must have the same length as time")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(mode_data)):
        raise ValueError("time and complex mode must be finite")
    if h22 is not None and not np.all(np.isfinite(h22)):
        raise ValueError("h22 must be finite")
    if h22 is not None:
        peak = float(time[int(np.argmax(np.abs(h22)))])
        meta.setdefault("h22_peak_time", peak)
        meta.setdefault("peak_time", peak)
    meta["h22_peak_time_convention"] = (
        "input time coordinate at the sample maximizing |h22|; no time shift"
        if h22 is not None else "not available in input NPZ")
    if expected_sha256 is not None:
        meta["npz_sha256"] = str(expected_sha256).lower()
    return ComplexCase(str(path), mode, time, mode_data, h22, meta)


def _prepare_uniform(
    time: np.ndarray,
    mode: np.ndarray,
    controls: FFTControls,
    relaxation_time: float | None,
    peak_time: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    time_array = np.asarray(time, dtype=float).reshape(-1)
    mode_array = np.asarray(mode, dtype=complex).reshape(-1)
    if time_array.size != mode_array.size:
        raise ValueError("time and mode must have the same length")
    order = np.argsort(time_array)
    t = time_array[order]
    h = mode_array[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, h = t[unique], h[unique]
    if t.size < 32 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(h)):
        raise ValueError("time/mode must be finite and contain at least 32 unique samples")
    relaxation = t[0] if relaxation_time is None else float(relaxation_time)
    if not np.isfinite(relaxation):
        raise ValueError("relaxation_time must be finite")
    keep = t >= relaxation
    t, h = t[keep], h[keep]
    if t.size < 32:
        raise ValueError("relaxation_time leaves fewer than 32 samples")
    dt = float(controls.sample_dt)
    duration_steps = (t[-1] - t[0]) / dt
    # A translated copy of an otherwise uniform record can land one ulp
    # below its integer duration (for example 1998.9999999999998).  Resolve
    # that representation artifact so the phase benchmark is epoch invariant.
    nearest_steps = round(duration_steps)
    if abs(duration_steps - nearest_steps) <= 1e-9:
        duration_steps = float(nearest_steps)
    n_uniform = int(np.floor(duration_steps)) + 1
    tu = t[0] + np.arange(n_uniform, dtype=float) * dt
    if tu.size < 32 or tu[-1] > t[-1] + 1e-12:
        raise ValueError("uniform grid extrapolates beyond retained record")
    try:
        from scipy.interpolate import CubicSpline
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("phase extraction requires scipy>=1.8") from exc
    hu = CubicSpline(t, h.real, extrapolate=False)(tu)
    hu = hu + 1j * CubicSpline(t, h.imag, extrapolate=False)(tu)
    start = relaxation + float(controls.start_taper_duration)
    end = t[-1] - float(controls.end_taper_duration)
    if peak_time is not None and not (start <= float(peak_time) <= end):
        raise ValueError("h22 peak_time is outside the flat FFT window")
    window = _endpoint_window(tu, start, end)
    return tu, hu * window, {
        "relaxation_time": float(relaxation),
        "retained_time": [float(tu[0]), float(tu[-1])],
        "taper_flat_window": [float(start), float(end)],
        "native_dt": float(np.median(np.diff(t))),
        "sample_dt": dt,
        "n_samples": int(hu.size),
        "nfft": int(controls.pad_factor * hu.size),
    }


def transform_complex(
    time: Sequence[float] | np.ndarray,
    mode: Sequence[complex] | np.ndarray,
    controls: FFTControls | None = None,
    *,
    relaxation_time: float | None = None,
    h22_peak_time: float | None = None,
) -> ComplexSpectrum:
    """Transform complex h33 using the frozen amplitude-pilot preprocessing.

    The stored SXS h33 convention is conjugated before the numpy FFT, so its
    positive physical frequencies are retained.  ``raw_spectrum`` is relative
    to the first retained sample ``fft_epoch``.  ``spectrum`` applies
    ``exp(-2*pi*i*f*(fft_epoch-h22_peak_time))`` and is therefore referenced to
    the h22 peak at time zero.  The returned spectrum is for conjugated h33;
    the equivalent negative-m mode is ``h_(3,-3) = -conj(h33)``.
    """
    controls = FFTControls() if controls is None else controls
    tu, windowed, prep = _prepare_uniform(
        np.asarray(time), np.asarray(mode), controls, relaxation_time, h22_peak_time)
    nfft = int(controls.pad_factor) * windowed.size
    if nfft < windowed.size:
        raise ValueError("pad_factor must be a positive integer")
    raw_full = np.fft.fft(np.conj(windowed), n=nfft) * float(controls.sample_dt)
    full_frequency = np.fft.fftfreq(nfft, float(controls.sample_dt))
    positive = (full_frequency > 0) & (full_frequency <= float(controls.fmax))
    if int(positive.sum()) < 8:
        raise ValueError("FFT has too few positive-frequency samples below fmax")
    frequency = full_frequency[positive]
    raw = raw_full[positive]
    fft_epoch = float(tu[0])
    if h22_peak_time is None:
        phase_reference = "fft_epoch"
        correction = np.ones_like(frequency, dtype=complex)
    else:
        peak = float(h22_peak_time)
        if not np.isfinite(peak):
            raise ValueError("h22_peak_time must be finite")
        phase_reference = "h22_peak"
        correction = np.exp(-2j * np.pi * frequency * (fft_epoch - peak))
    corrected = raw * correction
    prep.update({
        "fft_epoch": fft_epoch,
        "h22_peak_time": (None if h22_peak_time is None else float(h22_peak_time)),
        "phase_reference": phase_reference,
        "epoch_correction": "exp(-2*pi*i*f*(fft_epoch-h22_peak_time))",
        "frequency_convention": "conjugate h33; numpy FFT f>0",
        "positive_mode_mapping": "h_(3,-3) = -conj(h33)",
        "spectrum_units": "time-domain amplitude multiplied by sample_dt",
    })
    return ComplexSpectrum(frequency, corrected, raw, prep)


def transform_case(
    case: ComplexCase,
    controls: FFTControls | None = None,
    *,
    relaxation_time: float | None = None,
    h22_peak_time: float | None = None,
) -> ComplexSpectrum:
    """Transform a loaded case, using its recorded h22 peak by default."""
    if h22_peak_time is None:
        value = case.metadata.get("h22_peak_time", case.metadata.get("peak_time"))
        h22_peak_time = None if value is None else float(value)
    if relaxation_time is None and "relaxation_time" in case.metadata:
        relaxation_time = float(case.metadata["relaxation_time"])
    return transform_complex(case.time, case.mode, controls,
                             relaxation_time=relaxation_time,
                             h22_peak_time=h22_peak_time)


def interpolate_complex_common_grid(
    frequency: Sequence[float] | np.ndarray,
    spectrum: Sequence[complex] | np.ndarray,
    common_frequency: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Linearly interpolate real and imaginary parts, rejecting extrapolation."""
    f = np.asarray(frequency, dtype=float).reshape(-1)
    z = np.asarray(spectrum, dtype=complex).reshape(-1)
    grid = np.asarray(common_frequency, dtype=float).reshape(-1)
    if f.size < 2 or f.size != z.size or grid.size == 0:
        raise ValueError("frequency, spectrum, and common grid have incompatible sizes")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(grid)):
        raise ValueError("frequency, spectrum, and common grid must be finite")
    order = np.argsort(f)
    f, z = f[order], z[order]
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequency must contain unique samples")
    if np.any(np.diff(grid) < 0):
        raise ValueError("common grid must be increasing")
    if grid[0] < f[0] or grid[-1] > f[-1]:
        raise ValueError("common grid requires extrapolation")
    return np.interp(grid, f, z.real) + 1j * np.interp(grid, f, z.imag)


# Short aliases for callers that prefer the operation name first.
load_complex_mode = load_complex_case
common_grid_interpolate = interpolate_complex_common_grid
