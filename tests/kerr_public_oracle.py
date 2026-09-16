"""Adapter for the public LAL Kerr ringdown time-domain callable.

This module treats ``SimBlackHoleRingdown`` as a data-producing public API.
The Fourier transform is an explicit composite-Simpson quadrature of the
returned causal samples; no QNM or angular implementation is imported.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _simpson_fft(samples: np.ndarray, delta_t: float, n_fft: int) -> np.ndarray:
    """Integrate causal uniform samples with Simpson weights and zero padding."""
    if (isinstance(n_fft, bool) or not isinstance(n_fft, (int, np.integer))
            or n_fft <= 0):
        raise ValueError("n_fft must be a positive integer")
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("samples must be a one-dimensional sequence of length >= 2")
    if not np.isfinite(y).all() or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("samples and delta_t must be finite; delta_t must be positive")
    # Simpson's rule requires an odd number of points.  The appended zero is
    # outside the public causal support and therefore only completes the rule.
    if y.size % 2 == 0:
        y = np.concatenate((y, np.zeros(1)))
    if n_fft < y.size:
        raise ValueError("n_fft must contain every time-domain sample")
    weights = np.ones(y.size)
    weights[1:-1:2] = 4.
    weights[2:-1:2] = 2.
    padded = np.zeros(n_fft, dtype=np.float64)
    padded[:y.size] = weights * y
    return np.fft.rfft(padded) * (delta_t / 3.)


def public_ringdown_fd(mass_solar: float, spin: float, distance_mpc: float,
                       inclination: float = 0., phase: float = 0.,
                       *, sample_rate: int = 131072, delta_f: float = .25,
                       fractional_mass_loss: float = .01,
                       f_max: float | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(frequency, h_plus, h_cross)`` from the public TD callable.

    Inputs follow the documented public SI interface after conversion.  The
    result uses the physical epoch in the Fourier phase and has no model
    implementation dependency.  ``f_max`` only trims the returned arrays.
    """
    if not all(math.isfinite(float(x)) for x in
               (mass_solar, spin, distance_mpc, inclination, phase, fractional_mass_loss)):
        raise ValueError("ringdown inputs must be finite")
    if not all(math.isfinite(float(x)) for x in (sample_rate, delta_f)):
        raise ValueError("sample_rate and delta_f must be finite")
    if (mass_solar <= 0 or distance_mpc <= 0 or abs(spin) >= 1 or
            sample_rate <= 0 or delta_f <= 0 or int(sample_rate) != sample_rate):
        raise ValueError("invalid ringdown input or sampling configuration")
    if fractional_mass_loss < 0 or fractional_mass_loss >= 1:
        raise ValueError("fractional_mass_loss must be in [0, 1)")
    if f_max is not None and (not math.isfinite(f_max) or f_max <= 0):
        raise ValueError("f_max must be finite and positive")
    if f_max is not None and f_max > sample_rate / 2:
        raise ValueError("f_max exceeds the sampling Nyquist frequency")
    ratio = float(sample_rate) / float(delta_f)
    n_fft = int(round(ratio))
    if not math.isclose(ratio, n_fft, rel_tol=0, abs_tol=1e-10):
        raise ValueError("sample_rate/delta_f must be an integer")
    import lal
    import lalsimulation
    td = lalsimulation.SimBlackHoleRingdown(
        lal.LIGOTimeGPS(0), float(phase), 1. / sample_rate,
        float(mass_solar) * lal.MSUN_SI, float(spin), float(fractional_mass_loss),
        float(distance_mpc) * 1.e6 * lal.PC_SI, float(inclination), 2, 2)
    hp_td, hc_td = td
    n_time = int(hp_td.data.length)
    if int(hc_td.data.length) != n_time or float(hp_td.deltaT) != float(hc_td.deltaT):
        raise ValueError("public polarizations have mismatched support")
    # Keep the requested frequency spacing exactly; an FFT length need not be
    # a power of two for this public-data adapter.
    hp = _simpson_fft(np.asarray(hp_td.data.data, dtype=np.float64), float(hp_td.deltaT), n_fft)
    hc = _simpson_fft(np.asarray(hc_td.data.data, dtype=np.float64), float(hc_td.deltaT), n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=float(hp_td.deltaT))
    epoch = float(hp_td.epoch)
    if epoch != float(hc_td.epoch):
        raise ValueError("public polarizations have mismatched epochs")
    if epoch:
        phase_shift = np.exp(-2j * np.pi * freqs * epoch)
        hp, hc = hp * phase_shift, hc * phase_shift
    if f_max is not None:
        keep = freqs <= f_max
        freqs, hp, hc = freqs[keep], hp[keep], hc[keep]
    return freqs, hp, hc


def public_ringdown_fd_converged(*args, sample_rate=65536, max_sample_rate=524288,
                                 delta_f=.25, comparison_band=(50., 500.), **kwargs):
    """Refine the public Simpson transform until adjacent rates agree."""
    values = (sample_rate, max_sample_rate, delta_f,
              comparison_band[0], comparison_band[1])
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("sampling and comparison-band values must be finite")
    if sample_rate <= 0 or max_sample_rate < 2 * sample_rate:
        raise ValueError("max_sample_rate must be at least twice sample_rate")
    if delta_f <= 0 or comparison_band[0] < 0 or comparison_band[1] <= comparison_band[0]:
        raise ValueError("invalid frequency spacing or comparison band")
    requested_fmax = kwargs.get("f_max")
    available_fmax = float(sample_rate) / 2
    if requested_fmax is not None:
        if not math.isfinite(float(requested_fmax)) or float(requested_fmax) < 0:
            raise ValueError("f_max must be finite and nonnegative")
        available_fmax = min(available_fmax, float(requested_fmax))
    if comparison_band[1] > available_fmax:
        raise ValueError("comparison band exceeds the initial available frequency support")
    if math.ceil(comparison_band[0] / delta_f) * delta_f > comparison_band[1]:
        raise ValueError("comparison band has no frequency-grid sample")
    if int(sample_rate) != sample_rate or int(max_sample_rate) != max_sample_rate:
        raise ValueError("sample rates must be integers")
    rate = int(sample_rate)
    previous = None
    while rate <= int(max_sample_rate):
        current = public_ringdown_fd(*args, sample_rate=rate, delta_f=delta_f, **kwargs)
        if previous is not None:
            f, hp, hc = current
            fp, hp0, hc0 = previous
            mask = (f >= comparison_band[0]) & (f <= comparison_band[1])
            mask0 = (fp >= comparison_band[0]) & (fp <= comparison_band[1])
            if (not mask.any() or not mask0.any() or f[mask][0] > comparison_band[0]
                    or f[mask][-1] < comparison_band[1]
                    or fp[mask0][0] > comparison_band[0]
                    or fp[mask0][-1] < comparison_band[1]):
                raise ValueError("comparison band is not fully represented on both grids")
            if not np.array_equal(f[mask], fp[mask0]):
                raise ValueError("refinement grids do not share exact comparison samples")
            common, ia, ib = np.intersect1d(f[mask], fp[mask0], return_indices=True)
            if common.size == 0:
                raise ValueError("comparison band has no common frequency samples")
            def relative(a, b):
                numerator = np.linalg.norm(a - b)
                denominator = np.linalg.norm(b)
                if denominator == 0:
                    return 0.0 if numerator == 0 else math.inf
                return numerator / denominator
            err = max(relative(hp[mask][ia], hp0[mask0][ib]),
                       relative(hc[mask][ia], hc0[mask0][ib]))
            if not math.isfinite(float(err)):
                raise RuntimeError("non-finite public ringdown convergence error")
            if err <= 1e-7:
                return current + ({"sample_rate": rate, "previous_sample_rate": rate // 2,
                                   "comparison_band": list(comparison_band), "relative_l2": float(err)},)
        previous = current
        rate *= 2
    raise RuntimeError("public ringdown Simpson transform did not converge")


__all__ = ["public_ringdown_fd", "public_ringdown_fd_converged", "_simpson_fft"]
