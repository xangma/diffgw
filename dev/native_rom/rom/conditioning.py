"""Development-only frequency-domain gauge conditioning for ROM training.

The input is a common, resolved positive-frequency geometric grid and complex
waveforms from a separately qualified generator. A local quartic estimates the
phase and phase slope at a fixed reference frequency; both are removed as
extrinsic phase/time gauge. This does not validate the generator or its grid.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConditionedModes:
    log_amplitude: np.ndarray
    phase: np.ndarray
    reference_frequency: float
    removed_phase: np.ndarray
    removed_phase_slope: np.ndarray


def condition_modes(frequency, modes, reference_frequency):
    """Convert complex spectra to log amplitude and reference-gauge phase.

    At least five frequency nodes are needed for the local quartic. The caller
    must ensure the original phase is sampled densely enough to unwrap.
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.complex128)
    f_ref = float(reference_frequency)
    if (frequency.ndim != 1 or frequency.size < 5
            or not np.isfinite(frequency).all() or frequency[0] <= 0
            or not np.all(np.diff(frequency) > 0)
            or not np.isfinite(f_ref) or not frequency[0] <= f_ref <= frequency[-1]):
        raise ValueError("reference frequency must lie on a finite positive increasing grid")
    if (modes.shape[-1:] != frequency.shape or modes.ndim < 1
            or not np.isfinite(modes).all() or np.any(abs(modes) <= 0)):
        raise ValueError("modes must be finite, nonzero, and share the frequency grid")

    phase = np.unwrap(np.angle(modes), axis=-1)
    center = int(np.searchsorted(frequency, f_ref))
    first = min(max(center - 2, 0), len(frequency) - 5)
    nodes = frequency[first:first+5]
    span = nodes[-1] - nodes[0]
    chart = (nodes - f_ref) / span
    vandermonde = np.stack([chart**k for k in range(5)], axis=-1)
    selected = phase[..., first:first+5]
    coefficients = np.linalg.solve(vandermonde, selected.reshape(-1, 5).T)
    reference_phase = coefficients[0].reshape(selected.shape[:-1])
    reference_slope = coefficients[1].reshape(selected.shape[:-1]) / span
    conditioned = phase - reference_phase[..., None] - reference_slope[..., None] * (
        frequency - f_ref)
    return ConditionedModes(np.log(abs(modes)), conditioned, f_ref,
                            reference_phase, reference_slope)
