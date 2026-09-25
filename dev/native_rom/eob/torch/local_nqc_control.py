"""Opt-in local matching diagnostics for the development SEOBNRv4 route."""

from __future__ import annotations

import math

import torch

from torchwave.approximants.eob.nqc import nqc_22_basis
from torchwave.approximants.eob.numerical_attachment import (
    _eval_batch_scalar,
    _unwrap_mode_phase,
    natural_cubic_spline,
)


def stationary_peak_time(time, orbital_frequency):
    """Find the negative-curvature spline root nearest the sampled maximum."""
    spline = natural_cubic_spline(time, orbital_frequency)
    sampled_maximum = int(torch.argmax(orbital_frequency))
    roots = []
    for index in range(max(0, sampled_maximum - 16),
                       min(time.numel() - 1, sampled_maximum + 16)):
        left, right = float(time[index]), float(time[index + 1])
        left_slope = float(_eval_batch_scalar(spline, time.new_tensor(left), derivative=1))
        right_slope = float(_eval_batch_scalar(spline, time.new_tensor(right), derivative=1))
        if left_slope < 0 or right_slope > 0:
            continue
        for _ in range(48):
            middle = (left + right) / 2
            slope = float(_eval_batch_scalar(spline, time.new_tensor(middle), derivative=1))
            if slope > 0:
                left = middle
            else:
                right = middle
        root = (left + right) / 2
        if float(_eval_batch_scalar(spline, time.new_tensor(root), derivative=2)) < 0:
            roots.append(root)
    if not roots:
        raise ValueError("no negative-curvature stationary orbital peak")
    root = min(roots, key=lambda coordinate: abs(coordinate - float(time[sampled_maximum])))
    if abs(float(_eval_batch_scalar(spline, time.new_tensor(root), derivative=1))) > 1e-8:
        raise ValueError("stationary orbital peak root residual exceeds tolerance")
    return time.new_tensor(root)


def local_nqc_controls(time, state, orbital_frequency, base_mode, match_time,
                       *, half_width_M, degree):
    """Fit the amplitude, phase-basis, and phase-RHS derivatives together."""
    if not math.isfinite(float(half_width_M)) or half_width_M <= 0 or degree < 2:
        raise ValueError("invalid local NQC polynomial")
    selected = (time >= match_time - half_width_M) & (time <= match_time + half_width_M)
    chart = (time[selected] - match_time) / half_width_M
    if (chart.numel() < degree + 3
            or not bool((chart.min() < 0) & (chart.max() > 0))):
        raise ValueError("local NQC fit requires two-sided samples")
    amplitude = base_mode.abs()
    amplitude_basis, phase_basis = nqc_22_basis(
        state[:, 0], state[:, 2], orbital_frequency)
    values = torch.cat((amplitude[:, None], amplitude[:, None] * amplitude_basis,
                        phase_basis, _unwrap_mode_phase(base_mode)[:, None]), dim=-1)
    matrix = torch.stack(tuple(chart**power for power in range(degree + 1)), dim=-1)
    coefficients = torch.linalg.lstsq(matrix, values[selected], driver="gels").solution
    first = coefficients[1] / half_width_M
    second = 2 * coefficients[2] / half_width_M**2
    return (torch.stack((first[1:4], second[1:4])),
            torch.stack((first[4:6], second[4:6])),
            torch.stack((torch.stack((first[0], second[0])),
                         torch.stack((first[6], second[6])))))
