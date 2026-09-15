"""Resolved trajectory events, independent of an EOB calibration prescription.

Times and frequencies use t/M and M*Omega. The local interpolation below is
a numerical choice, not an assertion about a target library's interpolation.
"""

from typing import NamedTuple

import torch


class OrbitalFrequencyPeak(NamedTuple):
    time: torch.Tensor
    frequency: torch.Tensor
    curvature: torch.Tensor
    sample_index: torch.Tensor


def orbital_frequency_peak(time, frequency, *, valid=None):
    """Refine an interior maximum with a seven-sample degree-six interpolation.

    Samples occupy the last axis. Optional ``valid`` masks must be contiguous
    prefixes, permitting trajectories with independently terminated batches.
    A seven-point stencil is used when available; near a stopped trajectory
    it is shifted backward while retaining a local bracket around the sample.
    derivative root must be bracketed by its immediate neighbours and have
    negative curvature. Missing/unresolved peaks raise rather than silently
    selecting a trajectory endpoint. Verify convergence by reducing ODE dt.

    Autograd follows the chosen stencil and root, but the discrete selection
    of a different maximum is not differentiable.
    """
    if not isinstance(time, torch.Tensor) or not isinstance(frequency, torch.Tensor):
        raise TypeError("time and frequency must be torch tensors")
    if (time.dtype not in (torch.float32, torch.float64) or frequency.dtype != time.dtype
            or frequency.device != time.device):
        raise TypeError("time and frequency must share float32/float64 dtype and device")
    if time.ndim < 1 or frequency.ndim < 1 or time.shape[-1] != frequency.shape[-1]:
        raise ValueError("time and frequency must share a last-axis sample count")
    time, frequency = torch.broadcast_tensors(time, frequency)
    if time.shape[-1] < 7:
        raise ValueError("a resolved peak requires at least seven samples")
    if not bool(torch.isfinite(time).all() & torch.isfinite(frequency).all()):
        raise ValueError("time and frequency must be finite")
    if bool((time.diff(dim=-1) <= 0).any()):
        raise ValueError("sample times must be strictly increasing")
    if valid is None:
        valid = torch.ones_like(frequency, dtype=torch.bool)
    elif not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.device != time.device:
        raise TypeError("valid must be a boolean tensor on time's device")
    valid = valid.expand(frequency.shape)
    if not bool(valid[..., 0].all()) or bool((valid[..., 1:] & ~valid[..., :-1]).any()):
        raise ValueError("valid samples must form contiguous nonempty prefixes")
    # The event feeds the SEOBNRv4 NQC and merger prescriptions, whose
    # frequency convention is positive -d arg(h22)/d tau. Invalid suffixes
    # are padding and do not participate in the event definition.
    if bool(((frequency <= 0) & valid).any()):
        raise ValueError("valid orbital frequency samples must be positive")
    count = valid.sum(-1)
    if bool((count < 7).any()):
        raise ValueError("a resolved peak requires at least seven valid samples")
    index = frequency.masked_fill(~valid, -torch.inf).argmax(-1)
    if bool(((index < 1) | (index + 1 >= count)).any()):
        raise ValueError("orbital-frequency peak is unresolved: need samples on both sides")
    # Prefer a centered seven-point stencil; trajectories stopped immediately
    # after the event retain only two post-peak samples, so shift the same
    # degree-six stencil backward while keeping the derivative bracket local.
    start = (index - 3).clamp_min(0)
    start = torch.minimum(start, count - 7)
    indices = start.unsqueeze(-1) + torch.arange(7, device=time.device)
    t = torch.gather(time, -1, indices)
    y = torch.gather(frequency, -1, indices)
    center = index-start
    origin = torch.gather(t, -1, center.unsqueeze(-1)).squeeze(-1)
    scale = (t[..., -1] - t[..., 0])/2
    x = (t-origin.unsqueeze(-1))/scale.unsqueeze(-1)
    matrix = torch.stack(tuple(x.pow(k) for k in range(7)), dim=-1)
    # Subtract the central value so a large constant frequency cannot spoil
    # the conditioning of the much smaller local derivatives.
    coef = torch.linalg.solve(matrix, (y-torch.gather(y, -1, center.unsqueeze(-1))).unsqueeze(-1)).squeeze(-1)

    def derivatives(z):
        first = sum(k*coef[..., k]*z.pow(k-1) for k in range(1, 7))
        second = sum(k*(k-1)*coef[..., k]*z.pow(k-2) for k in range(2, 7))
        return first, second

    left, right = x.gather(-1, (center-1).unsqueeze(-1)).squeeze(-1), x.gather(-1, (center+1).unsqueeze(-1)).squeeze(-1)
    if bool(((derivatives(left)[0] <= 0) | (derivatives(right)[0] >= 0)).any()):
        raise ValueError("local orbital-frequency maximum has no derivative bracket")
    root = torch.zeros_like(origin)
    for _ in range(24):
        first, second = derivatives(root)
        safe_second = torch.where(second != 0, second, torch.ones_like(second))
        newton = root-first/safe_second
        candidate = torch.where((second < 0) & (newton >= left) & (newton <= right),
                                newton, (left+right)/2)
        slope = derivatives(candidate)[0]
        left = torch.where(slope > 0, candidate, left)
        right = torch.where(slope < 0, candidate, right)
        root = candidate
    slope, second = derivatives(root)
    tolerance = 128*torch.finfo(time.dtype).eps*coef.abs().sum(-1)
    if bool(((slope.abs() > tolerance) | (second >= 0)).any()):
        raise ValueError("orbital-frequency peak refinement did not converge to a maximum")
    value = torch.gather(y, -1, center.unsqueeze(-1)).squeeze(-1) + sum(coef[..., k]*root.pow(k) for k in range(7))
    return OrbitalFrequencyPeak(origin+scale*root, value, second/scale.square(), index)
