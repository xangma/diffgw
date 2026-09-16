"""Band-limited, alignment-aware phase and complex-waveform metrics.

The reference is ``target`` and the waveform being corrected is ``model``.
Corrections multiply the model by ``exp(1j * (constant + 2*pi*f*dt))``;
therefore a positive fitted ``dt`` advances model phase at positive frequency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class PhaseMetrics:
    bins: int
    constant_phase: float
    affine_phase: float
    time_shift: float
    raw_relative_l2: float
    constant_relative_l2: float
    affine_relative_l2: float
    raw_overlap: float
    constant_overlap: float
    affine_overlap: float
    raw_mismatch: float
    constant_mismatch: float
    affine_mismatch: float
    raw_phase_rms: float
    constant_phase_rms: float
    affine_phase_rms: float

    def as_dict(self):
        return asdict(self)


def _support(f, target, model, weights, amplitude_floor, phase_mask):
    f = np.asarray(f, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=complex).reshape(-1)
    model = np.asarray(model, dtype=complex).reshape(-1)
    if f.size != target.size or f.size != model.size or f.size < 3:
        raise ValueError("frequency and waveform arrays need >=3 equal-length bins")
    if not np.all(np.isfinite(f)) or not np.all(np.diff(f) > 0):
        raise ValueError("frequency grid must be finite and strictly increasing")
    if weights is None:
        w = np.ones(f.size)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != f.size:
            raise ValueError("weights have the wrong length")
    if not (np.all(np.isfinite(target.real)) and np.all(np.isfinite(target.imag))
            and np.all(np.isfinite(model.real)) and np.all(np.isfinite(model.imag))
            and np.all(np.isfinite(w)) and np.all(w > 0)):
        raise ValueError("waveforms and weights must be finite; weights must be positive")
    finite = np.ones(f.size, dtype=bool)
    if phase_mask is not None:
        mask = np.asarray(phase_mask, dtype=bool).reshape(-1)
        if mask.size != f.size:
            raise ValueError("phase_mask has the wrong length")
        finite &= mask
    if amplitude_floor is not None:
        if amplitude_floor < 0 or not np.isfinite(amplitude_floor):
            raise ValueError("amplitude_floor must be finite and nonnegative")
        scale = max(np.max(np.abs(target[finite])) if np.any(finite) else 0,
                    np.max(np.abs(model[finite])) if np.any(finite) else 0)
        finite &= np.maximum(np.abs(target), np.abs(model)) >= amplitude_floor * scale
    if np.count_nonzero(finite) < 3:
        raise ValueError("insufficient finite, positively weighted bins")
    base_norm = np.sum(w * np.abs(target) ** 2)
    if not np.isfinite(base_norm) or base_norm <= 0:
        raise ValueError("target has zero weighted norm")
    retained = float(np.sum(w[finite] * np.abs(target[finite]) ** 2) / base_norm)
    f, target, model, w = f[finite], target[finite], model[finite], w[finite]
    if not np.isfinite(np.sum(w * np.abs(target) ** 2)) or np.sum(w * np.abs(target) ** 2) <= 0:
        raise ValueError("target has zero weighted norm")
    if not np.isfinite(np.sum(w * np.abs(model) ** 2)) or np.sum(w * np.abs(model) ** 2) <= 0:
        raise ValueError("model has zero weighted norm")
    return f, target, model, w, retained


def _l2(a, b, w):
    return float(np.sqrt(np.sum(w * np.abs(a - b) ** 2) / np.sum(w * np.abs(b) ** 2)))


def _overlap(a, b, w, *, absolute=True):
    cross = np.sum(w * np.conj(b) * a)
    value = np.abs(cross) if absolute else np.real(cross)
    return float(value / np.sqrt(np.sum(w * np.abs(a) ** 2) * np.sum(w * np.abs(b) ** 2)))


def _affine_phase(f, target, model, w, dt_bounds):
    # Unwrap target/model phase difference, then fit target-model phase.
    y = np.unwrap(np.angle(target * np.conj(model)))
    # Phase least squares weights include the reference amplitude squared.
    w = w * np.abs(target) ** 2
    x = np.column_stack((np.ones(f.size), 2 * np.pi * f))
    normal = x.T @ (w[:, None] * x)
    rhs = x.T @ (w * y)
    try:
        beta = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("frequency support is insufficient for affine fit") from exc
    if dt_bounds is not None:
        lo, hi = map(float, dt_bounds)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ValueError("invalid dt_bounds")
        if beta[1] < lo or beta[1] > hi:
            # For fixed dt, the weighted-optimal constant is analytic. Choose
            # the better feasible endpoint; this is the bounded LS solution.
            candidates = []
            for dt in (lo, hi):
                c = float(np.sum(w * (y - 2 * np.pi * f * dt)) / np.sum(w))
                candidates.append((np.sum(w * (y - c - 2 * np.pi * f * dt) ** 2), c, dt))
            _, c, dt = min(candidates)
            beta = np.array([c, dt])
    return float(beta[0]), float(beta[1]), y


def phase_metrics(f, candidate, reference, *, weights=None, phase_mask=None,
                  time_bounds=(-200.0, 200.0), amplitude_floor=None, dt_bounds=None):
    """Return raw, constant-phase, and affine phase-aligned metrics.

    ``candidate`` is corrected toward ``reference``. ``raw`` compares inputs directly. Constant alignment maximizes normalized
    overlap. Affine alignment is weighted least squares in unwrapped phase;
    ``dt_bounds=(lo, hi)`` bounds the fitted time shift in the same units as
    ``f`` inverse units. Mismatch is ``1 - normalized overlap``.
    ``phase_mask`` limits only phase fitting and phase RMS; waveform metrics use
    the full validated band. ``time_bounds`` bounds the affine time shift (``dt_bounds`` is accepted as a compatibility
    alias). The result is a JSON-serializable dictionary.
    """
    if dt_bounds is not None:
        time_bounds = dt_bounds
    f, target, model, w, _ = _support(f, reference, candidate, weights, amplitude_floor, None)
    if phase_mask is None:
        phase_mask = np.ones(f.size, dtype=bool)
    else:
        phase_mask = np.asarray(phase_mask, dtype=bool).reshape(-1)
        if phase_mask.size != f.size or np.count_nonzero(phase_mask) < 3:
            raise ValueError("phase_mask has wrong length or insufficient bins")
    pf, pt, pm, pw, retained = f[phase_mask], target[phase_mask], model[phase_mask], w[phase_mask], float(np.sum(w[phase_mask] * np.abs(target[phase_mask])**2) / np.sum(w * np.abs(target)**2))
    c0 = float(np.angle(np.sum(w * np.conj(model) * target)))
    constant = model * np.exp(1j * c0)
    c1, dt, _ = _affine_phase(pf, pt, pm, pw, time_bounds)
    affine = model * np.exp(1j * (c1 + 2 * np.pi * f * dt))
    vals = [model, constant, affine]
    l2 = [_l2(a, target, w) for a in vals]
    ov = [_overlap(model, target, w, absolute=False),
          _overlap(constant, target, w), _overlap(affine, target, w)]
    phase_models = [pm, pm * np.exp(1j*c0), pm * np.exp(1j*(c1 + 2*np.pi*pf*dt))]
    phase = [np.unwrap(np.angle(pt * np.conj(a))) for a in phase_models]
    phasew = pw * np.abs(pt) ** 2
    rms = [float(np.sqrt(np.sum(phasew * p ** 2) / np.sum(phasew))) for p in phase]
    result = PhaseMetrics(f.size, c0, c1, dt, *l2, *ov, *(1 - x for x in ov), *rms).as_dict()
    result["phase_mask_retained_reference_norm_fraction"] = retained
    return result


def optimized_overlap(f, candidate, reference, *, weights=None,
                      time_bounds=(-200.0, 200.0)):
    """Maximize weighted normalized overlap over time and constant phase.

    A coarse grid has spacing no larger than ``1/(8*bandwidth)``; every grid
    local maximum and both bounds are refined, so a boundary result is flagged.
    """
    f, ref, cand, w, _ = _support(f, reference, candidate, weights, None, None)
    lo, hi = map(float, time_bounds)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError("invalid time_bounds")
    bw = f[-1] - f[0]
    step = 1.0 / (8.0 * bw)
    n = max(2, int(np.ceil((hi - lo) / step)) + 1)
    grid = np.linspace(lo, hi, n)
    nr = np.sum(w * np.abs(ref) ** 2)
    nc = np.sum(w * np.abs(cand) ** 2)
    def score(dt):
        cross = np.sum(w * np.conj(cand) * ref * np.exp(-2j * np.pi * f * dt))
        return float(np.abs(cross) / np.sqrt(nr * nc))
    vals = np.array([score(x) for x in grid])
    candidates = [(vals[0], lo), (vals[-1], hi)]
    intervals = [(0, 1), (n - 2, n - 1)]
    intervals += [(i - 1, i + 1) for i in range(1, n - 1)
                  if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]]
    for left, right in intervals:
        r = minimize_scalar(lambda x: -score(x), bounds=(grid[left], grid[right]), method="bounded",
                            options={"xatol": 1e-10})
        if not r.success or not np.isfinite(r.fun):
            raise ValueError("time-overlap optimizer failed")
        candidates.append((min(1.0, max(0.0, -float(r.fun))), float(r.x)))
    best, dt = max(candidates, key=lambda x: x[0])
    cross = np.sum(w * np.conj(cand) * ref * np.exp(-2j * np.pi * f * dt))
    phase = float(np.angle(cross))
    return {"time_shift": dt, "constant_phase": phase, "overlap": best,
            "mismatch": 1.0 - best, "boundary": bool(np.isclose(dt, lo) or np.isclose(dt, hi)),
            "grid_step": float(grid[1] - grid[0]), "bins": int(f.size)}
