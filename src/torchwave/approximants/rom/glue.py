"""Clean-room SEOBNRv4 ROM frequency glue.

The 31-node choice is the discrete interpretation with fifteen tabulated
phase knots on either side of the first knot above ``Mf=.01``.  It is a
published interval construction choice (Pürrer, arXiv:1512.02248, §IV.1),
not a continuously fitted parameter.
"""
from __future__ import annotations
import torch
from itertools import product
from torchwave.constants import MSUN_SEC, MPC_SI, C_SI
from .bspline import evaluate_stitched_frequency

class OfficialROMGlue:
    """Evaluate externally loaded low/high ROM patches with physical inputs.

    Frequencies are positive Hz; masses are solar masses and distance is Mpc.
    ``ringdown_mf`` must be supplied by the caller from a public QNM contract.
    The returned pair is plus/cross complex frequency series with the standard
    distance, inclination, phase, reference-frequency, and coalescence-time
    factors. Inputs may be scalar or broadcastable torch tensors.
    """
    def __init__(self, low, high, *, ringdown_mf, node_policy="31_first_above",
                 f_lower_hz=None, f_final_hz=None):
        self.low, self.high = low, high
        self.ringdown_mf = ringdown_mf
        nodes = high.mf_grid_phase[(high.mf_grid_phase >= .0074) & (high.mf_grid_phase <= .014)]
        i = int(torch.searchsorted(nodes, torch.as_tensor(.01, dtype=nodes.dtype, device=nodes.device)).item())
        if node_policy == "31_first_above": self.fit_nodes = nodes[i-15:i+16]
        elif node_policy == "30_split": self.fit_nodes = torch.cat((nodes[i-15:i], nodes[i:i+15]))
        elif node_policy == "all66": self.fit_nodes = nodes
        else: raise ValueError("unknown node policy")
        self.f_lower_hz = f_lower_hz
        self.f_final_hz = f_final_hz

    def __call__(self, frequency_hz, mass1, mass2, chi1, chi2, *, distance_mpc,
                 inclination=0., phic=0., fref_hz=20., tc=0.,
                 f_lower_hz=None, f_final_hz=None):
        f = torch.as_tensor(frequency_hz, dtype=self.low.amp_coeff.dtype, device=self.low.amp_coeff.device)
        if f.ndim == 0:
            f = f.reshape(1)
        elif f.ndim != 1:
            raise ValueError("frequency_hz must be one-dimensional")
        if torch.any(~torch.isfinite(f)) or torch.any(f < 0):
            raise ValueError("frequency_hz must be finite and non-negative")

        def _tensor(value):
            return torch.as_tensor(value, dtype=f.dtype, device=f.device)

        m1, m2 = _tensor(mass1), _tensor(mass2)
        s1, s2 = _tensor(chi1), _tensor(chi2)
        distance, inc = _tensor(distance_mpc), _tensor(inclination)
        phic, tc, fref = _tensor(phic), _tensor(tc), _tensor(fref_hz)
        params = torch.broadcast_shapes(m1.shape, m2.shape, s1.shape, s2.shape,
                                        distance.shape, inc.shape, phic.shape,
                                        tc.shape, fref.shape)
        m1, m2, s1, s2 = (v.expand(params) for v in (m1, m2, s1, s2))
        distance, inc, phic, tc, fref = (v.expand(params) for v in (distance, inc, phic, tc, fref))
        # The ROM parameterization is ordered by the heavier component.
        swap = m2 > m1
        old_m1, old_s1 = m1, s1
        m1 = torch.where(swap, m2, m1)
        m2 = torch.where(swap, old_m1, m2)
        s1 = torch.where(swap, s2, s1)
        s2 = torch.where(swap, old_s1, s2)
        if torch.any(~torch.isfinite(distance)) or torch.any(distance <= 0):
            raise ValueError("distance_mpc must be finite and positive")
        if torch.any(~torch.isfinite(m1)) or torch.any(~torch.isfinite(m2)) or torch.any(m1 <= 0) or torch.any(m2 <= 0):
            raise ValueError("component masses must be finite and positive")
        if torch.any(~torch.isfinite(s1)) or torch.any(~torch.isfinite(s2)) or torch.any(s1.abs() > 1) or torch.any(s2.abs() > 1):
            raise ValueError("spins must be finite and within [-1, 1]")
        mass = m1 + m2
        eta = (m1 * m2) / mass.square()
        eta_lo = torch.as_tensor(self.low.eta_grid.min(), dtype=f.dtype, device=f.device)
        eta_hi = torch.as_tensor(self.low.eta_grid.max(), dtype=f.dtype, device=f.device)
        if torch.any((eta < eta_lo) | (eta > eta_hi)):
            raise ValueError("mass ratio lies outside ROM eta domain")
        lower = self.f_lower_hz if f_lower_hz is None else f_lower_hz
        final = self.f_final_hz if f_final_hz is None else f_final_hz
        # A value is usable only where both amplitude and phase tables exist.
        amp_lo = torch.as_tensor(self.low.mf_grid_amp[0], dtype=f.dtype, device=f.device)
        phase_lo = torch.as_tensor(self.low.mf_grid_phase[0], dtype=f.dtype, device=f.device)
        amp_hi = torch.as_tensor(self.high.mf_grid_amp[-1], dtype=f.dtype, device=f.device)
        phase_hi = torch.as_tensor(self.high.mf_grid_phase[-1], dtype=f.dtype, device=f.device)
        model_lower = torch.maximum(amp_lo, phase_lo) / (mass * MSUN_SEC)
        model_final = torch.minimum(amp_hi, phase_hi) / (mass * MSUN_SEC)
        if lower is not None and (not torch.all(torch.isfinite(_tensor(lower)))):
            raise ValueError("f_lower_hz must be finite")
        if final is not None and (not torch.all(torch.isfinite(_tensor(final)))):
            raise ValueError("f_final_hz must be finite")
        lower_t = model_lower if lower is None else torch.broadcast_to(_tensor(lower), params)
        final_t = model_final if final is None else torch.broadcast_to(_tensor(final), params)
        if torch.any(lower_t <= 0) or torch.any(final_t <= lower_t):
            raise ValueError("f_lower_hz and f_final_hz must be positive and increasing")
        # Zero is the public sentinel for the effective lower support.
        fref = torch.where(fref == 0, lower_t, fref)
        if torch.any(~torch.isfinite(fref)) or torch.any(fref <= 0):
            raise ValueError("fref_hz must be finite and strictly positive, or zero")
        ringdown = self._ringdown_tensor(m1, m2, s1, s2, params, f)
        if torch.any(~torch.isfinite(ringdown)) or torch.any(ringdown <= 0):
            raise ValueError("ringdown_mf must be finite and positive")

        # The public ROM tables are scalar in intrinsic parameters.  Evaluate
        # each broadcast point independently and stack; this keeps all tensor
        # operations differentiable while providing the standard (batch, freq)
        # interface.
        hp, hc = [], []
        indices = list(product(*(range(n) for n in params))) if params else [()]
        for idx in indices:
            mi = mass[idx]
            # The public frequency interval is half-open at its upper edge.
            # The endpoint is retained in the caller's array as a zero sentinel.
            model_mask = ((f > 0) & (f >= lower_t[idx]) & (f < final_t[idx])
                          & (f >= model_lower[idx]) & (f <= model_final[idx]))
            if not torch.any(model_mask):
                hp.append(torch.zeros(f.shape, dtype=torch.complex128, device=f.device))
                hc.append(torch.zeros(f.shape, dtype=torch.complex128, device=f.device))
                continue
            mf_points = f[model_mask] * mi * MSUN_SEC
            rd = ringdown[idx]
            amp, phase = evaluate_stitched_frequency(
                self.low, self.high, mf_points, (m1[idx] * m2[idx]) / mi.square(), s1[idx], s2[idx],
                ringdown_frequency=rd, fit_nodes=self.fit_nodes,
                reference_mf=fref[idx] * mi * MSUN_SEC)
            z = torch.polar(amp, phase)
            # Positive-frequency convention used by SimInspiralChooseFDWaveform.
            shift = torch.exp(1j * (2 * torch.pi * f[model_mask] * tc[idx] + 2 * phic[idx]))
            z = z * (mi * MSUN_SEC) * (C_SI * mi * MSUN_SEC) / (distance[idx] * MPC_SI) * shift
            ci = torch.cos(inc[idx])
            hpv = .5 * (1 + ci.square()) * .5 * z
            hcv = -ci * .5j * z
            hpfull = torch.zeros(f.shape, dtype=hpv.dtype, device=f.device)
            hcfull = torch.zeros(f.shape, dtype=hcv.dtype, device=f.device)
            hp.append(hpfull.index_copy(0, model_mask.nonzero().flatten(), hpv))
            hc.append(hcfull.index_copy(0, model_mask.nonzero().flatten(), hcv))
        shape = params + f.shape
        return torch.stack(hp).reshape(shape), torch.stack(hc).reshape(shape)

    def _ringdown_tensor(self, m1, m2, s1, s2, params, frequency):
        value = self.ringdown_mf(m1, m2, s1, s2) if callable(self.ringdown_mf) else self.ringdown_mf
        return torch.as_tensor(value, dtype=frequency.dtype, device=frequency.device).expand(params)
