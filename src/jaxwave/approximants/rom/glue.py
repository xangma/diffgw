"""Clean-room SEOBNRv4 ROM frequency glue in JAX.

Port of torchwave.approximants.rom.glue for jaxwave.
"""

from __future__ import annotations

from itertools import product
from typing import Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from jaxwave.approximants.rom.bspline import evaluate_stitched_frequency, OfficialROMSurrogate


class OfficialROMGlue:
    """Evaluate externally loaded low/high ROM patches with physical inputs in JAX."""

    def __init__(
        self,
        low: OfficialROMSurrogate,
        high: OfficialROMSurrogate,
        *,
        ringdown_mf,
        node_policy: str = "31_first_above",
        f_lower_hz: Optional[float] = None,
        f_final_hz: Optional[float] = None,
    ):
        self.low, self.high = low, high
        self.ringdown_mf = ringdown_mf
        nodes = high.mf_grid_phase[(high.mf_grid_phase >= 0.0074) & (high.mf_grid_phase <= 0.014)]
        i = int(np.searchsorted(np.asarray(nodes), 0.01))
        if node_policy == "31_first_above":
            self.fit_nodes = nodes[i - 15:i + 16]
        elif node_policy == "30_split":
            self.fit_nodes = jnp.concatenate((nodes[i - 15:i], nodes[i:i + 15]))
        elif node_policy == "all66":
            self.fit_nodes = nodes
        else:
            raise ValueError("unknown node policy")
        self.f_lower_hz = f_lower_hz
        self.f_final_hz = f_final_hz

    def __call__(
        self,
        frequency_hz,
        mass1,
        mass2,
        chi1,
        chi2,
        *,
        distance_mpc,
        inclination=0.0,
        phic=0.0,
        fref_hz=20.0,
        tc=0.0,
        f_lower_hz=None,
        f_final_hz=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        f = jnp.asarray(frequency_hz, dtype=jnp.float64)
        if f.ndim == 0:
            f = f.reshape(1)
        elif f.ndim != 1:
            raise ValueError("frequency_hz must be one-dimensional")

        def _arr(v):
            return jnp.asarray(v, dtype=jnp.float64)

        m1, m2 = _arr(mass1), _arr(mass2)
        s1, s2 = _arr(chi1), _arr(chi2)
        dist, inc = _arr(distance_mpc), _arr(inclination)
        phi_c, t_c, fref = _arr(phic), _arr(tc), _arr(fref_hz)

        params_shape = np.broadcast_shapes(
            m1.shape, m2.shape, s1.shape, s2.shape,
            dist.shape, inc.shape, phi_c.shape, t_c.shape, fref.shape,
        )

        m1 = jnp.broadcast_to(m1, params_shape)
        m2 = jnp.broadcast_to(m2, params_shape)
        s1 = jnp.broadcast_to(s1, params_shape)
        s2 = jnp.broadcast_to(s2, params_shape)
        dist = jnp.broadcast_to(dist, params_shape)
        inc = jnp.broadcast_to(inc, params_shape)
        phi_c = jnp.broadcast_to(phi_c, params_shape)
        t_c = jnp.broadcast_to(t_c, params_shape)
        fref = jnp.broadcast_to(fref, params_shape)

        swap = m2 > m1
        old_m1, old_s1 = m1, s1
        m1 = jnp.where(swap, m2, m1)
        m2 = jnp.where(swap, old_m1, m2)
        s1 = jnp.where(swap, s2, s1)
        s2 = jnp.where(swap, old_s1, s2)

        mass = m1 + m2
        eta = (m1 * m2) / (mass**2)

        lower = self.f_lower_hz if f_lower_hz is None else f_lower_hz
        final = self.f_final_hz if f_final_hz is None else f_final_hz

        amp_lo = self.low.mf_grid_amp[0]
        phase_lo = self.low.mf_grid_phase[0]
        amp_hi = self.high.mf_grid_amp[-1]
        phase_hi = self.high.mf_grid_phase[-1]
        model_lower = jnp.maximum(amp_lo, phase_lo) / (mass * MSUN_SEC)
        model_final = jnp.minimum(amp_hi, phase_hi) / (mass * MSUN_SEC)

        lower_t = model_lower if lower is None else jnp.broadcast_to(_arr(lower), params_shape)
        final_t = model_final if final is None else jnp.broadcast_to(_arr(final), params_shape)

        fref = jnp.where(fref == 0.0, lower_t, fref)

        ringdown = self._ringdown_tensor(m1, m2, s1, s2, params_shape)

        hp, hc = [], []
        indices = list(product(*(range(n) for n in params_shape))) if params_shape else [()]
        for idx in indices:
            mi = mass[idx]
            model_mask = (
                (f > 0.0)
                & (f >= lower_t[idx])
                & (f < final_t[idx])
                & (f >= model_lower[idx])
                & (f <= model_final[idx])
            )
            if not np.any(np.asarray(model_mask)):
                hp.append(jnp.zeros(f.shape, dtype=jnp.complex128))
                hc.append(jnp.zeros(f.shape, dtype=jnp.complex128))
                continue

            mf_points = f[model_mask] * mi * MSUN_SEC
            rd = ringdown[idx]
            amp, phase = evaluate_stitched_frequency(
                self.low,
                self.high,
                mf_points,
                (m1[idx] * m2[idx]) / (mi**2),
                s1[idx],
                s2[idx],
                ringdown_frequency=rd,
                fit_nodes=self.fit_nodes,
                reference_mf=fref[idx] * mi * MSUN_SEC,
            )
            z = amp * jnp.exp(1j * phase)
            shift = jnp.exp(1j * (2.0 * jnp.pi * f[model_mask] * t_c[idx] + 2.0 * phi_c[idx]))
            z = z * (mi * MSUN_SEC) * (C_SI * mi * MSUN_SEC) / (dist[idx] * MPC_SI) * shift
            ci = jnp.cos(inc[idx])
            hpv = 0.5 * (1.0 + ci**2) * 0.5 * z
            hcv = -ci * 0.5j * z

            hpfull = jnp.zeros(f.shape, dtype=jnp.complex128)
            hcfull = jnp.zeros(f.shape, dtype=jnp.complex128)
            hpfull = hpfull.at[model_mask].set(hpv)
            hcfull = hcfull.at[model_mask].set(hcv)
            hp.append(hpfull)
            hc.append(hcfull)

        out_shape = params_shape + f.shape
        return jnp.stack(hp).reshape(out_shape), jnp.stack(hc).reshape(out_shape)

    def _ringdown_tensor(self, m1, m2, s1, s2, params_shape):
        if callable(self.ringdown_mf):
            return self.ringdown_mf(m1, m2, s1, s2)
        return jnp.broadcast_to(jnp.asarray(self.ringdown_mf, dtype=jnp.float64), params_shape)


__all__ = ["OfficialROMGlue"]
