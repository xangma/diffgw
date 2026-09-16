"""Public-facing clean-room wrapper for the externally supplied SEOBNRv4 ROM in JAX.

Port of torchwave.approximants.rom.official_wrapper.
"""
from __future__ import annotations

from pathlib import Path
from itertools import product
from typing import Optional, Union
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI
from jaxwave.conversions import as_array
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from .official import OFFICIAL_ROM_SHA256, resolve_official_rom_path, sha256_file
from .bspline import load_official_rom
from .glue import OfficialROMGlue

_UNSET = object()


def independent_ringdown_mf(mass1, mass2, chi1, chi2):
    """Return total-mass QNM frequency in cycles, from independent fits."""
    mf = eob_final_mass_fraction(mass1, mass2, chi1, chi2)
    af = hofmann_final_spin(mass1, mass2, chi1, chi2, fit_order=(3, 4))
    return evaluate_kerr_qnm_220(af)[0] / (2.0 * PI * mf)


def table_convention_ringdown_mf(mass1, mass2, chi1, chi2):
    """Return ROM ringdown frequency using the validated rounded-knot convention."""
    mf = eob_final_mass_fraction(mass1, mass2, chi1, chi2)
    af = hofmann_final_spin(mass1, mass2, chi1, chi2, fit_order=(3, 4))
    return signed_qnm_220_table_convention(af)[0] / (2.0 * PI * mf)


class OfficialSEOBNRv4ROM:
    """Evaluate the public external ROM without a LAL runtime dependency in JAX."""

    def __init__(
        self,
        model_path=None,
        *,
        node_policy: str = "31_first_above",
        ringdown_mf=None,
        flow: Optional[float] = None,
        fhigh: Optional[float] = None,
    ):
        path = resolve_official_rom_path(model_path, verify=False)
        digest = sha256_file(path)
        if digest != OFFICIAL_ROM_SHA256:
            raise ValueError(
                f"Unexpected SHA-256 for {Path(path).name}: {digest}; "
                f"expected {OFFICIAL_ROM_SHA256}"
            )
        self.model_path = Path(path)
        self._flow = flow
        self._fhigh = fhigh
        low = load_official_rom(path, "sub1")
        high = load_official_rom(path, "sub2")
        high_spin = load_official_rom(path, "sub3")
        provider = table_convention_ringdown_mf if ringdown_mf is None else ringdown_mf
        self._normal = OfficialROMGlue(
            low,
            high,
            ringdown_mf=provider,
            node_policy=node_policy,
            f_lower_hz=flow,
            f_final_hz=fhigh,
        )
        self._high_spin = OfficialROMGlue(
            low,
            high_spin,
            ringdown_mf=provider,
            node_policy=node_policy,
            f_lower_hz=flow,
            f_final_hz=fhigh,
        )
        self._ringdown = provider

    def __call__(
        self,
        frequency_hz=None,
        mass1=None,
        mass2=None,
        chi1=0.0,
        chi2=0.0,
        *,
        f=None,
        distance=_UNSET,
        distance_mpc=None,
        inclination=0.0,
        phic=0.0,
        tc=0.0,
        f_ref=None,
        flow=None,
        fhigh=None,
    ):
        if frequency_hz is None:
            frequency_hz = f
        elif f is not None:
            raise TypeError("provide either frequency_hz or f, not both")
        if frequency_hz is None:
            raise TypeError("frequency_hz (or f) is required")
        if mass1 is None or mass2 is None:
            raise TypeError("mass1 and mass2 are required")
        distance_given = distance is not _UNSET
        if not distance_given:
            distance = 100.0
        if distance_mpc is not None:
            if distance_given:
                raise TypeError("provide either distance or distance_mpc, not both")
            distance = distance_mpc

        input_frequency = as_array(frequency_hz, dtype=jnp.float64)
        out_dtype = jnp.complex64 if getattr(frequency_hz, "dtype", None) == jnp.float32 else jnp.complex128
        f_arr = input_frequency

        dtype = jnp.float64
        m1, m2, s1 = (as_array(x, dtype=dtype) for x in (mass1, mass2, chi1))
        vals = [as_array(x, dtype=dtype) for x in
                (m1, m2, s1, chi2, distance, inclination, phic, tc,
                 0.0 if f_ref is None else f_ref)]

        for name, value in (("distance", distance), ("inclination", inclination),
                            ("phic", phic), ("tc", tc), ("f_ref", f_ref)):
            if value is not None:
                check = as_array(value, dtype=dtype)
                if not isinstance(check, jax.core.Tracer):
                    if not bool(jnp.all(jnp.isfinite(check))):
                        raise ValueError(f"{name} must be finite")
        if f_ref is not None:
            check_ref = as_array(f_ref, dtype=dtype)
            if not isinstance(check_ref, jax.core.Tracer):
                if bool(jnp.any(check_ref < 0)):
                    raise ValueError("f_ref must be non-negative or None")

        effective_flow = self._flow if flow is None else flow
        effective_fhigh = self._fhigh if fhigh is None else fhigh
        bound_values = [as_array(x, dtype=dtype) for x in
                        (effective_flow, effective_fhigh) if x is not None]

        # Broadcast parameters
        all_to_broadcast = vals + bound_values
        broadcasted = jnp.broadcast_arrays(*all_to_broadcast)
        params_shape = broadcasted[0].shape
        vals_b = broadcasted[:len(vals)]

        m1, m2, s1, s2, dist, inc, phi, tcv, fref = vals_b
        swap = m2 > m1
        old_m1, old_s1 = m1, s1
        m1 = jnp.where(swap, m2, m1)
        m2 = jnp.where(swap, old_m1, m2)
        s1 = jnp.where(swap, s2, s1)
        s2 = jnp.where(swap, old_s1, s2)
        eta = (m1 * m2) / (m1 + m2)**2
        highspin = (s1 >= 0.995) & (eta <= 0.025)

        vals_p, vals_c = [], []
        indices = list(product(*(range(n) for n in params_shape))) if params_shape else [()]
        for idx in indices:
            def _bound(value):
                if value is None:
                    return None
                bound = as_array(value, dtype=dtype)
                return jnp.broadcast_to(bound, params_shape)[idx] if params_shape else bound

            kw = dict(
                distance_mpc=float(dist[idx]) if params_shape else float(dist),
                inclination=float(inc[idx]) if params_shape else float(inc),
                phic=float(phi[idx]) if params_shape else float(phi),
                tc=float(tcv[idx]) if params_shape else float(tcv),
                fref_hz=float(fref[idx]) if params_shape else float(fref),
                f_lower_hz=float(_bound(effective_flow)) if _bound(effective_flow) is not None else None,
                f_final_hz=float(_bound(effective_fhigh)) if _bound(effective_fhigh) is not None else None,
            )
            m1_val = m1[idx] if params_shape else m1
            m2_val = m2[idx] if params_shape else m2
            s1_val = s1[idx] if params_shape else s1
            s2_val = s2[idx] if params_shape else s2
            args = (m1_val, m2_val, s1_val, s2_val)
            is_hs = bool(highspin[idx]) if params_shape else bool(highspin)
            engine = self._high_spin if is_hs else self._normal
            p, c = engine(f_arr, *args, **kw)
            vals_p.append(p.astype(out_dtype))
            vals_c.append(c.astype(out_dtype))

        if not params_shape:
            return vals_p[0], vals_c[0]
        return (
            jnp.stack(vals_p).reshape(params_shape + f_arr.shape),
            jnp.stack(vals_c).reshape(params_shape + f_arr.shape),
        )


__all__ = [
    "OfficialSEOBNRv4ROM",
    "independent_ringdown_mf",
    "table_convention_ringdown_mf",
]
