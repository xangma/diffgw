"""IMRPhenomXP Precessing Dominant-Harmonic Frequency-Domain Approximant in JAX.

Mathematical references for the inherited model and audited corrections:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412)
"""

from typing import Optional, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    component_masses_to_delta,
)
from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS, XASParameterFits
from jaxwave.approximants.precession_msa import (
    compute_effective_chip,
    compute_msa_final_spin,
    compute_l_3pn,
    evaluate_msa_precession,
    resolve_precession_version,
)


def wigner_small_d_matrix(l: int, beta: jnp.ndarray) -> jnp.ndarray:
    """Compute exact Wigner small-d matrix d^l_{m, m'}(beta) for l in {2, 3, 4}.

    Indices range from -l to l (matrix shape (..., 2l+1, 2l+1)).
    Formula:
        d^l_{m, m'}(beta) = sum_k (-1)^k * [sqrt((l+m)!(l-m)!(l+m')!(l-m')!) /
                            ((l+m-k)! k! (l-m'-k)! (k-m+m')!)] * (cos beta/2)^(2l+m-m'-2k) * (sin beta/2)^(2k-m+m')
    """
    if l == 2:
        c = jnp.cos(0.5 * beta)
        s = jnp.sin(0.5 * beta)
        c2 = c * c
        s2 = s * s
        c4 = c2 * c2
        s4 = s2 * s2
        c2s2 = c2 * s2
        cs = c * s
        c3s = c2 * cs
        cs3 = s2 * cs
        sqrt6 = math.sqrt(6.0)
        sqrt6_c2s2 = sqrt6 * c2s2
        sqrt6_cs_c2ms2 = sqrt6 * cs * (c2 - s2)
        c2_c2m3s2 = c2 * (c2 - 3.0 * s2)
        s2_3c2ms2 = s2 * (3.0 * c2 - s2)
        c4_4c2s2_s4 = c4 - 4.0 * c2s2 + s4

        r0 = [c4, 2.0 * c3s, sqrt6_c2s2, 2.0 * cs3, s4]
        r1 = [-2.0 * c3s, c2_c2m3s2, sqrt6_cs_c2ms2, s2_3c2ms2, 2.0 * cs3]
        r2 = [sqrt6_c2s2, -sqrt6_cs_c2ms2, c4_4c2s2_s4, sqrt6_cs_c2ms2, sqrt6_c2s2]
        r3 = [-2.0 * cs3, s2_3c2ms2, -sqrt6_cs_c2ms2, c2_c2m3s2, 2.0 * c3s]
        r4 = [s4, -2.0 * cs3, sqrt6_c2s2, -2.0 * c3s, c4]

        rows = [jnp.stack(r, axis=-1) for r in (r0, r1, r2, r3, r4)]
        return jnp.stack(rows, axis=-2)

    c = jnp.cos(0.5 * beta)
    s = jnp.sin(0.5 * beta)
    fact = math.factorial
    rows = []
    for m in range(-l, l + 1):
        row = []
        for mp in range(-l, l + 1):
            k_min = max(0, m - mp)
            k_max = min(l + m, l - mp)
            val = jnp.zeros_like(beta)
            for k in range(k_min, k_max + 1):
                sign = (-1) ** k
                num = math.sqrt(fact(l + m) * fact(l - m) * fact(l + mp) * fact(l - mp))
                den = fact(l + m - k) * fact(k) * fact(l - mp - k) * fact(k - m + mp)
                c_pow = 2 * l + m - mp - 2 * k
                s_pow = 2 * k - m + mp
                coeff = sign * (num / den)
                term = coeff
                if c_pow > 0:
                    term = term * (c ** c_pow)
                if s_pow > 0:
                    term = term * (s ** s_pow)
                val = val + term
            row.append(val)
        rows.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows, axis=-2)


@jax.tree_util.register_pytree_node_class
class IMRPhenomXP:
    """IMRPhenomXP Precessing Dominant-Harmonic (l=2, |m|=2) Waveform Approximant in JAX.

    Twists up underlying coprecessing dominant (2, 2) mode from IMRPhenomXAS
    using NNLO angles or the published two-spin MSA approximation
    with Wigner small-d matrix frame rotations and polarization basis alignment.
    """

    def __new__(cls, *args, **kwargs):
        if args or kwargs:
            return cls.evaluate(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self):
        self.xas_model = IMRPhenomXAS()

    def tree_flatten(self):
        return ((), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    @staticmethod
    def evaluate(
        f: Optional[jnp.ndarray] = None,
        mass1: Union[float, jnp.ndarray] = 30.0,
        mass2: Union[float, jnp.ndarray] = 30.0,
        spin1x: Union[float, jnp.ndarray] = 0.0,
        spin1y: Union[float, jnp.ndarray] = 0.0,
        spin1z: Optional[Union[float, jnp.ndarray]] = None,
        spin2x: Union[float, jnp.ndarray] = 0.0,
        spin2y: Union[float, jnp.ndarray] = 0.0,
        spin2z: Optional[Union[float, jnp.ndarray]] = None,
        chi1: Optional[Union[float, jnp.ndarray]] = None,
        chi2: Optional[Union[float, jnp.ndarray]] = None,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        phi0: Union[float, jnp.ndarray] = 0.0,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[jnp.ndarray] = None,
        precession_method: str = "msa",
        prec_version: Optional[int] = None,
        convention: int = 1,
        final_spin_mod: Optional[int] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate precessing dominant-mode polarizations in JAX."""
        p_ver = resolve_precession_version(precession_method, prec_version)
        final_spin_version = kwargs.pop("final_spin_version", None)
        if final_spin_version is not None:
            if isinstance(final_spin_version, bool) or final_spin_version not in (0, 3):
                raise ValueError("final_spin_version must be 0 (NNLO) or 3 (MSA).")
            if final_spin_mod is not None and final_spin_mod != final_spin_version:
                raise ValueError(
                    "final_spin_mod and final_spin_version must agree when both are supplied."
                )
            final_spin_mod = final_spin_version
        if isinstance(final_spin_mod, bool) or final_spin_mod not in (None, 0, 3):
            raise ValueError("final_spin_mod must be None, 0 (NNLO), or 3 (MSA).")
        if isinstance(convention, bool) or convention not in (0, 1, 5, 6, 7):
            raise ValueError(f"Unsupported precession convention: {convention}.")
        if final_spin_mod is None and p_ver == 223:
            final_spin_mod = 3

        freqs = f if f is not None else sample_frequencies
        if freqs is None:
            raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        dtype = jnp.float64
        complex_dtype = jnp.complex128

        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1x = as_array(spin1x, dtype=dtype)
        s1y = as_array(spin1y, dtype=dtype)
        s2x = as_array(spin2x, dtype=dtype)
        s2y = as_array(spin2y, dtype=dtype)

        s1z_val = spin1z if spin1z is not None else (chi1 if chi1 is not None else 0.0)
        s2z_val = spin2z if spin2z is not None else (chi2 if chi2 is not None else 0.0)
        s1z = as_array(s1z_val, dtype=dtype)
        s2z = as_array(s2z_val, dtype=dtype)

        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)
        phi_c = as_array(kwargs.pop("phi_c", phic), dtype=dtype)
        t_c = as_array(kwargs.pop("t_c", tc), dtype=dtype)
        phi_0 = as_array(phi0, dtype=dtype)
        fref = as_array(f_ref, dtype=dtype) if f_ref is not None else None
        fref_eval = fref if fref is not None else jnp.asarray(20.0, dtype=dtype)

        (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
         dist, inc, phi_c, t_c, phi_0, fref_eval) = jnp.broadcast_arrays(
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
            dist, inc, phi_c, t_c, phi_0, fref_eval,
        )
        if fref is not None:
            fref = fref_eval

        # Canonical body ordering: primary mass m1 >= m2
        swap_mask = m1 < m2
        m1_eff = jnp.where(swap_mask, m2, m1)
        m2_eff = jnp.where(swap_mask, m1, m2)
        s1x_eff = jnp.where(swap_mask, s2x, s1x)
        s1y_eff = jnp.where(swap_mask, s2y, s1y)
        s1z_eff = jnp.where(swap_mask, s2z, s1z)
        s2x_eff = jnp.where(swap_mask, s1x, s2x)
        s2y_eff = jnp.where(swap_mask, s1y, s2y)
        s2z_eff = jnp.where(swap_mask, s1z, s2z)
        m1, m2 = m1_eff, m2_eff
        s1x, s1y, s1z = s1x_eff, s1y_eff, s1z_eff
        s2x, s2y, s2z = s2x_eff, s2y_eff, s2z_eff

        in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
        is_aligned = (in_plane_sq == 0.0)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        m_sec = total_mass * MSUN_SEC

        final_spin_override = None
        if final_spin_mod == 3:
            v0_remnant = (PI * m_sec * fref_eval) ** (1.0 / 3.0)
            final_spin_override = compute_msa_final_spin(
                v0_remnant, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
            )
        elif final_spin_mod == 0 or p_ver in (101, 102, 103, 104):
            chip = compute_effective_chip(m1, m2, s1x, s1y, s2x, s2y)
            final_spin_override = XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chip)

        is_batched = total_mass.ndim >= 1
        freqs = as_array(freqs, dtype=dtype)
        f_eval = freqs if (not is_batched or freqs.ndim > 1) else freqs[None, :]

        def unsqueeze_param(t: jnp.ndarray) -> jnp.ndarray:
            if is_batched and f_eval.ndim > t.ndim:
                return jnp.expand_dims(t, axis=-1)
            return t

        f_angles = jnp.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval))
        v = (PI * unsqueeze_param(m_sec) * f_angles) ** (1.0 / 3.0)
        v0 = (PI * unsqueeze_param(m_sec) * unsqueeze_param(fref_eval)) ** (1.0 / 3.0)

        alpha, beta, gamma, theta_jn, zeta, carrier_phase = evaluate_msa_precession(
            v=v,
            v0=v0,
            m1=unsqueeze_param(m1),
            m2=unsqueeze_param(m2),
            s1x=unsqueeze_param(s1x),
            s1y=unsqueeze_param(s1y),
            s1z=unsqueeze_param(s1z),
            s2x=unsqueeze_param(s2x),
            s2y=unsqueeze_param(s2y),
            s2z=unsqueeze_param(s2z),
            iota=unsqueeze_param(inc),
            phi_ref=unsqueeze_param(phi_c),
            prec_version=p_ver,
            convention=convention,
            return_carrier_phase=True,
            observer_azimuth=unsqueeze_param(phi_0),
        )

        carrier_phi_c = carrier_phase.squeeze(-1) if is_batched else carrier_phase
        hp_carrier, _ = IMRPhenomXAS.evaluate(
            f=f_eval,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=jnp.zeros_like(inc),
            phi_c=carrier_phi_c,
            t_c=t_c,
            f_ref=fref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            _final_spin_override=final_spin_override,
        )

        d_beta = wigner_small_d_matrix(2, beta)
        d_jn = wigner_small_d_matrix(2, unsqueeze_param(theta_jn))

        term_plus = jnp.zeros(f_eval.shape, dtype=complex_dtype)
        term_cross = jnp.zeros(f_eval.shape, dtype=complex_dtype)

        for m_val in range(-2, 3):
            m_idx = m_val + 2
            dm_minus = d_beta[..., m_idx, -2 + 2]
            dm_plus = d_beta[..., m_idx, 2 + 2]
            y_lm = d_jn[..., m_idx, 2 + 2]

            m_phase = float(m_val) * (-alpha)
            cos_m = jnp.cos(m_phase)
            sin_m = jnp.sin(m_phase)

            exp_minus_ima = cos_m + 1j * sin_m
            exp_plus_ima = cos_m - 1j * sin_m

            a_m_minus = exp_minus_ima * dm_minus * y_lm
            a_m_plus_star = exp_plus_ima * dm_plus * y_lm

            term_plus = term_plus + (a_m_minus + a_m_plus_star)
            term_cross = term_cross + (a_m_minus - a_m_plus_star)

        two_gamma = 2.0 * gamma
        exp_two_gamma = jnp.cos(two_gamma) + 1j * jnp.sin(two_gamma)

        hp_raw = exp_two_gamma * hp_carrier * term_plus
        hc_raw = 1j * exp_two_gamma * hp_carrier * term_cross

        two_zeta = 2.0 * unsqueeze_param(zeta)
        cos_2z = jnp.cos(two_zeta)
        sin_2z = jnp.sin(two_zeta)

        hp_precess = -(cos_2z * hp_raw + sin_2z * hc_raw)
        hc_precess = -(-sin_2z * hp_raw + cos_2z * hc_raw)

        if convention == 1:
            hp_al, hc_al = IMRPhenomXAS.evaluate(
                f=f_eval,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phi_c=phi_c - phi_0,
                t_c=t_c,
                f_ref=fref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )
            is_al_u = unsqueeze_param(is_aligned)
            hp_final = jnp.where(is_al_u, hp_al, hp_precess)
            hc_final = jnp.where(is_al_u, hc_al, hc_precess)
        else:
            hp_final = hp_precess
            hc_final = hc_precess

        hp_final = jnp.where(f_eval > 0.0, hp_final, jnp.zeros_like(hp_final))
        hc_final = jnp.where(f_eval > 0.0, hc_final, jnp.zeros_like(hc_final))

        return hp_final, hc_final

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


__all__ = [
    "IMRPhenomXP",
    "wigner_small_d_matrix",
]
