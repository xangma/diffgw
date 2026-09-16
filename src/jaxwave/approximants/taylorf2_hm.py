"""TaylorF2_HM Multipolar Frequency-Domain Gravitational Waveform Approximant in JAX.

Clean-room implementation based on published literature:
- Kidder (2008), Phys. Rev. D 77, 044016 (arXiv:0710.0614)
- Arun, Buonanno, Faye, Ochsner (2009), Phys. Rev. D 79, 104023 (arXiv:0810.5336)
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Mezzasoma & Yunes (2022), Phys. Rev. D 106, 024026 (arXiv:2203.08939)
- Bardeen, Press, Teukolsky (1972), Astrophys. J. 178, 347
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
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
    kerr_isco_frequency,
)
from jaxwave.approximants.taylorf2 import TaylorF2


def _swsh_projection_coefficients(
    l: int,
    m: int,
    iota: jnp.ndarray,
    phi0: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute polarization projection coefficients d_plus and d_cross for TaylorF2_HM.

    For positive frequencies, the contribution of modes (l, m) and (l, -m)
    to h_+ and h_x is given by (García-Quirós et al. (2020), Phys. Rev. D 102, 064002, Eq. 3):
        d_plus  = 0.5 * (Y_{l,-m} + (-1)^l * conj(Y_{l,m}))
        d_cross = 0.5i * (Y_{l,-m} - (-1)^l * conj(Y_{l,m}))
    where Y_{l,m} = {}_{-2}Y_{lm}(\\iota, \\phi_0).

    Returns:
        (d_plus, d_cross) complex arrays.
    """
    cos_i = jnp.cos(iota)
    sin_i = jnp.sin(iota)

    one_plus_cos = 1.0 + cos_i
    one_minus_cos = 1.0 - cos_i

    # Phase factors exp(i * m * phi0)
    m_phi0 = float(m) * phi0
    exp_pos = jnp.cos(m_phi0) + 1j * jnp.sin(m_phi0)
    exp_neg = jnp.conj(exp_pos)

    if l == 2 and m == 2:
        c_norm = math.sqrt(5.0 / (64.0 * math.pi))
        ylm = c_norm * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    elif l == 2 and m == 1:
        c_norm = math.sqrt(5.0 / (16.0 * math.pi))
        ylm = c_norm * sin_i * one_plus_cos * exp_pos
        yl_negm = c_norm * sin_i * one_minus_cos * exp_neg
        sign_l = 1.0
    elif l == 3 and m == 3:
        c_norm = math.sqrt(21.0 / (128.0 * math.pi))
        ylm = -c_norm * sin_i * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * sin_i * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = -1.0
    elif l == 3 and m == 2:
        c_norm = math.sqrt(7.0 / (64.0 * math.pi))
        ylm = c_norm * (one_plus_cos * one_plus_cos) * (3.0 * cos_i - 2.0) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * (3.0 * cos_i + 2.0) * exp_neg
        sign_l = -1.0
    elif l == 3 and m == 1:
        c_norm = math.sqrt(70.0 / (256.0 * math.pi))
        ylm = c_norm * sin_i * one_plus_cos * (3.0 * cos_i - 1.0) * exp_pos
        yl_negm = c_norm * sin_i * one_minus_cos * (1.0 + 3.0 * cos_i) * exp_neg
        sign_l = -1.0
    elif l == 4 and m == 4:
        c_norm = math.sqrt(63.0 / (256.0 * math.pi))
        ylm = c_norm * (sin_i * sin_i) * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (sin_i * sin_i) * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    elif l == 4 and m == 2:
        c_norm = 3.0 / (8.0 * math.sqrt(math.pi))
        ylm = c_norm * (one_plus_cos * one_plus_cos) * (
            7.0 * cos_i * cos_i - 7.0 * cos_i + 1.0
        ) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * (
            7.0 * cos_i * cos_i + 7.0 * cos_i + 1.0
        ) * exp_neg
        sign_l = 1.0
    elif l == 4 and m == 3:
        c_norm = 3.0 * math.sqrt(14.0 / (256.0 * math.pi))
        ylm = c_norm * sin_i * (one_plus_cos * one_plus_cos) * (1.0 - 2.0 * cos_i) * exp_pos
        yl_negm = c_norm * sin_i * (one_minus_cos * one_minus_cos) * (1.0 + 2.0 * cos_i) * exp_neg
        sign_l = 1.0
    else:
        raise ValueError(f"Mode ({l}, {m}) not currently supported in TaylorF2_HM")

    ylm_star = jnp.conj(ylm)
    d_plus = 0.5 * (yl_negm + sign_l * ylm_star)
    diff = yl_negm - sign_l * ylm_star
    d_cross = -0.5 * diff.imag + 0.5j * diff.real

    return d_plus, d_cross


class TaylorF2_HM:
    """Multipolar Post-Newtonian Stationary Phase Approximation (SPA) model in JAX.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 1), (3, 2), (3, 3),
    (4, 2), (4, 3), (4, 4)}.
    """

    DEFAULT_MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4))

    def __init__(self, modes: Optional[Sequence[Tuple[int, int]]] = None):
        if modes is not None:
            normalized_modes = tuple((int(l), int(m)) for l, m in modes)
            for l, m in normalized_modes:
                if (l, m) not in self.DEFAULT_MODES:
                    raise ValueError(
                        f"Mode ({l}, {m}) is not supported. Supported modes: {self.DEFAULT_MODES}"
                    )
            self.modes = normalized_modes
        else:
            self.modes = self.DEFAULT_MODES

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("modes", self.modes)
        return self.evaluate(*args, **kwargs)

    @classmethod
    def evaluate(
        cls,
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        phi0: Union[float, jnp.ndarray] = 0.0,
        lambda1: Optional[Union[float, jnp.ndarray]] = None,
        lambda2: Optional[Union[float, jnp.ndarray]] = None,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        f_isco_cutoff: bool = True,
        isco_type: str = "schwarzschild",
        smooth_cutoff: bool = False,
        roll_off_delta: float = 0.1,
        spin_order: int = 7,
        return_modes: bool = False,
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray, Dict[Tuple[int, int], Tuple[jnp.ndarray, jnp.ndarray]]],
    ]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including higher multipoles."""
        dtype = jnp.float64
        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1 = as_array(chi1, dtype=dtype)
        s2 = as_array(chi2, dtype=dtype)
        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)
        phi_c = as_array(phic, dtype=dtype)
        t_c = as_array(tc, dtype=dtype)
        phi_0 = as_array(phi0, dtype=dtype)

        l1 = as_array(lambda1, dtype=dtype) if lambda1 is not None else None
        l2 = as_array(lambda2, dtype=dtype) if lambda2 is not None else None

        active_modes = (
            tuple((int(l), int(m)) for l, m in modes) if modes is not None else cls.DEFAULT_MODES
        )
        for mode in active_modes:
            if mode not in cls.DEFAULT_MODES:
                raise ValueError(
                    f"Mode {mode} is not supported. Supported modes: {cls.DEFAULT_MODES}"
                )

        is_batched = m1.ndim >= 1
        if is_batched:
            f_eval = jnp.expand_dims(f, 0) if f.ndim == 1 else f
            inc_eval = jnp.expand_dims(inc, -1) if inc.ndim < f_eval.ndim else inc
            phi0_eval = jnp.expand_dims(phi_0, -1) if phi_0.ndim < f_eval.ndim else phi_0
        else:
            f_eval = f
            inc_eval = inc
            phi0_eval = phi_0

        total_m = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        # Cutoff frequency at ISCO
        if f_isco_cutoff:
            m_sec = total_m * MSUN_SEC
            if is_batched and f_eval.ndim > m_sec.ndim:
                m_sec_cutoff = jnp.expand_dims(m_sec, -1)
                total_m_cutoff = jnp.expand_dims(total_m, -1)
                chi_eff_cutoff = jnp.expand_dims((m1 * s1 + m2 * s2) / total_m, -1)
            else:
                m_sec_cutoff = m_sec
                total_m_cutoff = total_m
                chi_eff_cutoff = (m1 * s1 + m2 * s2) / total_m

            if isco_type.lower() == "kerr":
                f_isco_22 = kerr_isco_frequency(total_m_cutoff, chi_eff_cutoff)
            else:
                f_isco_22 = 1.0 / (6.0 ** 1.5 * PI * m_sec_cutoff)

        amp0 = TaylorF2.evaluate_amplitude(f=f, mass1=m1, mass2=m2, distance=dist)

        m_sec_b = total_m * MSUN_SEC
        if is_batched and f_eval.ndim > m_sec_b.ndim:
            m_sec_b = jnp.expand_dims(m_sec_b, -1)
            eta_b = jnp.expand_dims(eta, -1)
            delta_b = jnp.expand_dims(delta, -1)
        else:
            eta_b = eta
            delta_b = delta

        safe_f = jnp.maximum(f_eval, 1e-8)
        v22 = (PI * m_sec_b * safe_f) ** (1.0 / 3.0)

        mode_norm = math.sqrt(16.0 * math.pi / 5.0) * amp0

        h_plus_total = jnp.zeros_like(amp0, dtype=jnp.complex128)
        h_cross_total = jnp.zeros_like(h_plus_total)
        mode_dict = {}

        for l, m in active_modes:
            f_scaled = (2.0 / float(m)) * f_eval
            f_ref_scaled = (2.0 / float(m)) * as_array(f_ref, dtype=dtype) if f_ref is not None else None

            psi_base = TaylorF2.evaluate_phase(
                f=f_scaled,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                lambda1=l1,
                lambda2=l2,
                tc=t_c,
                phic=phi_c,
                f_ref=f_ref_scaled,
                spin_order=spin_order,
            )

            delta_phi = (3.0 * PI / 4.0) * (1.0 - float(m) / 2.0)
            psi_lm = (float(m) / 2.0) * psi_base + delta_phi

            if (l, m) == (2, 2):
                r_lm = jnp.ones_like(v22)
            elif (l, m) == (2, 1):
                factor_21 = delta_b / (3.0 * (2.0 ** (1.0 / 3.0)))
                r_lm = factor_21 * v22
            elif (l, m) == (3, 1):
                factor_31 = delta_b / (12.0 * math.sqrt(7.0) * (2.0 ** (5.0 / 6.0)))
                r_lm = factor_31 * v22
            elif (l, m) == (3, 3):
                factor_33 = (
                    (3.0 / 4.0)
                    * math.sqrt(5.0 / 7.0)
                    * (3.0 / 2.0) ** (5.0 / 6.0)
                    * delta_b
                )
                r_lm = factor_33 * v22
            elif (l, m) == (3, 2):
                factor_32 = (1.0 / 3.0) * math.sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta_b)
                r_lm = factor_32 * (v22 * v22)
            elif (l, m) == (4, 4):
                factor_44 = (8.0 / 9.0) * math.sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta_b)
                r_lm = factor_44 * (v22 * v22)
            elif (l, m) == (4, 2):
                factor_42 = (math.sqrt(5.0) / 63.0) * (1.0 - 3.0 * eta_b)
                r_lm = factor_42 * (v22 * v22)
            elif (l, m) == (4, 3):
                factor_43 = (
                    (3.0 / 4.0)
                    * math.sqrt(3.0 / 35.0)
                    * (3.0 / 2.0) ** (1.0 / 6.0)
                    * delta_b
                    * (1.0 - 2.0 * eta_b)
                )
                r_lm = factor_43 * (v22 * v22 * v22)
            else:
                raise ValueError(f"Mode ({l}, {m}) not recognized")

            amp_lm = mode_norm * r_lm

            if f_isco_cutoff:
                f_cut_m = (float(m) / 2.0) * f_isco_22
                if not smooth_cutoff:
                    mask = (f_eval <= f_cut_m) & (f_eval > 0.0)
                    amp_lm = jnp.where(mask, amp_lm, jnp.zeros_like(amp_lm))
                else:
                    f_start_m = (1.0 - roll_off_delta) * f_cut_m
                    x = jnp.clip((f_eval - f_start_m) / (f_cut_m - f_start_m + 1e-30), 0.0, 1.0)
                    w = 0.5 * (1.0 + jnp.cos(PI * x))
                    w = jnp.where(f_eval <= f_start_m, 1.0, w)
                    w = jnp.where(f_eval >= f_cut_m, 0.0, w)
                    w = jnp.where(f_eval > 0.0, w, 0.0)
                    amp_lm = amp_lm * w

            exp_neg_i_psi = jnp.cos(psi_lm) - 1j * jnp.sin(psi_lm)
            h_mode = amp_lm * exp_neg_i_psi

            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi0_eval)
            hp_lm = d_plus * h_mode
            hc_lm = d_cross * h_mode

            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm

            if return_modes:
                mode_dict[(l, m)] = (hp_lm, hc_lm)

        if return_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total


__all__ = [
    "TaylorF2_HM",
    "_swsh_projection_coefficients",
]
