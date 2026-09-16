"""TaylorF2 Frequency-Domain Gravitational Waveform Approximant in JAX.

Clean-room implementation based on published literature:
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- Buonanno et al. (2009), Phys. Rev. D 80, 084043 (arXiv:0907.0700)
- Khan et al. (2016), Phys. Rev. D 93, 044007 (arXiv:1508.07253)
- Wade et al. (2014), Phys. Rev. D 89, 103012 (arXiv:1402.5156)
"""

from typing import Optional, Tuple, Union
import jax.numpy as jnp
from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    EULER_GAMMA,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    component_masses_to_delta,
    component_masses_to_chirp_mass,
    spin_combinations,
    tidal_combinations,
    kerr_isco_frequency,
)


class TaylorF2:
    """Post-Newtonian Stationary Phase Approximation (SPA) frequency-domain model in JAX."""

    @staticmethod
    def phase_coefficients(
        mass1: jnp.ndarray,
        mass2: jnp.ndarray,
        chi1: jnp.ndarray,
        chi2: jnp.ndarray,
        lambda1: Optional[jnp.ndarray] = None,
        lambda2: Optional[jnp.ndarray] = None,
        quad_monopole1: Optional[jnp.ndarray] = None,
        quad_monopole2: Optional[jnp.ndarray] = None,
        spin_order: int = 7,
        tidal_order: int = 12,
        phenom_d_calibration: bool = False,
    ) -> Tuple[jnp.ndarray, ...]:
        """Compute the 3.5PN point-particle, spin, and tidal phase coefficients."""
        eta = component_masses_to_eta(mass1, mass2)
        delta = component_masses_to_delta(mass1, mass2)
        chi_s, chi_a, _, _ = spin_combinations(mass1, mass2, chi1, chi2)

        total_mass = mass1 + mass2
        m1_ratio = mass1 / total_mass
        m2_ratio = mass2 / total_mass

        eta2 = eta * eta
        eta3 = eta2 * eta

        # 0PN
        phi0 = jnp.ones_like(eta)

        # 1PN
        phi2 = (3715.0 / 756.0) + (55.0 / 9.0) * eta

        # 1.5PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B2)
        phi3_pp = -16.0 * PI
        phi3_so = (113.0 / 3.0) * delta * chi_a + ((113.0 / 3.0) - (76.0 * eta / 3.0)) * chi_s
        phi3 = phi3_pp + (phi3_so if (spin_order >= 3 or spin_order < 0) else jnp.zeros_like(eta))

        # 2PN: Point-particle + Spin-Spin (Khan et al. 2016 Eq. B3, Poisson 1998)
        phi4_pp = (15293365.0 / 508032.0) + (27145.0 / 504.0) * eta + (3085.0 / 72.0) * eta2
        phi4_ss = (
            (-405.0 / 8.0 + 200.0 * eta) * (chi_a * chi_a)
            - (405.0 / 4.0) * delta * chi_a * chi_s
            + (-405.0 / 8.0 + 5.0 * eta / 2.0) * (chi_s * chi_s)
        )

        cq1 = 1.0 if quad_monopole1 is None else quad_monopole1
        cq2 = 1.0 if quad_monopole2 is None else quad_monopole2
        if cq1 != 1.0 or cq2 != 1.0:
            phi4_qm = -(50.0 / eta) * (
                (m1_ratio * m1_ratio) * (cq1 - 1.0) * (chi1 * chi1)
                + (m2_ratio * m2_ratio) * (cq2 - 1.0) * (chi2 * chi2)
            )
        else:
            phi4_qm = jnp.zeros_like(eta)

        phi4 = phi4_pp
        if spin_order >= 4 or spin_order < 0:
            phi4 = phi4 + phi4_ss + phi4_qm

        # 2.5PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B4)
        phi5_pp = (38645.0 * PI / 756.0) - (65.0 * PI * eta / 9.0)
        phi5_so = (
            delta * (-(732985.0 / 2268.0) - (140.0 * eta / 9.0)) * chi_a
            + (-(732985.0 / 2268.0) + (24260.0 * eta / 81.0) + (340.0 * eta2 / 9.0)) * chi_s
        )
        phi5_bracket = phi5_pp + (phi5_so if (spin_order >= 5 or spin_order < 0) else jnp.zeros_like(eta))
        phi5 = phi5_bracket
        phi5_log = 3.0 * phi5_bracket

        # 3PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B5) + Spin-Spin
        phi6_pp_const = (
            (11583231236531.0 / 4694215680.0)
            - (6848.0 / 21.0) * EULER_GAMMA
            - (640.0 / 3.0) * (PI * PI)
            + ((-15737765635.0 / 3048192.0) + (2255.0 / 12.0) * (PI * PI)) * eta
            + (76055.0 / 1728.0) * eta2
            - (127825.0 / 1296.0) * eta3
        )
        phi6_so = (2270.0 * PI / 3.0) * delta * chi_a + ((2270.0 * PI / 3.0) - 520.0 * PI * eta) * chi_s
        phi6_ss = (
            chi_s * chi_a * (75515.0 * delta / 144.0 - 8225.0 * delta * eta / 18.0)
            + (chi_a * chi_a) * (-480.0 * eta2 - 263245.0 * eta / 252.0 + 75515.0 / 288.0)
            + (chi_s * chi_s) * (1255.0 * eta2 / 9.0 - 232415.0 * eta / 504.0 + 75515.0 / 288.0)
        )
        phi6 = phi6_pp_const
        if spin_order >= 5 or spin_order < 0:
            phi6 = phi6 + phi6_so
        if not phenom_d_calibration and (spin_order >= 6 or spin_order < 0):
            phi6 = phi6 + phi6_ss
        phi6_log = -(6848.0 / 21.0) * jnp.ones_like(eta)

        # 3.5PN: Point-particle + Spin-Orbit (Khan et al. 2016 Eq. B6)
        phi7_pp = (
            (77096675.0 * PI / 254016.0)
            + (378515.0 * PI * eta / 1512.0)
            - (74045.0 * PI * eta2 / 756.0)
        )
        phi7_so = (
            delta * (-(25150083775.0 / 3048192.0) + (26804935.0 * eta / 6048.0) - (1985.0 * eta2 / 48.0)) * chi_a
            + (-(25150083775.0 / 3048192.0) + (10566655595.0 * eta / 762048.0) - (1042165.0 * eta2 / 3024.0) + (5345.0 * eta3 / 36.0)) * chi_s
        )
        phi7_sss = (
            chi_s * (-1140.0 * PI * delta * chi_a + (80.0 * eta2 - 7270.0 * eta + 14585.0 / 8.0) * (chi_a * chi_a))
            + (chi_a * chi_a * chi_a) * (14585.0 * delta / 24.0 - 2380.0 * delta * eta)
            + (chi_s * chi_s) * (chi_a * (14585.0 * delta / 8.0 - 215.0 * delta * eta / 2.0) + 40.0 * PI * eta - 570.0 * PI)
            + (chi_s * chi_s * chi_s) * (100.0 * eta2 / 3.0 - 475.0 * eta / 6.0 + 14585.0 / 24.0)
            + (2240.0 * PI * eta - 570.0 * PI) * (chi_a * chi_a)
        )
        phi7 = phi7_pp
        if spin_order >= 5 or spin_order < 0:
            phi7 = phi7 + phi7_so
        if not phenom_d_calibration and spin_order >= 8:
            phi7 = phi7 + phi7_sss

        # Tidal terms
        if lambda1 is not None and lambda2 is not None and (tidal_order >= 10 or tidal_order < 0):
            lambda_tilde, delta_lambda_tilde = tidal_combinations(mass1, mass2, lambda1, lambda2)
            phi10_tidal = -(39.0 / 2.0) * lambda_tilde
            if tidal_order >= 12 or tidal_order < 0:
                phi12_tidal = -(3115.0 / 64.0) * lambda_tilde + (6595.0 / 364.0) * delta * delta_lambda_tilde
            else:
                phi12_tidal = jnp.zeros_like(eta)
            if tidal_order >= 13 or tidal_order < 0:
                phi13_tidal = (39.0 / 2.0) * PI * lambda_tilde
            else:
                phi13_tidal = jnp.zeros_like(eta)
        else:
            phi10_tidal = jnp.zeros_like(eta)
            phi12_tidal = jnp.zeros_like(eta)
            phi13_tidal = jnp.zeros_like(eta)

        return phi0, phi2, phi3, phi4, phi5, phi5_log, phi6, phi6_log, phi7, phi10_tidal, phi12_tidal, phi13_tidal

    @staticmethod
    def evaluate_phase(
        f: jnp.ndarray,
        mass1: jnp.ndarray,
        mass2: jnp.ndarray,
        chi1: jnp.ndarray,
        chi2: jnp.ndarray,
        lambda1: Optional[jnp.ndarray] = None,
        lambda2: Optional[jnp.ndarray] = None,
        tc: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        quad_monopole1: Optional[jnp.ndarray] = None,
        quad_monopole2: Optional[jnp.ndarray] = None,
        spin_order: int = 5,
        tidal_order: int = 12,
        phenom_d_calibration: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
    ) -> jnp.ndarray:
        """Compute the full SPA phase Psi(f) with configurable phasing convention in JAX."""
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)

        if f.ndim == 1 and total_mass.ndim >= 1:
            if f.shape[0] == total_mass.shape[0]:
                f = jnp.expand_dims(f, -1)
            else:
                f = jnp.expand_dims(f, 0)

        f_safe = jnp.maximum(f, 1e-8)
        m_sec = total_mass * MSUN_SEC
        if m_sec.ndim >= 1 and f.ndim > m_sec.ndim:
            m_sec = jnp.expand_dims(m_sec, -1)
            eta = jnp.expand_dims(eta, -1)

        v = (PI * m_sec * f_safe) ** (1.0 / 3.0)
        v2 = v * v
        v3 = v2 * v
        v4 = v2 * v2
        v5 = v4 * v
        v6 = v3 * v3
        v7 = v6 * v

        (
            phi0, phi2, phi3, phi4, phi5, phi5_log,
            phi6, phi6_log, phi7, phi10_t, phi12_t, phi13_t
        ) = TaylorF2.phase_coefficients(
            mass1=mass1,
            mass2=mass2,
            chi1=chi1,
            chi2=chi2,
            lambda1=lambda1,
            lambda2=lambda2,
            quad_monopole1=quad_monopole1,
            quad_monopole2=quad_monopole2,
            spin_order=spin_order,
            tidal_order=tidal_order,
            phenom_d_calibration=phenom_d_calibration,
        )

        def expand_if_needed(t: jnp.ndarray) -> jnp.ndarray:
            if t.ndim >= 1 and f.ndim > t.ndim:
                return jnp.expand_dims(t, -1)
            return t

        phi0 = expand_if_needed(phi0)
        phi2 = expand_if_needed(phi2)
        phi3 = expand_if_needed(phi3)
        phi4 = expand_if_needed(phi4)
        phi5 = expand_if_needed(phi5)
        phi5_log = expand_if_needed(phi5_log)
        phi6 = expand_if_needed(phi6)
        phi6_log = expand_if_needed(phi6_log)
        phi7 = expand_if_needed(phi7)
        phi10_t = expand_if_needed(phi10_t)
        phi12_t = expand_if_needed(phi12_t)
        phi13_t = expand_if_needed(phi13_t)

        ln_v = jnp.log(v)
        ln_4v = jnp.log(4.0 * v)

        pn_sum = (
            phi0
            + phi2 * v2
            + phi3 * v3
            + phi4 * v4
            + (phi5 + phi5_log * ln_v) * v5
            + (phi6 + phi6_log * ln_4v) * v6
            + phi7 * v7
            + phi10_t * (v5 * v5)
            + phi12_t * (v6 * v6)
            + phi13_t * (v6 * v7)
        )

        psi_raw = -(PI / 4.0) + (3.0 / (128.0 * eta * v5)) * pn_sum

        t_c_t = expand_if_needed(as_array(tc, dtype=f.dtype))
        phi_c_t = expand_if_needed(as_array(phic, dtype=f.dtype))

        conv = reference_frame if reference_frame is not None else (
            phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
        )
        conv = conv.lower()

        if conv in ("reference_frequency", "reffreq", "ref"):
            ref_f = f_ref if f_ref is not None else (f.reshape(-1)[0] if f.size > 0 else 20.0)
            f_ref_t = as_array(ref_f, dtype=f.dtype)
            if f_ref_t.ndim == 0:
                f_ref_eval = f_ref_t.reshape(1)
            elif total_mass.ndim >= 1 and f_ref_t.ndim == 1 and f_ref_t.shape[0] == total_mass.shape[0]:
                f_ref_eval = f_ref_t.reshape(-1, 1)
            else:
                f_ref_eval = f_ref_t

            psi_ref = TaylorF2.evaluate_phase(
                f=f_ref_eval,
                mass1=mass1,
                mass2=mass2,
                chi1=chi1,
                chi2=chi2,
                lambda1=lambda1,
                lambda2=lambda2,
                tc=jnp.zeros_like(t_c_t),
                phic=jnp.zeros_like(phi_c_t),
                f_ref=None,
                quad_monopole1=quad_monopole1,
                quad_monopole2=quad_monopole2,
                spin_order=spin_order,
                tidal_order=tidal_order,
                phenom_d_calibration=phenom_d_calibration,
                phasing_convention="coalescence",
            )
            if total_mass.ndim >= 1 and f.ndim > total_mass.ndim:
                psi_ref = expand_if_needed(psi_ref)
            psi = (psi_raw - psi_ref) + (3.0 * PI / 4.0) + 2.0 * PI * f * t_c_t - 2.0 * phi_c_t
        else:
            psi = psi_raw + 2.0 * PI * f * t_c_t - 2.0 * phi_c_t

        return psi

    @staticmethod
    def evaluate_amplitude(
        f: jnp.ndarray,
        mass1: jnp.ndarray,
        mass2: jnp.ndarray,
        distance: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Newtonian leading-order amplitude A(f) = A0 * f^(-7/6) in JAX."""
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)

        m_sec = total_mass * MSUN_SEC
        d_sec = (distance * MPC_SI) / C_SI

        is_batched = total_mass.ndim >= 1
        if f.ndim == 1 and is_batched:
            f_eval = jnp.expand_dims(f, 0)
            m_sec = jnp.expand_dims(m_sec, -1)
            eta = jnp.expand_dims(eta, -1)
            d_sec = jnp.expand_dims(d_sec, -1)
        else:
            f_eval = f
            if m_sec.ndim >= 1 and f_eval.ndim > m_sec.ndim:
                m_sec = jnp.expand_dims(m_sec, -1)
                eta = jnp.expand_dims(eta, -1)
                d_sec = jnp.expand_dims(d_sec, -1)

        f_eval_safe = jnp.maximum(f_eval, 1e-8)
        prefactor = jnp.sqrt(5.0 * eta / 24.0) / (PI ** (2.0 / 3.0) * d_sec)
        a0 = prefactor * (m_sec ** (5.0 / 6.0))
        amp = a0 * (f_eval_safe ** (-7.0 / 6.0))
        amp = jnp.where(f_eval > 0.0, amp, jnp.zeros_like(amp))
        return amp

    @staticmethod
    def evaluate(
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
        lambda1: Optional[Union[float, jnp.ndarray]] = None,
        lambda2: Optional[Union[float, jnp.ndarray]] = None,
        f_isco_cutoff: bool = True,
        isco_type: str = "schwarzschild",
        smooth_cutoff: bool = False,
        roll_off_delta: float = 0.1,
        spin_order: int = 7,
        tidal_order: int = 12,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate frequency-domain polarizations (h_plus, h_cross) using TaylorF2 in JAX."""
        dtype = jnp.float64
        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1 = as_array(chi1, dtype=dtype)
        s2 = as_array(chi2, dtype=dtype)
        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)

        l1 = as_array(lambda1, dtype=dtype) if lambda1 is not None else None
        l2 = as_array(lambda2, dtype=dtype) if lambda2 is not None else None

        psi = TaylorF2.evaluate_phase(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            lambda1=l1,
            lambda2=l2,
            tc=tc,
            phic=phic,
            f_ref=f_ref,
            spin_order=spin_order,
            tidal_order=tidal_order,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
        )

        amp = TaylorF2.evaluate_amplitude(
            f=f,
            mass1=m1,
            mass2=m2,
            distance=dist,
        )

        cos_i = jnp.cos(inc)
        plus_factor = 0.5 * (1.0 + cos_i * cos_i)
        cross_factor = cos_i

        if (m1.ndim >= 1) and (f.ndim == 1 or f.shape[0] != m1.shape[0]):
            plus_factor = jnp.expand_dims(plus_factor, -1)
            cross_factor = jnp.expand_dims(cross_factor, -1)

        # Cutoff frequency
        if f_isco_cutoff:
            total_mass = m1 + m2
            if isco_type.lower() == "kerr":
                chi_s, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)
                f_isco = kerr_isco_frequency(total_mass, chi_pn)
            else:
                m_sec = total_mass * MSUN_SEC
                f_isco = 1.0 / (6.0 * jnp.sqrt(6.0) * PI * m_sec)

            if m1.ndim >= 1 and f.ndim > m1.ndim:
                f_isco = jnp.expand_dims(f_isco, -1)

            if smooth_cutoff:
                f_start = (1.0 - roll_off_delta) * f_isco
                theta = jnp.clip((f - f_start) / (f_isco - f_start), 0.0, 1.0)
                window = 0.5 * (1.0 + jnp.cos(PI * theta))
                window = jnp.where(f < f_start, 1.0, window)
                window = jnp.where(f > f_isco, 0.0, window)
                amp = amp * window
            else:
                mask = (f <= f_isco) & (f > 0.0)
                amp = jnp.where(mask, amp, 0.0)

        # Polarizations: h+ = amp * plus_factor * e^(-i*psi), hx = amp * cross_factor * e^(-i*(psi + pi/2))
        h_plus = (amp * plus_factor) * (jnp.cos(psi) - 1j * jnp.sin(psi))
        h_cross = (amp * cross_factor) * (jnp.cos(psi + PI / 2.0) - 1j * jnp.sin(psi + PI / 2.0))

        return h_plus, h_cross

    @staticmethod
    def __call__(*args, **kwargs):
        return TaylorF2.evaluate(*args, **kwargs)
