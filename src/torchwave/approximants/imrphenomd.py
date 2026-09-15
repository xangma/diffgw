"""IMRPhenomD Frequency-Domain Aligned-Spin Gravitational Waveform Approximant.

Waveform equations based on published literature:
- Husa, Khan, Hannam, Pürrer, Ohme, Forteza, Bohé (2016), Phys. Rev. D 93, 044006 (arXiv:1508.07250) [Paper I]
- Khan, Husa, Hannam, Ohme, Pürrer, Forteza, Bohé (2016), Phys. Rev. D 93, 044007 (arXiv:1508.07253) [Paper II]
- Stein (2019), JOSS 4(42), 1683 [independent Kerr eigenproblem solves]

Inherited unrounded calibration constants require separate provenance review;
the paper citations alone do not establish the origin of every stored digit.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import math

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    EULER_GAMMA,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
    spin_combinations,
    total_spin_for_remnant,
)
from torchwave.approximants.taylorf2 import TaylorF2
# Compatibility re-export for existing diagnostics and non-D callers.  The D
# remnant path below uses the independent cyclic sampler directly.
import torchwave.approximants.kerr_qnm_interp as _kerr_qnm_interp

evaluate_kerr_qnm_220 = _kerr_qnm_interp.evaluate_kerr_qnm_220
from torchwave.approximants.kerr_qnm_d_sampling_clean import evaluate_d_qnm_220


from torchwave.approximants.phenomd_unrounded_fits import (
    TABLE_V_UNROUNDED_COEFFS,
    FINAL_SPIN_ETA_POLY,
    FINAL_SPIN_S_COEFFS,
    ERAD_ETA_NUMERATOR,
    ERAD_S_NUMERATOR,
    ERAD_S_DENOMINATOR,
)


# Table V of Khan et al. (2016), arXiv:1508.07253
# Inherited unrounded fitting coefficients; see module provenance caveat.
TABLE_V_COEFFS: Dict[str, Tuple[float, ...]] = TABLE_V_UNROUNDED_COEFFS



class IMRPhenomD:
    """IMRPhenomD Frequency-Domain Aligned-Spin Phenomenological Waveform Model."""

    @staticmethod
    def amplitude_peak_frequency(f_rd, f_damp, gamma2, gamma3):
        """Evaluate Khan et al. Eq. (20), retaining the inherited no-root continuation.

        Rationalizing sqrt(1-gamma2**2)-1 avoids cancellation at gamma2=0.
        For gamma2 >= 1 the existing continuation replaces the square root
        with zero; this join frequency need not be an amplitude extremum.
        The branch point has no finite mathematical derivative. At equality
        autograd follows the continuation branch, without an epsilon shift.
        """
        discriminant = 1.0 - gamma2.square()
        has_root = discriminant > 0.0
        root = torch.sqrt(torch.where(has_root, discriminant, torch.ones_like(discriminant)))
        safe_gamma2 = torch.where(has_root, torch.ones_like(gamma2), gamma2)
        offset = torch.where(has_root, -gamma2 / (1.0 + root), -1.0 / safe_gamma2)
        return torch.abs(f_rd + f_damp * gamma3 * offset)

    @staticmethod
    def eval_table_v(name: str, eta: torch.Tensor, chi_pn: torch.Tensor) -> torch.Tensor:
        """Evaluate hierarchical polynomial mapping (Eq. 31 of Khan et al. 2016)."""
        coeffs = TABLE_V_COEFFS[name]
        c = [torch.tensor(val, dtype=eta.dtype, device=eta.device) for val in coeffs]

        eta2 = eta * eta
        x = chi_pn - 1.0
        x2 = x * x
        x3 = x2 * x

        poly = (
            c[0] + c[1] * eta
            + x * (c[2] + c[3] * eta + c[4] * eta2)
            + x2 * (c[5] + c[6] * eta + c[7] * eta2)
            + x3 * (c[8] + c[9] * eta + c[10] * eta2)
        )
        return poly

    @staticmethod
    def remnant_properties(
        mass1: torch.Tensor,
        mass2: torch.Tensor,
        chi1: torch.Tensor,
        chi2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute final mass M_f, final spin a_f, QNM frequency f_RD, and damping width f_damp.

        References:
        - Husa et al. (2016) PRD 93, 044006 (arXiv:1508.07250, Eqs. 16, 17)
        - Stein (2019), JOSS 4(42), 1683, independent Kerr eigenproblem solves

        Returns:
            m_final: Remnant mass in solar masses
            a_final: Dimensionless remnant Kerr spin parameter
            f_rd: Dimensionless ringdown frequency (M * f_RD)
            f_damp: Dimensionless ringdown damping frequency (M * f_damp)
        """
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta

        s, s_hat = total_spin_for_remnant(mass1, mass2, chi1, chi2)
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s

        # Final spin fit (Eq. 16 of Husa et al. 2016) with unrounded fits
        p_eta = FINAL_SPIN_ETA_POLY
        s_c = FINAL_SPIN_S_COEFFS
        a_final = (
            s + p_eta[0] * eta + p_eta[1] * eta2 + p_eta[2] * eta3 + p_eta[3] * eta4
            + (s_c[0][0] * s + s_c[1][0] * s2 + s_c[2][0] * s3 + s_c[3][0] * s4) * eta
            + (s_c[0][1] * s + s_c[1][1] * s2 + s_c[2][1] * s3 + s_c[3][1] * s4) * eta2
        )
        # Physical Kerr limit clamping to avoid complex QNM roots
        a_final = torch.clamp(a_final, -0.999, 0.999)

        # Radiated energy non-spinning fit
        e_eta = ERAD_ETA_NUMERATOR
        erad_ns = e_eta[0] * eta + e_eta[1] * eta2 + e_eta[2] * eta3 + e_eta[3] * eta4

        # Radiated energy full fit (Eq. 17 of Husa et al. 2016)
        e_num = ERAD_S_NUMERATOR
        e_den = ERAD_S_DENOMINATOR
        num = 1.0 + s_hat * (e_num[0] + e_num[1] * eta + e_num[2] * eta2)
        den = 1.0 + s_hat * (e_den[0] + e_den[1] * eta + e_den[2] * eta2)
        erad_over_m = erad_ns * (num / den)

        m_final = total_mass * (1.0 - erad_over_m)

        # Independent rounded-cubic Kerr QNM evaluation (l=2, m=2, n=0).
        # The sampler returns cyclic final-mass units (M_f f); only the
        # total-to-final mass ratio belongs in this D call site.
        f_rd_final, f_damp_final = evaluate_d_qnm_220(a_final)

        # Dimensionless QNM frequencies scaled to total binary mass M:
        m_ratio = total_mass / m_final
        f_rd = m_ratio * f_rd_final
        f_damp = m_ratio * f_damp_final

        return m_final, a_final, f_rd, f_damp

    @staticmethod
    def pn_amplitude_coefficients(
        eta: torch.Tensor,
        delta: torch.Tensor,
        chi_s: torch.Tensor,
        chi_a: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute the PN amplitude coefficients A_0 through A_6 (Eqs. 22-28 of Khan et al. 2016)."""
        eta2 = eta * eta
        eta3 = eta2 * eta

        chi_s2 = chi_s * chi_s
        chi_a2 = chi_a * chi_a

        a0 = torch.ones_like(eta)
        a1 = torch.zeros_like(eta)
        a2 = -(323.0 / 224.0) + (451.0 / 168.0) * eta
        a3 = (27.0 / 8.0) * delta * chi_a + ((27.0 / 8.0) - (11.0 / 6.0) * eta) * chi_s
        a4 = (
            -(27312085.0 / 8128512.0)
            - (1975055.0 / 338688.0) * eta
            + (105271.0 / 24192.0) * eta2
            + (-(81.0 / 32.0) + 8.0 * eta) * chi_a2
            - (81.0 / 16.0) * delta * chi_a * chi_s
            + (-(81.0 / 32.0) + (17.0 / 8.0) * eta) * chi_s2
        )
        a5 = (
            -(85.0 * PI / 64.0) + (85.0 * PI / 16.0) * eta
            + delta * ((285197.0 / 16128.0) - (1579.0 / 4032.0) * eta) * chi_a
            + ((285197.0 / 16128.0) - (15317.0 / 672.0) * eta - (2227.0 / 1008.0) * eta2) * chi_s
        )
        a6 = (
            -(177520268561.0 / 8583708672.0)
            + ((545384828789.0 / 5007163392.0) - (205.0 * PI * PI / 48.0)) * eta
            - (3248849057.0 / 178827264.0) * eta2
            + (34473079.0 / 6386688.0) * eta3
            + ((1614569.0 / 64512.0) - (1873643.0 / 16128.0) * eta + (2167.0 / 42.0) * eta2) * chi_a2
            + ((31.0 * PI / 12.0) - (7.0 * PI / 3.0) * eta) * chi_s
            + ((1614569.0 / 64512.0) - (61391.0 / 1344.0) * eta + (57451.0 / 4032.0) * eta2) * chi_s2
            + delta * chi_a * ((31.0 * PI / 12.0) + ((1614569.0 / 32256.0) - (165961.0 / 2688.0) * eta) * chi_s)
        )
        return a0, a1, a2, a3, a4, a5, a6

    @staticmethod
    def intermediate_amplitude_coeffs(
        f1: float,
        f3: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor,
        d1: torch.Tensor,
        d3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the 5x5 Vandermonde-Hermite collocation system for intermediate amplitude (Eq. 30).

        [ 1  f1  f1^2  f1^3  f1^4 ] [ delta_0 ]   [ v1 ]
        [ 1  f2  f2^2  f2^3  f2^4 ] [ delta_1 ]   [ v2 ]
        [ 1  f3  f3^2  f3^3  f3^4 ] [ delta_2 ] = [ v3 ]
        [ 0   1  2*f1  3*f1^2 4*f1^3 ] [ delta_3 ]   [ d1 ]
        [ 0   1  2*f3  3*f3^2 4*f3^3 ] [ delta_4 ]   [ d3 ]
        """
        batch_shape = f3.shape
        f3_flat = f3.reshape(-1)
        v1_flat = v1.reshape(-1)
        v2_flat = v2.reshape(-1)
        v3_flat = v3.reshape(-1)
        d1_flat = d1.reshape(-1)
        d3_flat = d3.reshape(-1)

        b_size = f3_flat.shape[0]
        dtype = f3.dtype
        device = f3.device

        f1_t = torch.full((b_size,), f1, dtype=dtype, device=device)
        f2_t = 0.5 * (f1_t + f3_flat)

        f1_2, f1_3, f1_4 = f1_t * f1_t, f1_t**3, f1_t**4
        f2_2, f2_3, f2_4 = f2_t * f2_t, f2_t**3, f2_t**4
        f3_2, f3_3, f3_4 = f3_flat * f3_flat, f3_flat**3, f3_flat**4

        ones = torch.ones(b_size, dtype=dtype, device=device)
        zeros = torch.zeros(b_size, dtype=dtype, device=device)

        row0 = torch.stack([ones, f1_t, f1_2, f1_3, f1_4], dim=-1)
        row1 = torch.stack([ones, f2_t, f2_2, f2_3, f2_4], dim=-1)
        row2 = torch.stack([ones, f3_flat, f3_2, f3_3, f3_4], dim=-1)
        row3 = torch.stack([zeros, ones, 2.0 * f1_t, 3.0 * f1_2, 4.0 * f1_3], dim=-1)
        row4 = torch.stack([zeros, ones, 2.0 * f3_flat, 3.0 * f3_2, 4.0 * f3_3], dim=-1)

        mat = torch.stack([row0, row1, row2, row3, row4], dim=1)  # (B, 5, 5)
        rhs = torch.stack([v1_flat, v2_flat, v3_flat, d1_flat, d3_flat], dim=-1).unsqueeze(-1)  # (B, 5, 1)

        sol = torch.linalg.solve(mat, rhs).squeeze(-1)  # (B, 5)

        d0 = sol[:, 0].reshape(batch_shape)
        d1_c = sol[:, 1].reshape(batch_shape)
        d2 = sol[:, 2].reshape(batch_shape)
        d3_c = sol[:, 3].reshape(batch_shape)
        d4 = sol[:, 4].reshape(batch_shape)

        return d0, d1_c, d2, d3_c, d4

    @staticmethod
    def compute_slope_peak(
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        *,
        remnant_provider=None,
    ) -> torch.Tensor:
        """Paper merger-ringdown phase derivative dPsi/d(Mf) evaluated at f_peak."""
        # Preserve the waveform parameter precision.  Coercing batched
        # float32 inputs to as_tensor's float64 default promotes the batched
        # gauge term, while PyTorch keeps a 0-D scalar operand weak, so
        # scalar and one-row calls otherwise take different precision paths.
        values = (mass1, mass2, chi1, chi2)
        input_dtype = None
        input_device = None
        for value in values:
            if isinstance(value, torch.Tensor):
                if input_device is None:
                    input_device = value.device
                if value.is_floating_point():
                    input_dtype = (value.dtype if input_dtype is None
                                   else torch.promote_types(input_dtype, value.dtype))
        if input_dtype is None:
            input_dtype = torch.float64

        def coerce(value: Union[float, torch.Tensor]) -> torch.Tensor:
            return torch.as_tensor(value, device=input_device, dtype=input_dtype)

        m1 = coerce(mass1)
        m2 = coerce(mass2)
        s1 = coerce(chi1)
        s2 = coerce(chi2)
        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        _, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)

        gamma2 = IMRPhenomD.eval_table_v("gamma2", eta, chi_pn)
        gamma3 = IMRPhenomD.eval_table_v("gamma3", eta, chi_pn)
        alpha1_raw = IMRPhenomD.eval_table_v("alpha1", eta, chi_pn)
        alpha2 = IMRPhenomD.eval_table_v("alpha2", eta, chi_pn)
        alpha3 = IMRPhenomD.eval_table_v("alpha3", eta, chi_pn)
        alpha4 = IMRPhenomD.eval_table_v("alpha4", eta, chi_pn)
        alpha5 = IMRPhenomD.eval_table_v("alpha5", eta, chi_pn)

        m_final, a_final, f_rd, f_damp = (IMRPhenomD.remnant_properties if remnant_provider is None else remnant_provider)(m1, m2, s1, s2)
        f_peak = IMRPhenomD.amplitude_peak_frequency(f_rd, f_damp, gamma2, gamma3)

        atan_lorentz_peak = (alpha4 * f_damp) / (f_damp * f_damp + (f_peak - alpha5 * f_rd) ** 2)
        slope_peak = (
            alpha1_raw
            + alpha2 * (f_peak ** -2.0)
            + alpha3 * (f_peak ** -0.25)
            + atan_lorentz_peak
        ) / eta
        return slope_peak

    @staticmethod

    def __call__(
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        window_type: Optional[str] = None,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        return_phase: bool = False,
        *,
        remnant_provider=None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate frequency-domain polarizations (h_plus, h_cross) using IMRPhenomD.

        Args:
            f: Frequency evaluation tensor in Hz, shape (N_f,) or (B, N_f)
            mass1: Primary mass in solar masses (m1 >= m2)
            mass2: Secondary mass in solar masses
            chi1: Primary aligned spin
            chi2: Secondary aligned spin
            distance: Luminosity distance in Mpc
            inclination: Binary inclination angle in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz for physical reference phasing
            window_type: Optional high-frequency taper ('cosine' or 'planck');
                None retains the analytic amplitude up to the existing Mf=0.2 cutoff
            phasing_convention: Phasing convention ('coalescence' or 'reference_frequency')
            reference_frame: Alias for phasing_convention

        Returns:
            (h_plus, h_cross) complex tensors with shape (B, N_f) or (N_f,)
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)
        chi_s, chi_a, chi_pn, chi_hat = spin_combinations(m1, m2, s1, s2)

        # Evaluate Table V phenomenological coefficients using unscaled reduced spin chi_pn
        # (Khan et al. 2016 arXiv:1508.07253, Eq. 24 and Tab. V)
        rho1 = IMRPhenomD.eval_table_v("rho1", eta, chi_pn)
        rho2 = IMRPhenomD.eval_table_v("rho2", eta, chi_pn)
        rho3 = IMRPhenomD.eval_table_v("rho3", eta, chi_pn)
        v2 = IMRPhenomD.eval_table_v("v2", eta, chi_pn)
        gamma1 = IMRPhenomD.eval_table_v("gamma1", eta, chi_pn)
        gamma2 = IMRPhenomD.eval_table_v("gamma2", eta, chi_pn)
        gamma3 = IMRPhenomD.eval_table_v("gamma3", eta, chi_pn)
        sigma1 = IMRPhenomD.eval_table_v("sigma1", eta, chi_pn)
        sigma2 = IMRPhenomD.eval_table_v("sigma2", eta, chi_pn)
        sigma3 = IMRPhenomD.eval_table_v("sigma3", eta, chi_pn)
        sigma4 = IMRPhenomD.eval_table_v("sigma4", eta, chi_pn)
        beta2 = IMRPhenomD.eval_table_v("beta2", eta, chi_pn)
        beta3 = IMRPhenomD.eval_table_v("beta3", eta, chi_pn)
        alpha1_raw = IMRPhenomD.eval_table_v("alpha1", eta, chi_pn)
        alpha2 = IMRPhenomD.eval_table_v("alpha2", eta, chi_pn)
        alpha3 = IMRPhenomD.eval_table_v("alpha3", eta, chi_pn)
        alpha4 = IMRPhenomD.eval_table_v("alpha4", eta, chi_pn)
        alpha5 = IMRPhenomD.eval_table_v("alpha5", eta, chi_pn)

        # Remnant black hole properties & QNM frequencies
        m_final, a_final, f_rd, f_damp = (IMRPhenomD.remnant_properties if remnant_provider is None else remnant_provider)(m1, m2, s1, s2)

        # Amplitude peak frequency (Khan et al. 2016, arXiv:1508.07253)
        f_peak = IMRPhenomD.amplitude_peak_frequency(f_rd, f_damp, gamma2, gamma3)

        # Transition boundaries
        f1_amp = 0.014
        f3_amp = f_peak
        f1_ph = 0.018
        f2_ph = 0.5 * f_rd

        # PN Amplitude expansion coefficients
        a_pn = IMRPhenomD.pn_amplitude_coefficients(eta, delta, chi_s, chi_a)

        # Collocation values at f1_amp and f3_amp
        def eval_scaled_amp_ins(mf_val: Union[float, torch.Tensor]) -> torch.Tensor:
            mf = torch.as_tensor(mf_val, dtype=dtype, device=device)
            v = (PI * mf) ** (1.0 / 3.0)
            v2 = v * v
            v3 = v2 * v
            v4 = v2 * v2
            v5 = v4 * v
            v6 = v3 * v3
            pn_part = (
                a_pn[0]
                + a_pn[1] * v
                + a_pn[2] * v2
                + a_pn[3] * v3
                + a_pn[4] * v4
                + a_pn[5] * v5
                + a_pn[6] * v6
            )
            # rho terms: rho_i * (Mf)^((6+i)/3)
            mf_73 = mf ** (7.0 / 3.0)
            mf_83 = mf ** (8.0 / 3.0)
            mf_3 = mf ** 3.0
            phen_part = rho1 * mf_73 + rho2 * mf_83 + rho3 * mf_3
            return pn_part + phen_part

        def eval_scaled_amp_ins_deriv(mf_val: float) -> torch.Tensor:
            mf = torch.as_tensor(mf_val, dtype=dtype, device=device)
            dpn = (
                (2.0 / 3.0) * a_pn[2] * (PI ** (2.0 / 3.0)) * (mf ** (-1.0 / 3.0))
                + a_pn[3] * PI
                + (4.0 / 3.0) * a_pn[4] * (PI ** (4.0 / 3.0)) * (mf ** (1.0 / 3.0))
                + (5.0 / 3.0) * a_pn[5] * (PI ** (5.0 / 3.0)) * (mf ** (2.0 / 3.0))
                + 2.0 * a_pn[6] * (PI * PI) * mf
            )
            dphen = (
                (7.0 / 3.0) * rho1 * (mf ** (4.0 / 3.0))
                + (8.0 / 3.0) * rho2 * (mf ** (5.0 / 3.0))
                + 3.0 * rho3 * (mf * mf)
            )
            return dpn + dphen

        def eval_scaled_amp_mr(mf: torch.Tensor) -> torch.Tensor:
            num = gamma1 * (gamma3 * f_damp)
            den = (mf - f_rd) ** 2 + (gamma3 * f_damp) ** 2
            exp_arg = -(gamma2 * (mf - f_rd)) / (gamma3 * f_damp)
            return (num / den) * torch.exp(exp_arg)

        v1_col = eval_scaled_amp_ins(f1_amp)
        d1_col = eval_scaled_amp_ins_deriv(f1_amp)
        v3_col = eval_scaled_amp_mr(f3_amp)
        # Differentiate Eq. (19): the no-real-root continuation of Eq. (20)
        # is not stationary, so imposing zero here would break C1 continuity.
        width = gamma3 * f_damp
        offset = f3_amp - f_rd
        d3_col = v3_col * (-2.0 * offset / (offset.square() + width.square()) - gamma2 / width)

        # Solve intermediate amplitude collocation system
        d0, d1_c, d2, d3_c, d4 = IMRPhenomD.intermediate_amplitude_coeffs(
            f1=f1_amp,
            f3=f3_amp,
            v1=v1_col,
            v2=v2,
            v3=v3_col,
            d1=d1_col,
            d3=d3_col,
        )

        # -------------------------------------------------------------
        # Phase Matching (C1 continuity at f1_ph and f2_ph)
        # -------------------------------------------------------------
        # Use exact Khan et al. 2016 Paper II Appendix B inspiral calibration
        (
            tf2_0, tf2_2, tf2_3, tf2_4, tf2_5, tf2_5_log,
            tf2_6, tf2_6_log, tf2_7, *_
        ) = TaylorF2.phase_coefficients(m1, m2, s1, s2, phenom_d_calibration=True)

        def eval_tf2_at(mf: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            u = torch.as_tensor(mf, dtype=dtype, device=device)
            v = (PI * u) ** (1.0 / 3.0)
            v2 = v * v
            v3 = v2 * v
            v4 = v2 * v2
            v5 = v4 * v
            v6 = v3 * v3
            v7 = v6 * v

            ln_v = torch.log(v)
            ln_4v = torch.log(4.0 * v)

            def match_dim(t: torch.Tensor) -> torch.Tensor:
                if u.dim() > t.dim():
                    return t.unsqueeze(-1)
                return t

            c0 = match_dim(tf2_0)
            c2 = match_dim(tf2_2)
            c3 = match_dim(tf2_3)
            c4 = match_dim(tf2_4)
            c5 = match_dim(tf2_5)
            c5_log = match_dim(tf2_5_log)
            c6 = match_dim(tf2_6)
            c6_log = match_dim(tf2_6_log)
            c7 = match_dim(tf2_7)
            eta_m = match_dim(eta)

            # Phase
            pn_sum = (
                c0
                + c2 * v2
                + c3 * v3
                + c4 * v4
                + (c5 + c5_log * ln_v) * v5
                + (c6 + c6_log * ln_4v) * v6
                + c7 * v7
            )
            phi_tf2 = (3.0 / (128.0 * eta_m * v5)) * pn_sum

            # Exact analytical derivative d phi_tf2 / d(Mf)
            dv_du = (PI / 3.0) * (PI * u) ** (-2.0 / 3.0)
            d_pn_sum_dv = (
                -5.0 * c0 * (v ** -6)
                - 3.0 * c2 * (v ** -4)
                - 2.0 * c3 * (v ** -3)
                - c4 * (v ** -2)
                + c5_log * (v ** -1)
                + (c6 + c6_log * ln_4v + c6_log)
                + 2.0 * c7 * v
            )
            dphi_tf2 = (3.0 / (128.0 * eta_m)) * d_pn_sum_dv * dv_du
            return phi_tf2, dphi_tf2

        # Inspiral phase and derivative at f1_ph = 0.018
        phi_tf2_f1, dphi_tf2_f1 = eval_tf2_at(f1_ph)
        phi_ins_f1 = phi_tf2_f1 + (1.0 / eta) * (
            sigma1 * f1_ph
            + (3.0 / 4.0) * sigma2 * (f1_ph ** (4.0 / 3.0))
            + (3.0 / 5.0) * sigma3 * (f1_ph ** (5.0 / 3.0))
            + 0.5 * sigma4 * (f1_ph ** 2.0)
        )
        dphi_ins_f1 = dphi_tf2_f1 + (1.0 / eta) * (
            sigma1
            + sigma2 * (f1_ph ** (1.0 / 3.0))
            + sigma3 * (f1_ph ** (2.0 / 3.0))
            + sigma4 * f1_ph
        )

        # Region IIa intermediate constants (beta_0, beta_1) from C1 match at f1_ph
        beta1 = eta * dphi_ins_f1 - beta2 * (f1_ph ** -1.0) - beta3 * (f1_ph ** -4.0)
        beta0 = (
            eta * phi_ins_f1
            - beta1 * f1_ph
            - beta2 * math.log(f1_ph)
            + (beta3 / 3.0) * (f1_ph ** -3.0)
        )

        # Intermediate phase and derivative at f2_ph = 0.5 * f_rd
        phi_int_f2 = (1.0 / eta) * (
            beta0
            + beta1 * f2_ph
            + beta2 * torch.log(f2_ph)
            - (beta3 / 3.0) * (f2_ph ** -3.0)
        )
        dphi_int_f2 = (1.0 / eta) * (
            beta1
            + beta2 * (f2_ph ** -1.0)
            + beta3 * (f2_ph ** -4.0)
        )

        # Region IIb merger-ringdown constants (alpha_0, alpha_1) from C1 match at f2_ph
        atan_arg_f2 = (f2_ph - alpha5 * f_rd) / f_damp
        d_atan_f2 = (alpha4 / f_damp) / (1.0 + atan_arg_f2 * atan_arg_f2)

        alpha1 = (
            eta * dphi_int_f2
            - alpha2 * (f2_ph ** -2.0)
            - alpha3 * (f2_ph ** -0.25)
            - d_atan_f2
        )
        alpha0 = (
            eta * phi_int_f2
            - alpha1 * f2_ph
            + alpha2 * (f2_ph ** -1.0)
            - (4.0 / 3.0) * alpha3 * (f2_ph ** 0.75)
            - alpha4 * torch.atan(atan_arg_f2)
        )

        # -------------------------------------------------------------
        # Evaluate on the frequency grid f
        # -------------------------------------------------------------
        m_sec = total_mass * MSUN_SEC
        is_batched = total_mass.dim() >= 1

        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)  # (1, N_f)
        else:
            f_eval = f

        if is_batched and f_eval.dim() > m_sec.dim():
            m_sec_eval = m_sec.unsqueeze(-1)  # (B, 1)
        else:
            m_sec_eval = m_sec

        f_eval_safe = torch.clamp(f_eval, min=1e-8)
        mf = m_sec_eval * f_eval_safe  # Dimensionless frequency M*f

        def unsqueeze_param(t: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        eta_u = unsqueeze_param(eta)
        f_rd_u = unsqueeze_param(f_rd)
        f_damp_u = unsqueeze_param(f_damp)
        f3_amp_u = unsqueeze_param(f3_amp)
        f2_ph_u = unsqueeze_param(f2_ph)

        rho1_u = unsqueeze_param(rho1)
        rho2_u = unsqueeze_param(rho2)
        rho3_u = unsqueeze_param(rho3)

        d0_u = unsqueeze_param(d0)
        d1_c_u = unsqueeze_param(d1_c)
        d2_u = unsqueeze_param(d2)
        d3_c_u = unsqueeze_param(d3_c)
        d4_u = unsqueeze_param(d4)

        gamma1_u = unsqueeze_param(gamma1)
        gamma2_u = unsqueeze_param(gamma2)
        gamma3_u = unsqueeze_param(gamma3)

        sigma1_u = unsqueeze_param(sigma1)
        sigma2_u = unsqueeze_param(sigma2)
        sigma3_u = unsqueeze_param(sigma3)
        sigma4_u = unsqueeze_param(sigma4)

        beta0_u = unsqueeze_param(beta0)
        beta1_u = unsqueeze_param(beta1)
        beta2_u = unsqueeze_param(beta2)
        beta3_u = unsqueeze_param(beta3)

        alpha0_u = unsqueeze_param(alpha0)
        alpha1_u = unsqueeze_param(alpha1)
        alpha2_u = unsqueeze_param(alpha2)
        alpha3_u = unsqueeze_param(alpha3)
        alpha4_u = unsqueeze_param(alpha4)
        alpha5_u = unsqueeze_param(alpha5)

        # Paper merger-ringdown phase derivative dPsi/d(Mf) evaluated at f_peak using raw fitted alpha1
        atan_lorentz_peak = (alpha4 * f_damp) / (f_damp * f_damp + (f_peak - alpha5 * f_rd) ** 2)
        slope_peak = (
            alpha1_raw
            + alpha2 * (f_peak ** -2.0)
            + alpha3 * (f_peak ** -0.25)
            + atan_lorentz_peak
        ) / eta
        slope_peak_u = unsqueeze_param(slope_peak)

        # 1. Piecewise Scaled Amplitude
        # Region I: Inspiral (Mf <= 0.014)
        v_eval = (PI * mf) ** (1.0 / 3.0)
        v2_eval = v_eval * v_eval
        v3_eval = v2_eval * v_eval
        v4_eval = v2_eval * v2_eval
        v5_eval = v4_eval * v_eval
        v6_eval = v3_eval * v3_eval

        pn_amp = (
            unsqueeze_param(a_pn[0])
            + unsqueeze_param(a_pn[1]) * v_eval
            + unsqueeze_param(a_pn[2]) * v2_eval
            + unsqueeze_param(a_pn[3]) * v3_eval
            + unsqueeze_param(a_pn[4]) * v4_eval
            + unsqueeze_param(a_pn[5]) * v5_eval
            + unsqueeze_param(a_pn[6]) * v6_eval
        )
        scaled_amp_ins = (
            pn_amp
            + rho1_u * (mf ** (7.0 / 3.0))
            + rho2_u * (mf ** (8.0 / 3.0))
            + rho3_u * (mf ** 3.0)
        )

        # Region IIa: Intermediate (0.014 < Mf <= f_peak)
        scaled_amp_int = (
            d0_u
            + d1_c_u * mf
            + d2_u * (mf * mf)
            + d3_c_u * (mf**3)
            + d4_u * (mf**4)
        )

        # Region IIb: Merger-Ringdown (Mf > f_peak)
        num_mr = gamma1_u * (gamma3_u * f_damp_u)
        den_mr = (mf - f_rd_u) ** 2 + (gamma3_u * f_damp_u) ** 2
        exp_arg_mr = -(gamma2_u * (mf - f_rd_u)) / (gamma3_u * f_damp_u)
        scaled_amp_mr = (num_mr / den_mr) * torch.exp(exp_arg_mr)

        scaled_amp = torch.where(
            mf <= f1_amp,
            scaled_amp_ins,
            torch.where(
                mf <= f3_amp_u,
                scaled_amp_int,
                scaled_amp_mr,
            ),
        )

        # Physical leading Newtonian amplitude A0(f)
        d_sec = (unsqueeze_param(dist) * MPC_SI) / C_SI
        prefactor = torch.sqrt(5.0 * eta_u / 24.0) / (PI ** (2.0 / 3.0) * d_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (f_eval_safe ** (-7.0 / 6.0))
        full_amp = a0_phys * scaled_amp

        # Khan Eqs. (19), (36) give an untapered merger-ringdown amplitude.
        # Extra high-frequency tapers are opt-in. The Mf=0.2 default cutoff
        # below is an inherited implementation boundary, not a fit in Eq. (19).
        if window_type is not None and window_type.lower() in ("planck", "cosine"):
            mf_cut = f_rd_u + 5.0 * f_damp_u
            mf_start = mf_cut - 2.0 * f_damp_u
            f_cut_phys = mf_cut / m_sec_eval
            f_start_phys = mf_start / m_sec_eval

            x_win = torch.clamp((f_eval - f_start_phys) / (f_cut_phys - f_start_phys + 1e-30), 0.0, 1.0)
            if window_type.lower() == "planck":
                z = 1.0 - x_win
                z_safe = torch.clamp(z, 1e-7, 1.0 - 1e-7)
                sigma_win = (1.0 / z_safe) + (1.0 / (z_safe - 1.0))
                w_cutoff = 1.0 / (torch.exp(torch.clamp(sigma_win, -50.0, 50.0)) + 1.0)
            else:
                w_cutoff = 0.5 * (1.0 + torch.cos(PI * x_win))

            w_cutoff = torch.where(f_eval <= f_start_phys, torch.ones_like(w_cutoff), w_cutoff)
            w_cutoff = torch.where(f_eval >= f_cut_phys, torch.zeros_like(w_cutoff), w_cutoff)

            full_amp = full_amp * w_cutoff
            full_amp = torch.where(f_eval < f_cut_phys, full_amp, torch.zeros_like(full_amp))
        else:
            mf_cut = 0.2 * torch.ones_like(f_rd_u)
            f_cut_phys = mf_cut / m_sec_eval
            full_amp = torch.where(f_eval <= f_cut_phys, full_amp, torch.zeros_like(full_amp))

        # 2. Piecewise Phase & Analytical Derivative
        # Region I: Inspiral (Mf <= 0.018)
        phi_tf2_grid, dphi_tf2_grid = eval_tf2_at(mf)
        phi_ins_grid = phi_tf2_grid + (1.0 / eta_u) * (
            sigma1_u * mf
            + (3.0 / 4.0) * sigma2_u * (mf ** (4.0 / 3.0))
            + (3.0 / 5.0) * sigma3_u * (mf ** (5.0 / 3.0))
            + 0.5 * sigma4_u * (mf ** 2.0)
        )
        dphi_ins_grid = dphi_tf2_grid + (1.0 / eta_u) * (
            sigma1_u
            + sigma2_u * (mf ** (1.0 / 3.0))
            + sigma3_u * (mf ** (2.0 / 3.0))
            + sigma4_u * mf
        )

        # Region IIa: Intermediate (0.018 < Mf <= 0.5 * f_rd)
        phi_int_grid = (1.0 / eta_u) * (
            beta0_u
            + beta1_u * mf
            + beta2_u * torch.log(mf)
            - (beta3_u / 3.0) * (mf ** -3.0)
        )
        dphi_int_grid = (1.0 / eta_u) * (
            beta1_u
            + beta2_u * (mf ** -1.0)
            + beta3_u * (mf ** -4.0)
        )

        # Region IIb: Merger-Ringdown (Mf > 0.5 * f_rd)
        atan_arg_grid = (mf - alpha5_u * f_rd_u) / f_damp_u
        phi_mr_grid = (1.0 / eta_u) * (
            alpha0_u
            + alpha1_u * mf
            - alpha2_u * (mf ** -1.0)
            + (4.0 / 3.0) * alpha3_u * (mf ** 0.75)
            + alpha4_u * torch.atan(atan_arg_grid)
        )
        d_atan_grid = (alpha4_u / f_damp_u) / (1.0 + atan_arg_grid * atan_arg_grid)
        dphi_mr_grid = (1.0 / eta_u) * (
            alpha1_u
            + alpha2_u * (mf ** -2.0)
            + alpha3_u * (mf ** -0.25)
            + d_atan_grid
        )

        phi_imr = torch.where(
            mf <= f1_ph,
            phi_ins_grid,
            torch.where(
                mf <= f2_ph_u,
                phi_int_grid,
                phi_mr_grid,
            ),
        )

        # Task A: Reference Frequency Phasing Transformation
        # Psi_raw(f) = -pi/4 + phi_imr
        psi_raw = -(PI / 4.0) + phi_imr

        t_c_u = unsqueeze_param(t_c)
        phi_c_u = unsqueeze_param(phi_c)

        conv = reference_frame if reference_frame is not None else (
            phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
        )
        conv = conv.lower()

        if conv in ("reference_frequency", "reffreq", "ref"):
            ref_f = f_ref if f_ref is not None else (f_eval.reshape(-1)[0] if f_eval.numel() > 0 else 20.0)
            fref_t = as_tensor(ref_f, device=device, dtype=dtype)
            if is_batched and f_eval.dim() > fref_t.dim():
                fref_u = fref_t.unsqueeze(-1)
            else:
                fref_u = fref_t

            mf_ref = m_sec_eval * fref_u
            phi_tf2_ref, dphi_tf2_ref = eval_tf2_at(mf_ref)

            phi_ins_ref = phi_tf2_ref + (1.0 / eta_u) * (
                sigma1_u * mf_ref
                + (3.0 / 4.0) * sigma2_u * (mf_ref ** (4.0 / 3.0))
                + (3.0 / 5.0) * sigma3_u * (mf_ref ** (5.0 / 3.0))
                + 0.5 * sigma4_u * (mf_ref ** 2.0)
            )
            dphi_ins_ref = dphi_tf2_ref + (1.0 / eta_u) * (
                sigma1_u
                + sigma2_u * (mf_ref ** (1.0 / 3.0))
                + sigma3_u * (mf_ref ** (2.0 / 3.0))
                + sigma4_u * mf_ref
            )

            phi_int_ref = (1.0 / eta_u) * (
                beta0_u
                + beta1_u * mf_ref
                + beta2_u * torch.log(mf_ref)
                - (beta3_u / 3.0) * (mf_ref ** -3.0)
            )
            dphi_int_ref = (1.0 / eta_u) * (
                beta1_u
                + beta2_u * (mf_ref ** -1.0)
                + beta3_u * (mf_ref ** -4.0)
            )

            atan_arg_ref = (mf_ref - alpha5_u * f_rd_u) / f_damp_u
            phi_mr_ref = (1.0 / eta_u) * (
                alpha0_u
                + alpha1_u * mf_ref
                - alpha2_u * (mf_ref ** -1.0)
                + (4.0 / 3.0) * alpha3_u * (mf_ref ** 0.75)
                + alpha4_u * torch.atan(atan_arg_ref)
            )
            d_atan_ref = (alpha4_u / f_damp_u) / (1.0 + atan_arg_ref * atan_arg_ref)
            dphi_mr_ref = (1.0 / eta_u) * (
                alpha1_u
                + alpha2_u * (mf_ref ** -2.0)
                + alpha3_u * (mf_ref ** -0.25)
                + d_atan_ref
            )

            phi_imr_ref = torch.where(
                mf_ref <= f1_ph,
                phi_ins_ref,
                torch.where(
                    mf_ref <= f2_ph_u,
                    phi_int_ref,
                    phi_mr_ref,
                ),
            )
            dphi_imr_ref = torch.where(
                mf_ref <= f1_ph,
                dphi_ins_ref,
                torch.where(
                    mf_ref <= f2_ph_u,
                    dphi_int_ref,
                    dphi_mr_ref,
                ),
            )

            # Phasing convention: reference_frequency aligns phase to reference frequency
            psi_ref = -(PI / 4.0) + phi_imr_ref

            # Intrinsic global time gauge: subtract slope * (Mf - Mf_ref) where slope is the
            # paper merger-ringdown phase derivative dPsi/d(Mf) evaluated at f_peak using raw fitted alpha1.
            total_phase = (
                psi_raw
                - psi_ref
                - slope_peak_u * (mf - mf_ref)
                + 2.0 * PI * f_eval * t_c_u
                - 2.0 * phi_c_u
            )
        else:
            total_phase = (
                psi_raw
                - slope_peak_u * mf
                + 2.0 * PI * f_eval * t_c_u
                - 2.0 * phi_c_u
            )

        # Inclination projections
        cos_inc = torch.cos(unsqueeze_param(inc))
        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        # Complex polarizations h_plus and h_cross
        exp_minus_i_phase = torch.complex(torch.cos(total_phase), -torch.sin(total_phase))

        h_plus = (plus_factor * full_amp).to(exp_minus_i_phase.dtype) * exp_minus_i_phase
        h_cross = torch.complex(
            -cross_factor * full_amp * torch.sin(total_phase),
            -cross_factor * full_amp * torch.cos(total_phase),
        )

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus = torch.where(f_eval > 0.0, h_plus, torch.zeros_like(h_plus))
        h_cross = torch.where(f_eval > 0.0, h_cross, torch.zeros_like(h_cross))

        if return_phase:
            return h_plus, h_cross, total_phase

        return h_plus, h_cross
