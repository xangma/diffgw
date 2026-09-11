"""IMRPhenomXPHM Precessing Multipolar Higher-Harmonic Frequency-Domain Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- Khan et al. (2020), Phys. Rev. Lett. 124, 081101 (arXiv:1911.06050)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
)
from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.imrphenom_hm import (
    IMRPhenomXHM,
    _swsh_projection_coefficients,
)


def wigner_small_d_matrix(l: int, beta: torch.Tensor) -> torch.Tensor:
    """Compute exact Wigner small-d matrix d^l_{m, m'}(beta) for l in {2, 3, 4}.

    Indices range from -l to l (matrix shape (..., 2l+1, 2l+1)).
    Formula:
        d^l_{m, m'}(beta) = sum_k (-1)^(k - m + m') * [sqrt((l+m)!(l-m)!(l+m')!(l-m')!) /
                            ((l+m-k)! k! (l-m'-k)! (k-m+m')!)] * (cos beta/2)^(2l+m-m'-2k) * (sin beta/2)^(2k-m+m')
    """
    c = torch.cos(0.5 * beta)
    s = torch.sin(0.5 * beta)
    shape = beta.shape + (2 * l + 1, 2 * l + 1)
    d = torch.zeros(shape, dtype=beta.dtype, device=beta.device)

    fact = math.factorial
    for i, m in enumerate(range(-l, l + 1)):
        for j, mp in enumerate(range(-l, l + 1)):
            k_min = max(0, m - mp)
            k_max = min(l + m, l - mp)
            val = torch.zeros_like(beta)
            for k in range(k_min, k_max + 1):
                sign = (-1) ** (k - m + mp)
                num = math.sqrt(fact(l + m) * fact(l - m) * fact(l + mp) * fact(l - mp))
                den = fact(l + m - k) * fact(k) * fact(l - mp - k) * fact(k - m + mp)
                c_pow = 2 * l + m - mp - 2 * k
                s_pow = 2 * k - m + mp
                coeff = sign * (num / den)
                val = val + coeff * (c ** c_pow) * (s ** s_pow)
            d[..., i, j] = val
    return d


class IMRPhenomXPHM(nn.Module):
    """IMRPhenomXPHM Precessing Multipolar Higher-Harmonic Waveform Approximant.

    Twists up underlying coprecessing multipolar modes from IMRPhenomXHM
    using next-to-next-to-leading order (NNLO) precession angles and the
    full Wigner D-matrix frame rotation.
    """

    SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))

    def __init__(self, modes: Optional[Sequence[Tuple[int, int]]] = None):
        super().__init__()
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        self.xhm_model = IMRPhenomXHM(modes=self.modes)

    def forward(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        spin1x: Union[float, torch.Tensor] = 0.0,
        spin1y: Union[float, torch.Tensor] = 0.0,
        spin1z: Optional[Union[float, torch.Tensor]] = None,
        spin2x: Union[float, torch.Tensor] = 0.0,
        spin2y: Union[float, torch.Tensor] = 0.0,
        spin2z: Optional[Union[float, torch.Tensor]] = None,
        chi1: Optional[Union[float, torch.Tensor]] = None,
        chi2: Optional[Union[float, torch.Tensor]] = None,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        phi0: Union[float, torch.Tensor] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
        precession_method: str = "nnlo",
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")
        """Generate frequency-domain precessing multipolar polarizations (h_plus, h_cross).

        Args:
            f: Frequency tensor in Hz
            mass1, mass2: Component masses in solar masses (mass1 >= mass2)
            spin1x, spin1y, spin1z: Dimensionless 3D spin of primary mass
            spin2x, spin2y, spin2z: Dimensionless 3D spin of secondary mass
            chi1, chi2: Aligned dimensionless spins (used if spin1z/spin2z not given)
            distance: Luminosity distance in Mpc
            inclination: Observer inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz
            phi0: Observer azimuth angle in radians
            modes: Optional sequence of harmonic modes (l, m) to evaluate
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1x = as_tensor(spin1x, device=device, dtype=dtype)
        s1y = as_tensor(spin1y, device=device, dtype=dtype)
        s2x = as_tensor(spin2x, device=device, dtype=dtype)
        s2y = as_tensor(spin2y, device=device, dtype=dtype)

        s1z_val = spin1z if spin1z is not None else (chi1 if chi1 is not None else 0.0)
        s2z_val = spin2z if spin2z is not None else (chi2 if chi2 is not None else 0.0)
        s1z = as_tensor(s1z_val, device=device, dtype=dtype)
        s2z = as_tensor(s2z_val, device=device, dtype=dtype)

        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)
        fref = as_tensor(f_ref, device=device, dtype=dtype) if f_ref is not None else None
        phi_0 = as_tensor(phi0, device=device, dtype=dtype)

        active_modes = tuple(modes) if modes is not None else self.modes

        # Binary parameters
        total_mass = m1 + m2
        m_sec = total_mass * MSUN_SEC
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        m1_norm = 0.5 * (1.0 + delta)
        m2_norm = 0.5 * (1.0 - delta)

        q = m1 / m2
        a1_coeff = 2.0 + 1.5 / q
        a2_coeff = 2.0 + 1.5 * q

        # Effective precession parameter chi_p (Schmidt et al. 2015, Pratten et al. 2021)
        s1_perp = (m1 * m1) * torch.sqrt(torch.clamp(s1x * s1x + s1y * s1y, min=1e-20))
        s2_perp = (m2 * m2) * torch.sqrt(torch.clamp(s2x * s2x + s2y * s2y, min=1e-20))
        s_p = torch.maximum(a1_coeff * s1_perp, a2_coeff * s2_perp)
        chip = s_p / (a1_coeff * m1 * m1)

        # In-plane spin initial orientation angle at f_ref: alpha(f_ref) = phi_s1
        s1_perp_sq = s1x * s1x + s1y * s1y
        phi_s1 = torch.where(
            s1_perp_sq > 1e-12,
            torch.atan2(s1y, s1x),
            torch.atan2(s2y, s2x),
        )

        # Aligned spin in coprecessing frame
        chi_l = m1_norm * s1z + m2_norm * s2z

        _, _, coprec_dict = self.xhm_model(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=torch.zeros_like(inc),
            phic=phi_c,
            tc=t_c,
            f_ref=fref,
            phi0=torch.zeros_like(phi_0),
            modes=active_modes,
            return_unprojected_modes=True,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
        )

        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
        else:
            f_eval = f

        def unsqueeze_param(t: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        m_sec_u = unsqueeze_param(m_sec)
        eta_u = unsqueeze_param(eta)
        delta_u = unsqueeze_param(delta)
        m1n_u = unsqueeze_param(m1_norm)
        m2n_u = unsqueeze_param(m2_norm)
        chip_u = unsqueeze_param(chip)
        chil_u = unsqueeze_param(chi_l)
        s1z_u = unsqueeze_param(s1z)
        s2z_u = unsqueeze_param(s2z)
        a1_u = unsqueeze_param(a1_coeff)
        inc_u = unsqueeze_param(inc)
        fref_eval = fref if fref is not None else torch.tensor(20.0, device=device, dtype=dtype)
        fref_u = unsqueeze_param(fref_eval)
        phi_s1_u = unsqueeze_param(phi_s1)
        phi0_u = unsqueeze_param(phi_0)

        if precession_method.lower() == "ode":
            from torchwave.approximants.precession_ode import SpinPrecessionODE

            alpha, beta, epsilon = SpinPrecessionODE.compute_angles_frequency(
                f=f_eval,
                mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                f_ref=fref_eval,
            )
        else:
            # NNLO precession angle polynomial coefficients
            (
                a_m3, a_m2, a_m1, a_1, a_log,
                e_m3, e_m2, e_m1, e_1, e_log,
            ) = IMRPhenomP.nnlo_angle_coefficients(
                eta=eta_u,
                delta=delta_u,
                m1_norm=m1n_u,
                chip=chip_u,
                chi_l=chil_u,
            )

            # Dimensionless velocity v = (pi * M * f)^(1/3)
            v = (PI * m_sec_u * f_eval) ** (1.0 / 3.0)
            v_ref = (PI * m_sec_u * fref_u) ** (1.0 / 3.0)

            alpha_dyn, epsilon = IMRPhenomP.evaluate_nnlo_angles(
                v=v,
                v_ref=v_ref,
                a_minus3=a_m3, a_minus2=a_m2, a_minus1=a_m1, a_1=a_1, a_log=a_log,
                e_minus3=e_m3, e_minus2=e_m2, e_minus1=e_m1, e_1=e_1, e_log=e_log,
            )
            alpha = alpha_dyn + phi_s1_u

            # Opening angle beta(f)
            ang_l = IMRPhenomP.orbital_angular_momentum(v, eta_u)
            sp_geom = chip_u * a1_u * (m1n_u * m1n_u)
            sl_geom = (m1n_u * m1n_u) * s1z_u + (m2n_u * m2n_u) * s2z_u

            l_plus_sl = ang_l + sl_geom
            cos_beta = torch.clamp(
                l_plus_sl / torch.sqrt(l_plus_sl * l_plus_sl + sp_geom * sp_geom + 1e-24),
                -1.0 + 1e-12,
                1.0 - 1e-12,
            )
            beta = torch.arccos(cos_beta)

        # If in-plane spins are effectively zero (aligned spin limit), beta -> 0, alpha -> 0, epsilon -> 0
        in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
        is_aligned = unsqueeze_param(in_plane_sq) < 1e-12
        beta = torch.where(is_aligned, torch.zeros_like(beta), beta)
        alpha = torch.where(is_aligned, torch.zeros_like(alpha), alpha)
        epsilon = torch.where(is_aligned, torch.zeros_like(epsilon), epsilon)

        h_plus_total = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)
        h_cross_total = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)
        mode_dict = {}

        # Evaluate Wigner small-d matrices for required multipoles
        unique_l = sorted(list(set(l for l, _ in active_modes)))
        d_mat_beta = {l: wigner_small_d_matrix(l, beta) for l in unique_l}
        d_mat_inc = {l: wigner_small_d_matrix(l, inc_u) for l in unique_l}

        norm_mode = math.sqrt(16.0 * PI / 5.0)

        for l, mp in active_modes:
            h_mode = coprec_dict[(l, mp)]

            # Precession scale factor: converts Wigner small-d basis to physical strain normalization
            sign_l = -1.0 if l == 3 else 1.0
            scale_l = (sign_l * math.sqrt((2 * l + 1) / 5.0)) / norm_mode

            d_beta_l = d_mat_beta[l]
            d_inc_l = d_mat_inc[l]

            term_plus = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)
            term_cross = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)

            # Sum over m in [-l, ..., l]
            for m_val in range(-l, l + 1):
                m_idx = m_val + l
                dm_minus = d_beta_l[..., m_idx, -mp + l]
                dm_plus = d_beta_l[..., m_idx, mp + l]

                # y_{l, m}(iota) = d^l_{m, -2}(iota)
                y_lm = d_inc_l[..., m_idx, -2 + l]

                m_phase = float(m_val) * (phi0_u - alpha)
                cos_m = torch.cos(m_phase)
                sin_m = torch.sin(m_phase)

                # a_{m, -m'} and a_{m, m'}^*
                exp_minus_ima = torch.complex(cos_m, sin_m)
                exp_plus_ima = torch.complex(cos_m, -sin_m)

                a_m_minus = exp_minus_ima * dm_minus * y_lm
                a_m_plus_star = exp_plus_ima * dm_plus * y_lm

                term_plus = term_plus + (a_m_minus + ((-1) ** l) * a_m_plus_star)
                term_cross = term_cross + (a_m_minus - ((-1) ** l) * a_m_plus_star)

            # Phase shift exp(-i * mp * epsilon)
            mp_eps = float(mp) * epsilon
            exp_minus_ime = torch.complex(torch.cos(-mp_eps), torch.sin(-mp_eps))

            # Multiply by scale and coprecessing mode
            hp_lm = scale_l * exp_minus_ime * h_mode * term_plus
            hc_lm = -1j * scale_l * exp_minus_ime * h_mode * term_cross

            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm

            if return_modes:
                mode_dict[(l, mp)] = (hp_lm, hc_lm)

        if return_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
