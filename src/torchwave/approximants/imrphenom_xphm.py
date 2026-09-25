"""IMRPhenomXPHM Precessing Multipolar Higher-Harmonic Frequency-Domain Approximant.

Mathematical references for the inherited model and audited corrections:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from diffgw_core.xphm_options import resolve_xphm_options
from torchwave.constants import (
    MSUN_SEC,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
)
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.precession_msa import (
    compute_effective_chip,
    compute_msa_final_spin,
    evaluate_msa_precession,
)


def wigner_small_d_matrix(l: int, beta: torch.Tensor) -> torch.Tensor:
    """Compute exact Wigner small-d matrix d^l_{m, m'}(beta) for l in {2, 3, 4}.

    Indices range from -l to l (matrix shape (..., 2l+1, 2l+1)).
    Formula:
        d^l_{m, m'}(beta) = sum_k (-1)^k * [sqrt((l+m)!(l-m)!(l+m')!(l-m')!) /
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
                sign = (-1) ** k
                num = math.sqrt(fact(l + m) * fact(l - m) * fact(l + mp) * fact(l - mp))
                den = fact(l + m - k) * fact(k) * fact(l - mp - k) * fact(k - m + mp)
                c_pow = 2 * l + m - mp - 2 * k
                s_pow = 2 * k - m + mp
                coeff = sign * (num / den)
                val = val + coeff * (c ** c_pow) * (s ** s_pow)
            d[..., i, j] = val
    return d


def twist_up_mode(
    l: int,
    mp: int,
    h_coprec: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    theta_jn: torch.Tensor,
    zeta: torch.Tensor,
    d_jn_l: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Twist one XHM carrier mode into positive-frequency polarizations.

    This is the tensor form of Pratten et al. (2021), Eqs. (110)--(113),
    plus the Appendix-D observer-basis rotation.  The ``i**mp`` and scale
    factors are solely the adapter from DiffGW's XHM mode convention.
    """
    if l not in (2, 3, 4) or mp <= 0 or mp > l:
        raise ValueError("XPHM rotation supports l in {2,3,4} and 0 < mp <= l.")
    h_mode = (1j ** mp) * h_coprec
    sign_l = 1.0 if (l % 2 == 0) else -1.0
    scale_l = sign_l * math.sqrt((2 * l + 1) / (16.0 * PI))
    d_beta_l = wigner_small_d_matrix(l, beta)
    if d_jn_l is None:
        d_jn_l = wigner_small_d_matrix(l, theta_jn)

    term_plus = torch.zeros_like(h_mode)
    term_cross = torch.zeros_like(h_mode)
    for m_val in range(-l, l + 1):
        m_idx = m_val + l
        dm_minus = d_beta_l[..., m_idx, -mp + l]
        dm_plus = d_beta_l[..., m_idx, mp + l]
        y_lm = d_jn_l[..., m_idx, 2 + l]
        exp_minus_ima = torch.exp(-1j * float(m_val) * alpha)
        exp_plus_ima = torch.exp(1j * float(m_val) * alpha)
        a_m_minus = exp_minus_ima * dm_minus * y_lm
        a_m_plus_star = exp_plus_ima * dm_plus * y_lm
        term_plus = term_plus + a_m_minus + sign_l * a_m_plus_star
        term_cross = term_cross + a_m_minus - sign_l * a_m_plus_star

    phase = torch.exp(1j * float(mp) * gamma)
    hp_raw = scale_l * phase * h_mode * term_plus
    hc_raw = 1j * scale_l * phase * h_mode * term_cross
    cos_2z = torch.cos(2.0 * zeta)
    sin_2z = torch.sin(2.0 * zeta)
    return (
        cos_2z * hp_raw + sin_2z * hc_raw,
        -sin_2z * hp_raw + cos_2z * hc_raw,
    )


class IMRPhenomXPHM(nn.Module):
    """IMRPhenomXPHM Precessing Multipolar Higher-Harmonic Waveform Approximant.

    Twists up underlying coprecessing multipolar modes from IMRPhenomXHM
    using NNLO or published two-spin MSA precession
    angles and the full Wigner D-matrix frame rotation.
    """

    SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        calibration: str = "sovereign",
        precession_method: str = "msa",
        prec_version: Optional[int] = None,
        convention: int = 1,
        final_spin_mod: Optional[int] = None,
    ):
        super().__init__()
        self.calibration = calibration.lower()
        self.default_precession_method = precession_method
        self.default_prec_version = prec_version
        self.default_convention = convention
        self.default_final_spin_mod = final_spin_mod
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        self.xhm_model = IMRPhenomXHM(modes=self.modes, calibration=self.calibration)

    def forward(
        self,
        f: Optional[torch.Tensor] = None,
        mass1: Union[float, torch.Tensor] = 30.0,
        mass2: Union[float, torch.Tensor] = 30.0,
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
        precession_method: Optional[str] = None,
        prec_version: Optional[int] = None,
        convention: Optional[int] = None,
        final_spin_mod: Optional[int] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]],
    ]:
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
            phi0: Additional source-frame observer azimuth angle in radians
            modes: Optional sequence of harmonic modes (l, m) to evaluate
            precession_method: 'nnlo', 'msa', 'spintaylor', or 'pnr' (default 'msa')
            prec_version: An explicit version takes precedence over method.
                When omitted, 'nnlo' selects 102 and 'msa' selects 223.
                MSA labels 220--224 share the published analytical equations;
                their undocumented implementation differences remain unresolved.
            convention: Euler angle offset convention (0, 1, 5, 6, 7; default 1)

        NNLO rotations with L+S_parallel<=0 are unsupported; exactly aligned
        convention-1 inputs use the nonprecessing carrier directly.

        ``final_spin_mod=0`` explicitly selects the NNLO remnant estimate;
        ``final_spin_mod=3`` selects the published MSA averaged remnant
        (Eq. 42), independently of the Euler-angle prescription. ``None``
        selects the published family default: 0 for NNLO and 3 for MSA.
        """
        if precession_method is None:
            precession_method = self.default_precession_method
        if prec_version is None:
            prec_version = self.default_prec_version
        if convention is None:
            convention = self.default_convention
        if final_spin_mod is None:
            final_spin_mod = self.default_final_spin_mod

        precession_family, p_ver, final_spin_mod = resolve_xphm_options(
            precession_method, prec_version, final_spin_mod
        )
        if isinstance(convention, bool) or convention not in (0, 1, 5, 6, 7):
            raise ValueError(f"Unsupported precession convention: {convention}.")
        freqs = f if f is not None else sample_frequencies
        if freqs is None:
            raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        device = freqs.device if isinstance(freqs, torch.Tensor) else None
        dtype = freqs.dtype if isinstance(freqs, torch.Tensor) and freqs.is_floating_point() else torch.float64
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
        fref_eval = fref if fref is not None else torch.tensor(20.0, device=device, dtype=dtype)

        (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
         dist, inc, phi_c, t_c, phi_0, fref_eval) = torch.broadcast_tensors(
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
            dist, inc, phi_c, t_c, phi_0, fref_eval,
        )
        if fref is not None:
            fref = fref_eval

        # Canonical body ordering: primary mass m1 >= m2
        swap_mask = m1 < m2
        m1_eff = torch.where(swap_mask, m2, m1)
        m2_eff = torch.where(swap_mask, m1, m2)
        s1x_eff = torch.where(swap_mask, s2x, s1x)
        s1y_eff = torch.where(swap_mask, s2y, s1y)
        s1z_eff = torch.where(swap_mask, s2z, s1z)
        s2x_eff = torch.where(swap_mask, s1x, s2x)
        s2y_eff = torch.where(swap_mask, s1y, s2y)
        s2z_eff = torch.where(swap_mask, s1z, s2z)
        m1, m2 = m1_eff, m2_eff
        s1x, s1y, s1z = s1x_eff, s1y_eff, s1z_eff
        s2x, s2y, s2z = s2x_eff, s2y_eff, s2z_eff

        active_modes = tuple(modes) if modes is not None else self.modes

        # In-plane spin check
        in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
        is_aligned = in_plane_sq == 0

        # Fast path: if purely aligned scalar, directly return IMRPhenomXHM
        if not (hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling()):
            if convention == 1 and is_aligned.dim() == 0 and bool(is_aligned):
                return self.xhm_model(
                f=freqs,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phic=phi_c,
                tc=t_c,
                f_ref=fref,
                phi0=phi_0,
                modes=active_modes,
                return_modes=return_modes,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )

        total_mass = m1 + m2
        # The remnant fit is parameterized by the symmetric mass ratio after
        # canonical body ordering and broadcasting above.
        eta = component_masses_to_eta(m1, m2)
        m_sec = total_mass * MSUN_SEC
        final_spin_override = None
        # Table V FinalSpinMod=0 is the NNLO (versions 101--104) remnant model.
        # An explicit remnant prescription is independent of the angle version.
        if final_spin_mod == 3:
            v0_remnant = (PI * m_sec * fref_eval) ** (1.0 / 3.0)
            final_spin_override = compute_msa_final_spin(
                v0_remnant, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
            )
        elif final_spin_mod == 2:
            from torchwave.approximants.precession_spintaylor import compute_chi_tot_perp

            chi_tot_perp = compute_chi_tot_perp(m1, m2, s1x, s1y, s2x, s2y)
            final_spin_override = XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chi_tot_perp)
        elif final_spin_mod == 0:
            chip = compute_effective_chip(m1, m2, s1x, s1y, s2x, s2y)
            final_spin_override = XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chip)
        is_batched = total_mass.dim() >= 1

        f_eval = freqs if (not is_batched or freqs.dim() > 1) else freqs.unsqueeze(0)

        def unsqueeze_param(t: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        # Precession angles evaluation
        if precession_family == "pnr":
            from torchwave.approximants.precession_pnr import evaluate_pnr_precession

            alpha, beta, gamma, theta_jn, zeta, carrier_phase = evaluate_pnr_precession(
                f=torch.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)),
                f_ref=fref_eval,
                mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                inclination=inc,
                phi_ref=phi_c,
                convention=convention,
                observer_azimuth=phi_0,
            )
        elif precession_family == "spintaylor":
            from torchwave.approximants.precession_spintaylor import evaluate_spintaylor_precession

            alpha, beta, gamma, theta_jn, zeta, carrier_phase = evaluate_spintaylor_precession(
                f=torch.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)),
                f_ref=fref_eval,
                mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                inclination=inc,
                phi_ref=phi_c,
                convention=convention,
                observer_azimuth=phi_0,
            )
        else:
            f_angles = torch.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval))
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

        # Coprecessing modes receive precisely the Table IV phiRef argument.
        _, _, coprec_dict = self.xhm_model(
            f=freqs,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=torch.zeros_like(inc),
            phic=carrier_phase.squeeze(-1) if is_batched else carrier_phase,
            tc=t_c,
            f_ref=fref,
            phi0=torch.zeros_like(phi_0),
            modes=active_modes,
            return_unprojected_modes=True,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            _final_spin_override=final_spin_override,
        )

        h_plus_re = torch.zeros_like(f_eval)
        h_plus_im = torch.zeros_like(f_eval)
        h_cross_re = torch.zeros_like(f_eval)
        h_cross_im = torch.zeros_like(f_eval)
        mode_dict = {}

        unique_l = sorted(list(set(l for l, _ in active_modes)))
        # E18/E19: the SPA orbital frequency is 2*pi*f/mp. Reference spin
        # data still refer to the common dominant-harmonic f_ref.
        angles_by_mp = {2: (alpha, beta, gamma)}
        if precession_family == "pnr":
            from torchwave.approximants.precession_pnr import evaluate_pnr_precession

            for mp in {mp for _, mp in active_modes} - {2}:
                f_mp = torch.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)) * (2.0 / float(mp))
                a_mp, b_mp, g_mp, _, _, _ = evaluate_pnr_precession(
                    f=f_mp,
                    f_ref=fref_eval,
                    mass1=m1,
                    mass2=m2,
                    spin1x=s1x,
                    spin1y=s1y,
                    spin1z=s1z,
                    spin2x=s2x,
                    spin2y=s2y,
                    spin2z=s2z,
                    inclination=inc,
                    phi_ref=phi_c,
                    convention=convention,
                    observer_azimuth=phi_0,
                )
                angles_by_mp[mp] = (a_mp, b_mp, g_mp)
        elif precession_family == "spintaylor":
            from torchwave.approximants.precession_spintaylor import evaluate_spintaylor_precession

            for mp in {mp for _, mp in active_modes} - {2}:
                f_mp = torch.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)) * (2.0 / float(mp))
                a_mp, b_mp, g_mp, _, _, _ = evaluate_spintaylor_precession(
                    f=f_mp,
                    f_ref=fref_eval,
                    mass1=m1,
                    mass2=m2,
                    spin1x=s1x,
                    spin1y=s1y,
                    spin1z=s1z,
                    spin2x=s2x,
                    spin2y=s2y,
                    spin2z=s2z,
                    inclination=inc,
                    phi_ref=phi_c,
                    convention=convention,
                    observer_azimuth=phi_0,
                )
                angles_by_mp[mp] = (a_mp, b_mp, g_mp)
        else:
            for mp in {mp for _, mp in active_modes} - {2}:
                a_mp, b_mp, g_mp, _, _ = evaluate_msa_precession(
                    v=v * (2.0 / mp) ** (1.0 / 3.0), v0=v0,
                    m1=unsqueeze_param(m1), m2=unsqueeze_param(m2),
                    s1x=unsqueeze_param(s1x), s1y=unsqueeze_param(s1y), s1z=unsqueeze_param(s1z),
                    s2x=unsqueeze_param(s2x), s2y=unsqueeze_param(s2y), s2z=unsqueeze_param(s2z),
                    iota=unsqueeze_param(inc), phi_ref=unsqueeze_param(phi_c),
                    prec_version=p_ver, convention=convention,
                    observer_azimuth=unsqueeze_param(phi_0),
                )
                angles_by_mp[mp] = (a_mp, b_mp, g_mp)
        d_mat_jn = {l: wigner_small_d_matrix(l, unsqueeze_param(theta_jn)) for l in unique_l}

        for l, mp in active_modes:
            h_coprec = coprec_dict[(l, mp)]
            alpha, beta, gamma = angles_by_mp.get(mp, angles_by_mp[2])
            hp_lm, hc_lm = twist_up_mode(
                l,
                mp,
                h_coprec,
                alpha,
                beta,
                gamma,
                unsqueeze_param(theta_jn),
                unsqueeze_param(zeta),
                d_jn_l=d_mat_jn[l],
            )
            h_plus_re = h_plus_re + hp_lm.real
            h_plus_im = h_plus_im + hp_lm.imag
            h_cross_re = h_cross_re + hc_lm.real
            h_cross_im = h_cross_im + hc_lm.imag

            if return_modes:
                mode_dict[(l, mp)] = (hp_lm, hc_lm)

        # If aligned-spin limit, blend with exact IMRPhenomXHM
        should_blend_aligned = (convention == 1) and (
            (hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling())
            or bool(is_aligned.any())
        )
        if should_blend_aligned:
            out_al = self.xhm_model(
                f=freqs,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phic=phi_c,
                tc=t_c,
                f_ref=fref,
                phi0=phi_0,
                modes=active_modes,
                return_modes=return_modes,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )
            is_al_u = unsqueeze_param(is_aligned)
            if return_modes:
                hp_al, hc_al, dict_al = out_al
                h_plus_re = torch.where(is_al_u, hp_al.real, h_plus_re)
                h_plus_im = torch.where(is_al_u, hp_al.imag, h_plus_im)
                h_cross_re = torch.where(is_al_u, hc_al.real, h_cross_re)
                h_cross_im = torch.where(is_al_u, hc_al.imag, h_cross_im)
                for k in active_modes:
                    hp_m, hc_m = mode_dict[k]
                    hp_m_al, hc_m_al = dict_al[k]
                    mode_dict[k] = (
                        torch.where(is_al_u, hp_m_al, hp_m),
                        torch.where(is_al_u, hc_m_al, hc_m),
                    )
            else:
                hp_al, hc_al = out_al
                h_plus_re = torch.where(is_al_u, hp_al.real, h_plus_re)
                h_plus_im = torch.where(is_al_u, hp_al.imag, h_plus_im)
                h_cross_re = torch.where(is_al_u, hc_al.real, h_cross_re)
                h_cross_im = torch.where(is_al_u, hc_al.imag, h_cross_im)

        h_plus_total = torch.view_as_complex(torch.stack([h_plus_re, h_plus_im], dim=-1))
        h_cross_total = torch.view_as_complex(torch.stack([h_cross_re, h_cross_im], dim=-1))

        if return_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
