"""IMRPhenomXP Precessing Dominant-Harmonic Frequency-Domain Approximant.

Mathematical references for the inherited model and audited corrections:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
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
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS, XASParameterFits
from torchwave.approximants.precession_msa import (
    compute_effective_chip,
    compute_msa_final_spin,
    compute_l_3pn,
    evaluate_msa_precession,
    resolve_precession_version,
)
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223
from torchwave.approximants.imrphenom_xphm import wigner_small_d_matrix


class IMRPhenomXP(nn.Module):
    """IMRPhenomXP Precessing Dominant-Harmonic (l=2, |m|=2) Waveform Approximant.

    Twists up underlying coprecessing dominant (2, 2) mode from IMRPhenomXAS
    using NNLO angles or the published two-spin MSA approximation
    with Wigner small-d matrix frame rotations and polarization basis alignment.
    """

    def __init__(self):
        super().__init__()
        self.xas_model = IMRPhenomXAS()

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
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
        precession_method: str = "msa",
        prec_version: Optional[int] = None,
        convention: int = 1,
        final_spin_mod: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate precessing dominant-mode polarizations.

        With ``prec_version`` omitted, the default ``precession_method='msa'``
        selects the clean MSA v223 route; ``precession_method='nnlo'`` selects
        v102. An explicit supported ``prec_version`` takes precedence.
        ``convention`` defaults to 1, and the MSA v223 route defaults to
        ``final_spin_mod=3``. Explicit NNLO choices, including
        ``final_spin_mod=0``, retain their existing behavior.
        MSA version labels share the published analytical prescription;
        undocumented implementation-specific differences are not reproduced.
        ``phi0`` is an additional source-frame observer azimuth. The clean
        helper has no separate observer-azimuth argument, so nonzero ``phi0``
        is evaluated through the established XP rotation path, where it is
        applied to the source-frame geometry and aligned carrier.
        A batch mixing zero and nonzero ``phi0`` rows is rejected on the clean
        route; evaluate those rows separately until a per-row geometry adapter
        is available.
        NNLO rotations with L+S_parallel<=0 are unsupported and use the
        established fallback row by row. Regular exact aligned and zero-spin
        rows remain on the clean v223 chart in either convention.

        ``final_spin_mod=0`` explicitly selects the NNLO remnant estimate;
        ``final_spin_mod=3`` selects the published MSA averaged remnant
        (Eq. 42), independently of the Euler-angle prescription. ``None``
        retains the existing behavior for other versions: NNLO
        remnant for versions 101--104 and the aligned carrier remnant
        otherwise. For supported v223, omission selects MSA remnant mode 3.
        """
        p_ver = resolve_precession_version(precession_method, prec_version)
        force_legacy = bool(kwargs.pop("_force_legacy", False))
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
        # The audited public default for the supported clean MSA route is the
        # averaged v223 remnant. Keep explicit zero and all other version
        # paths unchanged; only an omitted remnant choice is resolved here.
        if final_spin_mod is None and p_ver == 223:
            final_spin_mod = 3
        freqs = f if f is not None else sample_frequencies
        if freqs is None:
            raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        device = freqs.device if isinstance(freqs, torch.Tensor) else None
        dtype = freqs.dtype if isinstance(freqs, torch.Tensor) and freqs.is_floating_point() else torch.float64
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
        fref_eval = fref if fref is not None else torch.tensor(20.0, device=device, dtype=dtype)

        (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
         dist, inc, phi_c, t_c, phi_0, fref_eval) = torch.broadcast_tensors(
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
            dist, inc, phi_c, t_c, phi_0, fref_eval,
        )
        if fref is not None:
            fref = fref_eval

        # The clean native wrapper is the production MSA implementation for
        # the v223/MSA-remnant tuple when the observer azimuth is zero. The
        # helper has no separate observer-azimuth argument, so nonzero phi0
        # stays on the established XP rotation path rather than being dropped.
        exact_aligned = (
            (s1x == 0) & (s1y == 0) & (s2x == 0) & (s2y == 0)
        )
        has_nonzero_phi0 = bool(torch.any(phi_0 != 0))
        has_zero_phi0 = bool(torch.any(phi_0 == 0))
        if (p_ver == 223 and convention in (0, 1) and final_spin_mod == 3
                and has_nonzero_phi0 and has_zero_phi0):
            raise ValueError(
                "Mixed zero/nonzero phi0 rows are unsupported for clean MSA "
                "v223; evaluate rows separately."
            )
        clean_phi0 = not has_nonzero_phi0
        clean_candidate = (
            p_ver == 223 and convention in (0, 1) and final_spin_mod == 3
            and clean_phi0 and not force_legacy
        )
        guard_unsupported = torch.zeros_like(exact_aligned, dtype=torch.bool)
        if clean_candidate:
            # The clean MSA helper deliberately surfaces the singular
            # anti-aligned-J chart as a nonfinite alpha.  Keep the public
            # evaluator's documented L+S guard, including tiny nonzero
            # in-plane spins. Exact-zero aligned rows are regular for the
            # clean chart and are intentionally not routed away merely
            # because their in-plane components vanish.
            with torch.no_grad():
                guard_m1 = torch.where(m1 >= m2, m1, m2)
                guard_m2 = torch.where(m1 >= m2, m2, m1)
                guard_s1z = torch.where(m1 >= m2, s1z, s2z)
                guard_s2z = torch.where(m1 >= m2, s2z, s1z)
                guard_total = guard_m1 + guard_m2
                guard_n1 = guard_m1 / guard_total
                guard_n2 = guard_m2 / guard_total
                guard_eta = guard_n1 * guard_n2
                guard_delta = guard_n1 - guard_n2
                guard_s_ell = (
                    guard_n1.square() * guard_s1z
                    + guard_n2.square() * guard_s2z
                )
                guard_sigma_ell = guard_n2 * guard_s2z - guard_n1 * guard_s1z
                guard_v0 = (
                    PI * guard_total * MSUN_SEC * fref_eval
                ).pow(1.0 / 3.0)
                guard_l0 = compute_l_3pn(
                    guard_v0, guard_eta, guard_delta,
                    guard_s_ell, guard_sigma_ell,
                )
                guard_unsupported = guard_l0 + guard_s_ell <= 0
            # Ordinary exact aligned/zero-spin rows are nonsingular for the
            # clean v223 chart and must use the same helper as their
            # neighboring rows.  Only the physical L+S guard selects the
            # established fallback; keeping this mask row-local avoids a
            # singular row changing a valid neighbor's model.
            legacy_rows = guard_unsupported
        else:
            legacy_rows = exact_aligned
        if clean_candidate and legacy_rows.dim() > 0 and bool(torch.any(legacy_rows)):
            # Dispatch only the unsupported L+S rows independently so a
            # singular row cannot force a valid neighboring row onto the
            # legacy path. Regular exact aligned rows use the helper below.
            source_shape = tuple(m1.shape)
            clean_freqs = as_tensor(freqs, device=device, dtype=dtype)
            flat = tuple(value.reshape(-1) for value in (
                m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
                dist, inc, phi_c, t_c, phi_0, fref_eval, legacy_rows,
            ))
            row_outputs = []
            for index in range(m1.numel()):
                row_fref = None if f_ref is None else flat[13][index]
                row_hp, row_hc = self.forward(
                    f=freqs,
                    mass1=flat[0][index], mass2=flat[1][index],
                    spin1x=flat[2][index], spin1y=flat[3][index],
                    spin1z=flat[4][index], spin2x=flat[5][index],
                    spin2y=flat[6][index], spin2z=flat[7][index],
                    distance=flat[8][index], inclination=flat[9][index],
                    phic=flat[10][index], tc=flat[11][index],
                    f_ref=row_fref, phi0=flat[12][index],
                    phasing_convention=phasing_convention,
                    reference_frame=reference_frame,
                    precession_method=precession_method, prec_version=p_ver,
                    convention=convention, final_spin_mod=final_spin_mod,
                    _force_legacy=bool(flat[14][index]),
                )
                row_outputs.append((
                    row_hp.reshape(tuple(clean_freqs.shape)),
                    row_hc.reshape(tuple(clean_freqs.shape)),
                ))
            return (
                torch.stack([item[0] for item in row_outputs]).reshape(
                    source_shape + tuple(clean_freqs.shape)
                ),
                torch.stack([item[1] for item in row_outputs]).reshape(
                    source_shape + tuple(clean_freqs.shape)
                ),
            )
        if clean_candidate and not bool(torch.any(legacy_rows)):
            def opaque_xas_carrier(**carrier_kwargs):
                carrier_kwargs["phasing_convention"] = phasing_convention
                carrier_kwargs["reference_frame"] = reference_frame
                return self.xas_model(**carrier_kwargs)

            clean_freqs = as_tensor(freqs, device=device, dtype=dtype)
            clean_hp, clean_hc = waveform_v223(
                clean_freqs,
                m1,
                m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                distance=dist,
                inclination=inc,
                phic=phi_c,
                tc=t_c,
                f_ref=fref_eval,
                convention=convention,
                final_spin_mod=3,
                xas_model=opaque_xas_carrier,
            )
            # The clean helper flattens its frequency input internally.  The
            # public XP adapter retains the caller's frequency shape, with
            # broadcast source axes preceding it (including scalar grids).
            target_shape = tuple(m1.shape) + tuple(clean_freqs.shape)
            return clean_hp.reshape(target_shape), clean_hc.reshape(target_shape)

        # Canonical body ordering: primary mass m1 >= m2
        swap_mask = m1 < m2
        if torch.any(swap_mask):
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

        # Check aligned-spin limit
        in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
        is_aligned = in_plane_sq == 0

        # Fast path: if purely aligned scalar, directly return IMRPhenomXAS
        if convention == 1 and is_aligned.dim() == 0 and bool(is_aligned):
            return self.xas_model(
                f=freqs,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phi_c=phi_c-phi_0,
                t_c=t_c,
                f_ref=fref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )

        total_mass = m1 + m2
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
        elif final_spin_mod == 0 or p_ver in (101, 102, 103, 104):
            chip = compute_effective_chip(m1, m2, s1x, s1y, s2x, s2y)
            final_spin_override = XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chip)
        is_batched = total_mass.dim() >= 1

        f_eval = freqs if (not is_batched or freqs.dim() > 1) else freqs.unsqueeze(0)

        def unsqueeze_param(t: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        f_angles = torch.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval))
        v = (PI * unsqueeze_param(m_sec) * f_angles) ** (1.0 / 3.0)
        v0 = (PI * unsqueeze_param(m_sec) * unsqueeze_param(fref_eval)) ** (1.0 / 3.0)

        # Evaluate precession angles
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

        # Table IV supplies the carrier's independent orbital-phase argument.
        hp_carrier, _ = self.xas_model(
            f=f_eval,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=torch.zeros_like(inc),
            phi_c=carrier_phase.squeeze(-1) if is_batched else carrier_phase,
            t_c=t_c,
            f_ref=fref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            _final_spin_override=final_spin_override,
        )

        # Twist-up for (l=2, mp=2)
        scale_2 = 1.0
        d_beta = wigner_small_d_matrix(2, beta)
        d_jn = wigner_small_d_matrix(2, unsqueeze_param(theta_jn))

        term_plus = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)
        term_cross = torch.zeros(f_eval.shape, dtype=complex_dtype, device=device)

        for m_val in range(-2, 3):
            m_idx = m_val + 2
            dm_minus = d_beta[..., m_idx, -2 + 2]
            dm_plus = d_beta[..., m_idx, 2 + 2]
            y_lm = d_jn[..., m_idx, 2 + 2]

            m_phase = float(m_val) * (-alpha)
            cos_m = torch.cos(m_phase)
            sin_m = torch.sin(m_phase)

            exp_minus_ima = torch.complex(cos_m, sin_m)
            exp_plus_ima = torch.complex(cos_m, -sin_m)

            a_m_minus = exp_minus_ima * dm_minus * y_lm
            a_m_plus_star = exp_plus_ima * dm_plus * y_lm

            term_plus = term_plus + (a_m_minus + a_m_plus_star)
            term_cross = term_cross + (a_m_minus - a_m_plus_star)

        # Coprecessing angle factor: exp(2i * gamma) = exp(-2i * epsilon)
        two_gamma = 2.0 * gamma
        exp_two_gamma = torch.complex(torch.cos(two_gamma), torch.sin(two_gamma))

        hp_raw = scale_2 * exp_two_gamma * hp_carrier * term_plus
        hc_raw = 1j * scale_2 * exp_two_gamma * hp_carrier * term_cross

        # C22 / D6--D7: rotate to the observer basis. The minus is the m=2
        # source-azimuth conversion exp(i*m*pi/2), as in XPHM.
        two_zeta = 2.0 * unsqueeze_param(zeta)
        cos_2z = torch.cos(two_zeta)
        sin_2z = torch.sin(two_zeta)

        hp_precess = -(cos_2z * hp_raw + sin_2z * hc_raw)
        hc_precess = -(-sin_2z * hp_raw + cos_2z * hc_raw)

        # If aligned-spin limit, evaluate exact non-precessing waveform
        if convention == 1 and is_aligned.any():
            hp_al, hc_al = self.xas_model(
                f=f_eval,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phi_c=phi_c-phi_0,
                t_c=t_c,
                f_ref=fref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )
            is_al_u = unsqueeze_param(is_aligned)
            hp_final = torch.where(is_al_u, hp_al, hp_precess)
            hc_final = torch.where(is_al_u, hc_al, hc_precess)
        else:
            hp_final = hp_precess
            hc_final = hc_precess

        return hp_final, hc_final

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
