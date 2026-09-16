"""EccentricTaylorF2 Frequency-Domain Gravitational Waveform Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Moore, Favata, Arun, Mishra (2016), Phys. Rev. D 93, 124061 (arXiv:1605.00304)
- Yunes et al. (2009), Phys. Rev. D 80, 084018 (arXiv:0909.4263)
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    G_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    kerr_isco_frequency,
)
from torchwave.approximants.taylorf2 import TaylorF2

GAMMA_E = 0.577215664901532860606512090082402431
LN2 = 0.693147180559945309417232121458176568
LN3 = 1.098612288668109691395245236922525704

_UNSET = object()


class TaylorF2_Eccentric(nn.Module):
    """EccentricTaylorF2 Post-Newtonian Stationary Phase Approximation model.

    Implements full 3PN eccentric phase corrections from Moore et al. (2016)
    Eqs. (1438)-(1484) / Eq. (4.17).
    """

    def __init__(self):
        super().__init__()
        self.base_model = TaylorF2()

    @staticmethod
    def eccentric_phase_correction(
        f: Union[float, torch.Tensor],
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        e0: Union[float, torch.Tensor],
        f0: Union[float, torch.Tensor],
        ecc_order: int = 6,
    ) -> torch.Tensor:
        """Compute the eccentric phase correction under SPA up to 3PN order.

        Reference: Moore, Favata, Arun, Mishra (2016), Phys. Rev. D 93, 124061,
        Eqs. (1438)-(1484) [arXiv:1605.00304].

        Delta Psi_ecc(f, f0) = - (2355 / 1462) * (3 / (128 * eta * v^5)) * e0^2 * (v0 / v)^(19/3)
                               * { 1 + [1PN] + [1.5PN] + [2PN] + [2.5PN] + [3PN] }
        """
        # Determine device and dtype from inputs
        for param in [mass1, mass2, e0, f0, f]:
            if isinstance(param, torch.Tensor):
                device = param.device
                dtype = param.dtype if param.is_floating_point() else torch.float64
                break
        else:
            device = None
            dtype = torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        e0_t = as_tensor(e0, device=device, dtype=dtype)
        f0_t = as_tensor(f0, device=device, dtype=dtype)
        f_t = as_tensor(f, device=device, dtype=dtype)

        eta = component_masses_to_eta(m1, m2)
        total_mass = m1 + m2
        m_sec = total_mass * MSUN_SEC

        is_batched = total_mass.dim() >= 1
        if is_batched:
            if f_t.dim() == 2:
                f_eval = f_t
            elif f_t.dim() == 1 and f_t.shape[0] == total_mass.shape[0] and f_t.numel() == total_mass.numel():
                f_eval = f_t.reshape(-1, 1)
            else:
                f_eval = f_t.reshape(1, -1) if f_t.numel() > 1 else f_t.reshape(1, 1)
            m_sec_eval = m_sec.reshape(-1, 1)
            eta_eval = eta.reshape(-1, 1)
            e0_eval = e0_t.reshape(-1, 1)
            f0_eval = f0_t.reshape(-1, 1)
        else:
            f_eval = f_t
            m_sec_eval = m_sec
            eta_eval = eta
            e0_eval = e0_t
            f0_eval = f0_t

        safe_f = torch.clamp(f_eval, min=1e-8)
        safe_f0 = torch.clamp(f0_eval, min=1e-8)

        v = torch.pow(PI * m_sec_eval * safe_f, 1.0 / 3.0)
        v0 = torch.pow(PI * m_sec_eval * safe_f0, 1.0 / 3.0)

        v2 = v * v
        v3 = v2 * v
        v4 = v2 * v2
        v5 = v4 * v
        v6 = v3 * v3

        v02 = v0 * v0
        v03 = v02 * v0
        v04 = v02 * v02
        v05 = v04 * v0
        v06 = v03 * v03

        eta2 = eta_eval * eta_eval
        eta3 = eta2 * eta_eval

        # Leading factor: - (2355 / 1462) * (3 / (128 * eta * v^5)) * e0^2 * (v0 / v)^(19/3)
        pref = (
            -(2355.0 / 1462.0)
            * (3.0 / (128.0 * eta_eval * v5))
            * (e0_eval * e0_eval)
            * torch.pow(v0 / v, 19.0 / 3.0)
        )

        bracket = torch.ones_like(v)

        if ecc_order >= 2:  # 1PN
            c_v2 = (299076223.0 / 81976608.0) + (18766963.0 / 2927736.0) * eta_eval
            c_v02 = (2833.0 / 1008.0) - (197.0 / 36.0) * eta_eval
            bracket = bracket + c_v2 * v2 + c_v02 * v02

        if ecc_order >= 3:  # 1.5PN
            c_v3 = -(2819123.0 / 282600.0) * PI
            c_v03 = (377.0 / 72.0) * PI
            bracket = bracket + c_v3 * v3 + c_v03 * v03

        if ecc_order >= 4:  # 2PN
            c_v4 = (
                (16237683263.0 / 3330429696.0)
                + (24133060753.0 / 971375328.0) * eta_eval
                + (1562608261.0 / 69383952.0) * eta2
            )
            c_v2v02 = (
                (847282939759.0 / 82632420864.0)
                - (718901219.0 / 368894736.0) * eta_eval
                - (3697091711.0 / 105398496.0) * eta2
            )
            c_v04 = (
                -(1193251.0 / 3048192.0)
                - (66317.0 / 9072.0) * eta_eval
                + (18155.0 / 1296.0) * eta2
            )
            bracket = bracket + c_v4 * v4 + c_v2v02 * v2 * v02 + c_v04 * v04

        if ecc_order >= 5:  # 2.5PN
            c_v5 = (
                -(
                    (2831492681.0 / 118395270.0)
                    + (11552066831.0 / 270617760.0) * eta_eval
                )
                * PI
            )
            c_v3v02 = (
                -(7986575459.0 / 284860800.0)
                + (555367231.0 / 10173600.0) * eta_eval
            ) * PI
            c_v2v03 = (
                (112751736071.0 / 5902315776.0)
                + (7075145051.0 / 210796992.0) * eta_eval
            ) * PI
            c_v05 = (
                (764881.0 / 90720.0)
                - (949457.0 / 22680.0) * eta_eval
            ) * PI
            bracket = bracket + c_v5 * v5 + c_v3v02 * v3 * v02 + c_v2v03 * v2 * v03 + c_v05 * v05

        if ecc_order >= 6:  # 3PN
            c_v6 = (
                -(43603153867072577087.0 / 132658535116800000.0)
                + (536803271.0 / 19782000.0) * GAMMA_E
                + (15722503703.0 / 325555200.0) * (PI * PI)
                + (
                    (299172861614477.0 / 689135247360.0)
                    - (15075413.0 / 1446912.0) * (PI * PI)
                )
                * eta_eval
                + (3455209264991.0 / 41019955200.0) * eta2
                + (50612671711.0 / 878999040.0) * eta3
                + (3843505163.0 / 59346000.0) * LN2
                - (1121397129.0 / 17584000.0) * LN3
                + (536803271.0 / 39564000.0) * torch.log(16.0 * v2)
            )

            c_v4v02 = (
                (46001356684079.0 / 3357073133568.0)
                + (253471410141755.0 / 5874877983744.0) * eta_eval
                - (1693852244423.0 / 23313007872.0) * eta2
                - (307833827417.0 / 2497822272.0) * eta3
            )

            c_v3v03 = -(1062809371.0 / 20347200.0) * (PI * PI)

            c_v2v04 = (
                -(356873002170973.0 / 249880440692736.0)
                - (260399751935005.0 / 8924301453312.0) * eta_eval
                + (150484695827.0 / 35413894656.0) * eta2
                + (340714213265.0 / 3794345856.0) * eta3
            )

            c_v06 = (
                (26531900578691.0 / 168991764480.0)
                - (3317.0 / 126.0) * GAMMA_E
                + (122833.0 / 10368.0) * (PI * PI)
                + (
                    (9155185261.0 / 548674560.0)
                    - (3977.0 / 1152.0) * (PI * PI)
                )
                * eta_eval
                - (5732473.0 / 1306368.0) * eta2
                - (3090307.0 / 139968.0) * eta3
                + (87419.0 / 1890.0) * LN2
                - (26001.0 / 560.0) * LN3
                - (3317.0 / 252.0) * torch.log(16.0 * v02)
            )

            bracket = (
                bracket
                + c_v6 * v6
                + c_v4v02 * v4 * v02
                + c_v3v03 * v3 * v03
                + c_v2v04 * v2 * v04
                + c_v06 * v06
            )

        return pref * bracket

    @staticmethod
    def eccentric_amplitude_factor(
        f: Union[float, torch.Tensor],
        e0: Union[float, torch.Tensor],
        f0: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the leading-order Newtonian eccentric amplitude correction.

        Reference: Moore et al. (2016) Eq. (4.18)
        A_ecc(f) = A_circ(f) * [ 1 - (743 / 9216) * e0^2 * (f / f0)^(-19/9) ]
        """
        for param in [e0, f0, f]:
            if isinstance(param, torch.Tensor):
                device = param.device
                dtype = param.dtype if param.is_floating_point() else torch.float64
                break
        else:
            device = None
            dtype = torch.float64

        e0_t = as_tensor(e0, device=device, dtype=dtype)
        f0_t = as_tensor(f0, device=device, dtype=dtype)
        f_t = as_tensor(f, device=device, dtype=dtype)

        is_batched = e0_t.dim() >= 1
        if is_batched:
            f_eval = f_t.reshape(1, -1) if f_t.numel() > 1 else f_t.reshape(1, 1)
            e0_eval = e0_t.reshape(-1, 1)
            f0_eval = f0_t.reshape(-1, 1)
        else:
            f_eval = f_t
            e0_eval = e0_t
            f0_eval = f0_t

        safe_f = torch.clamp(f_eval, min=1e-8)
        safe_f0 = torch.clamp(f0_eval, min=1e-8)

        f_ratio = safe_f / safe_f0
        ratio_term = torch.pow(f_ratio, -19.0 / 9.0)
        e0_sq = e0_eval * e0_eval

        factor = 1.0 - (743.0 / 9216.0) * e0_sq * ratio_term
        return factor

    eccentric_amplitude_correction = eccentric_amplitude_factor

    def forward(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor, object] = _UNSET,
        chi2: Union[float, torch.Tensor, object] = _UNSET,
        e0: Optional[Union[float, torch.Tensor]] = None,
        eccentricity: Optional[Union[float, torch.Tensor]] = None,
        f0_ecc: Optional[Union[float, torch.Tensor]] = None,
        f_lower: Optional[Union[float, torch.Tensor]] = None,
        spin1z: Optional[Union[float, torch.Tensor]] = None,
        spin2z: Optional[Union[float, torch.Tensor]] = None,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        lambda1: Optional[Union[float, torch.Tensor]] = None,
        lambda2: Optional[Union[float, torch.Tensor]] = None,
        f_isco_cutoff: bool = True,
        isco_type: str = "schwarzschild",
        smooth_cutoff: bool = False,
        roll_off_delta: float = 0.1,
        spin_order: int = 7,
        ecc_order: int = 6,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        include_amplitude_correction: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including orbital eccentricity.

        Args:
            f: Frequency tensor in Hz
            mass1: Primary mass in solar masses (mass1 >= mass2)
            mass2: Secondary mass in solar masses
            chi1, chi2: Dimensionless aligned spins
            spin1z, spin2z: Aliases for chi1, chi2
            e0: Reference orbital eccentricity (0 <= e0 < 0.4)
            eccentricity: Alias for e0
            f0_ecc: Reference frequency in Hz at which eccentricity e0 is defined (defaults to f_lower, f_ref, or f[0])
            f_lower: Starting frequency of signal band
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz
            lambda1, lambda2: Dimensionless tidal deformabilities
            f_isco_cutoff: If True, set signal to zero above ISCO
            isco_type: ISCO formulation ("schwarzschild" or "kerr")
            smooth_cutoff: If True, smoothly roll off amplitude to zero at ISCO
            roll_off_delta: Fractional width of smooth roll-off window below ISCO
            spin_order: Highest PN spin order
            ecc_order: Highest PN eccentric order (default 6 for full 3PN)
            phasing_convention: Phasing convention
            reference_frame: Alias for phasing_convention
            include_amplitude_correction: Whether to apply the leading-order amplitude suppression factor (default False for LAL parity)
        """
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs.keys())
            raise TypeError(f"TaylorF2_Eccentric.forward() got unexpected keyword argument(s): {unexpected}")

        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)

        # Support spin1z / spin2z aliases with conflict checking
        if chi1 is not _UNSET and spin1z is not None:
            t_spin1z = as_tensor(spin1z, device=device, dtype=dtype)
            t_chi1 = as_tensor(chi1, device=device, dtype=dtype)
            if not torch.allclose(t_chi1, t_spin1z, atol=1e-12, rtol=1e-8):
                raise ValueError(f"Conflicting spin1 parameters: chi1={chi1} and spin1z={spin1z}")
            s1 = t_spin1z
        elif spin1z is not None:
            s1 = as_tensor(spin1z, device=device, dtype=dtype)
        elif chi1 is not _UNSET:
            s1 = as_tensor(chi1, device=device, dtype=dtype)
        else:
            s1 = as_tensor(0.0, device=device, dtype=dtype)

        if chi2 is not _UNSET and spin2z is not None:
            t_spin2z = as_tensor(spin2z, device=device, dtype=dtype)
            t_chi2 = as_tensor(chi2, device=device, dtype=dtype)
            if not torch.allclose(t_chi2, t_spin2z, atol=1e-12, rtol=1e-8):
                raise ValueError(f"Conflicting spin2 parameters: chi2={chi2} and spin2z={spin2z}")
            s2 = t_spin2z
        elif spin2z is not None:
            s2 = as_tensor(spin2z, device=device, dtype=dtype)
        elif chi2 is not _UNSET:
            s2 = as_tensor(chi2, device=device, dtype=dtype)
        else:
            s2 = as_tensor(0.0, device=device, dtype=dtype)

        # Support both e0 and eccentricity aliases
        if e0 is not None and eccentricity is not None:
            if not torch.allclose(as_tensor(e0, device=device, dtype=dtype), as_tensor(eccentricity, device=device, dtype=dtype)):
                raise ValueError("Conflicting eccentricity parameters: e0 and eccentricity both provided with different values")
            e_val = e0
        elif eccentricity is not None:
            e_val = eccentricity
        elif e0 is not None:
            e_val = e0
        else:
            e_val = 0.0

        e_0 = as_tensor(e_val, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)

        l1 = as_tensor(lambda1, device=device, dtype=dtype) if lambda1 is not None else None
        l2 = as_tensor(lambda2, device=device, dtype=dtype) if lambda2 is not None else None

        # Resolve reference frequency for eccentricity and phase alignment
        if f0_ecc is not None:
            f0_ref = as_tensor(f0_ecc, device=device, dtype=dtype)
        elif f_lower is not None:
            f0_ref = as_tensor(f_lower, device=device, dtype=dtype)
        elif f_ref is not None and (as_tensor(f_ref, device=device, dtype=dtype) > 0.0).all():
            f0_ref = as_tensor(f_ref, device=device, dtype=dtype)
        else:
            pos_f = f[f > 0.0]
            f0_val = pos_f.min() if pos_f.numel() > 0 else 20.0
            f0_ref = as_tensor(f0_val, device=device, dtype=dtype)

        # Baseline quasi-circular phase and amplitude from TaylorF2
        f_ref_circ = f0_ref if f_ref is None or (as_tensor(f_ref, device=device, dtype=dtype) <= 0.0).all() else f_ref

        psi_circ = TaylorF2.evaluate_phase(
            f=f,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            lambda1=l1,
            lambda2=l2,
            tc=t_c,
            phic=phi_c,
            f_ref=f_ref_circ,
            spin_order=spin_order,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
        )

        amp_circ = TaylorF2.evaluate_amplitude(
            f=f,
            mass1=m1,
            mass2=m2,
            distance=dist,
        )

        # Full 3PN eccentric phase corrections
        delta_psi_f = self.eccentric_phase_correction(
            f=f,
            mass1=m1,
            mass2=m2,
            e0=e_0,
            f0=f0_ref,
            ecc_order=ecc_order,
        )

        delta_psi_ref = self.eccentric_phase_correction(
            f=f_ref_circ,
            mass1=m1,
            mass2=m2,
            e0=e_0,
            f0=f0_ref,
            ecc_order=ecc_order,
        )

        # Phase alignment relative to f_ref
        delta_psi_ecc = delta_psi_f - delta_psi_ref

        total_psi = psi_circ + delta_psi_ecc
        total_amp = amp_circ

        if include_amplitude_correction:
            amp_factor = self.eccentric_amplitude_correction(
                f=f,
                e0=e_0,
                f0=f0_ref,
            )
            total_amp = total_amp * amp_factor

        f_eval = f.unsqueeze(0) if (m1.dim() >= 1 and f.dim() == 1) else f

        # Cutoff above ISCO if requested
        if f_isco_cutoff:
            total_m = m1 + m2
            m_sec = total_m * MSUN_SEC
            if m_sec.dim() >= 1 and total_amp.dim() > m_sec.dim():
                m_sec = m_sec.unsqueeze(-1)
                total_m_for_cutoff = total_m.unsqueeze(-1)
                chi_eff = ((m1 * s1 + m2 * s2) / total_m).unsqueeze(-1)
            else:
                total_m_for_cutoff = total_m
                chi_eff = (m1 * s1 + m2 * s2) / total_m

            if isco_type.lower() == "kerr":
                f_isco = kerr_isco_frequency(total_m_for_cutoff, chi_eff)
            else:
                f_isco = 1.0 / (6.0 ** 1.5 * PI * m_sec)

            if not smooth_cutoff:
                mask = (f_eval <= f_isco) & (f_eval > 0.0)
                total_amp = torch.where(mask, total_amp, torch.zeros_like(total_amp))
            else:
                f_start = (1.0 - roll_off_delta) * f_isco
                x = torch.clamp((f_eval - f_start) / (f_isco - f_start + 1e-30), 0.0, 1.0)
                w = 0.5 * (1.0 + torch.cos(PI * x))
                w = torch.where(f_eval <= f_start, torch.ones_like(w), w)
                w = torch.where(f_eval >= f_isco, torch.zeros_like(w), w)
                w = torch.where(f_eval > 0.0, w, torch.zeros_like(w))
                total_amp = total_amp * w

        # Inclination projections:
        # h_plus = 0.5 * (1 + cos^2(iota)) * A(f) * exp(-i * Psi(f))
        # h_cross = -i * cos(iota) * A(f) * exp(-i * Psi(f))
        cos_inc = torch.cos(inc)
        if cos_inc.dim() >= 1 and total_amp.dim() > cos_inc.dim():
            cos_inc = cos_inc.unsqueeze(-1)

        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        exp_minus_i_psi = torch.complex(torch.cos(total_psi), -torch.sin(total_psi))

        h_plus = (plus_factor * total_amp).to(exp_minus_i_psi.dtype) * exp_minus_i_psi
        h_cross = torch.complex(
            -cross_factor * total_amp * torch.sin(total_psi),
            -cross_factor * total_amp * torch.cos(total_psi),
        )

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus = torch.where(f_eval > 0.0, h_plus, torch.zeros_like(h_plus))
        h_cross = torch.where(f_eval > 0.0, h_cross, torch.zeros_like(h_cross))

        return h_plus, h_cross

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
