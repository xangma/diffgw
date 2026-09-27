"""IMRPhenomHM & IMRPhenomXHM Multipolar Higher-Mode Frequency-Domain Approximants.

Strict Clean-Room Implementation based exclusively on published literature:
- London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
- Mishra et al. (2016), Phys. Rev. D 94, 084053 (arXiv:1609.00727)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
- Kidder (2008), Phys. Rev. D 77, 044016 (arXiv:0710.0614)
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
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
    spin_combinations,
    total_spin_for_remnant,
)
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_hm_qnm_n61 import evaluate_n61_qnm


# Kerr Quasinormal Mode (QNM) perturbation table fits (Berti, Cardoso, Will 2006 Table VIII)
# M * omega_lmn = f1 + f2 * (1 - j)^f3
# Q_lmn = q1 + q2 * (1 - j)^q3
# f_RD = (M / M_f) * (omega_lmn / (2 * pi)), f_damp = f_RD / (2 * Q_lmn)
QNM_FITS: Dict[Tuple[int, int], Dict[str, Tuple[float, float, float]]] = {
    (2, 2): {
        "f": (1.5251, -1.1568, 0.1292),
        "q": (0.7000, 1.4187, -0.4990),
    },
    (2, 1): {
        "f": (0.6000, -0.2339, 0.4175),
        "q": (-0.3000, 2.3561, -0.2277),
    },
    (3, 3): {
        "f": (1.8956, -1.3043, 0.1818),
        "q": (0.9000, 2.3430, -0.4810),
    },
    (3, 2): {
        "f": (1.1481, -0.5552, 0.3002),
        "q": (0.8313, 2.3773, -0.3655),
    },
    (4, 4): {
        "f": (2.3000, -1.5056, 0.2244),
        "q": (1.1929, 3.1191, -0.4825),
    },
    (4, 3): {
        "f": (1.6869, -0.8862, 0.2822),
        "q": (1.4812, 2.8096, -0.4271),
    },
}


def _swsh_projection_coefficients_cartesian(
    l: int,
    m: int,
    iota: torch.Tensor,
    phi0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute polarization projection coefficients (dp_re, dp_im, dc_re, dc_im) for mode (l, m) in pure Cartesian real arithmetic.

    For positive frequencies, the coefficient of the positive-m mode and its
    negative-m partner are combined as
        d_plus  = 0.5 * (Y_{l,m} + (-1)^l * conj(Y_{l,-m}))
        d_cross = -0.5i * (Y_{l,m} - (-1)^l * conj(Y_{l,-m}))
    where Y_{l,m} = {}_{-2}Y_{lm}(iota, phi0).

    Returns:
        (dp_re, dp_im, dc_re, dc_im) real tensors.
    """
    cos_i = torch.cos(iota)
    sin_i = torch.sin(iota)

    one_plus_cos = 1.0 + cos_i
    one_minus_cos = 1.0 - cos_i

    m_phi0 = float(m) * phi0
    cos_m = torch.cos(m_phi0)
    sin_m = torch.sin(m_phi0)

    if l == 2 and m == 2:
        c_norm = math.sqrt(5.0 / (64.0 * math.pi))
        A = c_norm * (one_plus_cos * one_plus_cos)
        B = c_norm * (one_minus_cos * one_minus_cos)
        sign_l = 1.0
    elif l == 2 and m == 1:
        c_norm = math.sqrt(5.0 / (16.0 * math.pi))
        A = c_norm * sin_i * one_plus_cos
        B = c_norm * sin_i * one_minus_cos
        sign_l = 1.0
    elif l == 3 and m == 3:
        c_norm = math.sqrt(21.0 / (128.0 * math.pi))
        A = -c_norm * sin_i * (one_plus_cos * one_plus_cos)
        B = c_norm * sin_i * (one_minus_cos * one_minus_cos)
        sign_l = -1.0
    elif l == 3 and m == 2:
        c_norm = math.sqrt(7.0 / (4.0 * math.pi)) * 0.25
        A = c_norm * (one_plus_cos * one_plus_cos) * (3.0 * cos_i - 2.0)
        B = c_norm * (one_minus_cos * one_minus_cos) * (3.0 * cos_i + 2.0)
        sign_l = -1.0
    elif l == 4 and m == 4:
        c_norm = math.sqrt(63.0 / (256.0 * math.pi))
        A = c_norm * (sin_i * sin_i) * (one_plus_cos * one_plus_cos)
        B = c_norm * (sin_i * sin_i) * (one_minus_cos * one_minus_cos)
        sign_l = 1.0
    elif l == 4 and m == 3:
        c_norm = 3.0 * math.sqrt(14.0) / (16.0 * math.sqrt(math.pi))
        A = c_norm * sin_i * (one_plus_cos * one_plus_cos) * (1.0 - 2.0 * cos_i)
        B = c_norm * sin_i * (one_minus_cos * one_minus_cos) * (1.0 + 2.0 * cos_i)
        sign_l = 1.0
    else:
        raise ValueError(f"Mode ({l}, {m}) not currently supported.")

    k_plus = 0.5 * (A + sign_l * B)
    k_cross = 0.5 * (A - sign_l * B)
    dp_re = k_plus * cos_m
    dp_im = k_plus * sin_m
    dc_re = k_cross * sin_m
    dc_im = -k_cross * cos_m
    return dp_re, dp_im, dc_re, dc_im


def _swsh_projection_coefficients(
    l: int,
    m: int,
    iota: torch.Tensor,
    phi0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute polarization projection coefficients d_plus and d_cross for mode (l, m).

    Returns:
        (d_plus, d_cross) complex tensors.
    """
    dp_re, dp_im, dc_re, dc_im = _swsh_projection_coefficients_cartesian(l, m, iota, phi0)
    d_plus = torch.view_as_complex(torch.stack([dp_re, dp_im], dim=-1))
    d_cross = torch.view_as_complex(torch.stack([dc_re, dc_im], dim=-1))
    return d_plus, d_cross


def frequency_mapping_amplitude(
    f: torch.Tensor,
    f0_amp: Union[float, torch.Tensor],
    f_rd_lm: Union[float, torch.Tensor],
    f_rd_22: Union[float, torch.Tensor],
    m: int,
) -> torch.Tensor:
    """London et al. (2018) piecewise amplitude frequency mapping f_22^A(f).

    Args:
        f: Frequency tensor
        f0_amp: Transition frequency f_0^A
        f_rd_lm: Ringdown frequency of (l, m) mode
        f_rd_22: Ringdown frequency of (2, 2) mode
        m: Azimuthal mode number
    """
    s2 = (f_rd_22 - (2.0 / float(m)) * f0_amp) / (f_rd_lm - f0_amp)
    reg1 = (2.0 / float(m)) * f
    reg2 = s2 * (f - f0_amp) + (2.0 / float(m)) * f0_amp
    reg3 = f - (f_rd_lm - f_rd_22)
    return torch.where(f <= f0_amp, reg1, torch.where(f <= f_rd_lm, reg2, reg3))


def frequency_mapping_phase(
    f: torch.Tensor,
    f0_phi: Union[float, torch.Tensor],
    f_rd_lm: Union[float, torch.Tensor],
    f_rd_22: Union[float, torch.Tensor],
    m: int,
) -> torch.Tensor:
    """London et al. (2018) piecewise phase frequency mapping f_22^phi(f).

    Args:
        f: Frequency tensor
        f0_phi: Transition frequency f_0^phi
        f_rd_lm: Ringdown frequency of (l, m) mode
        f_rd_22: Ringdown frequency of (2, 2) mode
        m: Azimuthal mode number
    """
    s2 = (f_rd_22 - (2.0 / float(m)) * f0_phi) / (f_rd_lm - f0_phi)
    s3 = f_rd_22 / f_rd_lm
    reg1 = (2.0 / float(m)) * f
    reg2 = s2 * (f - f0_phi) + (2.0 / float(m)) * f0_phi
    reg3 = s3 * f
    return torch.where(f <= f0_phi, reg1, torch.where(f <= f_rd_lm, reg2, reg3))


def _d_ringdown_phase_correction(
    mf: torch.Tensor,
    eta: torch.Tensor,
    alpha4: torch.Tensor,
    alpha5: torch.Tensor,
    f_rd_22: torch.Tensor,
    f_damp_22: torch.Tensor,
    f_rd_lm: torch.Tensor,
    f_damp_lm: torch.Tensor,
) -> torch.Tensor:
    """Replace the D ringdown arctangent by London Eq. (9), with C1 matching.

    All frequencies are geometric. Subtracting the value and tangent of the
    arctangent difference at D's merger-ringdown join recomputes alpha_0 and
    alpha_1 without changing the inspiral/intermediate phase or global gauge.
    This is an intrinsic correction; reference-frequency anchoring is applied
    by the caller using the same reference as the carrier.
    """
    ratio = f_rd_22 / f_rd_lm
    width = ratio * f_damp_lm
    center = alpha5 * f_rd_22
    join = 0.5 * f_rd_22

    def difference(x: torch.Tensor) -> torch.Tensor:
        return (alpha4 / eta) * (
            ratio * torch.atan((x - center) / width)
            - torch.atan((x - center) / f_damp_22)
        )

    offset = join - center
    tangent = (alpha4 / eta) * (
        ratio * width / (width.square() + offset.square())
        - f_damp_22 / (f_damp_22.square() + offset.square())
    )
    correction = difference(mf) - difference(join) - (mf - join) * tangent
    return torch.where(mf > join, correction, torch.zeros_like(correction))


def h_hat_multipole(
    l: int,
    m: int,
    v_m: torch.Tensor,
    delta: torch.Tensor,
    eta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Evaluate leading PN multipole amplitude ratio H_hat_lm(f).

    References:
    - Mishra et al. (2016), Phys. Rev. D 94, 084053
    - London et al. (2018), Phys. Rev. Lett. 120, 161102
    """
    if (l, m) == (2, 2):
        return -torch.ones_like(v_m)
    elif (l, m) == (3, 3):
        c = -0.75 * math.sqrt(5.0 / 7.0)
        return c * delta * v_m
    elif (l, m) == (4, 4):
        c = -(4.0 / 9.0) * math.sqrt(10.0 / 7.0)
        return c * (1.0 - 3.0 * eta) * (v_m ** 2.0)
    elif (l, m) == (2, 1):
        c = -math.sqrt(2.0) / 3.0
        return c * (delta * v_m - 1.5 * (chi_a + delta * chi_s) * (v_m ** 2.0))
    elif (l, m) == (3, 2):
        c = -(1.0 / 3.0) * math.sqrt(5.0 / 7.0)
        return c * ((1.0 - 3.0 * eta) * (v_m ** 2.0) + 4.0 * eta * chi_s * (v_m ** 3.0))
    elif (l, m) == (4, 3):
        c = -0.75 * math.sqrt(3.0 / 35.0)
        return c * (delta * (1.0 - 2.0 * eta) * (v_m ** 3.0) + 2.5 * eta * (chi_a - delta * chi_s) * (v_m ** 4.0))
    else:
        raise ValueError(f"Mode ({l}, {m}) not supported.")


# The production route keeps the public leading-PN helper above for API
# compatibility.  London Eq. (4), however, needs the complex Arun-compatible
# 2,1 factor used by the independently audited assembly.  Keeping this small
# transcription local avoids a runtime dependency on diagnostic modules.
def _production_hhat(
    l: int,
    m: int,
    v_m: torch.Tensor,
    delta: torch.Tensor,
    eta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    if (l, m) == (2, 2):
        return -torch.ones_like(v_m)
    if (l, m) == (3, 3):
        return -(3.0 / 4.0) * torch.sqrt(v_m.new_tensor(5.0 / 7.0)) * delta * v_m
    if (l, m) == (4, 4):
        return -(4.0 / 9.0) * torch.sqrt(v_m.new_tensor(10.0 / 7.0)) * (1.0 - 3.0 * eta) * v_m.square()
    if (l, m) == (2, 1):
        # London et al. retain the absolute-V^3 terms in this mode and use
        # the complex tail/spin correction through the next displayed order.
        c_spa = delta * (
            (113.0 / 24.0 - 19.0 * eta / 6.0) * chi_s
            + (113.0 / 24.0) * delta * chi_a
        )
        c_spa = c_spa - 1.5 * (chi_a + delta * chi_s) * (
            743.0 / 672.0 + 11.0 * eta / 8.0
        )
        c_td = (0.5 - 205.0 * eta / 14.0) * chi_a + delta * (
            0.5 - 33.0 * eta / 14.0
        ) * chi_s
        one = torch.ones_like(v_m)
        c_tail = delta * torch.complex(
            -math.pi * one,
            (-0.5 - 2.0 * math.log(2.0)) * one,
        )
        c = c_tail + c_spa - c_td
        return -(torch.sqrt(v_m.new_tensor(2.0)) / 3.0) * (
            delta * v_m
            - 1.5 * (chi_a + delta * chi_s) * v_m.square()
            + delta * (335.0 / 672.0 + 117.0 * eta / 56.0) * v_m.pow(3.0)
            + c * v_m.pow(4.0)
        )
    if (l, m) == (3, 2):
        return -(1.0 / 3.0) * torch.sqrt(v_m.new_tensor(5.0 / 7.0)) * (
            (1.0 - 3.0 * eta) * v_m.square()
        )
    if (l, m) == (4, 3):
        # The London production truncation retains the leading antisymmetric
        # mass term for (4,3).  The public ``h_hat_multipole`` helper above
        # remains the full restricted-PN expression for callers that need its
        # displayed higher-order spin correction.
        return -(3.0 / 4.0) * torch.sqrt(v_m.new_tensor(3.0 / 35.0)) * (
            delta * (1.0 - 2.0 * eta) * v_m.pow(3.0)
        )
    raise ValueError(f"Mode ({l}, {m}) not supported.")


def _tensor_like(value: Union[float, torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """Convert a scalar without detaching a tensor already on the graph."""
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _reshape_if_same_numel(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Keep carrier callback boundary values broadcastable for batched calls."""
    if value.shape != shape and value.numel() == math.prod(shape):
        return value.reshape(shape)
    return value


def _production_eq14_delta_c1(
    mf: torch.Tensor,
    eta: torch.Tensor,
    chi_pn: torch.Tensor,
    hm_fr22: torch.Tensor,
    hm_flm: torch.Tensor,
    hm_fd22: torch.Tensor,
    hm_fdlm: torch.Tensor,
    alpha4: torch.Tensor,
    alpha5: torch.Tensor,
    *,
    base_fr22: Optional[torch.Tensor] = None,
    base_fd22: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """London Eq. (14) correction with the D carrier QNM contract.

    The outer map receives the HM pair.  The two arctangents retain the
    absolute D-carrier pair, with HM information entering only through
    ``R=fRD_HM/fRD_lm`` and ``T=fdamp_lm/fdamp_HM``.  The difference is
    C1-completed at ``0.5*fRD_D``.
    """
    fr22 = hm_fr22
    flm = hm_flm
    fd22 = hm_fd22
    fdlm = hm_fdlm
    ratio = fr22 / flm
    damping_ratio = fdlm / fd22
    carrier_fr22 = fr22 if base_fr22 is None else base_fr22
    carrier_fd22 = fd22 if base_fd22 is None else base_fd22
    center = alpha5 * carrier_fr22
    width_lm = carrier_fd22 * ratio * damping_ratio
    join = 0.5 * carrier_fr22

    def raw(x: torch.Tensor) -> torch.Tensor:
        return (alpha4 / eta) * (
            ratio * torch.atan((x - center) / width_lm)
            - torch.atan((x - center) / carrier_fd22)
        )

    offset = join - center
    at_join_slope = (alpha4 / eta) * (
        ratio * width_lm / (width_lm.square() + offset.square())
        - carrier_fd22 / (carrier_fd22.square() + offset.square())
    )
    correction = raw(mf) - raw(join) - at_join_slope * (mf - join)
    return torch.where(mf > join, correction, torch.zeros_like(correction))


def _production_mapped_phase_c1(
    f: torch.Tensor,
    f0_phi: torch.Tensor,
    f_rd_lm: torch.Tensor,
    f_rd_22: torch.Tensor,
    m: int,
    ell: int,
    carrier_phase,
) -> torch.Tensor:
    """Map a positive-D carrier while preserving value and derivative joins."""
    mid_slope = (f_rd_22 - 2.0 * f0_phi / float(m)) / (f_rd_lm - f0_phi)
    g0 = 2.0 * f0_phi / float(m)
    g_mid = mid_slope * (f - f0_phi) + g0
    g_high = (f_rd_22 / f_rd_lm) * f
    g = torch.where(
        f <= f0_phi,
        2.0 * f / float(m),
        torch.where(f <= f_rd_lm, g_mid, g_high),
    )
    k_mid = 1.0 / mid_slope
    k_high = f_rd_lm / f_rd_22
    low_delta = f.new_tensor(
        math.pi / 2.0 * (3 * ell + ((ell + m) % 2)) - math.pi
    )

    phi_g0 = _reshape_if_same_numel(carrier_phase(g0), g0.shape)
    delta_mid = low_delta + (float(m) / 2.0 - k_mid) * phi_g0
    phi_fr = _reshape_if_same_numel(carrier_phase(f_rd_22), f_rd_22.shape)
    delta_high = k_mid * phi_fr + delta_mid - k_high * phi_fr

    low = (float(m) / 2.0) * carrier_phase(g) + low_delta
    middle = k_mid * carrier_phase(g) + delta_mid
    high = k_high * carrier_phase(g) + delta_high
    return torch.where(f <= f0_phi, low, torch.where(f <= f_rd_lm, middle, high))


class IMRPhenomHM(nn.Module):
    """IMRPhenomHM Multipolar Higher-Harmonic IMR Gravitational Waveform Approximant.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)}.
    Literature:
    - London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
    - Mishra et al. (2016), Phys. Rev. D 94, 084053 (arXiv:1609.00727)
    - Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
    """

    SUPPORTED_MODES: Tuple[Tuple[int, int], ...] = (
        (2, 2),
        (2, 1),
        (3, 3),
        (3, 2),
        (4, 4),
        (4, 3),
    )

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        carrier: str = "D",
        base_22: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        if base_22 is not None:
            self.base_22 = base_22
        elif carrier.upper() in ("XAS", "IMRPHENOMXAS"):
            from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
            self.base_22 = IMRPhenomXAS()
        else:
            self.base_22 = IMRPhenomD()

    @staticmethod
    def mode_qnm_frequencies(
        l: int,
        m: int,
        mass1: torch.Tensor,
        mass2: torch.Tensor,
        chi1: torch.Tensor,
        chi2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fundamental QNM frequency f_RD and damping width f_damp for mode (l, m).

        HM modes use the independent N=61 London polynomial artifact in
        final-mass units, scaled to the total binary mass.  The artifact
        records its sampling-grid hypothesis and payload hash in
        :mod:`imrphenom_hm_qnm_n61`; evaluation remains Torch-native so
        final-spin gradients, dtype, and device are preserved.

        Returns ``(f_rd, f_damp)`` in dimensionless total-mass units ``M*f``.
        """
        total_mass = mass1 + mass2
        m_final, a_final, _, _ = IMRPhenomD.remnant_properties(mass1, mass2, chi1, chi2)
        omega_real_final, omega_imag_final = evaluate_n61_qnm((l, m), a_final)
        m_ratio = total_mass / m_final
        scale = m_ratio / (2.0 * PI)
        return scale * omega_real_final, scale * omega_imag_final

    def _legacy_forward(
        self,
        f: Optional[torch.Tensor] = None,
        mass1: Union[float, torch.Tensor] = 30.0,
        mass2: Union[float, torch.Tensor] = 30.0,
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        f_lower: Optional[float] = None,
        phi0: Union[float, torch.Tensor] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        return_unprojected_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]],
    ]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including higher multipoles.

        Args:
            f: Frequency tensor in Hz
            mass1, mass2: Component masses in solar masses (mass1 >= mass2)
            chi1, chi2: Aligned dimensionless spins
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz
            f_lower: Optional dispatcher frequency lower bound; output masking is applied by the wrapper
            phi0: Observer azimuth angle in radians (default 0.0)
            modes: Optional tuple of modes to evaluate (defaults to all supported modes)
            return_modes: If True, also return dictionary of mode waveforms
            return_unprojected_modes: If True, also return dictionary of unprojected h_mode waveforms

        Returns:
            (h_plus, h_cross) or (h_plus, h_cross, mode_dict)
        """
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

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
        phi_0 = as_tensor(phi0, device=device, dtype=dtype)

        active_modes = tuple(modes) if modes is not None else self.modes

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)
        chi_s = 0.5 * (s1 + s2)
        chi_a = 0.5 * (s1 - s2)
        m_sec = total_mass * MSUN_SEC

        is_batched = total_mass.dim() >= 1
        if is_batched:
            if f.dim() == 1:
                f_eval = f.unsqueeze(0)
            else:
                f_eval = f

            def _unsqueeze(t: torch.Tensor) -> torch.Tensor:
                if t.dim() < f_eval.dim():
                    return t.unsqueeze(-1)
                return t

            m_sec_eval = _unsqueeze(m_sec)
            eta_eval = _unsqueeze(eta)
            delta_eval = _unsqueeze(delta)
            chi_s_eval = _unsqueeze(chi_s)
            chi_a_eval = _unsqueeze(chi_a)
            inc_eval = _unsqueeze(inc)
            phi0_eval = _unsqueeze(phi_0)
            dist_eval = _unsqueeze(dist)
            phic_eval = _unsqueeze(phi_c)
            tc_eval = _unsqueeze(t_c)
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_eval = eta
            delta_eval = delta
            chi_s_eval = chi_s
            chi_a_eval = chi_a
            inc_eval = inc
            phi0_eval = phi_0
            dist_eval = dist
            phic_eval = phi_c
            tc_eval = t_c

        # Evaluate base (2, 2) mode at face-on (iota=0)
        try:
            hp_22_raw, _, phi_22_raw = self.base_22(
                f=f,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
                f_ref=f_ref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                return_phase=True,
            )
        except TypeError:
            hp_22_raw, _ = self.base_22(
                f=f,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
                f_ref=f_ref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
            )
            phi_22_raw = -torch.angle(hp_22_raw)

        amp_22 = torch.abs(hp_22_raw)
        phi_22 = phi_22_raw + 2.0 * PI * f_eval * tc_eval - 2.0 * phic_eval

        h_plus_total = torch.zeros_like(
            amp_22, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64
        )
        h_cross_total = torch.zeros_like(h_plus_total)
        mode_dict = {}

        # 22 QNM ringdown frequency
        f_rd_22_geom, f_damp_22_geom = self.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)
        if is_batched:
            f_rd_22_u = _unsqueeze(f_rd_22_geom)
            f_damp_22_u = _unsqueeze(f_damp_22_geom)
        else:
            f_rd_22_u = f_rd_22_geom
            f_damp_22_u = f_damp_22_geom
        f_rd_22_phys = f_rd_22_u / m_sec_eval

        # Eq. (9) refines the D ansatz specifically; arbitrary carriers retain
        # the model-agnostic phase mapping.
        use_d_ringdown = isinstance(self.base_22, IMRPhenomD)
        if use_d_ringdown:
            _, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)
            alpha4 = IMRPhenomD.eval_table_v("alpha4", eta, chi_pn)
            alpha5 = IMRPhenomD.eval_table_v("alpha5", eta, chi_pn)
            if is_batched:
                alpha4 = _unsqueeze(alpha4)
                alpha5 = _unsqueeze(alpha5)
            phase_convention = reference_frame if reference_frame is not None else (
                phasing_convention if phasing_convention is not None else
                ("reference_frequency" if f_ref is not None else "coalescence")
            )
            reference_anchored = phase_convention.lower() in ("reference_frequency", "reffreq", "ref")

        norm_mode = math.sqrt(16.0 * PI / 5.0)

        for l, m in active_modes:
            if (l, m) == (2, 2):
                exp_neg_i_phi22 = torch.complex(torch.cos(phi_22), -torch.sin(phi_22))
                h_mode = (norm_mode * amp_22).to(exp_neg_i_phi22.dtype) * exp_neg_i_phi22
            else:
                # 1. QNM ringdown for mode (l, m)
                f_rd_lm_geom, f_damp_lm_geom = self.mode_qnm_frequencies(l, m, m1, m2, s1, s2)
                if is_batched:
                    f_rd_lm_u = _unsqueeze(f_rd_lm_geom)
                    f_damp_lm_u = _unsqueeze(f_damp_lm_geom)
                else:
                    f_rd_lm_u = f_rd_lm_geom
                    f_damp_lm_u = f_damp_lm_geom
                f_rd_lm_phys = f_rd_lm_u / m_sec_eval

                if use_d_ringdown:
                    def ringdown_correction(frequencies: torch.Tensor) -> torch.Tensor:
                        def intrinsic(mf: torch.Tensor) -> torch.Tensor:
                            return _d_ringdown_phase_correction(
                                mf, eta_eval, alpha4, alpha5, f_rd_22_u,
                                f_damp_22_u, f_rd_lm_u, f_damp_lm_u,
                            )

                        result = intrinsic(frequencies * m_sec_eval)
                        if reference_anchored:
                            reference = as_tensor(
                                f_ref if f_ref is not None else frequencies.reshape(-1)[0],
                                device=device, dtype=dtype,
                            )
                            if is_batched:
                                reference = _unsqueeze(reference)
                            result = result - intrinsic(reference * m_sec_eval)
                        return result

                # Transition frequencies in physical Hz (PhenomD amp cut Mf=0.014, phase cut Mf=0.018)
                f0_amp = (0.014 / m_sec_eval) * (f_rd_lm_phys / f_rd_22_phys)
                f0_phi = (0.018 / m_sec_eval) * (f_rd_lm_phys / f_rd_22_phys)

                # 2. London et al. (2018) Frequency Mappings
                f_map_amp = frequency_mapping_amplitude(f_eval, f0_amp, f_rd_lm_phys, f_rd_22_phys, m)
                f_map_phi = frequency_mapping_phase(f_eval, f0_phi, f_rd_lm_phys, f_rd_22_phys, m)

                # 3. Base 22 amplitude at mapped frequency f_22^A
                f_map_amp_safe = torch.clamp(f_map_amp, min=1e-8)
                hp_amp_eval, _ = self.base_22(
                    f=f_map_amp_safe,
                    mass1=m1,
                    mass2=m2,
                    chi1=s1,
                    chi2=s2,
                    distance=dist,
                    inclination=0.0,
                    phic=0.0,
                    tc=0.0,
                    f_ref=f_ref,
                    phasing_convention=phasing_convention,
                    reference_frame=reference_frame,
                )
                amp_22_mapped = torch.abs(hp_amp_eval)

                # 4. Base 22 phase at mapped frequency f_22^phi
                f_map_phi_safe = torch.clamp(f_map_phi, min=1e-8)
                try:
                    _, _, phi_22_mapped = self.base_22(
                        f=f_map_phi_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                        return_phase=True,
                    )
                except TypeError:
                    hp_phi_eval, _ = self.base_22(
                        f=f_map_phi_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                    )
                    phi_22_mapped = -torch.angle(hp_phi_eval)

                if use_d_ringdown:
                    phi_22_mapped = phi_22_mapped + ringdown_correction(f_map_phi_safe)

                # 5. Base 22 phase at boundaries f_b0 = 2*f0_phi/m and f_bRD = f_rd_22_phys
                f_b0 = (2.0 / float(m)) * f0_phi
                f_bRD = f_rd_22_phys
                if is_batched:
                    f_boundaries = torch.cat([f_b0, f_bRD], dim=-1)
                else:
                    f_boundaries = torch.stack([f_b0, f_bRD])
                f_boundaries_safe = torch.clamp(f_boundaries, min=1e-8)
                try:
                    _, _, phi_bounds = self.base_22(
                        f=f_boundaries_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                        return_phase=True,
                    )
                except TypeError:
                    hp_bounds, _ = self.base_22(
                        f=f_boundaries_safe,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=0.0,
                        tc=0.0,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                    )
                    phi_bounds = -torch.angle(hp_bounds)

                if use_d_ringdown:
                    phi_bounds = phi_bounds + ringdown_correction(f_boundaries_safe)

                if is_batched:
                    phi_22_f0 = phi_bounds[..., 0:1]
                    phi_22_frd = phi_bounds[..., 1:2]
                else:
                    phi_22_f0 = phi_bounds[0]
                    phi_22_frd = phi_bounds[1]

                # 6. Phase Connection
                slope_2 = (f_rd_22_phys - (2.0 / float(m)) * f0_phi) / (f_rd_lm_phys - f0_phi)
                slope_3 = f_rd_22_phys / f_rd_lm_phys
                kappa_1 = float(m) / 2.0
                kappa_2 = 1.0 / slope_2
                kappa_3 = 1.0 / slope_3

                delta_1 = (PI / 2.0) * float(3 * l + ((l + m) % 2)) - PI
                delta_2 = (kappa_1 - kappa_2) * phi_22_f0 + delta_1
                delta_3 = (kappa_2 - kappa_3) * phi_22_frd + delta_2

                kappa_f = torch.where(f_eval <= f0_phi, kappa_1, torch.where(f_eval <= f_rd_lm_phys, kappa_2, kappa_3))
                delta_f = torch.where(f_eval <= f0_phi, delta_1, torch.where(f_eval <= f_rd_lm_phys, delta_2, delta_3))

                phi_lm_int = kappa_f * phi_22_mapped + delta_f
                if f_ref is not None:
                    slope_peak = IMRPhenomD.compute_slope_peak(m1, m2, s1, s2)
                    if is_batched:
                        slope_peak_u = _unsqueeze(slope_peak)
                        fref_u = _unsqueeze(as_tensor(f_ref, device=device, dtype=dtype))
                    else:
                        slope_peak_u = slope_peak
                        fref_u = as_tensor(f_ref, device=device, dtype=dtype)
                    delta_t_gauge = (float(m) / 2.0 - 1.0) * slope_peak_u * m_sec_eval * fref_u
                    psi_lm = phi_lm_int - delta_t_gauge + 2.0 * PI * f_eval * tc_eval - float(m) * phic_eval
                else:
                    psi_lm = phi_lm_int + 2.0 * PI * f_eval * tc_eval - float(m) * phic_eval

                # 7. Amplitude Rescaling: A_lm(f) = |beta_lm(f)| * A_22[f_22^A(f)]
                v_m = torch.pow(PI * m_sec_eval * torch.clamp((2.0 / float(m)) * f_eval, min=1e-8), 1.0 / 3.0)
                h_hat = h_hat_multipole(l, m, v_m, delta_eval, eta_eval, chi_s_eval, chi_a_eval)
                # The alternative after London Eq. (4) is H_lm(f)/H_22(f_map).
                # H_hat above removes the common SPA v^(-7/2) envelope; restore
                # its ratio here. It equals one only in the inspiral mapping.
                spa_ratio = torch.pow(
                    torch.clamp((2.0 / float(m)) * f_eval, min=1e-8) / f_map_amp_safe,
                    -7.0 / 6.0,
                )
                beta_lm = -h_hat * spa_ratio
                amp_lm = torch.abs(beta_lm) * amp_22_mapped

                # Complex mode strain (with odd-l parity factor (-1)^l)
                parity = (-1.0) ** l
                exp_neg_i_psi = torch.complex(torch.cos(psi_lm), -torch.sin(psi_lm))
                h_mode = parity * (norm_mode * torch.sign(beta_lm) * amp_lm).to(exp_neg_i_psi.dtype) * exp_neg_i_psi
                h_mode = torch.where(f_eval > 0.0, h_mode, torch.zeros_like(h_mode))

            # Spin-weighted spherical harmonics projection
            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi0_eval)
            hp_lm = d_plus * h_mode
            hc_lm = d_cross * h_mode

            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm

            if return_modes:
                mode_dict[(l, m)] = (hp_lm, hc_lm)
            elif return_unprojected_modes:
                mode_dict[(l, m)] = h_mode

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus_total = torch.where(f_eval > 0.0, h_plus_total, torch.zeros_like(h_plus_total))
        h_cross_total = torch.where(f_eval > 0.0, h_cross_total, torch.zeros_like(h_cross_total))

        if return_modes or return_unprojected_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def forward(
        self,
        f: Optional[torch.Tensor] = None,
        mass1: Union[float, torch.Tensor] = 30.0,
        mass2: Union[float, torch.Tensor] = 30.0,
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        f_lower: Optional[float] = None,
        phi0: Union[float, torch.Tensor] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        return_unprojected_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
        hm_mapping_transitions: str = "phenomd_runtime",
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]],
    ]:
        """Generate the Torch-native London HM assembly.

        HM QNM values control the London frequency maps.  For the D carrier,
        its own absolute QNM pair is retained inside the Eq. (14) arctangent;
        the HM pair enters there only as the two London ratios.  All boundary
        evaluations and projections stay in Torch so parameter gradients are
        retained in scalar and batched calls.
        """
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        if hm_mapping_transitions not in ("phenomd_runtime", "london_2018"):
            raise ValueError(f"Unknown HM mapping transitions: {hm_mapping_transitions}")
        if hm_mapping_transitions == "london_2018" and not isinstance(self.base_22, IMRPhenomD):
            raise ValueError("london_2018 HM mapping transitions require the D carrier")

        # Keep custom/XAS carriers on the established opaque-carrier path.
        # The Torch-native London route below relies on the IMRPhenomD
        # reference gauge and its remnant contract; a custom carrier retains
        # the legacy behavior rather than being silently coerced into D.
        if not isinstance(self.base_22, IMRPhenomD):
            return self._legacy_forward(
                f=f,
                mass1=mass1,
                mass2=mass2,
                chi1=chi1,
                chi2=chi2,
                distance=distance,
                inclination=inclination,
                phic=phic,
                tc=tc,
                f_ref=f_ref,
                f_lower=f_lower,
                phi0=phi0,
                modes=modes,
                return_modes=return_modes,
                return_unprojected_modes=return_unprojected_modes,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                sample_frequencies=sample_frequencies,
            )
        if not isinstance(f, torch.Tensor):
            f = torch.as_tensor(f)
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64
        f = f.to(dtype=dtype)

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)
        phi_0 = as_tensor(phi0, device=device, dtype=dtype)

        active_modes = tuple(modes) if modes is not None else self.modes
        for lm in active_modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")

        # Form the broadcast batch shape from every parameter and from any
        # explicit frequency batch axes.  A scalar mass is therefore allowed
        # with batched spins, angles, distances, or phases.  A one-dimensional
        # frequency vector is shared by all parameter rows; an explicit
        # (..., N) frequency tensor contributes its leading batch shape.
        fref_input = None if f_ref is None else _tensor_like(f_ref, f)
        batch_shape = torch.Size()
        for value in (m1, m2, s1, s2, dist, inc, phi_c, t_c, phi_0):
            try:
                batch_shape = torch.broadcast_shapes(batch_shape, value.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"HM parameter shapes are not broadcastable: {batch_shape} and {tuple(value.shape)}"
                ) from exc
        if f.dim() > 1:
            try:
                batch_shape = torch.broadcast_shapes(batch_shape, f.shape[:-1])
            except RuntimeError as exc:
                raise ValueError(
                    f"HM frequency batch shape {tuple(f.shape[:-1])} is not broadcastable with parameters {tuple(batch_shape)}"
                ) from exc
        if fref_input is not None:
            try:
                batch_shape = torch.broadcast_shapes(batch_shape, fref_input.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"HM reference-frequency shape {tuple(fref_input.shape)} is not broadcastable with parameters {tuple(batch_shape)}"
                ) from exc

        is_batched = len(batch_shape) > 0
        has_frequency_axis = f.dim() > 0

        def broadcast_batch(value: torch.Tensor) -> torch.Tensor:
            return value.expand(batch_shape) if value.shape != batch_shape else value

        def expand_param(value: torch.Tensor) -> torch.Tensor:
            value = broadcast_batch(value)
            return value.unsqueeze(-1) if is_batched and has_frequency_axis else value

        if is_batched:
            if f.dim() == 0:
                f_eval = f.expand(batch_shape)
            elif f.dim() == 1:
                f_eval = f.reshape((1,) * len(batch_shape) + (f.shape[0],)).expand(
                    batch_shape + (f.shape[0],)
                )
            else:
                f_eval = f.expand(batch_shape + (f.shape[-1],))
        else:
            f_eval = f

        # Keep the unexpanded parameter tensors for calls whose API expects a
        # batch-shaped scalar, and use the trailing singleton form for calls
        # evaluated on the frequency axis.
        m1 = broadcast_batch(m1)
        m2 = broadcast_batch(m2)
        s1 = broadcast_batch(s1)
        s2 = broadcast_batch(s2)
        dist = broadcast_batch(dist)
        fref_batch = None if fref_input is None else broadcast_batch(fref_input)
        fref_eval = None if fref_batch is None else expand_param(fref_batch)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)
        chi_s = 0.5 * (s1 + s2)
        chi_a = 0.5 * (s1 - s2)
        _, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)
        m_sec = total_mass * MSUN_SEC

        m_sec_eval = expand_param(m_sec)
        m1_eval = expand_param(m1)
        m2_eval = expand_param(m2)
        s1_eval = expand_param(s1)
        s2_eval = expand_param(s2)
        dist_eval = expand_param(dist)
        # London's mappings and Eq. (14) use dimensionless geometric
        # frequency ``Mf``.  The public HM API accepts Hz, so retain both
        # coordinates: ``f_eval`` for the carrier/output and ``mf_eval`` for
        # the HM assembly.
        mf_eval = f_eval * m_sec_eval
        eta_eval = expand_param(eta)
        delta_eval = expand_param(delta)
        chi_s_eval = expand_param(chi_s)
        chi_a_eval = expand_param(chi_a)
        chi_pn_eval = expand_param(chi_pn)
        inc_eval = expand_param(inc)
        phi0_eval = expand_param(phi_0)
        phic_eval = expand_param(phi_c)
        tc_eval = expand_param(t_c)

        # The base callback is deliberately kept opaque: production HM only
        # consumes its face-on amplitude and positive-D phase coordinate.
        def _base_call(
            geom_frequency: torch.Tensor,
            *,
            reference_frequency: Optional[Union[float, torch.Tensor]],
            need_phase: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            frequency_hz = torch.clamp(geom_frequency / m_sec_eval, min=1.0e-8)
            # IMRPhenomD treats a one-dimensional frequency tensor as a
            # shared grid whenever its intrinsic parameters are batched.  A
            # scalar public frequency has no frequency axis, so add a private
            # singleton axis for that carrier call and restore the scalar
            # public shape below.
            carrier_frequency = frequency_hz
            if is_batched and not has_frequency_axis:
                carrier_frequency = carrier_frequency.unsqueeze(-1)
            kwargs = {
                "f": carrier_frequency,
                "mass1": m1_eval,
                "mass2": m2_eval,
                "chi1": s1_eval,
                "chi2": s2_eval,
                "distance": dist_eval,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
                "f_ref": reference_frequency,
                "phasing_convention": phasing_convention,
                "reference_frame": reference_frame,
            }
            if need_phase:
                result = self.base_22(return_phase=True, **kwargs)
            else:
                result = self.base_22(**kwargs)

            hp_raw, hc_raw = result[0], result[1]
            phase_raw = result[2] if len(result) >= 3 else None
            hp_raw = _reshape_if_same_numel(hp_raw, geom_frequency.shape)
            hc_raw = _reshape_if_same_numel(hc_raw, geom_frequency.shape)
            if phase_raw is not None:
                phase_raw = _reshape_if_same_numel(phase_raw, geom_frequency.shape)
            if need_phase and phase_raw is None:
                zero_iota = torch.zeros_like(geom_frequency)
                dplus, _ = _swsh_projection_coefficients(2, 2, zero_iota, zero_iota)
                phase_raw = -torch.angle(hp_raw / dplus)
            return hp_raw, hc_raw, phase_raw

        def amplitude_22(geom_frequency: torch.Tensor) -> torch.Tensor:
            hp_raw, _, _ = _base_call(
                geom_frequency,
                reference_frequency=None,
                need_phase=False,
            )
            zero_iota = torch.zeros_like(geom_frequency)
            dplus, _ = _swsh_projection_coefficients(2, 2, zero_iota, zero_iota)
            return torch.abs(hp_raw / dplus)

        def carrier_positive(geom_frequency: torch.Tensor) -> torch.Tensor:
            _, _, phase_raw = _base_call(
                geom_frequency,
                reference_frequency=fref_eval,
                need_phase=True,
            )
            if phase_raw is None:
                raise RuntimeError("base carrier did not provide a phase")
            return _reshape_if_same_numel(phase_raw, geom_frequency.shape)

        # D's remnant call provides the absolute carrier pair in total-mass
        # units.  HM mode_qnm_frequencies supplies the independent N=61 pair.
        m_final, a_final, base_fr22, base_fd22 = IMRPhenomD.remnant_properties(m1, m2, s1, s2)
        hm_qnm: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {
            lm: self.mode_qnm_frequencies(lm[0], lm[1], m1, m2, s1, s2)
            for lm in active_modes
        }
        if (2, 2) not in hm_qnm:
            hm_qnm[(2, 2)] = self.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)

        hm_fr22 = expand_param(hm_qnm[(2, 2)][0])
        hm_fd22 = expand_param(hm_qnm[(2, 2)][1])
        base_fr22_eval = expand_param(base_fr22)
        base_fd22_eval = expand_param(base_fd22)
        alpha4 = expand_param(IMRPhenomD.eval_table_v("alpha4", eta, chi_pn))
        alpha5 = expand_param(IMRPhenomD.eval_table_v("alpha5", eta, chi_pn))

        if f_ref is None:
            reference_gauge_base = torch.zeros_like(f_eval)
        else:
            slope_peak = expand_param(IMRPhenomD.compute_slope_peak(m1, m2, s1, s2))
            reference_gauge_base = slope_peak * m_sec_eval * fref_eval

        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        h_plus_total = torch.zeros_like(f_eval, dtype=complex_dtype)
        h_cross_total = torch.zeros_like(h_plus_total)
        mode_dict: Dict[Tuple[int, int], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
        use_eq14 = isinstance(self.base_22, IMRPhenomD)

        for l, m in active_modes:
            hm_fr_lm, hm_fd_lm = hm_qnm[(l, m)]
            hm_fr_lm = expand_param(hm_fr_lm)
            hm_fd_lm = expand_param(hm_fd_lm)
            if hm_mapping_transitions == "london_2018":
                f0_amp = 0.018 * hm_fr_lm / hm_fr22
                f0_phi = 0.014 * hm_fr_lm / hm_fr22
            else:
                f0_amp = 0.014 * hm_fr_lm / hm_fr22
                f0_phi = 0.018 * hm_fr_lm / hm_fr22
            f_map_amp = frequency_mapping_amplitude(
                mf_eval, f0_amp, hm_fr_lm, hm_fr22, m
            )
            f_map_amp_safe = torch.clamp(f_map_amp, min=1.0e-8)
            amp_22_mapped = amplitude_22(f_map_amp_safe)

            # Eq. (4)'s amplitude factor is evaluated at both the original
            # mode frequency and the mapped 22 frequency.  This is an
            # amplitude-only ratio, so complex PN corrections enter through
            # their absolute value.
            if (l, m) == (2, 2):
                ratio = torch.ones_like(f_eval)
            else:
                x_safe = torch.clamp(mf_eval, min=1.0e-8)
                ga_safe = f_map_amp_safe
                v_ga = torch.pow(2.0 * PI * ga_safe / float(m), 1.0 / 3.0)
                v_x = torch.pow(2.0 * PI * x_safe / float(m), 1.0 / 3.0)
                hhat_ga = _production_hhat(l, m, v_ga, delta_eval, eta_eval, chi_s_eval, chi_a_eval)
                hhat_x = _production_hhat(l, m, v_x, delta_eval, eta_eval, chi_s_eval, chi_a_eval)
                hlm_ga = hhat_ga * v_ga.pow(-3.5)
                hlm_x = hhat_x * v_x.pow(-3.5)
                h22_ga = -torch.pow(2.0 * PI * ga_safe / 2.0, 1.0 / 3.0).pow(-3.5)
                mode_at_22_x = 2.0 * x_safe / float(m)
                v_22_x = torch.pow(2.0 * PI * mode_at_22_x / float(m), 1.0 / 3.0)
                h22_at_x = _production_hhat(l, m, torch.pow(2.0 * PI * mode_at_22_x / float(m), 1.0 / 3.0), delta_eval, eta_eval, chi_s_eval, chi_a_eval) * v_22_x.pow(-3.5)
                tiny = torch.finfo(dtype).tiny
                nonzero = torch.abs(h22_at_x) > tiny
                safe_h22_at_x = torch.where(nonzero, h22_at_x, torch.ones_like(h22_at_x))
                ratio_raw = torch.abs((hlm_ga / h22_ga) * (hlm_x / safe_h22_at_x))
                ratio = torch.where(nonzero, ratio_raw, torch.zeros_like(ratio_raw))
                # The London Eq. (4) ratio is magnitude-only.  Restore the
                # odd-m body-exchange sign without changing the positive
                # mass-ordering branch used by the public HM cases.  At
                # equal masses the antisymmetric spin controls that sign.
                if m % 2:
                    sign_source = torch.where(
                        torch.abs(delta_eval) > tiny, delta_eval, chi_a_eval
                    )
                    mode_factor = torch.sgn(sign_source)
                    mode_factor = torch.where(
                        torch.abs(hhat_ga) > tiny,
                        mode_factor,
                        torch.zeros_like(mode_factor),
                    )
                else:
                    mode_factor = torch.ones_like(ratio)

            def carrier_lm(y: torch.Tensor) -> torch.Tensor:
                z = carrier_positive(y)
                if (l, m) == (2, 2) or not use_eq14:
                    return z
                return z + _production_eq14_delta_c1(
                    y,
                    eta_eval,
                    chi_pn_eval,
                    hm_fr22,
                    hm_fr_lm,
                    hm_fd22,
                    hm_fd_lm,
                    alpha4,
                    alpha5,
                    base_fr22=base_fr22_eval,
                    base_fd22=base_fd22_eval,
                )

            mapped_phase = _production_mapped_phase_c1(
                mf_eval,
                f0_phi,
                hm_fr_lm,
                hm_fr22,
                m,
                l,
                carrier_lm,
            )
            # The D reference-frequency gauge is defined for the (2,2)
            # carrier.  Harmonic m carries the relative factor m/2 - 1 in
            # the mapped HM phase contract.
            reference_gauge = (float(m) / 2.0 - 1.0) * reference_gauge_base
            physical_phase = -mapped_phase + reference_gauge + float(m) * phic_eval
            phase_factor = torch.complex(torch.cos(physical_phase), torch.sin(physical_phase))
            if (l, m) == (2, 2):
                mode_factor = torch.ones_like(ratio)
            h_mode = amp_22_mapped * ratio * mode_factor * phase_factor
            h_mode = torch.where(f_eval > 0.0, h_mode, torch.zeros_like(h_mode))

            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi0_eval)
            hp_lm = d_plus * h_mode
            hc_lm = d_cross * h_mode
            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm
            if return_modes:
                mode_dict[(l, m)] = (hp_lm, hc_lm)
            elif return_unprojected_modes:
                mode_dict[(l, m)] = h_mode

        # The time shift is applied to the complete complex positive-mode sum,
        # matching the public D convention while retaining full complex scale.
        if not torch.compiler.is_compiling() and not torch.any(t_c != 0):
            pass
        else:
            angle = -2.0 * PI * f_eval * tc_eval
            shift = torch.complex(torch.cos(angle), torch.sin(angle))
            h_plus_total = h_plus_total * shift
            h_cross_total = h_cross_total * shift
            if return_modes or return_unprojected_modes:
                for lm, value in mode_dict.items():
                    if isinstance(value, tuple):
                        mode_dict[lm] = (value[0] * shift, value[1] * shift)
                    else:
                        mode_dict[lm] = value * shift

        valid = f_eval > 0.0
        if f_lower is not None:
            valid = valid & (f_eval >= float(f_lower))
        h_plus_total = torch.where(valid, h_plus_total, torch.zeros_like(h_plus_total))
        h_cross_total = torch.where(valid, h_cross_total, torch.zeros_like(h_cross_total))

        if return_modes or return_unprojected_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



# Re-export clean-room calibrated IMRPhenomXHM
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM

__all__ = [
    "IMRPhenomHM",
    "IMRPhenomXHM",
    "frequency_mapping_amplitude",
    "frequency_mapping_phase",
    "h_hat_multipole",
    "_swsh_projection_coefficients",
    "_swsh_projection_coefficients_cartesian",
    "QNM_FITS",
]
