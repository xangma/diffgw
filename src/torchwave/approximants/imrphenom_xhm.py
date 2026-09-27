"""Clean-room implementation of IMRPhenomXHM calibrated higher-order multipolar approximant.

Literature citations:
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
  "Multimode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries: IMRPhenomXHM"
- Pratten et al. (2020), Phys. Rev. D 101, 024029 (arXiv:2001.10911)
  "Setting the cornerstone for next-generation models of gravitational reactions: IMRPhenomXAS"
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
  "Quasinormal modes of Kerr black holes: from astrophysics to string theory"

The fresh published-equation checks and remaining calibration gaps are recorded
in docs/reviews/2026-09-12-clean-room/pass-2-higher-modes.md. Inherited fit
assembly and phase conventions are not certified by those checks.
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
)
from torchwave.approximants.imrphenom_xas import (
    IMRPhenomXAS,
    XASParameterFits,
    safe_sqrt,
)
from torchwave.approximants.xhm_fits import XHMParameterFits, _nonnegative_sqrt
from torchwave.approximants.xhm_native_fits import (
    eval_native_ringdown_params,
    eval_native_phase_params,
)
from torchwave.approximants.imrphenom_hm import (
    _swsh_projection_coefficients_cartesian,
)
from torchwave.approximants.polynomial_interval import polynomial_has_nonpositive
from torchwave.approximants.xhm_inspiral_phase_clean import rescale_carrier_phase


# Pre-inverted 3x3 matrix for inspiral collocation points
# Collocation points at 0.5 f_Ins, 0.75 f_Ins, 1.0 f_Ins
# Powers: 7/3, 8/3, 3
_INSP_MAT = torch.tensor([
    [0.5 ** (7.0 / 3.0), 0.5 ** (8.0 / 3.0), 0.5 ** 3.0],
    [0.75 ** (7.0 / 3.0), 0.75 ** (8.0 / 3.0), 0.75 ** 3.0],
    [1.0, 1.0, 1.0],
], dtype=torch.float64)
_INSP_MAT_INV = torch.linalg.inv(_INSP_MAT)


def _published_parameter_amplitude_veto(
    mode: Tuple[int, int],
    mass_ratio: torch.Tensor,
    eta: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
) -> torch.Tensor:
    """Remove both inner amplitude nodes in the unconditional paper regions.

    García-Quirós et al., arXiv:2001.10914v2, Sec. V.A.3 and Table V.
    Absolute-amplitude vetoes and the conditional q > 40 rows require the
    separately specified, ordered checks and are not implemented here.
    """
    return (
        ((mode == (2, 1)) & (eta < 0.23) & (chi1 > 0.7) & (chi2 < -0.5))
        | ((mode == (3, 2)) & (mass_ratio > 2.5) & (chi1 < -0.6) & (chi2 > 0.0))
        | ((mode == (3, 2)) & (chi1 < -0.9) & (chi2 < -0.9))
        | ((mode in ((3, 3), (4, 4))) & (mass_ratio > 7.0) & (chi1 > 0.95))
    )


def _ordered_amplitude_vetoes(
    mode: Tuple[int, int],
    mass_ratio: torch.Tensor,
    eta: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
    h_ins: torch.Tensor,
    h_int1: torch.Tensor,
    h_int2: torch.Tensor,
    h_rd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply source-supported ordered intermediate-amplitude vetoes.

    Table VI's numeric amplitude thresholds are deliberately not activated:
    the paper does not resolve raw ``A`` versus ``A/A0`` threshold units.
    The monotonicity check requires all values in a common convention with
    frequency-independent normalization.  ``alive1`` and ``alive2`` identify
    surviving inner values; the third result is reserved for a future,
    explicitly normalized threshold implementation.
    """
    alive1 = torch.ones_like(h_ins, dtype=torch.bool)
    alive2 = torch.ones_like(h_ins, dtype=torch.bool)
    remove_derivatives = torch.zeros_like(h_ins, dtype=torch.bool)

    # The paper does not explicitly identify the sequence used in its
    # monotonicity check.  We use the full four-point
    # sequence here as an explicit implementation choice; all four values
    # are in the common A/A0 convention at the production call site.
    both_alive = alive1 & alive2
    monotonic = ((h_ins <= h_int1) & (h_int1 <= h_int2) & (h_int2 <= h_rd)) | (
        (h_ins >= h_int1) & (h_int1 >= h_int2) & (h_int2 >= h_rd)
    )
    remove_middle = both_alive & ~monotonic & ((mode == (2, 1)) | (mode == (3, 2)))
    alive2 = alive2 & ~remove_middle

    # The bad-a_lambda and unconditional parameter rows are applied in the
    # published order, before the conditional high-q rows.
    if mode in ((3, 3), (4, 4)):
        bad_lambda = (mass_ratio > 7.0) & (chi1 > 0.95)
        alive1 = torch.where(bad_lambda, torch.zeros_like(alive1), alive1)
        alive2 = torch.where(bad_lambda, torch.zeros_like(alive2), alive2)

    unconditional = _published_parameter_amplitude_veto(mode, mass_ratio, eta, chi1, chi2)
    alive1 = torch.where(unconditional, torch.zeros_like(alive1), alive1)
    alive2 = torch.where(unconditional, torch.zeros_like(alive2), alive2)

    conditional = (mass_ratio > 40.0) & (chi1 > 0.9)
    if mode in ((2, 1), (3, 2), (3, 3), (4, 4)):
        alive1 = torch.where(conditional, torch.zeros_like(alive1), alive1)
        alive2 = torch.where(conditional, torch.zeros_like(alive2), alive2)

    return alive1, alive2, remove_derivatives


def _unsqueeze_to(tensor: Union[float, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Unsqueeze tensor along trailing dimensions until it matches target's dimensionality."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, dtype=target.dtype, device=target.device)
    res = tensor
    while res.dim() < target.dim():
        res = res.unsqueeze(-1)
    return res


def _ringdown_cut_frequencies(f_ring, f_damp):
    """Default XHM Eq. (36)/Table 2 amplitude and phase cuts from supplied QNMs."""
    amplitude = {
        "21": 0.75 * f_ring["21"],
        "33": 0.95 * f_ring["33"],
        "32": f_ring["22"],
        "44": 0.90 * f_ring["44"],
    }
    phase = {
        "21": f_ring["21"] - f_damp["21"],
        "33": f_ring["33"] - f_damp["33"],
        "32": f_ring["22"] - 0.5 * f_damp["22"],
        "44": f_ring["44"] - f_damp["44"],
    }
    return amplitude, phase


def _ringdown_amplitude_ratio(frequency, ring, damp, sigma, a_lambda, lambda_fit):
    """XHM Eq. (42) for supplied ringdown parameters, with native guards."""
    width = _unsqueeze_to(damp * sigma, frequency)
    ring = _unsqueeze_to(ring, frequency)
    a_lambda = _unsqueeze_to(a_lambda, frequency)
    lambda_fit = _unsqueeze_to(lambda_fit, frequency)
    offset = frequency - ring
    denominator = offset**2 + width**2
    exponent = torch.clamp(offset * lambda_fit / width, min=-80.0, max=80.0)
    frequency_power = torch.pow(torch.clamp(frequency, min=1e-8), 1.0 / 12.0)
    return (1.0 / frequency_power) * (torch.abs(a_lambda) * width / torch.clamp(denominator, min=1e-12)) * torch.exp(-exponent)


def _nonmix_ringdown_phase_shifts(join, phase_at_join, slope_at_join, ring, damp, a2, a_lambda):
    """XHM Eq. (43)--(44) C1 attachment for supplied join data."""
    lorentzian = damp / (damp**2 + (join - ring)**2)
    time_shift = slope_at_join - (a2 * ring**2 / join**2 + a_lambda * lorentzian)
    base_phase = -a2 * ring**2 / join + a_lambda * torch.atan((join - ring) / damp) + time_shift * join
    phase_shift = phase_at_join - base_phase
    return time_shift, phase_shift


def _nonmix_ringdown_phase(frequency, ring, damp, a2, a_lambda, time_shift, phase_shift):
    """XHM Eq. (44) for non-mixing modes with supplied C1 shifts."""
    ring = _unsqueeze_to(ring, frequency)
    damp = _unsqueeze_to(damp, frequency)
    a2 = _unsqueeze_to(a2, frequency)
    a_lambda = _unsqueeze_to(a_lambda, frequency)
    time_shift = _unsqueeze_to(time_shift, frequency)
    phase_shift = _unsqueeze_to(phase_shift, frequency)
    return -a2 * ring**2 / frequency + a_lambda * torch.atan((frequency - ring) / damp) + time_shift * frequency + phase_shift


def _spheroidal_32_ringdown_phase_derivative(frequency, ring, damp, a2, a4, a_lambda, time_shift):
    """XHM Eq. (46) with caller-supplied (3,2) coefficients and QNM pair."""
    ring, damp, a2, a4, a_lambda, time_shift = (
        _unsqueeze_to(value, frequency) for value in (ring, damp, a2, a4, a_lambda, time_shift)
    )
    lorentzian = damp / (damp**2 + (frequency - ring)**2)
    return a2 / frequency**2 + a4 / frequency**4 + a_lambda * lorentzian + time_shift


def _spheroidal_32_ringdown_phase(frequency, ring, damp, a2, a4, a_lambda, time_shift, phase_shift):
    """XHM Eq. (47) primitive with caller-supplied phase offset."""
    ring, damp, a2, a4, a_lambda, time_shift, phase_shift = (
        _unsqueeze_to(value, frequency)
        for value in (ring, damp, a2, a4, a_lambda, time_shift, phase_shift)
    )
    return (
        -a2 / frequency
        - a4 / (3.0 * frequency**3)
        + a_lambda * torch.atan((frequency - ring) / damp)
        + time_shift * frequency
        + phase_shift
    )


def _spheroidal_32_alignment(
    ring22, damp22, ring32, damp32, a2, a4, a_lambda,
    base_time, carrier_slope, carrier_phase, delta_t, delta_phi,
):
    """XHM Eqs. (49)-(50) offsets at the author notebook's reference nodes."""
    f0 = ring22 + damp22
    f1 = ring22
    unaligned_slope = _spheroidal_32_ringdown_phase_derivative(
        f0, ring32, damp32, a2, a4, a_lambda, base_time,
    )
    final_time = base_time + carrier_slope + delta_t - unaligned_slope
    unaligned_phase = _spheroidal_32_ringdown_phase(
        f1, ring32, damp32, a2, a4, a_lambda, final_time, 0.0,
    )
    final_phase = carrier_phase + delta_phi - unaligned_phase
    return final_time, final_phase


def _value_and_derivative(function, frequency: torch.Tensor):
    """Value and pointwise frequency derivative using forward-mode JVP, retaining parameter gradients."""
    is_scalar = frequency.dim() == 0
    freq_input = frequency.unsqueeze(0) if is_scalar else frequency
    tangent = torch.ones_like(freq_input)
    val, deriv = torch.func.jvp(function, (freq_input,), (tangent,))
    if is_scalar:
        return val.squeeze(0), deriv.squeeze(0)
    return val, deriv


def _allow_internal_frequency_derivatives(function):
    """Passthrough preserving function signature without forcing inference-mode tensor cloning."""
    return function


def _pn_phase_lambda_coeff(l, m, eta, delta):
    if (l, m) == (2, 1):
        coefficient = torch.where(delta != 0, torch.full_like(delta, -0.5 * (1.0 + math.log(16.0))), torch.zeros_like(delta))
    elif (l, m) == (3, 3):
        coefficient = torch.where(delta != 0, torch.full_like(delta, -21.0 / 5.0 + 6.0 * math.log(1.5)), torch.zeros_like(delta))
    elif (l, m) == (3, 2):
        coefficient = (3.0 - 66.0 * eta / 5.0) / (1.0 - 3.0 * eta)
    elif (l, m) == (4, 4):
        coefficient = (eta * (3579.0 - 2880.0 * math.log(2.0)) - 1008.0 + 960.0 * math.log(2.0)) / (120.0 * (3.0 * eta - 1.0))
    else:
        raise ValueError(f"Unsupported PN phase mode {(l, m)}")
    return coefficient * (2.0 * PI / float(m))


def _pn_phase_lambda(l, m, frequency, eta, delta):  # noqa: E741
    """Frequency-dependent part of Eq. (4.9), expanded through linear f.

    The constant PN argument is absorbed in the separate phase convention.
    These coefficients follow by dividing the first imaginary coefficient in
    Eqs. (E10), (E11), (E12), (E14) by the leading real coefficient, after
    removing the constant complex prefactor. The odd-mode imaginary terms
    vanish identically at equal mass at the PN order available here.
    """
    eta = _unsqueeze_to(eta, frequency)
    delta = _unsqueeze_to(delta, frequency)
    coeff = _pn_phase_lambda_coeff(l, m, eta, delta)
    return coeff * frequency


def _spherical_32_from_spheroidal_cartesian(h22_re, h22_im, h32_s, r_re, r_im, c_re, c_im):
    """Eliminate h22,S from the two simultaneous equations in Eq. (A3) using Cartesian real components."""
    mixed_re = (r_re * h22_re - r_im * h22_im) + c_re * h32_s
    mixed_im = (r_re * h22_im + r_im * h22_re) + c_im * h32_s
    return mixed_re, mixed_im


def _spherical_32_from_spheroidal(h22, h32_s, alpha222, alpha232, alpha322, alpha332):
    """Eliminate h22,S from the two simultaneous equations in Eq. (A3)."""
    ratio = alpha322 / alpha222
    return ratio * h22 + (alpha332 - ratio * alpha232) * h32_s


def _nonmix_intermediate_phase_nodes(eta, f_ins, f_ring):
    """Default [C1,C2,E2,E3,E5] nodes of XHM Table 7."""
    start = (1.0 + 0.001 * (0.25 / torch.clamp(eta, min=1e-6) - 1.0)) * f_ins
    return torch.stack((
        start,
        ((math.sqrt(3.0) + 2.0) * start - (math.sqrt(3.0) - 2.0) * f_ring) / 4.0,
        (f_ring + 3.0 * start) / 4.0,
        (f_ring + start) / 2.0,
        (7.0 * f_ring + start) / 8.0,
    ), dim=-1)


def _nonmix_intermediate_phase_coefficients(nodes, targets, ring, damp):
    """Eq. (39): non-mixing five-point phase-derivative collocation solve."""
    ring_u = ring.unsqueeze(-1) if ring.dim() > 0 else ring
    damp_u = damp.unsqueeze(-1) if damp.dim() > 0 else damp
    matrix = torch.stack((
        damp_u / (damp_u**2 + (nodes - ring_u)**2),
        torch.ones_like(nodes),
        1.0 / nodes,
        1.0 / nodes**2,
        1.0 / nodes**4,
    ), dim=-1)
    return torch.linalg.solve(matrix, targets)


def _nonmix_intermediate_phase_derivative(frequency, coefficients, ring, damp):
    """Eq. (38), evaluated at positive dimensionless frequency."""
    lam, a0, a1, a2, a4 = (_unsqueeze_to(coefficients[..., i], frequency) for i in range(5))
    ring, damp = _unsqueeze_to(ring, frequency), _unsqueeze_to(damp, frequency)
    return lam * damp / (damp**2 + (frequency - ring)**2) + a0 + a1 / frequency + a2 / frequency**2 + a4 / frequency**4


def _nonmix_intermediate_phase(frequency, coefficients, ring, damp):
    """Antiderivative of Eq. (38), with zero additive constant."""
    lam, a0, a1, a2, a4 = (_unsqueeze_to(coefficients[..., i], frequency) for i in range(5))
    ring, damp = _unsqueeze_to(ring, frequency), _unsqueeze_to(damp, frequency)
    return (lam * torch.atan((frequency - ring) / damp) + a0 * frequency
            + a1 * torch.log(torch.clamp(frequency, min=1e-12)) - a2 / frequency
            - a4 / (3.0 * frequency**3))


def _intermediate_32_phase_nodes(eta, f_ins, ring22, damp22):
    """First four Table-7 nodes, ending at the distinct (3,2) frequency."""
    start = (1.0 + 0.001 * (0.25 / eta - 1.0)) * f_ins
    end = ring22 - 0.5 * damp22
    return torch.stack((
        start,
        ((math.sqrt(3.0) + 2.0) * start - (math.sqrt(3.0) - 2.0) * end) / 4.0,
        (end + 3.0 * start) / 4.0,
        (end + start) / 2.0,
    ), dim=-1)


def _intermediate_32_phase_coefficients(nodes, targets, join, dphi_join, d2phi_join, ring, damp):
    """Eq. (40): four phase-derivative targets and a C2 ringdown join."""

    def row(f):
        lorentz = damp / (damp**2 + (f - ring)**2)
        return torch.stack((lorentz, torch.ones_like(f), 1 / f, 1 / f**2, 1 / f**3, 1 / f**4), dim=-1)

    dlorentz = -2 * damp * (join - ring) / (damp**2 + (join - ring)**2)**2
    derivative_row = torch.stack((dlorentz, torch.zeros_like(join), -1 / join**2,
                                  -2 / join**3, -3 / join**4, -4 / join**5), dim=-1)
    matrix = torch.stack((*[row(nodes[..., i]) for i in range(4)], row(join), derivative_row), dim=-2)
    rhs = torch.cat((targets, dphi_join[..., None], d2phi_join[..., None]), dim=-1)
    return torch.linalg.solve(matrix, rhs)


def _intermediate_32_phase_derivative(frequency, coefficients, ring, damp):
    """Eq. (38), including the nonzero inverse-cubic (3,2) term."""
    lam, a0, a1, a2, a3, a4 = (_unsqueeze_to(coefficients[..., i], frequency) for i in range(6))
    ring, damp = _unsqueeze_to(ring, frequency), _unsqueeze_to(damp, frequency)
    return (lam * damp / (damp**2 + (frequency - ring)**2) + a0 + a1 / frequency
            + a2 / frequency**2 + a3 / frequency**3 + a4 / frequency**4)


def _intermediate_32_phase(frequency, coefficients, ring, damp):
    """Antiderivative of Eq. (38), with zero additive constant."""
    lam, a0, a1, a2, a3, a4 = (_unsqueeze_to(coefficients[..., i], frequency) for i in range(6))
    ring, damp = _unsqueeze_to(ring, frequency), _unsqueeze_to(damp, frequency)
    return (lam * torch.atan((frequency - ring) / damp) + a0 * frequency
            + a1 * torch.log(torch.clamp(frequency, min=1e-12)) - a2 / frequency
            - a3 / (2.0 * frequency**2) - a4 / (3.0 * frequency**3))


def _carrier_h22_ratio_cartesian(strain, phase, phase_reference, a0):
    """Put the physical carrier in the H_lm normalization of Eq. (4.6) in Cartesian real components."""
    amp = torch.abs(strain / a0)
    dphi = phase - phase_reference
    return amp * torch.cos(dphi), -amp * torch.sin(dphi)


def _carrier_h22_ratio(strain, phase, phase_reference, a0):
    """Put the physical carrier in the H_lm normalization of Eq. (4.6)."""
    amp = torch.abs(strain / a0)
    dphi = phase - phase_reference
    return torch.complex(amp * torch.cos(dphi), -amp * torch.sin(dphi))


_DEFAULT_CARRIER_H22_RATIO = _carrier_h22_ratio
_DEFAULT_SPHERICAL_32_FROM_SPHEROIDAL = _spherical_32_from_spheroidal


def _eval_f_isco(chi: torch.Tensor) -> torch.Tensor:
    """Kerr ISCO frequency for dimensionless spin chi."""
    chi_clamped = torch.clamp(chi, min=-0.9999, max=0.9999)
    one_m_chi2 = torch.clamp(1.0 - chi_clamped**2, min=1e-12)
    z1 = 1.0 + torch.pow(one_m_chi2, 1.0 / 3.0) * (
        torch.pow(1.0 + chi_clamped, 1.0 / 3.0) + torch.pow(1.0 - chi_clamped, 1.0 / 3.0)
    )
    z2 = safe_sqrt(3.0 * chi_clamped**2 + z1**2)
    diff = torch.clamp((3.0 - z1) * (3.0 + z1 + 2.0 * z2), min=0.0)
    r_isco = 3.0 + z2 - torch.sign(chi_clamped) * safe_sqrt(diff)
    r_isco_clamped = torch.clamp(r_isco, min=1.0)
    return 1.0 / (PI * (r_isco_clamped**1.5 + chi_clamped))


def _pn_amplitude_h21(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Magnitude of the expanded PN H_21 (arXiv:2001.10914v2, Eq. E10)."""
    c1 = (1.0 / 3.0) * math.sqrt(2.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t1_re = v * delta
    t1_im = torch.zeros_like(v)

    t2_re = v2 * (-1.5 * delta * chi_s - 1.5 * chi_a)
    t2_im = torch.zeros_like(v)

    t3_re = v3 * ((117.0 / 56.0) * delta * eta + (335.0 / 672.0) * delta)
    t3_im = torch.zeros_like(v)

    t4_re = v4 * (
        -(965.0 / 336.0) * delta * eta * chi_s
        + (3427.0 / 1344.0) * delta * chi_s
        - PI * delta
        - (2101.0 / 336.0) * eta * chi_a
        + (3427.0 / 1344.0) * chi_a
    )
    t4_im = v4 * (-0.5 * delta - 0.5 * delta * math.log(16.0))

    t5_re = v5 * (
        (21365.0 / 8064.0) * delta * (eta**2)
        + 10.0 * delta * eta * (chi_a**2)
        + (39.0 / 8.0) * delta * eta * (chi_s**2)
        - (36529.0 / 12544.0) * delta * eta
        - (307.0 / 32.0) * delta * (chi_a**2)
        - (307.0 / 32.0) * delta * (chi_s**2)
        + 3.0 * PI * delta * chi_s
        - (964357.0 / 8128512.0) * delta
        + (213.0 / 4.0) * eta * chi_a * chi_s
        - (307.0 / 16.0) * chi_a * chi_s
        + 3.0 * PI * chi_a
    )
    t5_im = torch.zeros_like(v)

    t6_re = v6 * (
        -(547.0 / 768.0) * delta * (eta**2) * chi_s
        - 15.0 * delta * eta * (chi_a**2) * chi_s
        - (3.0 / 16.0) * delta * eta * (chi_s**3)
        - (7049629.0 / 225792.0) * delta * eta * chi_s
        + (417.0 / 112.0) * PI * delta * eta
        + (729.0 / 64.0) * delta * (chi_a**2) * chi_s
        + (243.0 / 64.0) * delta * (chi_s**3)
        + (143063173.0 / 5419008.0) * delta * chi_s
        - (2455.0 / 1344.0) * PI * delta
        + (42617.0 / 1792.0) * (eta**2) * chi_a
        - 15.0 * eta * (chi_a**3)
        - (489.0 / 16.0) * eta * chi_a * (chi_s**2)
        - (22758317.0 / 225792.0) * eta * chi_a
        + (243.0 / 64.0) * (chi_a**3)
        + (729.0 / 64.0) * chi_a * (chi_s**2)
        + (143063173.0 / 5419008.0) * chi_a
    )
    t6_im = v6 * (
        -(1489.0 / 112.0) * delta * eta
        - (89.0 / 28.0) * delta * eta * math.log(2.0)
        - (335.0 / 1344.0) * delta
        - (335.0 / 336.0) * delta * math.log(2.0)
    )

    t_re = t1_re + t2_re + t3_re + t4_re + t5_re + t6_re
    t_im = t1_im + t2_im + t3_im + t4_im + t5_im + t6_im

    h_re = -c1 * t_im
    h_im = c1 * t_re
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h33(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Magnitude of the expanded PN H_33 (arXiv:2001.10914v2, Eq. E11)."""
    c1 = -(3.0 / 4.0) * math.sqrt(5.0 / 7.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t1_re = v * delta
    t1_im = torch.zeros_like(v)

    t3_re = v3 * delta * ((27.0 / 8.0) * eta - 1945.0 / 672.0)
    t3_im = torch.zeros_like(v)

    t4_re = v4 * (
        -(2.0 / 3.0) * delta * eta * chi_s
        + (65.0 / 24.0) * delta * chi_s
        + PI * delta
        - (28.0 / 3.0) * eta * chi_a
        + (65.0 / 24.0) * chi_a
    )
    t4_im = v4 * (-(21.0 / 5.0) * delta + 6.0 * delta * math.log(1.5))

    t5_re = v5 * (
        (420389.0 / 63360.0) * delta * (eta**2)
        + 10.0 * delta * eta * (chi_a**2)
        + (1.0 / 8.0) * delta * eta * (chi_s**2)
        - (11758073.0 / 887040.0) * delta * eta
        - (81.0 / 32.0) * delta * (chi_a**2)
        - (81.0 / 32.0) * delta * (chi_s**2)
        - (1077664867.0 / 447068160.0) * delta
        + (81.0 / 4.0) * eta * chi_a * chi_s
        - (81.0 / 16.0) * chi_a * chi_s
    )
    t5_im = torch.zeros_like(v)

    t6_re = v6 * (
        -(67.0 / 24.0) * delta * (eta**2) * chi_s
        - (58745.0 / 4032.0) * delta * eta * chi_s
        + (131.0 / 16.0) * PI * delta * eta
        + (163021.0 / 16128.0) * delta * chi_s
        - (5675.0 / 1344.0) * PI * delta
        - (137.0 / 24.0) * (eta**2) * chi_a
        - (148501.0 / 4032.0) * eta * chi_a
        + (163021.0 / 16128.0) * chi_a
    )
    t6_im = v6 * (
        -(440957.0 / 9720.0) * delta * eta
        + (69.0 / 4.0) * delta * eta * math.log(1.5)
        + (389.0 / 32.0) * delta
        - (1945.0 / 112.0) * delta * math.log(1.5)
    )

    t_re = t1_re + t3_re + t4_re + t5_re + t6_re
    t_im = t1_im + t3_im + t4_im + t5_im + t6_im

    h_re = -c1 * t_im
    h_im = c1 * t_re
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h32(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Magnitude of the expanded PN H_32 (arXiv:2001.10914v2, Eq. E12)."""
    c1 = (1.0 / 3.0) * math.sqrt(5.0 / 7.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t2_re = v2 * (1.0 - 3.0 * eta)
    t3_re = v3 * (4.0 * eta * chi_s)
    t4_re = v4 * (
        -(589.0 / 72.0) * (eta**2)
        + (12325.0 / 2016.0) * eta
        - (10471.0 / 10080.0)
    )

    t5_re = v5 * (
        eta * ((113.0 / 8.0) * delta * chi_a + (1081.0 / 84.0) * chi_s)
        + (1.0 / 24.0) * (-113.0 * delta * chi_a - 113.0 * chi_s)
        - 15.0 * (eta**2) * chi_s
    )
    t5_im = v5 * (eta * (-66.0 / 5.0) + (72.0 / 24.0))

    t6_re = v6 * (
        eta * (
            -(1633.0 / 48.0) * delta * chi_a * chi_s
            - (563.0 / 32.0) * (chi_a**2)
            - (2549.0 / 96.0) * (chi_s**2)
            + 8.0 * PI * chi_s
            - (8689883.0 / 149022720.0)
        )
        + (81.0 / 16.0) * delta * chi_a * chi_s
        + (837223.0 / 63360.0) * (eta**3)
        + (eta**2) * (30.0 * (chi_a**2) + (313.0 / 24.0) * (chi_s**2) - 78584047.0 / 2661120.0)
        + (81.0 / 32.0) * (chi_a**2)
        + (81.0 / 32.0) * (chi_s**2)
        + (824173699.0 / 447068160.0)
    )
    t6_im = torch.zeros_like(v)

    t_re = t2_re + t3_re + t4_re + t5_re + t6_re
    t_im = t5_im + t6_im

    h_re = c1 * t_re
    h_im = c1 * t_im
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h44(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Magnitude of the expanded PN H_44 (arXiv:2001.10914v2, Eq. E14)."""
    c1 = (4.0 / 9.0) * math.sqrt(10.0 / 7.0)
    v2 = v * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v5 * v

    t2_re = v2 * (3.0 * eta - 1.0)
    t4_re = v4 * (
        (1063.0 / 88.0) * (eta**2)
        - (128221.0 / 7392.0) * eta
        + (158383.0 / 36960.0)
    )

    term5_inner_re = (
        -eta * (1695.0 * delta * chi_a + 2075.0 * chi_s)
        + 565.0 * delta * chi_a
        + 1140.0 * (eta**2) * chi_s
        + 565.0 * chi_s
    )
    term5_inner_im = -eta * (-3579.0 + 2880.0 * math.log(2.0)) - 1008.0 + 960.0 * math.log(2.0)
    t5_re = v5 * (PI * (2.0 - 6.0 * eta) + (1.0 / 120.0) * term5_inner_re)
    t5_im = v5 * ((1.0 / 120.0) * term5_inner_im)

    t6_re = v6 * (
        eta * (
            (243.0 / 16.0) * delta * chi_a * chi_s
            + (563.0 / 32.0) * (chi_a**2)
            + (247.0 / 32.0) * (chi_s**2)
            - (22580029007.0 / 880588800.0)
        )
        - (81.0 / 16.0) * delta * chi_a * chi_s
        - (7606537.0 / 274560.0) * (eta**3)
        + (eta**2) * (-30.0 * (chi_a**2) - (3.0 / 8.0) * (chi_s**2) + 901461137.0 / 11531520.0)
        - (81.0 / 32.0) * (chi_a**2)
        - (81.0 / 32.0) * (chi_s**2)
        + (7888301437.0 / 29059430400.0)
    )
    t6_im = torch.zeros_like(v)

    t_re = t2_re + t4_re + t5_re + t6_re
    t_im = t5_im + t6_im

    h_re = c1 * t_re
    h_im = c1 * t_im
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_ratio(
    l: int,  # noqa: E741
    m: int,
    f: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Return |A_lm^PN|/A_22^0 using arXiv:2001.10914v2 Eq. (E8).

    The SPA envelope is v_m^(-7/2), with v_m=(2*pi*f/m)^(1/3).
    Dividing by the quadrupole envelope at the same f gives (m/2)^(7/6).
    """
    m_flt = float(m)
    v = torch.pow(torch.clamp((2.0 * PI * f) / m_flt, min=1e-12), 1.0 / 3.0)
    factor = (m_flt / 2.0) ** (7.0 / 6.0)
    eta_u = _unsqueeze_to(eta, v)
    delta_u = _unsqueeze_to(delta, v)
    chi_s_u = _unsqueeze_to(chi_s, v)
    chi_a_u = _unsqueeze_to(chi_a, v)
    if l == 2 and m == 1:
        h_val = _pn_amplitude_h21(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 3 and m == 3:
        h_val = _pn_amplitude_h33(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 3 and m == 2:
        h_val = _pn_amplitude_h32(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 4 and m == 4:
        h_val = _pn_amplitude_h44(v, eta_u, delta_u, chi_s_u, chi_a_u)
    else:
        raise ValueError(f"Mode ({l}, {m}) not supported.")
    return factor * h_val


class IMRPhenomXHM(nn.Module):
    """IMRPhenomXHM Calibrated Higher-Harmonic IMR Frequency-Domain Approximant.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)}.
    Literature:
    - García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
    - Pratten et al. (2020), Phys. Rev. D 101, 024029 (arXiv:2001.10911)

    Uses the carrier-based inspiral and value-only phase connection.
    The unvalidated experimental phase-connection option is unavailable.
    """

    SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
    supported_modes = SUPPORTED_MODES

    @property
    def get_supported_modes(self):
        return self.SUPPORTED_MODES

    SUPPORTED_CALIBRATIONS = ("sovereign", "torchwave_nr", "native", "torchwave")

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        carrier: str = "XAS",
        base_22: Optional[nn.Module] = None,
        *,
        experimental_phase_connections: bool = False,
        calibration: str = "sovereign",
    ):
        self.carrier = carrier
        super().__init__()
        self.calibration = calibration.lower()
        if self.calibration not in self.SUPPORTED_CALIBRATIONS:
            raise ValueError(
                f"Unsupported calibration '{calibration}'. Sovereign calibration is the sole supported standard."
            )
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        self.base_22 = base_22 if base_22 is not None else IMRPhenomXAS()
        if experimental_phase_connections:
            raise ValueError("experimental_phase_connections=True is an unvalidated prototype and is not available in releases.")

    @_allow_internal_frequency_derivatives
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
        phi0: Union[float, torch.Tensor] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        return_unprojected_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
        **kwargs,
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
            phi0: Observer azimuth angle in radians (default 0.0)
            modes: Optional tuple of modes to evaluate
            return_modes: If True, return projected mode polarizations (hp, hc)
            return_unprojected_modes: If True, return unprojected complex mode strains h_lm
        """
        if kwargs.pop("experimental_phase_connections", False):
            raise ValueError("experimental_phase_connections=True is an unvalidated prototype and is not available in releases.")
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1_in = as_tensor(mass1, device=device, dtype=dtype)
        m2_in = as_tensor(mass2, device=device, dtype=dtype)
        s1_in = as_tensor(chi1, device=device, dtype=dtype)
        s2_in = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(kwargs.pop("phic", phic), device=device, dtype=dtype)
        t_c = as_tensor(kwargs.pop("tc", tc), device=device, dtype=dtype)
        phi_0 = as_tensor(kwargs.pop("phi0", phi0), device=device, dtype=dtype)
        final_spin_override = kwargs.pop("_final_spin_override", None)
        if final_spin_override is not None:
            final_spin_override = as_tensor(final_spin_override, device=device, dtype=dtype)

        active_modes = tuple(modes) if modes is not None else self.modes

        # Body exchange canonical ordering (m1 >= m2), breaking equal-mass ties
        # by spin so that Eq. (2.13) also holds for unequal-spin equal masses.
        # Binary relabeling 1 <-> 2: under 180-deg orbital rotation, odd-m modes pick up a factor (-1)^m = -1.
        swap_mask = (m1_in < m2_in) | ((m1_in == m2_in) & (s1_in < s2_in))
        m1 = torch.where(swap_mask, m2_in, m1_in)
        m2 = torch.where(swap_mask, m1_in, m2_in)
        s1 = torch.where(swap_mask, s2_in, s1_in)
        s2 = torch.where(swap_mask, s1_in, s2_in)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta, _, _, S = XHMParameterFits.compute_spin_variables(eta, s1, s2)
        chi_s = 0.5 * (s1 + s2)
        chi_a = 0.5 * (s1 - s2)
        chi_eff = (m1 * s1 + m2 * s2) / total_mass
        m_sec = total_mass * MSUN_SEC

        # Final remnant properties from XAS fits
        chi_f = (
            XASParameterFits.fit_final_spin(eta, s1, s2)
            if final_spin_override is None else final_spin_override
        )
        erad = XASParameterFits.fit_erad(eta, s1, s2)
        m_f = 1.0 - erad

        # ISCO and MECO frequencies for (2,2)
        f_isco_22 = _eval_f_isco(chi_f)
        f_meco_22 = XASParameterFits.fit_f_meco(eta, s1, s2)
        diff_isco_meco = torch.abs(f_isco_22 - f_meco_22)

        # Mode QNMs in geometric units (rescaled by remnant mass M_f)
        f_ring: Dict[str, torch.Tensor] = {}
        f_damp: Dict[str, torch.Tensor] = {}
        for lm_str in ("21", "22", "32", "33", "44"):
            f_ring[lm_str] = getattr(XHMParameterFits, f"qnm_fring_{lm_str}")(chi_f) / m_f
            f_damp[lm_str] = getattr(XHMParameterFits, f"qnm_fdamp_{lm_str}")(chi_f) / m_f

        # Cutting frequencies per mode
        eps_amp = {
            "21": torch.tensor(1.0, device=device, dtype=dtype),
            "33": torch.tensor(1.0, device=device, dtype=dtype),
            "32": f_ring["32"] / f_ring["22"],
            "44": torch.tensor(1.0, device=device, dtype=dtype),
        }
        del_amp = {
            "21": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * (chi_eff**2),
            "33": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * chi_eff,
            "32": (0.75 - 0.235 * torch.abs(chi_eff)) * (f_ring["32"] / f_ring["22"]),
            "44": 0.75 - 0.235 * chi_eff,
        }
        m_vals = {"21": 1, "32": 2, "33": 3, "44": 4}

        f_ins_amp = {
            lm_str: (m_vals[lm_str] / 2.0) * (f_meco_22 * eps_amp[lm_str] + diff_isco_meco * del_amp[lm_str])
            for lm_str in ("21", "32", "33", "44")
        }
        f_rd_amp, f_rd_phi = _ringdown_cut_frequencies(f_ring, f_damp)
        f_ins_phi = {
            lm_str: (m_vals[lm_str] / 2.0) * f_meco_22
            for lm_str in ("21", "32", "33", "44")
        }

        # Amplitude sigma fits
        sigma_amp = {
            "21": XHMParameterFits.amp_sigma_21(eta, s1, s2, S),
            "33": torch.tensor(1.30, device=device, dtype=dtype),
            "32": torch.tensor(1.33, device=device, dtype=dtype),
            "44": torch.tensor(1.33, device=device, dtype=dtype),
        }

        # 22-mode phase parameter for ringdown rescaling (Eq. 6.5)
        alpha2_22 = XHMParameterFits.phase_rd_alpha2(eta, s1, s2, S)

        # Batching alignment
        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
            m_sec_eval = m_sec.unsqueeze(-1)
            eta_eval = eta.unsqueeze(-1)
            dist_eval = dist.unsqueeze(-1)
            inc_eval = inc.unsqueeze(-1)
            phi0_eval = phi_0.unsqueeze(-1)
            phic_eval = phi_c.unsqueeze(-1)
            tc_eval = t_c.unsqueeze(-1)
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_eval = eta
            dist_eval = dist
            inc_eval = inc
            phi0_eval = phi_0
            phic_eval = phi_c
            tc_eval = t_c

        # Geometric frequency grid: Mf
        f_safe = torch.clamp(f_eval, min=1e-8)
        mf_grid = m_sec_eval * f_safe

        # Evaluate carrier (2, 2) mode at face-on (iota=0)
        carrier_state = None
        if hasattr(self.base_22, "build_state"):
            carrier_state = self.base_22.build_state(
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                f_ref=f_ref,
                _final_spin_override=final_spin_override,
                device=device,
                dtype=dtype,
            )
            hp_22_raw, _, phi_22_carrier = carrier_state.eval_waveform(
                f=f_eval,
                distance=dist,
                inclination=0.0,
                phi_c=0.0,
                t_c=0.0,
                return_phase=True,
            )
        else:
            hp_22_raw, _, phi_22_carrier = self.base_22(
                f=f_eval,
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
                _final_spin_override=final_spin_override,
            )
        amp_22_phys = torch.abs(hp_22_raw)

        # Base physical leading quadrupole factor A0_phys
        dist_sec = (dist_eval * MPC_SI) / C_SI
        prefactor = torch.sqrt(5.0 * eta_eval / 24.0) / (PI ** (2.0 / 3.0) * dist_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (f_safe ** (-7.0 / 6.0))
        # Mode normalization factor relates mode amplitude to physical strain
        mode_norm = math.sqrt(16.0 * PI / 5.0)

        def eval_carrier_at(mf_target: torch.Tensor):
            # An explicit singleton frequency axis keeps batch axes distinct.
            target_grid = mf_target.unsqueeze(-1) if mf_target.ndim == m_sec.ndim else mf_target
            target_hz = target_grid / _unsqueeze_to(m_sec, target_grid)
            if carrier_state is not None:
                hp, _, phase = carrier_state.eval_waveform(
                    f=target_hz,
                    distance=m_sec * C_SI / MPC_SI,
                    inclination=0.0,
                    phi_c=0.0,
                    t_c=0.0,
                    return_phase=True,
                )
            else:
                hp, _, phase = self.base_22(
                    f=target_hz, mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                    distance=m_sec * C_SI / MPC_SI, inclination=0.0, phic=0.0, tc=0.0,
                    f_ref=f_ref, phasing_convention=phasing_convention,
                    reference_frame=reference_frame, return_phase=True,
                    _final_spin_override=final_spin_override,
                )
            if mf_target.ndim == m_sec.ndim:
                return hp.squeeze(-1), phase.squeeze(-1)
            return hp, phase

        def eval_carrier_phase_at(mf_target: torch.Tensor):
            target_grid = mf_target.unsqueeze(-1) if mf_target.ndim == m_sec.ndim else mf_target
            target_hz = target_grid / _unsqueeze_to(m_sec, target_grid)
            if carrier_state is not None:
                phase = carrier_state.eval_phase(target_hz)
            else:
                phase = eval_carrier_at(mf_target)[1]
            if mf_target.ndim == m_sec.ndim:
                return phase.squeeze(-1)
            return phase

        def eval_carrier_phi_and_deriv(mf_target: torch.Tensor):
            return _value_and_derivative(eval_carrier_phase_at, mf_target)

        h_plus_re = torch.zeros_like(amp_22_phys)
        h_plus_im = torch.zeros_like(amp_22_phys)
        h_cross_re = torch.zeros_like(amp_22_phys)
        h_cross_im = torch.zeros_like(amp_22_phys)
        mode_dict = {}

        # Store h22 Cartesian components for (2,2) and mode-mixing with (3,2)
        h22_amp = mode_norm * amp_22_phys
        h22_re = h22_amp * torch.cos(phi_22_carrier)
        h22_im = -h22_amp * torch.sin(phi_22_carrier)

        for l, m in active_modes:  # noqa: E741
            lm_str = f"{l}{m}"
            if (l, m) == (2, 2):
                h_mode_re = h22_re
                h_mode_im = h22_im
            else:
                m_flt = float(m)
                f_ins_a = f_ins_amp[lm_str]
                f_rd_a = f_rd_amp[lm_str]
                f_ins_p = f_ins_phi[lm_str]
                f_rd_p = f_rd_phi[lm_str]
                f_ring_lm = f_ring[lm_str]
                f_damp_lm = f_damp[lm_str]
                sigma_lm = sigma_amp[lm_str]

                # -------------------------------------------------------------
                # 1. Amplitude Reconstruction
                # -------------------------------------------------------------
                f1_ins = 0.5 * f_ins_a
                f2_ins = 0.75 * f_ins_a
                f3_ins = f_ins_a

                h1_pn = _pn_amplitude_ratio(l, m, f1_ins, eta, delta, chi_s, chi_a)
                h2_pn = _pn_amplitude_ratio(l, m, f2_ins, eta, delta, chi_s, chi_a)
                h3_pn = _pn_amplitude_ratio(l, m, f3_ins, eta, delta, chi_s, chi_a)

                h1_fit = getattr(XHMParameterFits, f"amp_insp1_{lm_str}")(eta, s1, s2, S)
                h2_fit = getattr(XHMParameterFits, f"amp_insp2_{lm_str}")(eta, s1, s2, S)
                h3_fit = getattr(XHMParameterFits, f"amp_insp3_{lm_str}")(eta, s1, s2, S)
                if m % 2 != 0:
                    # The odd-m calibration surfaces are allowed to carry the
                    # sign of the full strain.  Reconstructed amplitudes use
                    # their non-negative magnitudes (arXiv:2001.10914v2,
                    # Sec. V.A.2), so fold these collocation targets before
                    # solving for the inspiral reconstruction coefficients.
                    h1_fit = torch.abs(h1_fit)
                    h2_fit = torch.abs(h2_fit)
                    h3_fit = torch.abs(h3_fit)

                rhs_insp = torch.stack([h1_fit - h1_pn, h2_fit - h2_pn, h3_fit - h3_pn], dim=-1)
                inv_mat = _INSP_MAT_INV.to(device=device, dtype=dtype)
                abc = torch.matmul(rhs_insp, inv_mat.T)
                alpha_insp = abc[..., 0]
                beta_insp = abc[..., 1]
                gamma_insp = abc[..., 2]

                # Ringdown amplitude parameters (Sovereign SXS NR calibrated)
                alambda_fit, lambda_fit = eval_native_ringdown_params(l, m, eta, delta)

                def eval_h_rd(f_in: torch.Tensor) -> torch.Tensor:
                    return _ringdown_amplitude_ratio(
                        f_in, f_ring_lm, f_damp_lm, sigma_lm, alambda_fit, lambda_fit,
                    )

                if (l, m) == (3, 2):
                    # The three inherited calibration conditions below are not
                    # the four data specified in Table IX. Preserve that gap;
                    # do not fabricate a fourth fit (see the pass-2 audit).
                    alpha2_32 = alpha2_22
                    g1 = XHMParameterFits.phase_32_s_dphi_p1(eta, s1, s2, S)
                    g2 = XHMParameterFits.phase_32_s_dphi_fring32(eta, s1, s2, S)
                    g3 = XHMParameterFits.phase_32_s_dphi_p4(eta, s1, s2, S)

                    f_nodes_s = torch.stack([
                        f_ring["22"],
                        f_ring["32"],
                        f_ring["32"] + 0.5 * f_damp["32"],
                    ], dim=-1)

                    f_damp_32_u = f_damp["32"].unsqueeze(-1) if f_damp["32"].dim() > 0 else f_damp["32"]
                    f_ring_32_u = f_ring["32"].unsqueeze(-1) if f_ring["32"].dim() > 0 else f_ring["32"]
                    lor_nodes_s = f_damp_32_u / ((f_damp_32_u**2) + (f_nodes_s - f_ring_32_u)**2)
                    m_s = torch.stack([
                        1.0 / (f_nodes_s**4),
                        lor_nodes_s,
                        torch.ones_like(f_nodes_s),
                    ], dim=-1)
                    rhs_s = torch.stack([
                        g1 - alpha2_32 / (f_nodes_s[..., 0]**2),
                        g2 - alpha2_32 / (f_nodes_s[..., 1]**2),
                        g3 - alpha2_32 / (f_nodes_s[..., 2]**2),
                    ], dim=-1)
                    sol_s = torch.linalg.solve(m_s, rhs_s)

                    alpha4_32 = sol_s[..., 0]
                    alpha_lam_32 = sol_s[..., 1]
                    dt_rd_32 = sol_s[..., 2]

                    # Alignment to (2,2) carrier in spheroidal basis
                    f0_align = f_ring["22"] + f_damp["22"]
                    f1_align = f_ring["22"]

                    delta_t_32_s = XHMParameterFits.phase_32_s_deltadphi(eta, s1, s2, S)
                    delta_phi_32_s = XHMParameterFits.phase_32_s_deltaphi(eta, s1, s2, S)

                    _, carrier_dp_at_f0 = eval_carrier_phi_and_deriv(f0_align)
                    carrier_p_at_f1, _ = eval_carrier_phi_and_deriv(f1_align)

                    dt_rd_32, phi_rd_32_s_shift = _spheroidal_32_alignment(
                        f_ring["22"], f_damp["22"], f_ring["32"], f_damp["32"],
                        alpha2_32, alpha4_32, alpha_lam_32, dt_rd_32,
                        carrier_dp_at_f0, carrier_p_at_f1, delta_t_32_s, delta_phi_32_s,
                    )

                    def eval_phi_32_s(f_in: torch.Tensor) -> torch.Tensor:
                        return _spheroidal_32_ringdown_phase(
                            f_in, f_ring["32"], f_damp["32"],
                            alpha2_32, alpha4_32, alpha_lam_32, dt_rd_32, phi_rd_32_s_shift,
                        )

                    eval_h_rd_s = eval_h_rd
                    a222_re, a222_im = XHMParameterFits.mixing_alpha222(chi_f)
                    a232_re, a232_im = XHMParameterFits.mixing_alpha223(chi_f)
                    a322_re, a322_im = XHMParameterFits.mixing_alpha322(chi_f)
                    a332_re, a332_im = XHMParameterFits.mixing_alpha333(chi_f)

                    # ratio r = alpha322 / alpha222
                    d = a222_re**2 + a222_im**2
                    r_re = (a322_re * a222_re + a322_im * a222_im) / d
                    r_im = (a322_im * a222_re - a322_re * a222_im) / d

                    # c = alpha332 - ratio * alpha232
                    ra_re = r_re * a232_re - r_im * a232_im
                    ra_im = r_re * a232_im + r_im * a232_re
                    c_re = a332_re - ra_re
                    c_im = a332_im - ra_im

                    def eval_mixed_rd_relative(f_in: torch.Tensor):
                        strain22, phase22 = eval_carrier_at(f_in)
                        # The carrier probe uses distance=M, so all amplitude
                        # derivatives are evaluated before the tiny physical
                        # distance normalization is applied.
                        a0 = _unsqueeze_to(m_sec * torch.sqrt(5.0 * eta / 24.0) / PI ** (2.0 / 3.0), f_in) * f_in ** (-7.0 / 6.0)
                        phase_s = eval_phi_32_s(f_in)
                        if _carrier_h22_ratio is not _DEFAULT_CARRIER_H22_RATIO or _spherical_32_from_spheroidal is not _DEFAULT_SPHERICAL_32_FROM_SPHEROIDAL:
                            rel22 = _carrier_h22_ratio(strain22, phase22, phase_s, a0)
                            alpha222 = torch.complex(a222_re, a222_im)
                            alpha232 = torch.complex(a232_re, a232_im)
                            alpha322 = torch.complex(a322_re, a322_im)
                            alpha332 = torch.complex(a332_re, a332_im)
                            mixed = _spherical_32_from_spheroidal(
                                rel22, eval_h_rd_s(f_in),
                                _unsqueeze_to(alpha222, f_in), _unsqueeze_to(alpha232, f_in),
                                _unsqueeze_to(alpha322, f_in), _unsqueeze_to(alpha332, f_in),
                            )
                            return mixed.real, mixed.imag
                        # Eq. (4.6) defines H_lm against A^0_22. The common
                        # sqrt(16*pi/5) factor in the paper's A^0_22 and mode
                        # amplitude cancels, leaving the native carrier ratio.
                        rel22_re, rel22_im = _carrier_h22_ratio_cartesian(strain22, phase22, phase_s, a0)
                        return _spherical_32_from_spheroidal_cartesian(
                            rel22_re, rel22_im, eval_h_rd_s(f_in),
                            _unsqueeze_to(r_re, f_in), _unsqueeze_to(r_im, f_in),
                            _unsqueeze_to(c_re, f_in), _unsqueeze_to(c_im, f_in),
                        )

                    def eval_phi_32_rd(f_in: torch.Tensor):
                        # Factoring out the smooth spheroidal phase avoids
                        # wrapping its many orbital cycles into [-pi, pi].
                        mixed_re, mixed_im = eval_mixed_rd_relative(f_in)
                        return eval_phi_32_s(f_in) - torch.atan2(mixed_im, mixed_re)

                    def eval_h_rd(f_in: torch.Tensor) -> torch.Tensor:
                        mixed_re, mixed_im = eval_mixed_rd_relative(f_in)
                        return torch.hypot(mixed_re, mixed_im)

                def eval_dh_rd(f_in: torch.Tensor) -> torch.Tensor:
                    return _value_and_derivative(eval_h_rd, f_in)[1]

                def eval_h_ins(f_in: torch.Tensor) -> torch.Tensor:
                    pn_part = _pn_amplitude_ratio(l, m, f_in, eta, delta, chi_s, chi_a)
                    f_ins_u = _unsqueeze_to(f_ins_a, f_in)
                    a_u = _unsqueeze_to(alpha_insp, f_in)
                    b_u = _unsqueeze_to(beta_insp, f_in)
                    g_u = _unsqueeze_to(gamma_insp, f_in)
                    f_ratio = f_in / f_ins_u
                    pseudo = (
                        a_u * torch.pow(f_ratio, 7.0 / 3.0)
                        + b_u * torch.pow(f_ratio, 8.0 / 3.0)
                        + g_u * torch.pow(f_ratio, 3.0)
                    )
                    return pn_part + pseudo

                def eval_dh_ins(f_in: torch.Tensor) -> torch.Tensor:
                    return _value_and_derivative(eval_h_ins, f_in)[1]

                # Intermediate amplitude collocation matching
                f_int1 = f_ins_a + (1.0 / 3.0) * (f_rd_a - f_ins_a)
                f_int2 = f_ins_a + (2.0 / 3.0) * (f_rd_a - f_ins_a)

                v_int1_fit = getattr(XHMParameterFits, f"amp_int1_{lm_str}")(eta, s1, s2, S)
                v_int2_fit = getattr(XHMParameterFits, f"amp_int2_{lm_str}")(eta, s1, s2, S)
                if m % 2 != 0:
                    # Eq. (5.3) fits the intermediate amplitude itself.  The
                    # odd-m fit may be negative because its sign is folded
                    # from the strain; the reconstructed amplitude is |fit|.
                    v_int1_fit = torch.abs(v_int1_fit)
                    v_int2_fit = torch.abs(v_int2_fit)

                h_ins_at_cut = eval_h_ins(f_ins_a)
                dh_ins_at_cut = eval_dh_ins(f_ins_a)
                p_ins = torch.pow(PI * f_ins_a, 7.0 / 6.0) / torch.clamp(h_ins_at_cut, min=1e-12)
                dp_ins = torch.pow(PI * f_ins_a, 7.0 / 6.0) * (
                    (7.0 / 6.0) / (f_ins_a * torch.clamp(h_ins_at_cut, min=1e-12))
                    - dh_ins_at_cut / torch.clamp(h_ins_at_cut**2, min=1e-12)
                )

                # Eq. (5.3) fits A^Inter_lm/A0, where A0 = pi*sqrt(2*eta/3)
                # is frequency independent.  Converting that target to the
                # Eq. (4.6) H_lm ratio supplies the common (pi*f)^(7/6)
                # factor only when evaluating H_lm, hence p=1/(A^Inter/A0).
                p_int1 = 1.0 / torch.clamp(v_int1_fit, min=1e-12)
                p_int2 = 1.0 / torch.clamp(v_int2_fit, min=1e-12)

                h_rd_at_cut = eval_h_rd(f_rd_a)
                dh_rd_at_cut = eval_dh_rd(f_rd_a)
                p_rd = torch.pow(PI * f_rd_a, 7.0 / 6.0) / torch.clamp(h_rd_at_cut, min=1e-12)
                dp_rd = torch.pow(PI * f_rd_a, 7.0 / 6.0) * (
                    (7.0 / 6.0) / (f_rd_a * torch.clamp(h_rd_at_cut, min=1e-12))
                    - dh_rd_at_cut / torch.clamp(h_rd_at_cut**2, min=1e-12)
                )

                def make_vandermonde_rows(f_node: torch.Tensor, is_deriv: bool = False) -> torch.Tensor:
                    if not is_deriv:
                        return torch.stack([
                            torch.ones_like(f_node), f_node, f_node**2, f_node**3, f_node**4, f_node**5
                        ], dim=-1)
                    return torch.stack([
                        torch.zeros_like(f_node), torch.ones_like(f_node), 2.0 * f_node, 3.0 * (f_node**2), 4.0 * (f_node**3), 5.0 * (f_node**4)
                    ], dim=-1)

                v_mat = torch.stack([
                    make_vandermonde_rows(f_ins_a),
                    make_vandermonde_rows(f_ins_a, is_deriv=True),
                    make_vandermonde_rows(f_int1),
                    make_vandermonde_rows(f_int2),
                    make_vandermonde_rows(f_rd_a),
                    make_vandermonde_rows(f_rd_a, is_deriv=True),
                ], dim=-2)
                rhs_amp = torch.stack([p_ins, dp_ins, p_int1, p_int2, p_rd, dp_rd], dim=-1)
                delta_amp = torch.linalg.solve(v_mat, rhs_amp)

                # Parameter-space veto cubic: retain C1 matching at the
                # endpoints after removing both intermediate values.
                def make_hermite_rows(f_node: torch.Tensor, is_deriv: bool = False) -> torch.Tensor:
                    if not is_deriv:
                        return torch.stack([
                            torch.ones_like(f_node), f_node, f_node**2, f_node**3
                        ], dim=-1)
                    return torch.stack([
                        torch.zeros_like(f_node), torch.ones_like(f_node), 2.0 * f_node, 3.0 * (f_node**2)
                    ], dim=-1)

                v_mat3 = torch.stack([
                    make_hermite_rows(f_ins_a),
                    make_hermite_rows(f_ins_a, is_deriv=True),
                    make_hermite_rows(f_rd_a),
                    make_hermite_rows(f_rd_a, is_deriv=True),
                ], dim=-2)
                rhs_amp3 = torch.stack([p_ins, dp_ins, p_rd, dp_rd], dim=-1)
                delta_amp3 = torch.linalg.solve(v_mat3, rhs_amp3)

                # Published pole fallback: keep all four positive amplitude
                # values and remove only the two boundary derivatives. This
                # differs from the parameter-space veto above, which removes
                # both inner values and retains C1 boundary matching.
                v_mat4 = torch.stack([
                    make_hermite_rows(f_ins_a),
                    make_hermite_rows(f_int1),
                    make_hermite_rows(f_int2),
                    make_hermite_rows(f_rd_a),
                ], dim=-2)
                rhs_amp4 = torch.stack([p_ins, p_int1, p_int2, p_rd], dim=-1)
                delta_amp4 = torch.linalg.solve(v_mat4, rhs_amp4)
                # A single surviving inner value leaves five conditions and
                # therefore a quartic inverse polynomial.  The two variants
                # are kept separate so a veto of fInt1 still preserves fInt2
                # (and conversely) in a later pole fallback.
                def make_quartic_value_row(f_node: torch.Tensor) -> torch.Tensor:
                    return torch.stack([
                        torch.ones_like(f_node), f_node, f_node**2, f_node**3, f_node**4,
                    ], dim=-1)

                def make_quartic_deriv_row(f_node: torch.Tensor) -> torch.Tensor:
                    return torch.stack([
                        torch.zeros_like(f_node), torch.ones_like(f_node),
                        2.0 * f_node, 3.0 * f_node**2, 4.0 * f_node**3,
                    ], dim=-1)

                v_mat5_1 = torch.stack([
                    make_quartic_value_row(f_ins_a), make_quartic_deriv_row(f_ins_a),
                    make_quartic_value_row(f_int1), make_quartic_value_row(f_rd_a),
                    make_quartic_deriv_row(f_rd_a),
                ], dim=-2)
                v_mat5_2 = torch.stack([
                    make_quartic_value_row(f_ins_a), make_quartic_deriv_row(f_ins_a),
                    make_quartic_value_row(f_int2), make_quartic_value_row(f_rd_a),
                    make_quartic_deriv_row(f_rd_a),
                ], dim=-2)
                delta_amp5_1 = torch.linalg.solve(
                    v_mat5_1, torch.stack([p_ins, dp_ins, p_int1, p_rd, dp_rd], dim=-1),
                )
                delta_amp5_2 = torch.linalg.solve(
                    v_mat5_2, torch.stack([p_ins, dp_ins, p_int2, p_rd, dp_rd], dim=-1),
                )
                # Lower-order value-only branches preserve every surviving
                # amplitude value when a denominator pole is found.
                def make_value_row(f_node: torch.Tensor) -> torch.Tensor:
                    return torch.stack([
                        torch.ones_like(f_node), f_node, f_node**2, f_node**3,
                    ], dim=-1)

                v_mat2_1 = torch.stack([
                    make_value_row(f_ins_a), make_value_row(f_int1), make_value_row(f_rd_a),
                ], dim=-2)[..., :3, :3]
                v_mat2_2 = torch.stack([
                    make_value_row(f_ins_a), make_value_row(f_int2), make_value_row(f_rd_a),
                ], dim=-2)[..., :3, :3]
                delta_amp2_1 = torch.linalg.solve(v_mat2_1, torch.stack([p_ins, p_int1, p_rd], dim=-1))
                delta_amp2_2 = torch.linalg.solve(v_mat2_2, torch.stack([p_ins, p_int2, p_rd], dim=-1))
                v_mat1 = torch.stack([
                    torch.ones_like(f_ins_a), f_ins_a,
                    torch.ones_like(f_rd_a), f_rd_a,
                ], dim=-1).reshape(f_ins_a.shape + (2, 2))
                delta_amp1 = torch.linalg.solve(v_mat1, torch.stack([p_ins, p_rd], dim=-1))
                # The source's numeric Table VI thresholds are deferred: it
                # does not resolve raw A versus A/A0 threshold units.
                h_int1_at_cut = v_int1_fit
                h_int2_at_cut = v_int2_fit
                # Monotonicity compares the four collocation
                # amplitudes in the common A/A0 convention.  The endpoint
                # values are 1/p at the same cuts; using H there would mix in
                # a frequency-dependent (pi*f)^(7/6) factor.
                alive1, alive2, remove_derivatives = _ordered_amplitude_vetoes(
                    (l, m), m1 / m2, eta, s1, s2,
                    1.0 / torch.clamp(p_ins, min=1e-12),
                    h_int1_at_cut, h_int2_at_cut,
                    1.0 / torch.clamp(p_rd, min=1e-12),
                )

                # Select coefficients per intrinsic row before checking them.
                # An unselected reconstruction must never trigger a fallback.
                def pad_coefficients(coefficients: torch.Tensor, count: int) -> torch.Tensor:
                    return torch.nn.functional.pad(coefficients, (0, count - coefficients.shape[-1]))

                full = (alive1 & alive2).unsqueeze(-1)
                one1 = (alive1 & ~alive2).unsqueeze(-1)
                one2 = (~alive1 & alive2).unsqueeze(-1)
                active_coefficients = torch.where(full, delta_amp, pad_coefficients(delta_amp3, 6))
                active_coefficients = torch.where(one1, pad_coefficients(delta_amp5_1, 6), active_coefficients)
                active_coefficients = torch.where(one2, pad_coefficients(delta_amp5_2, 6), active_coefficients)

                # Dropping both boundary derivatives retains every surviving
                # value. The paper does not specify a one-derivative-at-a-time
                # order; removing both together is the retained interpretation.
                value_coefficients = torch.where(full, delta_amp4, pad_coefficients(delta_amp1, 4))
                value_coefficients = torch.where(one1, pad_coefficients(delta_amp2_1, 4), value_coefficients)
                value_coefficients = torch.where(one2, pad_coefficients(delta_amp2_2, 4), value_coefficients)
                active_coefficients = torch.where(
                    remove_derivatives.unsqueeze(-1), pad_coefficients(value_coefficients, 6), active_coefficients,
                )
                use_pole_fallback = polynomial_has_nonpositive(active_coefficients, f_ins_a, f_rd_a)
                fallback_bad = polynomial_has_nonpositive(value_coefficients, f_ins_a, f_rd_a)
                selected_coefficients = torch.where(
                    use_pole_fallback.unsqueeze(-1), pad_coefficients(value_coefficients, 6), active_coefficients,
                )
                selected_coefficients = torch.where(
                    (use_pole_fallback & fallback_bad).unsqueeze(-1),
                    pad_coefficients(delta_amp1, 6), selected_coefficients,
                )
                # Numerical stationary-point checks replace the old eleven
                # samples and five-percent floor. Decisions are detached;
                # the selected polynomial retains parameter gradients.
                p_poly_active = torch.zeros_like(mf_grid)
                for power in range(5, -1, -1):
                    p_poly_active = p_poly_active * mf_grid + _unsqueeze_to(selected_coefficients[..., power], mf_grid)
                p_poly_active = torch.clamp(p_poly_active, min=1e-12)
                f_ins_a_u = _unsqueeze_to(f_ins_a, mf_grid)
                f_rd_a_u = _unsqueeze_to(f_rd_a, mf_grid)
                h_ins_grid = eval_h_ins(mf_grid)
                h_int_grid = torch.pow(PI * mf_grid, 7.0 / 6.0) / p_poly_active
                h_rd_grid = eval_h_rd(mf_grid)
                h_ratio_grid = torch.where(
                    mf_grid <= f_ins_a_u,
                    h_ins_grid,
                    torch.where(
                        mf_grid <= f_rd_a_u,
                        h_int_grid,
                        h_rd_grid,
                    ),
                )
                if m % 2 != 0:
                    zero_odd_mask = _unsqueeze_to((delta == 0.0) & (chi_a == 0.0), mf_grid)
                    h_ratio_grid = torch.where(zero_odd_mask, torch.zeros_like(h_ratio_grid), h_ratio_grid)

                amp_lm_phys = mode_norm * a0_phys * h_ratio_grid

                # -------------------------------------------------------------
                # 2. Phase Reconstruction
                # -------------------------------------------------------------
                if (l, m) != (3, 2):
                    # Modes without mixing: (2,1), (3,3), (4,4)
                    f_p1 = getattr(XHMParameterFits, f"phase_p1_{lm_str}")(eta, s1, s2, S)
                    f_p2 = getattr(XHMParameterFits, f"phase_p2_{lm_str}")(eta, s1, s2, S)
                    f_p3 = getattr(XHMParameterFits, f"phase_p3_{lm_str}")(eta, s1, s2, S)
                    f_p4 = getattr(XHMParameterFits, f"phase_p4_{lm_str}")(eta, s1, s2, S)
                    f_p6 = getattr(XHMParameterFits, f"phase_p6_{lm_str}")(eta, s1, s2, S)

                    # Table VII / Fig. 8 default subset: [C1,C2,E2,E3,E5]
                    # = [f1,f2,f3,f4,f6].  The fifth table point is reserved
                    # for the paper's sparse-calibration subset.
                    f_nodes = _nonmix_intermediate_phase_nodes(eta, f_ins_p, f_ring_lm)
                    rhs_phase = torch.stack([f_p1, f_p2, f_p3, f_p4, f_p6], dim=-1)
                    a_int = _nonmix_intermediate_phase_coefficients(f_nodes, rhs_phase, f_ring_lm, f_damp_lm)

                    def eval_dphi_int(f_in: torch.Tensor) -> torch.Tensor:
                        return _nonmix_intermediate_phase_derivative(f_in, a_int, f_ring_lm, f_damp_lm)

                    def eval_phi_int(f_in: torch.Tensor) -> torch.Tensor:
                        return _nonmix_intermediate_phase(f_in, a_int, f_ring_lm, f_damp_lm)

                    # Ringdown phase connection at f_rd_p (Sovereign SXS NR calibrated)
                    alpha_2_lm, alpha_lambda_lm = eval_native_phase_params(l, m, eta)

                    dt_rd, dphi_rd_shift = _nonmix_ringdown_phase_shifts(
                        f_rd_p, eval_phi_int(f_rd_p), eval_dphi_int(f_rd_p),
                        f_ring_lm, f_damp_lm, alpha_2_lm, alpha_lambda_lm,
                    )

                    def eval_phi_rd(f_in: torch.Tensor) -> torch.Tensor:
                        return _nonmix_ringdown_phase(
                            f_in, f_ring_lm, f_damp_lm, alpha_2_lm,
                            alpha_lambda_lm, dt_rd, dphi_rd_shift,
                        )

                    # Inherited phase convention; its calibrated alignment
                    # remains outside the published-equation certification.
                    delta_target = (l % 2) * PI - (3.0 * PI / 8.0) * (m_flt - 2.0)
                    if l == 2 and m == 1:
                        ff = 0.008
                        a21_pn = (
                            -16.0 * delta * eta * ff * (math.pi**1.5) / (3.0 * math.sqrt(5.0))
                            + 4.0 * (2.0**(1.0/3.0)) * ((s1 - s2) + delta * (s1 + s2)) * eta * (ff**(4.0/3.0)) * (math.pi**(11.0/6.0)) / math.sqrt(5.0)
                            + 2.0 * (2.0**(2.0/3.0)) * eta * (306.0 * delta - 360.0 * delta * eta) * (ff**(5.0/3.0)) * (math.pi**(13.0/6.0)) / (189.0 * math.sqrt(5.0))
                        )
                        delta_target = delta_target + torch.where(a21_pn >= 0, torch.tensor(PI, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype))

                    # Evaluate carrier phase on scaled frequency grid (2/m) * f
                    f_scaled_grid = (2.0 / m_flt) * f_eval
                    if carrier_state is not None:
                        phi_22_scaled = carrier_state.eval_phase(f_scaled_grid)
                    else:
                        _, _, phi_22_scaled = self.base_22(
                            f=f_scaled_grid,
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
                            _final_spin_override=final_spin_override,
                        )
                    delta_target_u = _unsqueeze_to(delta_target, mf_grid)
                    carrier_p_at_ins, carrier_dp_at_ins = eval_carrier_phi_and_deriv((2.0 / m_flt) * f_ins_p)
                    phi_ins_grid = rescale_carrier_phase(phi_22_scaled, m) + delta_target_u
                    phi_ins_at_ins = rescale_carrier_phase(carrier_p_at_ins, m) + delta_target
                    dphi_int_shift = phi_ins_at_ins - eval_phi_int(f_ins_p)
                    dphi_int_shift_u = _unsqueeze_to(dphi_int_shift, mf_grid)

                    phi_int_grid = eval_phi_int(mf_grid) + dphi_int_shift_u
                    phi_rd_grid = eval_phi_rd(mf_grid) + dphi_int_shift_u

                    f_ins_p_u = _unsqueeze_to(f_ins_p, mf_grid)
                    f_rd_p_u = _unsqueeze_to(f_rd_p, mf_grid)
                    psi_lm = torch.where(
                        mf_grid <= f_ins_p_u,
                        phi_ins_grid,
                        torch.where(
                            mf_grid <= f_rd_p_u,
                            phi_int_grid,
                            phi_rd_grid,
                        ),
                    )
                    h_mode_re = amp_lm_phys * torch.cos(psi_lm)
                    h_mode_im = -amp_lm_phys * torch.sin(psi_lm)

                else:
                    # ---------------------------------------------------------
                    # Mode (3, 2): Spherical-Spheroidal Mode Mixing
                    # ---------------------------------------------------------
                    # Intermediate phase for (3,2)
                    f_p1_32 = XHMParameterFits.phase_p1_32(eta, s1, s2, S)
                    f_p2_32 = XHMParameterFits.phase_p2_32(eta, s1, s2, S)
                    f_p3_32 = XHMParameterFits.phase_p3_32(eta, s1, s2, S)
                    f_p4_32 = XHMParameterFits.phase_p4_32(eta, s1, s2, S)

                    f_nodes_int_32 = _intermediate_32_phase_nodes(eta, f_ins_p, f_ring["22"], f_damp["22"])

                    dphi_rd_at_join, d2phi_rd_at_join = _value_and_derivative(
                        lambda x: _value_and_derivative(eval_phi_32_rd, x)[1], f_rd_p,
                    )

                    targets_int_32 = torch.stack((f_p1_32, f_p2_32, f_p3_32, f_p4_32), dim=-1)
                    sol_int_32 = _intermediate_32_phase_coefficients(
                        f_nodes_int_32, targets_int_32, f_rd_p, dphi_rd_at_join, d2phi_rd_at_join,
                        f_ring["32"], f_damp["32"],
                    )

                    def eval_dphi_int_32(f_in: torch.Tensor) -> torch.Tensor:
                        return _intermediate_32_phase_derivative(f_in, sol_int_32, f_ring["32"], f_damp["32"])

                    def eval_phi_int_32(f_in: torch.Tensor) -> torch.Tensor:
                        return _intermediate_32_phase(f_in, sol_int_32, f_ring["32"], f_damp["32"])

                    c_int_32 = eval_phi_32_rd(f_rd_p) - eval_phi_int_32(f_rd_p)

                    # Inherited phase convention; its calibrated alignment
                    # remains outside the published-equation certification.
                    delta_target_32 = PI
                    carrier_p_at_ins, carrier_dp_at_ins = eval_carrier_phi_and_deriv(f_ins_p)
                    phi_ins_32_grid = rescale_carrier_phase(phi_22_carrier, m) + delta_target_32
                    phi_ins_at_ins_32 = rescale_carrier_phase(carrier_p_at_ins, m) + delta_target_32
                    dphi_int_shift_32 = phi_ins_at_ins_32 - (eval_phi_int_32(f_ins_p) + c_int_32)
                    dphi_int_shift_32_u = _unsqueeze_to(dphi_int_shift_32, mf_grid)
                    c_int_32_u = _unsqueeze_to(c_int_32, mf_grid)

                    phi_int_32_grid = eval_phi_int_32(mf_grid) + c_int_32_u + dphi_int_shift_32_u
                    phi_rd_32_grid = eval_phi_32_rd(mf_grid) + dphi_int_shift_32_u

                    f_ins_p_u = _unsqueeze_to(f_ins_p, mf_grid)
                    f_rd_p_u = _unsqueeze_to(f_rd_p, mf_grid)
                    phi_32_imr = torch.where(
                        mf_grid <= f_ins_p_u,
                        phi_ins_32_grid,
                        torch.where(
                            mf_grid <= f_rd_p_u,
                            phi_int_32_grid,
                            phi_rd_32_grid,
                        ),
                    )

                    # Both amplitude and phase are already spherical here.
                    # Each has its own published transition frequency (Table II).
                    h_mode_re = amp_lm_phys * torch.cos(phi_32_imr)
                    h_mode_im = -amp_lm_phys * torch.sin(phi_32_imr)

            # Apply the Fourier time shift and orbital phase once, after the
            # intrinsic joins and (3,2) mixing. Their factors must be common to
            # every frequency region (García-Quirós et al., Sec. II B and Eq. (2.13)).
            extrinsic_phase = 2.0 * PI * f_eval * tc_eval - float(m) * phic_eval
            cos_ext = torch.cos(extrinsic_phase)
            sin_ext = torch.sin(extrinsic_phase)
            # Complex multiplication: (h_re + i * h_im) * (cos_ext - i * sin_ext)
            # Re = h_re * cos_ext + h_im * sin_ext
            # Im = -h_re * sin_ext + h_im * cos_ext
            h_rot_re = h_mode_re * cos_ext + h_mode_im * sin_ext
            h_rot_im = -h_mode_re * sin_ext + h_mode_im * cos_ext

            # Apply body exchange swap phase (-1)^m for odd-m modes when swapped
            if m % 2 != 0:
                swap_u = _unsqueeze_to(swap_mask, h_rot_re)
                h_rot_re = torch.where(swap_u, -h_rot_re, h_rot_re)
                h_rot_im = torch.where(swap_u, -h_rot_im, h_rot_im)

            # Spin-weighted spherical harmonics projection
            # The positive-frequency carrier is the -m Fourier mode
            # (Pratten et al., Appendix B, Eqs. B6-B7); its observer-angle
            # factor is exp(-i*m*phi0). The shared SWSH helper represents
            # positive-m harmonics, so evaluate it at the opposite azimuth.
            # XHM v2 Eqs. (8)-(9) then require (-1)^l for the -m carrier:
            # d_{+,-m}=(-1)^l d_{+,+m}(-phi0), and likewise for d_cross.
            dp_re, dp_im, dc_re, dc_im = _swsh_projection_coefficients_cartesian(l, m, inc_eval, -phi0_eval)
            if l % 2:
                dp_re, dp_im, dc_re, dc_im = -dp_re, -dp_im, -dc_re, -dc_im
            hp_lm_re = dp_re * h_rot_re - dp_im * h_rot_im
            hp_lm_im = dp_re * h_rot_im + dp_im * h_rot_re
            hc_lm_re = dc_re * h_rot_re - dc_im * h_rot_im
            hc_lm_im = dc_re * h_rot_im + dc_im * h_rot_re

            h_plus_re = h_plus_re + hp_lm_re
            h_plus_im = h_plus_im + hp_lm_im
            h_cross_re = h_cross_re + hc_lm_re
            h_cross_im = h_cross_im + hc_lm_im

            if return_modes:
                mode_dict[(l, m)] = (
                    torch.view_as_complex(torch.stack([hp_lm_re, hp_lm_im], dim=-1)),
                    torch.view_as_complex(torch.stack([hc_lm_re, hc_lm_im], dim=-1)),
                )
            elif return_unprojected_modes:
                h_unproj = torch.view_as_complex(torch.stack([h_rot_re, h_rot_im], dim=-1))
                mode_dict[(l, m)] = h_unproj
                # Negative-m mode satisfies equatorial reflection symmetry: h_{l, -m} = (-1)^l * conj(h_{l, m})
                sign_l = 1.0 if l % 2 == 0 else -1.0
                mode_dict[(l, -m)] = torch.view_as_complex(
                    torch.stack([sign_l * h_rot_re, -sign_l * h_rot_im], dim=-1)
                )

        # Enforce exact finite zeros for DC and non-positive frequency bins on real floats
        pos_mask = f_eval > 0.0
        h_plus_re = torch.where(pos_mask, h_plus_re, torch.zeros_like(h_plus_re))
        h_plus_im = torch.where(pos_mask, h_plus_im, torch.zeros_like(h_plus_im))
        h_cross_re = torch.where(pos_mask, h_cross_re, torch.zeros_like(h_cross_re))
        h_cross_im = torch.where(pos_mask, h_cross_im, torch.zeros_like(h_cross_im))

        h_plus_total = torch.view_as_complex(torch.stack([h_plus_re, h_plus_im], dim=-1))
        h_cross_total = torch.view_as_complex(torch.stack([h_cross_re, h_cross_im], dim=-1))

        if return_modes or return_unprojected_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
