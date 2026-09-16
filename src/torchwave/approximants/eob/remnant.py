"""Published aligned-spin remnant inputs, with no implicit waveform selection.

Mass: Ossokine et al. (2020), PRD 102, 044055, Eq. (3.5).
Spin: Hofmann, Barausse & Rezzolla, arXiv:1605.01938v2, Eq. (7),
Table 1 and Eq. (11). Spin coefficients retain the printed precision.
"""

import math
from numbers import Real

import torch


# Rows are i, columns j in sum k_ij nu**(2+i) a_eff**j. The omitted k00
# is determined separately by Eq. (11), using its central value 0.68646.
_HOFMANN = {
    (1, 2): (.41616, (
        (0., -1.2019, -1.20764),
        (3.79245, 1.18385, 4.90494),
    )),
    (3, 3): (.463926, (
        (0., 2.87025, -1.53315, -3.78893),
        (32.9127, -62.9901, 10.0068, 56.1926),
        (-136.832, 329.32, -13.2034, -252.27),
        (210.075, -545.35, -3.97509, 368.405),
    )),
    (3, 4): (.474046, (
        (0., 3.39221, 4.48865, -5.77101, -13.0459),
        (35.1278, -72.9336, -86.0036, 93.7371, 200.975),
        (-146.822, 387.184, 447.009, -467.383, -884.339),
        (223.911, -648.502, -697.177, 753.738, 1166.89),
    )),
}


class _KerrISCORadius(torch.autograd.Function):
    """Bardeen radius with the derivative of its defining ISCO equation."""

    @staticmethod
    def forward(ctx, spin):
        # Hofmann Eqs. (4)-(6), algebraically rationalized near spin=0.
        # u=(1+a)^(1/3), v=(1-a)^(1/3), t=u*v, s=u+v, d=u²+uv+v².
        # (3-Z1)/a² = [s+4/(2+s)]/(1+t+t²) - 4/[(2+s)*d²].
        # This removes both sign(a)*sqrt(a²) and cancellation in 3-Z1.
        u, v = (1 + spin).pow(1/3), (1 - spin).pow(1/3)
        t, s = u*v, u + v
        d = u.square() + t + v.square()
        z1 = 1 + t*s
        z2 = torch.sqrt(3*spin.square() + z1.square())
        ratio = (s + 4/(2+s))/(1+t+t.square()) - 4/((2+s)*d.square())
        radius = 3 + z2 - spin*torch.sqrt(ratio*(3+z1+2*z2))
        radius = torch.where(spin == -1, 9., radius)
        radius = torch.where(spin == 0, 6., radius)
        radius = torch.where(spin == 1, 1., radius)
        ctx.save_for_backward(spin, radius)
        return radius

    @staticmethod
    def backward(ctx, gradient):
        spin, radius = ctx.saved_tensors
        root = radius.sqrt()
        # Differentiate r²-6r+8a*sqrt(r)-3a²=0. Unlike derivatives of
        # individual cube roots this is regular at a=0 and a=-1.
        derivative = (6*spin - 8*root)/(2*radius - 6 + 4*spin/root)
        return gradient*derivative


def _kerr_isco(spin):
    """(r/M, E, L/M) for positive orbital L and signed Kerr spin.

    The caller supplies a float tensor in [-1,1]. The radius has a divergent
    derivative at +1; no finite derivative is invented at that endpoint.
    """
    radius = _KerrISCORadius.apply(spin)
    energy = torch.sqrt(1 - 2/(3*radius))
    angular_momentum = 2/(3*math.sqrt(3))*(1 + 2*torch.sqrt(3*radius - 2))
    return radius, energy, angular_momentum


def _parameters(mass1, mass2, chi1, chi2):
    values = (mass1, mass2, chi1, chi2)
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if any(value.dtype not in (torch.float32, torch.float64) for value in tensors):
        raise TypeError("remnant inputs require float32 or float64 tensors")
    if any(not isinstance(value, (torch.Tensor, Real)) or isinstance(value, bool)
           for value in values):
        raise TypeError("remnant inputs must be real scalars or floating tensors")
    device = tensors[0].device if tensors else torch.device("cpu")
    if any(value.device != device for value in tensors):
        raise ValueError("remnant input tensors must share a device")
    dtype = torch.float32 if tensors else torch.float64
    for value in tensors:
        dtype = torch.promote_types(dtype, value.dtype)
    mass1, mass2, chi1, chi2 = torch.broadcast_tensors(*(
        torch.as_tensor(value, dtype=dtype, device=device) for value in values))
    if any(not bool(torch.isfinite(value).all()) for value in (mass1, mass2, chi1, chi2)):
        raise ValueError("remnant inputs must be finite")
    if bool(((mass1 <= 0) | (mass2 <= 0)).any()):
        raise ValueError("remnant component masses must be positive")
    if bool(((chi1.abs() > 1) | (chi2.abs() > 1)).any()):
        raise ValueError("remnant spin projections must lie in [-1,1]")

    # Scaling first avoids overflow in the sum/squares of dimensional masses.
    scale = torch.maximum(mass1, mass2)
    m1, m2 = mass1/scale, mass2/scale
    denominator = m1 + m2
    nu = (m1/denominator)*(m2/denominator)
    weighted_spin = (m1.square()*chi1 + m2.square()*chi2)/(m1.square()+m2.square())
    total_spin = (1 - 2*nu)*weighted_spin
    return nu, total_spin, weighted_spin, chi1, chi2


def eob_final_mass_fraction(mass1, mass2, chi1, chi2):
    """Return M_final/(mass1+mass2) from Ossokine Eq. (3.5).

    Masses are positive and in any common unit; chi1,chi2 are signed spin
    projections along orbital angular momentum in [-1,1]. Inputs describe
    broadcastable parameter batches. Tensor dtypes are promoted within
    float32/float64 on a common device; Python-only inputs use float64.
    This scalar projection fit does not evolve precessing spin directions.
    """
    nu, total_spin, weighted_spin, _, _ = _parameters(mass1, mass2, chi1, chi2)
    _, energy, _ = _kerr_isco(total_spin)
    equal_mass_radiation = .00258 - .0773/(weighted_spin - 1.6939)
    return 1 - nu*(1-4*nu)*(1-energy) - 16*nu.square()*equal_mass_radiation


def hofmann_final_spin(mass1, mass2, chi1, chi2, *, fit_order):
    """Return the signed final-spin projection from Hofmann Eq. (7).

    ``fit_order`` is REQUIRED: choose Table 1's (1,2), (3,3), or (3,4).
    No target waveform's choice is inferred. Inputs follow
    :func:`eob_final_mass_fraction`; the return value keeps its sign and is
    not clipped. This is the parallel-spin formula, not the generic spin
    magnitude/direction extension in Hofmann's later equations.
    """
    if (not isinstance(fit_order, tuple) or len(fit_order) != 2
            or any(type(order) is not int for order in fit_order)
            or fit_order not in _HOFMANN):
        raise ValueError("fit_order must be one of (1,2), (3,3), or (3,4)")
    xi, coefficients = _HOFMANN[fit_order]
    nu, total_spin, _, chi1, chi2 = _parameters(mass1, mass2, chi1, chi2)
    effective_spin = total_spin + xi*nu*(chi1+chi2)
    _, energy, angular_momentum = _kerr_isco(effective_spin)
    # Eq. (11): sqrt(3)/2 + sum_i k_i0/4**(2+i) = 0.68646.
    k00 = 16*(.68646 - math.sqrt(3)/2
              - sum(row[0]/4**(2+i) for i, row in enumerate(coefficients) if i))
    polynomial = torch.zeros_like(effective_spin)
    for i in range(len(coefficients)-1, -1, -1):
        row_value = torch.zeros_like(effective_spin)
        for j in range(len(coefficients[i])-1, -1, -1):
            coefficient = k00 if (i, j) == (0, 0) else coefficients[i][j]
            row_value = row_value*effective_spin + coefficient
        polynomial = polynomial*nu + row_value
    return (total_spin + nu*(angular_momentum - 2*total_spin*(energy-1))
            + nu.square()*polynomial)
