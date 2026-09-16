"""Direct PN/Kerr hybrid MECO; no waveform fits or calibration data.

Cabero et al., https://arxiv.org/html/1602.03134v3, Eqs. (2), (5)--(7),
(23)--(26); Pratten et al., https://arxiv.org/html/2001.11412v1,
Eqs. (48)--(49). This implements Cabero's black-hole hybrid convention:
mass-weighted effective spin, spin-orbit and spin-spin terms, no spin-cubic
term. It is not certified equivalent to the inherited XAS MECO fit.
"""

import math
from typing import Union

import torch


Real = Union[float, torch.Tensor]


def _real_tensors(*values):
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    device = tensors[0].device if tensors else None
    dtype = None
    for value in tensors:
        if value.device != device:
            raise ValueError("tensor inputs must share a device")
        if value.is_floating_point():
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError("floating tensors must use float32 or float64")
            dtype = value.dtype if dtype is None else torch.promote_types(dtype, value.dtype)
    dtype = dtype or torch.float64
    converted = []
    for value in values:
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise TypeError("inputs must be real and not boolean")
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if bool(torch.any(~torch.isfinite(tensor))):
            raise ValueError("inputs must be finite")
        converted.append(tensor)
    return torch.broadcast_tensors(*converted)


def _parameters(mass1, mass2, chi1, chi2, pn_order):
    if isinstance(pn_order, bool) or pn_order not in (3, 3.5):
        raise ValueError("pn_order must be 3 or 3.5")
    if bool(torch.any((mass1 < 0) | (mass2 < 0) | ((mass1 == 0) & (mass2 == 0)))):
        raise ValueError("masses must be nonnegative, with at least one positive mass")
    if bool(torch.any((chi1.abs() > 1) | (chi2.abs() > 1))):
        raise ValueError("component spins must lie in [-1, 1]")
    # Scaling before summing avoids overflow and retains exchange symmetry.
    scale = torch.maximum(mass1, mass2)
    u1, u2 = mass1 / scale, mass2 / scale
    x1, x2 = u1 / (u1 + u2), u2 / (u1 + u2)
    eta = x1 * x2
    # The symmetric mean/difference form preserves identical-spin endpoints
    # exactly even when the rounded mass fractions do not sum to one.
    delta = (u1 - u2) / (u1 + u2)
    chi = (chi1 + chi2) / 2 + delta * ((chi1 - chi2) / 2)
    if bool(torch.any(chi > 0.9999)):
        raise ValueError("the supported domain requires chi_eff <= 0.9999")
    spin_sum = chi1 + chi2
    spin = x1.square() * chi1 + x2.square() * chi2
    delta_sigma = (x1 - x2) * (x2 * chi2 - x1 * chi1)

    # E_h = E_Kerr - (1/2) sum_k c_k v^(k+2). Factor eta out of
    # every residual coefficient; all test-mass terms then vanish exactly.
    # For black holes the entire 2PN spin-spin term is -chi_eff^2.
    c2 = -eta / 12
    c3 = -2 * eta * spin_sum / 3
    c4 = eta * (19 / 8 - eta / 24)
    c5 = -eta * (5 * spin_sum + 61 * spin / 9 + 10 * delta_sigma / 3)
    ss3_residual = 5 * eta / 18 * (
        x1 * (7 * x1 + 22) * chi1.square()
        + x2 * (7 * x2 + 22) * chi2.square()
        + 2 * (10 - eta) * chi1 * chi2
    )
    c6 = eta * (34445 / 576 - 205 * math.pi**2 / 96
                - 155 * eta / 96 - 35 * eta.square() / 5184) + ss3_residual
    c7 = eta * (-81 * spin_sum / 4 - 367 * spin / 4 - 39 * delta_sigma
                + eta * (29 * spin / 12 + 5 * delta_sigma / 4))
    if pn_order == 3:
        c7 = torch.zeros_like(c7)
    return chi, (c2, c3, c4, c5, c6, c7)


def _energy(v, chi, coefficients, derivative):
    u = 1 - chi * v**3
    w = v.square() * u.pow(-2 / 3)
    sqrt_w = w.sqrt()
    denominator = 1 - 3 * w + 2 * chi * w * sqrt_w
    if derivative == 0:
        # Rationalize N/sqrt(D)-1 to retain the Newtonian energy for tiny v.
        numerator = 1 - 2 * w + chi * w * sqrt_w
        rad = denominator.sqrt()
        kerr = (-w + 4 * w.square() - 4 * chi * w.square() * sqrt_w
                + chi.square() * w**3) / (rad * (numerator + rad))
    else:
        stability = 1 - 6 * w + 8 * chi * w * sqrt_w - 3 * chi.square() * w.square()
        dw = 2 * v * u.pow(-5 / 3)
        if derivative == 1:
            kerr = -0.5 * dw * stability * denominator.pow(-1.5)
        else:
            ddw = 2 * u.pow(-5 / 3) + 10 * chi * v**3 * u.pow(-8 / 3)
            d_stability = -6 + 12 * chi * sqrt_w - 6 * chi.square() * w
            d_denominator = -3 + 3 * chi * sqrt_w
            kerr = -0.5 * denominator.pow(-1.5) * (
                ddw * stability + dw.square() * d_stability
                - 1.5 * dw.square() * stability * d_denominator / denominator
            )
    residual = torch.zeros_like(v)
    for power, coefficient in enumerate(coefficients, start=4):
        factor = 1 if derivative == 0 else power
        if derivative == 2:
            factor *= power - 1
        residual = residual + factor * coefficient * v**(power - derivative)
    return kerr - 0.5 * residual


def _photon_velocity(chi):
    # Physical exterior branch of D=0: r^3-6r^2+9r-4chi^2=0,
    # with its sign fixed by 1-3/r+2chi/r^(3/2)=0.
    radius = 2 * (1 + torch.cos((2 / 3) * torch.acos(-chi)))
    return (radius.pow(1.5) + chi).pow(-1 / 3)


def hybrid_binding_energy(
    velocity: Real, mass1: Real, mass2: Real, chi1: Real = 0., chi2: Real = 0.,
    *, pn_order: float = 3.5, derivative: int = 0,
) -> torch.Tensor:
    """Energy per reduced mass, or its first/second velocity derivative.

    ``velocity=(pi*M*f_22)^(1/3)``; masses share arbitrary units. This
    aligned-spin black-hole prescription includes nonspinning/SS terms
    through 3PN, SO through ``pn_order`` (3 or 3.5), and exact Kerr energy.
    It follows the omission of spin-cubic terms in Cabero's Eq. (6).

    Inputs broadcast. Tensor float32/float64 dtypes are promoted on their
    common device; Python-only inputs use float64. Masses may include one
    zero for the test-particle limit. Spins lie in [-1,1], chi_eff<=0.9999,
    and 0<velocity<the effective Kerr photon-orbit velocity. No tides,
    precession, neutron-star multipoles, or empirical corrections enter.
    """
    if isinstance(derivative, bool) or derivative not in (0, 1, 2):
        raise ValueError("derivative must be 0, 1, or 2")
    v, m1, m2, s1, s2 = _real_tensors(velocity, mass1, mass2, chi1, chi2)
    chi, coefficients = _parameters(m1, m2, s1, s2, pn_order)
    if bool(torch.any((v <= 0) | (v >= _photon_velocity(chi)))):
        raise ValueError("velocity must lie strictly between zero and the Kerr photon orbit")
    result = _energy(v, chi, coefficients, derivative)
    if bool(torch.any(~torch.isfinite(result))):
        raise ValueError("energy is unresolved at this precision near the photon orbit")
    return result


def hybrid_meco_velocity(
    mass1: Real, mass2: Real, chi1: Real = 0., chi2: Real = 0., *,
    pn_order: float = 3.5, scan_points: int = 256, bisection_steps: int = 64,
) -> torch.Tensor:
    """First stable minimum of the hybrid energy, in PN velocity units.

    Scan below the photon orbit for the first negative-to-positive E'
    crossing, bisect it, then apply three differentiable Newton refinements.
    Autograd first and second derivatives follow the stationary point;
    bracket selection is discrete. A failed bracket, nonpositive curvature,
    or unresolved root raises RuntimeError instead of returning a cutoff.
    The scan is a numerical search, not a proof of uniqueness. Increasing
    scan_points checks for overlooked extrema; increasing bisection_steps
    checks root convergence. Parameter conventions match hybrid_binding_energy.

    Float32 calculations are internally promoted to float64 for the root,
    then restored. Float64 must be available on the input device. The upper
    effective-spin limit excludes the degenerate extremal prograde orbit.
    """
    for name, value, minimum in (("scan_points", scan_points, 32),
                                 ("bisection_steps", bisection_steps, 20)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    arguments = _real_tensors(mass1, mass2, chi1, chi2)
    output_dtype = arguments[0].dtype
    arguments = tuple(value.to(torch.float64) for value in arguments)
    chi, coefficients = _parameters(*arguments, pn_order)
    with torch.no_grad():
        upper = (1 - 1e-5) * _photon_velocity(chi)
        fractions = torch.linspace(1e-4, 1., scan_points, dtype=chi.dtype, device=chi.device)
        grid = upper.unsqueeze(-1) * fractions
        slope = _energy(grid, chi.unsqueeze(-1), tuple(c.unsqueeze(-1) for c in coefficients), 1)
        crossings = (slope[..., :-1] < 0) & (slope[..., 1:] >= 0)
        if bool(torch.any(~crossings.any(-1))):
            raise RuntimeError("hybrid MECO search did not bracket an energy minimum")
        index = crossings.to(torch.int64).argmax(-1, keepdim=True)
        low = grid.gather(-1, index).squeeze(-1)
        high = grid.gather(-1, index + 1).squeeze(-1)
        for _ in range(bisection_steps):
            middle = (low + high) / 2
            negative = _energy(middle, chi, coefficients, 1) < 0
            low = torch.where(negative, middle, low)
            high = torch.where(negative, high, middle)
        velocity = (low + high) / 2
    # At a converged root, repeated Newton steps recover implicit first and
    # second derivatives without differentiating a sign-based bisection.
    for _ in range(3):
        velocity = velocity - _energy(velocity, chi, coefficients, 1) / _energy(
            velocity, chi, coefficients, 2)
    curvature = _energy(velocity, chi, coefficients, 2)
    residual = _energy(velocity, chi, coefficients, 1)
    invalid = (~torch.isfinite(velocity) | ~torch.isfinite(curvature)
               | ~torch.isfinite(residual) | (curvature <= 0)
               | (velocity <= 0) | (velocity >= upper)
               | (residual.abs() > 2e-11 * (1 + curvature.abs())))
    if bool(torch.any(invalid)):
        raise RuntimeError("hybrid MECO root failed the minimum/convergence checks")
    return velocity.to(output_dtype)


def hybrid_meco_frequency(
    mass1: Real, mass2: Real, chi1: Real = 0., chi2: Real = 0., **kwargs,
) -> torch.Tensor:
    """Dimensionless dominant-mode frequency M*f_22 = v_MECO**3/pi.

    See hybrid_meco_velocity for parameter conventions and numerical controls.
    Divide by the total mass in seconds to obtain Hz.
    """
    return hybrid_meco_velocity(mass1, mass2, chi1, chi2, **kwargs)**3 / math.pi
