"""Independent nonspinning TEOB post-adiabatic inspiral.

This module implements the bounded vacuum specialization of the alternating
post-adiabatic (PA) construction in Nagar and Rettegno,
arXiv:1805.03891v2, Sec. II.  It owns the radial-grid algorithm and uses the
existing :class:`TEOBResumSHamiltonian` and
:class:`TEOBResumSRadiationReaction` only as explicit conservative and
vacuum-radiation dependencies.

The implementation is deliberately scalar in its intrinsic parameters.  A
radial grid is vectorized, but parameter batches are rejected rather than
implicitly broadcast.  The supported sector is zero aligned spin and zero
quadrupolar/octupolar tide.  No merger, NQC, calibration, or waveform routing
is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from torchwave.constants import MSUN_SEC, PI
from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
)


_ALLOWED_ORDERS = (2, 4, 6, 8)
PA_INITIALIZER_POLICIES = ("analytic", "quadratic-value", "direct-quadratic-value")
_DEFAULT_RMIN = 14.0
_DEFAULT_DR = 0.1
_MIN_INTERVALS = 10


class TEOBPostAdiabaticError(ValueError):
    """Raised when a PA profile cannot be constructed from the contract."""


@dataclass(frozen=True)
class TEOBPostAdiabaticEndpoint:
    """Endpoint inherited by a later ODE evolution."""

    state: Tensor
    time: Tensor
    phase: Tensor


@dataclass(frozen=True)
class TEOBPostAdiabaticResult:
    """A scalar-intrinsic PA profile on a descending radial grid.

    ``state`` has final axis ``(r, phase, p_r*, p_phi)``.  ``frequency`` and
    ``omega`` are the dimensionless orbital angular frequency
    ``d phase / d t``.  ``time`` and ``phase`` include the caller supplied
    offsets.  A result with ``skipped=True`` contains the initialized radial
    grid and state but has no PA endpoint; callers should then choose their
    ODE initial-condition route.
    """

    radius: Tensor
    time: Tensor
    phase: Tensor
    pr_star: Tensor
    p_phi: Tensor
    omega: Tensor
    radial_velocity: Tensor
    pr_star_rate: Tensor
    torque: Tensor
    p_prime: Tensor
    j_prime: Tensor
    radial_balance_residual: Tensor
    angular_balance_residual: Tensor
    initializer_policy: str
    pa_order: int
    grid_spacing: float
    r_min: float
    start_radius: Tensor
    status: str
    skip_reason: Optional[str]
    endpoint_state: Optional[Tensor]
    endpoint_time: Optional[Tensor]
    endpoint_phase: Optional[Tensor]

    @property
    def r(self) -> Tensor:
        """Alias for ``radius``."""
        return self.radius

    @property
    def p(self) -> Tensor:
        """Alias for ``p_r*``."""
        return self.pr_star

    @property
    def p_r_star(self) -> Tensor:
        """Descriptive alias for ``pr_star``."""
        return self.pr_star

    @property
    def j(self) -> Tensor:
        """Alias for ``p_phi``."""
        return self.p_phi

    @property
    def pphi(self) -> Tensor:
        """Compatibility alias for ``p_phi``."""
        return self.p_phi

    @property
    def frequency(self) -> Tensor:
        """Dimensionless orbital frequency profile."""
        return self.omega

    @property
    def dphi_dt(self) -> Tensor:
        """Alias for the orbital frequency profile."""
        return self.omega

    @property
    def state(self) -> Tensor:
        """Canonical state profile ``(r, phase, p_r*, p_phi)``."""
        return torch.stack((self.radius, self.phase, self.pr_star, self.p_phi), dim=-1)

    @property
    def initial_state(self) -> Tensor:
        """First state on the radial grid, useful when PA was skipped."""
        return self.state[0]

    @property
    def endpoint(self) -> Optional[TEOBPostAdiabaticEndpoint]:
        """Endpoint bundle, or ``None`` for an explicitly skipped profile."""
        if self.endpoint_state is None:
            return None
        return TEOBPostAdiabaticEndpoint(
            state=self.endpoint_state,
            time=self.endpoint_time,
            phase=self.endpoint_phase,
        )

    @property
    def skipped(self) -> bool:
        """Whether the requested profile was too short for the PA stencil."""
        return self.status == "skipped"

    @property
    def complete(self) -> bool:
        """Whether a PA endpoint was produced."""
        return self.status == "complete"

    @property
    def metadata(self) -> dict[str, object]:
        """Grid and order metadata in a compact caller-friendly mapping."""
        return {
            "status": self.status,
            "pa_order": self.pa_order,
            "grid_spacing": self.grid_spacing,
            "r_min": self.r_min,
            "initializer_policy": self.initializer_policy,
            "initializer_value_only": self.initializer_policy in (
                "quadratic-value", "direct-quadratic-value"
            ),
            "start_radius": self.start_radius,
            "num_points": int(self.radius.numel()),
            "num_intervals": max(int(self.radius.numel()) - 1, 0),
            "skip_reason": self.skip_reason,
        }


def _reference_tensor(*values: object) -> Tensor:
    for value in values:
        if isinstance(value, Tensor):
            if value.is_complex():
                raise TypeError("TEOB PA inputs must be real tensors")
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError("TEOB PA inputs must use float32 or float64")
            return value
    return torch.empty((), dtype=torch.get_default_dtype())


def _scalar_tensor(value: object, reference: Tensor, name: str) -> Tensor:
    if isinstance(value, complex):
        raise TypeError(f"{name} must be a real scalar")
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.is_complex():
        raise TypeError(f"{name} must be a real scalar")
    if result.ndim != 0:
        raise ValueError(
            f"{name} must be scalar; batched intrinsic parameters are not supported"
        )
    return result


def _check_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise TEOBPostAdiabaticError(f"{name} is nonfinite")


def _check_scalar_finite(name: str, value: Tensor) -> None:
    _check_finite(name, value)


def circular_angular_momentum_squared(
    A: Tensor,
    Aprime: Tensor,
    r_c: Tensor,
    r_c_prime: Tensor | float = 1.0,
    Gprime: Tensor | float = 0.0,
) -> Tensor:
    """Return the opt-in general circular lower-root value diagnostic.

    This helper evaluates the squared circular stationarity equation from
    arXiv:1806.01772v2, Appendix C, Eq. (58), directly.  With ``x=j**2`` and
    ``U=A/r_c**2``, the coefficients are

    ``a=Uprime**2 - 4*U*Gprime**2``,
    ``b=2*Aprime*Uprime - 4*A*Gprime**2``, and ``c=Aprime**2``,

    where ``Uprime=r_c**-2*(Aprime-2*A*r_c_prime/r_c)``.  The lower numerical
    quadratic root is the explicit compatibility value; in particular this is
    the required value when ``Gprime`` is exactly zero.  The factored
    discriminant is clamped only when this factored radicand
    (``A'² U - A A' U' + A² G'²``) is negative.  The helper is opt-in for
    clean diagnostics and does not replace the analytic initializer in
    :func:`alternating8pa`.

    Inputs broadcast over their common shape and must be real float32/64
    tensors or scalar values.  A positive finite circular root is required.
    This is a value-level diagnostic of the squared equation: it does not
    establish an unsquared stationarity branch or the physical continuation
    of the lower root for a general nonzero ``Gprime``.  The exact
    ``Gprime``-factorized discriminant yields a value-only branch with
    an explicit ``|Gprime|`` cusp at the origin; that cusp is not expected to
    be differentiable in ``Gprime`` itself.
    """
    reference = _reference_tensor(A, Aprime, r_c, r_c_prime, Gprime)
    values: list[Tensor] = []
    for name, value in (
        ("A", A),
        ("A_prime", Aprime),
        ("r_c", r_c),
        ("r_c_prime", r_c_prime),
        ("G_prime", Gprime),
    ):
        if isinstance(value, complex):
            raise TypeError(f"{name} must be real")
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if tensor.is_complex() or tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use real float32/float64 values")
        values.append(tensor)
    A_t, Aprime_t, r_c_t, r_c_prime_t, Gprime_t = torch.broadcast_tensors(*values)
    for name, value in (
        ("A", A_t),
        ("A_prime", Aprime_t),
        ("r_c", r_c_t),
        ("r_c_prime", r_c_prime_t),
        ("G_prime", Gprime_t),
    ):
        _check_finite(name, value)
    if bool(torch.any(A_t <= 0)) or bool(torch.any(r_c_t <= 0)):
        raise TEOBPostAdiabaticError("A and r_c must be positive")

    U = A_t / r_c_t.square()
    Uprime = (
        Aprime_t - 2.0 * A_t * r_c_prime_t / r_c_t
    ) / r_c_t.square()
    Gprime2 = Gprime_t.square()
    a = Uprime.square() - 4.0 * U * Gprime2
    b = 2.0 * Aprime_t * Uprime - 4.0 * A_t * Gprime2
    if bool(torch.any(a == 0)):
        raise TEOBPostAdiabaticError(
            "circular quadratic coefficient is zero; no bounded lower root"
        )
    # Exact factorization:  b² - 4ac = 16 G'² (A'² U - A A' U' + A² G'²).
    # This is algebraically stable near G'=0 and keeps the exact lower-root
    # continuation (|G'| cusp at zero), so we do not claim differentiability
    # in G' at the exact origin.
    factor = (
        Aprime_t.square() * U
        - A_t * Aprime_t * Uprime
        + A_t.square() * Gprime2
    )
    sqrt_factor = torch.sqrt(
        torch.where(
            factor < 0.0,
            torch.zeros_like(factor),
            factor,
        )
    )
    sqrt_discriminant = torch.where(
        Gprime2 == 0.0,
        torch.zeros_like(a),
        4.0 * torch.abs(Gprime_t) * sqrt_factor,
    )
    root_minus = (-b - sqrt_discriminant) / (2.0 * a)
    root_plus = (-b + sqrt_discriminant) / (2.0 * a)
    root = torch.minimum(root_minus, root_plus)
    root = torch.where(Gprime_t == 0.0, -Aprime_t / Uprime, root)
    _check_finite("circular lower-root angular-momentum radicand", root)
    if bool(torch.any(root <= 0)):
        raise TEOBPostAdiabaticError(
            "circular lower-root angular-momentum radicand is nonpositive"
        )
    return root


def circular_angular_momentum_squared_direct(
    A: Tensor,
    Aprime: Tensor,
    r_c: Tensor,
    r_c_prime: Tensor | float = 1.0,
    Gprime: Tensor | float = 0.0,
) -> Tensor:
    """Evaluate the circular quadratic value using its direct discriminant.

    The public EOB stationarity equation is quadratic in ``x=j**2`` with
    ``a=Uprime**2 - 4*U*Gprime**2``, ``b=2*Aprime*Uprime -
    4*A*Gprime**2``, and ``c=Aprime**2``.  This compatibility helper forms
    ``b**2 - 4*a*c`` directly in the requested Torch dtype, clamps a negative
    roundoff result to zero, and chooses the minus root for nonpositive
    ``Gprime`` and the plus root for positive ``Gprime``.  It is value-only
    when selected through ``initializer_policy='direct-quadratic-value'``.

    The numerical intermediate ``u=1/r_c`` gives
    ``U=A*u**2`` and ``Uprime=u**2*(Aprime - 2*A*u*r_c_prime)``.  The direct
    arithmetic is intentionally separate from the stabilized
    :func:`circular_angular_momentum_squared` helper.
    """
    reference = _reference_tensor(A, Aprime, r_c, r_c_prime, Gprime)
    values: list[Tensor] = []
    for name, value in (
        ("A", A),
        ("A_prime", Aprime),
        ("r_c", r_c),
        ("r_c_prime", r_c_prime),
        ("G_prime", Gprime),
    ):
        if isinstance(value, complex):
            raise TypeError(f"{name} must be real")
        if isinstance(value, Tensor) and value.is_complex():
            raise TypeError(f"{name} must be real")
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if tensor.is_complex() or tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use real float32/float64 values")
        values.append(tensor)
    A_t, Aprime_t, r_c_t, r_c_prime_t, Gprime_t = torch.broadcast_tensors(*values)
    for name, value in (
        ("A", A_t),
        ("A_prime", Aprime_t),
        ("r_c", r_c_t),
        ("r_c_prime", r_c_prime_t),
        ("G_prime", Gprime_t),
    ):
        _check_finite(name, value)
    if bool(torch.any(A_t <= 0)) or bool(torch.any(r_c_t <= 0)):
        raise TEOBPostAdiabaticError("A and r_c must be positive")

    u = torch.reciprocal(r_c_t)
    U = A_t * u.square()
    Uprime = u.square() * (Aprime_t - 2.0 * A_t * u * r_c_prime_t)
    Gprime2 = Gprime_t.square()
    a = Uprime.square() - 4.0 * U * Gprime2
    b = 2.0 * Aprime_t * Uprime - 4.0 * A_t * Gprime2
    c = Aprime_t.square()
    if bool(torch.any(a == 0)):
        raise TEOBPostAdiabaticError(
            "circular quadratic coefficient is zero; no bounded direct root"
        )
    discriminant = b.square() - 4.0 * a * c
    discriminant = torch.where(
        discriminant < 0.0, torch.zeros_like(discriminant), discriminant
    )
    sqrt_discriminant = torch.sqrt(discriminant)
    root_minus = (-b - sqrt_discriminant) / (2.0 * a)
    root_plus = (-b + sqrt_discriminant) / (2.0 * a)
    root = torch.where(Gprime_t > 0.0, root_plus, root_minus)
    _check_finite("direct circular angular-momentum radicand", root)
    if bool(torch.any(root <= 0)):
        raise TEOBPostAdiabaticError(
            "direct circular angular-momentum radicand is nonpositive"
        )
    return root


def _check_hamiltonian(hamiltonian: object) -> TEOBResumSHamiltonian:
    if not isinstance(hamiltonian, TEOBResumSHamiltonian):
        raise TypeError("hamiltonian must be a TEOBResumSHamiltonian")

    tensor_names = (
        "nu", "chi1", "chi2", "lambda1", "lambda2",
        "kappa3_1", "kappa3_2",
    )
    for name in tensor_names:
        value = getattr(hamiltonian, name, None)
        if not isinstance(value, Tensor):
            raise TypeError(f"hamiltonian.{name} must be a tensor")
        if value.ndim != 0:
            raise ValueError(
                "TEOB PA accepts scalar intrinsic parameters; "
                f"hamiltonian.{name} has shape {tuple(value.shape)}"
            )
        if value.is_complex() or value.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "TEOB PA Hamiltonian parameters must be real float32/float64 tensors"
            )
        _check_finite(f"hamiltonian.{name}", value)

    if bool(hamiltonian.nu <= 0) or bool(hamiltonian.nu > 0.25):
        raise TEOBPostAdiabaticError("hamiltonian.nu must lie in (0, 1/4]")
    for name in ("chi1", "chi2", "lambda1", "lambda2", "kappa3_1", "kappa3_2"):
        if bool(getattr(hamiltonian, name) != 0):
            raise TEOBPostAdiabaticError(
                "bounded TEOB PA supports only zero spins and zero tides"
            )
    return hamiltonian


def _validate_descending_grid(radius: Tensor, *, minimum_points: int = 1) -> None:
    if not isinstance(radius, Tensor):
        raise TypeError("radius must be a torch.Tensor")
    if radius.is_complex() or radius.dtype not in (torch.float32, torch.float64):
        raise TypeError("radius must use a real float32/float64 dtype")
    if radius.ndim != 1:
        raise ValueError("radius must be one-dimensional")
    if radius.numel() < minimum_points:
        raise ValueError(f"radius must contain at least {minimum_points} points")
    _check_finite("radius", radius)
    if radius.numel() > 1 and bool(torch.any(radius[1:] >= radius[:-1])):
        raise ValueError("radius must be strictly descending")


def _coerce_grid_argument(value: object, reference: Tensor, name: str) -> Tensor:
    result = _scalar_tensor(value, reference, name)
    _check_scalar_finite(name, result)
    return result


def _quartic_stencil_indices(n: int, *, device: torch.device) -> Tensor:
    centers = torch.arange(n, device=device, dtype=torch.long)
    starts = torch.clamp(centers - 2, min=0, max=n - 5)
    offsets = torch.arange(5, device=device, dtype=torch.long)
    return starts[:, None] + offsets[None, :]


def _exact_derivative_weights(radius: Tensor) -> tuple[Tensor, Tensor]:
    """Return five-node derivative weights and their vectorized indices.

    For each target node, the rows of the local Vandermonde system impose
    exactness on ``1, x, x**2, x**3, x**4``.  Centered five-node stencils are
    used where available; the first and last five nodes serve the boundaries.
    """
    _validate_descending_grid(radius, minimum_points=5)
    n = int(radius.numel())
    indices = _quartic_stencil_indices(n, device=radius.device)
    nodes = radius[indices]
    offsets = nodes - radius[:, None]
    # Solve in a local dimensionless coordinate.  Besides expressing the
    # polynomial exactness equations directly, this avoids an ill-conditioned
    # Vandermonde solve when a float32 grid has a small physical spacing.
    scale = offsets.abs().amax(dim=-1).clamp_min(torch.finfo(radius.dtype).tiny)
    local_offsets = offsets / scale[:, None]
    powers = torch.stack([local_offsets.pow(power) for power in range(5)], dim=-2)
    rhs = torch.zeros((n, 5), dtype=radius.dtype, device=radius.device)
    rhs[:, 1] = 1.0
    weights = torch.linalg.solve(powers, rhs.unsqueeze(-1)).squeeze(-1)
    return weights / scale[:, None], indices


def quartic_derivative(values: Tensor, radius: Tensor | float) -> Tensor:
    """Differentiate values with a local quartic through five grid nodes.

    ``radius`` is normally the descending one-dimensional coordinate grid.
    A scalar second argument is interpreted as a spacing and is accepted for
    small standalone polynomial tests; it must be negative for a descending
    grid.  Leading axes on ``values`` are preserved.
    """
    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.is_complex() or values.dtype not in (torch.float32, torch.float64):
        raise TypeError("values must be a real float32/float64 tensor")
    if values.ndim < 1:
        raise ValueError("values must have a grid axis")
    n = int(values.shape[-1])
    reference = _reference_tensor(values, radius)
    radius_input = torch.as_tensor(radius, dtype=reference.dtype, device=reference.device)
    if radius_input.ndim == 0:
        spacing = _scalar_tensor(radius_input, reference, "spacing")
        if not bool(spacing < 0):
            raise ValueError("a scalar spacing must be negative for descending grids")
        radius_t = torch.arange(
            n, dtype=reference.dtype, device=reference.device
        ) * spacing
    else:
        radius_t = radius_input
    if radius_t.ndim != 1:
        raise ValueError("radius must be one-dimensional")
    if int(radius_t.numel()) != n:
        raise ValueError("radius and values must have the same grid length")
    weights, indices = _exact_derivative_weights(radius_t)
    selected = values[..., indices]
    return (selected * weights).sum(dim=-1)


def _cubic_stencil_indices(n: int, *, device: torch.device) -> Tensor:
    intervals = torch.arange(n - 1, device=device, dtype=torch.long)
    if n == 4:
        starts = torch.zeros_like(intervals)
    else:
        starts = torch.where(
            intervals == 0,
            torch.zeros_like(intervals),
            torch.where(
                intervals == n - 2,
                torch.full_like(intervals, n - 4),
                intervals - 1,
            ),
        )
    offsets = torch.arange(4, device=device, dtype=torch.long)
    return starts[:, None] + offsets[None, :]


def _exact_cubic_integral_weights(radius: Tensor) -> tuple[Tensor, Tensor]:
    """Return four-node cubic interval weights and their vectorized indices."""
    _validate_descending_grid(radius, minimum_points=4)
    n = int(radius.numel())
    indices = _cubic_stencil_indices(n, device=radius.device)
    left = radius[:-1]
    right = radius[1:]
    spacing = right - left
    nodes = radius[indices]
    local_x = (nodes - left[:, None]) / spacing[:, None]
    powers = torch.stack([local_x.pow(power) for power in range(4)], dim=-2)
    moments = torch.tensor(
        [1.0, 0.5, 1.0 / 3.0, 0.25],
        dtype=radius.dtype,
        device=radius.device,
    ).expand(n - 1, -1)
    weights = torch.linalg.solve(powers, moments.unsqueeze(-1)).squeeze(-1)
    return weights * spacing[:, None], indices


def cubic_interval_integral(values: Tensor, radius: Tensor) -> Tensor:
    """Integrate a cubic interpolant on every descending-grid interval."""
    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.is_complex() or values.dtype not in (torch.float32, torch.float64):
        raise TypeError("values must be a real float32/float64 tensor")
    if values.ndim < 1:
        raise ValueError("values must have a grid axis")
    radius_t = torch.as_tensor(radius, dtype=values.dtype, device=values.device)
    if radius_t.ndim != 1 or int(radius_t.numel()) != int(values.shape[-1]):
        raise ValueError("radius and values must have the same one-dimensional grid")
    weights, indices = _exact_cubic_integral_weights(radius_t)
    selected = values[..., indices]
    return (selected * weights).sum(dim=-1)


def cumulative_cubic_integral(
    values: Tensor,
    radius: Tensor,
    *,
    offset: Tensor | float = 0.0,
) -> Tensor:
    """Recover a cumulative integral using the four-node cubic rule."""
    intervals = cubic_interval_integral(values, radius)
    reference = _reference_tensor(values, radius, offset)
    offset_t = _scalar_tensor(offset, reference, "offset")
    offset_t = offset_t.to(dtype=values.dtype, device=values.device)
    # ``intervals`` may have arbitrary leading axes.  Build the initial
    # offset column from its first interval so stacked profiles retain their
    # shape instead of trying to concatenate a scalar-shaped column with a
    # matrix.
    offset_column = offset_t + torch.zeros_like(intervals[..., :1])
    cumulative = offset_t + torch.cumsum(intervals, dim=-1)
    return torch.cat((offset_column, cumulative), dim=-1)


# Short aliases are useful in diagnostics and keep the numerical primitives
# discoverable without exposing implementation-specific class internals.
quartic_derivative_weights = _exact_derivative_weights
cubic_integral_weights = _exact_cubic_integral_weights
cubic_integral = cubic_interval_integral
cumulative_cubic_quadrature = cumulative_cubic_integral
_quartic_derivative = quartic_derivative
_cubic_integral = cubic_interval_integral
_cumulative_cubic_integral = cumulative_cubic_integral


@dataclass(frozen=True)
class _PAQuantities:
    A: Tensor
    B: Tensor
    Aprime: Tensor
    U: Tensor
    Uprime: Tensor
    Q: Tensor
    Heff: Tensor
    E: Tensor
    C: Tensor
    radial_velocity: Tensor
    omega: Tensor
    pr_star_rate: Tensor


def _state_quantities(
    hamiltonian: TEOBResumSHamiltonian,
    radius: Tensor,
    pr_star: Tensor,
    p_phi: Tensor,
    A: Tensor,
    B: Tensor,
    Aprime: Tensor,
) -> _PAQuantities:
    """Evaluate the nonspinning equations in the coordinator contract."""
    nu = hamiltonian.nu
    z3 = 2.0 * nu * (4.0 - 3.0 * nu)
    U = A / radius.square()
    Uprime = Aprime / radius.square() - 2.0 * A / radius.pow(3)
    ratio = A / B
    if bool(torch.any(A <= 0)) or bool(torch.any(B <= 0)) or bool(torch.any(ratio <= 0)):
        raise TEOBPostAdiabaticError("metric potentials must remain positive")
    Q = torch.sqrt(ratio)
    heff_radicand = pr_star.square() + A * (
        1.0 + p_phi.square() / radius.square() + z3 * pr_star.pow(4) / radius.square()
    )
    if bool(torch.any(heff_radicand <= 0)):
        raise TEOBPostAdiabaticError("effective Hamiltonian radicand is nonpositive")
    Heff = torch.sqrt(heff_radicand)
    e_radicand = 1.0 + 2.0 * nu * (Heff - 1.0)
    if bool(torch.any(e_radicand <= 0)):
        raise TEOBPostAdiabaticError("real Hamiltonian radicand is nonpositive")
    E = torch.sqrt(e_radicand)
    C = Q * (1.0 + 2.0 * z3 * A * pr_star.square() / radius.square()) / (E * Heff)
    radial_velocity = pr_star * C
    omega = A * p_phi / (radius.square() * E * Heff)
    pr_star_rate = -Q * (
        Aprime + p_phi.square() * Uprime + z3 * pr_star.pow(4) * Uprime
    ) / (2.0 * E * Heff)
    for name, value in (
        ("A", A), ("B", B), ("A_prime", Aprime), ("U_prime", Uprime),
        ("Q", Q), ("Heff", Heff), ("E", E), ("C", C),
        ("radial_velocity", radial_velocity), ("omega", omega),
        ("pr_star_rate", pr_star_rate),
    ):
        _check_finite(name, value)
    if bool(torch.any(omega <= 0)):
        raise TEOBPostAdiabaticError("orbital frequency must remain positive")
    return _PAQuantities(
        A=A, B=B, Aprime=Aprime, U=U, Uprime=Uprime, Q=Q,
        Heff=Heff, E=E, C=C, radial_velocity=radial_velocity,
        omega=omega, pr_star_rate=pr_star_rate,
    )


def _vacuum_torque(
    radiation_reaction: object,
    quantities: _PAQuantities,
    radius: Tensor,
    pr_star: Tensor,
    p_phi: Tensor,
    hamiltonian: TEOBResumSHamiltonian,
    include_horizon_absorption: bool,
) -> Tensor:
    """Call the reviewed clean vacuum torque with no NQC source."""
    try:
        result = radiation_reaction.radiation_reaction_forces(
            quantities.omega,
            pr_star,
            p_phi,
            r=radius,
            hamiltonian=hamiltonian,
            include_horizon_absorption=include_horizon_absorption,
        )
    except TypeError as exc:
        # A small custom test double may expose the older public signature;
        # the supplied production reaction still receives the explicit policy
        # above.  No alternate radiation model is selected here.
        if "include_horizon_absorption" not in str(exc):
            raise
        result = radiation_reaction.radiation_reaction_forces(
            quantities.omega,
            pr_star,
            p_phi,
            r=radius,
            hamiltonian=hamiltonian,
        )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("radiation_reaction_forces must return (radial_force, torque)")
    radial_force, torque = result
    if not isinstance(torque, Tensor):
        raise TypeError("radiation torque must be a torch.Tensor")
    radial_force = torch.as_tensor(
        radial_force, dtype=radius.dtype, device=radius.device,
    )
    if torch.any(radial_force != 0):
        raise TEOBPostAdiabaticError("vacuum PA requires zero radial radiation force")
    torque = torch.broadcast_to(torque, radius.shape)
    _check_finite("radiation torque", torque)
    return torque


def _resolve_order(pa_order: int, order: Optional[int]) -> int:
    if order is not None:
        if pa_order != 8 and pa_order != order:
            raise ValueError("pa_order and order disagree")
        pa_order = order
    if isinstance(pa_order, bool) or not isinstance(pa_order, int):
        raise TypeError("pa_order must be one of 2, 4, 6, or 8")
    if pa_order not in _ALLOWED_ORDERS:
        raise ValueError("pa_order must be one of 2, 4, 6, or 8")
    return pa_order


def _resolve_initializer_policy(initializer_policy: str) -> str:
    """Validate the named PA angular-momentum initializer policy."""
    if not isinstance(initializer_policy, str):
        raise TypeError(
            "initializer_policy must be one of "
            f"{PA_INITIALIZER_POLICIES}"
        )
    if initializer_policy not in PA_INITIALIZER_POLICIES:
        raise ValueError(
            "initializer_policy must be one of "
            f"{PA_INITIALIZER_POLICIES}; got {initializer_policy!r}"
        )
    return initializer_policy


def _reject_tracked_quadratic_initializer(
    hamiltonian: TEOBResumSHamiltonian,
    *values: object,
) -> None:
    """Keep the squared-root probe explicitly value-only.

    The squared stationarity equation has a branch choice and can have
    non-finite derivatives at a zero-spin discriminant crossing.  This probe
    therefore has no autograd contract; callers needing gradients must use
    the analytic initializer or a separately derived unsquared branch.
    """
    names = (
        "nu", "chi1", "chi2", "lambda1", "lambda2", "kappa3_1", "kappa3_2",
    )
    tracked_hamiltonian = any(
        getattr(hamiltonian, name).requires_grad for name in names
    )
    tracked_values = any(
        isinstance(value, Tensor) and value.requires_grad for value in values
    )
    if tracked_hamiltonian or tracked_values:
        raise TEOBPostAdiabaticError(
            "quadratic initializer is value-only and does not support "
            "autograd-tracked inputs"
        )


def _resolve_start_radius(
    hamiltonian: TEOBResumSHamiltonian,
    reference: Tensor,
    *,
    f_start: object | None,
    omega_start: object | None,
    r0: object | None,
    total_mass_msun: object | None,
    total_mass: object | None,
    mass_total: object | None,
    mass1: object | None,
    mass2: object | None,
    r_min: Tensor,
) -> tuple[Tensor, Tensor, str]:
    supplied_frequency = sum(value is not None for value in (f_start, omega_start))
    if supplied_frequency > 1:
        raise ValueError("supply only one of f_start and omega_start")
    if r0 is not None and supplied_frequency:
        raise ValueError("r0 cannot be combined with f_start or omega_start")

    if r0 is not None:
        start_radius = _scalar_tensor(r0, reference, "r0")
        _check_scalar_finite("r0", start_radius)
        if bool(start_radius < r_min):
            raise ValueError("r0 must be at least r_min")
        return start_radius, torch.zeros_like(start_radius), "explicit_r0"

    if f_start is not None:
        frequency = _scalar_tensor(f_start, reference, "f_start")
        _check_scalar_finite("f_start", frequency)
        if bool(frequency <= 0):
            raise ValueError("f_start must be finite and positive")

        masses = [value for value in (total_mass_msun, total_mass, mass_total) if value is not None]
        if len(masses) > 1:
            raise ValueError("specify only one total-mass argument")
        if mass1 is not None or mass2 is not None:
            if mass1 is None or mass2 is None:
                raise ValueError("mass1 and mass2 must be supplied together")
            if masses:
                raise ValueError("do not combine component masses with total mass")
            m1 = _scalar_tensor(mass1, reference, "mass1")
            m2 = _scalar_tensor(mass2, reference, "mass2")
            if bool(m1 <= 0) or bool(m2 <= 0):
                raise ValueError("component masses must be finite and positive")
            total = m1 + m2
        else:
            if not masses:
                raise ValueError(
                    "f_start requires total_mass_msun or both component masses"
                )
            total = _scalar_tensor(masses[0], reference, "total_mass_msun")
            if bool(total <= 0):
                raise ValueError("total_mass_msun must be finite and positive")
        omega = PI * total * MSUN_SEC * frequency
        _check_scalar_finite("omega_start", omega)
        if bool(omega <= 0):
            raise ValueError("omega_start must be finite and positive")
        kepler_radius = omega.pow(-2.0 / 3.0)
        start_radius = torch.maximum(r_min, kepler_radius)
        return start_radius, omega, "f_start"

    if omega_start is not None:
        omega = _scalar_tensor(omega_start, reference, "omega_start")
        _check_scalar_finite("omega_start", omega)
        if bool(omega <= 0):
            raise ValueError("omega_start must be finite and positive")
        start_radius = torch.maximum(r_min, omega.pow(-2.0 / 3.0))
        return start_radius, omega, "omega_start"

    raise ValueError("one of r0, omega_start, or f_start is required")


def _make_skipped_result(
    radius: Tensor,
    quantities: _PAQuantities,
    p_phi: Tensor,
    *,
    start_time: Tensor,
    start_phase: Tensor,
    pa_order: int,
    initializer_policy: str,
    grid_spacing: float,
    r_min: float,
    start_radius: Tensor,
    reason: str,
) -> TEOBPostAdiabaticResult:
    zeros = torch.zeros_like(radius)
    # Addition keeps caller supplied scalar offsets attached to autograd while
    # broadcasting them over the initialized radial profile.
    time = start_time + torch.zeros_like(radius)
    phase = start_phase + torch.zeros_like(radius)
    endpoint_state = None
    return TEOBPostAdiabaticResult(
        radius=radius,
        time=time,
        phase=phase,
        pr_star=zeros,
        p_phi=p_phi,
        omega=quantities.omega,
        radial_velocity=zeros,
        pr_star_rate=zeros,
        torque=zeros,
        p_prime=zeros,
        j_prime=zeros,
        radial_balance_residual=zeros,
        angular_balance_residual=zeros,
        initializer_policy=initializer_policy,
        pa_order=pa_order,
        grid_spacing=grid_spacing,
        r_min=r_min,
        start_radius=start_radius,
        status="skipped",
        skip_reason=reason,
        endpoint_state=endpoint_state,
        endpoint_time=None,
        endpoint_phase=None,
    )


def alternating8pa(
    hamiltonian: TEOBResumSHamiltonian,
    radiation_reaction: Optional[object] = None,
    f_start: object | None = None,
    *,
    omega_start: object | None = None,
    r0: object | None = None,
    r_min: object = _DEFAULT_RMIN,
    dr: object = _DEFAULT_DR,
    pa_order: int = 8,
    order: Optional[int] = None,
    total_mass_msun: object | None = None,
    total_mass: object | None = None,
    mass_total: object | None = None,
    mass1: object | None = None,
    mass2: object | None = None,
    start_time: object = 0.0,
    start_phase: object = 0.0,
    include_horizon_absorption: bool = True,
    initializer_policy: str = "analytic",
) -> TEOBPostAdiabaticResult:
    """Construct a bounded scalar nonspinning vacuum PA profile.

    The start can be supplied as an explicit ``r0`` (the useful bounded
    diagnostic form), as dimensionless ``omega_start``, or as physical
    ``f_start`` in Hz.  For ``f_start``, pass ``total_mass_msun`` or
    ``mass1``/``mass2`` so the Kepler mapping has an explicit physical mass.

    ``initializer_policy='analytic'`` is the canonical differentiable
    ``sqrt(-A'/U')`` initializer.  The opt-in ``'quadratic-value'`` policy
    evaluates the stabilized lower-root diagnostic.  The opt-in
    ``'direct-quadratic-value'`` policy evaluates the same published
    quadratic using its direct binary64/float32 discriminant and explicit
    root branch.  Both quadratic policies are value-only and run under
    ``no_grad``; use the analytic policy for autograd.

    ``pa_order`` supports 2, 4, 6, and 8 alternating orders.  The grid has
    ``floor((r0-r_min)/dr)+1`` descending nodes.  Profiles with fewer than
    ten intervals return ``status='skipped'`` so a caller can select an ODE
    initial-condition route explicitly.
    """
    hamiltonian = _check_hamiltonian(hamiltonian)
    initializer_policy = _resolve_initializer_policy(initializer_policy)
    if initializer_policy in ("quadratic-value", "direct-quadratic-value"):
        _reject_tracked_quadratic_initializer(hamiltonian)
    if not isinstance(include_horizon_absorption, bool):
        raise TypeError("include_horizon_absorption must be a bool")
    pa_order = _resolve_order(pa_order, order)
    if radiation_reaction is None:
        radiation_reaction = TEOBResumSRadiationReaction(
            nu=hamiltonian.nu,
            chi1=hamiltonian.chi1,
            chi2=hamiltonian.chi2,
            include_horizon_absorption=include_horizon_absorption,
        )
    if not hasattr(radiation_reaction, "radiation_reaction_forces"):
        raise TypeError("radiation_reaction must provide radiation_reaction_forces")
    if hasattr(radiation_reaction, "include_horizon_absorption"):
        radiation_reaction.include_horizon_absorption = include_horizon_absorption

    reference = hamiltonian.nu
    r_min_t = _coerce_grid_argument(r_min, reference, "r_min")
    dr_t = _coerce_grid_argument(dr, reference, "dr")
    if bool(r_min_t <= 0):
        raise ValueError("r_min must be finite and positive")
    if bool(dr_t <= 0):
        raise ValueError("dr must be finite and positive")
    start_time_t = _scalar_tensor(start_time, reference, "start_time")
    start_phase_t = _scalar_tensor(start_phase, reference, "start_phase")
    _check_scalar_finite("start_time", start_time_t)
    _check_scalar_finite("start_phase", start_phase_t)

    start_radius, resolved_omega, start_source = _resolve_start_radius(
        hamiltonian,
        reference,
        f_start=f_start,
        omega_start=omega_start,
        r0=r0,
        total_mass_msun=total_mass_msun,
        total_mass=total_mass,
        mass_total=mass_total,
        mass1=mass1,
        mass2=mass2,
        r_min=r_min_t,
    )
    del resolved_omega, start_source
    if initializer_policy in ("quadratic-value", "direct-quadratic-value"):
        _reject_tracked_quadratic_initializer(
            hamiltonian, r_min_t, dr_t, start_time_t, start_phase_t, start_radius,
        )
    if bool(start_radius < r_min_t):
        raise ValueError("start radius must be at least r_min")

    # The integer grid length is metadata.  Keep the tensor-valued first node
    # in the profile so frequency and parameter gradients remain connected.
    ratio = (start_radius - r_min_t) / dr_t
    _check_scalar_finite("grid interval count", ratio)
    n_intervals = int(torch.floor(ratio).detach().cpu().item())
    if n_intervals < 0:
        raise ValueError("start radius must not be below r_min")
    n_points = n_intervals + 1
    # Reject a huge grid before materializing its node index and potential
    # arrays; bounded PA should fail explicitly rather than allocate blindly.
    if n_points > 200_000:
        raise TEOBPostAdiabaticError("requested PA radial grid is too large")
    node_index = torch.arange(
        n_points, dtype=reference.dtype, device=reference.device
    )
    radius = start_radius - node_index * dr_t
    _validate_descending_grid(radius)

    A, B, _ = hamiltonian.metric_potentials(radius)
    Aprime = hamiltonian.metric_potential_derivative(radius)
    for name, value in (("A", A), ("B", B), ("A_prime", Aprime)):
        _check_finite(name, value)
    if bool(torch.any(A <= 0)) or bool(torch.any(B <= 0)):
        raise TEOBPostAdiabaticError("metric potentials must remain positive")
    Uprime = Aprime / radius.square() - 2.0 * A / radius.pow(3)
    if initializer_policy == "analytic":
        circular_radicand = -Aprime / Uprime
    elif initializer_policy == "quadratic-value":
        # This branch is intentionally value-only.  Do not replace it with a
        # straight-through derivative: the lower root is a diagnostic choice,
        # not a generally justified differentiable physical branch.
        with torch.no_grad():
            circular_radicand = circular_angular_momentum_squared(
                A, Aprime, radius, r_c_prime=1.0, Gprime=0.0,
            )
        circular_radicand = circular_radicand.detach()
    else:
        # This branch is intentionally value-only.  It preserves the direct
        # discriminant and explicit branch arithmetic for compatibility probes.
        with torch.no_grad():
            circular_radicand = circular_angular_momentum_squared_direct(
                A, Aprime, radius, r_c_prime=1.0, Gprime=0.0,
            )
        circular_radicand = circular_radicand.detach()
    _check_finite("circular angular-momentum radicand", circular_radicand)
    if bool(torch.any(circular_radicand <= 0)):
        raise TEOBPostAdiabaticError(
            "adiabatic circular angular-momentum radicand is nonpositive"
        )
    p_phi = torch.sqrt(circular_radicand)
    pr_star = torch.zeros_like(radius)
    quantities = _state_quantities(
        hamiltonian, radius, pr_star, p_phi, A, B, Aprime
    )

    if n_intervals < _MIN_INTERVALS:
        return _make_skipped_result(
            radius,
            quantities,
            p_phi,
            start_time=start_time_t,
            start_phase=start_phase_t,
            pa_order=pa_order,
            initializer_policy=initializer_policy,
            grid_spacing=float(dr_t.detach().cpu().item()),
            r_min=float(r_min_t.detach().cpu().item()),
            start_radius=start_radius,
            reason=(
                f"only {n_intervals} radial intervals remain; "
                f"at least {_MIN_INTERVALS} are required for bounded PA"
            ),
        )

    # The initial circular state has no radial motion; the first odd order
    # supplies the inward momentum from the previous circular torque.
    torque = _vacuum_torque(
        radiation_reaction,
        quantities,
        radius,
        pr_star,
        p_phi,
        hamiltonian,
        include_horizon_absorption,
    )
    p_prime = torch.zeros_like(radius)
    j_prime = quartic_derivative(p_phi, radius)
    if bool(torch.any(j_prime == 0)) or not bool(torch.isfinite(j_prime).all()):
        raise TEOBPostAdiabaticError("zero or nonfinite angular balance denominator")

    for current_order in range(1, pa_order + 1):
        if current_order % 2 == 1:
            # Odd order: p_new = F_phi / (j'_old C_old).
            denominator = j_prime * quantities.C
            _check_finite("odd PA balance denominator", denominator)
            if bool(torch.any(denominator == 0)):
                raise TEOBPostAdiabaticError("zero angular balance denominator")
            pr_new = torque / denominator
            _check_finite("odd PA radial momentum", pr_new)
            pr_star = pr_new
        else:
            # Even order: solve the radial balance for j_new.  Every p and
            # p' on the right hand side is from the preceding odd order.
            p_prime = quartic_derivative(pr_star, radius)
            nu = hamiltonian.nu
            z3 = 2.0 * nu * (4.0 - 3.0 * nu)
            correction = 1.0 + 2.0 * z3 * A * pr_star.square() / radius.square()
            angular_radicand = -(
                Aprime
                + 2.0 * pr_star * p_prime * correction
                + z3 * pr_star.pow(4) * Uprime
            ) / Uprime
            _check_finite("even PA angular-momentum radicand", angular_radicand)
            if bool(torch.any(angular_radicand < 0)):
                raise TEOBPostAdiabaticError(
                    "even PA angular-momentum radicand is negative"
                )
            p_phi = torch.sqrt(angular_radicand)
            _check_finite("even PA angular momentum", p_phi)
            if bool(torch.any(p_phi <= 0)):
                raise TEOBPostAdiabaticError("even PA angular momentum is nonpositive")

        quantities = _state_quantities(
            hamiltonian, radius, pr_star, p_phi, A, B, Aprime
        )
        if current_order >= 1 and bool(torch.any(quantities.radial_velocity >= 0)):
            raise TEOBPostAdiabaticError(
                "PA profile contains noninward radial motion"
            )
        torque = _vacuum_torque(
            radiation_reaction,
            quantities,
            radius,
            pr_star,
            p_phi,
            hamiltonian,
            include_horizon_absorption,
        )
        j_prime = quartic_derivative(p_phi, radius)

    # Derivatives in the returned balance diagnostics are always evaluated on
    # the final profile.  The radial equation is solved by each even update;
    # the angular residual records the finite-order lag of the preceding odd
    # update and is useful for order/refinement diagnostics.
    p_prime = quartic_derivative(pr_star, radius)
    j_prime = quartic_derivative(p_phi, radius)
    radial_balance = quantities.pr_star_rate - p_prime * quantities.radial_velocity
    angular_balance = j_prime * quantities.radial_velocity - torque
    _check_finite("radial balance residual", radial_balance)
    _check_finite("angular balance residual", angular_balance)
    if bool(torch.any(quantities.radial_velocity >= 0)):
        raise TEOBPostAdiabaticError("PA profile contains noninward radial motion")

    time_integrand = torch.reciprocal(quantities.radial_velocity)
    phase_integrand = quantities.omega * time_integrand
    _check_finite("time integrand", time_integrand)
    _check_finite("phase integrand", phase_integrand)
    time = cumulative_cubic_integral(time_integrand, radius, offset=start_time_t)
    phase = cumulative_cubic_integral(phase_integrand, radius, offset=start_phase_t)
    _check_finite("PA time", time)
    _check_finite("PA phase", phase)
    if bool(torch.any(time[1:] <= time[:-1])):
        raise TEOBPostAdiabaticError("cumulative PA time is not increasing")
    if bool(torch.any(phase[1:] <= phase[:-1])):
        raise TEOBPostAdiabaticError("cumulative PA phase is not increasing")

    endpoint_state = torch.stack((
        radius[-1], phase[-1], pr_star[-1], p_phi[-1]
    ))
    endpoint_time = time[-1]
    endpoint_phase = phase[-1]
    grid_spacing = float(dr_t.detach().cpu().item())
    r_min_value = float(r_min_t.detach().cpu().item())
    return TEOBPostAdiabaticResult(
        radius=radius,
        time=time,
        phase=phase,
        pr_star=pr_star,
        p_phi=p_phi,
        omega=quantities.omega,
        radial_velocity=quantities.radial_velocity,
        pr_star_rate=quantities.pr_star_rate,
        torque=torque,
        p_prime=p_prime,
        j_prime=j_prime,
        radial_balance_residual=radial_balance,
        angular_balance_residual=angular_balance,
        initializer_policy=initializer_policy,
        pa_order=pa_order,
        grid_spacing=grid_spacing,
        r_min=r_min_value,
        start_radius=start_radius,
        status="complete",
        skip_reason=None,
        endpoint_state=endpoint_state,
        endpoint_time=endpoint_time,
        endpoint_phase=endpoint_phase,
    )


def compute_teob_post_adiabatic(*args, **kwargs) -> TEOBPostAdiabaticResult:
    """Descriptive alias for :func:`alternating8pa`."""
    return alternating8pa(*args, **kwargs)


def integrate_teob_postadiabatic(*args, **kwargs) -> TEOBPostAdiabaticResult:
    """Compatibility alias for the bounded PA profile builder."""
    return alternating8pa(*args, **kwargs)


def teob_postadiabatic(*args, **kwargs) -> TEOBPostAdiabaticResult:
    """Short alias for :func:`alternating8pa`."""
    return alternating8pa(*args, **kwargs)


class TEOBPostAdiabatic:
    """Small stateful wrapper around :func:`alternating8pa`."""

    def __init__(
        self,
        hamiltonian: TEOBResumSHamiltonian,
        radiation_reaction: Optional[object] = None,
        **defaults,
    ):
        self.hamiltonian = _check_hamiltonian(hamiltonian)
        self.radiation_reaction = radiation_reaction
        self.defaults = dict(defaults)

    def integrate(self, **kwargs) -> TEOBPostAdiabaticResult:
        options = dict(self.defaults)
        options.update(kwargs)
        return alternating8pa(
            self.hamiltonian,
            self.radiation_reaction,
            **options,
        )

    __call__ = integrate


# Readable class aliases for callers that name the component after its role.
TEOBVacuumPostAdiabatic = TEOBPostAdiabatic
TEOBPostAdiabaticInspiral = TEOBPostAdiabatic
alternating8PA = alternating8pa


__all__ = [
    "TEOBPostAdiabaticError",
    "TEOBPostAdiabaticEndpoint",
    "TEOBPostAdiabaticResult",
    "TEOBPostAdiabatic",
    "TEOBVacuumPostAdiabatic",
    "TEOBPostAdiabaticInspiral",
    "alternating8pa",
    "alternating8PA",
    "compute_teob_post_adiabatic",
    "integrate_teob_postadiabatic",
    "teob_postadiabatic",
    "quartic_derivative",
    "quartic_derivative_weights",
    "circular_angular_momentum_squared",
    "circular_angular_momentum_squared_direct",
    "cubic_interval_integral",
    "cubic_integral",
    "cubic_integral_weights",
    "cumulative_cubic_integral",
    "cumulative_cubic_quadrature",
]
