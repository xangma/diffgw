"""Angular Teukolsky eigensolutions from a spherical spectral expansion.

Reconstructed from Cook & Zalutskiy, arXiv:1410.7698v2, Eqs. (4),
(49)--(56). No numerical waveform implementation is used. The phase is
fixed by a real-positive spherical coefficient at the requested ell.
"""

import math

import torch


def _angular_matrix(m: int, c: torch.Tensor, ell_max: int) -> torch.Tensor:
    """Represent A0 - c² x² + 2cs x, for spin weight s=-2."""
    ell_min = max(2, abs(m))
    # Include one extra basis function before squaring x. Truncating x first
    # would incorrectly omit the last diagonal contribution of x².
    ell = torch.arange(ell_min, ell_max + 2, dtype=c.real.dtype, device=c.device)
    diagonal = 2.0 * m / (ell * (ell + 1.0))
    upper_ell = ell[1:]
    off_diagonal = torch.sqrt(
        (upper_ell.square() - m * m) * (upper_ell.square() - 4.0)
        / (upper_ell.square() * (4.0 * upper_ell.square() - 1.0))
    )
    x = torch.diag(diagonal) + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
    x2 = (x @ x)[:-1, :-1]
    x = x[:-1, :-1]
    a0 = torch.diag(ell[:-1] * (ell[:-1] + 1.0) - 2.0)
    return a0 - c[..., None, None].square() * x2 - 4.0 * c[..., None, None] * x


def spheroidal_coefficients(
    ell: int, m: int, c: torch.Tensor, *, ell_max: int = 18,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the separation constant A and unit-norm spherical coefficients.

    The final axis of C runs from max(2, |m|) to ell_max. A branch is followed
    from c=0 along a straight path, as prescribed after Eq. (56). This solver
    is restricted to |c|<=2, encompassing the supported low Kerr overtones;
    a small spectral tail is required. Parameter batches and gradients are
    supported. Angular eigenvalues must remain simple along that path.
    """
    if ell < max(2, abs(m)) or ell_max < ell + 3:
        raise ValueError("Require ell >= max(2, |m|) and ell_max >= ell+3.")
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.complex128)
    elif c.dtype not in (torch.complex64, torch.complex128):
        c = c.to(torch.complex128 if c.dtype == torch.float64 else torch.complex64)
    if not bool(torch.all(torch.isfinite(c) & (c.abs() <= 2.0))):
        raise ValueError("Angular solver requires finite |a*omega| <= 2.")

    # Discrete branch labels have no derivative. Evaluate the selected final
    # eigensolution with autograd enabled, including its phase normalization.
    with torch.no_grad():
        previous = torch.full_like(c, ell * (ell + 1) - 2)
        for step in range(1, 8):
            values = torch.linalg.eigvals(_angular_matrix(m, c * (step / 8.0), ell_max))
            nearest = (values - previous[..., None]).abs().argmin(-1)
            previous = values.gather(-1, nearest[..., None]).squeeze(-1)
    values, vectors = torch.linalg.eig(_angular_matrix(m, c, ell_max))
    nearest = (values - previous[..., None]).abs().argmin(-1)
    a = values.gather(-1, nearest[..., None]).squeeze(-1)
    indices = nearest[..., None, None].expand(*vectors.shape[:-1], 1)
    coefficients = vectors.gather(-1, indices).squeeze(-1)
    anchor = coefficients[..., ell - max(2, abs(m))]
    coefficients = coefficients * (anchor.conj() / anchor.abs())[..., None]
    tolerance = 1e-10 if c.dtype == torch.complex128 else 2e-5
    if bool(torch.any(torch.linalg.vector_norm(coefficients[..., -2:], dim=-1) > tolerance)):
        raise ValueError("Angular spectral tail is too large; increase ell_max.")
    return a, coefficients


def spin_weighted_spherical_harmonic(
    ell: int, m: int, theta: torch.Tensor, phi: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Evaluate spin -2 Y_lm using the finite Wigner-d factorial sum.

    Y_lm = sqrt((2l+1)/(4pi)) d^l_{m,2}(theta) exp(i m phi), with
    the Condon--Shortley convention, including Y_22(0)=sqrt(5/(4pi)).
    """
    if ell < 2 or abs(m) > ell:
        raise ValueError("Spin -2 harmonics require ell>=2 and |m|<=ell.")
    cosine, sine = torch.cos(theta / 2.0), torch.sin(theta / 2.0)
    value = torch.zeros_like(theta)
    fact = math.factorial
    numerator = math.sqrt(fact(ell + m) * fact(ell - m) * fact(ell + 2) * fact(ell - 2))
    for k in range(max(0, m - 2), min(ell + m, ell - 2) + 1):
        denominator = fact(ell + m - k) * fact(k) * fact(ell - 2 - k) * fact(k - m + 2)
        value = value + ((-1) ** k * numerator / denominator) * (
            cosine.pow(2 * ell + m - 2 - 2 * k) * sine.pow(2 * k - m + 2)
        )
    azimuth = torch.as_tensor(phi, dtype=theta.dtype, device=theta.device) * m
    return math.sqrt((2 * ell + 1) / (4.0 * math.pi)) * value * torch.exp(1j * azimuth)


def spin_weighted_spheroidal_harmonic(
    ell: int, m: int, c: torch.Tensor, theta: torch.Tensor,
    phi: torch.Tensor | float = 0.0, *, ell_max: int = 18,
) -> torch.Tensor:
    """Evaluate spin -2 S_lm(theta,phi;c), normalized over the sphere.

    c, theta, and phi use ordinary elementwise broadcasting. To evaluate a
    batch on an independent angular grid, append a singleton dimension to c.
    """
    _, coefficients = spheroidal_coefficients(ell, m, c, ell_max=ell_max)
    basis = torch.stack([
        spin_weighted_spherical_harmonic(j, m, theta, phi)
        for j in range(max(2, abs(m)), ell_max + 1)
    ], dim=-1)
    return (coefficients * basis).sum(-1)


def leaver_frobenius_coefficients(
    ell: int, c: torch.Tensor, *, n_terms: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaver Frobenius solution for the ``s=-2, m=2`` angular equation.

    This is an independent implementation of the power series in Leaver,
    *Proc. R. Soc. A* **402**, 285 (1985), Eqs. (16)--(21), specialized to the ``(2,2)``
    gravitational family.  With ``z=(1+cos(theta))/2`` we use

    ``S = exp(c*cos(theta)) (1+cos(theta))**2 sum(a_n z**n)``.

    Substitution in the Teukolsky angular equation gives

    ``(n+1)(n+5)a[n+1] + beta[n]a[n] - 4*c*n*a[n-1] = 0``

    where ``beta[n] = A+c**2+6*c-4-n*(n+5)+4*c*n``.  The regular solution is
    the minimal recurrence solution; a finite determinant supplies its A. The
    convention ``a[0]=1`` fixes phase before the positive real sphere norm
    is applied.  This routine intentionally has no dependency on the
    spherical-harmonic matrix solver above.

    The series is specialized to ``m=2`` because this is the independently
    verified Kerr ringdown mode.  ``n_terms`` is both the determinant and
    coefficient truncation order; increase it for stricter convergence.
    """
    if ell < 2:
        raise ValueError("Require ell >= 2.")
    if n_terms < ell + 8:
        raise ValueError("Require n_terms >= ell+8 for a converged tail.")
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c, dtype=torch.complex128)
    elif c.dtype not in (torch.complex64, torch.complex128):
        c = c.to(torch.complex128 if c.dtype == torch.float64 else torch.complex64)
    if not torch.compiler.is_compiling():
        if not bool(torch.all(torch.isfinite(c))):
            raise ValueError("c must be finite.")

    n = torch.arange(n_terms, dtype=c.real.dtype, device=c.device)
    def recurrence_matrix(cvalue):
        shape = (*cvalue.shape, n_terms, n_terms)
        out = torch.zeros(shape, dtype=cvalue.dtype, device=cvalue.device)
        out[..., torch.arange(n_terms), torch.arange(n_terms)] = (n * (n + 5.0) - 4.0 * cvalue[..., None] * n).to(cvalue.dtype)
        out[..., torch.arange(n_terms - 1), torch.arange(1, n_terms)] = (
            -(n[:-1] + 1.0) * (n[:-1] + 5.0)
        ).to(cvalue.dtype)
        out[..., torch.arange(1, n_terms), torch.arange(n_terms - 1)] = 4.0 * cvalue[..., None] * n[1:]
        return out

    matrix = recurrence_matrix(c)
    # The recurrence determinant is singular when beta's additive constant
    # q=A+c^2+6c-4 equals an eigenvalue of this tridiagonal pencil.
    values, vectors = torch.linalg.eig(matrix)
    target = torch.full_like(c, (ell - 2) * (ell + 3))
    with torch.no_grad():
        previous = target
        for step in range(1, 9):
            vals = torch.linalg.eigvals(recurrence_matrix(c * (step / 8.0)))
            pick = (vals - previous[..., None]).abs().argmin(-1)
            previous = vals.gather(-1, pick[..., None]).squeeze(-1)
    index = (values - previous[..., None]).abs().argmin(-1)
    q = values.gather(-1, index[..., None]).squeeze(-1)
    terms = [torch.ones_like(c)]
    for j in range(n_terms - 1):
        b = q - n[j] * (n[j] + 5.0) + 4.0 * c * n[j]
        if j:
            b = b * terms[j] - 4.0 * c * n[j] * terms[j - 1]
        else:
            b = b * terms[j]
        terms.append(-b / ((j + 1.0) * (j + 5.0)))
    a = torch.stack(terms, dim=-1)
    A = q - c.square() - 6.0 * c + 4.0
    return A, a


def leaver_frobenius_harmonic(
    ell: int, c: torch.Tensor, theta: torch.Tensor,
    phi: torch.Tensor | float = 0.0, *, n_terms: int = 48,
    quadrature_order: int = 256,
    phase_convention: str = "spherical_projection",
) -> torch.Tensor:
    """Evaluate the unit-sphere-norm Leaver series.

    ``phase_convention="spherical_projection"`` (the default) makes the
    requested spherical-harmonic projection real and positive.  The exact
    alternative ``"a0_real_positive"`` retains the Frobenius gauge
    ``a[0]=1`` after normalization; it applies no projection phase rotation.
    Both are gauge conventions for the same equation-defined solution.
    """
    if phase_convention not in {"spherical_projection", "a0_real_positive"}:
        raise ValueError(
            "phase_convention must be 'spherical_projection' or 'a0_real_positive'."
        )
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c, dtype=torch.complex128)
    theta = torch.as_tensor(theta)
    phi = torch.as_tensor(phi, dtype=theta.dtype, device=theta.device)
    # Solve the recurrence only on the caller's c shape. Angular coordinates
    # may be a large grid and must not replicate identical eigensolutions.
    c_shape = c.shape
    _, coefficients = leaver_frobenius_coefficients(ell, c, n_terms=n_terms)
    coefficients_orig = coefficients
    out_shape = torch.broadcast_shapes(c.shape, theta.shape, phi.shape)
    c_b, theta_b, phi_b = torch.broadcast_tensors(c, theta, phi)
    flat_c, flat_theta = c_b.reshape(-1), theta_b.reshape(-1)
    coefficients = coefficients.reshape(c_shape + (n_terms,)).expand(out_shape + (n_terms,))
    coeff_flat = coefficients.reshape(-1, n_terms)
    x = torch.cos(flat_theta)
    z = (1.0 + x) / 2.0
    powers = z[..., None] ** torch.arange(n_terms, dtype=theta.dtype, device=theta.device)
    raw = torch.exp(flat_c * x) * (1.0 + x) ** 2 * (coeff_flat * powers).sum(-1)
    # L2 normalization is evaluated by independent Gauss-Legendre quadrature.
    import numpy as np
    qnodes, qweights = np.polynomial.legendre.leggauss(quadrature_order)
    nodes, weights = torch.from_numpy(qnodes), torch.from_numpy(qweights)
    nodes, weights = nodes.to(theta), weights.to(theta)
    zz = (1.0 + nodes) / 2.0
    pp = zz[..., None] ** torch.arange(n_terms, dtype=theta.dtype, device=theta.device)
    # Norm and phase are per original c value; reduce only over quadrature.
    unique_c = c.reshape(-1)
    rq = torch.exp(unique_c[..., None] * nodes) * (1.0 + nodes) ** 2 * (
        coefficients_orig.reshape(-1, n_terms)[..., None, :] * pp
    ).sum(-1)
    norm = torch.sqrt(2.0 * math.pi * (weights * rq.abs().square()).sum(-1)).real
    if phase_convention == "spherical_projection":
        # Fixed phase: the requested spherical-harmonic coefficient is real
        # positive, matching the c=0 basis.  This is equation-independent
        # gauge selection, not residual-minimizing phase fitting.
        anchor = spin_weighted_spherical_harmonic(
            ell, 2, torch.acos(nodes)
        ).conj()
        projection = 2.0 * math.pi * (weights * rq * anchor).sum(-1) / norm
        fixed_phase = projection.conj() / projection.abs()
    else:
        # The recurrence starts with a[0]=1, hence this is the exact
        # a0-real-positive Frobenius gauge.  Normalization is real positive,
        # so no additional phase factor is needed.
        fixed_phase = torch.ones_like(norm, dtype=coefficients.dtype)
    norm_b = norm.reshape(c_shape).expand(out_shape).reshape(-1)
    phase_b = fixed_phase.reshape(c_shape).expand(out_shape).reshape(-1)
    phase = torch.exp(2j * phi_b.reshape(-1))
    return (raw / norm_b * phase_b * phase).reshape(out_shape)
