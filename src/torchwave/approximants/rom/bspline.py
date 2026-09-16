"""Clean-room tensor-product B-splines and reduced-basis reconstruction.

The published metadata contains parameter grids and coefficient dimensions
with two additional planes.  This module keeps the numerical primitives
independent of any waveform implementation or data file; callers supply
tables explicitly.  The table packing convention remains an explicit loader
assumption pending a complete public-data verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC, PI


TensorLike = Union[torch.Tensor, Sequence[float]]


def linear_interpolate(x: TensorLike, grid: TensorLike, values: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation on a strictly increasing ROM grid.

    ``values`` has its last axis aligned with ``grid``; leading dimensions
    broadcast with ``x``.  Endpoints are included and extrapolation is
    rejected because the published ROM domain ends at its tabulated grids.
    """
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=values.dtype, device=values.device)
    grid = grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid, dtype=values.dtype, device=values.device)
    x = x.to(device=values.device, dtype=values.dtype)
    grid = grid.to(device=values.device, dtype=values.dtype)
    if grid.ndim != 1 or values.shape[-1] != grid.numel() or grid.numel() < 2:
        raise ValueError("grid must be one-dimensional and match values' final axis")
    if torch.any(~torch.isfinite(grid)) or torch.any(grid[1:] <= grid[:-1]):
        raise ValueError("grid must be finite and strictly increasing")
    if torch.any(~torch.isfinite(x)) or torch.any(x < grid[0]) or torch.any(x > grid[-1]):
        raise ValueError("frequency points must lie within the tabulated ROM grid")
    idx = torch.searchsorted(grid, x).clamp(1, grid.numel() - 1)
    lo, hi = idx - 1, idx
    weight = (x - grid[lo]) / (grid[hi] - grid[lo])
    if values.ndim == 1:
        left, right = values[lo], values[hi]
    else:
        if values.shape[:-1] != x.shape:
            raise ValueError("x shape must match values leading dimensions")
        left = torch.gather(values, -1, lo.unsqueeze(-1)).squeeze(-1)
        right = torch.gather(values, -1, hi.unsqueeze(-1)).squeeze(-1)
    return left + weight * (right - left)


def cubic_spline_interpolate(x: TensorLike, grid: TensorLike, values: torch.Tensor) -> torch.Tensor:
    """Evaluate the not-a-knot cubic spline through tabulated values.

    The ROM papers specify cubic splines with a not-a-knot endpoint condition
    for the frequency interpolant.  This implementation solves for the
    second derivatives in torch, so interpolation remains differentiable in
    both query frequencies and (when supplied) tabulated values.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=values.dtype, device=values.device)
    grid = grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid, dtype=values.dtype, device=values.device)
    x, grid = x.to(device=values.device, dtype=values.dtype), grid.to(device=values.device, dtype=values.dtype)
    if grid.ndim != 1 or values.ndim != 1 or values.numel() != grid.numel() or grid.numel() < 2:
        raise ValueError("grid and one-dimensional values must have the same length")
    if torch.any(~torch.isfinite(grid)) or torch.any(grid[1:] <= grid[:-1]):
        raise ValueError("grid must be finite and strictly increasing")
    if torch.any(~torch.isfinite(x)) or torch.any(x < grid[0]) or torch.any(x > grid[-1]):
        raise ValueError("frequency points must lie within the tabulated ROM grid")
    n = grid.numel()
    if n == 2:
        return linear_interpolate(x, grid, values)
    if n == 3:
        # With three knots the not-a-knot conditions collapse to the unique
        # quadratic interpolant; solving this Vandermonde system avoids the
        # singular endpoint system.
        vand = torch.stack((torch.ones_like(grid), grid, grid.square()), dim=-1)
        coef = torch.linalg.solve(vand, values)
        return coef[0] + coef[1] * x + coef[2] * x.square()
    h = grid[1:] - grid[:-1]
    mat = torch.zeros((n, n), dtype=values.dtype, device=values.device)
    rhs = torch.zeros((n,), dtype=values.dtype, device=values.device)
    # Not-a-knot: the first two and last two cubic pieces are identical.
    mat[0, 0], mat[0, 1], mat[0, 2] = h[1], -(h[0] + h[1]), h[0]
    mat[-1, -3], mat[-1, -2], mat[-1, -1] = h[-1], -(h[-2] + h[-1]), h[-2]
    slope = (values[1:] - values[:-1]) / h
    interior = torch.arange(1, n - 1, device=values.device)
    mat[interior, interior - 1] = h[:-1]
    mat[interior, interior] = 2 * (h[:-1] + h[1:])
    mat[interior, interior + 1] = h[1:]
    rhs[1:-1] = 6 * (slope[1:] - slope[:-1])
    second = torch.linalg.solve(mat, rhs)
    idx = torch.searchsorted(grid, x).clamp(1, n - 1)
    j = idx - 1
    dx = x - grid[j]
    hj = h[j]
    a = (grid[j + 1] - x) / hj
    b = dx / hj
    return (a * values[j] + b * values[j + 1]
            + ((a**3 - a) * second[j] + (b**3 - b) * second[j + 1]) * hj**2 / 6)


def _knots(breaks: torch.Tensor, order: int) -> torch.Tensor:
    if breaks.ndim != 1 or breaks.numel() < 2:
        raise ValueError("breaks must be a one-dimensional sequence with at least two values")
    if order < 1:
        raise ValueError("order must be positive")
    if torch.any(~torch.isfinite(breaks)) or torch.any(breaks[1:] <= breaks[:-1]):
        raise ValueError("breaks must be finite and strictly increasing")
    # Open/clamped knot vector.  Thus ncoeff = nbreak + order - 2.
    return torch.cat((breaks[:1].expand(order), breaks[1:-1], breaks[-1:].expand(order)))


def bspline_basis(
    x: TensorLike,
    breaks: TensorLike,
    *,
    order: int = 4,
    validate_domain: bool = True,
) -> torch.Tensor:
    """Evaluate all clamped B-spline basis functions at ``x``.

    The result has shape ``x.shape + (len(breaks) + order - 2,)``.  The right
    endpoint belongs to the final basis function, and values outside the
    closed domain raise ``ValueError`` when validation is enabled.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    elif not x.is_floating_point():
        x = x.to(dtype=torch.get_default_dtype())
    if not isinstance(breaks, torch.Tensor):
        breaks = torch.as_tensor(breaks, dtype=x.dtype, device=x.device)
    else:
        breaks = breaks.to(device=x.device, dtype=x.dtype)
    knots = _knots(breaks, order)
    if validate_domain and (torch.any(~torch.isfinite(x)) or torch.any(x < breaks[0]) or torch.any(x > breaks[-1])):
        raise ValueError(f"evaluation points must lie in [{breaks[0].item()}, {breaks[-1].item()}]")

    # Cox-de Boor recurrence, retaining x's leading dimensions.
    flat = x.reshape(-1, 1)
    ncoeff = breaks.numel() + order - 2
    n0 = knots.numel() - 1
    left, right = knots[:-1], knots[1:]
    values = (flat >= left) & (flat < right)
    # Include the closed right endpoint in the final non-zero interval (the
    # trailing repeated knot spans are zero width).
    endpoint_interval = (right == knots[-1]) & (left < right)
    values = values | ((flat[:, 0] == knots[-1]).unsqueeze(-1) & endpoint_interval)
    values = values.to(x.dtype)
    for degree in range(1, order):
        count = n0 - degree
        out = torch.zeros((flat.shape[0], count), dtype=x.dtype, device=x.device)
        den_left = knots[degree:degree + count] - knots[:count]
        den_right = knots[degree + 1:degree + 1 + count] - knots[1:count + 1]
        if degree == 1:
            prev = values
        else:
            prev = values
        # Replace zero-width spans before division: torch.where evaluates
        # both branches, and raw 0/0 would otherwise poison autograd.
        safe_left = torch.where(den_left != 0, den_left, torch.ones_like(den_left))
        safe_right = torch.where(den_right != 0, den_right, torch.ones_like(den_right))
        a = (flat - knots[:count]) / safe_left
        b = (knots[degree + 1:degree + 1 + count] - flat) / safe_right
        out = torch.where(den_left != 0, a * prev[:, :count], torch.zeros_like(out))
        out = out + torch.where(den_right != 0, b * prev[:, 1:count + 1], torch.zeros_like(out))
        values = out
    return values[:, :ncoeff].reshape(x.shape + (ncoeff,))


def tensor_product_basis(
    eta: TensorLike,
    chi1: TensorLike,
    chi2: TensorLike,
    grids: Tuple[TensorLike, TensorLike, TensorLike],
    *,
    shifts: Optional[Tuple[float, float, float]] = None,
    order: int = 4,
) -> torch.Tensor:
    """Return flattened tensor-product basis, with the last axis C-packed."""
    vals = [eta, chi1, chi2]
    if shifts is None:
        shifts = (0.0, 0.0, 0.0)
    tensors = []
    for value, grid, shift in zip(vals, grids, shifts):
        v = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.get_default_dtype())
        g = grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid, dtype=v.dtype, device=v.device)
        tensors.append(bspline_basis(v + shift, g + shift, order=order))
    a, b, c = tensors
    leading = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1], c.shape[:-1])
    a, b, c = (v.expand(leading + (v.shape[-1],)) for v in (a, b, c))
    return (a[..., :, None, None] * b[..., None, :, None] * c[..., None, None, :]).reshape(
        leading + (a.shape[-1] * b.shape[-1] * c.shape[-1],)
    )


def reconstruct_reduced_basis(basis: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Project coefficients through basis rows shaped ``(n_basis, n_frequency)``."""
    if basis.ndim != 2 or coefficients.shape[-1] != basis.shape[0]:
        raise ValueError("basis must be (n_basis, n_frequency), coefficients must end in n_basis")
    return torch.matmul(coefficients, basis)


class OfficialROMSurrogate:
    """Evaluate one externally supplied official ROM HDF5 submodel.

    Coefficients are currently interpreted in frequency-first, C-contiguous
    order as ``(n_basis, n_eta+2, n_chi1+2, n_chi2+2)``.  Basis rows are
    ``(n_basis, n_frequency)``.  These are explicit
    loader assumption, checked against supplied arrays, pending verification
    against the complete public data metadata.
    """

    def __init__(self, group, *, name: str = ""):
        self.name = name
        self.eta_grid = torch.as_tensor(group["etavec"][...], dtype=torch.float64)
        self.chi1_grid = torch.as_tensor(group["chi1vec"][...], dtype=torch.float64)
        self.chi2_grid = torch.as_tensor(group["chi2vec"][...], dtype=torch.float64)
        self.bamp = torch.as_tensor(group["Bamp"][...], dtype=torch.float64)
        self.bphase = torch.as_tensor(group["Bphase"][...], dtype=torch.float64)
        self.mf_grid_amp = torch.as_tensor(group["Mf_grid_Amp"][...], dtype=torch.float64)
        self.mf_grid_phase = torch.as_tensor(group["Mf_grid_Phi"][...], dtype=torch.float64)
        arrays = (self.eta_grid, self.chi1_grid, self.chi2_grid, self.bamp, self.bphase, self.mf_grid_amp, self.mf_grid_phase)
        if any(torch.any(~torch.isfinite(value)) for value in arrays):
            raise ValueError("ROM grids and basis matrices must contain only finite values")
        parameter_shape = (self.eta_grid.numel() + 2, self.chi1_grid.numel() + 2, self.chi2_grid.numel() + 2)
        amp_shape = (self.bamp.shape[0],) + parameter_shape
        phase_shape = (self.bphase.shape[0],) + parameter_shape
        self.amp_coeff = torch.as_tensor(group["Amp_ciall"][...], dtype=torch.float64)
        self.phase_coeff = torch.as_tensor(group["Phase_ciall"][...], dtype=torch.float64)
        if torch.any(~torch.isfinite(self.amp_coeff)) or torch.any(~torch.isfinite(self.phase_coeff)):
            raise ValueError("ROM coefficient tables must contain only finite values")
        if self.amp_coeff.numel() != int(torch.tensor(amp_shape).prod()) or self.phase_coeff.numel() != int(torch.tensor(phase_shape).prod()):
            raise ValueError("coefficient tables do not match basis and parameter dimensions")
        self.amp_coeff = self.amp_coeff.reshape(amp_shape)
        self.phase_coeff = self.phase_coeff.reshape(phase_shape)

    def evaluate(self, eta: TensorLike, chi1: TensorLike, chi2: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tables are loaded as float64 CPU tensors.  Promote Python/scalar
        # inputs before basis evaluation so conversion never silently starts
        # in float32; callers can move the model tensors for another device.
        target = self.amp_coeff
        def _parameter(value):
            return torch.as_tensor(value, device=target.device, dtype=target.dtype)
        p = tensor_product_basis(_parameter(eta), _parameter(chi1), _parameter(chi2),
                                 (self.eta_grid, self.chi1_grid, self.chi2_grid),
                                 shifts=(0.0, 0.0, 0.0))
        amp_c = torch.einsum("...p,fp->...f", p, self.amp_coeff.reshape(self.amp_coeff.shape[0], -1))
        phase_c = torch.einsum("...p,fp->...f", p, self.phase_coeff.reshape(self.phase_coeff.shape[0], -1))
        return reconstruct_reduced_basis(self.bamp, amp_c), reconstruct_reduced_basis(self.bphase, phase_c)

    def evaluate_frequency(self, mf: TensorLike, eta: TensorLike, chi1: TensorLike, chi2: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return interpolated reduced amplitude and phase at dimensionless ``Mf``."""
        amp_nodes, phase_nodes = self.evaluate(eta, chi1, chi2)
        if amp_nodes.ndim != 1 or phase_nodes.ndim != 1:
            raise ValueError("evaluate_frequency currently supports scalar parameter points only")
        return (cubic_spline_interpolate(mf, self.mf_grid_amp, amp_nodes),
                cubic_spline_interpolate(mf, self.mf_grid_phase, phase_nodes))

    def evaluate_complex(self, mf: TensorLike, eta: TensorLike, chi1: TensorLike, chi2: TensorLike) -> torch.Tensor:
        """Return the dimensionless ROM mode ``amplitude * exp(i phase)``.

        Physical mass/distance scaling, polarization projection, and reference
        time/phase conditioning are intentionally outside this data-only
        evaluator because they are not encoded in the HDF5 metadata.
        """
        amplitude, phase = self.evaluate_frequency(mf, eta, chi1, chi2)
        return torch.polar(amplitude, phase)

    def evaluate_polarizations(
        self, frequency_hz: TensorLike, total_mass_msun: TensorLike,
        distance_mpc: TensorLike, eta: TensorLike, chi1: TensorLike,
        chi2: TensorLike, *, inclination: TensorLike = 0.0,
        phic: TensorLike = 0.0, tc: TensorLike = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return positive-frequency ``(h_plus, h_cross)`` for the ROM mode.

        This is the explicitly scoped dominant-mode convention of Eqs.
        (6.11)--(6.15) in Pürrer (arXiv:1402.4146): the stored complex mode
        is the conjugated ``(2,2)`` mode with positive-frequency power, and
        its missing parameter-dependent normalization is set to one because
        the supplied HDF5 has no ``N_A`` table.  The dimensional factor is
        ``M_sec*(C_SI*M_sec)/(D_mpc*MPC_SI)`` (Fourier strain has seconds).
        ``tc`` and ``phic`` apply the stated
        Fourier time/phase shifts; this method does not claim a hidden ROM
        phase gauge or a higher-mode spin-weighted projection.
        """
        target = self.amp_coeff
        f = torch.as_tensor(frequency_hz, dtype=target.dtype, device=target.device)
        mass = torch.as_tensor(total_mass_msun, dtype=target.dtype, device=target.device)
        distance = torch.as_tensor(distance_mpc, dtype=target.dtype, device=target.device)
        inc = torch.as_tensor(inclination, dtype=target.dtype, device=target.device)
        if torch.any(~torch.isfinite(f)) or torch.any(f < 0):
            raise ValueError("frequency_hz must be finite and non-negative")
        if torch.any(mass <= 0) or torch.any(distance <= 0):
            raise ValueError("total_mass_msun and distance_mpc must be positive")
        mf = f * mass * MSUN_SEC
        mode = self.evaluate_complex(mf, eta, chi1, chi2)
        phase_shift = torch.exp(1j * (2 * PI * f * torch.as_tensor(tc, dtype=target.dtype, device=target.device) - 2 * torch.as_tensor(phic, dtype=target.dtype, device=target.device)))
        m_sec = mass * MSUN_SEC
        scale = m_sec * (C_SI * m_sec) / (distance * MPC_SI)
        mode = mode * scale * phase_shift
        ci = torch.cos(inc)
        hplus = 0.5 * (1 + ci**2) * (0.5 * mode)
        hcross = ci * (0.5j * mode)
        return hplus, hcross


def evaluate_stitched_frequency(
    low: OfficialROMSurrogate, high: OfficialROMSurrogate,
    mf: TensorLike, eta: TensorLike, chi1: TensorLike, chi2: TensorLike,
    *, transition: float = 0.01, ringdown_frequency: Optional[float] = None,
    overlap: Optional[Tuple[float, float]] = (0.0074, 0.014), fit_points: Optional[int] = None,
    fit_nodes: Optional[TensorLike] = None, reference_mf: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Glue low/high official ROM patches using Pürrer's published rule.

    Fit cubic polynomials to both phases on ``overlap`` and apply the
    value/slope affine correction at ``transition`` (arXiv:1512.02248 §IV.1).
    The common affine gauge is selected so ``dphi/dMf`` vanishes at the
    explicit ringdown frequency (Appendix A).
    """
    dtype, device = low.amp_coeff.dtype, low.amp_coeff.device
    outer_grad = torch.is_grad_enabled()
    if ringdown_frequency is None:
        raise ValueError("ringdown_frequency must be supplied explicitly")
    rd = torch.as_tensor(ringdown_frequency, dtype=dtype, device=device)
    if rd.numel() != 1 or not torch.isfinite(rd):
        raise ValueError("ringdown_frequency must be a finite scalar")
    if not (high.mf_grid_phase[0] <= rd <= high.mf_grid_phase[-1]):
        raise ValueError("ringdown_frequency must lie in the high-ROM phase domain")
    tr = torch.as_tensor(transition, dtype=dtype, device=device)
    if tr.numel() != 1 or not torch.isfinite(tr):
        raise ValueError("transition must be a finite scalar")
    if not (low.mf_grid_phase[0] < tr < low.mf_grid_phase[-1] and high.mf_grid_phase[0] < tr < high.mf_grid_phase[-1]):
        raise ValueError("transition must lie inside both ROM phase domains")
    if fit_points is not None and (not isinstance(fit_points, int) or fit_points < 4):
        raise ValueError("fit_points must be an integer >= 4 when supplied")
    if overlap is None:
        lo = max(float(low.mf_grid_phase[0]), float(high.mf_grid_phase[0]))
        hi = min(float(low.mf_grid_phase[-1]), float(high.mf_grid_phase[-1]))
    else:
        if len(overlap) != 2:
            raise ValueError("overlap must contain (lower, upper)")
        lo, hi = map(float, overlap)
    if not (lo < hi and lo <= float(tr) <= hi):
        raise ValueError("overlap must be increasing and contain transition")
    zmid = torch.as_tensor((lo + hi) / 2, dtype=dtype, device=device)
    zscale = torch.as_tensor((hi - lo) / 2, dtype=dtype, device=device)
    def _sparse_fit_grid(model):
        # The fit is made on tabulated sparse frequency nodes.  Selecting
        # nodes from the published grid preserves its nonuniform spacing;
        # coordinates are never replaced by an artificial linspace.
        grid = model.mf_grid_phase
        if fit_nodes is None:
            nodes = grid[(grid >= lo) & (grid <= hi)]
        else:
            nodes = torch.as_tensor(fit_nodes, dtype=dtype, device=device)
            if nodes.ndim != 1 or torch.any(nodes[1:] <= nodes[:-1]) or torch.any(nodes < lo) or torch.any(nodes > hi):
                raise ValueError("fit_nodes must be increasing and inside overlap")
        if fit_points is not None and nodes.numel() != fit_points:
            raise ValueError("fit_points must equal the explicit fit_nodes count")
        if nodes.numel() < 4:
            raise ValueError("overlap must contain at least four tabulated phase nodes")
        return nodes

    # Both cubics use the same high-patch sparse abscissae.  The low phase is
    # evaluated at those abscissae before fitting, as specified by §IV.1.
    fit_x_high = _sparse_fit_grid(high)
    fit_x_low = fit_x_high

    def _fit(model, fit_x):
        z = (fit_x - zmid) / zscale
        design = torch.stack((torch.ones_like(z), z, z.square(), z.square() * z), dim=-1)
        _, values = model.evaluate_frequency(fit_x, eta, chi1, chi2)
        return torch.linalg.lstsq(design, values.unsqueeze(-1)).solution.squeeze(-1)

    cl, ch = _fit(low, fit_x_low), _fit(high, fit_x_high)

    def _poly(c, x):
        zz = (x - zmid) / zscale
        return (c[0] + zz * (c[1] + zz * (c[2] + zz * c[3])),
                (c[1] + zz * (2 * c[2] + zz * 3 * c[3])) / zscale)

    pl, dl = _poly(cl, tr)
    ph, dh = _poly(ch, tr)
    slope = dl - dh
    intercept = pl - ph - slope * tr
    x = torch.as_tensor(mf, dtype=dtype, device=device)
    if torch.any(~torch.isfinite(x)) or torch.any(x < low.mf_grid_amp[0]) or torch.any(x > high.mf_grid_amp[-1]):
        raise ValueError("Mf lies outside the combined ROM frequency domain")
    low_mask = x <= transition
    # Amplitude data are joined at the transition and represented by one
    # cubic spline, as the public ROM construction describes.  This avoids
    # evaluating two independent splines with an artificial branch seam.
    amp_low_grid = low.mf_grid_amp[low.mf_grid_amp <= tr]
    amp_high_grid = high.mf_grid_amp[high.mf_grid_amp > tr]
    if amp_low_grid.numel() < 2 or amp_high_grid.numel() < 2:
        raise ValueError("transition must leave at least two amplitude nodes per patch")
    amp_low_values, _ = low.evaluate_frequency(amp_low_grid, eta, chi1, chi2)
    amp_high_values, _ = high.evaluate_frequency(amp_high_grid, eta, chi1, chi2)
    amp_grid = torch.cat((amp_low_grid, amp_high_grid))
    amp_values = torch.cat((amp_low_values, amp_high_values))
    amp = cubic_spline_interpolate(x, amp_grid, amp_values)
    phase_low_grid = low.mf_grid_phase[low.mf_grid_phase <= tr]
    phase_high_grid = high.mf_grid_phase[high.mf_grid_phase > tr]
    phase_low_values = low.evaluate_frequency(phase_low_grid, eta, chi1, chi2)[1]
    phase_high_values = high.evaluate_frequency(phase_high_grid, eta, chi1, chi2)[1]
    # Apply the fitted affine correction to the high data nodes, then use one
    # interpolant. This makes the actual joined spline the Appendix-A object.
    phase_grid = torch.cat((phase_low_grid, phase_high_grid))
    phase_values = torch.cat((phase_low_values, phase_high_values + intercept + slope * phase_high_grid))
    with torch.enable_grad():
        rd_query = rd.clone().requires_grad_(True)
        joined_rd_phase = cubic_spline_interpolate(rd_query, phase_grid, phase_values)
        joined_rd_slope = torch.autograd.grad(joined_rd_phase, rd_query, create_graph=True)[0]
    gauge = joined_rd_slope
    with torch.enable_grad():
        phase = cubic_spline_interpolate(x, phase_grid, phase_values) - gauge * (x - rd)
        if reference_mf is not None:
            ref = torch.as_tensor(reference_mf, dtype=dtype, device=device)
            if ref.numel() != 1 or not torch.isfinite(ref) or ref < phase_grid[0] or ref > phase_grid[-1]:
                raise ValueError("reference_mf must be a finite scalar in the joined phase domain")
            phase = phase - (cubic_spline_interpolate(ref, phase_grid, phase_values) - gauge * (ref - rd))
    if not outer_grad:
        amp, phase = amp.detach(), phase.detach()
    return amp, phase


def load_official_rom(path: Union[str, Path], group: str = "sub1") -> OfficialROMSurrogate:
    """Load one ROM group from an explicit external HDF5 path."""
    from torchwave.approximants.rom.official import inspect_official_rom
    inspect_official_rom(path, verify_hash=True)
    import h5py
    with h5py.File(Path(path), "r") as handle:
        if group not in handle:
            raise KeyError(f"missing ROM group {group!r}")
        # Copy arrays before closing the file; no data is bundled or cached.
        copied = {k: handle[group][k][...] for k in ("etavec", "chi1vec", "chi2vec", "Bamp", "Bphase", "Mf_grid_Amp", "Mf_grid_Phi", "Amp_ciall", "Phase_ciall")}
    return OfficialROMSurrogate(copied, name=group)
