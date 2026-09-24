"""Clean-room composition for the MSA XP prescription.

The module deliberately stops at the composition boundary.  The MSA angle
assembler and source-frame geometry are imported from their separate clean
modules, while the non-precessing carrier is an opaque :class:`IMRPhenomXAS`
callable. The explicit v223 wrapper supplies the supported production XP
route; alternative equation-level choices remain composition helpers.

The low-level :func:`compose_msa_waveform` function takes already evaluated
angles, geometry, and a signed final-spin scalar.  This makes it possible to
test angle, carrier, and projection choices independently.  Angles supplied
as a sequence use the conventional ``(alpha, beta, gamma)`` order.  A mapping
may instead use ``alpha``, ``beta`` and ``gamma`` (or ``cos_beta``).  Relative
angles are anchored with ``alpha0`` from the geometry; public angle APIs can
be injected by setting ``angles_are_absolute=True``.

The final-spin helpers use source-total-mass-normalized angular momenta.  In
those units Eq. (afin_L_fin) of Pratten et al. (arXiv:2004.06503) is

``L_fin = (M_fin/M)**2 * a_parallel - (S1.Lhat + S2.Lhat)``.

The printed ``finalspin_3`` equation divides by ``M**2`` whereas the preceding
definition of the Kerr parameter divides by ``M_fin**2``.  Both choices are
available explicitly through ``normalization='total'`` and
``normalization='remnant'``.  The equation-level wrapper uses remnant
normalization for the carrier's Kerr input together with the separate
``l_fin_normalization='initial'`` mode.  The explicit native v223 wrapper
uses total normalization for the carrier's Kerr input and the same
initial-total-M² ``L_fin`` compatibility mode.  The named
``finalspin_3`` helper defaults to the printed paper denominator.  No fitted
correction is applied to bridge the two conventions.

The optional ``initial_s1_l`` and ``initial_s2_l`` arguments are an explicit
compatibility path for the reference MSA final-spin assembly.  When supplied,
their initial aligned projections are used in ``L_fin`` while
``mean_s1_l + mean_s2_l`` remains in the precession-averaged cross term.  If
they are omitted, the equation helper uses the mean projections for both
places.  ``l_fin_normalization='remnant'`` (the default) applies the explicit
``(M_fin/M)**2`` factor to a remnant-normalized Kerr input.  The native
compatibility path uses ``l_fin_normalization='initial'``: the aligned fit
scalar is treated as the published fit for ``L_fin`` in initial-total-M²
units, as stated in the paper, so no remnant-mass factor is applied.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from torchwave.constants import MSUN_SEC
from .imrphenom_xas import IMRPhenomXAS, XASParameterFits
from .precession_msa_angles_clean import (
    assemble_precession_msa_angles, _compatibility_variant,
)
from .precession_msa_pn_clean import precession_averages
from .precession_msa_reference_l_clean import orbital_angular_momentum_nonconserved_3pn
from .precession_source_frame_clean import source_frame_geometry
from .precession_twisting_clean import project_twisted_positive_modes


_DTYPE = torch.float64
_SQRT_H22 = math.sqrt(5.0 / (4.0 * math.pi))


def _device_of(values):
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _as_f64(value, *, device=None):
    return torch.as_tensor(value, dtype=_DTYPE, device=device)


def _broadcast_f64(*values):
    device = _device_of(values)
    return torch.broadcast_tensors(*(_as_f64(value, device=device) for value in values))


def _require_finite(name: str, value: torch.Tensor):
    if bool(torch.any(~torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")


def _stable_acos(value):
    """Differentiate the endpoint values of ``acos`` without ``inf*0``."""
    value = torch.clamp(_as_f64(value), -1.0, 1.0)
    endpoint = value.abs() >= 1.0
    # Evaluate the inactive branch at a regular interior point.  The selected
    # endpoint branch is exactly 0 or pi and has the expected zero gauge
    # derivative.
    sin2 = torch.where(endpoint, torch.ones_like(value),
                       torch.clamp(1.0 - value.square(), min=0.0))
    raw = torch.atan2(torch.sqrt(sin2), value)
    endpoint_value = torch.where(value < 0.0, torch.full_like(value, math.pi),
                                 torch.zeros_like(value))
    return torch.where(endpoint, endpoint_value, raw)


def physical_zero_nutation_cos_beta(frequency, mass1, mass2, spin1, spin2):
    """Opening angle from the unsplit spin vector (Pratten et al., Eq. 17).

    The v223 MSA auxiliary root separation is needed in singular integrals,
    but is not a physical transverse spin.  On a zero-nutation branch the
    spin vector is constant, so Eq. (25) reduces to Eq. (17) with the same
    3PN orbital angular momentum used by the MSA opening angle.  In
    particular, its aligned limit is exactly zero for ``L+S_parallel>0``.
    """
    f = _as_f64(frequency)
    m1, m2, s1, s2 = mass1, mass2, spin1, spin2
    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    spin = n1[..., None].square() * s1 + n2[..., None].square() * s2
    s_parallel = spin[..., 2, None]
    s_perp = torch.linalg.vector_norm(spin[..., :2], dim=-1)[..., None]
    l_orb = orbital_angular_momentum_nonconserved_3pn(
        m1[..., None], m2[..., None], f,
        s1[..., 2, None], s2[..., 2, None],
    )
    parallel = l_orb + s_parallel
    return parallel / torch.hypot(parallel, s_perp)


def _normalization_factor(normalization: str, remnant_mass_fraction: torch.Tensor):
    if normalization == "remnant":
        return remnant_mass_fraction.square()
    if normalization in ("total", "paper"):
        # ``paper`` names the denominator printed in Eq. (finalspin_3),
        # namely the source total mass.  Inputs to this module are already
        # normalized by that mass, so the factor is unity.
        return torch.ones_like(remnant_mass_fraction)
    raise ValueError("normalization must be 'remnant' or 'total' ('paper' is an alias for 'total')")


def _msa_final_spin_squared_from_sav_sq(
    sav_sq: torch.Tensor,
    mean_s1_l: torch.Tensor,
    mean_s2_l: torch.Tensor,
    aligned_kerr_parameter: torch.Tensor,
    remnant_mass_fraction: torch.Tensor,
    initial_s1_l: torch.Tensor,
    initial_s2_l: torch.Tensor,
    *,
    l_fin_normalization: str,
):
    """Equation-level implementation with ``sav_sq=<S_av^2>/M^4``."""
    if l_fin_normalization == "remnant":
        # Eq. (afin_L_fin) is expressed in source-total-M^2 units.  This mode
        # interprets the scalar as a remnant-normalized Kerr parameter and
        # applies the explicit (M_fin/M)^2 factor.
        aligned_l_fin = remnant_mass_fraction.square() * aligned_kerr_parameter
    elif l_fin_normalization == "initial":
        # The paper notes that the aligned ``a_parallel`` fit is constructed
        # as a fit for L_fin.  In this compatibility mode its scalar is
        # already normalized by the initial total mass squared.
        aligned_l_fin = aligned_kerr_parameter
    else:
        raise ValueError("l_fin_normalization must be 'remnant' or 'initial'")
    # The orbital subtraction uses initial aligned spin projections; the cross
    # term in Eq. (MSA_finalSpin) uses the precession averages.
    l_fin = aligned_l_fin - (initial_s1_l + initial_s2_l)
    mean_sum = mean_s1_l + mean_s2_l
    return sav_sq + l_fin.square() + 2.0 * l_fin * mean_sum


def msa_final_spin_squared(
    s_av,
    mean_s1_l,
    mean_s2_l,
    aligned_kerr_parameter,
    remnant_mass_fraction,
    *,
    average_is_squared: bool = False,
    initial_s1_l=None,
    initial_s2_l=None,
    l_fin_normalization: str = "remnant",
):
    """Return the MSA Eq. (MSA_finalSpin) squared final spin.

    ``s_av`` is the dimensionless, source-total-mass-normalized magnitude
    ``S_av/M^2`` by default.  Set ``average_is_squared=True`` when the first
    input already is ``S_av^2/M^4``.  The two mean projections are
    ``<S_i.Lhat>/M^2``; ``aligned_kerr_parameter`` is the signed aligned Kerr
    fit ``a_parallel``; and ``remnant_mass_fraction`` is ``M_fin/M``.

    The return value is ``<S_fin^2>/M^4``.  Eq. (afin_L_fin)'s remnant-mass
    factor is applied before the MSA identity when
    ``l_fin_normalization='remnant'``, with no clamping or fitted affine
    adjustment.  ``l_fin_normalization='initial'`` treats the aligned fit as
    an initial-total-M² ``L_fin`` scalar and therefore omits that factor.
    By default the two mean projections also provide the aligned subtraction
    in ``L_fin``.  Supplying both ``initial_s1_l`` and ``initial_s2_l`` selects
    the published compatibility contract: initial projections provide that
    subtraction, while the mean projections remain in the averaged cross
    term.  Supplying only one initial projection is an error.  All inputs
    broadcast and retain PyTorch gradients.
    """
    if (initial_s1_l is None) != (initial_s2_l is None):
        raise ValueError("initial_s1_l and initial_s2_l must be supplied together")
    values = [s_av, mean_s1_l, mean_s2_l, aligned_kerr_parameter, remnant_mass_fraction]
    if initial_s1_l is not None:
        values.extend((initial_s1_l, initial_s2_l))
    broadcast = _broadcast_f64(*values)
    sav, s1_l, s2_l, a_parallel, mf = broadcast[:5]
    if initial_s1_l is None:
        initial_1, initial_2 = s1_l, s2_l
    else:
        initial_1, initial_2 = broadcast[5:]
    checks = (("s_av", sav), ("mean_s1_l", s1_l),
              ("mean_s2_l", s2_l), ("aligned_kerr_parameter", a_parallel),
              ("remnant_mass_fraction", mf))
    if initial_s1_l is not None:
        checks += (("initial_s1_l", initial_1), ("initial_s2_l", initial_2))
    for name, value in checks:
        _require_finite(name, value)
    if bool(torch.any(mf <= 0.0)):
        raise ValueError("remnant_mass_fraction must be positive")
    sav_sq = sav if average_is_squared else sav.square()
    if bool(torch.any(sav_sq < 0.0)):
        raise ValueError("S_av^2 must be nonnegative")
    return _msa_final_spin_squared_from_sav_sq(
        sav_sq, s1_l, s2_l, a_parallel, mf, initial_1, initial_2,
        l_fin_normalization=l_fin_normalization,
    )


def msa_final_spin(
    s_av,
    mean_s1_l,
    mean_s2_l,
    aligned_kerr_parameter,
    remnant_mass_fraction,
    *,
    normalization: str = "remnant",
    average_is_squared: bool = False,
    initial_s1_l=None,
    initial_s2_l=None,
    l_fin_normalization: str = "remnant",
):
    """Return the signed MSA ``finalspin_3`` Kerr parameter.

    ``normalization='remnant'`` returns ``sign(a_parallel)*sqrt(<S_fin^2>) /
    (M_fin/M)^2`` and is the normalization implied by Eq. (afin_L_fin).
    ``normalization='total'`` (or ``'paper'``) returns the quantity printed in
    Eq. (finalspin_3), ``sign(a_parallel)*sqrt(<S_fin^2>)``.  The sign follows
    the paper's prescription for the aligned carrier.  At exactly zero
    ``a_parallel`` the positive branch is selected because the sign is
    undefined by the source equations.  ``l_fin_normalization`` is forwarded
    to :func:`msa_final_spin_squared`; use ``'initial'`` only when the aligned
    scalar is already an initial-total-M² ``L_fin`` fit.
    """
    squared = msa_final_spin_squared(
        s_av,
        mean_s1_l,
        mean_s2_l,
        aligned_kerr_parameter,
        remnant_mass_fraction,
        average_is_squared=average_is_squared,
        initial_s1_l=initial_s1_l,
        initial_s2_l=initial_s2_l,
        l_fin_normalization=l_fin_normalization,
    )
    a_parallel, mf = _broadcast_f64(
        aligned_kerr_parameter, remnant_mass_fraction
    )
    if bool(torch.any(squared < -1e-12)):
        raise ValueError("MSA final-spin squared magnitude is negative")
    magnitude = torch.sqrt(torch.clamp(squared, min=0.0))
    return torch.where(a_parallel < 0.0, -magnitude, magnitude) / _normalization_factor(normalization, mf)


def finalspin_3(*args, **kwargs):
    """Named implementation of the paper's MSA ``finalspin_3`` equation.

    The default ``normalization='paper'`` reproduces the denominator printed
    in Eq. (finalspin_3).  Pass ``normalization='remnant'`` to use the Kerr
    normalization implied by Eq. (afin_L_fin), and use
    :func:`msa_final_spin_normalization_report` to inspect their aligned-limit
    discrepancy.
    """
    kwargs.setdefault("normalization", "paper")
    return msa_final_spin(*args, **kwargs)


def msa_final_spin_normalization_report(aligned_kerr_parameter, remnant_mass_fraction):
    """Report the two source-equation normalizations in the aligned limit.

    For ``S_av=|S_parallel|`` and ``<S1.L>+<S2.L>=S_parallel``, the remnant
    normalization returns ``a_parallel`` while the printed total-mass
    normalization returns ``(M_fin/M)^2*a_parallel``.  This helper exposes the
    two values without inferring or fitting either one.
    """
    a_parallel, mf = _broadcast_f64(aligned_kerr_parameter, remnant_mass_fraction)
    _require_finite("aligned_kerr_parameter", a_parallel)
    _require_finite("remnant_mass_fraction", mf)
    if bool(torch.any(mf <= 0.0)):
        raise ValueError("remnant_mass_fraction must be positive")
    return {
        "aligned_remnant_normalized": a_parallel,
        "aligned_total_mass_normalized": mf.square() * a_parallel,
        "total_minus_remnant": (mf.square() - 1.0) * a_parallel,
    }


def _unpack_angles(angles, *, angle_order: str):
    if isinstance(angles, Mapping):
        try:
            alpha = angles["alpha"]
            gamma = angles["gamma"]
        except KeyError as exc:
            raise ValueError("angles mapping must contain 'alpha' and 'gamma'") from exc
        if "beta" in angles:
            beta = angles["beta"]
        elif "cos_beta" in angles:
            beta = _stable_acos(angles["cos_beta"])
        else:
            raise ValueError("angles mapping must contain 'beta' or 'cos_beta'")
        return alpha, beta, gamma
    if isinstance(angles, (str, bytes)):
        raise TypeError("angles must be a mapping or a three-element sequence")
    try:
        values = tuple(angles)
    except TypeError as exc:
        raise TypeError("angles must be a mapping or a three-element sequence") from exc
    if len(values) != 3:
        raise ValueError("angles must contain exactly three values")
    if angle_order == "alpha_beta_gamma":
        return values
    if angle_order == "alpha_gamma_cosbeta":
        return values[0], _stable_acos(values[2]), values[1]
    if angle_order == "alpha_gamma_beta":
        return values[0], values[2], values[1]
    raise ValueError("angle_order must be 'alpha_beta_gamma', 'alpha_gamma_beta', or 'alpha_gamma_cosbeta'")


def _unpack_geometry(geometry):
    if isinstance(geometry, Mapping):
        try:
            alpha0 = geometry["alpha0"] if "alpha0" in geometry else geometry["alpha"]
            theta_jn = geometry["theta_jn"] if "theta_jn" in geometry else geometry["theta"]
            zeta = geometry["zeta"]
        except KeyError as exc:
            raise ValueError(f"geometry mapping must contain '{exc.args[0]}'") from exc
        if "phi_j" in geometry:
            phi_j = geometry["phi_j"]
        elif "phiJL0" in geometry:
            # Public source-frame metadata reports phiJL0=-phi_j.
            phi_j = -_as_f64(geometry["phiJL0"])
        else:
            raise ValueError("geometry mapping must contain 'phi_j'")
        return alpha0, theta_jn, zeta, phi_j
    try:
        values = tuple(geometry)
    except TypeError as exc:
        raise TypeError("geometry must be a mapping or (alpha0, theta_jn, zeta, phi_j)") from exc
    if len(values) != 4:
        raise ValueError("geometry must contain exactly (alpha0, theta_jn, zeta, phi_j)")
    return values


def _principal_phase_delta(current, reference):
    """Return the differentiable principal phase difference ``current-reference``."""
    delta = current - reference
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def transport_msa_angles_for_phi_ref(
    angles,
    geometry_at_zero,
    geometry_at_phi,
    *,
    convention: int = 0,
    aligned: bool = False,
    angle_order: str = "alpha_beta_gamma",
):
    """Transport relative MSA angles between source-frame ``phiRef`` values.

    Appendix C and Table IV define the source-frame coordinate change used by
    the XP conventions.  ``angles`` are relative to the reference geometry at
    ``phiRef=0``.  The returned angle mapping and geometry anchor can be
    passed directly to :func:`compose_msa_waveform`:

    * alpha receives ``alpha0(phiRef)-alpha0(0)``;
    * convention 1 receives ``-[phiJ(phiRef)-phiJ(0)]`` in gamma;
    * convention 1 uses the zero-phase ``phiJ`` anchor after that gamma
      transport, while convention 0 keeps the current ``phiJ`` for its
      carrier phase;
    * on the legacy near-aligned gauge branch, convention-1 gamma transport is
      omitted because ``phiJ`` is undefined, and the current clean fallback
      branch is retained.  The native v223 wrapper applies its stricter
      exact-zero endpoint chart correction after this transport.

    The theta and polarization-basis coordinates are taken from the current
    geometry.  Phase deltas use the principal representative through
    ``atan2(sin(delta), cos(delta))``; this changes no Wigner-D rotation and
    avoids an artificial discontinuity at a branch cut.  No public angle or
    geometry provider is consulted here.

    Returns ``(shifted_angles, geometry_anchor, details)``.  ``details`` is a
    tensor-valued record intended for diagnostics and remains in the autograd
    graph.  ``aligned`` is an explicit scalar branch choice made by the
    native wrapper from the source spins.
    """
    if int(convention) not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    alpha, beta, gamma = _unpack_angles(angles, angle_order=angle_order)
    a0_zero, _theta_zero, _zeta_zero, phi_zero = _unpack_geometry(geometry_at_zero)
    _a0_phi, theta_phi, zeta_phi, phi_phi = _unpack_geometry(geometry_at_phi)
    device = _device_of((alpha, beta, gamma, a0_zero, theta_phi, zeta_phi, phi_zero, phi_phi))
    alpha = _as_f64(alpha, device=device)
    beta = _as_f64(beta, device=device)
    gamma = _as_f64(gamma, device=device)
    a0_zero = _as_f64(a0_zero, device=device)
    theta_phi = _as_f64(theta_phi, device=device)
    zeta_phi = _as_f64(zeta_phi, device=device)
    phi_zero = _as_f64(phi_zero, device=device)
    phi_phi = _as_f64(phi_phi, device=device)
    delta_alpha = _principal_phase_delta(_as_f64(_a0_phi, device=device), a0_zero)
    delta_phi_j = _principal_phase_delta(phi_phi, phi_zero)
    is_tensor_aligned = isinstance(aligned, torch.Tensor)
    if is_tensor_aligned:
        aligned_omitted = aligned & (int(convention) == 1)
        delta_phi_j_u = delta_phi_j.unsqueeze(-1) if (gamma.ndim > delta_phi_j.ndim and delta_phi_j.ndim > 0) else delta_phi_j
        delta_alpha_u = delta_alpha.unsqueeze(-1) if (alpha.ndim > delta_alpha.ndim and delta_alpha.ndim > 0) else delta_alpha
        apply_gamma_shift = (int(convention) == 1) & (~aligned_omitted)
        apply_u = apply_gamma_shift.unsqueeze(-1) if gamma.ndim > apply_gamma_shift.ndim else apply_gamma_shift
        gamma_shift = torch.where(apply_u, -delta_phi_j_u, torch.zeros_like(delta_phi_j_u))
        shifted = {"alpha": alpha + delta_alpha_u, "gamma": gamma + gamma_shift}
        phi_anchor = torch.where(aligned_omitted | (int(convention) == 0), phi_phi, phi_zero)
        gamma_omitted = aligned_omitted
    else:
        gamma_shift = torch.zeros_like(delta_phi_j)
        gamma_omitted = bool(aligned and int(convention) == 1)
        if int(convention) == 1 and not gamma_omitted:
            gamma_shift = -delta_phi_j
        delta_alpha_u = delta_alpha.unsqueeze(-1) if (alpha.ndim > delta_alpha.ndim and delta_alpha.ndim > 0) else delta_alpha
        gamma_shift_u = gamma_shift.unsqueeze(-1) if (gamma.ndim > gamma_shift.ndim and gamma_shift.ndim > 0) else gamma_shift
        shifted = {"alpha": alpha + delta_alpha_u, "gamma": gamma + gamma_shift_u}
        phi_anchor = phi_phi if int(convention) == 0 or gamma_omitted else phi_zero
    if isinstance(angles, Mapping) and "beta" not in angles and "cos_beta" in angles:
        shifted["cos_beta"] = _as_f64(angles["cos_beta"], device=device)
    else:
        shifted["beta"] = beta
    # In convention 1 the gamma offset in compose_msa_waveform is
    # ``pi-phi_j``.  Pairing it with phi_zero after the explicit -delta_phi_j
    # transport gives the same total phase as the current geometry.  The
    # aligned branch deliberately keeps its current fallback phi_j.
    anchor = {
        "alpha0": a0_zero,
        "theta_jn": theta_phi,
        "zeta": zeta_phi,
        "phi_j": phi_anchor,
    }
    details = {
        "delta_alpha": delta_alpha,
        "delta_phi_j": delta_phi_j,
        "delta_gamma": gamma_shift,
        "aligned_gamma_transport_omitted": gamma_omitted,
        "convention": int(convention),
        "phi_j_anchor": phi_anchor,
        "source": "Appendix C/Table IV coordinate transport; no public provider",
    }
    return shifted, anchor, details


def _align_frequency_tensor(value, batch_shape, n_frequency, *, name):
    """Broadcast a scalar/batch/frequency-last angle to ``batch+frequency``."""
    value = _as_f64(value)
    if value.ndim == 0:
        return value.expand(tuple(batch_shape) + (n_frequency,))
    if value.shape[-1] == n_frequency:
        prefix = value.shape[:-1]
        try:
            target = torch.broadcast_shapes(batch_shape, prefix)
        except RuntimeError as exc:
            raise ValueError(f"{name} batch shape {tuple(prefix)} is not broadcastable to {tuple(batch_shape)}") from exc
        return torch.broadcast_to(value, tuple(target) + (n_frequency,))
    try:
        target = torch.broadcast_shapes(batch_shape, value.shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} shape {tuple(value.shape)} is not a batch shape") from exc
    return value.reshape(tuple(value.shape) + (1,)).expand(tuple(target) + (n_frequency,))


def _angle_batch_shape(value, n_frequency, *, name):
    """Return an angle's parameter prefix before the frequency axis."""
    tensor = _as_f64(value)
    if tensor.ndim == 0:
        return torch.Size()
    if tensor.shape[-1] == n_frequency:
        return tensor.shape[:-1]
    return tensor.shape


def _compose_broadcast_parameters(values):
    tensors = _broadcast_f64(*values)
    return tensors[0].shape, tensors


def compose_msa_waveform(
    frequency,
    mass1,
    mass2,
    chi1,
    chi2,
    angles,
    geometry,
    final_spin,
    *,
    distance=100.0,
    tc=0.0,
    f_ref=20.0,
    convention: int = 0,
    angle_order: str = "alpha_beta_gamma",
    angles_are_absolute: bool = False,
    xas_model=None,
):
    """Compose an XP waveform from supplied MSA angles and geometry.

    Parameters are masses in solar masses, aligned carrier spins ``chi1`` and
    ``chi2``, frequency in Hz, distance in Mpc, and a signed final Kerr
    parameter.  ``geometry`` is ``(alpha0, theta_jn, zeta, phi_j)`` from
    :func:`source_frame_geometry`.  Relative angles are interpreted as
    ``(alpha-alpha0, beta, gamma)`` and anchored by ``alpha0``.  Set
    ``angles_are_absolute=True`` when injecting the values returned by the
    public angle API.  Convention 0 uses ``-phi_j`` as the face-on XAS
    carrier phase and leaves the source-frame polarization basis unchanged;
    convention 1 adds ``pi-phi_j`` to relative ``gamma``, shifts that basis by
    ``-pi/2``, and uses carrier phase zero, following the validated generic
    Eqs. (11)--(13) assembly and Table IV of arXiv:2004.06503.

    The opaque carrier is called only at inclination zero with
    ``_final_spin_override=final_spin``.  Its positive-frequency ``h22`` is
    reconstructed from ``h_plus`` and passed to the fresh quadrupole projector.
    Output tensors have parameter axes preceding the final frequency axis.
    """
    if int(convention) not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    alpha_in, beta_in, gamma_in = _unpack_angles(angles, angle_order=angle_order)
    alpha0, theta_jn, zeta, phi_j = _unpack_geometry(geometry)

    fi = torch.as_tensor(frequency)
    if fi.ndim == 0:
        fi = fi.reshape(1)
    if not fi.is_floating_point():
        fi = fi.to(_DTYPE)
    f = fi.to(_DTYPE).reshape(-1)
    _require_finite("frequency", f)
    positive = f[f > 0.0]
    if positive.numel() == 0:
        raise ValueError("frequency grid must contain a positive frequency")
    f_safe = torch.where(f > 0.0, f, positive.min())

    # Geometry is per source and therefore contributes to the batch shape;
    # angle arrays are aligned to that shape only after this broadcast.
    physical_batch_shape, params = _compose_broadcast_parameters(
        (mass1, mass2, chi1, chi2, distance, tc, f_ref, final_spin,
         alpha0, theta_jn, zeta, phi_j)
    )
    angle_batch_shapes = tuple(
        _angle_batch_shape(value, f.numel(), name=name)
        for name, value in (("alpha", alpha_in), ("beta", beta_in), ("gamma", gamma_in))
    )
    try:
        batch_shape = torch.broadcast_shapes(physical_batch_shape, *angle_batch_shapes)
    except RuntimeError as exc:
        raise ValueError(
            "angle parameter prefixes must broadcast with the physical parameter batch"
        ) from exc
    if batch_shape != physical_batch_shape:
        params = tuple(value.expand(batch_shape) for value in params)
    m1, m2, c1, c2, dist, tc_t, fr, fs, a0, theta, zeta_t, phi = params
    for name, value in (("mass1", m1), ("mass2", m2), ("chi1", c1), ("chi2", c2),
                        ("distance", dist), ("tc", tc_t), ("f_ref", fr),
                        ("final_spin", fs), ("alpha0", a0), ("theta_jn", theta),
                        ("zeta", zeta_t), ("phi_j", phi)):
        _require_finite(name, value)
    if bool(torch.any((m1 <= 0.0) | (m2 <= 0.0))):
        raise ValueError("masses must be positive")
    if bool(torch.any(dist <= 0.0)):
        raise ValueError("distance must be positive")
    if bool(torch.any(fr <= 0.0)):
        raise ValueError("f_ref must be positive")

    alpha = _align_frequency_tensor(alpha_in, batch_shape, f.numel(), name="alpha")
    beta = _align_frequency_tensor(beta_in, batch_shape, f.numel(), name="beta")
    gamma = _align_frequency_tensor(gamma_in, batch_shape, f.numel(), name="gamma")
    for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        _require_finite(name, value)
    if not angles_are_absolute:
        alpha = alpha + a0.reshape(batch_shape + (1,))
    if int(convention) == 1 and not angles_are_absolute:
        gamma = gamma + math.pi - phi.reshape(batch_shape + (1,))
        zeta_t = zeta_t - math.pi / 2.0
        carrier_phase = torch.zeros_like(phi)
    elif int(convention) == 1:
        # Absolute public angle injection already contains the convention's
        # gamma offset; geometry still controls the published polarization
        # basis and carrier phase.
        zeta_t = zeta_t - math.pi / 2.0
        carrier_phase = torch.zeros_like(phi)
    else:
        # ``source_frame_geometry`` returns the azimuth of J's transverse
        # projection.  The generic clean-room Eqs. (11)--(13) assembly uses the
        # opposite carrier phase and the geometry's polarization basis as
        # returned.  Keep this choice shared by relative and absolute angle
        # injection so a supplied public angle triplet changes only the
        # twisting dynamics.
        carrier_phase = -phi

    carrier = IMRPhenomXAS() if xas_model is None else xas_model
    if isinstance(carrier, type):
        carrier = carrier()
    if not callable(carrier):
        raise TypeError("xas_model must be a callable carrier or carrier class")

    n_rows = math.prod(batch_shape) if batch_shape else 1
    flat = tuple(value.reshape(-1) for value in (m1, m2, c1, c2, dist, tc_t, fr, fs))
    angle_rows = tuple(value.reshape((n_rows, f.numel())) for value in (alpha, beta, gamma))
    geom_rows = tuple(value.reshape(-1) for value in (theta, zeta_t, carrier_phase))
    if n_rows > 1:
        try:
            h_carrier = carrier(
                f=f_safe,
                mass1=flat[0].reshape(batch_shape),
                mass2=flat[1].reshape(batch_shape),
                chi1=flat[2].reshape(batch_shape),
                chi2=flat[3].reshape(batch_shape),
                distance=flat[4].reshape(batch_shape),
                inclination=torch.zeros(batch_shape, dtype=_DTYPE, device=f.device),
                phi_c=geom_rows[2].reshape(batch_shape),
                t_c=flat[5].reshape(batch_shape),
                f_ref=flat[6].reshape(batch_shape),
                _final_spin_override=flat[7].reshape(batch_shape),
            )
            if isinstance(h_carrier, (tuple, list)) and len(h_carrier) >= 1:
                h22 = torch.as_tensor(h_carrier[0], dtype=torch.complex128)
                if h22.shape[-1] == f.numel() and h22.numel() == n_rows * f.numel():
                    hneg = 2.0 * h22 / _SQRT_H22
                    hp, hc = project_twisted_positive_modes(
                        {(2, 2): hneg},
                        alpha,
                        gamma,
                        beta,
                        theta.unsqueeze(-1),
                        zeta_t.unsqueeze(-1),
                    )
                    support = f > 0.0
                    hp = torch.where(support, hp, torch.zeros_like(hp))
                    hc = torch.where(support, hc, torch.zeros_like(hc))
                    out_dtype = torch.complex64 if fi.dtype == torch.float32 else torch.complex128
                    return (hp.reshape(tuple(batch_shape) + (f.numel(),)).to(out_dtype),
                            hc.reshape(tuple(batch_shape) + (f.numel(),)).to(out_dtype))
        except Exception:
            pass
    h_plus_rows, h_cross_rows = [], []
    for index in range(n_rows):
        h_carrier = carrier(
            f=f_safe,
            mass1=flat[0][index],
            mass2=flat[1][index],
            chi1=flat[2][index],
            chi2=flat[3][index],
            distance=flat[4][index],
            inclination=torch.zeros((), dtype=_DTYPE, device=f.device),
            phi_c=geom_rows[2][index],
            t_c=flat[5][index],
            f_ref=flat[6][index],
            _final_spin_override=flat[7][index],
        )
        if not isinstance(h_carrier, (tuple, list)) or len(h_carrier) < 1:
            raise TypeError("opaque XAS carrier must return (h_plus, h_cross)")
        h22 = torch.as_tensor(h_carrier[0], dtype=torch.complex128).reshape(-1)
        if h22.numel() != f.numel():
            raise ValueError("opaque XAS carrier returned an unexpected frequency shape")
        hneg = 2.0 * h22 / _SQRT_H22
        hp, hc = project_twisted_positive_modes(
            {(2, 2): hneg}, angle_rows[0][index], angle_rows[2][index],
            angle_rows[1][index], geom_rows[0][index], geom_rows[1][index]
        )
        support = f > 0.0
        h_plus_rows.append(torch.where(support, hp, torch.zeros_like(hp)))
        h_cross_rows.append(torch.where(support, hc, torch.zeros_like(hc)))
    out_dtype = torch.complex64 if fi.dtype == torch.float32 else torch.complex128
    return (torch.stack(h_plus_rows).reshape(tuple(batch_shape) + (f.numel(),)).to(out_dtype),
            torch.stack(h_cross_rows).reshape(tuple(batch_shape) + (f.numel(),)).to(out_dtype))


def _canonical_source_parameters(mass1, mass2, spin1, spin2, *extra):
    m1, m2 = _as_f64(mass1), _as_f64(mass2)
    s1, s2 = _as_f64(spin1), _as_f64(spin2)
    if s1.ndim == 0 or s2.ndim == 0 or s1.shape[-1] != 3 or s2.shape[-1] != 3:
        raise ValueError("spin vectors must end in dimension 3")
    rest = [_as_f64(value) for value in extra]
    batch_shape = torch.broadcast_shapes(
        m1.shape, m2.shape, s1.shape[:-1], s2.shape[:-1],
        *(value.shape for value in rest),
    )
    m1 = m1.expand(batch_shape)
    m2 = m2.expand(batch_shape)
    s1 = s1.expand(batch_shape + (3,))
    s2 = s2.expand(batch_shape + (3,))
    rest = [value.expand(batch_shape) for value in rest]
    swap = m2 > m1
    m1_c = torch.where(swap, m2, m1)
    m2_c = torch.where(swap, m1, m2)
    s1_c = torch.where(swap[..., None], s2, s1)
    s2_c = torch.where(swap[..., None], s1, s2)
    return m1_c, m2_c, s1_c, s2_c, rest


def _native_msa_final_spin(m1, m2, spin1, spin2, f_ref, angles, *,
                           compatibility=None, variant=None,
                           version=None, v223=False):
    """Recover the reference MSA couplings already used by the assembler."""
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        version=version, v223=v223,
    )
    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    eta = n1 * n2
    q = n2 / n1
    v0 = (math.pi * total * MSUN_SEC * f_ref).pow(1.0 / 3.0)
    s1v = n1[..., None].square() * spin1
    s2v = n2[..., None].square() * spin2
    s1sq = torch.sum(s1v.square(), dim=-1)
    s2sq = torch.sum(s2v.square(), dim=-1)
    xi = (1.0 + q) * s1v[..., 2] + (1.0 + 1.0 / q) * s2v[..., 2]
    if all(key in angles for key in ("mean_s1_l", "mean_s2_l")):
        # The angle assembler has already regularized the native repeated
        # root branch and restored its physical initial projections.  Reuse
        # those values so final-spin evaluation cannot reintroduce the
        # equal-mass 1/(1-q) singularity.
        av = {
            "s1l": angles["mean_s1_l"],
            "s2l": angles["mean_s2_l"],
        }
    else:
        roots = angles["roots_reference"]
        s30, sm0, sp0 = roots.unbind(-1)
        av = precession_averages(
            angles["c1"], xi, q, eta, sp0, sm0, s1sq, s2sq, v0
        )
    # The XAS aligned fit and radiated-energy fit are used only as opaque
    # public fit callables.  Their source bodies are outside this clean-room
    # composition module.
    a_parallel = XASParameterFits.fit_final_spin(eta, spin1[..., 2], spin2[..., 2])
    mf = 1.0 - XASParameterFits.fit_erad(eta, spin1[..., 2], spin2[..., 2])
    final_spin = msa_final_spin(
        # ``angles['sav']`` is already the invariant S_av^2 used by the MSA
        # identity.  Keep it on the squared-input path so an exactly zero
        # spin row does not differentiate an unnecessary sqrt(0).
        angles["sav"],
        av["s1l"], av["s2l"], a_parallel, mf,
        normalization="total" if selected == "v223" else "remnant",
        average_is_squared=True,
        initial_s1_l=s1v[..., 2], initial_s2_l=s2v[..., 2],
        l_fin_normalization="initial",
    )
    return final_spin, {"s_av_squared": angles["sav"], "mean_s1_l": av["s1l"],
                        "mean_s2_l": av["s2l"], "initial_s1_l": s1v[..., 2],
                        "initial_s2_l": s2v[..., 2],
                        "aligned_kerr_parameter": a_parallel,
                        "remnant_mass_fraction": mf,
                        "l_fin": a_parallel - (s1v[..., 2] + s2v[..., 2]),
                        "l_fin_normalization": "initial_m2_aligned_fit",
                        "normalization": "total" if selected == "v223" else "remnant",
                        "compatibility": selected}


def waveform(
    frequency,
    mass1,
    mass2,
    spin1x=0.0,
    spin1y=0.0,
    spin1z=0.0,
    spin2x=0.0,
    spin2y=0.0,
    spin2z=0.0,
    distance=100.0,
    inclination=0.0,
    phic=0.0,
    tc=0.0,
    f_ref=None,
    *,
    convention: int = 0,
    final_spin_mod: int = 3,
    order: int = 5,
    average_mode: str = "leading",
    beta_pn: bool = False,
    beta_pn_version: int = 102,
    beta_compat_102: bool = False,
    compatibility=None,
    variant=None,
    version=None,
    v223: bool = False,
    xas_model=None,
    final_spin_override=None,
    boundary: str = "v223",
):
    """Native MSA composition with an explicit clean beta choice.

    This wrapper calls the fresh equation-level MSA angle assembler with
    ``order=5`` and ``beta_pn=False`` by default, so its twist uses the clean
    Newtonian orbital-angular-momentum prescription for ``cos(beta)``.
    ``beta_pn=True`` selects the separately implemented PN beta prescription;
    ``compatibility='v223'`` (or :func:`waveform_v223`) selects the native
    nonconserved 3PN opening-beta value explicitly.
    It intentionally makes no claim to implement any public
    ``prec_version``; public versions 220/223/224 belong to the diagnostic's
    black-box reference calls.  The MSA ``final_spin_mod=3`` calculation is
    available from the explicit paper helper and is the wrapper default.
    Selecting ``compatibility='v223'`` (or using :func:`waveform_v223`)
    activates the bounded native composition: reference-spin flux couplings,
    the native angle convention, source-frame 259/75 geometry, and
    total-mass-normalized final spin.  The equation default remains unchanged.
    An explicit ``final_spin_override`` may be supplied for composition tests.
    Equal-mass and degenerate MSA rows are rejected by the equation-level
    angle assembler; the explicit native v223 branch uses its documented
    zero-nutation limiting state.  For v223 convention 1 with exactly zero
    transverse spin, the wrapper applies the Table-IV endpoint chart relation
    ``delta_alpha=-phi_J`` and ``delta_gamma=+phi_J`` before projection.  The
    finite ``cos(beta)`` from the clean angle assembler is retained; the
    correction is not applied to near-aligned nonzero transverse spins.
    """
    selected = _compatibility_variant(
        compatibility=compatibility, variant=variant,
        version=version, v223=v223,
    )
    if boundary not in ("v223", "physical"):
        raise ValueError("boundary must be 'v223' or 'physical'")
    if boundary == "physical" and selected != "v223":
        raise ValueError("physical boundary requires the v223 MSA prescription")
    if int(final_spin_mod) != 3 and final_spin_override is None:
        raise ValueError("clean native MSA supports final_spin_mod=3 only unless final_spin_override is supplied")
    if int(convention) not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    fi = torch.as_tensor(frequency)
    if fi.ndim == 0:
        fi = fi.reshape(1)
    if not fi.is_floating_point():
        fi = fi.to(_DTYPE)
    f = fi.to(_DTYPE).reshape(-1)
    _require_finite("frequency", f)
    positive = f[f > 0.0]
    if positive.numel() == 0:
        raise ValueError("frequency grid must contain a positive frequency")
    f_safe = torch.where(f > 0.0, f, positive.min())
    spin1_components = tuple(_as_f64(x, device=f.device) for x in (spin1x, spin1y, spin1z))
    spin2_components = tuple(_as_f64(x, device=f.device) for x in (spin2x, spin2y, spin2z))
    m1 = _as_f64(mass1, device=f.device)
    m2 = _as_f64(mass2, device=f.device)
    scalar_values = tuple(
        _as_f64(value, device=f.device)
        for value in (distance, inclination, phic, tc,
                      positive.min() if f_ref is None else f_ref)
    )
    # Broadcast scalar component inputs before stacking.  Stacking first
    # would reject valid mixed scalar/tensor component calls such as
    # ``spin1x=torch.tensor([.1,.2])`` with scalar y and z components.
    broadcast = torch.broadcast_tensors(
        m1, m2, *spin1_components, *spin2_components, *scalar_values
    )
    m1, m2 = broadcast[:2]
    spin1_components = broadcast[2:5]
    spin2_components = broadcast[5:8]
    scalar_values = broadcast[8:]
    s1 = torch.stack(spin1_components, dim=-1)
    s2 = torch.stack(spin2_components, dim=-1)
    dist, inc, phi0, tc_t, fr = scalar_values
    for name, value in (("mass1", m1), ("mass2", m2), ("spin1", s1), ("spin2", s2),
                        ("distance", dist), ("inclination", inc), ("phic", phi0),
                        ("tc", tc_t), ("f_ref", fr)):
        _require_finite(name, value)
    if bool(torch.any((m1 <= 0.0) | (m2 <= 0.0))):
        raise ValueError("masses must be positive")
    if bool(torch.any(dist <= 0.0)):
        raise ValueError("distance must be positive")
    if bool(torch.any(fr <= 0.0)):
        raise ValueError("f_ref must be positive")
    if bool(torch.any(torch.linalg.vector_norm(s1, dim=-1) > 1.0)) or bool(torch.any(torch.linalg.vector_norm(s2, dim=-1) > 1.0)):
        raise ValueError("spin-vector norm must not exceed 1")

    # The clean angle assembler canonicalizes internally; this wrapper uses
    # the same canonical state for its source-frame geometry and remnant fit.
    m1, m2, s1, s2, rest = _canonical_source_parameters(m1, m2, s1, s2, dist, inc, phi0, tc_t, fr)
    dist, inc, phi0, tc_t, fr = rest
    # Fully vectorized evaluation path
    angle_kwargs = dict(
        f_ref=fr, order=order, beta_pn=beta_pn,
        beta_pn_version=beta_pn_version,
        beta_compat_102=beta_compat_102, average_mode=average_mode,
    )
    if selected == "v223":
        angle_kwargs["compatibility"] = "v223"
    angle_values = assemble_precession_msa_angles(
        f_safe, m1, m2, s1, s2, **angle_kwargs,
    )
    geometry_kwargs = {"compatibility": "v223"} if selected == "v223" else {}
    geom = source_frame_geometry(
        m1, m2, s1[..., 0], s1[..., 1], s1[..., 2], s2[..., 0], s2[..., 1], s2[..., 2],
        inc, phi0, fr, **geometry_kwargs,
    )
    geom_zero = source_frame_geometry(
        m1, m2, s1[..., 0], s1[..., 1], s1[..., 2], s2[..., 0], s2[..., 1], s2[..., 2],
        inc, torch.zeros_like(phi0), fr, **geometry_kwargs,
    )
    transverse_sq = (
        torch.sum(s1[..., :2].square(), dim=-1) + torch.sum(s2[..., :2].square(), dim=-1)
    )
    legacy_aligned = transverse_sq <= 1.0e-28
    exact_transverse_zero = (
        torch.all(s1[..., :2] == 0.0, dim=-1) & torch.all(s2[..., :2] == 0.0, dim=-1)
    )
    if selected == "v223" and torch.any(legacy_aligned):
        alpha_shift = torch.as_tensor(
            math.pi, dtype=geom[0].dtype, device=geom[0].device,
        )
        geom = (torch.where(legacy_aligned, geom[0] + alpha_shift, geom[0]), *geom[1:])
        geom_zero = (torch.where(legacy_aligned, geom_zero[0] + alpha_shift, geom_zero[0]), *geom_zero[1:])
    transported_angles, composition_geometry, _transport = transport_msa_angles_for_phi_ref(
        {"alpha": angle_values["alpha"], "gamma": angle_values["gamma"],
         "cos_beta": angle_values["cos_beta"]},
        geom_zero, geom, convention=convention, aligned=legacy_aligned,
    )
    if boundary == "physical":
        # A separated auxiliary root has no physical spin amplitude.  Use
        # the unsplit zero-nutation opening only where one spin is aligned.
        # A nearly repeated auxiliary root alone does not prove constant S:
        # it may also occur for a generic two-spin configuration. Keep its
        # MSA Eq. (25) opening untouched.
        zero_nutation = (
            torch.all(s1[..., :2] == 0.0, dim=-1)
            | torch.all(s2[..., :2] == 0.0, dim=-1)
        )
        physical_cos_beta = physical_zero_nutation_cos_beta(
            f_safe, m1, m2, s1, s2,
        )
        transported_angles["cos_beta"] = torch.where(
            zero_nutation[..., None], physical_cos_beta,
            transported_angles["cos_beta"],
        )
    if selected == "v223" and convention == 1 and torch.any(exact_transverse_zero):
        endpoint_phi_j = _as_f64(
            composition_geometry["phi_j"], device=f.device,
        )
        ep_u = endpoint_phi_j.unsqueeze(-1) if transported_angles["alpha"].ndim > endpoint_phi_j.ndim else endpoint_phi_j
        exact_u = exact_transverse_zero.unsqueeze(-1) if transported_angles["alpha"].ndim > exact_transverse_zero.ndim else exact_transverse_zero
        transported_angles["alpha"] = torch.where(
            exact_u, transported_angles["alpha"] - ep_u, transported_angles["alpha"]
        )
        transported_angles["gamma"] = torch.where(
            exact_u, transported_angles["gamma"] + ep_u, transported_angles["gamma"]
        )
    if final_spin_override is None:
        rfinal, _ = _native_msa_final_spin(
            m1, m2, s1, s2, fr, angle_values,
            compatibility=selected,
        )
    else:
        rfinal = _as_f64(final_spin_override, device=f.device).expand(m1.shape)

    hp, hc = compose_msa_waveform(
        f, m1, m2, s1[..., 2], s2[..., 2],
        transported_angles, composition_geometry, rfinal,
        distance=dist, tc=tc_t, f_ref=fr, convention=convention,
        angle_order="alpha_beta_gamma", angles_are_absolute=False, xas_model=xas_model,
    )
    out_dtype = torch.complex64 if fi.dtype == torch.float32 else torch.complex128
    return hp.to(out_dtype), hc.to(out_dtype)


native_msa_waveform = waveform


def waveform_v223(*args, **kwargs):
    """Explicit native v223 MSA waveform entry point."""
    kwargs.setdefault("compatibility", "v223")
    return waveform(*args, **kwargs)


native_msa_waveform_v223 = waveform_v223


__all__ = [
    "compose_msa_waveform",
    "finalspin_3",
    "msa_final_spin",
    "msa_final_spin_normalization_report",
    "msa_final_spin_squared",
    "native_msa_waveform",
    "native_msa_waveform_v223",
    "transport_msa_angles_for_phi_ref",
    "waveform",
    "waveform_v223",
]
