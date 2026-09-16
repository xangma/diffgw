"""Clean-room London HM mode assembly primitives.

This module contains only equation-level composition.  It deliberately takes
the PhenomD carrier and amplitude as callables, so no waveform implementation
or calibration table is embedded here.  Frequencies are dimensionless ``Mf``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import torch

from .imrphenom_hm_clean import map_amplitude_frequency
from .imrphenom_hm_phase_clean import mapped_phase_c1, phenomd_eq14_phase_delta_c1
from .hm_21_pn_clean import hhat21_arun_compat_v4
from .precession_twisting_clean import wigner_d


MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))

# London et al. state these values immediately below Eq. (5).  The old clean
# diagnostic recipe used the two values in the opposite slots; keep that pair
# named so callers can request it explicitly without changing the clean
# default by accident.
PRINTED_AMPLITUDE_JOIN = 0.018
PRINTED_PHASE_JOIN = 0.014
COMPATIBILITY_AMPLITUDE_JOIN = 0.014
COMPATIBILITY_PHASE_JOIN = 0.018


def assemble_mode(
    mf: torch.Tensor,
    mode: tuple[int, int],
    *,
    eta: float | torch.Tensor,
    delta: float | torch.Tensor,
    chi_s: float | torch.Tensor,
    chi_a: float | torch.Tensor,
    carrier_positive: Callable[[torch.Tensor], torch.Tensor],
    amplitude_22: Callable[[torch.Tensor], torch.Tensor],
    amp_qnm_22: float | torch.Tensor,
    amp_qnm_lm: float | torch.Tensor,
    phase_qnm_22: float | torch.Tensor,
    phase_qnm_lm: float | torch.Tensor,
    phase_damp_22: float | torch.Tensor,
    phase_damp_lm: float | torch.Tensor,
    alpha4: float | torch.Tensor,
    alpha5: float | torch.Tensor,
    chi_pn: float | torch.Tensor,
    phic: float = 0.0,
    reference_gauge: float | torch.Tensor = 0.0,
    use_eq14: bool = True,
    base_fr22: float | torch.Tensor | None = None,
    base_fd22: float | torch.Tensor | None = None,
    amplitude_join: float | torch.Tensor = PRINTED_AMPLITUDE_JOIN,
    phase_join: float | torch.Tensor = PRINTED_PHASE_JOIN,
) -> torch.Tensor:
    """Assemble one positive-``m`` unprojected mode.

    London's printed joins are ``.018`` for the amplitude map and ``.014``
    for the phase map (the defaults).  The previous clean diagnostic recipe
    used ``amplitude_join=.014, phase_join=.018``; callers that need those
    numbers must pass them explicitly, preferably via
    :data:`COMPATIBILITY_AMPLITUDE_JOIN` and
    :data:`COMPATIBILITY_PHASE_JOIN`.  Eq. (14) is evaluated in the positive-D
    phase coordinate before mapping; the physical mode phase is its negative
    because the public D callable returns ``p0`` with ``h22=exp(-i*p0)``.
    ``phase_qnm_*`` remain the HM QNM values used by the outer frequency map.
    When supplied, ``base_fr22`` and ``base_fd22`` are the absolute D-carrier
    QNM values used consistently by Eq. (14)'s arctangent correction and its
    C1 join; omitting them preserves the helper's HM-22 default.
    ``reference_gauge`` is an explicit caller-supplied linear reference
    correction, never inferred.
    """
    x = torch.as_tensor(mf)
    dtype, device = x.dtype, x.device
    ell, m = mode
    f0_a = torch.as_tensor(amplitude_join, dtype=dtype, device=device) * torch.as_tensor(amp_qnm_lm, dtype=dtype, device=device) / torch.as_tensor(amp_qnm_22, dtype=dtype, device=device)
    f0_p = torch.as_tensor(phase_join, dtype=dtype, device=device) * torch.as_tensor(phase_qnm_lm, dtype=dtype, device=device) / torch.as_tensor(phase_qnm_22, dtype=dtype, device=device)
    ga = map_amplitude_frequency(x, f0_a, amp_qnm_lm, amp_qnm_22, m)
    ap = amplitude_22(ga)
    v = (2 * torch.pi * ga / m).pow(1 / 3)
    vx = (2 * torch.pi * x / m).pow(1 / 3)
    if mode == (2, 2):
        ratio = torch.ones_like(x)
    else:
        # London Eq. (4) evaluates the mode factor at both the mapped and
        # original frequencies; the common SPA powers therefore do not cancel
        # to a simple H-hat ratio.  This is an amplitude-only factor.
        from .imrphenom_hm_clean import pn_hhat_lm
        def hhat(y):
            vy = (2 * torch.pi * y / m).pow(1 / 3)
            if mode == (2, 1):
                return hhat21_arun_compat_v4(vy, eta, delta, chi_s, chi_a)
            if mode == (3, 2):
                et = torch.as_tensor(eta, dtype=dtype, device=device)
                return -(1 / 3) * torch.sqrt(vy.new_tensor(5 / 7)) * (1 - 3 * et) * vy**2
            return pn_hhat_lm(mode, vy, eta, delta, chi_s, chi_a)
        hlm_ga = hhat(ga) * v.pow(-3.5)
        hlm_x = hhat(x) * vx.pow(-3.5)
        h22_ga = -((2 * torch.pi * ga / 2).pow(1 / 3)).pow(-3.5)
        h22_at_x = hhat(2 * x / m) * ((2 * torch.pi * (2 * x / m) / m).pow(1 / 3)).pow(-3.5)
        # Equal-mass/equal-spin odd modes vanish identically.  Guard the
        # analytic ratio's 0/0 form without perturbing nonzero modes.
        tiny = torch.finfo(dtype).tiny
        safe_h22_at_x = torch.where(torch.abs(h22_at_x) > tiny, h22_at_x, torch.ones_like(h22_at_x))
        ratio_raw = torch.abs((hlm_ga / h22_ga) * (hlm_x / safe_h22_at_x))
        ratio = torch.where(torch.abs(h22_at_x) > tiny, ratio_raw, torch.zeros_like(ratio_raw))
    if use_eq14:
        def carrier_lm(y):
            z = carrier_positive(y)
            if mode == (2, 2):
                return z
            return z + phenomd_eq14_phase_delta_c1(
                y, eta, chi_pn, phase_qnm_22, phase_qnm_lm,
                phase_damp_22, phase_damp_lm, alpha4, alpha5,
                base_fr22=base_fr22, base_fd22=base_fd22,
            )
    else:
        carrier_lm = carrier_positive
    mapped = mapped_phase_c1(x, f0_p, phase_qnm_lm, phase_qnm_22, m, carrier_lm, ell=ell)
    ref = torch.as_tensor(reference_gauge, dtype=dtype, device=device)
    phi = torch.as_tensor(phic, dtype=dtype, device=device)
    physical_phase = -mapped + ref + m * phi
    return ap * ratio * torch.exp(1j * physical_phase)


def project_positive_modes(
    modes: Mapping[tuple[int, int], torch.Tensor], inclination: float | torch.Tensor,
    harmonic: Callable[[float | torch.Tensor, float | torch.Tensor, int, int, int], complex] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project positive-m modes using the paper's ``(-1)^ell`` symmetry.

    With ``harmonic=None`` (the default), the spin-weighted harmonics are
    evaluated as ``Y_lm(theta, 0) = sqrt((2*l+1)/(4*pi)) d^l_{m,2}(theta)``
    using the clean Torch Wigner-``d`` helper.  This path preserves the
    inclination's dtype, device, and autograd graph and supports any
    inclination shape that broadcasts with the supplied mode tensors.

    A custom ``harmonic`` callback remains available for scalar or vectorized
    external conventions.  Its return value is converted to the mode's
    complex dtype/device without going through Python ``complex``, so a Tensor
    result can retain its graph.  A callback that only accepts scalar angles
    must be called with scalar inclination.
    """
    if not modes:
        raise ValueError("at least one positive-m mode is required")

    first = next(iter(modes.values()))
    if not isinstance(first, torch.Tensor):
        first = torch.as_tensor(first)
    if first.is_complex():
        complex_dtype = first.dtype
        real_dtype = first.real.dtype
    elif first.is_floating_point():
        real_dtype = first.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    else:
        real_dtype = torch.get_default_dtype()
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    def as_mode_tensor(value: torch.Tensor | object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=first.device, dtype=complex_dtype)
        return torch.as_tensor(value, device=first.device, dtype=complex_dtype)

    def as_harmonic_tensor(value: torch.Tensor | object, z: torch.Tensor) -> torch.Tensor:
        # ``Tensor.to`` is differentiable; ``torch.as_tensor`` is appropriate
        # only for callback results that did not originate as Torch tensors.
        if isinstance(value, torch.Tensor):
            return value.to(device=z.device, dtype=z.dtype)
        return torch.as_tensor(value, device=z.device, dtype=z.dtype)

    iota = torch.as_tensor(inclination, dtype=real_dtype, device=first.device)
    zero = torch.zeros_like(iota)
    hp = None
    hc = None

    for (ell, m), value in modes.items():
        z = as_mode_tensor(value)
        if harmonic is None:
            norm = (2 * ell + 1) / (4 * torch.pi)
            yp = torch.sqrt(iota.new_tensor(norm)) * wigner_d(ell, m, 2, iota)
            yn = torch.sqrt(iota.new_tensor(norm)) * wigner_d(ell, -m, 2, iota)
        else:
            try:
                yp = harmonic(zero, iota, -2, ell, m)
                yn = harmonic(zero, iota, -2, ell, -m)
            except (TypeError, ValueError) as exc:
                if iota.ndim:
                    raise ValueError(
                        "custom harmonic callback must support vectorized "
                        "inclination or be called with a scalar inclination"
                    ) from exc
                raise
            yp = as_harmonic_tensor(yp, z)
            yn = as_harmonic_tensor(yn, z)

        parity = (-1) ** ell
        hp_lm = 0.5 * z * (yp + parity * torch.conj(yn))
        hc_lm = -0.5j * z * (yp - parity * torch.conj(yn))
        hp = hp_lm if hp is None else hp + hp_lm
        hc = hc_lm if hc is None else hc + hc_lm

    return hp, hc
