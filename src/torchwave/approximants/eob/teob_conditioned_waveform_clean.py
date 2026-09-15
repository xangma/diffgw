"""Opt-in clean TEOB time-domain to frequency-domain orchestration.

This module is an assembly adapter for the reviewed clean components.  It is
deliberately separate from the canonical waveform driver and does not change
any production routing.  The supported sector is scalar, nonspinning,
zero-tide TEOBResumS with the ``(2,2)`` mode.

The two attribution choices which can materially change a trajectory are
required arguments:

``radial_acceleration_policy``
    ``"conservative"`` evaluates the conservative radial acceleration from
    the Hamiltonian.  ``"frozen-mobility"`` uses the separately named clean
    ``C_current * p_dot_current`` diagnostic branch.  ``"zero"`` is an
    explicit control which supplies zero to the NQC basis.
    ``"frozen-mobility-stage-aware"`` applies the PA-prefix ``H_real**2``
    factor to that series while retaining ODE knots unchanged.
``horizon_policy``
    ``"current"`` uses the clean current TEOBResumS reaction.  The
    ``"leading-order"`` choice adds the published leading-order horizon flux
    to the clean infinity flux.
``remnant_policy``
    ``"uib2016"`` retains the existing standalone UIB2016 mass and spin fit.
    ``"teob-healy-jf"`` is an explicit compatibility diagnostic which uses
    the clean Healy-2014 final mass and printed Jiménez-Forteza final spin.
``pa_initializer_policy``
    ``"analytic"`` retains the canonical differentiable PA initializer.
    ``"quadratic-value"`` is an explicit value-only diagnostic using the
    lower zero-``G'`` root of the squared circular-stationarity equation.
    ``"direct-quadratic-value"`` is an explicit value-only diagnostic using
    the direct binary64/float32 discriminant and signed root branch.

No policy is selected implicitly.  In particular, this adapter is not a
parameter search or a trajectory validation driver.  It is intended for
bounded component attribution and seam tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from torchwave.constants import PI
from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
)

from .conditioned_waveform_clean import project_22_pair_to_polarizations
from .teob_adaptive_continuation_clean import (
    TEOBAdaptiveContinuationResult,
    continue_teob_adaptive,
)
from .teob_merger_nqc_clean import (
    PUBLISHED_2017_NQC_PROVENANCE,
    TEOBMergerNQCResult,
    assemble_teob_merger_nqc,
    published_2017_nqc_targets,
)
from .teob_postadiabatic_clean import (
    PA_INITIALIZER_POLICIES,
    TEOBPostAdiabaticResult,
    alternating8pa,
)
from .teob_radial_acceleration_clean import frozen_mobility_radial_acceleration
from .teob_ringdown_assembly_clean import RingdownAssemblyResult, assemble_teob_ringdown_22
from .teob_trajectory import conservative_trajectory_quantities
from .teob_healy2014_remnant_clean import teob_hybrid_nonspinning_remnant
from .teob_uib2016_remnant import uib2016_nonspinning_remnant
from .teob_waveform_state_clean import teob_waveform_state_mode
from .td_fd_conditioning_clean import (
    ConditionedFD,
    FourierSampling,
    TDSupportPlan,
    condition_td_to_fd,
    fft_length_and_sampling,
    plan_td_support,
)
from .teob_radiation_clean import TEOBRadiationClean, non_keplerian_radius


RADIAL_ACCELERATION_POLICIES = (
    "conservative", "frozen-mobility", "frozen-mobility-stage-aware", "zero",
)
HORIZON_POLICIES = ("current", "leading-order")
REMNANT_POLICIES = ("uib2016", "teob-healy-jf")
NQC_TARGET_POLICIES = ("published-2017",)


class TEOBConditionedWaveformError(ValueError):
    """Raised when the bounded TEOB assembly contract is not met."""


def _stage_aware_frozen_mobility_radial_acceleration(
    hamiltonian: TEOBResumSHamiltonian,
    continuation: TEOBAdaptiveContinuationResult,
) -> Tensor:
    """Apply the opt-in PA acceleration convention on the exact prefix.

    Independently, the published EOB Hamiltonian relation gives
    ``H_real/M = sqrt(1 + 2*nu*(H_eff/mu - 1))``; in this normalized clean
    Hamiltonian ``real_hamiltonian`` is therefore the real energy per total
    mass, ``E = H_real``.  The ``E**2`` multiplication here is the supplied
    compatibility convention for PA NQC data, rather than a prescription
    asserted by those papers.
    """
    state = continuation.state
    acceleration = frozen_mobility_radial_acceleration(
        hamiltonian, state[:, 0], state[:, 2], state[:, 3],
    )
    pa_time = getattr(continuation, "pa_time", None)
    pa_state = getattr(continuation, "pa_state", None)
    if pa_time is None or pa_state is None or pa_time.numel() == 0:
        return acceleration
    n_pa = pa_time.numel()
    if n_pa > state.shape[0] or pa_state.shape != state[:n_pa].shape:
        raise TEOBConditionedWaveformError(
            "continuation PA prefix is inconsistent with combined trajectory"
        )
    energy = hamiltonian.real_hamiltonian(
        pa_state[:, 0], pa_state[:, 2], pa_state[:, 3],
    )
    result = acceleration.clone()
    result[:n_pa] = result[:n_pa] * energy.square()
    return result


def _infer_real_dtype_device(*values: Any, dtype: Optional[torch.dtype] = None,
                             device: Optional[torch.device | str] = None):
    tensors = [value for value in values if isinstance(value, Tensor)]
    if any(value.is_complex() for value in tensors):
        raise TypeError("TEOB orchestration parameters must be real")
    if any(value.dtype == torch.bool for value in tensors) or any(
        isinstance(value, bool) for value in values
    ):
        raise TypeError("TEOB orchestration parameters must be real numbers")
    if device is None:
        devices = [value.device for value in tensors]
        if devices and any(item != devices[0] for item in devices):
            raise ValueError("TEOB orchestration tensors must share a device")
        target_device = devices[0] if devices else torch.device("cpu")
    else:
        target_device = torch.device(device)
    if dtype is None:
        floating = [value.dtype for value in tensors if value.dtype.is_floating_point]
        target_dtype = torch.float64 if not floating else floating[0]
        for item in floating[1:]:
            target_dtype = torch.promote_types(target_dtype, item)
    else:
        target_dtype = dtype
    if target_dtype not in (torch.float32, torch.float64):
        raise TypeError("TEOB orchestration dtype must be float32 or float64")
    return target_dtype, target_device


def _scalar_real(value: Any, *, name: str, dtype: torch.dtype,
                 device: torch.device, positive: bool = False,
                 nonnegative: bool = False) -> Tensor:
    if isinstance(value, bool) or (isinstance(value, Tensor) and value.dtype == torch.bool):
        raise TypeError(f"{name} must be a real scalar")
    if isinstance(value, Tensor) and value.is_complex():
        raise TypeError(f"{name} must be real")
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    if not bool(torch.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if positive and bool(result <= 0):
        raise ValueError(f"{name} must be finite and positive")
    if nonnegative and bool(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _policy(value: Any, *, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be one of {choices}; it is required explicitly")
    if value not in choices:
        raise ValueError(f"unsupported {name} {value!r}; choose one of {choices}")
    return value


def _select_remnant(
    mass1: Tensor,
    mass2: Tensor,
    policy: str,
) -> tuple[Tensor, Tensor]:
    """Select one explicitly named remnant prescription for this adapter.

    The standalone UIB2016 helper remains the adapter default so existing
    callers retain their numerical route.  The Healy/Jiménez choice is only
    reachable through the opt-in policy and deliberately combines the two
    published model versions exposed by its clean helper.
    """

    if policy == "uib2016":
        return uib2016_nonspinning_remnant(mass1, mass2)
    if policy == "teob-healy-jf":
        return teob_hybrid_nonspinning_remnant(mass1, mass2)
    raise TEOBConditionedWaveformError(f"unsupported remnant_policy {policy!r}")


def _copy_options(value: Optional[Mapping[str, Any]], *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of bounded component options")
    return dict(value)


def _reject_options(options: Mapping[str, Any], names: tuple[str, ...], *, name: str) -> None:
    selected = [item for item in names if item in options]
    if selected:
        joined = ", ".join(selected)
        raise ValueError(f"{name} cannot override orchestration-owned option(s): {joined}")


class LeadingOrderHorizonRadiationReaction:
    """Clean leading-order horizon policy used only by this adapter.

    The flux is the nonspinning leading-order absorption term written in the
    same dimensionless EOB variables as the clean radiation component.  The
    infinity flux still comes from :class:`TEOBRadiationClean`; this class
    only supplies the explicitly selected horizon addition and the TEOB
    reduced azimuthal force.
    """

    def __init__(self, *, include_horizon_absorption: bool = True):
        if not isinstance(include_horizon_absorption, bool):
            raise TypeError("include_horizon_absorption must be a bool")
        self.include_horizon_absorption = include_horizon_absorption
        self._radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")

    def radiation_reaction_forces(
        self,
        omega: Tensor,
        pr_star: Tensor,
        pphi: Tensor,
        *,
        r: Optional[Tensor] = None,
        hamiltonian: Optional[TEOBResumSHamiltonian] = None,
        include_horizon_absorption: Optional[bool] = None,
    ) -> tuple[Tensor, Tensor]:
        if r is None or hamiltonian is None:
            raise ValueError("leading-order horizon forces require r and hamiltonian")
        if include_horizon_absorption is None:
            include_horizon_absorption = self.include_horizon_absorption
        if not isinstance(include_horizon_absorption, bool):
            raise TypeError("include_horizon_absorption must be a bool or None")
        h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
            r, pr_star, pphi,
        )
        infinity_flux = self._radiation.flux(
            r,
            omega,
            torch.zeros_like(omega),
            pr_star,
            pphi,
            h_eff,
            h_real,
            h_circ,
            a_prime,
            hamiltonian.m1,
            hamiltonian.m2,
            include_horizon=False,
        )
        if include_horizon_absorption:
            r_omega = non_keplerian_radius(r, h_circ, a_prime)
            v_phi = r_omega * omega
            total_mass = hamiltonian.m1 + hamiltonian.m2
            x1 = hamiltonian.m1 / total_mass
            x2 = hamiltonian.m2 / total_mass
            horizon_ratio = (x1.pow(4) + x2.pow(4)) * v_phi.pow(8)
            horizon_flux = (
                (32.0 / 5.0)
                * hamiltonian.nu.pow(2)
                * r_omega.pow(4)
                * omega.pow(5)
                * horizon_ratio
            )
            flux = infinity_flux + horizon_flux
        else:
            flux = infinity_flux
        nu = hamiltonian.nu
        if omega.dim() > nu.dim():
            nu = nu.unsqueeze(-1)
        return torch.zeros_like(flux), -flux / nu


def _make_hamiltonian(mass1: Tensor, mass2: Tensor) -> TEOBResumSHamiltonian:
    total = mass1 + mass2
    # TEOBResumS uses the positive mass-difference convention.  The dynamics
    # are symmetric in the supplied component masses, so sorting only the
    # signed delta keeps the adapter usable for either component order.
    delta = torch.abs(mass1 - mass2) / total
    nu = torch.minimum(mass1 * mass2 / total.square(), mass1.new_tensor(0.25))
    zero = torch.zeros_like(nu)
    return TEOBResumSHamiltonian(
        nu,
        zero,
        zero,
        zero,
        zero,
        delta=delta,
    )


@dataclass(frozen=True)
class ConditionedTEOBFD:
    """All outputs and stage boundaries of one bounded TEOB assembly."""

    frequencies: Tensor
    h_plus: Tensor
    h_cross: Tensor
    fd: ConditionedFD
    sampling: FourierSampling
    conditioning: Any
    support: TDSupportPlan
    td_time: Tensor
    mode_22: Tensor
    mode_2m2: Tensor
    h_plus_td: Tensor
    h_cross_td: Tensor
    omega_start: Tensor
    initial_radius: Tensor
    total_mass_seconds: Tensor
    coordinate_epoch: Tensor
    physical_epoch_seconds: Tensor
    pa: TEOBPostAdiabaticResult
    continuation: TEOBAdaptiveContinuationResult
    nqc: TEOBMergerNQCResult
    ringdown: RingdownAssemblyResult
    hamiltonian: TEOBResumSHamiltonian
    radial_acceleration_policy: str
    horizon_policy: str
    remnant_policy: str
    nqc_target_policy: str
    nqc_target_provenance: str
    pa_initializer_policy: str
    mass1: Tensor
    mass2: Tensor
    f_low: Tensor
    f_max: Tensor
    delta_f: Tensor
    distance: Tensor
    inclination: Tensor
    phic: Tensor
    tc: Tensor
    matching_method: str = "grid"

    @property
    def hplus(self) -> Tensor:
        return self.h_plus

    @property
    def hcross(self) -> Tensor:
        return self.h_cross

    @property
    def coordinate_start(self) -> Tensor:
        """The ringdown coordinate start, not a peak-shifted observer epoch."""
        return self.coordinate_epoch

    @property
    def metadata(self) -> dict[str, Any]:
        """Named units, policy, and event metadata for audit reports."""
        return {
            "mode": (2, 2),
            "radial_acceleration_policy": self.radial_acceleration_policy,
            "horizon_policy": self.horizon_policy,
            "remnant_policy": self.remnant_policy,
            "nqc_target_policy": self.nqc_target_policy,
            "nqc_target_provenance": self.nqc_target_provenance,
            "pa_initializer_policy": self.pa_initializer_policy,
            "matching_method": self.matching_method,
            "coordinate_epoch": self.coordinate_epoch,
            "physical_epoch_seconds": self.physical_epoch_seconds,
            "omega_start": self.omega_start,
            "initial_radius": self.initial_radius,
            "pa_status": self.pa.status,
            "continuation_status": self.continuation.status,
            "continuation_fixed_step": self.continuation.fixed_step,
            "nqc_matching_index": self.nqc.matching_index,
            "ringdown_events": self.ringdown.events,
            "sampling": self.sampling,
        }


def project_teob22_generic(
    h22: Tensor,
    h2m2: Optional[Tensor] = None,
    *,
    mass1: Any,
    mass2: Any,
    distance: Any = 100.0,
    inclination: Any = 0.0,
    phic: Any = 0.0,
) -> tuple[Tensor, Tensor]:
    """Project the clean TEOB pair in the generic observer frame."""
    return project_22_pair_to_polarizations(
        h22,
        h2m2,
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        inclination=inclination,
        phic=phic,
    )


def project_teob22_direct(
    h22: Tensor,
    h2m2: Optional[Tensor] = None,
    *,
    mass1: Any,
    mass2: Any,
    distance: Any = 100.0,
    inclination: Any = 0.0,
    phic: Any = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return the direct-TEOB projection convention.

    The reviewed direct convention is the negative of the generic observer
    frame for the same pair and extrinsic parameters.  Keeping this relation
    explicit makes a frame-sign check independent of waveform generation.
    """
    generic_plus, generic_cross = project_teob22_generic(
        h22,
        h2m2,
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        inclination=inclination,
        phic=phic,
    )
    return -generic_plus, -generic_cross


# One descriptive alias keeps the pair projection hand-off discoverable.
project_teob22_pair_to_polarizations = project_teob22_generic


def _make_reaction(
    horizon_policy: str,
    hamiltonian: TEOBResumSHamiltonian,
) -> object:
    if horizon_policy == "current":
        return TEOBResumSRadiationReaction(
            nu=hamiltonian.nu,
            chi1=hamiltonian.chi1,
            chi2=hamiltonian.chi2,
            include_horizon_absorption=True,
        )
    return LeadingOrderHorizonRadiationReaction(include_horizon_absorption=True)


def _resolve_targets(
    hamiltonian: TEOBResumSHamiltonian,
    *,
    policy: str,
    targets: object,
    provenance: Optional[str],
) -> tuple[object, str]:
    if targets is not None:
        if provenance is None or not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("nqc_target_provenance is required with explicit nqc_targets")
        return targets, provenance
    if policy == "published-2017":
        return (
            published_2017_nqc_targets(hamiltonian.nu, hamiltonian.nu.new_zeros(())),
            PUBLISHED_2017_NQC_PROVENANCE,
        )
    raise TEOBConditionedWaveformError(f"unsupported NQC target policy {policy!r}")


def conditioned_teob_fd(
    mass1: Any,
    mass2: Any,
    *,
    f_low: Any,
    f_max: Any,
    delta_f: Any,
    distance: Any = 100.0,
    inclination: Any = 0.0,
    phic: Any = 0.0,
    tc: Any = 0.0,
    chi1z: Any = 0.0,
    chi2z: Any = 0.0,
    mode: tuple[int, int] = (2, 2),
    radial_acceleration_policy: Optional[str] = None,
    horizon_policy: Optional[str] = None,
    remnant_policy: str = "uib2016",
    nqc_target_policy: str = "published-2017",
    matching_method: str = "grid",
    pa_initializer_policy: str = "analytic",
    nqc_targets: object = None,
    nqc_target_provenance: Optional[str] = None,
    pa_kwargs: Optional[Mapping[str, Any]] = None,
    continuation_kwargs: Optional[Mapping[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> ConditionedTEOBFD:
    """Generate and condition one scalar clean TEOB 22 waveform.

    ``f_low``, ``f_max`` and ``delta_f`` are detector-frame Hz.  Native
    support planning determines the physical sampling interval; that sample
    interval is converted to geometric TEOB time for ringdown output and is
    then passed back to :func:`condition_td_to_fd`.  The initial PA radius is
    always ``max(r_min, omega_start**(-2/3))`` from the native support start
    frequency, rather than a fixed physical radius.

    The adapter intentionally requires the trajectory and horizon attribution
    policies.  The remnant policy defaults to the pre-existing UIB2016 route
    and must be named as ``"teob-healy-jf"`` to opt into the hybrid diagnostic.
    ``matching_method="continuous"`` shares an accepted-trajectory spline
    peak between NQC and ringdown, with matching at the exact physical time.
    The default ``"grid"`` preserves the existing compatibility convention.
    """
    if type(mode) is not tuple or mode != (2, 2):
        raise NotImplementedError("the clean orchestration adapter supports only mode (2,2)")
    radial_policy = _policy(
        radial_acceleration_policy,
        name="radial_acceleration_policy",
        choices=RADIAL_ACCELERATION_POLICIES,
    )
    horizon = _policy(horizon_policy, name="horizon_policy", choices=HORIZON_POLICIES)
    remnant = _policy(remnant_policy, name="remnant_policy", choices=REMNANT_POLICIES)
    target_policy = _policy(
        nqc_target_policy,
        name="nqc_target_policy",
        choices=NQC_TARGET_POLICIES,
    )
    matching = _policy(matching_method, name="matching_method", choices=("grid", "continuous"))
    pa_initializer = _policy(
        pa_initializer_policy,
        name="pa_initializer_policy",
        choices=PA_INITIALIZER_POLICIES,
    )

    real_dtype, target_device = _infer_real_dtype_device(
        mass1, mass2, f_low, f_max, delta_f, distance, inclination, phic, tc,
        chi1z, chi2z, dtype=dtype, device=device,
    )
    m1 = _scalar_real(mass1, name="mass1", dtype=real_dtype, device=target_device, positive=True)
    m2 = _scalar_real(mass2, name="mass2", dtype=real_dtype, device=target_device, positive=True)
    flow = _scalar_real(f_low, name="f_low", dtype=real_dtype, device=target_device, positive=True)
    fmax = _scalar_real(f_max, name="f_max", dtype=real_dtype, device=target_device, positive=True)
    df = _scalar_real(delta_f, name="delta_f", dtype=real_dtype, device=target_device, positive=True)
    dist = _scalar_real(distance, name="distance", dtype=real_dtype, device=target_device, positive=True)
    inc = _scalar_real(inclination, name="inclination", dtype=real_dtype, device=target_device)
    phase_c = _scalar_real(phic, name="phic", dtype=real_dtype, device=target_device)
    tc_t = _scalar_real(tc, name="tc", dtype=real_dtype, device=target_device)
    spin1 = _scalar_real(chi1z, name="chi1z", dtype=real_dtype, device=target_device)
    spin2 = _scalar_real(chi2z, name="chi2z", dtype=real_dtype, device=target_device)
    if bool(spin1 != 0) or bool(spin2 != 0):
        raise NotImplementedError("the clean orchestration adapter supports only zero aligned spins")
    if bool(fmax < flow):
        raise ValueError("f_max must be at least f_low")

    support = plan_td_support(m1, m2, flow, spin1, spin2, dtype=real_dtype, device=target_device)
    sampling = fft_length_and_sampling(fmax, df)
    # The Fourier helper inherits dtype/device from f_max/delta_f.  Coerce
    # metadata to the assembled parameter dtype without disconnecting graph
    # edges from f_max or delta_f.
    if sampling.dt.dtype != real_dtype or sampling.dt.device != target_device:
        sampling = type(sampling)(
            sampling.n_time,
            sampling.n_frequency,
            sampling.n_nyquist_intervals,
            sampling.f_max_requested.to(dtype=real_dtype, device=target_device),
            sampling.f_nyquist.to(dtype=real_dtype, device=target_device),
            sampling.delta_f.to(dtype=real_dtype, device=target_device),
            sampling.dt.to(dtype=real_dtype, device=target_device),
            sampling.automatic,
        )
    omega_start = torch.as_tensor(PI, dtype=real_dtype, device=target_device) * support.total_mass_seconds * support.start_frequency
    r_min_default = 14.0
    pa_options = _copy_options(pa_kwargs, name="pa_kwargs")
    continuation_options = _copy_options(continuation_kwargs, name="continuation_kwargs")
    _reject_options(
        pa_options,
        ("f_start", "omega_start", "r0", "total_mass_msun", "total_mass", "mass_total", "mass1", "mass2", "radiation_reaction", "include_horizon_absorption", "initializer_policy"),
        name="pa_kwargs",
    )
    _reject_options(
        continuation_options,
        ("pa_result", "initial_state", "initial_step", "radiation_reaction", "include_horizon_absorption", "start_time"),
        name="continuation_kwargs",
    )
    r_min = _scalar_real(
        pa_options.get("r_min", r_min_default), name="r_min", dtype=real_dtype, device=target_device, positive=True,
    )
    # Keep the radius construction in this adapter visible in metadata.  PA
    # repeats the same formula when passed omega_start and r_min; passing r0
    # would obscure that provenance, so it is prohibited above.
    initial_radius = torch.maximum(r_min, omega_start.pow(-2.0 / 3.0))

    hamiltonian = _make_hamiltonian(m1, m2)
    reaction = _make_reaction(horizon, hamiltonian)
    pa_options.setdefault("r_min", r_min)
    pa_options.setdefault("omega_start", omega_start)
    pa_options.setdefault("include_horizon_absorption", True)
    pa_options.setdefault("initializer_policy", pa_initializer)
    pa = alternating8pa(
        hamiltonian,
        reaction,
        **pa_options,
    )
    if not pa.complete or pa.endpoint_state is None:
        raise TEOBConditionedWaveformError(
            f"PA stage did not complete: status={pa.status!r}, reason={pa.skip_reason!r}"
        )

    continuation_options.setdefault("include_horizon_absorption", True)
    continuation = continue_teob_adaptive(
        hamiltonian,
        pa,
        reaction,
        **continuation_options,
    )
    if not continuation.complete:
        raise TEOBConditionedWaveformError(
            "adaptive/fixed continuation did not complete: "
            f"status={continuation.status!r}, reason={continuation.skip_reason!r}"
        )
    if continuation.fixed_step is None:
        raise TEOBConditionedWaveformError("continuation completed without a fixed output step")

    if radial_policy == "conservative":
        radial_acceleration = conservative_trajectory_quantities(
            hamiltonian, continuation.state,
        ).radial_acceleration
    elif radial_policy == "frozen-mobility":
        radial_acceleration = frozen_mobility_radial_acceleration(
            hamiltonian,
            continuation.state[:, 0],
            continuation.state[:, 2],
            continuation.state[:, 3],
        )
    elif radial_policy == "frozen-mobility-stage-aware":
        radial_acceleration = _stage_aware_frozen_mobility_radial_acceleration(
            hamiltonian, continuation,
        )
    else:
        radial_acceleration = torch.zeros_like(continuation.frequency)
    mode22 = teob_waveform_state_mode(
        continuation.state,
        hamiltonian,
        2,
        2,
        Omega=continuation.frequency,
        residual_phase="zero-unpublished",
    )
    targets, target_provenance = _resolve_targets(
        hamiltonian,
        policy=target_policy,
        targets=nqc_targets,
        provenance=nqc_target_provenance,
    )
    nqc = assemble_teob_merger_nqc(
        continuation.time,
        continuation.state,
        orbital_frequency=continuation.frequency,
        radial_acceleration=radial_acceleration,
        nu=hamiltonian.nu,
        fixed_step=continuation.fixed_step,
        targets=targets,
        target_provenance=target_provenance,
        mode=mode22,
        orbital_phase=continuation.state[:, 1],
        matching_method=matching,
    )
    final_mass_fraction, final_spin = _select_remnant(m1, m2, remnant)
    dt_geom = sampling.dt / support.total_mass_seconds
    ringdown = assemble_teob_ringdown_22(
        nqc.original.time,
        nqc.original.corrected_amplitude,
        nqc.original.corrected_phase,
        nqc.auxiliary.time,
        nqc.auxiliary.corrected_amplitude,
        nqc.auxiliary.corrected_phase,
        nqc.auxiliary.orbital_frequency,
        hamiltonian.nu,
        final_mass_fraction,
        final_spin,
        output_step=dt_geom,
        matching_method=matching,
        peak_time=nqc.t_peak if matching == "continuous" else None,
    )
    mode2m2 = torch.conj(ringdown.mode)
    h_plus_td, h_cross_td = project_teob22_generic(
        ringdown.mode,
        mode2m2,
        mass1=m1,
        mass2=m2,
        distance=dist,
        inclination=inc,
        phic=phase_c,
    )
    # condition_td_to_fd derives the same Fourier rate from (f_max, delta_f)
    # and receives the native physical dt explicitly for an auditable seam.
    fd = condition_td_to_fd(
        h_plus_td,
        h_cross_td,
        dt=sampling.dt,
        delta_f=df,
        f_max=fmax,
        support=support,
        tc=tc_t,
        return_metadata=True,
    )
    if not isinstance(fd, ConditionedFD):  # pragma: no cover - type guard
        raise RuntimeError("condition_td_to_fd did not return metadata")
    # The public TEOB epoch is tied to the first accepted orbital-frequency
    # decrease event.  The ringdown assembly also computes a continuous
    # amplitude/peak polynomial, but that interpolated peak is a distinct
    # diagnostic and must not silently become the observer epoch.
    if continuation.peak_event_time is None:
        raise TEOBConditionedWaveformError(
            "completed continuation has no accepted frequency-decrease event"
        )
    physical_epoch_seconds = (
        ringdown.time[0] - continuation.peak_event_time
    ) * support.total_mass_seconds
    return ConditionedTEOBFD(
        frequencies=fd.frequencies,
        h_plus=fd.h_plus,
        h_cross=fd.h_cross,
        fd=fd,
        sampling=fd.sampling,
        conditioning=fd.conditioning,
        support=support,
        td_time=ringdown.time,
        mode_22=ringdown.mode,
        mode_2m2=mode2m2,
        h_plus_td=h_plus_td,
        h_cross_td=h_cross_td,
        omega_start=omega_start,
        initial_radius=initial_radius,
        total_mass_seconds=support.total_mass_seconds,
        coordinate_epoch=ringdown.epoch,
        physical_epoch_seconds=physical_epoch_seconds,
        pa=pa,
        continuation=continuation,
        nqc=nqc,
        ringdown=ringdown,
        hamiltonian=hamiltonian,
        radial_acceleration_policy=radial_policy,
        horizon_policy=horizon,
        remnant_policy=remnant,
        nqc_target_policy=target_policy,
        nqc_target_provenance=target_provenance,
        pa_initializer_policy=pa_initializer,
        mass1=m1,
        mass2=m2,
        f_low=flow,
        f_max=fmax,
        delta_f=df,
        distance=dist,
        inclination=inc,
        phic=phase_c,
        tc=tc_t,
        matching_method=matching,
    )


# Descriptive aliases do not alter routing or introduce another production
# implementation path.
condition_teob_to_fd = conditioned_teob_fd
teob_conditioned_fd = conditioned_teob_fd
teob_td_to_fd = conditioned_teob_fd
conditioned_teob_waveform = conditioned_teob_fd


__all__ = [
    "RADIAL_ACCELERATION_POLICIES",
    "HORIZON_POLICIES",
    "REMNANT_POLICIES",
    "NQC_TARGET_POLICIES",
    "PA_INITIALIZER_POLICIES",
    "TEOBConditionedWaveformError",
    "LeadingOrderHorizonRadiationReaction",
    "ConditionedTEOBFD",
    "project_teob22_generic",
    "project_teob22_direct",
    "project_teob22_pair_to_polarizations",
    "conditioned_teob_fd",
    "condition_teob_to_fd",
    "teob_conditioned_fd",
    "teob_td_to_fd",
    "conditioned_teob_waveform",
]
