"""Static catalog of the currently scoped executable waveform references.

Keep this metadata independent of ``diffgw_reference`` imports: the latter
needs optional symbolic dependencies and remains an independent evaluator.
The component IDs and versions below mirror the reference modules exactly.
"""

from __future__ import annotations

from .schema import (
    ComponentDefinition, ConventionDefinition, OptionDefinition,
    ParameterDefinition, SourceDefinition, SourceRef, UnitDefinition,
    VerificationClaim, WaveformDefinition, validate_catalog,
)


def _source(source_id: str, location: str) -> tuple[SourceRef, ...]:
    return (SourceRef(source_id, location),)


_TF2_VERSION = "taylorf2-nonspinning-3.5pn/v1"
_TF2_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"tf2.ns35.{name}.v1", name=name.replace("_", " ").title(),
        spec_version=_TF2_VERSION, scope="nonspinning point-particle 3.5PN",
        variant="paper" if name == "paper_phase" else "project_adapter" if name in ("coalescence_phase", "reference_phase") else "canonical_reference",
        implementation="reference_only" if name == "paper_phase" else "native_and_reference",
        evidence=("source_transcription",) if name == "paper_phase" else
                 ("source_transcription", "sampled_native_comparison"),
        source_refs=_source(
            "project" if name in ("coalescence_phase", "reference_phase") else
            "arun2008" if name in ("amplitude", "polarizations") else "buonanno2009",
            "phase-origin convention" if name in ("coalescence_phase", "reference_phase") else
            "Eqs. 6.13, D1" if name in ("amplitude", "polarizations") else "Eq. 3.18",
        ),
        reference_module="diffgw_reference.taylorf2",
    )
    for name in (
        "phi0", "phi2", "phi3", "phi4", "phi5", "phi5_log", "phi6",
        "phi6_log", "phi7", "intrinsic_phase", "paper_phase", "amplitude",
        "polarizations", "coalescence_phase", "reference_phase",
    )
)


_D_AMP = "imrphenomd-amplitude-joins-paper-precision/v1"
_D_PHASE = "imrphenomd-phase-joins-paper-precision/v1"
_D_REMNANT = "imrphenomd-remnant-qnm-paper-precision/v1"
_D_WAVEFORM = "imrphenomd-composed-polarizations/v1"
_D_INTRINSIC = "imrphenomd-intrinsic-composition/v1"


def _d_component(module: str, version: str, component_id: str, location: str,
                 *, source_id: str = "khan2016", variant: str = "paper_precision",
                 implementation: str = "native_and_reference",
                 scope: str = "aligned-spin dominant-harmonic component",
                 caveats: tuple[str, ...] = ()) -> ComponentDefinition:
    return ComponentDefinition(
        id=component_id, name=component_id.rsplit(".", 1)[0].split(".")[-1].replace("_", " ").title(),
        spec_version=version, scope=scope, variant=variant,
        implementation=implementation,
        evidence=("source_transcription", "conditional_native_comparison")
                 if implementation == "native_and_reference" else ("source_transcription",),
        source_refs=_source(source_id, location),
        reference_module=f"diffgw_reference.{module}", caveats=caveats,
    )


_D_COMPONENTS = (
    *tuple(_d_component(
        "imrphenomd_amplitude", _D_AMP, f"phenomd.amp.fit.{name}.v1",
        "Eq. 31; Table 5", caveats=("Production retains inherited extra digits with unresolved publication provenance.",)
    ) for name in ("rho1", "rho2", "rho3", "v2", "gamma1", "gamma2", "gamma3")),
    *tuple(_d_component(
        "imrphenomd_amplitude", _D_AMP, f"phenomd.amp.{name}.v1",
        location, variant="project_adapter" if name == "peak" else "paper_precision",
    ) for name, location in (
        ("spin_coordinates", "Eqs. 2–4"), ("pn", "Eqs. 54–60"),
        ("inspiral", "Eqs. 29–30"), ("merger_ringdown", "Eq. 19"),
        ("peak", "Eq. 20; continuation for |gamma2| >= 1"),
        ("intermediate", "Eqs. 21–26; Table 2"),
    )),
    *tuple(_d_component(
        "imrphenomd_phase", _D_PHASE, f"phenomd.phase.fit.{name}.v1",
        "Eq. 31; Table 5",
        implementation="reference_only" if name == "beta1" else "native_and_reference",
        caveats=(
            "Printed beta1 is retained as source evidence; the waveform uses the continuity-determined value."
            if name == "beta1" else
            "Production retains inherited extra digits with unresolved publication provenance.",
        )
    ) for name in ("sigma1", "sigma2", "sigma3", "sigma4", "beta1", "beta2", "beta3",
                   "alpha1", "alpha2", "alpha3", "alpha4", "alpha5")),
    *tuple(_d_component(
        "imrphenomd_phase", _D_PHASE, f"phenomd.phase.{name}.v1", location,
    ) for name, location in (
        ("pn", "Eqs. 27, 41–53"), ("inspiral", "Eq. 28"),
        ("intermediate", "Eqs. 15–16"), ("merger_ringdown", "Eqs. 13–14"),
        ("joins", "Eq. 37"),
    )),
    _d_component("imrphenomd_phase", _D_PHASE, "phenomd.phase.frontend_gauge.v1",
                 "raw fitted alpha1 at amplitude peak", source_id="project",
                 variant="carrier_convention"),
    *tuple(_d_component(
        "imrphenomd_remnant", _D_REMNANT, component_id, location,
        source_id=source_id, variant=variant,
        scope="remnant and QNM input composition",
    ) for component_id, location, source_id, variant in (
        ("phenomd.remnant.coordinates.v1", "Sec. III.1", "husa2016", "paper_precision"),
        ("phenomd.remnant.spin.v1", "Eq. 22", "husa2016", "paper_precision"),
        ("phenomd.remnant.energy.v1", "Eqs. 23–24", "husa2016", "paper_precision"),
        ("phenomd.remnant.mass.v1", "Eq. 24", "husa2016", "paper_precision"),
        ("phenomd.qnm.total_mass_scaling.v1", "Eq. 17", "husa2016", "paper_precision"),
        ("phenomd.qnm.natural_cubic_compatibility.v1", "numerical interpolation convention", "project", "carrier_convention"),
    )),
    *tuple(_d_component(
        "imrphenomd_waveform", _D_WAVEFORM,
        f"phenomd.waveform.{name}.v1", location,
        source_id=source_id, variant=variant,
        scope="composed complex polarizations with supplied QNM/fit inputs",
    ) for name, location, source_id, variant in (
        ("physical_scale", "Eq. 63; SI/IAU unit conversion", "cutler_flanagan1994", "project_adapter"),
        ("phase_adapter", "Eq. 7; project phase and time origin", "khan2016", "project_adapter"),
        ("polarizations", "Eqs. 5–7, 62", "khan2016", "project_adapter"),
        ("default_cutoff", "Mf=0.2", "project", "carrier_convention"),
        ("polarization_derivatives", "product rule", "project", "derived"),
    )),
    *tuple(_d_component(
        "imrphenomd_intrinsic", _D_INTRINSIC,
        f"phenomd.intrinsic.{name}.v1", location,
        source_id=source_id, variant="conditional_composition",
        scope="full intrinsic composition with declared QNM knots and fit rows",
    ) for name, location, source_id in (
        ("remnant_qnm", "Eqs. 17, 22–24; supplied numerical QNM data", "husa2016"),
        ("polarizations", "Eqs. 5–7, 17–18, 36–39", "khan2016"),
        ("derivatives", "derived finite-difference checks", "project"),
    )),
)


_XAS_QUARTIC = "imrphenomxas-intermediate-amplitude-quartic/v1"
_XAS_FIFTH = "imrphenomxas-intermediate-amplitude-fifth-order/v1"
_XAS_AMP = "imrphenomxas-inspiral-ringdown-amplitude/v1"
_XAS_PHASE = "imrphenomxas-intermediate-ringdown-phase/v1"
_XAS_INSPIRAL = "imrphenomxas-inspiral-phase/v1"
_XAS_WAVEFORM = "imrphenomxas-composed-polarizations-carrier/v1"
_XAS_MASS_SCALING = "imrphenomxas-total-mass-scaling/v1"


def _xas_component(module: str, version: str, component_id: str,
                   location: str, *, variant: str = "carrier",
                   implementation: str = "native_and_reference",
                   source_id: str = "pratten2020v1",
                   caveats: tuple[str, ...] = ()) -> ComponentDefinition:
    return ComponentDefinition(
        id=component_id, name=component_id.rsplit(".", 1)[0].split(".")[-1].replace("_", " ").title(),
        spec_version=version, scope="aligned-spin (2,2) component with supplied calibration outputs",
        variant=variant, implementation=implementation,
        evidence=("source_transcription", "conditional_native_comparison")
                 if implementation in ("native_and_reference", "carrier_native_paper_reference")
                 else ("source_transcription",),
        source_refs=_source(source_id, location),
        reference_module=f"diffgw_reference.{module}", caveats=caveats,
    )


_XAS_COMPONENTS = (
    *tuple(_xas_component(
        "imrphenomxas_intermediate", _XAS_QUARTIC,
        f"phenomxas.amp.intermediate.{name}.v1", location,
        variant="paper_and_carrier" if name in ("quartic", "collocation") else "paper",
        implementation="carrier_native_paper_reference" if name in ("quartic", "collocation")
                       else "native_and_reference",
        caveats=("Printed A0(f)/P and carrier K/P have different frequency dependence.",)
                if name in ("quartic", "collocation") else (),
    ) for name, location in (
        ("quartic", "Eq. 71"), ("newtonian_factor", "Eq. 67"),
        ("collocation", "Eqs. 68–69, 79–84; Table 2"),
    )),
    ComponentDefinition(
        id="phenomxas.amp.intermediate.constant_normalization.v1",
        name="Constant normalization",
        spec_version=_XAS_QUARTIC,
        scope="Aligned-spin (2,2) carrier intermediate amplitude K/P with supplied calibration outputs",
        variant="carrier",
        implementation="native_and_reference",
        evidence=("declared_carrier_convention", "sampled_binary_oracle_comparison"),
        source_refs=_source("project", "Quartic intermediate amplitude: K/P carrier convention"),
        reference_module="diffgw_reference.imrphenomxas_intermediate",
        caveats=("This normalization is distinct from the printed Eq. 71; binary agreement does not establish source fidelity.",),
    ),
    *tuple(_xas_component(
        "imrphenomxas_fifth_order", _XAS_FIFTH, component_id, location,
        variant="paper", implementation="reference_only",
        caveats=("Current native XAS carriers use the quartic option.",),
    ) for component_id, location in (
        ("phenomxas.amp.intermediate.fifth_order.v1", "Eqs. 67, 72"),
        ("phenomxas.amp.intermediate.fifth_order_collocation.v1", "Eqs. 73–78; Table 1"),
    )),
    *tuple(_xas_component(
        "imrphenomxas_amplitude", _XAS_AMP, component_id, location,
        variant=variant, implementation=implementation,
    ) for component_id, location, variant, implementation in (
        ("phenomxas.amp.inspiral.pn.v1", "Eqs. 67, 69, 119–126", "paper", "native_and_reference"),
        ("phenomxas.amp.inspiral.pseudo_pn.v1", "Eq. 68", "paper", "native_and_reference"),
        ("phenomxas.amp.inspiral.collocation.v1", "Eqs. 52, 68, 70", "carrier", "native_and_reference"),
        ("phenomxas.amp.ringdown.lorentzian.v1", "Eq. 84", "paper", "native_and_reference"),
        ("phenomxas.amp.ringdown.peak_printed.v1", "Eq. 59", "paper", "reference_only"),
        ("phenomxas.amp.ringdown.peak_stationary.v1", "stationary point of Eq. 84", "carrier", "native_and_reference"),
    )),
    *tuple(_xas_component(
        "imrphenomxas_phase_branches", _XAS_PHASE, component_id, location,
        variant=variant, implementation=implementation,
        caveats=("Printed Eq. 90 and the native carrier select opposite Lorentzian signs.",)
                if "intermediate" in component_id else (),
    ) for component_id, location, variant, implementation in (
        ("phenomxas.phase.ringdown.ansatz.v1", "Eq. 96", "paper", "native_and_reference"),
        ("phenomxas.phase.ringdown.collocation.v1", "Eqs. 96–97", "carrier", "native_and_reference"),
        ("phenomxas.phase.intermediate.printed.v1", "Eq. 90", "paper", "reference_only"),
        ("phenomxas.phase.intermediate.collocation.v1", "Eqs. 90, 92–95", "paper_and_carrier", "carrier_native_paper_reference"),
        ("phenomxas.phase.intermediate.carrier_sign.v1", "Eq. 90; declared carrier sign", "carrier", "native_and_reference"),
        ("phenomxas.phase.intermediate_ringdown.join.v1", "Eqs. 90, 96", "carrier", "native_and_reference"),
    )),
    *tuple(_xas_component(
        "imrphenomxas_inspiral_phase", _XAS_INSPIRAL,
        f"phenomxas.phase.inspiral.{name}.v1", location,
        variant="paper" if name in ("pn", "spin_tail", "pseudo_pn") else "carrier",
    ) for name, location in (
        ("pn", "Eqs. 86, 129–137"), ("spin_tail_4pn", "Eq. 138"),
        ("pseudo_pn", "Eq. 85"), ("collocation", "Eqs. 87–88"),
        ("join", "Eq. 85; Sec. VII.1"),
    )),
    *tuple(_xas_component(
        "imrphenomxas_waveform", _XAS_WAVEFORM,
        f"phenomxas.waveform.{name}.v1", location,
        variant="carrier" if name in ("branch_selection", "phase_adapter", "nonpositive_mask") else "project_adapter",
        source_id="project" if name in ("nonpositive_mask", "extrinsic_derivatives",
                                       "frequency_derivatives", "mass_derivatives",
                                       "intrinsic_composition_derivatives") else "pratten2020v1",
    ) for name, location in (
        ("branch_selection", "Eqs. 52, 71, 85, 90, 96; carrier join convention"),
        ("physical_scale", "Eqs. 14–16, 66–67; dimensional restoration"),
        ("phase_adapter", "Eqs. 14–16; carrier phase convention"),
        ("polarizations", "Eqs. 14–16; carrier polarization convention"),
        ("nonpositive_mask", "f <= 0 carrier convention"),
        ("extrinsic_derivatives", "derived product rule"),
        ("frequency_derivatives", "derived product rule"),
        ("mass_scaling", "Eqs. 1, 14–16, 66–67; dimensional similarity"),
        ("mass_derivatives", "derived chain rule"),
        ("intrinsic_composition_derivatives", "derived chain rule; supplied author fits"),
    )),
)

# The scaling identity was separately versioned within the composition module.
_XAS_COMPONENTS = tuple(
    ComponentDefinition(
        id=component.id, name=component.name,
        spec_version=_XAS_MASS_SCALING if component.id == "phenomxas.waveform.mass_scaling.v1" else component.spec_version,
        scope=component.scope, variant=component.variant,
        implementation=component.implementation, evidence=component.evidence,
        source_refs=component.source_refs, reference_module=component.reference_module,
        caveats=component.caveats,
    ) for component in _XAS_COMPONENTS
)


_COMMON_PARAMETERS = (
    ParameterDefinition("mass1", "First component mass", "nominal solar mass", "> 0"),
    ParameterDefinition("mass2", "Second component mass", "nominal solar mass", "> 0"),
    ParameterDefinition("distance", "Luminosity distance", "Mpc", "> 0"),
    ParameterDefinition("inclination", "Orbital inclination", "rad", "0 <= inclination <= pi"),
    ParameterDefinition("phic", "Orbital phase parameter", "rad", "finite"),
    ParameterDefinition("tc", "Time shift", "s", "finite"),
    ParameterDefinition("f_ref", "Optional phase reference frequency", "Hz", "absent or > 0"),
    ParameterDefinition("sample_frequencies", "Evaluation frequencies", "Hz", "model-dependent frequency mask"),
)
_SPINS = (
    ParameterDefinition("chi1", "Aligned spin of body 1", "dimensionless", "-1 <= chi1 <= 1"),
    ParameterDefinition("chi2", "Aligned spin of body 2", "dimensionless", "-1 <= chi2 <= 1"),
)
_COMMON_UNITS = (
    UnitDefinition("component mass", "nominal solar mass", "M_seconds = (mass1 + mass2) * T_sun_nominal"),
    UnitDefinition("frequency", "Hz", "dimensionless frequency Mf = M_seconds * f_Hz"),
    UnitDefinition("luminosity distance", "Mpc", "D_seconds = distance_Mpc * Mpc_SI / c_SI"),
    UnitDefinition("inclination and orbital phase", "rad", "angles are radians"),
    UnitDefinition("time shift", "s", "phase shift is 2*pi*f_Hz*tc"),
)
_COMMON_CONVENTIONS = (
    ConventionDefinition("fourier", "Positive-frequency strain uses exp(-i*Psi)."),
    ConventionDefinition("polarizations", "h_plus=(1+cos(iota)^2) A exp(-i*Psi)/2; h_cross=-i cos(iota) A exp(-i*Psi)."),
)

TAYLORF2 = WaveformDefinition(
    id="TaylorF2", name="TaylorF2 nonspinning reference", aliases=(),
    domain="frequency", modes=((2, 2),),
    scope="Nonspinning 3.5PN inspiral with restricted quadrupole amplitude and positive frequencies.",
    reference_version=_TF2_VERSION, reference_module="diffgw_reference.taylorf2",
    spec_page="math/taylorf2.md",
    components=_TF2_COMPONENTS,
    sources=(
        SourceDefinition("buonanno2009", "Buonanno et al., nonspinning PN phase", "https://arxiv.org/pdf/0907.0700v1", "0907.0700v1"),
        SourceDefinition("arun2008", "Arun et al., restricted amplitude and polarizations", "https://arxiv.org/pdf/0810.5336v1", "0810.5336v1"),
        SourceDefinition("project", "DiffGW TaylorF2 phase adapter", "docs/math/taylorf2.md", _TF2_VERSION, kind="project_convention"),
    ),
    parameters=_COMMON_PARAMETERS,
    units=_COMMON_UNITS,
    conventions=(*_COMMON_CONVENTIONS,
        ConventionDefinition("phase_gauge", "Canonical intrinsic phase uses log(v); paper Eq. 3.18 uses log(v/v_lso), differing by a constant phase gauge."),
    ),
    options=(OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                              "Reference-frequency mode subtracts intrinsic phase at f_ref; its phase origin is a project convention. "
                              "Native calls select that mode when f_ref is supplied without an explicit convention; "
                              "coalescence is the default when f_ref is omitted."),),
    carrier_parameters=(), capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("symbolic_source_transcription", "sampled_torch_jax_comparison"),
    caveats=("This catalog entry covers the nonspinning scalar reference only; native TaylorF2 also has spin options.",
             "The analytic positive-frequency domain is broader than the physically reliable slow-inspiral regime."),
)

_TF2_ECC_VERSION = "taylorf2-eccentric-nonspinning-e2-3pn/v1"
_TF2_ECC_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"tf2.ecc.{name}.v1", name=name.replace("_", " ").title(),
        spec_version=_TF2_ECC_VERSION,
        scope="nonspinning, comparable-mass, order-e0² 3PN eccentric phase with restricted amplitude",
        variant="paper" if name == "paper_phase" else "project_adapter" if name == "reference_phase" else "composed_reference",
        implementation="native_and_reference",
        evidence=("source_transcription", "sampled_torch_jax_comparison", "two_case_lal_holdout"),
        source_refs=_source("moore2016" if name == "paper_phase" else "project",
                            "Eq. (6.26)" if name == "paper_phase" else "explicit f_ref subtraction and amplitude-off composition"),
        reference_module="diffgw_reference.taylorf2_eccentric",
    ) for name in ("paper_phase", "reference_phase", "polarizations")
)

TAYLORF2_ECCENTRIC = WaveformDefinition(
    id="TaylorF2_Eccentric", name="TaylorF2_Eccentric nonspinning low-e slice",
    aliases=(), domain="frequency", modes=((2, 2),),
    scope="Nonspinning comparable-mass (q <= 2), 0 < e0 <= 0.1, positive-frequency inspiral: restricted quadrupole amplitude and 3PN order-e0² eccentric phase at explicit f0_ecc and f_ref, with amplitude correction off.",
    reference_version=_TF2_ECC_VERSION,
    reference_module="diffgw_reference.taylorf2_eccentric",
    spec_page="math/taylorf2-eccentric.md",
    components=(*tuple(component for component in _TF2_COMPONENTS
                       if component.id not in {"tf2.ns35.paper_phase.v1", "tf2.ns35.coalescence_phase.v1"}),
                *_TF2_ECC_COMPONENTS),
    sources=(
        SourceDefinition("buonanno2009", "Buonanno et al., nonspinning PN phase", "https://arxiv.org/pdf/0907.0700v1", "0907.0700v1"),
        SourceDefinition("arun2008", "Arun et al., restricted amplitude and polarizations", "https://arxiv.org/pdf/0810.5336v1", "0810.5336v1"),
        SourceDefinition("moore2016", "Moore et al., low-eccentricity 3PN SPA phase", "https://arxiv.org/pdf/1605.00304v2", "1605.00304v2"),
        SourceDefinition("project", "DiffGW explicit reference-frequency composition", "docs/math/taylorf2-eccentric.md", _TF2_ECC_VERSION, kind="project_convention"),
    ),
    parameters=(*tuple(parameter for parameter in _COMMON_PARAMETERS if parameter.name != "f_ref"),
                ParameterDefinition("f_ref", "Explicit phase reference frequency", "Hz", "> 0"),
                ParameterDefinition("f0_ecc", "Frequency defining e0", "Hz", "> 0"),
                ParameterDefinition("e0", "Eccentricity at f0_ecc", "dimensionless", "0 < e0 <= 0.1"),
                ParameterDefinition("chi1", "Aligned spin of body 1", "dimensionless", "= 0"),
                ParameterDefinition("chi2", "Aligned spin of body 2", "dimensionless", "= 0")),
    units=_COMMON_UNITS,
    conventions=(*_COMMON_CONVENTIONS,
                 ConventionDefinition("eccentricity_reference", "e0 belongs to f0_ecc, separately from phase-origin f_ref."),
                 ConventionDefinition("phase_gauge", "Subtract both circular and eccentric intrinsic phases at explicit f_ref; this is a project convention.")),
    options=(
        OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                         "Only explicit reference_frequency is included in this versioned slice; coalescence remains a native option."),
        OptionDefinition("ecc_order", ("0", "2", "3", "4", "5", "6"), "6",
                         "Only 6 (relative 3PN, order e0²) is included in the composed model slice."),
        OptionDefinition("include_amplitude_correction", ("false", "true"), "false",
                         "Only false is qualified; the optional true branch has unresolved source provenance."),
        OptionDefinition("f_isco_cutoff", ("false", "true"), "true",
                         "The holdout bands end below Schwarzschild ISCO, so the default hard cutoff is inactive."),
    ),
    carrier_parameters=(), capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("symbolic_source_transcription", "sampled_torch_jax_comparison", "two_case_strict_lal_holdout"),
    caveats=("The native API accepts aligned spin and a wider eccentricity range; these are outside this definition.",
             "Physical accuracy is not established beyond the low-eccentricity, slow-inspiral comparable-mass regime.",
             "The hard cutoff boundary, optional amplitude factor, and aligned-spin eccentric composition remain provisional."),
    claims=(VerificationClaim(
        component_ids=("tf2.ecc.polarizations.v1",),
        scope="Two frozen nonspinning, nonzero-eccentricity, off-catalog binaries below Schwarzschild ISCO",
        variant="amplitude_off_reference_frequency_3pn",
        backends=("torch", "jax", "lal"),
        options=(("phasing_convention", "reference_frequency"), ("ecc_order", "6"),
                 ("include_amplitude_correction", "false"), ("f_isco_cutoff", "true")),
        domain="positive-frequency, q <= 2, e0 <= 0.1", precision="float64",
        device="cpu", category="external_oracle", status="sampled_pass",
        evidence_path="docs/validation/taylorf2_eccentric_nonspinning_holdout_20260927.json",
    ),),
)

_TF2_HM_VERSION = "taylorf2-hm-leading-carrier/v1"
_TF2_HM_MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4))
_TF2_HM_COMPONENTS = (
    *tuple(ComponentDefinition(
        id=f"tf2.hm.ratio_{l}{m}.v1", name=f"({l},{m}) carrier amplitude ratio",
        spec_version=_TF2_HM_VERSION, scope="leading nonspinning mode amplitude",
        variant="leading_spa_carrier", implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source("mezzasoma2022", "Eqs. 20–21, 30"),
        reference_module="diffgw_reference.taylorf2_hm",
    ) for l, m in _TF2_HM_MODES),
    *tuple(ComponentDefinition(
        id=f"tf2.hm.projection_{l}{m}.v1", name=f"({l},{m}) harmonic projection",
        spec_version=_TF2_HM_VERSION, scope="spin-weighted angular projection",
        variant="negative_m_positive_frequency", implementation="native_and_reference",
        evidence=("wigner_d_construction", "sampled_native_comparison"),
        source_refs=_source("garciaquiros2020", "Eqs. 2–3, 9"),
        reference_module="diffgw_reference.taylorf2_hm",
    ) for l, m in _TF2_HM_MODES),
    *tuple(ComponentDefinition(
        id=f"tf2.hm.{name}.v1", name=name.replace("_", " ").title(),
        spec_version=_TF2_HM_VERSION, scope="leading-amplitude aligned-spin carrier",
        variant="project_adapter", implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source(source, location),
        reference_module="diffgw_reference.taylorf2_hm",
    ) for name, source, location in (
        ("mode_phase", "garciaquiros2020", "Eqs. 4–9; project phase origin"),
        ("polarizations", "garciaquiros2020", "Eqs. 2–3; negative-m storage"),
        ("cutoff_frequency", "project_hm", "Schwarzschild ISCO mode mask"),
    )),
)

TAYLORF2_HM = WaveformDefinition(
    id="TaylorF2_HM", name="TaylorF2_HM leading eight-mode carrier",
    aliases=(), domain="frequency", modes=_TF2_HM_MODES,
    scope="Eight leading nonspinning SPA mode amplitudes with inherited aligned-spin TaylorF2 phase and negative-m polarization projection.",
    reference_version=_TF2_HM_VERSION, reference_module="diffgw_reference.taylorf2_hm",
    spec_page="math/taylorf2-hm.md", components=_TF2_HM_COMPONENTS,
    sources=(
        SourceDefinition("mezzasoma2022", "Mezzasoma and Yunes, higher-harmonic SPA amplitudes", "https://arxiv.org/html/2203.15934", "2203.15934"),
        SourceDefinition("mishra2016", "Mishra et al., multipolar frequency-domain inspiral", "https://arxiv.org/html/1601.05588", "1601.05588v1"),
        SourceDefinition("garciaquiros2020", "García-Quirós et al., frequency-domain mode conventions", "https://arxiv.org/html/2001.10914", "2001.10914"),
        SourceDefinition("project_hm", "DiffGW leading carrier and cutoff conventions", "docs/math/taylorf2-hm.md", _TF2_HM_VERSION, kind="project_convention"),
    ),
    parameters=(*_COMMON_PARAMETERS, *_SPINS,
                ParameterDefinition("phi0", "Observer azimuth", "rad", "finite")),
    units=_COMMON_UNITS,
    conventions=(
        ConventionDefinition("fourier", "Positive-frequency modes use exp(-i*Psi_lm) and store h_{l,-m}(f)."),
        ConventionDefinition("mode_phase", "Psi_lm=(m/2)Psi_22(2f/m)+(3*pi/4)(1-m/2); reference frequency is scaled by 2/m."),
        ConventionDefinition("polarizations", "d_plus=[Y_l,-m+(-1)^l conjugate(Y_l,m)]/2; d_cross=i[Y_l,-m-(-1)^l conjugate(Y_l,m)]/2."),
    ),
    options=(
        OptionDefinition("spin_order", ("-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"), "7", "Integer selectors use the inherited aligned-spin option map. Values below zero have the -1 recipe; values above eight have the 8 recipe. The sampled HM comparison uses the public default 7."),
        OptionDefinition("f_isco_cutoff", ("false", "true"), "true", "The reference supplies the Schwarzschild mode threshold; direct complex comparison samples use false."),
        OptionDefinition("smooth_cutoff", ("false", "true"), "false", "The current reference does not evaluate the native half-cosine taper."),
    ),
    carrier_parameters=(), capabilities=("fd", "higher_modes", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("leading_mode_source_transcription", "sampled_torch_jax_comparison"),
    caveats=(
        "The independent scalar comparison covers the unwindowed carrier and Schwarzschild hard-cutoff thresholds, not Kerr ISCO or taper derivatives.",
        "Mode amplitudes omit higher PN and spin-amplitude terms; aligned-spin phase inherits the separately scoped TaylorF2 reference.",
        "No exact external all-eight-mode TaylorF2 oracle or physical validation is claimed; historical LAL comparison covers only (2,2).",
    ),
)

IMRPHENOMD = WaveformDefinition(
    id="IMRPhenomD", name="IMRPhenomD aligned-spin dominant harmonic", aliases=(),
    domain="frequency", modes=((2, 2),),
    scope="Aligned-spin (2,2) IMR composition with declared fit rows and QNM data.",
    reference_version=_D_INTRINSIC, reference_module="diffgw_reference.imrphenomd_intrinsic",
    spec_page="math/imrphenomd-intrinsic.md",
    components=_D_COMPONENTS,
    sources=(
        SourceDefinition("khan2016", "Khan et al., IMRPhenomD waveform", "https://arxiv.org/html/1508.07253v2", "1508.07253v2"),
        SourceDefinition("husa2016", "Husa et al., remnant fits", "https://arxiv.org/html/1508.07250v2", "1508.07250v2"),
        SourceDefinition("cutler_flanagan1994", "Cutler and Flanagan, stationary-phase amplitude", "https://arxiv.org/pdf/gr-qc/9402014v1", "gr-qc/9402014v1"),
        SourceDefinition("project", "DiffGW IMRPhenomD adapter and numerical conventions", "docs/math/imrphenomd-waveform.md", _D_WAVEFORM, kind="project_convention"),
    ),
    parameters=(*_COMMON_PARAMETERS, *_SPINS), units=_COMMON_UNITS,
    conventions=(*_COMMON_CONVENTIONS,
        ConventionDefinition("phase_origin", "Coalescence phase uses -pi/4; finite-reference mode subtracts intrinsic phase at f_ref."),
        ConventionDefinition("time_gauge", "Native carrier uses the raw fitted alpha1 at the amplitude peak."),
    ),
    options=(
        OptionDefinition("calibration", ("inherited", "published"), "inherited",
                         "Published uses printed Khan Table V and Husa remnant coefficients throughout; inherited preserves historical extra digits. Generated QNM data are shared numerical inputs."),
        OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                         "Supplying f_ref without an explicit convention selects reference_frequency; finite-reference phase subtraction and amplitude-peak time gauge are project conventions."),
    ),
    carrier_parameters=(), capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("paper_precision_source_transcription", "conditional_sampled_torch_jax_comparison"),
    caveats=("Inherited extra production fit digits have unresolved publication provenance.",
             "Full intrinsic checks require declared numerical QNM knots and fit rows; agreement does not establish physical accuracy."),
)

_HM_WAVEFORM = "imrphenomhm-independent-waveform/v1"
_HM_COMPOSITION = "imrphenomhm-independent-waveform/v3"
_HM_ORBITAL = "imrphenomhm-orbital-reference-phase/v1"
_HM_SPA = "imrphenomhm-spa-sxs-carrier/v1"
_HM_COMPONENTS = tuple(
    ComponentDefinition(
        id=component_id, name=name, spec_version=_HM_WAVEFORM,
        scope="Six-mode D-carrier composition with printed fits, source PN ratios and declared generated QNM data",
        variant="published_calibration_source_pn",
        implementation="native_and_reference", evidence=("source_transcription", "independent_scalar_composition"),
        source_refs=(SourceRef("london2018", "Eqs. 1–9"),
                     SourceRef("mishra2016", "Eqs. 11–12"),
                     SourceRef("khan2016", "Appendix C Table V, PDF p. 27; amplitude and phase equations"),
                     SourceRef("husa2016", "PDF pp. 10–11, Eqs. 3.6–3.8; HTML Eqs. 22–24"),
                     SourceRef("london2018qnm", "Eqs. 19–21; polynomial form, with independently generated coefficients"),
                     SourceRef("generated_qnm", "Independent Kerr data and six-mode N61 polynomial coefficients"),
                     SourceRef("project_hm", "Fourier, phase/time gauge and positive-frequency mode-pair adapter")),
        reference_module="diffgw_reference.imrphenomhm_waveform",
        caveats=("Generated numerical QNM data are shared; no production fit helper or carrier evaluator is imported by the reference.",),
    )
    for component_id, name in (
        ("hm.waveform.independent_published.v1", "Independent published-calibration polarizations"),
        ("hm.waveform.independent_published_derivatives.v1", "Independent published-calibration smooth derivatives"),
    )
)
_HM_ORBITAL_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"hm.waveform.orbital_reference.{suffix}.v1", name=name,
        spec_version=_HM_ORBITAL,
        scope="Published-calibration six-mode composition with an orbital reference-phase origin",
        variant="published_calibration_source_pn_orbital",
        implementation="native_and_reference",
        evidence=("independent_scalar_composition", "orbital_rotation_identity"),
        source_refs=(SourceRef("london2018", "Eqs. 1, 6–9; constant carrier-phase mapping"),
                     SourceRef("garciaquiros2020", "Eq. 14; orbital rotation law, not the HM API convention"),
                     SourceRef("project_hm", "Orbital reference-phase extension and retained legacy convention")),
        reference_module="diffgw_reference.imrphenomhm_waveform",
        caveats=("Project API convention; no new calibration or physical-accuracy claim. Unchanged carrier components retain their original provenance.",),
    ) for suffix, name in (
        ("polarizations", "Orbital-reference six-mode polarizations"),
        ("derivatives", "Orbital-reference smooth derivatives"),
        ("covariance", "Independent orbital-reference covariance identity"),
    )
)

_HM_SPA_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"hm.waveform.spa_sxs.{suffix}.v1", name=name,
        spec_version=_HM_SPA,
        scope="Published-calibration six-mode composition with the explicit SXS-basis leading-SPA carrier convention",
        variant="published_calibration_source_pn_spa_sxs",
        implementation="native_and_reference",
        evidence=("independent_scalar_composition", "signed_pair_phase_transport"),
        source_refs=(SourceRef("blanchet2008", "Eqs. 8.2–8.3, 9.3, 9.4a,b,d,e,h,i; leading TD coefficient phases"),
                     SourceRef("boyle2019", "Appendix C Eqs. C.21–C.24 and Appendix A.3.1; strain basis and version sign"),
                     SourceRef("garciaquiros2020", "Sec. II.1 Eqs. 2, 4–9; signed-mode projection"),
                     SourceRef("project_hm", "Derived 22-preserving phase transport; explicit project carrier convention")),
        reference_module="diffgw_reference.imrphenomhm_waveform",
        caveats=("A source-derived convention option, not a physical polarization rotation or an NR-accuracy qualification. Frequency-dependent phase and amplitude errors are separate.",),
    ) for suffix, name in (
        ("polarizations", "SPA-convention six-mode polarizations"),
        ("derivatives", "SPA-convention smooth derivatives"),
    )
)

IMRPHENOMHM = WaveformDefinition(
    id="IMRPhenomHM", name="IMRPhenomHM published-calibration D-carrier composition", aliases=(),
    domain="frequency", modes=((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)),
    scope="Bounded independent mathematical composition for published D calibration and named source PN variants; inherited defaults remain separate.",
    reference_version=_HM_COMPOSITION, reference_module="diffgw_reference.imrphenomhm_waveform",
    spec_page="math/imrphenomhm-waveform.md", components=(*_HM_COMPONENTS, *_HM_ORBITAL_COMPONENTS, *_HM_SPA_COMPONENTS),
    sources=(
        *IMRPHENOMD.sources,
        SourceDefinition("london2018", "London et al., IMRPhenomHM", "https://arxiv.org/pdf/1708.00404v4", "1708.00404v4"),
        SourceDefinition("garciaquiros2020", "García-Quirós et al., higher-mode conventions", "https://arxiv.org/html/2001.10914v2", "2001.10914v2"),
        SourceDefinition("blanchet2008", "Blanchet et al., post-Newtonian waveform modes", "https://arxiv.org/pdf/0802.1249v3", "0802.1249v3"),
        SourceDefinition("boyle2019", "Boyle et al., SXS waveform conventions", "https://arxiv.org/pdf/1904.04831v2", "1904.04831v2"),
        SourceDefinition("mishra2016", "Mishra et al., spinning post-Newtonian modes", "https://arxiv.org/html/1601.05588v1", "1601.05588v1"),
        SourceDefinition("london2018qnm", "London and Fauchon-Jones, Kerr QNM representations", "https://arxiv.org/pdf/1810.03550v2", "1810.03550v2"),
        SourceDefinition("generated_qnm", "Independent Kerr and HM N61 numerical assets", "docs/provenance_ledger.md", "repository-versioned", kind="generated_asset",
                         caveats=("Generated data, not a transcription of a published numerical table or historical calibration identity.",)),
        SourceDefinition("project_hm", "DiffGW HM composition conventions", "docs/math/imrphenomhm-waveform.md", _HM_COMPOSITION, kind="project_convention"),
    ),
    parameters=(*_COMMON_PARAMETERS, *_SPINS,
                ParameterDefinition("phi0", "Spin-weighted-harmonic azimuth", "rad", "finite")),
    units=_COMMON_UNITS,
    conventions=(
        ConventionDefinition("fourier", "Positive-frequency complex modes use exp(-i*phase); polarizations sum mode-pair projections."),
        ConventionDefinition("mode_projection", "Positive-frequency mode-pair adapter for h=h_plus-i*h_cross; orbital phase exp(i*m*phic), time exp(-2*pi*i*f*tc)."),
        ConventionDefinition("carrier", "Exact native D carrier for published calibration; XAS and custom carriers are excluded."),
        ConventionDefinition("phase_gauge", "Finite f_ref subtracts D reference phase and peak slope. Legacy retains an extra HM-relative factor; orbital omits it so reference changes are orbital rotations on supported carrier branches. The reference uses positive f_ref or no reference."),
    ),
    options=(
        OptionDefinition("calibration", ("inherited", "published"), "inherited",
                         "Only published is independently composed here. It changes D fit/remnant coefficients consistently, without selecting PN ratios or mapping transitions."),
        OptionDefinition("hm_mapping_transitions", ("phenomd_runtime", "london_2018"), "phenomd_runtime",
                         "Runtime uses amplitude/phase constants 0.014/0.018; London uses printed 0.018/0.014, each scaled by the HM-to-22 ringdown-frequency ratio."),
        OptionDefinition("hm_reference_phase", ("legacy", "orbital"), "legacy",
                         "Legacy preserves the historical extra HM reference factor. Orbital omits it, preserving the native D carrier and (2,2) mode; only the exact native D carrier is supported."),
        OptionDefinition("hm_phase_convention", ("legacy", "spa_sxs_v1"), "legacy",
                         "spa_sxs_v1 multiplies positive-frequency modes by exp[i(m/2-1)3*pi/4] before projection, preserving both m=2 modes. Requires the exact native D carrier; independent of the reference-phase option. No physical-accuracy claim."),
        OptionDefinition("hm_21_pn_variant", ("binary_compatible", "mishra_eq12b"), "binary_compatible",
                         "Only the named Mishra source expression is included in the independent composition."),
        OptionDefinition("hm_32_pn_variant", ("binary_compatible", "mishra_eq12d"), "binary_compatible",
                         "Only the named Mishra source expression is included in the independent composition."),
        OptionDefinition("hm_43_pn_variant", ("binary_compatible", "mishra_eq12g"), "binary_compatible",
                         "Only the named Mishra source expression is included in the independent composition."),
        OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                         "No f_ref uses coalescence; positive f_ref selects reference_frequency unless explicitly overridden. The independent composition follows those two routes."),
    ),
    carrier_parameters=(), capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("paper_precision_source_transcription", "independent_scalar_composition"),
    caveats=(
        "Executable reference and catalog entry do not by themselves qualify a model; scoped comparison claims are recorded only after validation.",
        "Inherited calibration and binary-compatible PN defaults are not qualified by this published-source composition.",
        "The recorded sampled implementation review is complete; this is not uniform-domain or whole-family qualification.",
        "Legacy finite-reference covariance remains defective despite native/reference agreement; the orbital option has separate bounded covariance evidence.",
        "No same-model binary or NR physical-accuracy claim is made for the published variant; previous default-variant outcomes, including red NR cases, remain unchanged.",
        "Shared generated QNM inputs have separate off-grid evidence, not an exact or full-domain physical-accuracy guarantee.",
    ),
    claims=tuple(VerificationClaim(
        component_ids=(component_id,),
        scope=scope, variant="published_calibration_source_pn",
        backends=("torch", "jax", "scalar_reference"),
        options=(("calibration", "published"),
                 ("hm_21_pn_variant", "mishra_eq12b"),
                 ("hm_32_pn_variant", "mishra_eq12d"),
                 ("hm_43_pn_variant", "mishra_eq12g"),
                 ("hm_mapping_transitions", transition),
                 ("hm_reference_phase", "legacy"),
                 ("hm_phase_convention", "legacy"),
                 ("phasing_convention", gauge)),
        domain=domain, precision="float64", device="cpu",
        category="implementation_verification", status="sampled_pass",
        evidence_path="docs/validation/imrphenomhm_published_composition_20260927.md",
    ) for transition in ("london_2018", "phenomd_runtime")
      for component_id, scope, domain, gauge in (
          ("hm.waveform.independent_published.v1",
           "Six individual complex modes, per-mode projection and summed polarizations for two binaries at mapped joins and offsets",
           "(35,20,0.2,-0.1) and (30,30,0.35,-0.2); d=100 Mpc; declared angles/time; no f_ref",
           "coalescence"),
          ("hm.waveform.independent_published.v1",
           "Six individual complex modes, per-mode projection and summed polarizations for two binaries at mapped joins and offsets",
           "(35,20,0.2,-0.1) and (30,30,0.35,-0.2); d=100 Mpc; declared angles/time; f_ref=40 Hz",
           "reference_frequency"),
          ("hm.waveform.independent_published_derivatives.v1",
           "Smooth frequency, inclination, azimuth, orbital phase, time and distance derivatives of six modes and summed polarizations",
           "(35,20,0.2,-0.1); Mf=0.005,0.035,0.12; d=100 Mpc; declared angles/time; f_ref=40 Hz",
           "reference_frequency"),
      )) + tuple(VerificationClaim(
        component_ids=(f"hm.waveform.orbital_reference.{suffix}.v1",),
        scope=scope, variant="published_calibration_source_pn_orbital",
        backends=("torch", "jax", "scalar_reference"),
        options=(("calibration", "published"),
                 ("hm_21_pn_variant", "mishra_eq12b"),
                 ("hm_32_pn_variant", "mishra_eq12d"),
                 ("hm_43_pn_variant", "mishra_eq12g"),
                 ("hm_mapping_transitions", transition),
                 ("hm_reference_phase", "orbital"),
                 ("hm_phase_convention", "legacy")) + (
                     () if suffix == "covariance" else
                     (("phasing_convention", "reference_frequency"),)),
        domain=domain, precision="float64", device="cpu",
        category="implementation_verification", status="sampled_pass",
        evidence_path="docs/validation/imrphenomhm_orbital_reference_20260928.md",
    ) for suffix, scope, domain, transitions in (
        ("polarizations", "Six complex modes and summed polarizations at branch/join samples",
         "(35,20,0.2,-0.1); d=100 Mpc; declared angles/time; f_ref=40 Hz",
         ("london_2018", "phenomd_runtime")),
        ("covariance", "Independent carrier-constant prediction and compensated mode/polarization equality",
         "(35,20,0.2,-0.1) and (30,30,0.35,-0.2); Mf=.005,.025,.06; None/30/40 Hz references; supported carrier branches",
         ("london_2018", "phenomd_runtime")),
        ("derivatives", "Smooth primary-mass and reference-frequency derivatives of six modes and summed polarizations",
         "(35,20,0.2,-0.1); d=100 Mpc; declared angles/time; f=130 Hz; f_ref=40 Hz",
         ("london_2018",)),
    ) for transition in transitions) + tuple(VerificationClaim(
        component_ids=(f"hm.waveform.spa_sxs.{suffix}.v1",),
        scope=scope, variant="published_calibration_source_pn_spa_sxs",
        backends=("torch", "jax", "scalar_reference"),
        options=(("calibration", "published"),
                 ("hm_21_pn_variant", "mishra_eq12b"),
                 ("hm_32_pn_variant", "mishra_eq12d"),
                 ("hm_43_pn_variant", "mishra_eq12g"),
                 ("hm_mapping_transitions", transition),
                 ("hm_reference_phase", reference_phase),
                 ("hm_phase_convention", "spa_sxs_v1"),
                 ("phasing_convention", gauge)),
        domain="(35,20,0.2,-0.1); d=100 Mpc; phi0=.27; phic=.37; tc=.008 s; " + domain,
        precision="float64", device="cpu",
        category="implementation_verification", status="sampled_pass",
        evidence_path="docs/validation/imrphenomhm_spa_native_20260928.md",
    ) for suffix, scope, domain, reference_phase, gauge, transitions in (
        ("polarizations", "Six individual complex modes, per-mode projection and summed polarizations at branch/join samples",
         "iota=.93; no f_ref; mapped joins at their centers and factors 1±1e-6; Mf=.005,.035,.12",
         "legacy", "coalescence", ("london_2018", "phenomd_runtime")),
        ("polarizations", "Six individual complex modes, per-mode projection and summed polarizations at branch/join samples",
         "iota=2.2; f_ref=40 Hz; mapped joins at their centers and factors 1±1e-6; Mf=.005,.035,.12",
         "legacy", "reference_frequency", ("london_2018", "phenomd_runtime")),
        ("polarizations", "Six individual complex modes, per-mode projection and summed polarizations at branch/join samples",
         "iota=2.2; no f_ref; mapped joins at their centers and factors 1±1e-6; Mf=.005,.035,.12",
         "orbital", "coalescence", ("london_2018", "phenomd_runtime")),
        ("polarizations", "Six individual complex modes, per-mode projection and summed polarizations at branch/join samples",
         "iota=.93; f_ref=40 Hz; mapped joins at their centers and factors 1±1e-6; Mf=.005,.035,.12",
         "orbital", "reference_frequency", ("london_2018", "phenomd_runtime")),
        ("derivatives", "Smooth frequency, inclination, azimuth, orbital phase, time and distance derivatives of six modes and summed polarizations",
         "iota=.93; f_ref=40 Hz; Mf=.005,.035,.12; no derivative claim at joins",
         "legacy", "reference_frequency", ("london_2018", "phenomd_runtime")),
        ("derivatives", "Smooth primary-mass and reference-frequency derivatives of six modes and summed polarizations",
         "iota=.93; f=130 Hz; f_ref=40 Hz; no derivative claim at joins",
         "orbital", "reference_frequency", ("london_2018",)),
    ) for transition in transitions),
)

_XAS_CARRIER_PARAMETERS = (
    "f_rd", "f_damp", "width_rd", "lam", "a_r", "f1_col", "f4_col",
    "f_trans_amp", "f_peak", "f_trans_phase", "f_join_rd", "tshift_sec",
    "dt_rd", "dphi_rd", "c_rd_val", "c1_val", "c2_val", "c4_val", "c0_val",
    "rho1", "rho2", "rho3", "c0_amp", "c1_amp", "c2_amp", "c3_amp", "c4_amp",
    "sig0", "sig1", "sig2", "sig3", "sig4",
    "b0_val", "b1_val", "b2_val", "b3_val", "b4_val",
)

IMRPHENOMXAS = WaveformDefinition(
    id="IMRPhenomXAS", name="IMRPhenomXAS aligned-spin dominant harmonic",
    aliases=("PhenomXAS", "XAS"), domain="frequency", modes=((2, 2),),
    scope="Aligned-spin (2,2) carrier composition with supplied fit and QNM outputs.",
    reference_version=_XAS_WAVEFORM, reference_module="diffgw_reference.imrphenomxas_waveform",
    spec_page="math/imrphenomxas.md",
    components=_XAS_COMPONENTS,
    sources=(
        SourceDefinition("pratten2020v1", "Pratten et al., IMRPhenomXAS", "https://arxiv.org/html/2001.11412v1", "2001.11412v1"),
        SourceDefinition("author_fit_asset", "IMRPhenomXAS parameter-space fits, APS Mathematica supplement",
                         "https://journals.aps.org/prd/supplemental/10.1103/PhysRevD.102.064001/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m",
                         "APS public release candidate", kind="external_asset", status="opt_in_not_redistributed",
                         sha256="94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52",
                         caveats=("Redistribution terms unresolved; sampled verifier uses local user-supplied asset.",
                                  "Carrier midpoint fit uses sTotR while deposited function uses chiPNhat.")),
        SourceDefinition("project", "DiffGW XAS carrier conventions", "docs/math/imrphenomxas.md", _XAS_WAVEFORM, kind="project_convention"),
    ),
    parameters=(*_COMMON_PARAMETERS, *_SPINS), units=_COMMON_UNITS,
    conventions=(*_COMMON_CONVENTIONS,
        ConventionDefinition("dimensionless_frequency", "Intrinsic branches use u=M_seconds*f_Hz."),
        ConventionDefinition("amplitude_normalization", "Native intermediate quartic uses K/P; the printed paper alternative is K*u^(-7/6)/P."),
        ConventionDefinition("intermediate_phase_sign", "Native carrier uses the declared Lorentzian sign opposite the printed Eq. 90 form."),
        ConventionDefinition("phase_origin", "Carrier phase includes optional reference subtraction, time shift and +3*pi/4 offset."),
    ),
    options=(
        OptionDefinition("intermediate_amplitude", ("carrier_quartic", "paper_quartic", "paper_fifth_order"), "carrier_quartic",
                         "Only carrier_quartic is used by the current native Torch/JAX carriers; other forms are reference-only alternatives.",
                         native_choices=("carrier_quartic",)),
        OptionDefinition("intermediate_phase", ("carrier_sign", "printed_sign"), "carrier_sign",
                         "The printed and carrier Lorentzian signs are separate mathematical variants.",
                         native_choices=("carrier_sign",)),
        OptionDefinition("branch_joins", ("lower_branch_at_join",), "lower_branch_at_join",
                         "Both join frequencies belong to the lower-frequency branch; derivatives are smooth-branch claims only."),
    ),
    carrier_parameters=_XAS_CARRIER_PARAMETERS,
    capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("source_transcription", "conditional_sampled_torch_jax_comparison",
              "opt_in_author_fit_asset_comparison"),
    caveats=("The scalar composition takes named calibrated fits, QNM values and joins as inputs; it does not independently derive the full parameter-space calibration.",
             "The printed quartic normalization, printed phase sign and fifth-order amplitude are not native carrier options.",
             "A sampled external asset comparison is not physical validation or a uniform error bound."),
    fit_asset_ids=("author_fit_asset",),
)

_KERR_DIRECT_VERSION = "kerr-ringdown-220-energy-direct-class/v1"
_KERR_DIRECT_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"kerr.direct220.{name}.v1", name=name.replace("_", " ").title(),
        spec_version=_KERR_DIRECT_VERSION,
        scope="Direct KerrRingdown class, one physical-energy spheroidal (2,2,0) mode",
        variant="independent_qnm_spectral_angular_native_pair",
        implementation="native_and_reference",
        evidence=("source_transcription", "independent_scalar_whole_waveform_sample"),
        source_refs=_source(source, location),
        reference_module="diffgw_reference.kerr_ringdown",
    )
    for name, source, location in (
        ("qnm", "cook2014", "Eqs. 23–43; independently generated QNM assets"),
        ("angular", "cook2014", "Eqs. 4, 48–57; spectral projection"),
        ("energy_amplitude", "ruiz2007", "Eq. 2.8; finite-Q energy integral"),
        ("causal_td_fd", "project_kerr", "Direct-class carrier, mirror and Fourier-minus convention"),
    )
)

KERR_RINGDOWN_220_DIRECT = WaveformDefinition(
    id="KerrRingdown220Direct", name="KerrRingdown direct-class physical-energy 220",
    aliases=(), domain="time_frequency", modes=((2, 2),),
    scope="Direct Torch/JAX KerrRingdown class only: one physical-energy spheroidal (2,2,0) mode with independent QNM; selected FP64 inputs.",
    reference_version=_KERR_DIRECT_VERSION,
    reference_module="diffgw_reference.kerr_ringdown",
    spec_page="math/kerr-ringdown-direct-class.md", components=_KERR_DIRECT_COMPONENTS,
    sources=(
        SourceDefinition("cook2014", "Cook and Zalutskiy, Kerr QNM eigenproblem and angular harmonics", "https://arxiv.org/html/1410.7698v2", "1410.7698v2"),
        SourceDefinition("ruiz2007", "Ruiz et al., mode energy flux", "https://arxiv.org/pdf/0707.4654v3", "0707.4654v3"),
        SourceDefinition("fh1997", "Flanagan and Hughes, angular norm", "https://arxiv.org/abs/gr-qc/9701039", "gr-qc/9701039v2"),
        SourceDefinition("torch_qnm_asset", "Project-generated Torch 220 QNM spline", "src/torchwave/approximants/kerr_qnm_generated.py", "8f894d9", kind="project_generated_asset", sha256="e69d64e615dfd7bf054c2c7a9b721509688a1f126a482c7d9d0047f53b370dd3"),
        SourceDefinition("jax_qnm_asset", "Project-generated JAX 220 QNM spline", "src/diffgw_core/kerr_qnm_220_data.py", "8f894d9", kind="project_generated_asset", sha256="51b13ae3649ec332b45d4c538a669334ec31f4244105e3b4c2b0c24197f4c5e4"),
        SourceDefinition("project_kerr", "DiffGW direct-class pair and Fourier convention", "docs/math/kerr-ringdown-direct-class.md", _KERR_DIRECT_VERSION, kind="project_convention"),
    ),
    parameters=(
        ParameterDefinition("final_mass", "Remnant mass", "nominal solar mass", "positive"),
        ParameterDefinition("final_spin", "Dimensionless Kerr spin", "1", "selected 0.35 and 0.7"),
        ParameterDefinition("distance", "Luminosity distance", "Mpc", "positive"),
        ParameterDefinition("energy_fraction", "Energy in one mode", "1", "positive"),
        ParameterDefinition("inclination", "Viewing inclination", "rad", "selected off-axis values"),
        ParameterDefinition("phi", "Observer azimuth", "rad", "finite"),
        ParameterDefinition("t0", "Causal onset", "s", "finite"),
    ),
    units=(
        UnitDefinition("remnant mass", "s", "M_s=final_mass*MSUN_SEC"),
        UnitDefinition("distance", "s", "D_s=distance*MPC_SI/C_SI"),
        UnitDefinition("frequency", "Hz", "angular frequency is 2*pi*f"),
    ),
    conventions=(
        ConventionDefinition("qnm", "Independent generated 220 spline, spin weight -2, positive real frequency and damping rate."),
        ConventionDefinition("angular", "Unit-norm spheroidal spectral branch and equatorial mirror at complex c=a*M*omega."),
        ConventionDefinition("pair_amplitude", "Native pair coefficient is sqrt(2) times the single-mode finite-Q energy coefficient."),
        ConventionDefinition("fourier", "Causal H(0)=1 waveform; Fourier kernel exp(-2*pi*i*f*t)."),
    ),
    options=(
        OptionDefinition("call_route", ("direct_class",), "direct_class", "Only KerrRingdown.time_domain/frequency_domain class methods are covered."),
        OptionDefinition("energy_mode", ("(2,2,0)",), "(2,2,0)", "Exactly one fundamental 220 mode; amplitudes is omitted."),
        OptionDefinition("angular_basis", ("spheroidal",), "spheroidal", "Source-backed spectral projection."),
        OptionDefinition("qnm_backend", ("independent",), "independent", "Generated project QNM asset, checked against scalar equation solves."),
    ),
    carrier_parameters=(), capabilities=("td", "fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("independent_scalar_whole_waveform_sample", "selected_extrinsic_gradients"),
    caveats=(
        "This does not cover the KerrRingdown dispatcher, other class options or modes, or the opt-in public-clean evaluator.",
        "The existing LAL binary holdout exercises the dispatcher/public-clean route, not this direct class.",
        "JAX spin/oblateness derivatives are evaluated natively via pure differentiable inverse-iteration without pure_callback.",
        "Sampled scalar-to-native implementation agreement does not establish binary compatibility, a uniform domain, or physical accuracy; the direct-class public-binary diagnostic has a material residual.",
    ),
    claims=(VerificationClaim(
        component_ids=tuple(component.id for component in _KERR_DIRECT_COMPONENTS),
        scope="Two frozen moderate-spin off-axis inputs; both polarizations before/at/after onset and signed frequencies",
        variant="independent_qnm_spectral_angular_native_pair", backends=("torch", "jax", "scalar_reference"),
        options=(("call_route", "direct_class"), ("energy_mode", "(2,2,0)"),
                 ("angular_basis", "spheroidal"), ("qnm_backend", "independent")),
        domain="sampled time and signed frequency", precision="float64", device="cpu",
        category="implementation_verification", status="sampled_pass",
        evidence_path="tests/test_kerr_ringdown_class_math_reference.py",
    ),),
)

_CW_DIRECT_VERSION = "cw-monochromatic-h1-direct-class/v2"
_CW_DIRECT_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"cw.direct_h1.{name}.v1", name=name.replace("_", " ").title(),
        spec_version="cw-monochromatic-h1-direct-class/v1",
        scope="Monochromatic principal-axis 2f direct ContinuousWave class at H1",
        variant="jks_2f_with_declared_circular_orbit_and_utc_clock",
        implementation="native_and_reference",
        evidence=("source_transcription", "independent_scalar_whole_waveform_sample"),
        source_refs=_source(source, location) + (
            _source("iers_c49", "2015-07 leap offset")
            + _source("iers_eopc04_2015", "MJD 57279 UT1-UTC diagnostic")
            if name == "clock" else ()
        ),
        reference_module="diffgw_reference.continuous_wave_direct_class",
    )
    for name, source, location in (
        ("amplitude", "jks1998", "PDF Eqs. 23–24"),
        ("clock", "aoki1982", "Eqs. 13, 19; UTC project approximation"),
        ("delay_phase", "jks1998", "PDF Eqs. 14, 18; circular-orbit project epoch"),
        ("antenna", "jks1998", "PDF Eqs. 1–6; supplied H1 arms"),
        ("strain", "jks1998", "PDF Eqs. 19–22, 32–36"),
    )
) + (ComponentDefinition(
    id="cw.direct_h1.site.v1", name="H1 surveyed site",
    spec_version="cw-h1-site-survey/v1",
    scope="H1 Earth-fixed and geodetic vertex plus arm directions at LIGO survey precision",
    variant="ligo_t980044_10_table_4",
    implementation="native_and_reference",
    evidence=("source_transcription", "survey_precision_comparison"),
    source_refs=_source("ligo_t980044_10", "Section 2.1 and Table 4, WGS-84 conversion, Hanford vertex and arm direction cosines"),
    reference_module="diffgw_reference.continuous_wave_direct_class",
    caveats=("Published vertex rounds to 0.1 m; arm direction error is estimated at 1.93 microrad. Full native digits remain unverified.",),
),)

CONTINUOUS_WAVE_H1_DIRECT = WaveformDefinition(
    id="ContinuousWaveH1Direct", name="ContinuousWave monochromatic H1 direct class",
    aliases=(), domain="time", modes=((2, 2),),
    scope="Direct Torch/JAX ContinuousWave class: selected float64 CPU monochromatic principal-axis 2f strain at H1, GPS epoch in 2015–2016.",
    reference_version=_CW_DIRECT_VERSION,
    reference_module="diffgw_reference.continuous_wave_direct_class",
    spec_page="math/continuous-wave-direct-class.md", components=_CW_DIRECT_COMPONENTS,
    sources=(
        SourceDefinition("jks1998", "Jaranowski, Królak and Schutz, spinning-neutron-star signal", "https://arxiv.org/pdf/gr-qc/9804014v1", "gr-qc/9804014v1"),
        SourceDefinition("anderson2001", "Anderson et al., corrected H1 arm directions", "https://arxiv.org/pdf/gr-qc/0008066v1", "gr-qc/0008066v1"),
        SourceDefinition("aoki1982", "Aoki et al., IAU 1982 GMST definition", "https://ui.adsabs.harvard.edu/abs/1982A%26A...105..359A/abstract", "1982 A&A 105 359"),
        SourceDefinition("iers_c49", "IERS Bulletin C49, 2015-07 leap second", "https://datacenter.iers.org/data/html/bulletinc-049.html", "C49"),
        SourceDefinition("ligo_t980044_10", "LIGO as-built site coordinate survey", "https://dcc.ligo.org/public/0002/T980044/000/T980044-10.pdf", "LIGO-T980044-10"),
        SourceDefinition("iers_eopc04_2015", "IERS EOP 14 C04 2015, IAU1980", "https://datacenter.iers.org/data/html/eopc04_14.15.html", "MJD 57279"),
        SourceDefinition("project_cw", "Declared circular orbit and UTC clock", "docs/math/continuous-wave-direct-class.md", _CW_DIRECT_VERSION, kind="project_convention"),
    ),
    parameters=(
        ParameterDefinition("t", "Sample GPS times", "s", "2015-07 to 2016-12; selected offsets"),
        ParameterDefinition("t0", "Frequency and circular-orbit reference GPS epoch", "s", "1126259462 in sampled gate"),
        ParameterDefinition("ra", "Right ascension", "rad", "selected sky points"),
        ParameterDefinition("dec", "Declination", "rad", "selected sky points"),
        ParameterDefinition("f0", "Gravitational-wave frequency, twice rotation frequency", "Hz", "positive; selected 30 and 71"),
        ParameterDefinition("h0", "Characteristic strain, alternative to ellipticity and distance", "1", "positive"),
        ParameterDefinition("ellipticity", "Principal-axis equatorial ellipticity", "1", "selected 1e-5"),
        ParameterDefinition("distance", "Source distance when ellipticity is supplied", "m", "positive"),
        ParameterDefinition("cos_iota", "Cosine of source inclination", "1", "selected interior values"),
        ParameterDefinition("psi", "Polarization angle", "rad", "selected angles"),
        ParameterDefinition("phi0", "Initial GW phase", "rad", "selected angles"),
    ),
    units=(
        UnitDefinition("time", "GPS s", "t and t0 are GPS seconds"),
        UnitDefinition("distance", "m", "ellipticity amplitude uses SI distance and Izz in kg m²"),
        UnitDefinition("strain", "1", "detector projection of dimensionless 2f polarizations"),
    ),
    conventions=(
        ConventionDefinition("orbit", "Circular 1-AU orbit in the ecliptic, zero orbital phase at t0; sidereal year 365.256363004 days."),
        ConventionDefinition("clock", "Aoki IAU 1982 GMST with UTC substituted for UT1 and 17-second GPS-UTC offset in the selected epoch."),
        ConventionDefinition("site", "Native H1 vertex and normalized arms are conditioned inputs for tight scalar checks; LIGO T980044-10 Table 4 separately checks survey precision, not inherited full digits."),
        ConventionDefinition("phase", "JKS first-order Roemer phase at f1=f2=0; phi0 is GW phase and f0 is GW frequency."),
    ),
    options=(
        OptionDefinition("call_route", ("direct_class",), "direct_class", "Only ContinuousWave class calls are covered."),
        OptionDefinition("detector", ("H1",), "H1", "Only the H1 site is covered."),
        OptionDefinition("spindown", ("f1=f2=0",), "f1=f2=0", "Only monochromatic input is covered."),
        OptionDefinition("amplitude_input", ("h0", "ellipticity_distance"), "h0", "Both amplitude input routes are sampled."),
    ),
    carrier_parameters=(), capabilities=("td", "detector_strain", "torch", "jax", "scalar_reference"),
    evidence=("independent_scalar_whole_waveform_sample", "selected_frequency_and_phase_gradients", "survey_precision_site_and_clock_diagnostic"),
    caveats=(
        "The tight scalar comparison conditions on native H1 full-precision digits; the separate LIGO survey gate supports only printed position and estimated angular precision.",
        "UTC replaces UT1 without Earth-orientation corrections; IERS C04 reports UT1-UTC = +0.2553711 s at 2015-09-14 0h UTC. The circular orbit is not a Solar System ephemeris.",
        "Native nonzero f1/f2 includes higher Roemer-delay powers absent from JKS Eq. 14 and is outside this claim.",
        "No external binary, phase-coherent astrophysical, uniform-domain, or physical-validation claim is made.",
    ),
    claims=(VerificationClaim(
        component_ids=tuple(component.id for component in _CW_DIRECT_COMPONENTS),
        scope="Two sky/frequency/inclination cases, five GPS samples, both amplitude routes, selected f0 and phi0 gradients; separate H1 survey-precision site gate",
        variant="jks_2f_with_declared_circular_orbit_and_utc_clock",
        backends=("torch", "jax", "scalar_reference"),
        options=(("call_route", "direct_class"), ("detector", "H1"), ("spindown", "f1=f2=0")),
        domain="selected 2015 GPS samples", precision="float64", device="cpu",
        category="implementation_verification", status="sampled_pass",
        evidence_path="tests/test_continuous_wave_direct_class_math_reference.py",
    ), VerificationClaim(
        component_ids=("cw.direct_h1.site.v1",),
        scope="H1 vertex at published 0.1 m grid and independently converted geodetic point; arms within 1.93 microrad survey angular error",
        variant="ligo_t980044_10_table_4", backends=("torch", "jax", "scalar_reference"),
        options=(("detector", "H1"),), domain="static Earth-fixed site",
        precision="float64", device="cpu", category="source_fidelity",
        status="sampled_pass", evidence_path="tests/test_continuous_wave_direct_class_math_reference.py",
    )),
)

_MEMORY_22_VERSION = "memory-dominant-22-direct-class/v1"
_MEMORY_22_COMPONENTS = (
    ComponentDefinition(
        id="memory.dominant_22.v1", name="Dominant symmetric pair and full angular pattern",
        spec_version=_MEMORY_22_VERSION,
        scope="Supplied physical-strain h22 and equatorial-symmetry mirror, relative memory offset",
        variant="published_dominant_pair", implementation="native_and_reference",
        evidence=("source_transcription", "independent_scalar_sampled_comparison"),
        source_refs=(SourceRef("favata2010", "Sec. 6.2, Eq. 21"),
                     SourceRef("talbot2018", "Eqs. 1–9")),
        reference_module="diffgw_reference.memory_dominant_22",
        caveats=("Only the symmetric (2,±2) source pair; h20 is returned but the full plus pattern also contains l=4,m=0.",),
    ),
    ComponentDefinition(
        id="memory.sampled_flux.v1", name="Nonuniform sampled flux integral",
        spec_version=_MEMORY_22_VERSION,
        scope="First-order unequal-grid derivative and adjacent trapezoids, zero offset at first sample",
        variant="project_sampled_adapter", implementation="native_and_reference",
        evidence=("declared_numerical_rule", "independent_scalar_sampled_comparison"),
        source_refs=_source("project_memory", "Sampling rule in direct-class specification"),
        reference_module="diffgw_reference.memory_dominant_22",
    ),
)

MEMORY_22_DIRECT = WaveformDefinition(
    id="MemoryWaveform22Direct", name="MemoryWaveform direct-class dominant-pair memory",
    aliases=(), domain="time", modes=((2, 0), (4, 0)),
    scope="Only Torch/JAX MemoryWaveform.evaluate_22 with supplied physical-strain h22, symmetric mirror, and relative memory offset on selected CPU float64 grids.",
    reference_version=_MEMORY_22_VERSION,
    reference_module="diffgw_reference.memory_dominant_22",
    spec_page="math/memory-dominant-22-direct-class.md", components=_MEMORY_22_COMPONENTS,
    sources=(
        SourceDefinition("favata2010", "Favata, nonlinear gravitational-wave memory", "https://arxiv.org/pdf/1003.3486v1", "1003.3486v1"),
        SourceDefinition("talbot2018", "Talbot et al., gravitational-wave memory from compact binaries", "https://arxiv.org/pdf/1807.00990v2", "1807.00990v2"),
        SourceDefinition("project_memory", "DiffGW direct-class sampling convention", "docs/math/memory-dominant-22-direct-class.md", _MEMORY_22_VERSION, kind="project_convention"),
    ),
    parameters=(
        ParameterDefinition("times", "Strictly increasing time samples", "s", "one-dimensional, at least two"),
        ParameterDefinition("h22", "Complex physical-strain oscillatory mode at distance_m", "strain", "complex samples matching times"),
        ParameterDefinition("distance_m", "Distance of supplied mode", "m", "> 0"),
        ParameterDefinition("inclination", "Orbital inclination", "rad", "selected 0, 0.4, pi/2, 2.1, 0.73"),
    ),
    units=(
        UnitDefinition("time", "s", "SI seconds"),
        UnitDefinition("mode", "strain", "physical h22 at distance_m, not a distance-scaled mode"),
        UnitDefinition("distance", "m", "R/c restores SI seconds in the Favata integral"),
    ),
    conventions=(
        ConventionDefinition("mode", "h=h_plus-i*h_cross=sum h_lm * spin-minus-two Y_lm; h2,-2=conj(h22)."),
        ConventionDefinition("offset", "Memory equals zero at the first supplied sample; no prior-history estimate."),
        ConventionDefinition("polarization", "Full symmetric dominant-pair plus pattern includes l=4,m=0; cross is zero."),
    ),
    options=(
        OptionDefinition("call_route", ("evaluate_22",), "evaluate_22", "Only the explicit direct-class method is covered."),
        OptionDefinition("mirror", ("equatorial_symmetry",), "equatorial_symmetry", "The negative-m mode is the conjugate mirror."),
        OptionDefinition("grid_rule", ("gradient_trapezoid",), "gradient_trapezoid", "Endpoint secants, quadratic-interpolant interior derivative, adjacent trapezoids."),
    ),
    carrier_parameters=(), capabilities=("td", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("published_pattern_projection", "independent_scalar_sampled_comparison", "selected_parameter_gradients"),
    caveats=(
        "The legacy MemoryWaveform.forward/__call__ polarization-only diagnostic carrier is excluded.",
        "Supplied h22 physical accuracy, missing earlier offset, arbitrary-mode coupling, precession, and nonlinear iteration are unqualified.",
        "No LAL same-model or numerical-relativity comparison, continuous-domain error bound, or physical validation is claimed.",
    ),
    claims=(VerificationClaim(
        component_ids=tuple(component.id for component in _MEMORY_22_COMPONENTS),
        scope="Two- and four-sample linear inputs, four inclinations, and one nonlinear complex polynomial sample; selected slope, quadratic-coefficient, and inclination derivatives",
        variant="published_dominant_pair_with_project_sampled_adapter",
        backends=("torch", "jax", "scalar_reference"),
        options=(("call_route", "evaluate_22"), ("mirror", "equatorial_symmetry"),
                 ("grid_rule", "gradient_trapezoid")),
        domain="sampled time", precision="float64", device="cpu",
        category="implementation_verification", status="sampled_pass",
        evidence_path="tests/test_memory_dominant_22_math_reference.py",
    ),),
)

_TF2_ALIGNED_SPIN_VERSION = "taylorf2-aligned-bh-spin-3.5pn/v1"
_TF2_ALIGNED_SPIN_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"tf2.bhspin.{name}.v1",
        name=name.replace("_", " ").title(),
        spec_version=_TF2_ALIGNED_SPIN_VERSION,
        scope="aligned black-hole spin 3.5PN" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else "aligned black-hole spin phase correction",
        variant="project_adapter" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else "canonical_reference",
        implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source(
            "project" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else
            "pratten2020" if name == "SS7" else
            "mishra2016" if name in ("SO6", "SS6", "SO7", "SSS7") else "wade2013",
            "Aligned-spin TaylorF2 composition" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else
            "Eqs. A19, A24" if name == "SS7" else
            "Eqs. 5, 6a, 7" if name in ("SO6", "SS6") else
            "Eqs. 5, 6b, 7" if name in ("SO7", "SSS7") else "Eq. 4",
        ),
        reference_module="diffgw_reference.taylorf2_aligned_spin",
    )
    for name in (
        "SO3", "SS4", "SO5", "SO5_log", "SO6", "SS6", "SO7", "SS7", "SSS7",
        "intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations",
    )
)
TF2_ALIGNED_SPIN_COMPONENTS = _TF2_ALIGNED_SPIN_COMPONENTS

TAYLORF2_ALIGNED_SPIN = WaveformDefinition(
    id="taylorf2-aligned-bh-spin-3.5pn/v1",
    name="TaylorF2 aligned black-hole spin 3.5PN",
    aliases=("TaylorF2AlignedSpin",),
    domain="frequency",
    modes=((2, 2),),
    scope="Aligned black-hole spin phase corrections through 3.5PN conservative dynamics and flux to infinity with restricted quadrupole amplitude.",
    reference_version=_TF2_ALIGNED_SPIN_VERSION,
    reference_module="diffgw_reference.taylorf2_aligned_spin",
    spec_page="math/taylorf2-aligned-spin.md",
    components=(*_TF2_COMPONENTS, *_TF2_ALIGNED_SPIN_COMPONENTS),
    sources=(
        SourceDefinition("buonanno2009", "Buonanno et al., nonspinning PN phase", "https://arxiv.org/pdf/0907.0700v1", "0907.0700v1"),
        SourceDefinition("arun2008", "Arun et al., restricted amplitude and polarizations", "https://arxiv.org/pdf/0810.5336v1", "0810.5336v1"),
        SourceDefinition("wade2013", "Wade et al., aligned-spin 1.5–2.5PN phase", "https://arxiv.org/pdf/1306.3901v2", "1306.3901v2"),
        SourceDefinition("mishra2016", "Mishra et al., aligned-spin 3–3.5PN phase polynomials", "https://arxiv.org/pdf/1601.05588v1", "1601.05588v1"),
        SourceDefinition("bohe2015", "Bohé et al., spin-spin energy and flux", "https://arxiv.org/pdf/1501.01529v1", "1501.01529v1"),
        SourceDefinition("messina2017", "Messina and Nagar, 3.5PN spin-spin tail flux", "https://arxiv.org/pdf/1703.08107v4", "1703.08107v4"),
        SourceDefinition("pratten2020", "Pratten et al., aligned-spin 3.5PN tail phase", "https://arxiv.org/pdf/2001.11412v1", "2001.11412v1"),
        SourceDefinition("project", "DiffGW aligned-spin TaylorF2 composition", "docs/math/taylorf2-aligned-spin.md", _TF2_ALIGNED_SPIN_VERSION, kind="project_convention"),
    ),
    parameters=(*_COMMON_PARAMETERS, *_SPINS),
    units=_COMMON_UNITS,
    conventions=(
        *_COMMON_CONVENTIONS,
        ConventionDefinition("spin_phasing", "Aligned black-hole spin corrections through 3.5PN; horizon absorption excluded."),
        ConventionDefinition("phase_gauge", "Spin 2.5PN logarithm uses Wade v_ref=1 gauge log(v); baseline uses log(v)."),
    ),
    options=(
        OptionDefinition("spin_order", ("-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"), "7",
                         "Integer selectors use the aligned-spin option map. Public waveform default is 7; 8 selects all terms including SS7 and SSS7."),
        OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                         "Phase convention matching baseline TaylorF2."),
        OptionDefinition("phenom_d_calibration", ("false", "true"), "false",
                         "Suppresses SS6, SS7, and SSS7 at any selector for compatibility with legacy calibration recipes."),
    ),
    carrier_parameters=(),
    capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("source_transcription", "sampled_native_comparison"),
    caveats=(
        "Horizon absorption at 2.5PN and above is excluded.",
        "Does not include precession, eccentricity, or spin-dependent amplitude corrections.",
    ),
    claims=(),
)

_TF2_TIDAL_VERSION = "taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1"
_TF2_TIDAL_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"tf2.tidal65.{name}.v1",
        name=name.replace("_", " ").title(),
        spec_version=_TF2_TIDAL_VERSION,
        scope="adiabatic quadrupolar tides 6.5PN" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else "adiabatic quadrupolar tidal phase correction",
        variant="project_adapter" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else "canonical_reference",
        implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source(
            "project" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else
            "damour2012" if name == "T13" else "wade2014",
            "Adiabatic quadrupolar tidal TaylorF2 composition" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else
            "Eqs. 6, B8–B12" if name == "T13" else
            "Eq. 5" if name == "lambda_tilde" else
            "Eq. 6" if name == "delta_lambda_tilde" else "Eq. A26",
        ),
        reference_module="diffgw_reference.taylorf2_tidal",
    )
    for name in (
        "T10", "T12", "T13", "lambda_tilde", "delta_lambda_tilde",
        "intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations",
    )
)
TF2_TIDAL_COMPONENTS = _TF2_TIDAL_COMPONENTS

TAYLORF2_TIDAL = WaveformDefinition(
    id="taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1",
    name="TaylorF2 adiabatic quadrupolar tides 6.5PN",
    aliases=("TaylorF2Tidal",),
    domain="frequency",
    modes=((2, 2),),
    scope="Adiabatic quadrupolar gravitoelectric TaylorF2 tidal phase corrections through 6.5PN added to aligned-spin TaylorF2.",
    reference_version=_TF2_TIDAL_VERSION,
    reference_module="diffgw_reference.taylorf2_tidal",
    spec_page="math/taylorf2-tidal.md",
    components=(*_TF2_COMPONENTS, *_TF2_ALIGNED_SPIN_COMPONENTS, *_TF2_TIDAL_COMPONENTS),
    sources=(
        SourceDefinition("buonanno2009", "Buonanno et al., nonspinning PN phase", "https://arxiv.org/pdf/0907.0700v1", "0907.0700v1"),
        SourceDefinition("arun2008", "Arun et al., restricted amplitude and polarizations", "https://arxiv.org/pdf/0810.5336v1", "0810.5336v1"),
        SourceDefinition("wade2013", "Wade et al., aligned-spin 1.5–2.5PN phase", "https://arxiv.org/pdf/1306.3901v2", "1306.3901v2"),
        SourceDefinition("mishra2016", "Mishra et al., aligned-spin 3–3.5PN phase polynomials", "https://arxiv.org/pdf/1601.05588v1", "1601.05588v1"),
        SourceDefinition("bohe2015", "Bohé et al., spin-spin energy and flux", "https://arxiv.org/pdf/1501.01529v1", "1501.01529v1"),
        SourceDefinition("messina2017", "Messina and Nagar, 3.5PN spin-spin tail flux", "https://arxiv.org/pdf/1703.08107v4", "1703.08107v4"),
        SourceDefinition("pratten2020", "Pratten et al., aligned-spin 3.5PN tail phase", "https://arxiv.org/pdf/2001.11412v1", "2001.11412v1"),
        SourceDefinition("wade2014", "Wade et al., adiabatic tidal 5PN/6PN phase", "https://arxiv.org/pdf/1402.5156v1", "1402.5156v1"),
        SourceDefinition("damour2012", "Damour et al., adiabatic tidal 6.5PN tail factor", "https://arxiv.org/pdf/1203.4352v1", "1203.4352v1"),
        SourceDefinition("project", "DiffGW adiabatic tidal TaylorF2 composition", "docs/math/taylorf2-tidal.md", _TF2_TIDAL_VERSION, kind="project_convention"),
    ),
    parameters=(
        *_COMMON_PARAMETERS, *_SPINS,
        ParameterDefinition("lambda1", "Dimensionless tidal deformability of body 1", "dimensionless", ">= 0"),
        ParameterDefinition("lambda2", "Dimensionless tidal deformability of body 2", "dimensionless", ">= 0"),
    ),
    units=_COMMON_UNITS,
    conventions=(
        *_COMMON_CONVENTIONS,
        ConventionDefinition("tidal_phasing", "Linear adiabatic electric quadrupole tides through 6.5PN tail; magnetic and octupolar tides excluded."),
    ),
    options=(
        OptionDefinition("spin_order", ("-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"), "7",
                         "Inherited aligned-spin option map."),
        OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                         "Phase convention matching baseline TaylorF2."),
        OptionDefinition("phenom_d_calibration", ("false", "true"), "false",
                         "Suppresses higher spin terms for legacy compatibility."),
    ),
    carrier_parameters=(),
    capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("source_transcription", "sampled_native_comparison"),
    caveats=("Excludes dynamical tides, magnetic tides, octupolar tides, and tidal amplitude corrections.",),
    claims=(),
)

_TF2_SPIN_QUAD_VERSION = "taylorf2-spin-induced-quadrupole-3.5pn/v1"
_TF2_SPIN_QUAD_COMPONENTS = tuple(
    ComponentDefinition(
        id=f"tf2.spinq.{name}.v1",
        name=name.replace("_", " ").title(),
        spec_version=_TF2_SPIN_QUAD_VERSION,
        scope="spin-induced quadrupole 3.5PN" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else "spin-induced quadrupole phase correction",
        variant="project_adapter" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else "canonical_reference",
        implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source(
            "project" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else
            "dietrich2019" if name == "Q7_cubic" else "nagar2018",
            "Spin-induced quadrupole TaylorF2 composition" if name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations") else
            "Eqs. 26, 27" if name == "Q7_cubic" else
            "Eqs. 2, 38, 44, 50" if name == "Q4" else
            "Eqs. 2, 38, 46, 51" if name == "Q6" else "Eqs. 2, 38, 47, 52",
        ),
        reference_module="diffgw_reference.taylorf2_spin_quadrupole",
    )
    for name in (
        "Q4", "Q6", "Q7_tail", "Q7_cubic",
        "intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations",
    )
)
TF2_SPIN_QUAD_COMPONENTS = _TF2_SPIN_QUAD_COMPONENTS

TAYLORF2_SPIN_QUADRUPOLE = WaveformDefinition(
    id="taylorf2-spin-induced-quadrupole-3.5pn/v1",
    name="TaylorF2 spin-induced quadrupole 3.5PN",
    aliases=("TaylorF2SpinQuadrupole",),
    domain="frequency",
    modes=((2, 2),),
    scope="Spin-induced quadrupole moment phase corrections through 3.5PN added to aligned-spin and tidal TaylorF2.",
    reference_version=_TF2_SPIN_QUAD_VERSION,
    reference_module="diffgw_reference.taylorf2_spin_quadrupole",
    spec_page="math/taylorf2-spin-quadrupole.md",
    components=(*_TF2_COMPONENTS, *_TF2_ALIGNED_SPIN_COMPONENTS, *_TF2_TIDAL_COMPONENTS, *_TF2_SPIN_QUAD_COMPONENTS),
    sources=(
        SourceDefinition("buonanno2009", "Buonanno et al., nonspinning PN phase", "https://arxiv.org/pdf/0907.0700v1", "0907.0700v1"),
        SourceDefinition("arun2008", "Arun et al., restricted amplitude and polarizations", "https://arxiv.org/pdf/0810.5336v1", "0810.5336v1"),
        SourceDefinition("wade2013", "Wade et al., aligned-spin 1.5–2.5PN phase", "https://arxiv.org/pdf/1306.3901v2", "1306.3901v2"),
        SourceDefinition("mishra2016", "Mishra et al., aligned-spin 3–3.5PN phase polynomials", "https://arxiv.org/pdf/1601.05588v1", "1601.05588v1"),
        SourceDefinition("bohe2015", "Bohé et al., spin-spin energy and flux", "https://arxiv.org/pdf/1501.01529v1", "1501.01529v1"),
        SourceDefinition("messina2017", "Messina and Nagar, 3.5PN spin-spin tail flux", "https://arxiv.org/pdf/1703.08107v4", "1703.08107v4"),
        SourceDefinition("pratten2020", "Pratten et al., aligned-spin 3.5PN tail phase", "https://arxiv.org/pdf/2001.11412v1", "2001.11412v1"),
        SourceDefinition("wade2014", "Wade et al., adiabatic tidal 5PN/6PN phase", "https://arxiv.org/pdf/1402.5156v1", "1402.5156v1"),
        SourceDefinition("damour2012", "Damour et al., adiabatic tidal 6.5PN tail factor", "https://arxiv.org/pdf/1203.4352v1", "1203.4352v1"),
        SourceDefinition("nagar2018", "Nagar et al., spin-induced quadrupole phase corrections", "https://arxiv.org/pdf/1812.07923v1", "1812.07923v1"),
        SourceDefinition("dietrich2019", "Dietrich et al., spin-induced quadrupole cubic phase", "https://arxiv.org/pdf/1905.06011v2", "1905.06011v2"),
        SourceDefinition("project", "DiffGW spin-induced quadrupole composition", "docs/math/taylorf2-spin-quadrupole.md", _TF2_SPIN_QUAD_VERSION, kind="project_convention"),
    ),
    parameters=(
        *_COMMON_PARAMETERS, *_SPINS,
        ParameterDefinition("lambda1", "Dimensionless tidal deformability of body 1", "dimensionless", ">= 0"),
        ParameterDefinition("lambda2", "Dimensionless tidal deformability of body 2", "dimensionless", ">= 0"),
        ParameterDefinition("quad_monopole1", "Spin-induced quadrupole parameter C_Q1 of body 1", "dimensionless", "finite"),
        ParameterDefinition("quad_monopole2", "Spin-induced quadrupole parameter C_Q2 of body 2", "dimensionless", "finite"),
    ),
    units=_COMMON_UNITS,
    conventions=(
        *_COMMON_CONVENTIONS,
        ConventionDefinition("quadrupole_phasing", "Spin-induced quadrupole phase difference from Kerr (C_Qi=1); spin-induced octupoles remain at Kerr value."),
    ),
    options=(
        OptionDefinition("spin_order", ("-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"), "7",
                         "Inherited aligned-spin and quadrupole option map."),
        OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                         "Phase convention matching baseline TaylorF2."),
        OptionDefinition("phenom_d_calibration", ("false", "true"), "false",
                         "Suppresses higher terms for legacy calibration compatibility."),
    ),
    carrier_parameters=(),
    capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("source_transcription", "sampled_native_comparison"),
    caveats=("Spin-induced octupoles are fixed to Kerr value (C_Oc=1); not a complete NS model.",),
    claims=(),
)

_PPE_VERSION = "ppe-supported-equations/v2"
_PPE_COMPONENTS = (
    ComponentDefinition(
        id="ppe.inspiral.fa.v1",
        name="Ppe Inspiral Fa",
        spec_version=_PPE_VERSION,
        scope="Parameterized post-Einstein inspiral amplitude modification",
        variant="paper",
        implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source("yunes2009", "Eqs. 1, 47"),
        reference_module="diffgw_reference.ppe_equations",
    ),
    ComponentDefinition(
        id="ppe.inspiral.deltapsi_ppe.v1",
        name="Ppe Inspiral Deltapsi Ppe",
        spec_version=_PPE_VERSION,
        scope="Parameterized post-Einstein inspiral phase increment",
        variant="paper",
        implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source("yunes2009", "Eqs. 1, 47"),
        reference_module="diffgw_reference.ppe_equations",
    ),
    ComponentDefinition(
        id="ppe.dispersion.deltapsi_disp.v1",
        name="Ppe Dispersion Deltapsi Disp",
        spec_version=_PPE_VERSION,
        scope="Massive-graviton dispersion phase increment",
        variant="paper",
        implementation="native_and_reference",
        evidence=("source_transcription", "sampled_native_comparison"),
        source_refs=_source("mirshekari2012", "Eqs. 14–16, 28–31"),
        reference_module="diffgw_reference.ppe_equations",
    ),
)
PPE_COMPONENTS = _PPE_COMPONENTS

PARAMETERIZED_POST_EINSTEIN_INSPIRAL = WaveformDefinition(
    id="ppe-supported-equations/v2",
    name="ParameterizedPostEinstein inspiral amplitude, phase and dispersion",
    aliases=("ParameterizedPostEinstein",),
    domain="frequency",
    modes=((2, 2),),
    scope="Parameterized post-Einstein inspiral amplitude modification, phase increment, and massive-graviton dispersion applied to supported GR carriers.",
    reference_version=_PPE_VERSION,
    reference_module="diffgw_reference.ppe_equations",
    spec_page="math/ppe-phase-dispersion.md",
    components=_PPE_COMPONENTS,
    sources=(
        SourceDefinition("yunes2009", "Yunes and Pretorius, parameterized post-Einstein framework", "https://arxiv.org/pdf/0909.3328v2", "0909.3328v2"),
        SourceDefinition("mirshekari2012", "Mirshekari et al., massive graviton dispersion", "https://arxiv.org/pdf/1110.2720v2", "1110.2720v2"),
        SourceDefinition("project_ppe", "DiffGW ppE overlay algebra and distance conversion", "docs/math/ppe-phase-dispersion.md", _PPE_VERSION, kind="project_convention"),
    ),
    parameters=(
        *_COMMON_PARAMETERS,
        ParameterDefinition("alpha_ppe", "ppE amplitude coefficient alpha", "1", "dimensionless amplitude modifier"),
        ParameterDefinition("a_ppe", "ppE amplitude exponent a", "1", "dimensionless exponent"),
        ParameterDefinition("beta_ppe", "ppE phase coefficient beta", "1", "dimensionless phase modifier"),
        ParameterDefinition("b_ppe", "ppE phase exponent b (b_ppe = 3*b_paper)", "1", "dimensionless exponent"),
        ParameterDefinition("lambda_g", "Graviton Compton wavelength", "m", "> 0 or infinite"),
        ParameterDefinition("redshift", "Cosmological redshift z", "1", ">= 0"),
        ParameterDefinition("d_alpha", "Optional distance parameter D0", "m", "> 0"),
    ),
    units=(
        *_COMMON_UNITS,
        UnitDefinition("wavelength", "m", "graviton Compton wavelength in SI meters"),
        UnitDefinition("dispersion distance", "m", "D0 distance parameter in SI meters"),
    ),
    conventions=(
        ConventionDefinition("fourier", "Negative-Fourier-phase carrier multiplier exp(-i*(deltaPsi_ppe + deltaPsi_disp))."),
        ConventionDefinition("cosmology", "Flat matter/Lambda cosmology with Omega_m=0.3, Omega_Lambda=0.7 for distance conversion."),
    ),
    options=(
        OptionDefinition("carrier", ("TaylorF2", "IMRPhenomD", "IMRPhenomXAS"), "TaylorF2",
                         "Underlying GR carrier waveform model."),
        OptionDefinition("distance_variant", ("supplied_d0", "flat_matter_lambda_from_dl"), "flat_matter_lambda_from_dl",
                         "Distance conversion route for massive-graviton dispersion."),
    ),
    carrier_parameters=(),
    capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("source_transcription", "sampled_native_comparison"),
    caveats=("Qualifies the overlay algebra and composition, not the GR carrier itself or a complete IMR waveform.",),
    claims=(),
)

_EMRI_VERSION = "emri-newtonian-eccentric-quadrupole/v1"
_EMRI_COMPONENTS = (
    ComponentDefinition(
        id="emri.orbit.peters_kepler_rk4.v1",
        name="Emri Orbit Peters Kepler Rk4",
        spec_version=_EMRI_VERSION,
        scope="Adiabatic Peters p,e radiation reaction and Keplerian mean anomaly",
        variant="rk4",
        implementation="native_and_reference",
        evidence=("source_transcription", "independent_scalar_whole_waveform_sample"),
        source_refs=_source("barack2004", "Eq. 27"),
        reference_module="diffgw_reference.emri_newtonian_quadrupole",
    ),
    ComponentDefinition(
        id="emri.harmonics.peters_mathews.v1",
        name="Emri Harmonics Peters Mathews",
        spec_version=_EMRI_VERSION,
        scope="Finite Peters-Mathews harmonic Bessel sum",
        variant="bessel_sum",
        implementation="native_and_reference",
        evidence=("source_transcription", "independent_scalar_whole_waveform_sample"),
        source_refs=_source("barack2004", "Eqs. 7–9"),
        reference_module="diffgw_reference.emri_newtonian_quadrupole",
    ),
    ComponentDefinition(
        id="emri.observer.fixed_polarizations.v1",
        name="Emri Observer Fixed Polarizations",
        spec_version=_EMRI_VERSION,
        scope="Source-frame plus and cross polarizations at fixed observer",
        variant="barack_cutler_eq10",
        implementation="native_and_reference",
        evidence=("source_transcription", "independent_scalar_whole_waveform_sample"),
        source_refs=_source("barack2004", "Eq. 10"),
        reference_module="diffgw_reference.emri_newtonian_quadrupole",
    ),
)
EMRI_COMPONENTS = _EMRI_COMPONENTS

NEWTONIAN_ECCENTRIC_EMRI = WaveformDefinition(
    id="emri.newtonian_eccentric_quadrupole.v1",
    name="NewtonianEccentricEMRI weak-field eccentric EMRI",
    aliases=("NewtonianEccentricEMRI",),
    domain="time",
    modes=((2, 2),),
    scope="Weak-field, leading-quadrupole Newtonian eccentric EMRI polarizations at a fixed observer with adiabatic Peters p,e evolution.",
    reference_version=_EMRI_VERSION,
    reference_module="diffgw_reference.emri_newtonian_quadrupole",
    spec_page="math/emri-kludge.md",
    components=_EMRI_COMPONENTS,
    sources=(
        SourceDefinition("barack2004", "Barack and Cutler, LISA EMRI analytic kludge", "https://arxiv.org/pdf/gr-qc/0310125", "gr-qc/0310125"),
        SourceDefinition("peters1963", "Peters and Mathews, gravitational radiation from point masses in Kepler orbit", "https://journals.aps.org/pr/abstract/10.1103/PhysRev.131.435", "1963PhRv..131..435P"),
        SourceDefinition("peters1964", "Peters, gravitational radiation and the motion of two point masses", "https://journals.aps.org/pr/abstract/10.1103/PhysRev.136.B1224", "1964PhRv..136.B1224P"),
        SourceDefinition("project_emri", "DiffGW Newtonian eccentric EMRI limit", "docs/math/emri-kludge.md", _EMRI_VERSION, kind="project_convention"),
    ),
    parameters=(
        ParameterDefinition("times", "Strictly increasing sample times", "s", "one-dimensional, at least two samples"),
        ParameterDefinition("total_mass", "Central black hole mass M", "nominal solar mass", "> 0"),
        ParameterDefinition("secondary_mass", "Secondary compact object mass mu", "nominal solar mass", "0 < mu <= 1e-3 * total_mass"),
        ParameterDefinition("p0", "Initial dimensionless semi-latus rectum", "1", ">= 12"),
        ParameterDefinition("e0", "Initial orbital eccentricity", "1", "0 <= e0 <= 0.6"),
        ParameterDefinition("inclination", "Fixed observer line-of-sight inclination", "rad", "0 <= inclination <= pi"),
        ParameterDefinition("periapsis", "Initial periapsis angle gamma", "rad", "finite"),
        ParameterDefinition("phase0", "Initial mean anomaly", "rad", "finite"),
        ParameterDefinition("distance", "Luminosity distance", "Mpc", "> 0"),
    ),
    units=(
        UnitDefinition("time", "s", "times in SI seconds"),
        UnitDefinition("mass", "nominal solar mass", "total_mass and secondary_mass in nominal solar masses"),
        UnitDefinition("distance", "Mpc", "distance in Mpc"),
        UnitDefinition("angles", "rad", "inclination, periapsis, and phase0 in radians"),
    ),
    conventions=(
        ConventionDefinition("polarization", "Barack-Cutler Eq. (10) source-frame polarizations at fixed line-of-sight inclination."),
        ConventionDefinition("orbit", "Adiabatic Peters & Mathews p,e radiation reaction with Keplerian mean anomaly."),
    ),
    options=(
        OptionDefinition("n_harmonics", ("2", "3", "4", "5", "6", "7", "8"), "5",
                         "Number of positive eccentric harmonics summed."),
    ),
    carrier_parameters=(),
    capabilities=("td", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("source_transcription", "independent_scalar_whole_waveform_sample", "selected_parameter_gradients"),
    caveats=(
        "Weak-field, leading-quadrupole, fixed-observer limit; does not include spin, precession, generic Kerr frequencies, higher-PN corrections, or LISA response.",
        "Requires M,mu > 0, mu/M <= 1e-3, p >= 12, 0 <= e <= 0.6 for the entire trajectory.",
    ),
    claims=(),
)

CATALOG = (
    TAYLORF2,
    TAYLORF2_ECCENTRIC,
    TAYLORF2_HM,
    IMRPHENOMD,
    IMRPHENOMHM,
    IMRPHENOMXAS,
    KERR_RINGDOWN_220_DIRECT,
    CONTINUOUS_WAVE_H1_DIRECT,
    MEMORY_22_DIRECT,
    TAYLORF2_ALIGNED_SPIN,
    TAYLORF2_TIDAL,
    TAYLORF2_SPIN_QUADRUPOLE,
    PARAMETERIZED_POST_EINSTEIN_INSPIRAL,
    NEWTONIAN_ECCENTRIC_EMRI,
)
WAVEFORMS = CATALOG
validate_catalog(CATALOG)

__all__ = [
    "CATALOG",
    "WAVEFORMS",
    "TAYLORF2",
    "TAYLORF2_ECCENTRIC",
    "TAYLORF2_HM",
    "IMRPHENOMD",
    "IMRPHENOMHM",
    "IMRPHENOMXAS",
    "KERR_RINGDOWN_220_DIRECT",
    "CONTINUOUS_WAVE_H1_DIRECT",
    "MEMORY_22_DIRECT",
    "TAYLORF2_ALIGNED_SPIN",
    "TAYLORF2_TIDAL",
    "TAYLORF2_SPIN_QUADRUPOLE",
    "PARAMETERIZED_POST_EINSTEIN_INSPIRAL",
    "NEWTONIAN_ECCENTRIC_EMRI",
    "TF2_ALIGNED_SPIN_COMPONENTS",
    "TF2_TIDAL_COMPONENTS",
    "TF2_SPIN_QUAD_COMPONENTS",
    "PPE_COMPONENTS",
    "EMRI_COMPONENTS",
]
