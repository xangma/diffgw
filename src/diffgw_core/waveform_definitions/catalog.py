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
    options=(OptionDefinition("phasing_convention", ("coalescence", "reference_frequency"), "coalescence",
                              "Supplying f_ref without an explicit convention selects reference_frequency; finite-reference phase subtraction and amplitude-peak time gauge are project conventions."),),
    carrier_parameters=(), capabilities=("fd", "plus_cross", "torch", "jax", "scalar_reference"),
    evidence=("paper_precision_source_transcription", "conditional_sampled_torch_jax_comparison"),
    caveats=("Inherited extra production fit digits have unresolved publication provenance.",
             "Full intrinsic checks require declared numerical QNM knots and fit rows; agreement does not establish physical accuracy."),
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

WAVEFORMS = (TAYLORF2, TAYLORF2_ECCENTRIC, TAYLORF2_HM, IMRPHENOMD, IMRPHENOMXAS)
validate_catalog(WAVEFORMS)
