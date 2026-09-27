# Mathematical component specifications

DiffGW's mathematical reference lays out each covered waveform as small, exact,
literature-evidenced expressions. These expressions can be read, differentiated,
and evaluated independently of the Torch and JAX implementations.
For concise model summaries and public metadata, start with the
[waveform library](../waveforms/index.md).
Contributors can start from the [canonical model specification template](waveform-spec-template.md),
which also records how to describe research components that are not yet
qualified for the public catalog.
Every handwritten waveform model, component, and extension page follows its
six-section outline; the documentation generator checks this structure and
fills the cataloged model pages' identity and option tables.

| Model specification | Covered | Pending |
| --- | --- | --- |
| [TaylorF2, `taylorf2-nonspinning-3.5pn/v1`](taylorf2.md) | Nonspinning point-particle phase through 3.5PN, Newtonian amplitude, polarizations, explicit phase conventions, selected derivatives | Spins, tides, eccentricity outside the separate low-e slice, higher harmonics, cutoffs and tapers |
| [TaylorF2_HM, `taylorf2-hm-leading-carrier/v1`](taylorf2-hm.md) | Eight leading-amplitude SPA modes, Wigner-d angular projections, inherited aligned-spin phase, source-composed nonspinning polarizations, complex Torch/JAX comparisons and selected derivatives | Higher-PN and spin-amplitude terms, independent external numerical eight-mode oracle, Kerr/taper checks and physical validation |
| [TaylorF2_Eccentric nonspinning composition, `taylorf2-eccentric-nonspinning-e2-3pn/v1`](taylorf2-eccentric.md) | Restricted-amplitude 3PN order-`e0²` inspiral polarizations; independent scalar composition, Torch/JAX checks, and two strict pinned-LAL holdouts | Aligned-spin eccentric composition, optional amplitude, wider-domain binary and physical validation |
| [TaylorF2_Eccentric phase, `ecc.phase-3pn-e2/v1`](taylorf2-eccentric-phase.md) | Moore et al. order-`e0²` SPA phase through relative 3PN, all six native PN truncations and the separate `f_ref` gauge, checked against Torch/JAX | Optional amplitude factor, spin-eccentric coupling, cutoffs and physical validation |
| [Aligned-spin TaylorF2, `taylorf2-aligned-bh-spin-3.5pn/v1`](taylorf2-aligned-spin.md) | Composes the unchanged baseline with separately evidenced black-hole spin terms; explicit term sets, legacy frontend profiles, spin derivatives, FP64 checks | General quadrupoles, tides, absorption, precession, spin amplitude corrections, cutoffs and tapers; spinning FP32 qualification |
| [Adiabatic tidal TaylorF2, `taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1`](taylorf2-tidal.md) | Composes both prior specifications with 5PN/6PN electric quadrupole tides and the 6.5PN tail; independent component/effective forms, explicit selectors and missing inputs, tidal derivatives | General neutron-star spin multipoles, magnetic/octupolar/dynamical tides, spin-tidal couplings, tidal amplitude corrections, cutoffs and tapers; tidal FP32 qualification |
| [NRTidal v1 nonspinning tidal phase, `nrtidalv1-nonspinning-tidal-phase/v1`](nrtidalv1-phase.md) | Published Padé phase and its three-decimal carrier realization, independent scalar math, Torch/JAX derivatives and direct complex-quotient LAL binary gate | Exact analytic 1PN constraint in the rounded carrier, merger/taper, spinning cases, inherited baseline and physical validation |
| [NRTidal v1 merger taper and composition, `nrtidalv1-nonspinning-merger-taper/v1`](nrtidalv1-merger-taper.md) | Published merger fit and Planck taper against Torch/JAX, complex tidal factor, six-case premerger full-carrier and zero-tide taper public-binary gates | Nonzero-tide binary taper discrepancy, full tapered waveform, spins and physical validation |
| [NRTidalv2 nonspinning tidal phase, `nrtidalv2-nonspinning-tidal-phase/v1`](nrtidalv2-phase.md) | Published Padé phase, mass-weighted deformability, selected Torch/JAX derivatives and direct complex-quotient LAL binary gate | Tidal amplitude, merger and taper, self-spin, other modes, full-family or physical validation |
| [NRTidalv2 nonspinning amplitude and taper, nrtidalv2-nonspinning-tidal-amplitude-taper/v1](nrtidalv2-amplitude-taper.md) | Eq. (24) additive amplitude is a separately verified Pv2 component; explicit `paper_2019` D omits it; historical default retains pinned-LAL binary compatibility with unresolved amplitude-normalization provenance | Self-spin, other modes, full-family or physical validation |
| [NRTidalv2 spinning paper-D composition, `nrtidalv2-spinning-paper-d-composition/v1`](nrtidalv2-paper-d-composition.md) | Printed tidal phase, 2PN+3PN self-spin and merger taper jointly checked as unaligned complex Torch/JAX quotients through the taper on three positive-Lambda spinning systems | Fig. 7 and pinned-binary merger/amplitude conflicts, reference-frequency adapter, full waveform and physical validation |
| [NRTidalv2 Pv2 aligned additive terms, `nrtidalv2-pv2-aligned-additive-terms/v1`](nrtidalv2-pv2-aligned-additive-terms.md) | Signed Eq. (24) amplitude, printed tidal phase, 2PN+3PN+3.5PN self-spin and merger taper jointly checked before precession rotation on three spinning systems | Public Pv2 BNS assembly, binary contract, Fig. 7 and merger-frequency conflicts, full waveform and physical validation |
| [NRTidalv2 Pv2 positive-22 projection, `nrtidalv2-pv2-positive-22-projection/v1`](nrtidalv2-pv2-positive-22-projection.md) | Published positive-frequency mode-to-polarization operator and pre-rotation linearity checked against independent scalar reference for Torch/JAX | BNS angle dynamics, full waveform and physical validation |
| [NRTidalv2 Pv2 tidal-mode normalization, `nrtidalv2-pv2-tidal-mode-normalization/v1`](nrtidalv2-pv2-tidal-mode-normalization.md) | Signed Eq. (24) face-on strain converted to positive-22 mode and projected at supplied angles against an independent scalar reference | BNS angle dynamics, reference-frequency convention and full waveform validation |
| [NRTidalv2 Pv2 aligned BNS carrier, `nrtidalv2-pv2-aligned-carrier/v1`](nrtidalv2-pv2-aligned-carrier.md) | Published finite-size phase, signed amplitude and merger taper composed with a supplied BBH carrier before rotation; Torch/JAX complex carrier and supplied-angle projections against 80-digit scalar reference | Native BNS precession-angle and reference-frequency conventions, public dispatch, source/binary conflicts and physical validation |
| [NRTidalv2 Pv2 public assembly boundary, `nrtidalv2-pv2-public-assembly/v0`](nrtidalv2-pv2-public-assembly-boundary.md) | Source audit of inherited angle, source frame and phase/time origin; records the native compatibility conventions without claiming a BNS dispatch | Paper-exact BNS angles, merger-time spline, reference-frequency phase gauge, complete complex waveform and physical validation |
| [NSBH London 220 QNM input, `nsbh.qnm.london-220.v1`](nsbh-london-qnm.md) | Printed Kerr 220 polynomial, frequency and quality factor checked against an independent scalar on 18 supplied remnant states in Torch/JAX | Remnant mass and spin providers, phase QNM, amplitude assembly and whole waveform |
| [Spin-induced quadrupole TaylorF2, `taylorf2-spin-induced-quadrupole-3.5pn/v1`](taylorf2-spin-quadrupole.md) | Composes the prior three specifications with non-Kerr 2PN/3PN/3.5PN quadrupole phase corrections, including cubic spin; selectors, derivatives and compiled frontend checks | Independent spin-induced octupoles, spin–tidal couplings, precession, amplitude corrections, cutoffs and merger |
| [SI/IAU units, `si-iau-waveform-units/v1`](units.md) | Independent definitions and FP64 rounding checks for speed of light, solar mass in seconds, and Mpc | Other physical constants |
| [IMRPhenomD amplitude, `imrphenomd-amplitude-joins-paper-precision/v1`](imrphenomd-amplitude.md) | Seven printed amplitude fits, PN and merger-ringdown amplitude, peak and quartic joins; Torch/JAX equation checks | Physical validation and provenance of inherited extra fit digits; waveform composition is covered separately below |
| [IMRPhenomD remnant and QNM inputs, `imrphenomd-remnant-qnm-paper-precision/v1`](imrphenomd-remnant-qnm.md) | Printed final-spin/radiated-energy fits, mass and QNM scaling, interior natural spline, high-spin generated continuation and numerical holdout checks for both frontends | Independent QNM eigenproblem specification, full-domain physical validation, and provenance of inherited extra remnant digits |
| [IMRPhenomD phase and joins, `imrphenomd-phase-joins-paper-precision/v1`](imrphenomd-phase.md) | Printed phase fits, aligned-spin PN phase, three integrated phase branches, two $C^1$ joins, frontend phase gauges; Torch/JAX checks | Provenance of inherited extra fit digits and physical validation; waveform composition is covered separately below |
| [IMRPhenomD composed polarizations, `imrphenomd-composed-polarizations/v1`](imrphenomd-waveform.md) | Composes the amplitude and phase slices with audited SI/IAU units; direct FP64 Torch/JAX complex-strain and selected derivative checks on declared QNM inputs and both phase conventions | Inherited fit provenance, FP32, full-domain intrinsic derivatives and physical validation; a selected full-chain intrinsic slice follows below |
| [IMRPhenomD intrinsic composition, `imrphenomd-intrinsic-composition/v1`](imrphenomd-intrinsic.md) | Connects published remnant fits, declared numerical QNM data, and unchanged composed polarizations; selected full-chain mass-ratio and spin derivatives on rounded, signed high-spin, and active-cap branches, plus complete high-spin waveform samples, against Torch/JAX | Inherited fit provenance, derivatives at the cap/switch itself, FP32, full-domain derivative qualification and physical validation |
| [IMRPhenomHM frequency map, `hm.fmap.eq5.v1` and `hm.fmap.phase.eq8.v1`](imrphenomhm-frequency-map.md) | London et al. piecewise amplitude and D-carrier phase maps; independent scalar values, joins, and selected Torch/JAX derivatives | QNM fit provenance, mode amplitude and phase-offset assembly, complete waveform and physical validation; research-only component |
| [IMRPhenomHM phase offsets, `hm.phase.offsets.eq9.v1`](imrphenomhm-phase-offsets.md) | London et al. three-branch phase multipliers and continuity offsets on a supplied carrier; scalar values and full frequency-input derivatives against live Torch/JAX helpers | Eq. (14) carrier correction, QNM input provenance, amplitude, full mode assembly and physical validation; research-only component |
| [IMRPhenomXAS amplitude, phase, composed polarizations, and mass scaling](imrphenomxas.md) | Paper-literal amplitude equations and all three phase branches, with carrier differences identified; complete quartic boundary adapter, composed complex strain, phase, and selected total-mass/mass-ratio/spin tangents against Torch/JAX; selected LAL binary amplitudes; opt-in author-asset fit values and parameter derivatives | Fifth-order amplitude frontend support, full-domain intrinsic derivative bounds, independent calibration and physical validation |
| [IMRPhenomXHM (3,2) transition and mixing](imrphenomxhm-transition-mixing.md) | Published intermediate collocation, spherical mixed-ringdown C2 join, spheroidal phase derivative, and two-mode projection checked against Torch/JAX | Current-release 122022 calibration and mixing asset provenance, full-waveform catalog and holdout gates; research-only component |
| [IMRPhenomXHM (3,2) spheroidal ringdown phase, `xhm.phase.32.spheroidal-ringdown-shape.v1`](imrphenomxhm-32-spheroidal-ringdown-phase.md) | Eqs. (46)–(47) phase and derivative on supplied QNM frequencies and coefficients, with scalar/Torch/JAX values and full Jacobians | Four-target Eq. (48) calibration, Eq. (49)–(50) fitted alignment, exceptional `(3,2)` cuts, and complete mode qualification |
| [KerrRingdown direct-class physical-energy 220, `kerr-ringdown-220-energy-direct-class/v1`](kerr-ringdown-direct-class.md) | Source-backed independent QNM and spectral angular factors composed with energy amplitude into causal TD/FD polarizations; two selected whole-waveform Torch/JAX checks and extrinsic derivatives | Direct-class public-binary mismatch, dispatcher and other options/modes, intrinsic JAX spin gradient, excitation physics, and domain-wide validation |
| [ContinuousWave monochromatic H1 direct class, `cw-monochromatic-h1-direct-class/v1`](continuous-wave-direct-class.md) | Selected principal-axis 2f strain, paper scalar composition, H1 detector response, both amplitude routes, and Torch/JAX values and gradients | Nonzero spin-down, other sites, independent full-precision H1 survey and UT1, ephemeris, external binary and physical validation |
| [Kerr 220 public-clean carrier, `kerr-220-energy-carrier-conditional/v1`](kerr-ringdown.md) | Scalar energy normalization, causal mode/mirror polarizations, and Fourier transform checked against both frontends on supplied inputs; separate selected 220 QNM and angular-input checks | [Opt-in public-clean composition is blocked by an independent angular mismatch](kerr-ringdown-public-clean-gate.md); excitation physics, other modes/overtones, and domain-wide validation |
| [Kerr angular projection, `kerr-angular-projection-scalar/v1`](kerr-angular-projection.md) | Independent scalar spherical and spheroidal angular factors, selected eigenvalues, and angle derivatives checked against Torch/JAX; Torch real-oblateness derivative | JAX oblateness gradient, all supported modes, angular-series stopping rule, and domain-wide validation |
| [Kerr selected 220 QNM inputs](kerr-qnm-scalar-qualification.md) | Independent radial continued-fraction frequency and angular eigenvalue checks, including selected spin derivatives, against Torch/JAX | Near-extremal spins, other modes and overtones, domain-wide validation |
| [Dominant-pair nonlinear memory, `memory.dominant_22/v1`](memory-dominant-22.md) | Favata's symmetric ±22 full plus pattern, its $h_{20}$ projection, and selected Torch/JAX values and derivatives with explicit SI distance | General mode-pair coupling, waveform generation, legacy polarization-only carrier, prior-history offset and physical validation |
| [MemoryWaveform22Direct, `memory-dominant-22-direct-class/v1`](memory-dominant-22-direct-class.md) | Scoped direct-class `evaluate_22` route with independent sampled-grid scalar comparison | Legacy class call, arbitrary-mode coupling, external same-model comparison, and physical validation |
| [EMRI prototype leading Peters flux, `emri.flux.peters_leading.v1`](emri-kludge.md) | Source-derived `p,e` secular rates and CPU float64 Torch/JAX checks; research-only component | Generic Kerr orbit, harmonic and LISA TDI response; full EMRI waveform qualification |
| [ppE inspiral amplitude, phase and dispersion](ppe-phase-dispersion.md) | Separate amplitude `v1` and phase/dispersion `v1` scalar checks against Torch/JAX, with a conditional carrier adapter | Merger/ringdown branches, carrier fidelity, and wider cosmologies |
| Other waveform families | No executable specification yet | Further family-by-family slices |

The Kerr carrier has separate, bounded input evidence for
[angular projection](kerr-angular-projection.md) and
[scalar 220 QNM frequencies and eigenvalues](kerr-qnm-scalar-qualification.md).

## How to read a specification

Each component has a stable ID, a formula, input units and domain, a pinned paper
version and equation number, and any transformations needed to reach the chosen
convention. Project choices and calibrated data must be identified separately
from published analytic results. Corrections that change numerical meaning need
a new specification version and an explanation of the difference.

The canonical expressions live in the optional `diffgw_reference` package.
SymPy keeps rational coefficients exact and produces symbolic derivatives;
mpmath evaluates the small scalar expression tree at a chosen precision. The
reference imports no production waveform functions, constants, or numerical
helpers. The frontend tests provide explicit unit conversions at the boundary.
The displayed coefficient table is generated from the canonical expressions.
The [component traceability map](taylorf2.md#component-traceability) records source
locations, reference functions, both frontend functions, and maintained test
names for every component, including explicitly reference-only expressions.
Structural tests detect missing IDs, dangling links, coefficient-order drift,
and stale rendered documentation without importing either frontend.
The [aligned-spin map](taylorf2-aligned-spin.md#component-traceability) retains
every inherited component version and adds separately sourced corrections and
composed expressions. Its [option table](taylorf2-aligned-spin.md#frontend-option-contract)
distinguishes the frontends' legacy selectors from physical PN orders.
The [tidal map](taylorf2-tidal.md#component-traceability) retains both earlier
specifications and records the separate tidal terms, effective parameters,
and [missing-input/selector contract](taylorf2-tidal.md#frontend-option-contract).
The [spin-quadrupole extension](taylorf2-spin-quadrupole.md) retains all three
earlier versions and traces each non-Kerr correction to pinned paper equations.

## What the evidence establishes

Three questions remain distinct:

1. **Literature fidelity:** do the recorded expressions and conventions represent
   the specified publications? Equation-level citations and transcription review
   address this. A numerical test cannot establish it by itself.
2. **Implementation verification:** do Torch/JAX implement those expressions?
   Component, value, and derivative comparisons address this on declared cases.
3. **Physical validation:** does the model describe the physical system accurately?
   This requires separate physical references and an appropriate parameter domain.

The reference is reviewable and versioned, rather than infallible. High precision
reduces numerical error; it does not fix a wrong equation. Frozen fixtures are
snapshots from the same reference, not a second independent derivation. No Lean
proof or Mathematica cross-check is claimed for this first version. Those can be
added as separate evidence without replacing the source and convention audit.

## Development workflow

Install `python -m pip install -e '.[reference,torch,jax,dev]'`. For each added
component, first document its source and assumptions, then build the independent
expression and converged examples, then compare each frontend against it.
Avoid using frontend results to define the expected coefficients or fixtures.
Record disagreements before deciding whether they are transcription errors,
numerical errors, or documented model choices.

Review changes to sources, formulas, conventions, and calibration inputs against
the affected component IDs. A change in mathematical meaning requires a new
specification version, an explanation of the difference, updated traces and
coverage, and reviewed regenerated artifacts. Numerical comparisons must not be
used to choose source equations simply because they match existing code.

The maintained CPU CI job runs the complete mathematical-reference test set in
`--require-math-reference` mode: both frontends and reference dependencies must
be present, every selected reference module must execute fully, and skips or
expected failures fail the gate. Normal optional-dependency runs remain useful
for local development but do not supply this complete qualification. The job
also reports dependency versions and checks the maintained equation/fixture
artifacts; the test set checks the generated component map.

A separate CPU CI job installs current LALSuite and PyCBC releases, then runs
maintained focused LAL tests and frozen strict binary-oracle comparisons. This includes
selected same-model IMRPhenomD intrinsic derivatives, TaylorF2 non-Kerr
quadrupoles, and nonspinning TaylorF2 electric tides. Its
[coverage and model boundaries](../validation.md) are distinct from the
paper-based mathematical-reference gate.

See the [TaylorF2 reproduction commands](taylorf2.md#reproduce-the-checks) and the
broader [validation evidence](../validation.md).
