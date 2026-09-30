# Waveform qualification status

This 29 September 2026 review separates implemented mathematical and numerical
scopes from remaining provenance, compatibility and physical-accuracy
questions. The linked specifications and dated receipts define the actual
options, samples and thresholds. Publication in the [waveform library](index.md)
or [dispatch inventory](coverage.md) does not qualify a complete waveform or
expand its tested domain.

## How to read the claims

Qualification belongs to an explicitly named model version, component,
convention, mode set, parameter domain and option tuple. Record it on separate
axes:

| Claim | Evidence and interpretation |
| --- | --- |
| Source and provenance | Pinned equations, conventions and calibration origins. Distinguish published mathematics, adapted author code, inferred compatibility choices and third-party data. |
| Independent reference | Independently executable scalar calculations with numerical-stability checks. Supplied fit targets or QNM inputs leave upstream provenance conditional. |
| Native frontends and derivatives | Reference-to-Torch/JAX values, selected derivatives, batching and JIT checks for the declared paths and precision. Torch/JAX agreement alone is insufficient. |
| External compatibility | Versioned binary or independent-code comparisons with matching options and frozen gates. Optimized matches differ from direct complex-waveform comparisons. |
| Physical accuracy | Independently chosen NR or other physical holdouts, data-resolution screens and frozen criteria. Internal references and binary matches do not establish this claim. |

Use **sampled pass**, **conditional**, **failed**, **incomplete** or **blocked**
for each axis rather than whole-model qualification tiers. Failed physical
gates do not invalidate independently closed implementation scopes; closed
components do not qualify assembled waveforms. Preserve failures and their
thresholds when adding evidence.

There is no universal 1e-12 parity bound or smoothness guarantee. Tolerances
depend on the component and reference. An analytic Planck taper is smooth
under its stated endpoint construction; full waveforms also contain masks,
chart switches, guards and numerical solvers. Derivatives need branch-specific
checks. Plateau and symmetry derivatives can legitimately be zero. The NSBH
project continuation below is only C1.

## Family scopes

| Family or scope | Source, reference and native evidence | External and physical limits |
| --- | --- | --- |
| [TaylorF2 base](../math/taylorf2.md) | Registered nonspinning 3.5PN point-particle phase and restricted Newtonian quadrupole amplitude with independent scalar/native checks and explicit phase/reference conventions | Selected LAL comparisons are bounded evidence. Spin, tides, eccentricity, higher modes and optional cutoffs have separate contracts. |
| [TaylorF2 aligned spin](../math/taylorf2-aligned-spin.md), [tides](../math/taylorf2-tidal.md), [spin quadrupole](../math/taylorf2-spin-quadrupole.md) | Separately versioned coefficient and waveform scopes with scalar references and Torch/JAX checks | The aligned-spin receipt reports unavailable optional external dependencies skipped, without an external-reference claim. Selected tidal and 2PN/3PN quadrupole binary checks do not certify every higher-order option or the full parameter domain. |
| [TaylorF2 eccentric](../math/taylorf2-eccentric.md) | Nonspinning, order-e0-squared 3PN phase, explicit eccentric reference frequency, 1 <= q <= 2, 0 < e0 <= 0.1, amplitude correction off | Two strict nonspinning LAL holdouts pass. This is same-model sampled evidence, not uniform accuracy or spinning-eccentric qualification; see the [27 September phase receipt](../validation/taylorf2_eccentric_phase_20260927.md). |
| [TaylorF2 higher modes](../math/taylorf2-hm.md) | Eight leading SPA modes and their composition have independent source/reference checks | Independent external multimode comparison remains open, including the minimal nonspinning (2,2)+(3,3) slice. Public LAL TaylorF2 mode-array output is a single carrier, not that oracle; see the [27 September audit](../validation/taylorf2-hm-minimal-multimode-oracle-20260927.md). |
| [IMRPhenomD](../math/imrphenomd-waveform.md) | Versioned carrier compositions and scoped scalar/native/binary comparisons, conditional on specified fit and QNM inputs | These checks do not independently validate every calibration input, extra fit digit or whole-domain physical accuracy. |
| [IMRPhenomXAS](../math/imrphenomxas.md) | Versioned compositions distinguish paper expressions, carrier choices and supplied fit inputs | The [27 September waveform receipt](../validation/imrphenomxas_whole_waveform_20260927.md) passes six CPU FP64 comparisons at its 1e-10 gate on nonzero binary support. It excludes the native tail beyond the binary mask and does not resolve calibration provenance, disputed source choices or NR accuracy. |
| [KerrRingdown220Direct](../math/kerr-ringdown-direct-class.md) | Cataloged physical-energy spectral (2,2,0) direct-class slice with sampled references and native JAX inverse-iteration value/gradient checks | Staged direct-class binary diagnostics have substantial convention/angular residuals. Passing dispatcher holdouts select another route. The separate [public-clean angular gate](../math/kerr-ringdown-public-clean-gate.md) remains blocked; details below. |
| [ContinuousWaveH1Direct](../math/continuous-wave-direct-class.md) | Sampled H1 antenna and circular 1-AU timing construction | Fixed GPS/UTC offset and UTC-for-UT1 approximation; no full ephemeris, Einstein/Shapiro delay, binary orbit or external physical-accuracy claim. The strict source spindown diagnostic remains an expected failure outside the catalog slice. |
| [MemoryWaveform22Direct](../math/memory-dominant-22-direct-class.md) | Dominant supplied physical h22 coupling to memory (2,0) and (4,0) with sampled reference/native checks | Legacy coupling diagnostic remains a strict expected failure. General mode pairs, precession and external LAL/NR accuracy are outside this scope. |
| [ParameterizedPostEinstein](../math/ppe-phase-dispersion.md) | Sampled independent inspiral amplitude, phase and massive-graviton dispersion equations, native values and selected gradients | Applying the algebra to an IMR carrier does not supply a unique paper-derived merger/ringdown model. No external full-waveform or physical-accuracy claim. |
| [EMRI_NewtonianEccentric](../math/emri-kludge.md) | Bounded Newtonian quadrupole, eight Peters–Mathews harmonics and Peters evolution with scalar/Torch/JAX checks | Shared RK4 discretization leaves possible common truncation error. No FEW comparison, relativistic AK/NK/Kerr physics, unrestricted harmonics, long-duration accuracy or detector-response qualification; see the [29 September receipt](../validation/emri_newtonian_quadrupole_20260929.md). |
| [IMRPhenomHM published composition](../math/imrphenomhm-waveform.md#implementation-closure-2026-09-28) | 22 CPU FP64 sampled implementation claims closed on 28 September, with separately versioned orbital and SPA conventions | Legacy finite-reference covariance defect, inherited-default provenance and untested domains remain. Multiple NR gates fail, as recorded below. |
| [IMRPhenomP/Pv2](research/phenom-precessing.md) | Native routes and bounded component/comparison evidence in a provisional ledger | P class aliases, public Pv2 dispatch and named binary variants are distinct routes with different default reference-frequency/convention behavior. Effective-spin, angle, frame, limit and full-complex option-tuple closure remain incomplete. |
| [IMRPhenomXP v223](research/xp-v223-reference-closure.md) | Scoped geometry, physical zero-nutation continuation, selected unequal-mass generic charts and mapped nonzero-azimuth composition pass their declared reference/binary checks | Compatibility-derived choices remain distinct from paper identities. Equal-mass/invalid-root angle fallbacks, remnant restrictions, other versions and near-aligned binary boundaries are not universally closed. [Mixed boundary batches and an aligned equal-mass scalar fixture](../validation/xp_xphm_boundary_qualification_20260925.md#2026-09-26-explicit-remnant-policy-amendment) can produce NaN reverse-mode common-mass gradients in JAX. New sampled boundary tests do not certify an entire chart. |
| [XPHM SpinTaylor](../validation/spintaylor_reference_angle_20260929.md) | Independent paper-derived reference-angle gauge and selected native angle checks; optional damp_beta continuation has sampled tests | The [29 September direct waveform comparison](../validation/spintaylor_residual_20260929.md) fails at about 25% relative L2. Optional damping does not resolve inspiral evolution, frame, carrier or the full attachment contract. |
| [IMRPhenomXHM/XPHM](research/xhm-xphm-qualification-20260929.md) | Independent supplied-target amplitude, phase, cuts, mixing, projection and precession components; 94 focused checks passed in the dated receipt | Release-122022 calibration/provider provenance and full carrier assembly remain incomplete. Four strict frozen binary cases fail, 0/4. This is compatibility evidence, not NR validation. The new supplied-target helper is not connected to production. |
| [IMRPhenomXPNR](research/imrphenomxpnr.md) | [37 versioned published-form expressions](../math/imrphenomxpnr-components.md) with independent SymPy/mpmath and native component checks | Terminal states, transitions, attachments, tuned carrier/asymmetry and calibrated assets remain incomplete. Isolated expressions and a mock angle fitter do not establish a full XPNR waveform. |
| [IMRPhenomT/THM/TP/TPHM](research/imrphenomt-family.md) | Selected inspiral, supplied-input merger/ringdown, rotation and precession RHS/trajectory scopes; [29 September merger checks](../validation/phenomt_merger_20260929.md) | The [validation record](../validation/phenomt.md) identifies adapted MIT author reconstruction and unprinted fits rather than an entirely independent published calibration. Full mode/fit closure is incomplete; four strict q=10 precession binary gates fail. |
| [NRTidal v1 D](../validation/nrtidal_family_qualification_20260929.md) | Printed nonspinning phase/taper and bounded native/binary checks use the pinned later revision of the coupling equation | The original arXiv v1 unequal-mass coupling differs and is not a native waveform option. Historical taper conventions retain their stated bounds; broader spinning/full-carrier scopes are separate. |
| [NRTidal v2 D](../math/nrtidalv2-paper-d-composition.md) | paper_2019 printed spinning tidal quotient through taper passes independent Torch/JAX value, derivative and JIT checks. The default binary_compatible option separately has premerger binary evidence | Default amplitude normalization is not derived from 2019 Eq. (24); exact binary parity is not claimed for paper_2019. Nonzero-tide default taper binary gates fail. Love–Q and merger-fit conflicts remain in the [provenance audit](../validation/nrtidalv2_provenance_20260927.md). |
| [NRTidal v2 Pv2 conditional carrier](../math/nrtidalv2-pv2-aligned-carrier.md) | Supplied BBH carrier and supplied-angle finite-size/projection components pass independent scalar/native checks | No public diffgw precessing BNS dispatch; BNS angle evolution, phase/time origin and reference anchoring remain [pending](../math/nrtidalv2-pv2-public-assembly-boundary.md). V1 Pv2 remains a capability gap. |
| [IMRPhenomNSBH](research/tidal-eccentric-higher-modes.md) | Bounded disruption, QNM and supplied-target polarization composition; sampled checks of a project C1 Lambda continuation and plus-window choice | Joining-frequency exception, source variants and remnant guards remain unresolved, as detailed below. [Accepted public options](research/nsbh-public-options-20260929.md) do not validate NS spin or extrapolated domains. |
| [Official SEOBNR_ROM / SEOBNRv4_ROM](../external-data.md) | Current native Torch/JAX evaluator uses checksum-pinned authors' SEOBNRv4ROM v3 HDF5 bases and interpolation coefficients | Third-party data retains its upstream terms. This is an external-data evaluation contract, not independently generated calibration or a newly qualified native EOB generator. |
| [Retired native EOB/ROM/TEOB prototypes](research/retired-release-surface.md) | Independently scoped equation components survive; native SEOBNR development has separate dated records | Former complete-waveform pilots are removed from release dispatch. Official ROM evidence cannot qualify them; failed native-generator gates are retained below. |

## Failed and blocked gates retained

### Kerr routes

The [direct-class specification](../math/kerr-ringdown-direct-class.md)
records staged LALSimulation 6.2.1 diagnostics for final mass 53 Msun,
spin 0.35 or 0.7, distance 230 Mpc, energy fraction 0.017, inclination 1.1,
phase 0.41 and 50–500 Hz at 1 Hz spacing. Direct plus/cross relative L2 is
1.232/1.554 and 1.238/1.536, respectively. Empirical phase/sign adjustments
reduce residuals but are not sourced input conventions. The passing
KerrRingdown_holdout public dispatcher does not select this direct class.
The direct-class positive-frequency time carrier also differs in sign from
the cited separated Teukolsky ansatz. Its declared scalar-reference agreement
does not establish that the carrier and spheroidal factor form that solution.

The [public-clean gate](../math/kerr-ringdown-public-clean-gate.md) remains
**blocked** on two off-axis CPU FP64 cases: spin/mass (0.35,53) and (0.7,92),
TD samples tc + (-0.001,0,0.002,0.008) seconds and FD samples
(-130,0,35,110,270) Hz. Independent spectral TD/FD relative errors are
1.68e-4/1.98e-4 and 9.44e-4/9.47e-4, despite Torch/JAX agreement near 1e-16.
The inferred angular stopping/normalization convention lacks a published
source. A separate source-spectral option passes selected checks without
promoting the default or direct-class binary claim.

### HM physical holdouts

The [27 September published-composition NR receipt](../validation/imrphenomhm_published_nr_receipt_20260927.md)
uses the declared published D calibration, PN choices and London mapping,
NR modes through ell=5, total mass 100 Msun, distance 500 Mpc and 30–1024 Hz
with the design PSD. Time, orbital phase and polarization are optimized.
Gates are match >=0.980 and norm ratio [0.85,1.15]:

| Retained failed case | Measured result |
| --- | --- |
| SXS:BBH:2374, q approximately 5, nonspinning, view (1.2,0.4) | Match 0.971607892, norm ratio 0.750351161: both fail. |
| SXS:BBH:2163, q approximately 3, aligned spins approximately +0.6,+0.6, view (1.2,0.4) | Match 0.972272538, norm ratio 1.223703469: both fail. |
| [SXS:BBH:2265](../validation/imrphenomhm_sxs_2265_receipt_20260927.md) and [3631](../validation/imrphenomhm_sxs_3631_receipt_20260927.md), default D-carrier comparisons | Each fails 9 of 10 mode/projection observables in both frontends; only (2,2) passes. Their NR resolution screens pass. |
| [SXS:BBH:3619](../validation/imrphenomhm_sxs_3619_projection_receipt_20260927.md), views (0.7,0) and (1.2,0.4) | Model match/norm checks pass, but NR resolution matches 0.991880/0.980143 fail the 0.995 screen: overall failure. |
| [SXS:BBH:3630](../validation/imrphenomhm_sxs_3630_projection_receipt_20260927.md), view (1.2,0.4) | Resolution screen and match pass; norm ratio 0.840431923454 fails 0.85: overall failure. |

Passing views in the same receipt and separate
[2158](../validation/imrphenomhm_sxs_2158_receipt_20260927.md) and
[3144](../validation/imrphenomhm_sxs_3144_receipt_20260927.md) receipts remain
valid for their sampled cases. They do not clear these failures or certify
all inclinations and parameter combinations.

### Precession and higher-mode binary gates

The [XP v223 boundary record](research/xp-v223-reference-closure.md) retains
an aligned convention-6 LAL relative L2 of about 0.118. Later internal up-down
continuity checks pass the unchanged 1e-6 threshold, superseding the older
internal-continuity failure while the binary-boundary failure remains.
Selected unequal-mass generic mapped-frame fixtures pass; equal-mass/root
fallback coverage and other option tuples remain separate.

The [29 September SpinTaylor receipt](../validation/spintaylor_residual_20260929.md)
uses masses (36,29) Msun, spins (0.3,0.2,-0.1) and (0.1,-0.2,0.2),
inclination 0.6, distance 100 Mpc, reference 37.3 Hz and 20–300 Hz at
0.25 Hz spacing against LALSimulation 5.3.1 XPHM version 320/final-spin 2.
Raw plus/cross relative L2 is 0.253636/0.255445; fitting time and constant
complex scale lowers it to 0.059416/0.060229 without passing the direct gate.
Even (2,2)-only raw residuals are 0.248621/0.249230. Optional continuation
tests do not close this whole-waveform failure.

The [29 September XHM/XPHM frozen receipt](research/xhm-xphm-qualification-20260929.md)
compares default-option catalog/holdout tuples against LALSuite 7.26.15,
LALSimulation 6.2.1 on len, over 20–512 Hz at 0.25 Hz spacing. Catalog
XHM/XPHM masses are (40,20)/(35,25) Msun; holdouts are (55,11)/(42,17).
The [raw receipt](research/xhm-xphm-frozen-binary-20260929.json) retains
their spins, extrinsics and reference-frequency choices. Every polarization
must satisfy relative L2 <=1e-6 and flat mismatch <=1e-8:

| Case | Plus L2 / mismatch | Cross L2 / mismatch |
| --- | ---: | ---: |
| XHM catalog | 0.374023 / 0.063006 | 0.366201 / 0.060753 |
| XPHM catalog | 0.228818 / 0.025734 | 0.226417 / 0.025321 |
| XHM holdout | 0.546374 / 0.127476 | 0.462455 / 0.094623 |
| XPHM holdout | 0.504113 / 0.112109 | 0.556021 / 0.134134 |

The strict verdict is **0/4**. A SOVEREIGN_NR_CALIBRATED harness label is not
a passed gate. These binary comparisons do not establish NR accuracy or
outcomes for every nondefault option.

The [27 September T-family blocker](../validation/phenomt_q10_gate_blocker_20260927.md)
retains four strict failures for (60,6) Msun, spins (0.2,0.1,0.8) and
(0.1,0,0.1), reference 20 Hz. TP/TPHM MSA223/final-spin 3 relative L2 is
0.0202333/0.0225165 against 0.01; mismatch 0.000200/0.000249 against 0.0001.
Numerical PV300/final-spin 4 L2 is 0.308418/0.307103 against 0.03; mismatch
0.04753/0.04700 against 0.0005. Paper-RHS agreement and nearby passing
aligned/NNLO cases do not resolve undocumented conventions/state maps.

### Matter source and binary boundaries

The [29 September NRTidal receipt](../validation/nrtidal_family_qualification_20260929.md)
retains default v2 D nonzero-tide taper errors 0.000153757344 (equal mass)
and 0.000189164604 (unequal mass), both above the frozen 0.00005 bound.
The [dense taper audit](../validation/nrtidalv2_d_taper_pv2_boundary_20260929.md)
localizes the discrepancy without identifying a source-backed coefficient
correction. Printed paper-D quotient and conditional Pv2 assembly checks
remain separate passes.

The [27 September NSBH joining receipt](../validation/nsbh_joining_frequencies_20260927.md)
finds maximum relative discrepancy 0.1505 in the low/no-torus joining center
against literal Table I on its frozen 210-input grid. The extra native outer
factor is a compatibility choice. Other named joining rows pass; the full
four-row component remains conditional. The
[remnant guard source gate](../math/nsbh-remnant-guard-source-boundary.md)
is blocked: the paper does not uniquely specify native factor clips or the
nu=0.188 fallback, including diagnostic (q,a,Lambda)=(3,0,100) and (5,-0.5,100).

The project continuation uses g(Lambda)=1+0.5*Lambda-0.25*Lambda**2 for
Lambda <= 1, then constant 1.25. Values and first derivatives meet at 1;
the second derivative jumps, so this is **C1**, not C-infinity. Printed
Eq. (26) has the opposite linear sign and jumps from 0.25 to 1.25.
The native +0.81248*omega_plus width choice likewise differs from the
printed minus window in Eq. (29). Sampled continuity and BBH-limit checks
verify these project choices without establishing paper equivalence or
closing full amplitude/phase assembly.

### Separate native SEOBNR development track

The retired generator is developed separately at
[`1f17173e`](https://github.com/xangma/diffgw/tree/1f17173e/dev/native_rom),
with [29 September tabulated-option evidence](https://github.com/xangma/diffgw/blob/aa1f365f/dev/native_rom/evidence/2026-09-29/tabulated_qualification_followup.md)
and a [27 September historical-fidelity decision](https://github.com/xangma/diffgw/blob/aa1f365f/dev/native_rom/evidence/2026-09-27/historical_fidelity_release.json).
Four release gates remain unmet: **source completeness, numerical convergence,
whole-waveform fidelity and independent ROM holdout**.

Both completed default-path campaigns fail the frozen 1e-6 ODE complex-L2
budget: 9.079206e-6 nonspinning and 4.938449e-5 spinning. The nonspinning grid
also fails its complex/decreasing-error gate (6.149855e-6 final); the spinning
grid passes but ODE amplitude L2 1.470160e-6 fails. The flux source audit
covers 31 of 35 eligible rows; Hamiltonian, QNM and initial-data questions
remain. No whole-waveform comparison was rerun at that exact tabulated revision.

The [opt-in continuous NQC derivative diagnostic](https://github.com/xangma/diffgw/blob/1f17173e/dev/native_rom/evidence/2026-09-29/ode_chain_derivatives/README.md)
passes both frozen numerical gates for the spinning row. Nonspinning ODE L2
2.614471e-6 still fails; grid errors 6.220681e-9 to 8.178640e-9 fail the
unchanged decreasing-error check. The
[residual follow-up](https://github.com/xangma/diffgw/blob/88e005003d1f5d85b1ae6ebb9669b5454de38f65/dev/native_rom/evidence/2026-09-29/ode_chain_residual/README.md)
localizes tolerance/match-state and grid-setup effects without justifying a
default rule. The frozen eager/JIT gradient check passes at 3.4370e-11
against 1e-10 for one initial-radius derivative with fixed mesh and attachment
decisions. General physical-parameter derivatives and old NQC sensitivities
do not follow from it.

The recorded SXS:BBH:2569 overlap near 0.98914 remains below 0.99, including
for the pinned historical public model. This physical failure is separate
from historical fidelity. Neither this track nor its evidence changes the
current official external-data ROM contract.

## Recent implementation checks and refit prototypes

New window, precession-boundary and differentiable-solver/NSBH tests exercise
sampled endpoint behavior, selected equal-mass/remnant and coalescing-root
algebra, spheroidal eigenvalues/projections, selected autodiff/JIT paths and
NSBH norm continuity. They add evidence for those cases without replacing the
independent reference, binary or physical campaigns above.

The TaylorF2 Kerr-ISCO cutoff correction now uses the same effective aligned
spin in JAX and Torch. Maintained windowing tests cover six sharp/cosine/Planck
prograde/retrograde parity cases across the actual taper, a selected spin
derivative against a scalar finite difference, and batching/JAX JIT.
All eight regressions failed before the correction; the focused window/Kerr
selection passed 31 tests afterward. A broader local CPU selection passed
630 tests with five skips. This repairs a demonstrated option-path parity
defect without qualifying every cutoff option or its physical accuracy.

The audited **mock-data fitting prototype** and its synthetic XHM amplitude
and XPNR angle coefficient packages were excluded from the release changes on
2026-09-30. They were not integrated into native XHM/XPHM/XPNR waveforms and
provided no completed SXS calibration or independent NR validation. Their
pipeline and integrity tests did not establish scientific fit quality.

The prototype also failed its declared numerical gates: XHM (2,1) v1 training
RMS was 0.367 against a 0.05 limit, and its conditioned log-amplitude
holdout/training RMS ratio was 11.40 against a limit of 3. XPNR B1/B2 ratios
were 5.36/5.34 against the same limit of 3. The source and distribution boundary
checks exclude these tools, packages and public loaders. Real calibration needs
a separately declared source/data contract and independent holdouts before
production integration. See the [refit status](research/phenom-higher-modes.md).

## Reproduction

Use an environment with the repository's reference, Torch, JAX and development
dependencies. These commands cover distinct workflows; no small selection
reproduces all qualification evidence. Dated receipts retain their original
hosts, dependency versions, inputs, gates and results. Expected failures and
unavailable optional dependencies must be reported separately from passes.

~~~sh
# Sampled new implementation and boundary checks.
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_windowing.py tests/test_precession_boundaries.py \
  tests/test_differentiable_solvers_and_nsbh.py

# Release boundary and built-distribution checks.
PYTHONPATH=src python -m pytest -q \
  tests/test_prototype_release_contract.py tests/test_release_distributions.py

# Required mathematical-reference collection; keep the complete globs.
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest \
  tests/test_waveform_definitions.py tests/test_*math_reference*.py \
  tests/test_nsbh_*reference.py \
  tests/test_emri_*reference.py tests/test_xp_msa_remnant_boundary_policy.py \
  tests/test_xphm_rotation_reference.py tests/test_xphm_nnlo_clean_room.py \
  tests/test_spintaylor_reference_angle.py \
  --require-math-reference --reference-device=cpu -ra

# Selected external binary gates; requires the pinned LAL environment.
DIFFGW_REQUIRE_LAL_BINARY=1 PYTHONPATH=src JAX_ENABLE_X64=1 \
  JAX_PLATFORMS=cpu python -m pytest -ra \
  tests/test_imrphenomxas_lal_binary.py \
  tests/test_taylorf2_eccentric_holdout.py tests/test_phenomt_family_lal.py

# Publication checks, separate from waveform qualification.
PYTHONPATH=src python scripts/generate_waveform_library_table.py --check
PYTHONPATH=src python scripts/generate_waveform_coverage.py --check
mkdocs build --strict
~~~

For NR and development-generator campaigns, follow the linked protocols and
version-pinned receipts. Unit tests or documentation builds do not rerun those
campaigns, deploy a waveform, or change a failed gate.
