# IMRPhenomT family: reference validation

This report retains the model-specific reference checks and diagnostic experiments.
For generation, parameters and supported transformations, use the [T-family guide](../imrphenomt.md).
Return to the [validation overview](../validation.md) for other evidence.

`IMRPhenomT` evaluates the aligned-spin dominant `(2,±2)` time-domain model
from Estellés et al. (2021), using the authors' MIT-licensed phenomxpy
calibration and reconstruction release pinned in the provenance ledger.
The family also provides `THM`, `TP` and `TPHM`, with NNLO, MSA and numerical
precession for the latter two. All four have native time-domain evaluation;
there is no direct frequency-domain implementation.

The sampling wrapper places the amplitude peak at `tc`, with a shared fixed
grid for batches and explicit `duration`. It does not promise LAL's automatically
selected epoch or array length. `phic` fixes orbital phase at `f_ref`.
Optional black-box LAL comparisons and Torch/JAX tests accompany the
implementation; LAL is not a runtime dependency.

The [pinned author-reference test](https://github.com/xangma/diffgw/blob/5226141/tests/test_phenomt_reference.py)
checks both frontends for masses `(60, 6)` and aligned spins `(0.9, 0.1)` at
five dimensionless times. Frequency and amplitude use `rtol=1e-11` and
`atol=1e-13`; phase, after subtracting its value at time zero, uses
`rtol=1e-11` and `atol=1e-10`. This is an implementation check using shared
author calibration, rather than independent physical validation.

## Aligned-spin and NNLO reference checks

The optional LAL regression tests define the following acceptance criteria:

| Model | Cases and sampling | Direct complex relative L2 | Mismatch |
| --- | --- | ---: | ---: |
| T | Five mass/spin configurations; 2048 Hz | `< 1e-3` | `< 1e-6` |
| THM | Three configurations; 2048 Hz; five mode pairs | `< 1e-3` | `< 1e-6` |
| TP / TPHM, NNLO | Three configurations; 2048 Hz, except the mass-ratio-10 case at 32768 Hz | `< 1e-2` | `< 1e-4` |

These are **test thresholds**, not measured worst-case errors over the model's
domain. Each comparison uses LAL's returned time grid. Relative L2 applies no
time, phase or amplitude adjustment; mismatch uses an unweighted complex inner
product maximized over constant phase, without a time shift. Exact inputs are
retained in the T reference test and
[family reference tests](https://github.com/xangma/diffgw/blob/main/tests/test_phenomt_family_lal.py).
The NNLO tests select `PhenomXPrecVersion=10210` and `PhenomXPFinalSpinMod=0`,
which differ from LAL's default numerical precession.

NNLO reference agreement is sensitive to sampling, particularly for the rapidly
precessing mass-ratio-10 case. DiffGW anchors ringdown continuation at physical
time `tc`, independently of the requested sampling grid, and uses the published
orbital-angular-momentum series. Grid-invariance tests and the precession
conventions are documented in the [T-family guide](../imrphenomt.md).

## MSA and numerical precession

For the additional precession methods, the same `(40, 15)` generic-spin
case at 2048 Hz gives the following direct complex-strain errors against
LALSimulation 6.2.1. The angle and remnant-spin options are matched explicitly.
The [MSA receipt](../validation/phenomt_precession_20260921.json) and updated
[numerical receipt](../validation/phenomt_numerical_20260922.json) record the
environment and reproduction commands. Use
`python scripts/validate_phenomt_precession.py` for the baseline comparison.

| Method / LAL options | TP relative L2 | TPHM relative L2 | Largest mismatch |
| --- | ---: | ---: | ---: |
| MSA / `22310`, `FinalSpinMod=3` | 0.00379 | 0.00388 | 0.00000754 |
| Numerical / `300`, `FinalSpinMod=4` | 0.0234 | 0.0236 | 0.000274 |

These measurements use the full returned LAL time grid without time, phase,
or amplitude adjustment for relative L2. Numerical precession uses the
PhenomT frequency evolution, fixed intrinsic integration nodes, and a
physical-time ringdown boundary. It now excludes the N3LO quadratic-spin
terms according to the public `phenomtp` API specification, retaining the
leading self-spin terms and N4LO spin-orbit terms. This reduces the baseline
error from about 2.8% to 2.4%; it does not eliminate the reference differences.
The full published spin equations remain available in the internal helper.

The matched-option [CPU/Torch holdout receipt](../validation/phenomt_holdouts_torch_20260925.json)
and [CUDA/JAX holdout receipt](../validation/phenomt_holdouts_jax_len_20260925.json)
repeat the baseline and strongly spinning `(60, 6)` cases for both MSA and
numerical precession. For MSA, TP and TPHM have relative L2 `0.0202` and
`0.0225`, with mismatch `0.000200` and `0.000249`. Both exceed the retained
MSA gates (`0.01` and `0.0001`); the failure is recorded as a strict expected
failure rather than absorbed by a wider tolerance.

A [fresh q=10 CPU receipt](../validation/phenomt_q10_rotation_reference_20260925.json)
repeats all eight TP/TPHM, MSA/numerical, Torch/JAX strain gates after adding
the independent rotation reference. No production waveform equation changed.
Compared with the prior holdout receipts, the largest absolute change is
`1.0e-11` in relative L2 and `1.4e-12` in mismatch. The MSA TP/TPHM L2
remains `0.0202333/0.0225165`, while numerical remains
`0.308418/0.307103`; all four q=10 whole-waveform gates are still red.

The [CPU/Torch MSA component receipt](../validation/phenomt_msa_components_torch_20260925.json)
and [CUDA/JAX rerun](../validation/phenomt_msa_components_jax_len_20260925.json)
compare public co-precessing Euler-angle outputs with native angles for every
supported PV223 suffix. Alpha and gamma are unwrapped and anchored only by
subtracting their first residual; beta is direct. The invariant result below
is the geodesic angle between `R(first)^T R(t)` triads. Each reported segment
is separately anchored at its own first sample, with no optimized frame or
time shift.

| Case | Version | Inspiral max rotation (rad) | Ringdown max rotation (rad) |
| --- | ---: | ---: | ---: |
| `(40, 15)` | 22300 | 0.00461 | 0.00146 |
| `(40, 15)` | 22310 | 0.00921 | 0.0442 |
| `(40, 15)` | 22311 | 0.00665 | 0.0445 |
| `(40, 15)` | 22321 | 0.00690 | 0.0354 |
| `(60, 6)` | 22300 | 0.0205 | 0.0377 |
| `(60, 6)` | 22310 | 0.0300 | 0.154 |
| `(60, 6)` | 22311 | 0.0215 | 0.154 |
| `(60, 6)` | 22321 | 0.0418 | 0.430 |

The JAX receipts were generated on host `len` with Python 3.12.3, JAX 0.11.2,
LALSimulation 6.2.1, FP64, and `cuda:0`. Across the table's component rotation
maxima, the largest absolute Torch/JAX difference is `3.2e-7` rad. Across all
eight baseline/holdout strain measurements, the largest absolute backend
difference is `4.4e-11` in relative L2 and `1.4e-12` in mismatch.

PV22300 uses analytical angles throughout, so its `(60, 6)` inspiral residual
shows that suffix continuation alone cannot explain the MSA holdout failure.
The suffix-dependent ringdown growth separately localizes continuation and
minimal-rotation gamma as follow-up components. These are black-box component
measurements, not equation provenance or physical-accuracy claims. Direct
rotation tests described below already exonerate the shared Wigner/SWSH
projection for binary-supplied modes and angles.

The [independent scalar-angle receipt](../validation/phenomt_msa_scalar_angles_20260925.json)
then tabulates the raw PV223 branch and implements the suffix rules from
PhenomTP Eqs. (5), (20), and (24) with independent scalar PCHIP derivatives
and adaptive quadrature. A 4,120-node table and its 2,065-node refinement
agree within `1.65e-7` rad. At nine reference/MECO/peak/ringdown probes, the
native versus independent maximum rotation is `0`, `1.43e-15`, `7.26e-6`,
and `1.16e-6` rad for suffixes `22300`, `22310`, `22311`, and `22321`.
The relative-rotation residual against the public binary is already `0.01623`
rad in the analytical inspiral for `22300`; its sampled ringdown residual is
`0.03774` rad. A full-grid check restricted to 5,326 samples strictly before
MECO first exceeds `1e-4` rad after `31.738` ms, `1e-3` rad after `0.72949` s,
and `1e-2` rad after `2.55908` s. No continuation, ringdown, or numerical-gamma
rule applies on this interval. The `22311` and `22321` continuations amplify
the sampled ringdown residual to `0.14664` and `0.42952` rad. Thus the earliest
localized q=10 MSA discrepancy is the raw analytical-angle branch, while
continuation/minimal-rotation gamma explains only additional suffix-dependent
growth. The public `22300` branch is an unmodified observation, and no binary
output is fitted.

The clean-room equation assembly now closes the source audit of that raw branch.
For the same q=10 binary on 401 frequencies from 20--220 Hz, production alpha,
gamma, and PN opening beta agree with the independent published-equation path
within `2.95e-10`, `2.51e-10`, and `8.49e-12`, respectively. The paths have
identical reference roots, `c1`, and leading spin average. The first material
split from the separately audited v223 compatibility path is its use of
initial-reference spin contractions in the radiation-reaction coefficients,
instead of the precession-averaged contractions in
[Chatziioannou et al.](https://arxiv.org/html/1703.03967). At q=10 this changes
`beta3` by `-0.00201653`. A later compatibility-only transformation changes
the quadratic nutation coefficient `psi2` from `9.41342` to `7.33672`.
`tests/test_clean_room_msa.py` fixes the common equation path and the location
of this split without treating binary behavior as equation provenance.

Composing that clean-room v223 compatibility path through the independent
source-frame construction reduces the public PV22300 relative-rotation
residual from `0.0162258` rad to `3.32e-6` rad across all 5,326 samples before
MECO, with no `1e-4` crossing. This makes the compatibility branch, beginning
at the flux contractions above, causal for the observed analytical-inspiral
failure rather than merely a nearby algebraic difference. Its fixed initial
Euler-gauge offset is removed by the same first-sample relative rotation used
for both comparisons.

Replacing the production coefficients with those compatibility choices would
therefore not be a source-backed correction. The published-equation
implementation is retained, while the external full-waveform gate remains red.

Additional numerical cases at 2048 Hz give the following errors. Both native
backends are included in the receipt; their reported errors agree to the
displayed precision. The spins, viewing geometry, reference frequency, and
separate inspiral/ringdown metrics are recorded there.

| Numerical case | TP relative L2 | TPHM relative L2 | Largest mismatch |
| --- | ---: | ---: | ---: |
| Single-spin, masses `(35, 20)` | 0.0263 | 0.0264 | 0.000343 |
| Equal masses `(30, 30)` | 0.0174 | 0.0173 | 0.000148 |
| Strongly spinning, masses `(60, 6)` | 0.308 | 0.307 | 0.0475 |

The large 10:1 residual persists at 16384 Hz (TP relative L2 `0.308`, TPHM
`0.307`); see the [higher-rate receipt](../validation/phenomt_numerical_highrate_20260922.json).
This rules out output sampling as its main cause for this case.

## Independent numerical-RHS reference

`diffgw_reference/phenomt_spin_evolution.py` is an 80-digit `mpmath` scalar
translation of [TPHM Eqs. (12)–(14)](https://arxiv.org/abs/2105.05872v2) and
[Sturani Eqs. (3), (21), (22), (24), and (25)](https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf).
It imports neither native backend and exposes the `v5` through `v9` spin
pieces, the nonspinning 3PN orbital angular momentum, and constrained orbital-
axis derivative separately. Tests match the total RHS to Torch and JAX within
`3e-14` relative error. A five-velocity Vandermonde solve recovers every PN
power independently and matches within `3e-10`; the production selection is
also checked to remove only the published `v8` N3LO quadratic-spin piece.

The [independent trajectory receipt](../validation/phenomt_independent_trajectory_20260925.json)
composes this RHS with a SciPy Brent event solve, adaptive DOP853 evolution,
minimal-rotation tangent transport, dense output, and independent Euler-angle
reconstruction. For the `(60, 6)` case, the reference event differs from the
native event by `1.91e-11 M` (`6.21e-15 s`). Tight/loose solves differ by at
most `2.27e-9` in state-vector norm and `1.66e-9` rad in rotation; the largest
constraint drift is `1.11e-12`. Native and independent trajectories differ by
at most `4.72e-9` in orbital-axis norm and `4.78e-9` rad in rotation. The
production angle reconstruction has a fixed `0.003066` rad J-frame chart
offset, but its relative evolution agrees with the independent reconstruction
within `7.96e-16` rad.

The same receipt compares the independent relative triad with rotations
reconstructed from the public PV300/FS4 binary's Euler-angle series for all
43,087 inspiral samples at 16384 Hz. The public binary does not expose its
internal orbital-axis or ODE state.
The initial relative-triad error is `1.38e-16` rad. It then exceeds `1e-4` rad
after `1.465` ms, `1e-3` rad after `14.282` ms, `1e-2` rad after `0.13934` s,
and `1e-1` rad after `1.16638` s, reaching maximum orbital-axis and triad
errors of `0.38911` and `0.39276` rad. The independent carrier event and public
epoch differ by only `1.12`
microseconds; the comparison anchors both rotation series at that defined
reference event, with no fitted time, phase, or rotation adjustment. This localizes the
numerical discrepancy to the pre-merger orientation/rotation path before
projection or ringdown. It does not distinguish public trajectory evolution
from public Euler-angle construction or carrier/time mapping; the small
reference-event offset does not exclude subsequent time-mapping drift.

This closes the native-versus-independent event/ODE/transport/angle
implementation check through the carrier peak, but not the public numerical-
precession model gate. In particular, the
independent published spin-even initial axis rate is `0.0686446799357 1/s` and
matches native to `8.33e-17 1/s`; the public angle-derived axis rate is
`3.27695843871e-8 1/s`. The positive SpinTaylor quadrupole control remains
present in the receipt. Published math therefore supports retaining the Kerr
self-quadrupole coefficient, while the black-box orientation discrepancy and
whole-waveform gate remain unresolved. No LAL source or binary-derived
coefficient enters the reference path.

The component localization is therefore:

| Failed q=10 path | Earliest paper-defined component | Evidence excluding later components | Remaining blocker |
| --- | --- | --- | --- |
| MSA PV223 | Public-binary raw analytical angles differ from the source-backed native/clean-room equation path | Production and independent equations agree below `3e-10`; the clean v223 compatibility path reduces the full pre-MECO relative residual from `0.0162258` to `3.32e-6` rad | The first compatibility substitution (initial-reference flux contractions, followed later by a `q^2` phase convention) is not stated by the published equations, so no source-backed production correction is authorized |
| Numerical PV300/FS4 | Pre-merger orientation/rotation path: carrier/time mapping, trajectory, or public angle construction unresolved | Relative-triad error reaches `1e-4` rad after `1.465` ms; projection closes below `8e-16`; ringdown is not sampled | Public Euler angles do not expose the internal trajectory, so the earliest of these three components cannot yet be selected |

The primary sources support the retained Kerr self-quadrupole term and the
implemented evolution equations. They do not support a narrow production
correction inferred from the public binary, so this slice changes diagnostics
and receipts only.

## Carrier and integration-resolution diagnostics

The [component receipt](../validation/phenomt_components_20260922.json) separates
the dominant co-precessing mode from the projected TP polarizations. The native
mode uses its own evolved remnant spin. Since the co-precessing API has an
independent phase convention, this diagnostic applies one constant phase rotation
fixed by the first mode sample, with no time shift or amplitude scaling. The
polarization comparison remains unaligned. Both Torch and JAX give these results:

| Numerical case | Co-precessing `(2,2)` relative L2, initial phase anchored | TP polarization relative L2, unaligned |
| --- | ---: | ---: |
| Baseline `(40, 15)` | 0.000138 | 0.0234 |
| Single-spin `(35, 20)` | 0.00000515 | 0.0263 |
| Equal masses `(30, 30)` | 0.000210 | 0.0174 |
| Strongly spinning `(60, 6)` | 0.000314 | 0.308 |

For the 10:1 case, doubling the intrinsic integration steps from 2048 backward /
1024 forward to 4096 / 2048 changes the native TP strain by relative L2
`2.32e-9`, while its reference error remains `0.308`; see the
[resolution receipt](../validation/phenomt_resolution_20260922.json). The opening
angle differs from the reference by about `0.0105` radians near the peak at both
resolutions. This focuses the remaining investigation on precession evolution
and frame/phase conventions: neither the dominant-mode shape nor integration
resolution explains the large residual. These diagnostics do not identify a
specific missing physical term or establish higher-mode carrier agreement.

## Quadrupole sensitivity

A separate [quadrupole sensitivity receipt](../validation/phenomt_quadrupole_20260922.json)
varies only the native self-quadrupole coupling in the spin equations. Setting
both coefficients to zero instead of their black-hole value of one reduces TP
relative L2 from `0.308` to `0.0181` for the 10:1 case and from `0.0234` to
`0.00364` for the baseline. Torch and JAX reproduce this sensitivity. This
experiment directs attention to the treatment of quadrupole coupling, but does
not establish which equations the reference uses. Production retains the
black-hole coupling: suppressing a physical term solely to improve reference
agreement would change the model. The mathematical basis is the orbit-averaged
spin equation in [Sturani, Eq. (21)](https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf)
and the [quadrupole-monopole contribution for Kerr black holes](https://arxiv.org/abs/0803.1820).

Reproduce the component check with
`python scripts/diagnose_phenomt_precession.py --output receipt.json`.
Add `--backend torch --case high_mass_ratio --backward-steps 128 512 2048 4096`
for the integration-resolution check. Both scripts use LAL binary outputs only;
no implementation source or fitted production correction is used.
For the separate sensitivity experiment, use
`--case baseline high_mass_ratio --quadrupole-kappa 1 0`.
The zero-coupling option exists only in the diagnostic script, not the waveform
API. It leaves the reference settings unchanged.

## Source-frame trajectories

A [source-frame trajectory receipt](../validation/phenomt_source_frame_20260922.json)
compares the orbital axis and transported tangent after removing the constant
reference rotation, using [PhenomTP Eq. (8)](https://arxiv.org/html/2105.05872).
It uses all inspiral samples at 2048 Hz and also checks elapsed time from each
model's reference frequency. The reference-time offsets are `-0.34 ns` for the
baseline and `1.12 microseconds` for the 10:1 case. Accounting for these offsets
leaves maximum orbital-axis differences of `0.0728` and `0.389` radians,
respectively, with both frontends agreeing. Suppressing the native quadrupole
term for diagnosis reduces those maxima to `0.0352` and `0.0635` radians.
Thus constant frame choices and reference-time offsets do not explain the
trajectory drift, and the zero-coupling experiment still leaves a measurable
evolution difference. Reproduce with the sensitivity command above; the receipt
records both clock conventions and initial-triad checks.

## Rotation and initial-rate diagnostics

Independent tests and binary diagnostics isolate the rotation and initial
spin response:

`diffgw_reference/phenomt_rotation.py` now constructs each degree-2 through
degree-5 Wigner small-d matrix by exponentiating the angular-momentum
generator at 60-digit precision. The strict
`tests/test_phenomt_rotation_math_reference.py` compares this independent
construction with both native backends, then checks signed-mode completion,
the spin-weighted-harmonic sum, observer polarization phase, and the fixed
inverse reference-frame rotation of [TPHM Eqs. (4), (8), and (9)](https://arxiv.org/html/2105.05872).
The generator calculation does not use the native factorial-sum algorithm.
It qualifies the mode rotation for supplied modes and Euler angles, but does
not establish the accuracy of the carrier or angle evolution.

- The [rotation receipt](../validation/phenomt_rotation_20260922.json) supplies
  binary co-precessing modes and Euler angles to the native `project_modes`,
  then compares against binary inertial modes projected with the reference
  spherical harmonics. It covers both frontends, all four cases, three viewing
  directions, and both the dominant-mode and full higher-mode selections.
  Relative L2 is below `8e-16`. This checks rotation conventions independently
  of the native carrier and precession evolution; it does not establish full
  waveform agreement. The inverse J-to-L0 frame check, for the three cases
  with `f_ref=f_lower=20 Hz`, has a separate higher-mode residual reaching
  `1.3e-9` in the full projected strain. Per-degree checks localize this
  residual to ℓ=5. A [reference-sample receipt](../validation/phenomt_rotation_reference_20260922.json)
  checks the baseline and 10:1 cases at 512, 2048, and 8192 Hz and distances
  of 50, 100, and 200 Mpc. At the reference sample, Eq. (8) requires the
  co-precessing and L0 modes to coincide. The binary L0 output instead contains
  ℓ=5, m=±2 components with amplitudes `1.44e-32` and `5.98e-31` at 100 Mpc,
  respectively, while those co-precessing components are zero. Inverting the
  binary J modes gives amplitudes below `2e-40` in these components. Native
  rotation matrices agree with an independent 80-digit factorial evaluation
  to relative error below `3e-16`; applying either matrix reproduces the binary
  J modes below `3e-16`. The residual is independent of sampling rate and its
  absolute size scales inversely with distance. These checks identify an
  inconsistency among the tested public frame outputs, without identifying
  its internal cause. Full-mode inverse results remain measurements, not a
  passing accuracy gate, and no native correction is introduced.
- The [initial-rate receipt](../validation/phenomt_initial_rates_20260922.json)
  reverses a single tilted spin at fixed masses and reference frequency. The
  companion spin is set to zero, and each run starts at `f_lower=f_ref`.
  Averaging the rates for opposite spins cancels the native spin-orbit terms
  and retains the self-quadrupole contribution of Sturani Eq. (21). The binary
  rate is differentiated from its returned orbital-axis angles and rotated
  back into the common initial source frame. The full initial Euler triad
  agrees with this frame mapping to below `5e-16`; the initial carrier
  frequency differs from `f_ref` by less than `3e-7` relative.

At 16384 Hz, the norms of these **spin-even initial orbital-axis rates** are:

| Single-spin configuration | Native black-hole rate (1/s) | PhenomTP binary rate (1/s) |
| --- | ---: | ---: |
| Masses `(35, 20)`, spin1 `(0.3, 0, 0.4)`, reference 30 Hz | 0.0447 | 0.000000202 |
| Masses `(60, 6)`, spin1 `(0.2, 0.1, 0.8)`, reference 20 Hz | 0.0686 | 0.0000000328 |

The receipt includes two sampling rates, two reference phases, two derivative
stencils, and controls with the spin in the orbital plane, where the native
self-quadrupole torque vanishes. It also includes a positive control using the
public `SpinTaylorT4` binary with explicit quadrupole parameters zero and one.
With one, its spin-even **individual-spin** derivative agrees with the native
published term to relative error below `5e-8` at 16384 Hz; with zero, that
response is suppressed. This control uses `spinO=5` and compares only the
initial spin-even derivative, not the full SpinTaylor and PhenomTP dynamics.

These results support a strongly suppressed self-quadrupole response in the
tested PhenomTP binary output. They do not uniquely identify its internal
quadrupole settings or equations: finite differences, frequency anchoring,
axis definitions, and spin-dependent angular-momentum denominators can affect
the inferred rates. The literature audit found no instruction in
[PhenomTP Sec. II.4.1](https://arxiv.org/html/2105.05872) to suppress the
black-hole quadrupole. Native production therefore retains it. This is a
documented reference difference, not a correction fitted to binary output.

The initial-rate receipt also separates the **spin-odd response** across three
mass ratios, reference frequencies of 20, 30, and 40 Hz, and two spin amplitudes.
An independent evaluation of [Sturani Eqs. (3), (5), (22), and (25)](https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf)
agrees with both native frontends below `9e-16` relative. No transcription error
was found in the `v^9` spin-orbit coefficient. Binary and native rate directions
agree within `1.8e-8` radians, but their magnitudes differ by `0.024%`–`0.959%`.
Doubling the spin amplitude preserves the binary odd-rate scaling within
`2.8e-7` relative; the receipt retains frequency anchoring and derivative-stencil
sensitivity alongside these measurements. Using the published nonspin 2PN
angular momentum instead of 3PN reduces the rate discrepancy, but leaves up
to `0.155%`. Thus changing that truncation alone does not explain the reference
behavior. These are diagnostic evaluations of published terms, with no fitted
coefficients or production changes.

Reproduce these checks with
`python scripts/diagnose_phenomt_rotation.py --output rotation.json` and
`python scripts/diagnose_phenomt_initial_rates.py --output initial-rates.json`.
Add `--reference-check` to the rotation command for the per-mode reference
controls, or `--odd-rate-sweep` to the initial-rate command for the published-order
sweep. The latter uses 16384 Hz and the selected frontends independently of the
main diagnostic's sampling-rate selection. Each receipt records its exact command.
All reference calls use public binaries; no reference implementation source
is read or incorporated.

Reproduce the new independent paths with
`python scripts/diagnose_phenomt_msa_scalar_angles.py --output msa-scalar.json`
and `python scripts/diagnose_phenomt_independent_trajectory.py --output trajectory.json`.
Their receipts record exact options, versions, tables, and residuals.
The committed receipts record their execution host, working directory, exact
command, and package versions. They were regenerated on `len` with Python 3.12.3, NumPy 2.5.3,
SciPy 1.18.1, Torch 2.14.0+cu130, LALSuite 7.26.15 / LALSimulation 6.2.1; the JAX
verification environment is recorded separately above.

## Regression scope and unresolved cases

The MSA and numerical regression checks are retained in
`tests/test_phenomt_family_lal.py`. Numerical checks cover the baseline,
single-spin, and equal-mass cases with relative L2 below `0.03` and mismatch
below `0.0005`. The 10:1 MSA case also exceeds its `0.01`/`0.0001` gates.
Both high-mass-ratio failures remain strict expected-failure tests, rather
than widening either regression tolerance. These checks
do not establish reference agreement throughout parameter space or physical
accuracy against numerical relativity. NNLO remains the default. See the
[precession options and numerical domain](../imrphenomt.md#higher-modes-and-precession).
