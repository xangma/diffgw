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

Two further diagnostics isolate the rotation and the initial spin response:

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

## Regression scope and unresolved cases

The MSA and numerical regression checks are retained in
`tests/test_phenomt_family_lal.py`. Numerical checks cover the baseline,
single-spin, and equal-mass cases with relative L2 below `0.03` and mismatch
below `0.0005`. The 10:1 case does **not** meet those bounds and is recorded
as a limitation, rather than widening the regression tolerance. These checks
do not establish reference agreement throughout parameter space or physical
accuracy against numerical relativity. NNLO remains the default. See the
[precession options and numerical domain](../imrphenomt.md#higher-modes-and-precession).
