# Accuracy and validation

DiffGW provides native tensor waveform models and inference tools. Each result
below supports a particular implementation, configuration, and numerical check.
A model name or available API does not establish accuracy throughout its
parameter space.

The [mathematical-reference CI check](math/taylorf2.md#reproduce-the-checks)
compares the covered equations and derivatives against each frontend, audits the
shared unit conversions independently, and fails if required checks are skipped
or omitted. Its numerical coverage and precision limits are stated in the
specification. The [component map](math/taylorf2.md#component-traceability)
connects the published sources, expressions, implementations, and tests. The
[aligned-spin extension](math/taylorf2-aligned-spin.md) composes that baseline
with separately sourced spin corrections and checks the historical frontend
option groups explicitly.
The [quadrupolar tidal extension](math/taylorf2-tidal.md) adds the 5PN, 6PN,
and 6.5PN phase terms, with exact equivalence between published parameterizations
and independent tidal/mass derivative checks for the composed waveforms.
The [spin-induced quadrupole extension](math/taylorf2-spin-quadrupole.md)
adds the non-Kerr 2PN, 3PN, and 3.5PN phase terms, with source-linked coefficient,
waveform, derivative, batching, and compiled-path checks.

On 24 September 2026, the `1315f6e` integration snapshot passed the strict
CPU gate on `len`: **1,097/1,097** required mathematical-reference cases,
zero skipped/xfail/deselected, and **1,109** tests passed overall. The fresh
Python 3.12.3 environment used NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0,
Torch 2.14.0, JAX/JAXLIB 0.11.2, and pytest 9.1.1. After installing
LALSuite 7.26.15 in that environment, the maintained IMRPhenomD and TaylorF2
binary-oracle tests passed **36/36**. Reproduce from that snapshot with
`PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1` and
`python -m pytest tests/test_waveform_definitions.py tests/test_*math_reference*.py --require-math-reference --reference-device=cpu -ra`, then
`python -m pytest -q tests/test_imrphenomd_lal_gradients.py tests/test_taylorf2_lal_quadrupole.py tests/test_taylorf2_lal_tidal.py -ra`.
These checks establish the stated implementation comparisons at tested points;
they do not certify the provisional waveform families or physical accuracy.

On 23 September 2026, the development tree was copied to `len` and the strict
math-reference gate passed **618/618** tests on CPU and **618/618** on the RTX
4090 CUDA device, with no skips, xfails, or deselections. The run used Python
3.12.3, Torch 2.5.0.dev20240619, JAX 0.4.29, and SymPy 1.12. The Torch
Inductor dynamic-batch and gradient regression also passed. These results
qualify implementation agreement with the versioned mathematical references;
they do not establish LAL agreement or physical accuracy.

After adding the IMRPhenomD remnant/QNM-input specification on the same date,
the complete strict gate passed **679/679** on `len` CPU, and its focused
remnant/QNM module passed **34/34** on the RTX 4090 CUDA device. The run used
Python 3.11.9, Torch 2.13.0+cu130, JAX 0.4.20, SymPy 1.14.0, and SciPy 1.13.0.
This older JAX/SciPy pairing required process-local `scipy.linalg.tril/triu`
aliases during test startup. The new checks qualify printed remnant equations,
agreement with explicitly supplied inherited digits, QNM interpolation and
mass scaling on selected interior spins, and selected derivatives. They do not
qualify endpoint QNM parity or full IMRPhenomD physical accuracy.

After adding the [IMRPhenomD phase specification](math/imrphenomd-phase.md),
the strict CPU gate passed **703/703** on `len`, with no skips, xfails or
deselections; the new phase module contributes **23** tests. This run used
Python 3.12.3, Torch 2.5.0.dev20240619, JAX 0.4.29, SymPy 1.12 and SciPy
1.13.1. It checks published phase expressions, conditional agreement when
inherited fit digits are supplied explicitly, $C^1$ joins, selected unwrapped
Torch/JAX phases, reference-frequency anchoring and frequency gradients.
The focused phase module also passed **23/23** on `len`'s RTX 4090 CUDA device.
These runs do not establish those digits' provenance or full-domain physical
accuracy.

The [composed IMRPhenomD polarization specification](math/imrphenomd-waveform.md)
then passed **10/10** focused FP64 tests on both `len` CPU and its RTX 4090
CUDA device. The complete strict CPU gate passed **713/713**, with no skips,
xfails or deselections. The new checks compare absolute complex plus and cross
strain and unwrapped phase against the scalar reference across selected binary
parameters, both phase conventions, branch joins and the default cutoff. They
use explicitly supplied inherited fit rows and a declared QNM pair, so this
agreement does not establish those inputs' provenance or physical accuracy.

The same composed specification now checks FP64 complex-polarization
derivatives with respect to frequency, distance, inclination, coalescence time,
coalescence phase, and a common scaling of both masses at fixed mass ratio,
spins, fit rows and dimensionless QNM pair. Interior points cover the three
amplitude/phase regions; the reference-frequency convention is also sampled.
On `len`, the focused module passed **34/34** on CPU and **34/34** on the RTX
4090 CUDA device. The complete strict CPU gate passed **737/737**, with no
skips, xfails or deselections. An initial concurrent CUDA run failed four
forward-value cases from GPU out-of-memory while the CPU gate held about 18 GiB
of GPU memory; the isolated CUDA rerun passed all 34. These derivatives do not
include the response of the remnant/QNM pipeline or establish generic
mass-ratio or spin gradients.

The [full intrinsic IMRPhenomD composition](math/imrphenomd-intrinsic.md)
extends that fixed-QNM slice with the published remnant expressions and a
declared numerical QNM spline. It compares default-path complex polarizations
and full-chain derivatives with respect to mass ratio at fixed total mass and
both aligned spins. The remnant/QNM tangents use symbolic differentiation and
the natural-spline slope; full complex-strain derivatives use a fourth-order
scalar stencil independent of Torch/JAX autodiff, guarded to stay in one QNM
spline cell and one waveform branch. Selected smooth points cover inspiral,
intermediate, merger-ringdown, both sides of the moving joins, the
reference-frequency convention, equal and unequal masses, and positive and
negative spins. Explicit crossing steps and spin/domain endpoints have separate
policy checks. The comparisons condition on inherited extra fit digits and
supplied QNM data. At that stage, source provenance, endpoint/clamp parity,
physical accuracy, and full-domain derivative error remained separate questions.

On `len`, the expanded focused polarization module passed **98/98** FP64
checks on CPU and **98/98** on the RTX 4090 CUDA device. The complete strict
CPU math-reference gate passed **801/801**, with no skips, xfails, or
deselections; all eight generated-artifact and traceability module checks passed.
The runs used Python 3.11.9, Torch 2.13.0+cu130, JAX 0.4.20, SymPy 1.14.0,
and SciPy 1.13.0. This older JAX/SciPy pairing required process-local
`scipy.linalg.tril/triu` aliases at test startup, as in the earlier remnant
campaign. The 64 intrinsic-composition cases cover the default remnant/QNM
chain, including $q$, $\chi_1$, and $\chi_2$ derivatives for the primary binary,
equal-mass and negative-spin cases, near-join samples, and stencil boundary
policy. They do not provide a uniform parameter-space error bound.

The subsequent remnant/QNM endpoint checks found two implementation defects:
compiled Torch silently used the rounded spline above $|a_f|=0.994$ while eager
Torch used generated roots, and JAX's natural-cubic tridiagonal solve used the
wrong lower-diagonal width adjacent to the nonuniform guard knots. Both were
corrected. At that stage the checks documented Torch's generated continuation,
JAX's rounded continuation, and their common $|a_f|=0.999$ remnant clamp. At
$a_f=0.995$, the generated and rounded cyclic QNM values differ by about
$1.5\times10^{-5}$ in the real channel and $3.7\times10^{-6}$ in damping.
The extreme mass-ratio clamp case checks implementation behavior, not physical
accuracy. The JAX continuation policy was subsequently revised as described
below.

On `len`, the complete QNM/remnant test files passed **65/65** on CPU and
**65/65** on the RTX 4090 CUDA device. The full strict CPU math-reference gate
passed **821/821** with no skips, xfails, or deselections. Direct QNM
`torch.compile` checks compared endpoint values and spin gradients with eager
Torch using `aot_eager` on CPU and CUDA; a separate default-Inductor CUDA probe
also matched eager values and gradients at $a_f=\pm0.995$. These checks do not
qualify a compiled full waveform at those extreme parameters.

The next high-spin audit found that JAX's rounded-spline continuation missed
the same-lineage positive-spin damping solves by up to about 1.0% near
$a_f=0.9986$. Both frontends now use the generated degree-5 Kerr spline for
$|a_f|>0.994$; they retain the natural cubic of six-significant-digit
ordinates inside that interval. The maintained endpoint checks compare all
93 stored high-spin eigenproblem holdouts and stricter solves with both
frontends, and two $|a|=0.9999$ rows from
[Lo, Sabani, and Cardoso's generalized Sasaki–Nakamura dataset](https://the-center-of-gravity.com/data-and-routines/)
provide a selected independent-method input check. The author's $a=0.9976$
damping row disagrees with its neighbors and the continued-fraction solve, so
the independent-method claim is limited to those
two near-extremal rows. Those checks validate the sampled QNM input and
continuation policy; by themselves they do not establish full-waveform
endpoint agreement or physical accuracy for extreme-mass-ratio binaries.

On `len`, the finalized remnant/QNM module passed **63/63** FP64 tests on CPU
and **63/63** on the RTX 4090 CUDA device. This includes JAX JIT values and
spin gradients on both high-spin branches, with gradients checked against a
fourth-order finite difference. The complete strict CPU math-reference gate
passed **829/829**, with no skips, xfails or deselections. These runs used the
Python 3.11.9, Torch 2.13.0+cu130, JAX 0.4.20, SymPy 1.14.0, and SciPy 1.13.0
environment described above, including the process-local SciPy compatibility
aliases.

The [full intrinsic composition](math/imrphenomd-intrinsic.md) now also tests
the **complete complex waveform** at high remnant spin. Its scalar reference
receives the generated Kerr spline as declared numerical data, evaluates it
independently with SciPy, applies the $|a_f|=0.999$ remnant cap and published
mass scaling, and composes the published amplitude and phase equations. Direct,
unaligned FP64 comparisons on `len` passed **12/12** focused cases on CPU and
**12/12** on CUDA, covering Torch/JAX, both phase conventions, both sides of
all moving joins, the cutoff, positive and negative high-spin continuations,
and an active positive-spin clamp. These extreme-mass-ratio cases check
implementation consistency; independent physical accuracy still depends on
the separate QNM evidence and waveform calibration.
The expanded waveform-reference module passed **110/110** on CUDA. The strict
CPU math-reference gate passed **841/841**, with no skips, xfails, or
deselections.

The subsequent high-spin derivative extension checks the real and imaginary
parts of both complex polarizations against an independent fourth-order scalar
stencil. At fixed total mass it varies $q$, $\chi_1$, and $\chi_2$ through
the positive and negative generated Kerr splines and an interior-spin point
on the active positive remnant cap. The reference guards QNM knots, the
$|a_f|=0.994$ interpolation switch, the $|a_f|=0.999$ cap transition,
moving waveform joins, and the hard cutoff; an explicit cap-crossing stencil
is rejected. A smaller starting step for generated-spline cases was checked
by halving it on all three high-spin branches. The cap case measures a
derivative within the capped region, not at its nonsmooth edge. These tests
establish sampled implementation agreement, not independent physical accuracy
at extreme mass ratios or a parameter-space-wide derivative bound.
The expanded waveform-reference module passed **168/168** FP64 tests on the
`len` RTX 4090 CUDA device, including Torch/JAX high-spin complex gradients
for both phase conventions and all three intrinsic parameters. The focused
high-spin derivative and cap-guard slice passed **55/55** on CPU; its three
additional independent stencil-convergence checks also passed on CPU.
The strict CPU mathematical-reference gate passed **899/899**, with no skips,
xfails, or deselections. Its CPU process ran with the GPU hidden while the
isolated CUDA suite used the device, avoiding a JAX/cuBLAS allocation conflict
between the concurrent processes.

### IMRPhenomD intrinsic derivatives against LALSuite (23 September 2026)

The optional [binary-oracle test](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomd_lal_gradients.py)
finite-differences the public LALSimulation `IMRPhenomD` FD waveform and
compares the resulting complex plus/cross slopes directly with Torch autograd
and JAX forward-mode autodiff. On `len`, **18/18** CPU FP64 cases passed with
LALSuite **7.21**. The three binaries are
$(36,24,0.2,-0.3)$, $(45,15,0.45,-0.25)$, and
$(36,9,-0.6,-0.4)$ in $(m_1/M_\odot,m_2/M_\odot,\chi_1,\chi_2)$.
At 40, 120, and 280 Hz, it varies $q=m_1/m_2$ at fixed total mass and each
aligned spin, with $f_{\rm ref}=20$ Hz, 400 Mpc, inclination 0.7, and zero
reference phase. Neither waveform nor derivative is phase, time, or amplitude
aligned. Relative to the LAL norm, the maximum direct waveform error was
$1.35\times10^{-12}$ and the maximum full complex derivative error was
$4.95\times10^{-10}$. Halving the fourth-order stencil step changed the LAL
derivative by at most $2.07\times10^{-8}$ relative to its norm; the maintained
thresholds are $10^{-9}$ for waveform values and $10^{-6}$ for derivatives and
stencil convergence. These are sampled same-model implementation comparisons,
not independent numerical-relativity validation or a bound near the high-spin
continuation, cap, or join transitions.

### TaylorF2 electric tides against LALSuite (23 September 2026)

The optional [tidal binary-oracle test](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_lal_tidal.py)
compares unaligned complex plus/cross polarizations for equal-mass
$(1.4,1.4,\Lambda_1=\Lambda_2=400)$ and unequal-mass
$(1.7,1.2,\Lambda_1=800,\Lambda_2=200)$ **nonspinning** binaries. Both
frontends and LALSuite 7.21 use the same 30–500 Hz sequence at 0.5 Hz spacing,
100 Mpc, inclination 0.3, restricted amplitude, phase order 7, spin order 7,
and tidal selectors 10, 12, and 13 (5PN, 6PN, and 6.5PN). LAL's
`dQuadMon_i=0` explicitly retains Kerr quadrupoles, and DiffGW uses
`phic=π/2` to match LAL's zero reference-phase carrier. The **12/12** CPU FP64
checks passed on `len`; maximum complex relative L2 was
$3.37\times10^{-12}$ and maximum unwrapped phase difference was
$9.43\times10^{-12}$ rad. The separate
[paper-based reference](math/taylorf2-tidal.md) verifies the tidal equations,
spinning composition, and derivatives against both frontends.

A diagnostic with nonzero spins $(0.15,-0.08)$, the unequal-mass deformabilities
above, tidal selector zero, and explicit `dQuadMon_i=0` gave a complex relative
L2 difference of **1.1766** from DiffGW's restricted spin-plus-tide model.
With zero deformabilities the same spinning configuration agreed to about
$3.1\times10^{-12}$. Thus this LAL binary call contains an additional
deformability-dependent spin effect even when its tidal PN selector is zero;
the nonspinning result must not be promoted to a spinning-tidal same-model
claim. This diagnostic does not identify the additional term or assess its
physical accuracy.

### TaylorF2 spin-induced quadrupoles against LALSuite (23 September 2026)

On `len`, the installed **LALSuite 7.21** TaylorF2 binary independently agreed
with the [paper-based quadrupole reference](math/taylorf2-spin-quadrupole.md)
at 2PN and 3PN. The maximum absolute error in a quadrupole coefficient was
`5.7e-14` across Kerr, BH–NS, and NS–NS cases. Both DiffGW frontends were then
compared with the binary's complex plus and cross polarizations on 1,081
frequencies from 30 to 300 Hz at 0.25 Hz spacing:

| Case | \(m_1,m_2\) (solar masses) | \(\chi_1,\chi_2\) | \(C_{Q1},C_{Q2}\) | Max complex relative L2, both frontends/polarizations |
| --- | --- | --- | --- | ---: |
| Kerr | 1.7, 1.2 | 0.4, −0.2 | 1, 1 | `3.1e-12` |
| BH–NS | 10, 1.4 | 0.3, 0.25 | 1, 5 | `8.3e-13` |
| NS–NS | 1.7, 1.2 | 0.4, −0.2 | 5, 2.5 | `3.1e-12` |

The largest absolute unwrapped phase difference, computed from the ratio of
complex polarizations, was `9.0e-12` rad; the largest relative amplitude
difference was `3.0e-15`. Relative L2 means the unweighted Euclidean norm of
the complex difference divided by the LAL norm. No fitted time or phase shift
was used. The inputs use `PNPhaseOrder=7`, `PNAmplitudeOrder=0`,
`PNSpinOrder=7`, `PNTidalOrder=0`, and LAL's `dQuadMon_i=C_Qi−1`; the DiffGW
options are `spin_order=7`, `tidal_order=0`, coalescence phasing, and no ISCO
cutoff. Both use 100 Mpc and inclination 0.3 rad. A fixed `phic=π/2` in DiffGW
matches LAL's `phiRef=0` carrier convention. The LAL sequence uses `f_ref=0`.
The [LAL parameter documentation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_8h.html)
defines the quadrupole-monopole coupling as the full coefficient diminished by one.
The Torch comparison used Python 3.11.9 and Torch 2.1.1; JAX 0.4.29 used
the same LAL-generated binary arrays in a separate Python 3.12.3 environment.
The [maintained optional binary-oracle test](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_lal_quadrupole.py)
now runs both frontends and passed **6/6** cases on `len` with LALSuite 7.21.
Together with the tidal and IMRPhenomD derivative oracles, the three focused
files passed **36/36** CPU tests. The maintained binary-oracle CI job installs
current LALSuite and PyCBC releases and runs all three files alongside strict
frozen catalog and holdout comparisons. These counts refer to the dated
campaign above, not future CI runs.

This LALSuite binary returns **zero** non-Kerr 3.5PN coefficient at its highest
accepted spin selector and rejects selector 8. Consequently, this comparison
qualifies only the 2PN and 3PN non-Kerr terms and their composed restricted
waveforms. The 3.5PN tail and cubic-spin terms retain their independent
paper-equation and SymPy/Torch/JAX checks, but have no same-model LAL binary
check in this campaign. These sampled numerical agreements do not establish
physical accuracy for neutron-star systems.

If you are replacing a LAL waveform, start with [Differences from LAL](differences_from_lal.md).
It identifies intentional modeling choices, unresolved discrepancies and their
practical implications. The reports below provide the underlying evidence.

## What each check establishes

| Check | Evidence | Scope |
| --- | --- | --- |
| Independent mathematical reference | [Versioned specifications](math/index.md), [nonspinning TaylorF2](math/taylorf2.md), [aligned-spin TaylorF2](math/taylorf2-aligned-spin.md), [quadrupolar tides](math/taylorf2-tidal.md), [spin-induced quadrupoles](math/taylorf2-spin-quadrupole.md), [IMRPhenomD amplitude](math/imrphenomd-amplitude.md), [remnant and QNM inputs](math/imrphenomd-remnant-qnm.md), [phase and joins](math/imrphenomd-phase.md), [composed polarizations](math/imrphenomd-waveform.md), [full intrinsic composition](math/imrphenomd-intrinsic.md) | TaylorF2 point-particle and aligned-spin black-hole phase through 3.5PN, 5PN/6PN/6.5PN electric tides, non-Kerr spin-induced quadrupole terms through 3.5PN, and restricted amplitude. IMRPhenomD covers scaled amplitude and phase branches, printed remnant fits, continuity joins, QNM mass scaling, selected full-chain complex-polarization mass-ratio/spin derivatives, and complete complex-waveform samples at high remnant spin. The scalar equations are independent of both frontends; inherited extra fit digits and QNM data are supplied explicitly. Unconditioned full-domain waveform agreement, other waveform families, and physical accuracy remain outside these specifications. |
| Waveform performance and LAL agreement | [FP64 package benchmarks](package_benchmarks.md), with committed CPU/CUDA receipts | Forward evaluation of TaylorF2, IMRPhenomD and IMRPhenomXAS on declared grids and binary samples. Both polarizations are checked; failed configurations remain visible. |
| PyTorch/JAX consistency | [Parity tests](https://github.com/xangma/diffgw/tree/main/tests/parity) | Agreement between two implementations for specified inputs and per-test tolerances. Shared coefficients or shared mistakes can produce agreement. |
| Time-domain performance and PhenomXPy agreement | [T/THM CPU comparison](package_benchmarks.md#phenomxpy-time-domain-comparison-22-september-2026), with a pinned author reference and receipt | Both polarizations on a fixed time segment; every binary checked, with strict-gate failures retained as diagnostic timings. Shared author fits mean this is not independent physical validation. |
| Broader model reference checks | [Differences from LAL](differences_from_lal.md#model-specific-differences), [manifest](model_manifest.json), linked archived runs, and CI `parity-oracle-report` artifacts | The dated archive has 17/21 passing catalog cases and 11/13 holdouts. CI gates the maintained frozen cases separately from whole-catalog coverage, which remains false while higher-mode oracle and model coverage gaps are open. The reports distinguish exact LAL comparisons from sovereign-calibration cases. New models have separate evidence, including the [T-family report](validation/phenomt.md). |
| Derivatives and Fisher matrices | [Inference parity tests](https://github.com/xangma/diffgw/blob/main/tests/parity/test_inference_parity.py), [Fisher example](../examples/02_fisher_analysis.py) | Selected parameters, models and derivative orders. Waveform-value agreement alone does not validate derivatives or posterior uncertainties. |
| Relative-binning likelihood | [Numerical tests](https://github.com/xangma/diffgw/blob/main/tests/test_relative_binning.py) and inference parity tests | [Local approximation and gradient checks](validation/inference.md#relative-binning). There is no published DiffGW timing campaign establishing a general likelihood speedup. |
| Physical calibration and provenance | [Provenance ledger](provenance_ledger.md) and [third-party notices](../THIRD_PARTY_NOTICES.md) | Literature, calibration sources, component checks, and unresolved asset provenance. A reference implementation check is distinct from independent physical validation. |

## Waveform agreement and physical validation

Start with [execution restrictions](models.md#execution-restrictions) for the
configuration you intend to use. The [differences from LAL](differences_from_lal.md#model-specific-differences)
summarize deliberate model differences and unresolved reference failures.
The [PhenomT report](validation/phenomt.md) contains its measured errors,
regression thresholds and component diagnostics.

The [dated FP64 campaign](package_benchmarks.md#numerical-qualification) checks
both polarizations of every timed binary, with direct errors and an explicitly
defined mismatch. Its strict thresholds differ from the model-specific catalog
and precession checks. Each threshold belongs to its stated reference and
configuration; a passing label from one check cannot qualify another.

Independent physical validation requires an appropriate physical reference and
coverage statement. The XHM/XPHM NR checks described in the LAL-differences page use a
restricted ringdown phase band and reference amplitudes; they do not establish
full-waveform amplitude and phase accuracy. Calibration provenance is tracked
in the [ledger](provenance_ledger.md), separately from implementation agreement.

## Derivative support and evidence

| Path | Documented support or check | Scope and missing evidence |
| --- | --- | --- |
| Torch/JAX waveform derivatives and Fisher matrices | [Fisher tutorial](tutorials/02_fisher_analysis.md) and [inference parity tests](https://github.com/xangma/diffgw/blob/main/tests/parity/test_inference_parity.py) | Selected models, parameters and orders. Cross-framework agreement is not an independent derivative reference or validation throughout parameter space. |
| IMRPhenomT family on fixed grids | First-order AD; implicit first derivative through frequency inversion | Higher orders through inversion are unsupported; automatic duration is eager and the start mask can be discontinuous. See [model restrictions](imrphenomt.md#arbitrary-grids-and-derivatives). |
| IMRPhenomD relative-binning likelihood | CPU FP64 gradient-vector check for distance and phase, with relative error below `1e-2` required by the test | One perturbation on the stated grid; no posterior-wide mass/spin gradient claim. See [exact test scope](validation/inference.md#relative-binning). |
| Other model/option/parameter combinations | Use model-specific documentation and tests | No complete model-by-parameter derivative qualification matrix is published here. Waveform-value checks and framework autodiff availability do not supply it. |

Automatic differentiation follows the implemented numerical operations. Precision,
interpolation, piecewise branches, solver tolerances and derivative order affect
the result. Independently check the required parameter path, for example against
converged finite differences. Published forward-only timings do not measure
[derivative performance](performance.md#workloads-without-published-measurements).

## Relative binning

The [inference validation report](validation/inference.md#relative-binning)
records the dense-versus-binned likelihood and gradient checks, their grids,
parameters and acceptance criteria. Its approximation must be checked over the
region used for inference; no general likelihood speedup is established by the
waveform campaign.

## Gradients and statistical interpretation

Fisher forecasts and MCMC results require statistical checks beyond derivative
agreement. Read the [inference report](validation/inference.md#gradients-and-statistical-interpretation)
for covariance regularization and interpretation. The [NUTS API walkthrough](tutorials/03_parameter_estimation_nuts.md)
is illustrative and does not supply a converged posterior or a proper box prior.

## Reproducing the inference checks

Use the [inference check commands](validation/inference.md#reproducing-the-inference-checks).
The [catalog commands](differences_from_lal.md#reproducing-model-checks) and
[PhenomT report](validation/phenomt.md) cover waveform checks. Record revisions,
configurations, results and skips; tests linked on `main` describe the current
checks, whereas dated reports describe their recorded runs. A skipped optional
reference test supplies no agreement evidence.
