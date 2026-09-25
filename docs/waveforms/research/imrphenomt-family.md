# IMRPhenomT family: provisional mathematical ledger

This research record covers the **native time-domain** IMRPhenomT, THM, TP,
and TPHM branches at foundation commit `5bc907d`. It is an audit and a route
to an independent executable reference, **not** that reference or a claim of
whole-model validation. It is deliberately outside the `WaveformDefinition`
catalog. The public behavior and measured residuals remain in the [generation
guide](../../imrphenomt.md) and [validation report](../../validation/phenomt.md).

Primary mathematical sources are Estellés et al. [original T/TP,
arXiv:2004.08302](https://arxiv.org/abs/2004.08302), [updated T and THM,
arXiv:2012.11923v1](https://arxiv.org/html/2012.11923), and [TPHM,
arXiv:2105.05872](https://arxiv.org/html/2105.05872). The later THM paper
explicitly replaces the original T paper's fixed MECO-region joins and extends
the dominant ansatz. The active carrier must therefore be checked against the
**updated** equations. For precession, also see [Sturani,
LIGO-T1500554](https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf),
[Chatziioannou et al., arXiv:1703.03967](https://arxiv.org/abs/1703.03967),
and [Pratten et al., arXiv:2004.06503](https://arxiv.org/abs/2004.06503).

## Scope and status

| Variant | Coprecessing/aligned input modes | Extra construction | Research status |
| --- | --- | --- | --- |
| T | `(2,±2)` | Updated aligned carrier | Native executable; independent scalar spec pending |
| THM | T plus `(2,±1),(3,±3),(4,±4),(5,±5)` | Mode-specific amplitude and merger/ringdown frequency | Native executable; independent scalar spec pending |
| TP | T pair | Euler angles, remnant adjustment, Wigner rotation | Native executable; numerical-precession qualification blocked |
| TPHM | THM set | Same precession/rotation for each degree | Native executable; numerical-precession qualification blocked |

These are quasi-circular black-hole models without tides, memory, `(3,2)`
mode mixing, or a native frequency-domain implementation. The source THM paper
discusses `(3,2)` mixing, but the native supported set excludes it. Status
words below describe whether an independently specified component can be
evaluated, not whether the native implementation is physically validated.

## Inputs, conventions, and options

Masses are detector-frame solar masses; distance is Mpc; `times`, `tc`, and
`delta_t` are seconds; `f_lower`/`f_ref` are Hz; angles are radians; spins are
dimensionless. Intrinsic ansätze use `u=(t-tc)/(M MSUN_SEC)` in total-mass
units, `eta=m1*m2/M²`, `delta=(m1-m2)/M`, and a positive `(2,2)` angular
frequency with `f22=omega22/(2*pi*M*MSUN_SEC)`. Native fits use
`S=(m1² chi1+m2² chi2)/(m1²+m2²)` and `Delta chi=chi1-chi2` with masses
normalized by `M`. THM Eq. (39a) prints squares on the spin symbols whereas
the author-fit variable and native code are linear in the spins; this source
discrepancy requires a source-level decision before a literal scalar spec.

The complex strain convention is `h=h_plus-i h_cross = sum h_lm * _{-2}Y_lm`;
for `m>0`, `h_lm=H_lm exp(-i phi_lm)` and aligned negative modes satisfy
`h_l,-m=(-1)^l conjugate(h_lm)` [THM Eqs. (1)–(3)]. `u=0` is the `(2,2)`
amplitude peak, even where a higher mode peaks at a fitted shifted time.
`phic` is the **orbital** phase at `f_ref`, not the `(2,2)` phase at merger.
The sampled wrapper treats omitted/zero `f_ref` as `f_lower`; direct classes
use 20 Hz. It can infer duration, otherwise a fixed duration may truncate the
inspiral. The wrapper zeros samples before its `f_lower` crossing, ends near
`t_pad`, and has no FFT taper. Direct model calls on a caller-supplied grid do
not apply that start mask. The `modes` selector chooses positive pairs and
their negative partners for polarization projection.

TP/TPHM accept Cartesian spin vectors in the reference `L0` source frame;
aligned variants reject nonzero transverse spin. The frame axes follow TPHM
Eq. (3): reference `z=L_N` and `x=(r1-r2)/|r1-r2|`. In the paper's labeling,
`q=m2/m1>=1`, so `x` points from the larger body 2 toward the smaller body 1;
DiffGW's public mass labels are canonicalized with `m1>=m2`. `y` completes a
right-handed triad. TPHM Eqs. (4)–(9) rotate from the
coprecessing frame through a `J` frame into `L0` and project spin-weight `-2`
harmonics. Native `project_modes` evaluates the equivalent `J`-frame sum at
azimuth zero, followed by a polarization-basis factor `exp(2i zeta)`; compare
the **complete** complex strain to avoid a frame-sign false failure.

`precession_method="nnlo"` and version `10210` are native compatibility
defaults, whereas the precession paper's default is numerical PV `300`.
Method/version and remnant mapping is:

| Method | Valid versions | Default | Remnant choice |
| --- | --- | --- | --- |
| `nnlo` | `10200`, `10210`, `10211`, `10221` | `10210` | `FinalSpinMod=0` |
| `msa` | `22300`, `22310`, `22311`, `22321` | `22310` | `FinalSpinMod=3` |
| `numerical` | `300` | `300` | `FinalSpinMod=4` |

Aliases `102`/`223` resolve to `10210`/`22310`. Suffix `00` evaluates
analytical angles throughout; `10` attaches ringdown at the peak; `11` also
integrates minimal-rotation gamma; `21` linearly continues alpha/beta from
MECO and integrates gamma. MSA needs unequal masses and a valid root/remnant
domain. Numerical evolution needs `f_lower>=1 Hz` and `f_ref` no higher than
the carrier frequency at its peak; direct requests before its integration
range produce NaN. Fixed-grid first derivatives are supported, with the
start mask and branch/event selection remaining piecewise; automatic duration
is eager and higher derivatives through frequency inversion are unsupported.

## Mathematics and components

Let `C22` be the aligned dominant mode and `C_HM` the four additional
positive modes. Then `T=C22`, `THM=C22+C_HM` by harmonic sum,
`TP=R_P(C22; alpha,beta,gamma,a_f)`, and
`TPHM=R_P(C22+C_HM; alpha,beta,gamma,a_f)`. The precessing equivalent-aligned
carrier uses reference spin projections and retains the aligned inspiral and
merger; the selected precessing final spin changes **ringdown QNM inputs**
[TPHM Secs. II.2–II.5]. T and THM share the same `(2,2)` carrier.

Stable IDs below are proposed `v0` research identifiers. `conditional` means
that published equations need pinned calibrated fits/QNM inputs;
`checked` means an independent executable scalar component is present and
matches both native backends, without qualifying its composed model;
`pending` means the independent scalar expression or a downstream gate is absent.
The native locations are implementation evidence, not independent sources.

| Component ID | Exact source equation/algorithm and dependency | Variants | Status | Native locations and next gate |
| --- | --- | --- | --- | --- |
| `T-PN-FREQ-v0` | THM Eqs. (7)–(11): `theta=[eta(t0-t)/(5M)]^(-1/8)` and six fitted frequency values at `theta={.33,.45,.55,.65,.75,.82}` solve Eq. (10). Eq. (8) defines orbital `omega_N=theta³/8`; the native `(2,2)` carrier uses twice the orbital TaylorT3 series and pseudo-PN sum, with leading `theta³/4`. T3 spin coefficients/phase primitive trace to original T App. A. | all | conditional | `imrphenomt.py` `_pn_coefficients`, `_coeff`, `_phase_insp_raw`; transcribe independent PN coefficients and fitted `t0` branch |
| `T-PN-AMP-v0` | THM Eqs. (12)–(16), App. A: complex PN amplitude and three real pseudo-PN orders `x^(k/2)` for `k=8,9,10`, `x=(omega22/2)^(2/3)`; collocation at `t/M=-2000,-250,-150`; rotate HM amplitude by Eq. (13) offsets `(0,pi/2,-pi/2,pi,pi/2)`. | T, THM, TP, TPHM | conditional | `_phenomt_amplitude.py`, `imrphenomthm.py` `_mode_pn`/`_pn_amp`; isolate PN complex coefficient tables |
| `T-HM-PHASE-v0` | THM Eqs. (17)–(21): `phi_lm^insp=(m/2)phi22+arg(H_lm^insp)-Delta phi_lm`; differentiate `arg` for cut frequency. | THM, TPHM | conditional | `imrphenomthm.py` `_pn_amp_phase_derivative`, `_positive`; establish sign at complex-amplitude zeros |
| `T-REMNANT-QNM-v0` | THM Sec. II.3, Eqs. (22)–(27): 2017 aligned remnant fits, fundamental/first-overtone Kerr frequency and damping fits, with `omega=2*pi*f/M_f` converted to total-mass time by `1/(M_f/M)`; fitted peak, `c2,c3,d3`. | all | conditional | `phenomt_fits.py`; `imrphenomt.py` `_state`/`_coeff`; `imrphenomthm.py` `_positive`; pin every fit symbol/asset and numeric domain |
| `T-RD-v0` | THM Eqs. (24)–(27): QNM-rescaled rational frequency correction `c1 c2(c3 e^(-c2 t)+2c4 e^(-2c2 t))/(1+c3 e^(-c2 t)+c4 e^(-2c2 t))`, and `bar H=d1 tanh[d2(t-t_peak)+d3]+d4`; amplitude carries `exp[-alpha1(t-t_peak)]`. Model calibrates `c2`, sets frequency `c4=0`, keeps amplitude `d2=(alpha2-alpha1)/2`; continuity determines the other coefficients. Eq. (25)'s printed `bar omega=omega-omega_RD` conflicts in sign with Eq. (27a)'s `c1` and the native `omega=omega_RD-correction`; retain as a transcription gate. | all | conditional | `imrphenomt.py` and `_phenomt_amplitude.py`; THM `_positive`; verify each mode's peak shift and precessing-QNM splice |
| `T-MERGER-v0` | THM Eqs. (28)–(33): `omega=omega_RD[1-sum_{k=0}^4 a_k asinh(alpha1 t)^k]`; `H=b0+b1 tau²+b2 sech(2 alpha1 tau)^(1/7)+b3 sech(alpha1 tau)`; values/slopes and fitted interior collocation determine coefficients. | all | conditional | `imrphenomt.py` `_coeff`/`_phase_inter_raw`, `_phenomt_amplitude.py`, THM `_positive`; specify distinct frequency and amplitude joins |
| `T-FITS-v0` | THM Sec. III Eqs. (34)–(40): six `(2,2)` frequency, three per-mode amplitude, two per-mode merger, peak, and three ringdown calibrated values (16 dominant/10 additional per paper), plus final-state/QNM fits. Hierarchical NR/Teukolsky/EOB calibration is **not** recreated here. | all | conditional | `diffgw_core/phenomt_fits.py`; pin author asset hash, fit function mapping and licenses |
| `P-ANGLES-v0` | TPHM Eqs. (4)–(7), (15)–(20), App. A: `alpha=atan2(Ly,Lx)`, `cos beta=Lz`, `dot gamma=-dot alpha cos beta`; NNLO effective single spin, double-spin MSA with `alpha=alpha_-1+alpha_0`, selected suffix continuation. | TP, TPHM | pending | `_phenomt_precession.py`, `_phenomt_msa.py`, `_phenomt_angle_options.py`; independent scalar angle tables and aligned-limit checks |
| `P-ANGLE-CONTINUATION-v0` | TPHM Eqs. (5), (20), and (24): minimal-rotation gamma, continuous linear alpha/beta continuation from MECO, and fixed-beta ringdown alpha/gamma evolution. | TP, TPHM MSA | checked | `diffgw_reference/phenomt_angle_continuation.py`; scalar-table refinement agrees within `1.65e-7` rad and native continuation within `7.26e-6` rad; the raw MSA formula remains pending |
| `P-NUMERICAL-RHS-v0` | TPHM Eqs. (12)–(14), Sturani Eqs. (3),(21),(22),(24),(25): both spin derivatives, the nonspinning 3PN orbital-angular-momentum magnitude, and constrained `dLhat/d(t/M)`. The scalar exposes the `v5` through `v9` pieces separately. Production excludes only Sturani's N3LO quadratic-spin Eq. (24), while retaining the leading Kerr self-quadrupole coefficient 1 and N4LO spin-orbit. | TP, TPHM PV300 | checked | `diffgw_reference/phenomt_spin_evolution.py`; 80-digit scalar totals and independently recovered PN-power coefficients match Torch/JAX |
| `P-NUMERICAL-v0` | Integrate `P-NUMERICAL-RHS-v0` against the **T carrier** frequency using TPHM Sec. II.4.1 and Eqs. (12)–(14), then construct Euler angles and event/ringdown continuation. | TP, TPHM PV300 | pending | `diffgw_reference/phenomt_numerical_trajectory.py`; the independent event/ODE/transport/angle path matches native rotation within `4.81e-9` rad through peak, but the black-box spin-even rate and whole-waveform gate remain unresolved |
| `P-RD-REMNANT-v0` | TPHM Eqs. (21)–(26): Eq. (25) supplies the magnitude `sqrt(chi_f,AS²+S_perp²/M⁴)`; native code restores the aligned final-spin sign for the QNM input. Ringdown uses `dot alpha=omega_122-omega_121`, beta fixed at peak, gamma from minimal rotation. | TP, TPHM | conditional | `_phenomt_dispatch.py`, `_phenomt_angle_options.py`, `_phenomt_numerical.py`; characterize FS0/3/4 inputs and signs |
| `P-ROTATION-v0` | TPHM Eqs. (4),(8),(9): Wigner-D rotation from coprecessing to `J`, fixed inverse reference rotation to `L0`, SWSH projection. | TP, TPHM | pending | `_phenomt_precession.py` `wigner_d`/`project_modes`; independent factorial/SWSH scalar and frame tests |
| `T-SAMPLING-v0` | Native adapter: seconds↔mass units, reference-phase anchoring, frequency inversion, `f_lower` mask, peak placement and fixed sample grid. This is interface mathematics rather than a paper ansatz. | all | pending | `torchwave/_phenomt_td.py`, `jaxwave/_phenomt_td.py`; scalar adapter and boundary tests |

The native `(2,2)` frequency join is `t_cut=-26.982976386771437/eta` and its
interior collocation time uses `theta=.95`; higher-mode frequency joins at
`-150M` with interior point `-25M`. Dominant and higher-mode amplitude joins
use `-150M`; higher-mode amplitude ringdown begins at its fitted `t_shift`,
while higher-mode frequency ringdown begins at `0`. These are **carrier
specializations**, not permission to replace every join with the single
schematic `t_cut` in THM Eqs. (4)–(6). The merger matching systems use values
and analytic derivatives; their coefficient solves are differentiable away
from branch/fit singularities. A scalar spec must preserve the carrier's
actual branch rules and phase primitives, and document where those rules
depart from the paper's generic matching statements.

## Implementation mapping

The Torch tree is `src/torchwave/approximants/` and the JAX tree is
`src/jaxwave/approximants/`. Both expose `imrphenomt.py`,
`imrphenomthm.py`, `imrphenomtp.py`, `imrphenomtphm.py`, and corresponding
`_phenomt_*` precession, amplitude, numerical, MSA, angle-option, and dispatch
modules. `src/diffgw_core/phenomt_options.py` validates method/version for
both; `src/diffgw_core/phenomt_fits.py` supplies algebraic calibration fits.
The wrapper routes through `src/torchwave/_phenomt_td.py` or
`src/jaxwave/_phenomt_td.py`; neither backend calls a scalar reference. No
independent scalar carrier or raw analytical-angle module exists yet. The
isolated `diffgw_reference/phenomt_spin_evolution.py` covers the numerical RHS;
`phenomt_numerical_trajectory.py` independently supplies the event, DOP853
evolution, tangent transport, and angle reconstruction; and
`phenomt_angle_continuation.py` supplies the paper's scalar continuation rules.
None imports production ODE, interpolation, angle, or continuation helpers.

The calibration/reconstruction adaptation is attributed to the authors'
MIT-licensed [`phenomxpy` revision
`4121fafa0484f3db2a7352f4eea697faaaf29f65`](https://gitlab.com/imrphenom-dev/phenomxpy/-/tree/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt):
`fits.py`, `internals.py`, and `numba_ansatze.py`, as recorded in the
[provenance ledger](../../provenance_ledger.md) and
[notices](https://github.com/xangma/diffgw/blob/5bc907d/THIRD_PARTY_NOTICES.md). The upstream MSA/numerical
modules identified GPL-derived code and were excluded from adaptation; no
LAL implementation source was used. A later scalar reference must derive its
precession expressions from the cited primary equations and state its own
coefficient transcription, rather than importing the native fit functions as
an independence claim.

Four source-fidelity checks are open. TPHM Eq. (20a,b) prints
`angle(t_MECO)+t*angle_dot(t_MECO)`; native suffix `21` uses
`angle(t_MECO)+(t-t_MECO)*angle_dot(t_MECO)`, which enforces continuity at
MECO. Also THM Eq. (39a)'s printed effective-spin expression has spin squares,
but the pinned author's fit variable and native expression use linear spins.
THM Eq. (25)'s ringdown-frequency correction sign conflicts with Eq. (27a)'s
coefficient sign, and Eqs. (12)–(15) print an amplitude normalization
`2 eta sqrt(16/5)` where the native PN code uses `2 eta sqrt(16 pi/5)`;
normalization may be distributed across mode and strain conventions. The
scalar reference should record a documented resolution for each before
declaring paper-literal agreement. It should also distinguish paper schematic
`C1` joins from signed-amplitude zero crossings, where phase can jump by pi.

## Evidence and limitations

Evidence layers must stay separate. The [validation report](../../validation/phenomt.md)
records: author-reference `(2,2)` values at five times, shared calibration and
thus implementation-level evidence; Torch/JAX fixed-grid and FP64 parity;
optional **binary** LAL comparison for T/THM and matching-option NNLO
TP/TPHM; and measured MSA/numerical cases. The published papers contain NR
studies of their models, but these do not independently validate this native
implementation. No held-out NR comparison of the native family is established.

The reference report's T/THM and NNLO LAL criteria are **thresholds**, not
measured global worst cases. For MSA `(40,15)`, TP/TPHM relative L2 is
`0.00379/0.00388`; for the strongly spinning `(60,6)` holdout it is
`0.0202/0.0225`, above the retained `0.01` gate. Numerical PV300/FS4 gives `0.0234/0.0236` there and
`0.308/0.307` for a strongly spinning `(60,6)` case, with mismatch `0.0475`.
The high-mass-ratio residual survives 16384 Hz output sampling and doubled
intrinsic integration steps. Its initial-phase-anchored coprecessing `(2,2)`
L2 is `0.000314`, so the large projected difference points to precession or
frame evolution. A rotation-only test using binary supplied modes and angles
agrees below `8e-16`; this localizes, but does not solve, the PV300 issue.

The quadrupole sensitivity diagnostic reduces `(60,6)` TP L2 to `0.0181`
when native self-quadrupole coupling is set to zero, but **production retains
the published black-hole coefficient 1**. The binary's spin-even initial axis
rate is strongly suppressed relative to the native result in two cases. The
new independent trajectory matches the native published rate to `8.33e-17 1/s`
and its rotation evolution to `4.81e-9` rad, yet this does not prove the
binary's internal coefficient choice. A separate binary
full-mode inverse-frame discrepancy reaches `1.3e-9` (localized to degree 5);
no native correction is justified. These are explicit qualification blockers
for numerical precession and a reason to keep component-level evidence
distinct from whole-family validation.

## Reproduce the checks

The existing maintained paths are `tests/test_phenomt_reference.py`,
`tests/test_phenomthm_reference.py`, `tests/test_phenomt_precession.py`,
`tests/test_phenomt_spin_evolution.py`, `tests/test_phenomt_angle_options.py`,
`tests/test_phenomt_numerical.py`, `tests/test_phenomt_spin_math_reference.py`,
`tests/test_phenomt_independent_trajectory.py`,
`tests/test_phenomt_family_lal.py`, and
`tests/parity/test_phenomt_parity.py` plus the THM/precession parity modules.
The LAL tests require optional binary bindings and skip otherwise. Receipts
and exact diagnostic commands are linked in the [validation report](../../validation/phenomt.md);
`python scripts/validate_phenomt_precession.py`,
`python scripts/diagnose_phenomt_msa.py --output receipt.json`, and
`python scripts/diagnose_phenomt_precession.py --output receipt.json` are
the targeted external checks.
The scalar continuation and independent trajectory diagnostics are
`python scripts/diagnose_phenomt_msa_scalar_angles.py --output receipt.json`
and `python scripts/diagnose_phenomt_independent_trajectory.py --output receipt.json`.

The staged path to the canonical six-section executable reference is:

1. Freeze a source/asset manifest for every `T-FITS-v0` symbol (author
   revision, function, fit coefficients and domains); resolve the four printed
   equation/convention discrepancies above and publish scalar conventions/cutoffs.
2. Build independent scalar expressions for `T-PN-FREQ`, `T-PN-AMP`,
   `T-MERGER`, `T-RD`, and `T-REMNANT-QNM`; check collocation solves,
   primitives, boundary value/slope, signed amplitude and phase at all mode
   branches before composing T and THM.
3. Complete scalar `P-ANGLES` NNLO/MSA raw expressions, remnant choices and
   `P-ROTATION`; the independent suffix continuation is present. Verify
   aligned limit, reference-frame identity, each Wigner/SWSH degree and
   phase convention before composing TP/TPHM analytical versions.
4. Extend the converged independent `P-NUMERICAL-RHS-v0` event/ODE/angle path
   through ringdown and waveform composition; resolve the remaining
   self-quadrupole/orbital-axis discrepancy with equation-level
   and binary-output evidence. Keep PV300 conditional until then.
5. Add scalar `T-SAMPLING`, independent Torch/JAX component and waveform
   checks, derivatives, parameter sweeps and held-out physical comparisons.
   Record metrics, domains, failure cases and optional-asset skips in the
   **Scope and status / Inputs, conventions, and options / Mathematics and
   components / Implementation mapping / Evidence and limitations /
   Reproduce the checks** six-section format. Catalog only fully specified
   variants after the gates pass.

When a source equation, fit asset, or branch convention changes, increment
the affected component ID version, regenerate comparison fixtures from the
independent reference, and re-review both native backends and the validation
receipts. This research ledger itself changes no runtime or catalog behavior.
