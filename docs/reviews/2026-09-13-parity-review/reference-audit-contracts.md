# Sanitized reference-audit contracts

Date: 2026-09-14. Authorized by the user's request to cross-check LAL and give
the findings to clean-room Luna implementers. The coordinator is the exposed
reference auditor. Implementers work independently from the functional
contracts below and primary publications. No reference source or coefficient
table is reproduced here. See the exposure disclosure in [review.md](review.md).

Reference: LALSuite v7.26.1, commit
`c90e1175f5e238741185744a4e516e58d42a9aec`; lal 7.7.0, lalsimulation 6.2.0.
Keep the existing strict acceptance cases and thresholds. Use public binary
outputs for validation; do not fit calibration constants or residual curves.

## XP default MSA

For the audited ordinary XP waveform, omitted options resolve to precession
version 223, convention 1, and final-spin mode 3. The public parameter lookup
sentinels 300 and 4 are resolved by the waveform and are not separate angle
or remnant equations. Preserve explicit alternatives. Verify this equivalence
with public complete-waveform outputs before changing the production default.
Acceptance checks must run the original frozen parameters; copies with
explicit options are separate diagnostic probes.

Versions 222/223 use the older orbital-angular-momentum representation with
non-conserved spin magnitudes. Versions 220/221/224 use conserved spin
magnitudes. Reconstruct the former independently from the cited PN papers;
do not vary the already specified Newtonian angular momentum used by the
leading MSA root dynamics. Reference geometry and the opening-angle angular
momentum are distinct contracts.

The older reference convention also differs from the printed 2.5PN
spin-orbit term. In the normalized, non-conserved-spin expression of
[Blanchet, Buonanno and Faye](https://arxiv.org/pdf/gr-qc/0605140v4),
Eq. (6.9), its coefficient multiplying `eta S_l` is `259/75`, whereas
the paper prints `259/72`. This is an audited compatibility fact, not a
paper correction or a calibrated coefficient. Retain the published function
and implement an explicitly named compatibility variant independently.
Equivalently, its difference in `L/M^2` is
`-(259/1800) eta^2 (S_l/M^2) x^2`. A coordinator check of this functional
contract against the 72 nonaligned v223 geometry rows, including the fresh
D case, has maximum absolute inferred-L error `8.55e-15`. No coefficient
was estimated from those outputs. This does not establish native MSA-angle
or default XP waveform parity.

The compatibility final-spin prescription uses initial-total-mass units.
Infer the final orbital angular momentum from the aligned remnant-spin
estimate by subtracting the *initial* aligned spin sum
`X1^2 chi1z + X2^2 chi2z`. Combine that orbital contribution with the MSA
spin averages without introducing a final-mass squared conversion.
Independently derive the vector expression from
[Pratten et al.](https://arxiv.org/abs/2004.06503v2), Eqs. (4.17)–(4.20).
Retain separate names for a published remnant-mass convention and this
compatibility convention. Validate carrier, frame and complete waveform
separately with full three-component spin inputs.

The opening-angle computation retains the published non-conserved 3PN
angular momentum, including `259/72`; only the reference-frame geometry uses
the compatibility coefficient above. The two must not share the same
reference-angular-momentum function by accident.

Additional v223 compatibility contracts, independently checked by the
coordinator against the public angle interface:

- The flux coefficients through `a5` use the initial spin projections and
  products, not the precession-averaged products. With `S_i = X_i^2 chi_i`,
  supply `S_1z`, `S_2z`, `S_1 dot S_2`, `S_1z^2`, `S_2z^2`, and
  `S_1z S_2z`. Retain the leading MSA averages for the root dynamics.
- Write the clean root-asymptotic nutation coefficient as
  `psi2 = 3 g2/g0 + R`. The compatibility value is
  `3 g2/g0 + q^2 R`, where `q=m2/m1<=1`; `psi1` is unchanged.
  This is a version convention, not permission to alter the published
  derivation or calibrate against the oracle.
- In the oscillatory corrections of [Chatziioannou et al.](https://arxiv.org/abs/1703.03967v3),
  Appendix B and F, the compatibility convention uses negative nutation
  derivative and absolute values of the two azimuthal residues. Its
  inverse-tangent differences use the principal tangent branches. Include
  the total-angular-momentum factor exactly once. The compatible third-angle
  correction omits the extra explicit `-sin(2 psi)/4` term. Implement a
  separately named version path with branch-continuity tests.
- The initial nutation phase and oscillatory contribution are disabled when
  the squared-spin-root gap is below `1e-5`. Aligned and degenerate systems
  need explicit limiting behavior; a finite general-case formula is not
  sufficient validation of that limit.

The coordinator's subsequent boundary checks expose an overbroad limiting
branch in the first implementation. A small root gap suppresses the
oscillatory correction, **not** secular alpha/gamma. At a current-frequency
gap below `1e-5`, the spin-norm expression uses the upper root, equivalent
to zero Jacobi sine. Keep the leading root mean and its associated `c1`
consistent. Do not replace both with arbitrary floors or zero the entire
precession solution. Equal mass is not itself a reason to remove precession.
The reference's singular/one-spin/aligned-root regularization has lower and
upper physical squared-spin roots `S_initial^2` and `S_initial^2+1e-9`,
with the third root zero. This is an explicitly audited numerical convention,
not a physical spin floor or an oracle-fitted calibration. The root mean
and the spin norm remain distinct in this branch. Derive secular limiting
expressions from the primary equations and validate complete polarizations
at exact and near alignment and exact and near equal mass before routing.

The zero-nutation initialization requires care with the phase coordinate.
In the published Eq. (51), write `psi(v)=C+P(v)`, where
`P(v)=-(3/4) g0 delta (1+psi1 v+psi2 v^2)/v^3`. An initial squared-root gap
below `1e-5` sets the additive integration constant `C` to zero. It does not
set the total phase at the reference frequency to zero. For ordinary initial
roots, `C=psi_initial-P(v_ref)`. This difference becomes observable when a
small initial root gap later exceeds the threshold: use `P(v)` in the Jacobi
and oscillatory expressions, without subtracting `P(v_ref)`. The small gap
at a current frequency still bypasses those evaluations. Keep this constant
distinct from the total reference phase in an API named `anchored_psi`.
The previous wording "initial nutation phase disabled" did not make this
distinction clear; this paragraph supersedes that ambiguous interpretation.

Apply that root regularization for an invalid trigonometric cubic, either
zero spin, or either spin exactly parallel or antiparallel to the initial
orbital angular momentum. A small gap or equal mass alone does not select
it. For the ordinary cubic, clamp the inverse-cosine argument to [-1,1].
A finite argument outside that interval still takes the clamped endpoint;
it does not select the invalid-cubic regularization, even if the overshoot
exceeds a roundoff tolerance. Keep invalid square roots and undefined
inverse-cosine inputs as explicit invalid branches. Take absolute values
of the two physical squared-spin roots; do not
move the inverse-cosine endpoints inward by an arbitrary epsilon. In the
published precession averages, the numerical convention adds `1e-16` to
each denominator factor `(1-q)^2` and `1-q^2`. Preserve the physical mass
ratio and initial invariants; replacing them with a synthetic mass ratio
changes the secular solution. These are numerical compatibility choices,
not changes to the underlying published physics.

Version 223 also permits failure of MSA initialization to select the existing
version-102 NNLO prescription. Its numerical validity guard concerns the
magnitude of the fifth secular Omega-z coefficient (above 1000), rather
than the root-gap threshold used for nutation. Keep these separate decisions;
a blanket equal-mass or small-spin switch to NNLO is not established.

With these contracts, the coordinator's three fixed nonaligned systems
reproduce reference-anchored alpha and gamma within `1.16e-13` radians and
`cos(beta)` within `8.89e-16`. These are component diagnostics, not routed
default-XP acceptance results. The source-isolated Luna implementation is
under review and must pass independent parameter and full-waveform checks.

## Pv2 carrier time shift

The compatibility time shift differentiates a natural cubic interpolant of
10 uniformly spaced phase samples from `0.8 fRD` to `min(1.2 fRD, fcut)`,
evaluated at `fRD`. The derivative must use the same phase coordinate and
ring frequency as the carrier. This is numerical discretization metadata,
not a fitted correction. Preserve a separately named analytic-derivative
option. Test the spline independently and verify signs against the actual
complex carrier; a small slope residual alone does not establish waveform
parity because its constant phase and source-frame factors also matter.

The precessing remnant uses the same D QNM sampling representation described
below. A coordinator diagnostic combining these two conventions reproduces
both polarizations to relative L2 below `9e-14` on the five fixed clean-Pv2
diagnostic cases over 20–511.75 Hz. The standard public interface zeros the
requested 512 Hz endpoint. Implement sequence and uniform-grid support as
explicit interface behavior; removing arbitrary last bins is not a fix.
The subsequent frozen production checkpoint passes both original Pv2 gates
below `6.4e-14`; see [catalog](verification/reference-contracts-checkpoint1/catalog.json)
and [holdout](verification/reference-contracts-checkpoint1/holdout.json).

## HM carrier QNM ratios

Keep HM QNM frequencies in the outer frequency maps. In the modified D
carrier phase of [London et al.](https://arxiv.org/abs/1708.00404), Eq. (14),
use HM data only through `R = f22_HM / flm_HM` and
`T = fdamp_lm_HM / fdamp_22_HM`. The D carrier retains its own absolute
ring-frequency center and damping scale: the modified arctangent has center
`alpha5 fRD_D`, width `R fdamp_D T` and multiplier `R`. Subtract the original
D arctangent and complete the phase with continuous value and derivative
at `fRD_D / 2`. Supplying distinct carrier QNM inputs must affect both the
modified branch and the continuity join. Test this with synthetic unequal
carrier/HM frequencies, same-mode identity and parameter gradients.

## D QNM numerical representation

The reference interpolates cyclic frequencies with a natural cubic spline
on a uniform spin grid of spacing 0.002, with extra endpoint knots at
±0.999 and ±1. Frequencies are represented to six significant digits.
Independently generate physical Kerr roots, convert angular frequencies to
cyclic frequencies *before* rounding, and assess the interpolation contract.
No reference-table ordinate may be copied. Some extreme and negative-spin
source values differ from independently solved roots; matching those values
is not authorization to replace the roots with reference data. Clearly
bound the independently verified domain and handle endpoints explicitly.

The frequency-sequence and standard conditioned waveform interfaces have
different high-frequency support. Match the requested interface when
reporting parity; do not count an expected truncation difference as a
carrier calibration error.

## SEOB orbital-peak epoch

The reference epoch uses a natural cubic spline over the full high-resolution
orbital-frequency series. Locate the first non-increasing sample. Choose an
adjacent interval according to the derivative at that sample: the following
interval for a positive derivative, the preceding interval otherwise. Apply
the generic derivative-sign-product bisection update until the interval width
is at most `1e-5 M`, including when its endpoints have the same derivative
sign. The returned estimate is the last evaluated midpoint. Same-sign
contraction can select a sample boundary rather than a physical extremum;
do not silently substitute a maximum-search algorithm. This is a numerical
convention, not an adjustable epoch offset.
The sign-product equality case moves the left endpoint to the midpoint;
only a strictly negative product moves the right endpoint. Include a
synthetic exact-zero derivative case in the numerical checks.

For the non-optimized aligned-spin path, angular velocity is evaluated as
`(d H_reduced / d p_y) / r` in the Cartesian chart with position `(r,0,0)`
and momentum `(prstar, pphi/r, 0)`. The numerical derivative is the public
GSL central-derivative routine with initial momentum step `1e-4`.
Validate the chart, derivative and spline independently before comparing
GPS-quantized epochs; do not prescribe an empirical nanosecond correction.
On the existing eight public state arrays, the natural-cubic/sign-product
contract reproduces five epochs to within `0.15 ns`; the three sign-bracketed
cases retain `7.46–11.09 ns` discrepancies with the clean Hamiltonian.
The derivative and attachment comparisons remain unresolved.

The NQC sampling contract also uses natural cubic interpolation over the full
high-resolution series, rather than local polynomial or analytic flow
derivatives. Starting from the published amplitude basis `q_j`, phase basis
`p_j`, amplitude `A`, and continuously unwrapped waveform argument, interpolate
each sampled product `A*q_j` independently. Its value, first derivative and
second derivative at the published matching time form the three amplitude
conditions. Interpolate the sampled phase bases independently for their first
and second derivatives. The base amplitude and waveform argument likewise use
their own natural cubic interpolants. Differentiating a product of separately
interpolated factors is a different numerical operation. Use one consistent
time origin for the entire series and matching time. These are numerical
compatibility requirements for the existing published NQC equations; no
calibration coefficients or fitted waveform residuals are supplied here.
In particular, obtain the base frequency and its derivative in the phase
matching right-hand side from the first and second derivatives of the
unwrapped waveform-phase spline. A spline of separately sampled analytic
frequency is a different operation. Normalize the phase convention to a
positive frequency, changing the derivative sign consistently. The reviewed
first implementation still uses the latter operation and needs correction.

Ringdown attachment is a separate discrete operation. Solve the NQC system at
the continuous published matching time, then attach at the preceding sample
of the high-resolution grid, using that grid's own time origin. Here
"preceding" means the greatest sample time less than or equal to the match;
an exact-knot match uses that knot. For the 22
mode, use the magnitude and unwrapped phase of the corrected waveform at
that sample, impose zero amplitude derivative, and obtain the positive
matching frequency from the backward phase difference over one high-grid
step. Do not replace these boundary values by the continuous NQC targets or
search for a new sampled amplitude maximum. The internal step in mass units
is the output step divided by `2^max(0,ceil(log2(50*dt_output/M)))`.
Reject an attachment whose floor sample has no preceding sample for the
backward difference; do not advance the boundary to a later grid point.
The high-resolution ringdown contains `floor(10/(damping*dt_high))` samples,
starting at elapsed time zero, followed by zero padding. Thus the last
nonzero sample precedes ten damping times. Apply this support rule before
downsampling onto the output grid.
Evaluate the published merger ansatz with those boundary data. A coordinator
control on two existing public state arrays reduces the post-attachment
100M complex relative L2 errors from `4.75e-4–4.93e-4` to
`7.79e-6–5.02e-5`; these are component diagnostics, not whole-waveform gates.
Remaining peak, calibration precision, QNM and output interpolation effects
must still be isolated without fitting any waveform residual.

The EOB fundamental QNM uses the 107-knot angular-frequency convention already
independently reconstructed in `kerr_qnm_table_convention.py`: regenerate
ordinates from the Leaver eigenproblem, round each angular-frequency component
to six significant figures, and apply a natural cubic spline. This is distinct
from D's 1001-knot cyclic-frequency convention. At the nonspinning 30+20 remnant,
the independent 107-knot result agrees with the public QNM API to `1.12e-16`;
the unsampled eigenfrequency differs by `1.49e-6` in initial-mass units.
Combining this with discrete attachment gives amplitude relative L2 below
`2.80e-7` on both 100M controls, while complex errors remain `3.21e-6–4.43e-5`.
The [phase-spline control](verification/eob-numerical-root-review/phase-spline-control.json)
uses the waveform spline in the phase-matching right-hand side and reduces
the complex errors from `6.50e-6–1.00e-4` to that range. This algebraic
control changes no fitted target or phase offset; the remaining error still
exceeds the strict waveform threshold.
The grid metadata have an earlier public-binary reconstruction; no reference
table ordinates are supplied. Keep inference-first initialization and subsequent
spin gradients valid when caching independently generated spline coefficients.

The [tail-window control](verification/eob-numerical-root-review/tail-window-control.json)
separates support from attachment error. On two paired public state/mode
fixtures the first 100M amplitude errors are `2.80e-7`; evaluating beyond the
reference tail raises whole-post-boundary amplitude errors to `7.25e-5` and
`7.36e-5`. Complex errors within the nonzero reference support are `1.00e-4`
and `6.50e-6`, so the phase-matching correction remains independently needed.
These are component checks with public states, not native-trajectory parity.

The numerical driver must actually generate the high-resolution trajectory,
not merely return its proposed step as metadata. Use two stages: retain a
coarse output-grid trajectory through the first orbital peak, then restart
from an existing sample roughly 150M before its end on the finer grid.
For a shorter initial trajectory, use approximately its latter half. Keep
the restart sample's state and phase, the fine grid's local origin, and the
mapping back to the initial-time chart explicit. The standard non-optimized
reference uses absolute tolerance `1e-10` and relative tolerance `1e-9` for
adaptive evolution; independently implemented adaptive integration and
refinement checks are appropriate. No source integrator is supplied.
The [adaptive-driver specification](eob-adaptive-driver-contract.md) records
accepted-state event checks and natural-cubic uniform resampling separately
from the underlying Fehlberg step. The coarse and fine stopping rules differ;
testing only at uniform output samples is a different numerical contract.
Find the attachment peak from the full fine-grid spline rather than passing
the coarse trajectory's local-polynomial peak into the attachment helper.

For spinning systems, the v4 factorized 22 phase selects the total-real-energy
interpretation of [Bohé et al.](https://arxiv.org/abs/1611.03703v3), Eq. (2.6).
It excludes both the older Pan test-particle `v^8` phase term and the v2-only
positive-spin `540` term. In the existing explicit phase API this corresponds
to `bohe_total_energy`, with the two optional inheritance flags false. This
selection needs spinning waveform validation; the zero-spin controls above
cannot distinguish it.

There is also an amplitude compatibility difference from the paper's stated
replacement of the spinning test-particle terms. The audited v4 convention
retains the cubic-spin contribution `a^3 v^7 / 3` from
[Pan et al.](https://arxiv.org/abs/1006.0431v2), Eq. (29a), in addition to the
finite-mass linear-spin expression in Bohé Eq. (2.5), where
`a=(1-2 eta) chiS + delta chiA` and `v=(M Omega)^(1/3)`.
Keep the printed-paper implementation and add an explicit compatibility
choice. This is a published analytic term with audited inheritance metadata,
not a coefficient inferred from waveform outputs.

## XHM release and inspiral assembly

The follow-up source audit distinguishes amplitude and phase versions:
release `122022` changes the amplitude calibration and assembly choices but
retains the earlier phase model. The proposed newer phase branch present in
the reference is inactive for this default. Do not require a new `122022`
phase artifact merely because the overall waveform selects that release.
For `(3,2)`, the active spheroidal phase has four conditions, consistent with
the public paper's Sec. VI.2.2, Eqs. (46)–(48). Torchwave's inherited route
instead uses three targets and fixes one coefficient to the quadrupole value.
The independently derived `xhm_phase_clean.py` implements the four-condition
algebra, but production integration requires targets at all four published
frequencies. Reusing or interpolating the old three targets is not a sourced
replacement. See the [updated inventory](xhm-calibration-critical-path.md).

The installed reference defaults to release `122022`. The original
[XHM publication](https://arxiv.org/abs/2001.10914v2) and the existing author
fit provider describe the earlier amplitude model. Keep the strict cases on
the default release. Pinning `122019` is a diagnostic control, not acceptance
of the current release or an interchangeable calibration.

For the earlier release, the inspiral provider takes the normalized spin
`chiPNHat = (chi_eff - 38 eta (chi1 + chi2)/113)/(1 - 76 eta/113)`.
Its `amp_insp1`, `amp_insp2`, `amp_insp3` magnitudes belong respectively to
`f_ins`, `0.75 f_ins`, `0.5 f_ins`. Thus the clean solver's ascending node
order is the absolute values of provider entries `3,2,1`. The original
`q<20` cuts use the published MECO/ISCO prescription already reconstructed
in the clean helper. Use the remnant spin in the Kerr ISCO frequency. The
33 cut's last spin term is linear, as printed.

Relative to the explicitly printed Appendix-E complex PN polynomials, the
compatibility 44 polynomial reverses the signs of the `v^2` and `v^4`
terms, keeping its `v^5` and `v^6` terms. The compatibility 32 polynomial
reverses its `v^2`, `v^3`, and `v^4` terms, keeping `v^5` and `v^6`.
Preserve the printed functions and name the compatibility variants. Derive
the polynomial and normalization independently; do not transcribe a source
expression. After the spin normalization and 44 sign changes, a coordinator
probe of four fixed intrinsic systems reproduces old-release physical
44-mode inspiral amplitudes to relative L2 below `2.32e-14`.

The 21 non-reexpanded SPA branch uses the full TaylorT4 derivative through
3.5PN, including its 3PN constant and logarithmic terms. The clean helper
currently transcribes the shorter displayed Appendix-E expression and omits
that relative `v^6` contribution. Recover it independently from primary
PN energy/flux publications, retaining an explicitly named printed
truncation. Do not infer its coefficients from waveform residuals.

An independent way to recover the missing term starts from the published
TaylorF2 phase `3 v^-5 sum[(p_k + l_k log(4v)) v^k]/(128 eta)`.
Differentiating twice via the stationary-phase time relation gives the
normalized `dt/dv` coefficient
`b_k = ((k-5)(k-8)(p_k+l_k log(4v)) + (2k-13)l_k)/40`.
Series inversion gives the TaylorT4 coefficient
`a6 = -b6 + 2 b2 b4 + b3^2 - b2^3`.
Include the published 3PN spin-spin phase contribution, for example
[Pratten et al.](https://arxiv.org/abs/2001.11412), Appendix A, rather than
the older PhenomD calibration's truncated phase. Adding only this missing
term reproduces old-release physical 21-mode inspiral amplitudes to relative
L2 below `4.3e-11` on the four coordinator controls. The root algebraic
diagnostic is not yet an integrated implementation or current-release gate.

For the four same old-release controls, all three 21/44 provider collocation
magnitudes agree with the public modes to floating-point accuracy. The 33
provider does not: differences are about 0.3–0.5 percent at the first
system's nodes. An audited comparison establishes a different calibration,
not merely a rounded representation. No alternative source coefficients
were transferred to implementers. The 21 between-node residual remains
about `2e-4`–`3.3e-4` before the full TaylorT4 correction.

The current `122022` amplitude release also changes the fit data, cuts,
intermediate collocation system and normalization. In its `q<20` branch,
the inspiral boundary is the mode-scaled MECO frequency, and its amplitude
collocations use geometric strain rather than the old common-H rescaling.
Those changes require a distinct release implementation and independently
admissible calibration inputs. Do not apply old provider values to the new
system or classify all remaining error as calibration before independently
checking the assembly contracts.

## TEOB radiation normalization and truncation

The general Newtonian mass polynomial
`c = X2^(ell+epsilon-1) + (-1)^m X1^(ell+epsilon-1)` already contains
the odd-mode mass difference. In particular, the `ell=5`, odd-m expression
is proportional to `(X2-X1)(1-2 eta)`. Applying another
`sqrt(1-4 eta)` duplicates that factor. The correction described in
[Nagar et al.](https://arxiv.org/abs/1806.01772v2), Sec. II.B.1, restores
a missing factor in an older representation; it is already present in the
general mass polynomial. Validate linear vanishing at equal mass, exchange
signs, and the resulting flux rather than preserving a test that expects
the duplicated factor.

The introductory paragraph before [Damour, Nagar and Bernuzzi](https://arxiv.org/abs/1212.4357v2),
Eqs. (D35)–(D44), excludes the highest test-mass phase terms for positive
eta. The finite-mass 32, 42 and 44 residual phases therefore omit their
`y^(9/2)` terms. Retain those terms in the separate test-mass limit.
The literal 21 Padé denominator `2456` in Eq. (D46) also occurs in the
audited reference; replacing it by `2465` would change compatibility.
The discrepancy with the expected weak-field normalization is recorded,
not silently corrected using the reference as a calibration target.
Nonspinning vacuum radiation reaction also needs the horizon contribution
specified in the same paper, Eqs. (36)–(45) and Table I. With `x=v_phi^2`,
its correction to the asymptotic Newtonian-normalized flux is
`(1-4 eta+2 eta^2) [x^4 Heff^2 rhoH22^4 + x^5 Sodd^2 rhoH21^4]`,
where `Sodd=p_phi/(r_omega v_phi)`. The public table supplies the two
fourth-degree residual polynomials. Multiply the sum of normalized
asymptotic and horizon contributions by
`-(32/5) eta r_omega^4 Omega^5` for the reduced angular torque.
The reviewed clean component now supplies both the 35-mode sum at infinity
and this horizon contribution. Driver integration must supply the evolving
`Hcirc` at zero radial momentum and the current angular momentum, together
with the radial derivative of the potential.
Derive and test normalization, leading powers and source factors independently;
do not insert an extra factor of eta when adding the horizon torque.
This contract is for nonspinning vacuum binaries; spinning and tidal cases
require their own prescribed treatment.

The stored PA radial acceleration used by full-trajectory waveform NQC has
a stage-specific energy convention. With `E=H_real/M`, multiply the ODE
frozen-mobility acceleration at the same state by `E^2` on every PA-prefix
knot, including its endpoint. Subsequent ODE knots retain the ODE value.
The merger auxiliary grid is entirely in the ODE stage. This is an audited
compatibility convention; derive the two Hamiltonian energy factors
independently from the published energy map and Hamilton equations. It is
not a new calibrated quantity or a claim that the paper prescribes this
stage-dependent storage. Validate its effect through the complete waveform.

These are component contracts; the complete TEOB driver remains unfinished.

## TD-to-FD conditioning

The strict SEOBNRv4 and TEOBResumS references use the conditioned
`SimInspiralFD` interface. A direct transform of raw TD output implements a
different array contract. The separate [conditioning handoff](td-fd-conditioning-contract.md)
specifies lower start frequency, two taper stages, zero-state Butterworth
filtering, final-sample cropping or leading padding, and FFT epoch semantics.
It includes public-binary diagnostic samples and independent filter checks.
No physical waveform, reference implementation, or calibration table is
supplied to implementers.

## Current implementation boundary

### SEOBNRv4 timing precision and public NQC observables

The version-4 numerical compatibility route evaluates the peak-time offset
with each coefficient of the published Bohé et al. equations (4.9) and
(4.15) rounded to six significant decimal digits before polynomial
evaluation. Rounding the final evaluated offset is a different operation.
Implement this convention independently from the paper's printed values;
retain the original published-precision helper for equation-level use.
This convention does not supply missing precision for the separate peak
amplitude, curvature, frequency, or frequency-slope targets.

Two public binary interfaces supply additional validation observables.
`SimIMRSpinAlignedEOBModes`, with version 4, a caller-owned length-10 real
vector and `nqcFlag=1`, fills the NQC vector and returns without waveform or
dynamics arrays. Its vector order is `a1,a2,a3,a3S,a4,a5,b1,b2,b3,b4`.
`SimIMRSpinAlignedEOBPeakFrequency` with version 4 returns the target
frequency in Hz; multiply by `2*pi*M_seconds` for the dimensionless GW
frequency used by the clean matching equations. These outputs are validation
data only, never runtime calibration or fitting inputs. A coefficient
disagreement alone does not distinguish trajectory, target, matching-time
or interpolation errors.

### MSA pole continuation near alignment

The small-transverse-spin boundary diagnostic uncovered a second issue after
the integration-constant correction. The native v223 compatibility operation
continues pole factors through negative values using their absolute square
roots. A negative factor alone is not an invalid-row condition. The ordinary
published real-pole helper can retain its stricter domain.

An independent partial-fraction derivation gives a compact handoff. Write
`R(u)=r0+ra/(1-la*u)+rb/(1-lb*u)`, with algebraic labels
`la=(-d2+s)/(2*d0)`, `lb=(-d2-s)/(2*d0)`, and
`s=sqrt(abs(d2*d2-4*d0*d4))`. Do not relabel by positivity. For each label let
`n=1-lambda`, `k=sqrt(abs(n))`, and
`B=atan(tan(psi))-atan(k*tan(psi))`, evaluated with a stable equivalent.
The two phi contributions in the compatibility convention are
`Ta=J/abs(psidot)*sign(la)*abs(ra)*B_a/k_a` and
`Tb=J/abs(psidot)*sign(d0*(d0+d2+d4))*sign(lb)*abs(rb)*B_b/k_b`.
Their zeta contributions carry the respective geometric factors
`A_theta+B_theta/la` and `A_theta+B_theta/lb`.
These identities follow by expressing the paper's pole coefficients as
`C=la*ra/(1-la)` and `D=lb*rb/(1-lb)`; the compatibility convention takes
the absolute value of C and retains the second pole's denominator sign.
Handle exact limiting values and nonfinite outputs explicitly. Do not change
the current-root-gap threshold or fit a boundary correction.

For the 35+25 solar-mass case with spins `(1e-5,0,0.2)` and
`(0,1e-5,-0.1)`, reference frequency 20 Hz, the clean 190 Hz sample currently
suppresses a finite compatibility correction because `n_a<0`. Around 196 Hz
the near-zero factor's sign is sensitive to cubic-root roundoff. Component
checks must report both this numerical sensitivity and raw waveform errors;
an algebraic extension by itself is not evidence of full boundary parity.

### Exact aligned convention-1 endpoint

At exactly zero transverse spins, retain the finite opening-angle cosine in
the source-frame projection. In the supported v223/convention-1 composition
apply the paired chart change `delta_alpha=-phi_J`, `delta_gamma=+phi_J`.
Applying only one angle offset at finite beta changes the waveform. This
operation is restricted to exactly zero transverse components and must not
absorb tiny nonzero-spin failures into an aligned-carrier shortcut.

### TEOB merger and postmerger

The [merger assembly specification](teob-merger-assembly-contract.md) records
the independent auxiliary-grid, discrete NQC, phase-anchor and support
requirements. The published postmerger phase uses `c2phi=alpha21`; its
amplitude uses `c2A=alpha21/2`. The earlier clean component incorrectly used
the amplitude exponent in both places. This has been independently corrected
and checked against the merger-frequency and asymptotic QNM conditions.

These findings establish actionable missing conventions. The latest completed
full-suite result is **17/21 catalog and 11/13 holdout**
in [checkpoint 3](verification/reference-contracts-checkpoint3/source-record.json).
Component changes after that run do not increase the counts. MIT metadata
and source-separated implementation do not certify the provenance of
inherited full-precision PhenomD/X calibration data.
