# Native ROM development

This branch-only workspace contains experimental EOB assemblers, the retired
pilot surrogate, and development checks. It is excluded from release packages.
The prototypes are not registered runtime models, qualified SEOBNRv4 generators,
or replacements for the official ROM data. No trained asset is promoted here.

The [first frozen qualification run](evidence/2026-09-22/README.md) failed the ODE
convergence budget. The generator is not qualified for independent ROM training.
An opt-in [fixed-restart numerical diagnostic](evidence/2026-09-23/README.md)
passes both convergence gates for one nonspinning row; the default route still
fails. Its sampled-peak public-binary comparison has a `0.02171` complex
relative L2 difference; [sub-sample peak anchoring](evidence/2026-09-23/README.md)
reduces it to about `0.0068–0.0069`. The same diagnostic choices fail both
numerical gates for an aligned-spin row; its sampled-peak difference of
`0.01336` falls to about `0.0078–0.0081` with sub-sample anchoring. These
comparisons do not validate the physical generator.

The later [fixed-trajectory output-sampling check](evidence/2026-09-23/README.md)
finds identical accepted trajectories across three grids and much smaller
Hermite-versus-DOP853 differences than natural-cubic differences through
`165 M`. In a predeclared
[six-row exploratory parameter campaign](evidence/2026-09-23/README.md),
both start-frequency rows pass the numerical gates, but four mass-ratio/spin
rows fail at least one. In particular, the strong aligned-spin row's refined
ODE complex error is `6.57e-6`, above the `1e-6` budget. The provisional
generator remains unqualified; no ROM training or release promotion follows.

A later development-only [six-row Fehlberg output-sampling control](evidence/2026-09-23/README.md)
passes the frozen numerical gates after fixing the first ODE trial step across
output grids. Its zero grid differences are an identity control. A separate
two-row off-grid check finds near-roundoff agreement with locally seeded
DOP853 at the NQC match, while the earlier Hermite sampler has larger local
errors. Neither check validates the provisional physical equations or changes
the release catalog.

The versioned [SEOBNRv4 paper reference](reference/SPEC.md) pins published
calibration, Hamiltonian, selected flux, leading initial-state, peak, and
ringdown components and maps applicable checks to the frontends. Its
[manifest](reference/manifest.json) marks unresolved generator components
explicitly. Resolve those entries and validate the complete generator before
extending the ROM work or promoting any model.
A separate [pinned public-binary component check](evidence/2026-09-23/README.md)
identifies the Hofmann `(3,4)` remnant-spin choice for the tested LAL version
and bounds native fundamental QNM differences at nine signed-spin points.
The [131-spin follow-up](evidence/2026-09-24/qnm_dense_spin.json) finds a
`1.72e-5` near-extremal interior frequency difference from that wheel,
exceeding the earlier nine-point `6e-7` diagnostic gate. Torch and JAX agree,
and the native value matches our original Kerr equation solver at the worst
spin to `2.3e-16`. A [pinned-source audit](evidence/2026-09-24/qnm_source_interpolation.json)
reproduces the wheel's historical GSL spline to `1.2e-16` on five witness
spins. Replacing its printed knots with native Kerr solutions leaves a
`1.75e-5` interpolation error at the worst positive-spin point, so knot
spacing and curvature dominate that discrepancy. The wheel tables are
third-party numbers and stay outside the native package. A
[Cook author-data check](evidence/2026-09-24/qnm_negative_author.json) finds
native counterrotating roots within `7e-13` on five sparse high-spin rows,
while the wheel differs by `0.0150` in frequency at `j=-0.9995`.
Historical compatibility, domain-wide accuracy, and whole-waveform impact
remain open.
The [post-hoc binary-input witnesses](evidence/2026-09-24/qnm_binary_reachability.json)
show that this is not confined to an unreachable final spin: at the paper's
`(q, chi1, chi2)=(8,0.85,0.85)` calibration configuration, the native/wheel
QNM frequency gap is `1.20e-6`, and it reaches `1.91e-5` for an extrapolated
`(76,1,-1)` binary. At `(80,1,-1)` the fitted remnant exceeds the pinned QNM
helper's `j=0.9996` limit. These are component findings, not full-waveform
errors or physical validation of the historical wheel interpolation.
The reference also checks sparse Berti and Cook author QNM values and the
published hybrid `(4,2)`, `(4,3)`, and `(5,5)` flux amplitudes. These component
results do not change the failed whole-waveform or release gates.
A frozen aligned-spin waveform replay tests the paper-derived higher-spin
`(2,1)` rho hybrid: peak-fitted complex relative L2 changes only from
`0.00807196` to `0.00806111`. This candidate remains development-only while
the compiled component and full model are unqualified.
The [historical `(3,3)` flux placement check](evidence/2026-09-24/README.md)
finds a substantial same-state torque improvement through `M Omega=0.12`,
but a residual near `245 ppm` at `0.15` and peak-anchor-sensitive mixed
whole-waveform results. That explicit Torch/JAX option remains diagnostic;
the physical and numerical release gates remain open.
The subsequent [historical Hamiltonian derivative check](evidence/2026-09-24/README.md)
finds that the pinned LAL source omits a term from an internal logarithmic
potential derivative. Reproducing that historical convention together with
the `(3,3)` flux choice reduces a frozen aligned-spin, peak-fitted whole-waveform
complex relative L2 difference from `0.00827564` to `0.00760014`. This
source-motivated variant remains development-only; the residual and open
reference-component gates prevent release qualification.
The same historical internal derivative is now available as a dev-only JAX
component. Its potential values, shifted derivative, and JIT evaluation pass
three frozen mass-ratio/spin checks; three same-state Torch/JAX regression
checks cover every metric field and effective Hamiltonian. This is frontend
component parity, not a
JAX whole-waveform qualification.
The JAX 35-mode pre-NQC flux also has a same-state Hamiltonian adapter for
reduced-mass radiation reaction. Scalar Torch/JAX torque checks and JAX batch,
JIT, and gradient checks pass; the JAX waveform and historical torque parity
are still open.
Combining the historical derivative with the fully native frequency warm start
lowers that one-row fitted difference to `0.00131140`. A pinned
public-start-state intervention
lowers it to `0.00116397`, so most of the residual remains. The same combined
candidate passes nested-output-grid stability but fails the refined ODE
tolerance budget (`3.53e-6` complex and `1.51e-6` amplitude, versus `1e-6`
limits). An opt-in local NQC second-derivative rule reduces those refined
differences to `1.05e-7` and `2.87e-9` on the same row while leaving the
physical comparison near `0.00131`. These [frozen diagnostics](evidence/2026-09-24/README.md)
remain on the feature branch; no model or ROM is release qualified.
A [three-row boundary campaign](evidence/2026-09-24/README.md) finds that a
mass-ratio-four nonspinning row passes its ODE gate, while strong aligned spin
still fails the complex gate and the original strong antialigned run has a
restart after its match time. A frozen antialigned rerun with an earlier
restart passes its ODE gate. Both ODE-passing rows pass a nested-grid identity
control. Exact-hash attribution traces the aligned-spin numerical failure
chiefly to the local NQC phase-basis second-derivative estimate; the raw
pre-NQC waveform is much more stable. None of these boundary checks qualifies
the physical generator or the independent ROM.
A post-hoc common `150 M` restart brings all three boundary rows inside the
fine trajectory and under the refined ODE budget; strong aligned spin still
differs from the pinned public waveform by `0.01985` complex relative L2.
A [predeclared wider-window intervention](evidence/2026-09-24/README.md)
passes the strong aligned-spin ODE numerical gates at `2.70e-7` complex
relative L2, while its fitted public-binary difference remains about `0.01985`.
This opt-in numerical choice still needs broader domain checks; the
historical-physics and ROM gates remain open.
A DOP853 dense-trajectory check on the frozen aligned-spin row finds the
`1 M` phase-basis window closest to the local derivative (`3.90e-11` maximum
matrix difference, versus `4.59e-10` at `0.5 M` and `1.63e-9` at `2 M`).
The widest window reduces waveform convergence drift but increases local
derivative bias here; the default remains unchanged and domain qualification
is still open.
The intermediate `1 M` window also passes the aligned-spin ODE gate at
`6.23e-7` refined complex relative L2, while its fitted public difference
remains about `0.01985`; it is still a development-only choice.
The same window passes the frozen ODE gates on nonspinning mass ratio four and
strong antialigned-spin rows (`1.41e-8` and `5.85e-7` refined complex relative
L2). These three rows support the numerical choice but do not bound its full
domain or establish physical accuracy.
The pinned non-HM v4 source also overwrites a `(3,2)` flux spin-square term.
An explicit source-faithful diagnostic choice reduces the same-state aligned
torque gap by more than 98% through `M Omega=0.12` and the aligned whole-waveform
fitted difference to `0.00587`; its refined numerical gate passes. The
antialigned boundary numerical gate also passes, but its already small fitted
public difference increases slightly. The remaining physical residual and
unresolved generator/ROM gates keep this choice and the model unqualified.
Adding the pinned non-HM v4 `(2,1)` and `(3,1)` source placements further lowers
the strong aligned-spin fitted difference to `0.00300`, with its numerical gate
passing. The post-peak comparison worsens, and domain/attachment checks remain
open, so this combined variant also stays diagnostic.
On that same combined variant, changing the NQC phase-derivative window from
`2 M` to `1 M` reduces refined ODE complex error from `3.26e-7` to `2.41e-7`;
the fitted public difference remains about `0.00300`. This supports a numerical
choice, not release qualification.
The same combined numerical and flux choice passes the frozen nonspinning and
strong antialigned-spin ODE gates. The nonspinning waveform is unchanged;
the antialigned fitted public difference increases from `0.000203842` to
`0.000212727`. These samples do not establish a full-domain bound.
At four early aligned-spin public states, the source-backed native torque
differs from a sampled public angular-momentum derivative by at most
`6.42e-7` relative on two output grids. This does not measure direct public
flux parity or explain the remaining accumulated waveform difference.
The pinned-source Hamiltonian coefficient initializer now feeds a verified
compiled adapter to the installed wheel's exported Hamiltonian wrapper. At
eight identical states, source and wheel Hamiltonians agree bitwise. The
paper-derived native value differs by up to `7.18e-6` relative at the latest
strong-spin state. A frozen two-factor diagnostic attributes that difference
to the pinned source's historical logarithmic-derivative convention and its
shorter calibration decimals: applying both in a development-only native
variant matches the wheel within `8.9e-16` absolute across all eight states.
This establishes the tested component path, not the wheel's private generator
initialization or a qualified waveform.
On the same aligned-spin waveform, applying only those shorter source
calibration decimals to the existing historical-derivative diagnostic lowers
peak-fitted complex relative L2 from `0.00299749` to `0.00271301`, with a
passing three-tolerance ODE check (`2.96e-7` refined complex L2). A frozen
early-state comparison reduces the maximum radial-momentum field gap on the
finer public grid from `4.70e-4` to `7.47e-5` relative, while the sampled
torque gap remains near `6.5e-7`. The remaining inspiral difference and open
domain, attachment, and ROM gates keep this source-convention variant on the
feature branch.
On the two boundary rows tested with the same source-backed flux and a `1 M`
NQC window, the rounded values also pass numerical convergence, but slightly
increase peak-fitted public waveform error. A fixed-flux initial-state check
reduces circular radius and angular-momentum gaps while leaving roughly
`2e-7` in inward momentum unexplained. The historical finite radial-response
step accounts for less than 1% of that momentum gap in an idealized native
check. These results prevent treating source-convention agreement as release
qualification.
A [pinned direct-source flux and initial-condition diagnostic](evidence/2026-09-24/source_flux_initial.json)
compiles a private snapshot of `lalsuite-v7.26.15`, the installed
`lalsimulation` 6.2.1 wheel's own clean VCS commit. Its checked SEOBNRv4
source and public-header files also have the same bytes at the older
`lalsimulation-v6.2.1` tag. At two identical circular states, source and
wheel flux per symmetric mass ratio differ by at most `2.27e-10` relative;
the source-calibrated native flux differs from source by at most `2.0e-9`.
The pinned-source full generator returns exactly the direct source legacy
initial state. Its initial radial momentum nevertheless differs from
the compiled public v4 initial state by `1.41e-7` to `2.53e-7` in absolute
value. Both `-O0` and `-O2` source builds return the same initial states, and
the compiled public first state is identical on three output grids. A
`-ffast-math` source-build control moves the initial state by more than this
gap but does not reproduce the wheel, so its exact numerical cause remains
open. At eight frozen noncircular trajectory states, including strong aligned
spin through `M Omega=0.12`, the source's raw pre-NQC `(2,2)` mode and native
mode agree within `1.8e-10` complex relative error and near-roundoff phase;
their supplied real energies agree within `1.2e-16`. This establishes the
tested source-mode component, not private wheel-mode parity, whole-waveform
accuracy, or domain validation.
The direct source evolution flux is now checked mode by mode at those eight
noncircular states and at the two circular starts. The previous native choice
misses total source flux by up to `27.8 ppm`, almost entirely from `(4,3)` and
`(8,8)`. The non-HM v4 source puts the higher `(4,3)` Kerr terms in `rho43^4`
and has no generic leading spin term in even `ell=8` modes. With those two
explicit diagnostic choices, all ten native/source flux comparisons agree
within `3.6e-10` relative, below the frozen `1e-8` gate. This closes the
tested source flux component; it does not establish compiled-wheel flux parity
or a qualified waveform.
The frozen three-level aligned-spin replay with precisely those two choices
passes its numerical gate (`3.64e-7` refined complex relative L2) and lowers
the fitted public-waveform difference from `0.0030000` to `0.0028067`. The
inspiral improves while the post-peak window is effectively unchanged. The
remaining waveform residual, untested domain, and open attachment and ROM
gates keep the variant on this feature branch.
The [pinned source/wheel dynamics comparison](evidence/2026-09-24/source_wheel_dynamics.json)
finds initial `pr_star` differences of `1.1e-7` to `2.5e-7`, but only
`7.0e-6` to `2.1e-5 rad` orbital-phase differences by `M Omega=0.12` across
three rows. The exact optimized v4 C evolution field agrees with the native
source-convention field at six fixed states to about `1.5e-11` relative in
torque, and the regular/optimized C fluxes agree within `2.5e-10` relative;
see the [optimized-field diagnostic](evidence/2026-09-24/optimized_source_field.json).
The [full generated mode comparison](evidence/2026-09-24/source_wheel_mode.json)
then isolates a larger source/wheel difference in their fitted NQC
coefficients. Before attachment, multiplying the wheel mode by the ratio of
the two NQC factors reduces source/wheel complex relative L2 from `0.00348`
to `6.21e-6` on the strong-spin row. This identifies NQC fitting as the main
source/wheel mode discrepancy on that window; it does not yet establish why
the two builds return different coefficients or qualify the native attachment.
In a [pinned-source compiler control](evidence/2026-09-24/source_nqc_build_controls.json),
`-O0`, `-O2`, and `-O3` give identical NQC coefficients on all three rows.
`-ffast-math` changes them substantially and moves the nonspinning fit closer
to the wheel, but does not reproduce the wheel, especially at strong spin.
The wheel's actual compilation settings and the coefficient gap's exact cause
remain unidentified.
The [native/source/wheel NQC replay](evidence/2026-09-24/native_source_nqc.json)
reproduces the frozen strong-spin native waveform bitwise and places its five
active coefficients between the locally compiled source and pinned wheel:
their L2 distances are `2.77` (native/source), `5.45` (native/wheel), and
`8.14` (source/wheel). Both the compiled source and wheel use the wheel's
hash-pinned GSL libraries. The native match-input and source/wheel build
differences still require direct isolation before this can be qualified.
The [instrumented pinned-source input capture](evidence/2026-09-24/source_nqc_inputs.json)
reproduces its uninstrumented coefficients bitwise. A
[native/source input replay](evidence/2026-09-24/native_source_nqc_inputs.json)
finds `54` to `230 ppm` differences among amplitude and orbital scalars at
matching; a pinned-GSL fit on the captured inputs produces a five-coefficient
gap of `2.742` (L2), close to the actual `2.768`. The
[crossed-input diagnostic](evidence/2026-09-24/mixed_nqc_inputs.json)
attributes most amplitude-coefficient drift to pre-NQC mode samples and most
phase-coefficient drift to orbital-state samples on this row. A `0.004768 M`
equal-frequency shift does not remove the curvature or coefficient gap.
The [source/wheel output-grid controls](evidence/2026-09-24/source_wheel_nqc_grid.json)
and [finer continuation](evidence/2026-09-24/source_wheel_nqc_fine_grid.json)
find a non-monotonic five-coefficient gap across six steps from `0.125` to
`0.00390625 M`: `8.139`, `1.800`, `3.284`, `1.318`, `19.582`, and `6.518`
(L2). Neither build's NQC coefficients have converged on these grids, so
the source/wheel build-numerics cause and the native attachment remain open.
The [source peak-grid capture](evidence/2026-09-24/source_nqc_peak_grid.json)
shows that two runs with the same orbital peak and internal step still differ
by `0.413%` in amplitude spline curvature at matching and `3.937` (L2) in
active NQC coefficients, despite at most `0.15 ppm` nonphase value drift.
The [fixed-window replay](evidence/2026-09-24/source_nqc_local_windows.json)
reproduces each full-series GSL fit exactly with a `0.5 M` or wider local
natural-cubic window; the output-grid drift is not a distant spline-boundary
effect. A [fixed-coarse/high-rate study](evidence/2026-09-24/source_nqc_high_rate.json)
varies only the pinned source's high-rate resampling factor. Its last two
steps share the same peak and match time, but their source GSL coefficients
differ by `25.05` (L2) as amplitude curvature changes by `2.69%` while the
nonphase values move by at most `0.30 ppm`. A diagnostic degree-four local
polynomial fit over `±2 M` reduces that pair's difference to `0.078`; at a
common match time, its three successive differences are `0.076`, `0.121`,
and `0.078`. These counterfactual fits are not a qualified NQC prescription.
The [additional-spin high-rate check](evidence/2026-09-24/source_nqc_high_rate_domain.json)
finds the same sensitivity on nonspinning and mild mixed-spin rows: at a
common match time, successive `±2 M` polynomial coefficient differences
decrease from `0.023` to `0.011` and from `0.026` to `0.012` (L2), while the
source cubic differences increase on the final refinement.
An [independent-integration derivative check](evidence/2026-09-24/source_placement_nqc_dense.json)
on the frozen current native strong-spin candidate finds a maximum nonphase
second-derivative error of `1.83e-8` for its GSL cubic, versus `4.20e-10` for
a diagnostic degree-six `±0.5 M` fit. The DOP853 reference uses the same
provisional equations, so this isolates native numerical differentiation on
one row; it neither validates source physics nor establishes a replacement
NQC rule. The project Python 3.13 environment matches the frozen LAL and
GSL binary hashes; the earlier system Python 3.12 wheel was a different build.
The [pinned-source shared-grid control](evidence/2026-09-24/source_nqc_shared_grid.json)
replays the two finest source grids exactly. Their amplitude samples differ by
at most `3.0e-9` within `±2 M` of matching, but their natural-cubic fitted
coefficients differ by `25.05` (L2). Decimating the finest run's own samples
changes its fit by `42.44` (L2), while removing its unmatched endpoint samples
changes nothing. This isolates a large grid-density sensitivity in the source
fit, but does not establish a validated replacement or source physics accuracy.
The [regular-source dense-field controls](evidence/2026-09-24/source_nqc_dense_field_strong_aligned_spin.json)
also compare the source's captured NQC cubic derivatives with explicit
pre-NQC modes evaluated along DOP853 integrations of the pinned **regular v4**
field, started from the source's interpolated state at matching. In the
[nonspinning](evidence/2026-09-24/source_nqc_dense_field_nonspinning.json),
[mild mixed-spin](evidence/2026-09-24/source_nqc_dense_field_mild_mixed_spin.json),
and strong aligned-spin rows, source cubic amplitude curvature differs from
the centered dense-field estimate by `3.23%`, `4.92%`, and `4.81%`, respectively.
Those figures use the factor-32 high-rate grid; at the source's default
factor-8 grid the respective absolute differences are `0.0011%`, `0.071%`, and
`0.326%`, then generally grow or change sign as sampling is refined.
The explicit source mode reproduces all 1,024 captured amplitudes within
`1.0e-10` in each row's `±2 M` window. The captured orbit's first derivative
also differs from its compiled vector field by up to `4.1e-7` among the three
rows. The regular generator resamples its adaptive orbit to a uniform grid
with a cubic spline before the NQC inputs are splined again; these controls
show a source trajectory/differentiation discrepancy, but cannot assign its
share to either interpolation stage or validate a replacement prescription.
The [frozen coefficient propagation](evidence/2026-09-24/source_nqc_coefficient_field.json)
replays all nine source NQC fits within `2.5e-4` absolute coefficient error
and then substitutes regular-field DOP853 derivatives at the same source
match states. At factor 32, the largest individual coefficient changes by
`2.03%` (strong aligned), `3.62%` (nonspinning), and `3.45%` (mild mixed).
Strong-spin source coefficients move `25.05` in L2 from factor 16 to 32,
whereas the dense-field counterfactual moves `0.0082` at the same match time.
This diagnoses the coefficient sensitivity; waveform impact and a validated
replacement still require independent checks.
The [source-state factor impact](evidence/2026-09-24/source_nqc_factor_impact.json)
then applies both coefficient sets to the same pinned-source pre-NQC arrays.
Within `2 M` before matching, the maximum complex `(2,2)` correction ratio
differs by `0.689%` (strong aligned), `0.252%` (nonspinning), and `0.558%`
(mild mixed) at factor 32. These are controlled source-mode counterfactuals,
not full-waveform residuals or evidence that the substituted fit is better.
At the source's default factor 8, the dense-field coefficients also move the
source-state NQC factor closer to the pinned wheel's public NQC coefficients:
the strongest pre-match complex-factor gap falls from `0.931%` to `0.512%` on
the aligned row, `0.0999%` to `0.00326%` on the nonspinning row, and `0.0973%`
to `0.0816%` on the mild mixed row. This narrows one source/wheel numerical
gap but does not identify the wheel's internal differentiation rule.
The [complete source-generator override](evidence/2026-09-24/source_nqc_whole_mode_override.json)
replays the frozen source and wheel `(2,2)` mode hashes, then substitutes only
the dense-field NQC coefficients into the compiled source generator. The
peak-fitted whole-mode complex L2 against the wheel drops from `6.48e-4` to
`2.97e-5` (nonspinning), `6.54e-4` to `5.33e-4` (mild mixed), and `5.80e-3`
to `3.19e-3` (strong aligned). The strong-spin discrepancy and small post-peak
regressions remain; this is a numerical attribution, not release validation.
The [wheel-coefficient override](evidence/2026-09-24/source_nqc_wheel_override.json)
keeps the same source generator and injects only the pinned wheel's reported
`(2,2)` NQC coefficients. Its peak-fitted whole-mode complex L2 falls to
`1.13e-5` (nonspinning), `2.46e-5` (mild mixed), and `8.58e-6` (strong
aligned). This bounds the residual from dynamics, raw modes, and attachment
on these three rows and isolates the source/wheel NQC fit as the dominant
whole-mode mismatch. Wheel coefficients are black-box reference outputs,
not native model data or a valid release prescription.
The [q4/q8 attachment comparison](evidence/2026-09-24/source_q4_q8_attachment.json)
finds that the source and native continuous NQC match offsets from orbital peak
agree to floating-point precision. The compiled source's boundary also agrees
with the independently inferred public ringdown origin within `0.00041 M`
after each waveform's own amplitude peak is fitted. In the **frozen baseline**,
the native three-sample fitted peak is `0.03777 M` earlier than source on q4
and `1.32536 M` later on q8 relative to orbital peak. The baseline's largest
coarse samples fall `0.296875 M` and `1.640625 M` after attachment, versus
source/public samples within `0.0625 M`. The later peak-preserving ringdown
variant has a different q8 fit and must not be mixed with that baseline.
Coarse-grid fits alone do not establish continuous-peak timing or an NQC cause.
Source and wheel have separate NQC coefficient drift, and neither is a
qualified native ROM training generator.
The [wheel-state NQC replay](evidence/2026-09-24/source_nqc_wheel_state_fit.json)
shows that the pinned source's first-decline peak rule returns a time
`0.0221 M` after the true cubic-spline maximum on the strong aligned-spin
row because its derivative interval is unbracketed. It returns the same
source-field peak time on the wheel's public high-rate states. Replacing
source states with wheel states reduces the source-fit-to-wheel-coefficient
L2 gap from `8.14` to `1.52` when the source generator's parameter construction
and per-state calls are replayed exactly. The pinned source C NQC routine
confirms the source-input reconstruction within `2e-6` per coefficient and
the wheel-state replay within `2.5e-5` of the Python/GSL fit. A simpler
algebraically equivalent adapter leaves a `0.215` L2 source control error:
the NQC fit is sensitive to floating-point operation order. The compiled
source-generator control shows that its public high-rate radius and radial
momentum equal its captured private NQC inputs. Source and wheel public
high-rate states differ by up to `8.89e-6` near matching.
The [three-row exact-source replay](evidence/2026-09-24/source_nqc_exact_rows.json)
confirms source C fit controls within `1.9e-6` L2 and leaves wheel-coefficient
gaps of `0.0687`, `0.220`, and `1.524` L2 on nonspinning, mild mixed-spin,
and strong aligned-spin wheel states, respectively.
The [pinned-wheel spline capture](evidence/2026-09-24/wheel_nqc_spline_capture.json)
now closes those gaps at the fit boundary: the pinned source C routine
reproduces all three wheel coefficient vectors to `1.01e-12` L2 or better
when given the wheel's actual private GSL arrays. The private NQC radius and
radial momentum equal public high-rate states to machine precision. A
single-factor replay attributes most of each earlier coefficient gap to
pre-NQC amplitude samples that differ from source explicit-state evaluation
on those same states by at most `1.61e-10` near matching. The explicit-state
adapter reproduces its own source generator's captured amplitude and
frequency exactly in the strong-spin control. The origin of the remaining
source-versus-wheel sample difference is unresolved. `-O0`, `-O2`, and `-O3`
source probes give the same amplitudes; `-ffast-math` worsens their RMS wheel
match on all three rows. Captured wheel arrays
remain black-box diagnostic outputs and do not supply a native prescription.
The aligned-spin trajectory replay finds a native-minus-public orbital phase
span of at most `1.27e-4 rad` from `M Omega=0.0425` through `0.13`, compared
with `3.36e-5 rad` maximum public-grid sensitivity. Over the later sampled
frequencies, an initial epoch-based public mode mapping gave raw pre-NQC `(2,2)`
amplitude and frequency gaps of `87 ppm` and `672 ppm`. The pinned-source
output-length rule places the public high-rate mode one sample earlier and
reduces those maxima to `0.923 ppm` and `92.0 ppm` on the same five frequencies.
This correction removes that output-index ambiguity, but it does not explain
the `0.0029975` fitted whole-waveform residual by itself.
The pinned wheel exposes no callable raw `(2,2)` mode at an explicit state, so
this is neither direct same-state parity nor an exact residual decomposition.
A three-grid public-dynamics check finds several-ppm differentiation sensitivity
in the remaining aligned torque residual near `M Omega=0.12`; that sampled
derivative is not direct compiled-flux parity. The pinned `(2,2)` coefficient
assignments match the native formula, but the full mode, NQC attachment, and
generator remain open.
A direct five-coefficient NQC diagnostic against the pinned public v4 binary
also reproduces both frozen native waveforms but finds `0.16–0.34%` aligned
and `0.36–1.55%` antialigned coefficient differences. Because the two fits
use different trajectories and pre-NQC modes, this remains an attachment
diagnostic rather than a localized NQC-formula error. Substituting the pinned
LAL source's full-precision targets into the same native matching matrices
changes the coefficients by at most `1.1%` and `6.0%` of those respective
public gaps in absolute magnitude. Target rounding does not explain the main
discrepancy; matching inputs and differentiation remain open.
On exact-hash native replays, replacing only the local phase-basis derivative
matrix with the full-series natural-cubic matrix slightly increases the
phase-coefficient distance from the public values on both strong-spin rows.
This rules out that matrix choice alone as the observed gap's explanation;
the distinct upstream inputs remain open.
The pinned wheel's GSL natural-cubic evaluator gives the same full-series
phase-basis derivative matrix on the exact native NQC samples, and applying
the source targets to those GSL matching systems also fails to close the
public coefficient gap. The wheel's V4 NQC routine remains private, so the
GSL check alone is same-input spline and matching evidence. A separate
pinned-source C harness now calls the V4 NQC routine on basis-equivalent
native samples. Its five coefficients agree with the GSL replay to at most
`1.4e-6` absolute on the aligned row and `1.0e-8` on the antialigned row.
The C-to-public coefficient distances remain `5.6238` and `0.66555` (L2).
This establishes source-compiled matching parity on native inputs, not
identity with the private wheel build or parity of its upstream inputs.
The pinned wheel's public C API also reproduces the same wheel's Python NQC
coefficients to `5.4e-11` or better on both rows, ruling out binding
marshalling as the observed gap. The locally compiled full generator does
not reproduce the wheel's evolution, so only the wheel remains the pinned
binary oracle for whole-waveform comparisons.
The [direct public-dynamics replay](evidence/2026-09-24/README.md) reconstructs
native pre-NQC orbital scalars from the frozen NQC samples. A fixed-time
comparison shows antialigned residuals near the inferred peak, but comparing
at equal orbital frequency reduces the radius and radial-momentum differences
to at most `3.4 ppm` and `47 ppm` on the two strong-spin rows. Their nearly
constant peak-relative time offsets explain most of the fixed-time signal.
A prespecified NQC match-time scan substantially reduces the antialigned
coefficient gap but not the aligned one; it is a sensitivity check with targets
held fixed, not a model correction. Dividing the public pre-attachment `(2,2)`
mode by its returned NQC factor gives raw amplitudes within `3.1 ppm` on the
antialigned row. The aligned row's nominal `38–87 ppm` amplitude discrepancy
falls below `1 ppm` under the pinned-source high-rate index at the frozen
`0.015625 M` step. Those mode-grid calls requested `0.015625 M` or
smaller output steps, whereas the frozen whole-waveform and public coefficient
comparison requested `0.125 M` and internally produced `0.015625 M` high-rate
dynamics. The [matched-output reference check](evidence/2026-09-24/README.md)
reproduces the latter coefficients exactly and finds fixed-frequency radius
and radial-momentum differences of at most `4.0` and `49.6 ppm`; it also finds
percent-level public coefficient changes when only the requested output step
changes. A [controlled native output-grid replay](evidence/2026-09-24/README.md)
keeps the native restart and `0.015625 M` high-rate step fixed: its five NQC
coefficients are identical across four output steps, whereas the public fit
changes by up to `0.27%` and `1.69%` on the two rows. The finer-grid mode
result is therefore not a same-run NQC input parity check. Raw mode slopes,
phase evolution, and centered matching inputs still require source-level
parity checks; neither the mode nor the remaining attachment difference is
qualified.

Two additional frozen stress rows now expose a much larger domain gap. The
source-convention candidate misses the q8 high-aligned-spin refined complex
numerical budget (`1.27e-6` versus `1e-6`) and differs from the pinned public
whole mode by `0.40226` complex relative L2 after amplitude-peak anchoring.
A further frozen ODE-tolerance step on the same q8 output grid reduces the
direct complex difference to `3.41e-7`, below that numerical budget; this
does not establish physical fidelity or domain-wide convergence.
A q4 mixed-spin row passes the numerical gate but differs by `0.01154`.
At q8, the native and public initial states at the same orbital frequency
agree to a few `1e-8`, and orbital phase spans through `M Omega=0.13` agree
within `2.1e-5 rad`. The large waveform gap is therefore downstream of the
tested pre-peak orbital dynamics. Anchoring each mode at its own prepeak
`M omega22=0.1` crossing reduces the q8 whole-mode difference to `0.00366`;
its amplitude peaks occur `1.34 M` apart, explaining most of the much larger
peak-anchored discrepancy. The remaining postpeak difference is under
attachment diagnosis. A development-only ringdown amplitude-peak bound derived
from the published ansatz reduces the q8 peak-time gap to `0.034 M` and its
peak-fitted whole-mode L2 to `0.01137`, but leaves a `0.00584` postpeak L2
under frequency anchoring. The ansatz-derived condition is now wired into the
feature-branch Torch and JAX attachment paths. A frozen integrated replay
reproduces the q8 counterfactual and unchanged q4 mode hashes exactly, with
the same public comparison metrics. This verifies the implementation route,
not domain-wide calibration of the new condition. The untouched q4 row has
only `0.051 M` peak lag
but still `0.00316` whole-mode and `0.00998` postpeak L2 at the same frequency
anchor. Its late QNM decay and phase slopes agree closely with the public
waveform, while an amplitude and phase offset settles through the attachment
transient. See the
[frozen stress evidence](evidence/2026-09-24/README.md).
The current peak-preserving q8 waveform also passes one additional frozen
ODE-tolerance step at `0.5 M`: direct complex relative L2 is `3.4133e-7`,
with unchanged event times. Further grid and domain-wide numerical checks,
physical waveform agreement, and ROM qualification remain open.
On the same q8 row, halving the requested output step from `0.5 M` to
`0.25 M` at the tighter tolerance leaves all nested mode samples and event
times unchanged. Both runs use the same `0.015625 M` internal step; this is
one-row grid evidence, not a domain-wide numerical bound.

The frozen q2 mixed-spin development row compares the native full `(2,2)`
mode with hash-pinned compiled LALSuite source and the installed public wheel
on their orbital-peak clocks. After one early phase alignment, the
amplitude/phase-interpolated full-mode complex relative L2 is `4.81e-4`
against source, while wheel/source is `5.72e-4`. A hash-matched native replay
with tighter ODE tolerance changes the q2 mode by only `4.32e-9` complex
relative L2, with unchanged sampled event times. Thus the q2 source residual
is not explained by measured ODE error. Halving the q2 requested output step
also leaves all shared complex samples identical at the same internal step.
The two reference binaries themselves differ, and this development row is
not a release holdout or a domain bound;
see the [frozen evidence](evidence/2026-09-24/README.md).
At the same q2 point, requesting `0.25 M` instead of `0.5 M` changes the
compiled source mode by `6.39e-4` complex relative L2 after peak-clock and
early-phase alignment; the public wheel changes by `1.31e-4`, and their
`0.25 M` difference is `1.33e-3`. These reference-grid effects prevent the
single `0.5 M` comparison from defining a physical release bound.
The compiled source's five active q2 NQC coefficients change by `0.426%`
between those steps, versus `0.018%` for the public wheel; native/source
at `0.5 M` differs by `0.159%` in coefficient space. Applying the
source/wheel NQC-factor ratio to the wheel mode on the same returned public
states through `-25 M` before the wheel peak reduces its complex relative L2
against source from `4.43e-4` to `2.36e-5` at `0.5 M`, and from `1.04e-3`
to `2.34e-5` at `0.25 M`. This coefficient-only counterfactual explains most
of the pre-attachment source/wheel difference on this row; it does not
resolve the native/source residual or qualify the generator.

Direct public-binary QNM calls now agree with the native fundamental frequency
and damping to float64 precision on both stress rows. The q4 native boundary
inputs are captured, but the public coarse-output mode does not expose its
hidden high-rate attachment inputs; a separate fine-output call changes the
trajectory and NQC coefficients. Replacing the historical compatibility QNM
with our independently generated Kerr root changes the q4 whole-mode complex
L2 by only `5.66e-8` and postpeak L2 by `2.27e-7` against the same public
waveform. That substitution cannot explain the residual on this row. The
remaining postpeak offset therefore
requires an exact coarse-call attachment or pre-attachment comparison.
An explicit development-only independent-Kerr-QNM choice now reproduces that
q4 counterfactual exactly, including the full mode hash and attachment inputs.
Neither row supports generator or ROM release qualification.

## Numerical evidence

From the repository root, using a development environment with Torch and NumPy:

```sh
PYTHONPATH=src python -m dev.native_rom.qualify_generator \
  --json-out /tmp/native-generator-evidence.json
```

The default scalar row is `mass1=30`, `mass2=20` solar masses, `chi1=chi2=0`,
and dimensionless orbital `omega_start=0.04`. Supply `--inputs /path/rows.json`
for a JSON list of rows with exactly those five keys. Run `--print-plan` with
the same command to inspect all settings without generating a waveform.

Each row runs five Torch CPU float64 solves with one thread: three ODE tolerance
pairs `(atol, rtol) = (1e-9, 1e-8), (1e-10, 1e-9), (1e-11, 1e-10)` at
`dt=0.5 M`, then the tight pair at `dt=0.25 M` and `0.125 M`. Ringdown is
`150 M`. Scientific choices stay fixed at `generic-leading-ell8`,
`bohe_total_energy`, no Pan-v8 or positive-spin-540 option, and final-spin fit
orders `(3, 4)`; no comparison selects coefficients or options.

The final pair in each refinement sequence must have direct complex and
amplitude relative L2 errors at most `1e-6`, nonincreasing complex error,
and peak/match/boundary time differences at most `0.05 M`. These are numerical
source budgets, not a physical-accuracy certification. Comparisons use exact
nested samples, include unmatched tails by zero filling, and never fit time,
phase, or amplitude. Changing the grid also changes the prototype's initial
step and restart sampling, so this checks the complete generator's sensitivity.

Each solve has a 600-second wall-clock cap, 200,000 derivative-evaluation cap,
20,000 accepted-step cap and 10,000 rejected-step cap. The default campaign has
at most five such solves per row. `--max-run-seconds` and `--max-evaluations`
can reduce resource bounds; a bounded-out run fails rather than passing with
missing evidence. Runtime is machine and input dependent. Ctrl-C records a
failed campaign. Use `/tmp` for evidence and logs.

An optional unaligned public-binary diagnostic requires exact module versions;
the local development environment currently supports:

```sh
PYTHONPATH=src python -m dev.native_rom.qualify_generator \
  --compare-lal --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out /tmp/native-generator-lal-evidence.json
```

This calls `SimInspiralChooseTDWaveform(..., SEOBNRv4)` only through its public
API, records wrapper and shared-library hashes, and does not inspect LAL source.
The native orbital-frequency peak defines its time origin; the LAL epoch stays
unchanged. Identical numeric phase input does not establish identical convention
semantics. This reference comparison reports discrepancies and makes no parity
pass claim. Requested reference errors fail the campaign.

JSON records exact configuration, input and source hashes, versions, run costs,
per-check errors and gates, and explicit false physics/training/production
qualification flags. Nonfinite outputs, exceptions, changed sources and failed
gates produce a nonzero exit. The runner exports no training arrays or ROM
asset. The adaptive prototype's historical-compatibility default uses
independently computed QNM values at knot locations inferred from a public
binary; that interpolation is not independent source material. An explicit
independent-Kerr diagnostic uses our own equation-solved representation, but
its broader parameter and waveform validation remains open.

The [q4/q8 pinned-source whole-mode comparison](evidence/2026-09-24/source_q4_q8_mode.json)
now separates a grid-comparison artifact from the model residual. Interpolating
complex samples onto the source grid had produced apparent post-attachment
amplitude errors of `1.275%` and `0.746%`; interpolating amplitude and
unwrapped phase lowers them to about `1e-4` and `7e-5`. The q8 phase remains
about `0.00245 rad` RMS over the later ringdown window. The independent
[source/wheel control](evidence/2026-09-24/source_q4_q8_wheel_mode.json)
agrees in post-attachment amplitude to better than `5e-10` on these rows.
The extended comparison covers the shared interval from about `-650 M` (q4)
or `-1197 M` (q8) before orbital peak through `100 M` afterward. With
amplitude/phase interpolation, whole-interval complex relative L2 is
`1.65e-4` (q4) and `0.00190` (q8); full-inspiral phase RMS before `-120 M`
is `0.000201` and `0.00211 rad`. Log-amplitude interpolation is a further
grid-bias control but does not improve every merger window. These two rows
remain audit-only. Wider parameter coverage, independent physics provenance,
and ROM training still need qualification.
The [q8 phase profile](evidence/2026-09-24/source_q8_phase_profile.json)
finds a smooth native-minus-source drift from `+0.00244 rad` at `-1100 M` to
`-0.00241 rad` at `+90 M`, relative to the fixed `-100 M` phase anchor.
The [fixed-trajectory NQC counterfactual](evidence/2026-09-24/source_q8_nqc_phase_counterfactual.json)
predicts source/wheel phase drift to `1.2e-5 rad` RMS, but the native/source
coefficient-only prediction at `-25 M` is `-0.00664 rad` versus an observed
`-0.00213 rad`. The residual is not explained by NQC coefficients alone;
the [pinned raw-phase decomposition](evidence/2026-09-24/source_q8_raw_phase.json)
now separates them at six late-inspiral probes. At `-25 M`, the anchored
trajectory phase contributes `+0.001882 rad` and NQC contributes
`-0.004345 rad`; the same-state raw-mode phase agrees to `2.9e-14 rad` and
the native orbital-frequency term is `+2.46e-6 rad`. The resulting
`-0.002461 rad` prediction differs from the observed `-0.002133 rad` by
`0.000327 rad`, which closes as separate native/source full-output
reconstruction errors. The post-hoc [output-interpolation control](evidence/2026-09-24/source_q8_output_interpolation.json)
reduces the source-side error at `-25 M` from `-0.000192 rad` to `-7.31e-8 rad`
under cubic interpolation. The [fine-trajectory control](evidence/2026-09-24/source_q8_fine_phase.json)
attributes `+0.000323 rad` to using coarse returned states instead of the
native attachment's fine states. Cubic output and fine states reduce the
pair remainder to `2.07e-7 rad` there. These sampling controls establish
internal consistency on one q8 row; they do not qualify the generator or ROM.

## Remaining promotion stages

A first independent [SXS:BBH:2123 comparison](evidence/2026-09-24/sxs_2123_plan.json)
uses archived Lev3/Lev4 `(2,2)` strain, an explicit frequency-crossing
alignment, and a native output-grid control. On the shared native support,
the fixed-anchor native/Lev4 complex relative L2 is `0.32436`, while the
Lev3/Lev4 control is `0.004160`; the native `0.25 M`/`0.125 M` difference
is `3.45e-8`. Under a separate post-hoc single time/phase adjustment on
the common `[-200,100] M` peak window, unweighted native/Lev4 overlap is
`0.9999468`, versus `0.9999533` for the pinned public model and
`0.999999965` between NR resolutions. The high optimized overlap is a
single-row diagnostic, not detector-noise-weighted faithfulness or a
release holdout. A separately frozen [ODE tolerance study](evidence/2026-09-24/sxs_2123_ode_plan.json)
and [extension](evidence/2026-09-24/sxs_2123_ode_extension_plan.json)
reduce the direct waveform difference to `1.29e-5`, still above the
predeclared `1e-6` numerical budget with the default natural-cubic sampler.
A fine-stage-only [sampling check](evidence/2026-09-24/sxs_2123_fine_sampling.json)
holds accepted ODE arrays identical and finds a `5.35e-6` waveform change.
Against a locally seeded independent
[DOP853 reference](evidence/2026-09-24/sxs_2123_sampling_reference.json),
the fine-stage RKF45 sample has at most `3.6e-15` state error near NQC
matching, versus `4.1e-12` for natural cubic. The frozen
[two-level RKF45 check](evidence/2026-09-24/sxs_2123_rkf_convergence.json)
then passes the same complex-mode numerical budget at `2.98e-9`, with
identical accepted trajectories and event times. At the same tightest
tolerance, its [fixed-anchor NR check](evidence/2026-09-24/sxs_2123_rkf_nr.json)
changes the native/Lev4 complex relative L2 only from `0.324417` to
`0.324419`. RKF45 remains an opt-in development diagnostic; this one-row
result is not a domain bound or a physical release gate. A subsequent
[frozen six-row boundary screen](evidence/2026-09-24/rkf_boundary_numerics.json)
uses the same fine-stage RKF45 sampler at `(atol, rtol)=(1e-14, 1e-13)`
and `(1e-15, 1e-14)` on previously declared mass-ratio, spin, and
start-frequency rows. All six pass the `1e-6` complete-mode numerical
budget; the worst complex relative L2 is `1.02e-8`, the worst amplitude
relative L2 is `3.81e-10`, and sampled match, peak, and boundary times
agree. This supports the numerical choice on those rows but does not
bound the intended domain or establish physical fidelity. See the
[evidence narrative](evidence/2026-09-24/README.md) for checksums,
conventions, and replay commands. The default numerical, physical-domain,
provenance, cross-backend, and independent-ROM gates below remain open.

A separately frozen [five-case SXS plan](evidence/2026-09-24/sxs_five_case_physical_plan.json)
selects low-eccentricity nonprecessing NR configurations across mass ratios
about 1–8 before retrieving their waveform files. Its
[fixed-frequency comparison](evidence/2026-09-24/sxs_five_case_physical.json)
finds native/highest-resolution complex relative L2 values of `0.317`,
`0.349`, `0.218`, `0.515`, and `0.399`; the corresponding next-highest/highest
resolution controls on native support are `0.00273`, `0.0142`, `0.0455`,
`0.00678`, and `0.00637`. All five native runs replay unchanged and the
earlier selected-row numerical controls are far smaller. A post-hoc
[public SEOBNRv4 control](evidence/2026-09-24/sxs_five_case_public_control.json)
on the same support finds complex errors `0.316`, `0.0775`, `0.283`,
`0.0213`, and `0.223` against NR. The public control therefore shares
some residuals but differs substantially on others, especially the
antialigned q≈3 case. These fixed-crossing errors are not maximized,
detector-noise-weighted faithfulness; NR resolution differences do not
bound every NR systematic. The comparison does not pass a physical release
gate or establish why the native and public models differ.
An explicitly post-hoc, bounded
[time/phase control](evidence/2026-09-24/sxs_five_case_alignment.json)
on the `[-200,100] M` peak window finds unweighted native/NR overlaps
`0.999863`, `0.999990`, `0.996270`, `0.999781`, and `0.999300` in the same
five-case order. Public/NR controls are `0.999919`, `0.999986`, `0.993603`,
`0.999856`, and `0.995644`. The q≈3 fixed-anchor `0.515` complex error,
for example, falls to `0.0209` after a `4.41 M` shift and phase rotation;
its native/public overlap is `0.999915`. This shows the earlier large
fixed-anchor errors are largely alignment offsets, but the control is
unweighted, uses short late-inspiral support, and was selected after the primary
result. It cannot replace the published detector-weighted, mass-dependent
validation or qualify either ROM.

A frozen [q≈5 earlier-start diagnostic](evidence/2026-09-24/sxs_2508_long_start.json)
begins the native and public models at orbital `M Omega=0.02`, below this
simulation's catalog reference frequency `0.02308`. The native/NR optimized
overlap on the same `[-200,100] M` peak window is `0.996245`, effectively
unchanged from the later-start `0.996270`; public/NR is `0.993648`.
Starting earlier therefore does not remove the remaining late-waveform shape
residual. This is still an unweighted, post-hoc single-case diagnostic, not
the detector-weighted 10–200 solar-mass physical release check.
The following [selected-case weighted sweep](evidence/2026-09-24/sxs_2508_weighted.json)
uses the NR metadata's settled frequency, a pinned Advanced LIGO design PSD,
time-domain tapering, and one global time/phase shift at 11 masses from 10 to
200 solar masses. Native/NR dominant-mode overlap stays above `0.99486`;
lower/higher NR resolution overlap stays above `0.99993`. The pinned public
SEOBNRv4 control reaches `0.98937`. Two alternative taper widths leave the
native minimum near `0.99486`. This passes a `0.99` threshold on the sampled
q≈5 rows, but it is a post-hoc single-case complex-mode proxy rather than
the paper's complete mass/parameter-domain and peak-time validation. The
physical generator and both ROM paths remain unqualified.

The [four additional earlier-start runs](evidence/2026-09-24/sxs_four_long_start_plan.json)
begin below their NR catalog reference frequencies and preserve the pinned
source hashes. On the late `[-200,100] M` window, their native/NR optimized
overlaps are `0.999914`, `0.999991`, `0.999837`, and `0.999300` for SXS
2496, 2499, 2139, and 2514. The resulting
[five-case weighted sweep](evidence/2026-09-24/sxs_five_case_weighted.json)
has native/NR minimum sampled overlaps `0.996651`, `0.998569`, `0.994862`,
`0.998946`, and `0.996964` in five-case plan order, with no time-shift
optimizer boundary hits. The public-model controls reach `0.989373` on
q≈5 and `0.988738` on q≈8; all lower/higher NR resolution controls exceed
`0.99992`. Alternative taper widths on four sampled masses per case preserve
native overlap above `0.99`. This is a post-hoc dominant-mode proxy over
five configurations and 11 discrete masses, not a continuum or full-domain
physical validation.

The [1 M⊙ follow-up grid](evidence/2026-09-24/sxs_five_case_dense_mass.json)
recomputes all five cases at every integer mass from 10 to 200 M⊙. Its
minimum native/NR overlaps are `0.996610`, `0.998562`, `0.994858`,
`0.998946`, and `0.996962` in plan order; the lowest occurs at 145 M⊙ for
q≈5. The original 11 masses reproduce within `1e-10`, no optimizer hits a
time-shift bound, and every NR resolution control remains above `0.99992`.
This dense post-hoc grid still does not bound the mass continuum or other
physical parameters.

A [q≈5 long-start tolerance study](evidence/2026-09-24/sxs_2508_tolerance.json)
finds `2.63e-8` direct complex relative drift between the tight baseline
and the next-looser solve, and at most `2.0e-9` change in the 145 M⊙ weighted
native/NR overlap across the three tolerance levels. Event-time budgets pass,
but the strict monotone error check fails near these float64 tolerances.
This supports a local numerical stability bound for the sampled waveform;
it does not remove the `0.994858` native/NR physical residual or establish
domain-wide convergence.

The independently selected [second five-case SXS holdout](evidence/2026-09-24/sxs_second_holdout_weighted.json)
fails its predeclared `0.99` dominant-mode overlap check: the q≈8 aligned-spin
SXS:BBH:2786 case reaches `0.950004` at 58 M⊙, while its NR-resolution
control exceeds `0.999694`. A post-hoc [pinned public SEOBNRv4 comparison](evidence/2026-09-24/sxs_2786_public_control.json)
reaches `0.995721` versus NR at 50 M⊙, compared with `0.950515` for the
native prototype under the same settings. This exposes a native-specific
problem in the default prototype. It remains outside production and cannot
seed a release ROM.

A post-hoc attempt to apply the previously source-backed physical choices to
this long-start case first hit two runtime limits. The first
[intervention plan](evidence/2026-09-24/sxs_2786_source_choice_plan.json)
timed out after 900 seconds because its fixed restart was 165 M after the
waveform began, leaving almost the entire 15,901 M inspiral to the fine
integrator. The [corrected near-peak plan](evidence/2026-09-24/sxs_2786_source_choice_near_peak_plan.json)
also [timed out](evidence/2026-09-24/sxs_2786_source_choice_near_peak_timeout.json)
after 900 seconds at extremely tight tolerances. The subsequent
[looser-tolerance replay](evidence/2026-09-24/sxs_2786_source_choice_loose.json)
completed in 154 seconds. Its detector-weighted overlap with the high-resolution
NR `(2,2)` mode is `0.995722` at 50 M⊙ and `0.995819` at 58 M⊙, compared with
`0.950515` and `0.950004` for the default prototype. It agrees with the pinned
public SEOBNRv4 waveform to `5.91e-5` anchored full complex relative L2 on
the common support. A [three-tolerance study](evidence/2026-09-24/sxs_2786_source_choice_convergence.json)
finds `8.38e-8` complex relative drift from middle to tighter tolerances.
The [191-mass post-hoc sweep](evidence/2026-09-24/sxs_2786_source_dense_mass.json)
has a `0.995701` minimum at 45 M⊙, with no time-shift optimizer at its bound.
A frozen development-only [calibration evaluator simplification](evidence/2026-09-24/sxs_2786_source_choice_optimization.json)
preserves that waveform to `1.24e-11` complex relative L2 and reduces the
measured end-to-end runtime to 101 seconds. This was a post-hoc selected case;
it is not independent validation or release qualification.

The [third prospectively selected five-case SXS holdout](evidence/2026-09-24/sxs_third_holdout_weighted.json)
uses the source-backed candidate. Four cases pass its frozen `0.99` dominant-mode
proxy; SXS:BBH:2569 reaches `0.989142` at 22 M⊙ while the low/high NR control
stays above `0.999924`. The [post-hoc pinned-public control](evidence/2026-09-24/sxs_2569_public_control.json)
reaches `0.989141` at the same mass and agrees with the native waveform to
`1.17e-4` anchored complex relative L2. This attributes the proxy failure to
a discrepancy shared with the public model under this protocol; it does not
turn the failed gate into a pass. A [10× tighter ODE replay](evidence/2026-09-24/sxs_2569_convergence.json)
changes the waveform by `6.65e-8` complex relative L2 and the 22 M⊙ overlap
by only `−2.87e-9`. The 2018 SXS run was not part of the NR catalog assessed
in the 2017 SEOBNRv4 paper. The remaining generator, frontend, and
independently trained ROM gates still need completion.
The [five-case pinned-public parity check](evidence/2026-09-24/sxs_third_public_parity.json)
passes its separately frozen post-hoc gates: native/public fixed-crossing
complex relative L2 is at most `3.47e-4`, and the largest difference in
NR overlap at each case's worst sampled mass is `2.33e-5`. This supports
implementation fidelity on those five rows, not physical accuracy across the
domain or a ROM qualification.

An independent [low-frequency phase-aligned peak-timing diagnostic](evidence/2026-09-24/sxs_five_case_peak_timing.json)
fits one time and phase shift in a 500 M early-inspiral window after NR
relaxation. Native/NR amplitude-peak lags are `-6.78`, `-9.94`, `-3.09`,
`-7.08`, and `-6.79 M` in the same order. The 300/800 M window controls
change some lags, and the q≈8 low/high NR control itself differs by `7.09 M`.
The exact early alignment window is not specified in
[Bohé et al., Section IV.A](https://arxiv.org/html/1611.03703), so these
lags are diagnostic and cannot be treated as a literal pass/fail of its
`5 M` peak-time criterion. They do expose an unresolved timing question;
neither the physical generator nor either ROM is release-qualified.

The independent ROM construction follows the published reduced-basis and
parameter-interpolation methods of [Pürrer (2014)](https://doi.org/10.1088/0264-9381/31/19/195010)
and [Pürrer (2016)](https://doi.org/10.1103/PhysRevD.93.064041).
[Bohé et al. (2017), Section VII](https://arxiv.org/html/1611.03703#S7)
describes the SEOBNRv4 ROM construction and its parameter patches. Independently
trained tables would define our own approximation to a qualified source model;
they would not become the authors' official ROM tables.

1. Qualify the physical generator against the published equations and independent
   references over the intended domain, resolve convention/provenance gaps, and
   establish gradients and numerical convergence. This runner covers only scalar
   CPU forward numerical stability; convergence alone cannot validate physics.
2. Freeze and hash that generator, construct independent ROM training data, and
   qualify an independent ROM against held-out generator waveforms, including
   parameter boundaries and interpolation derivatives. Identify it as an
   independently trained approximation rather than the official ROM asset.
3. Establish the same waveform, dtype/device, batching and derivative contracts
   in Torch and JAX, complete release validation and provenance review, and only
   then consider production registration and packaging.

Focused runner tests: `PYTHONPATH=src python -m pytest -q
dev/native_rom/tests/test_qualification.py`.
