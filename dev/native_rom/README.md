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
The private wheel fit inputs remain an unresolved release blocker.
The [three-row exact-source replay](evidence/2026-09-24/source_nqc_exact_rows.json)
confirms source C fit controls within `1.9e-6` L2 and leaves wheel-coefficient
gaps of `0.0687`, `0.220`, and `1.524` L2 on nonspinning, mild mixed-spin,
and strong aligned-spin wheel states, respectively.
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
asset. Its adaptive prototype currently uses independently computed QNM values
with black-box-inferred spline knots; resolving that provenance is still needed
for a wholly literature-specified generator.

## Remaining promotion stages

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
