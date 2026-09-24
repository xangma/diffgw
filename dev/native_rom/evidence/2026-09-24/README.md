# Strong-spin flux source audit and waveform checks

The later [strong-spin inspiral field](strong_spin_inspiral_field.json) freezes
four public strong aligned-spin states on two output grids, then evaluates the
native vector field at those same states. A separate
[start-alignment check](strong_spin_start_alignment.json) corrects an invalid
initial-state comparison in the field report: the earlier expression compares
the native warm-start endpoint at `M Omega=0.04` with the public first state
near `0.03162`. At the actual first frequency, native and public radius and
angular momentum agree within about `6e-8` and `5e-8`, respectively. Neither
comparison uses public state as a generator input.

The [frozen flux-choice check](strong_spin_flux_choices.json) and
[source-formula `(3,1)` check](mode_31_flux_placement.json) isolate mode
placements without fitting the public derivative. The pinned non-HM v4 source
puts the higher `(3,1)` Kerr-spin terms in `rho31`; changing that mode alone
moves the total torque by less than `0.1 ppm` at the tested frequencies. A
Taylor-`f33` counterfactual can close some error but disagrees with the pinned
v4 source, so it is not selected by numerical fit.

The same source explicitly overwrites the spin-square part of `rho32v2` for
non-HM version 4. The [predeclared same-state plan](mode_32_flux_placement_plan.json)
and [result](mode_32_flux_placement.json) change only that coefficient. At
`M Omega=0.045, 0.06, 0.09, 0.12`, native torque differences from the sampled
public derivative decrease from about `332, 721, 2256, 5208 ppm` to
`2.9, 6.5, 26, 59 ppm`. These are finite-difference comparisons, not direct
compiled-flux parity. The source revision and coefficient lines are pinned in
the plan; the implementation remains an explicit diagnostic choice.

The [frozen aligned-spin waveform plan](mode_32_whole_waveform_plan.json) and
[result](mode_32_whole_waveform.json) replay the previous baseline hash, then
apply only the `(3,2)` overwrite at three ODE tolerances. The independently
peak-fitted public waveform complex relative L2 decreases from `0.0198485` to
`0.0058721`; the refined native numerical difference is `3.15e-7` complex
relative L2, below the frozen `1e-6` budget, with all event gates passing.
The [antialigned boundary plan](mode_32_antialigned_plan.json) and
[result](mode_32_antialigned.json) likewise replay the prior baseline and pass
the numerical gate (`7.04e-7`), but its already small public comparison moves
from `0.0002039` to `0.0002225`. This mixed parameter behavior, the remaining
aligned-spin residual, and unresolved historical source and domain gates bar
release qualification. No ROM training or registration follows from these
diagnostics.

With historical `(3,2)` fixed, the [frozen `(2,1)`/`(3,1)` source-placement
plan](mode_21_31_32_flux_plan.json) and [same-state result](mode_21_31_32_flux.json)
find that the non-HM v4 `(2,1)` `rho` terms reduce the sampled strong-spin torque
gap to about `0.1–18.7 ppm` through `M Omega=0.12`; the `(3,1)` placement moves
total torque by less than `0.1 ppm`. The
[combined waveform plan](source_flux_whole_waveform_plan.json) and
[result](source_flux_whole_waveform.json) replay the `(3,2)` baseline hash and
improve the aligned-spin fitted complex relative L2 from `0.0058721` to
`0.0029975`. The refined numerical complex difference is `3.26e-7`, with all
gates passing. The inspiral and late-inspiral windows improve, but the
post-peak window worsens from `0.0004693` to `0.0008127`. Source placement
and one-row stability do not establish physical or domain qualification.

The [antialigned source-flux plan](source_flux_antialigned_plan.json) and
[result](source_flux_antialigned.json) replay the prior refined `(3,2)` waveform
hash. The three-tolerance numerical gate passes (`7.15e-7` refined complex
relative L2), and the fitted public difference moves from `0.0002225` to
`0.0002128`. This remains above the earlier `0.0002039` variant on that row;
the post-peak window is effectively unchanged. The source-faithful choices
therefore have mixed physical comparisons across rows despite numerical
stability at these boundaries.

The [frozen source-flux 1 M plan](source_flux_one_M_plan.json) and
[result](source_flux_one_M.json) change only the NQC phase-derivative window
on the aligned combined variant. Its refined ODE complex difference decreases
from `3.26e-7` at `2 M` to `2.41e-7` at `1 M`; all numerical gates pass.
The fitted public-binary difference changes from `0.00299749` to `0.00299999`.
The smaller window improves numerical convergence here but does not close the
physical residual or qualify the generator. Reproduce with `PYTHONPATH=src:.
python -m dev.native_rom.diagnose_source_flux_one_M --json-out
/tmp/source-flux-one-M.json` in the pinned development environment.

The [frozen early source-field plan](early_source_field_plan.json) and
[result](early_source_field.json) compare the source-backed native Hamiltonian
and flux at public strong aligned-spin states near `M Omega=0.032`, `0.035`,
`0.04`, and `0.05`. On public output grids `0.125` and `0.0625 M`, the largest
native torque difference from a five-point derivative of public angular
momentum is `6.42e-7` relative. The sampled public torque itself changes by
up to `2.11e-7` between grids. This bounds an early same-state vector-field
discrepancy at four samples, not direct public-flux parity or accumulated
trajectory and waveform error. The physical and ROM gates remain open.
Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_early_source_field --json-out
/tmp/early-source-field.json` in the pinned development environment.

The [frozen source-flux frequency-trace plan](source_flux_frequency_trace_plan.json)
and [result](source_flux_frequency_trace.json) replay the combined aligned-spin
waveform and both pinned public dynamics grids, then compare phase spans from
`M Omega=0.0425` at nine common frequencies through `0.13`. On the public
`0.0625 M` grid, the native-minus-public accumulated orbital phase reaches
`1.27e-4 rad`; the largest `0.125`-versus-`0.0625 M` public phase-span
difference is `3.36e-5 rad`. With the then-nominal epoch mapping, the native
raw pre-NQC mode amplitude differs by at most `87 ppm`, while its
gravitational-wave frequency differs by at most `672 ppm` relative to the
frozen reconstructed public raw mode. The exact source-backed mapping below
revises those two raw-mode figures on the same five frequencies. The prior fit
reports `0.0033308` local relative L2 in its inspiral window. These
comparisons do not decompose the fitted `0.0029975` whole-waveform residual
exactly or establish direct public-flux parity. The physical and ROM
gates remain open. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_source_flux_frequency_trace --json-out
/tmp/source-flux-frequency-trace.json` in the pinned development environment.

The [public mode-component capability plan](public_mode_component_capability_plan.json)
and [result](public_mode_component_capability.json) audit the pinned wheel and
`lalsimulation-v6.2.1` source for a raw `(2,2)` mode or factorized-amplitude
call at an explicit phase-space state. The relevant source routines are
`static` and absent from the wheel's exported symbols and Python bindings.
`SimIMRSpinAlignedEOBModes` generates a full waveform from binary inputs; its
signature does not accept an explicit EOB state. Thus no justified direct
same-state raw-mode comparison was performed. The separate exact-mapping
diagnostic below resolves the sampled output index without providing an
explicit-state mode oracle. A
separately compiled adapter would first need to reproduce the pinned wheel
numerically. Reproduce the read-only audit with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_public_mode_component_capability --source-repo
/path/to/pinned/lalsuite-git --json-out /tmp/public-mode-component-capability.json`.

The pinned v4 source's `rho22v2` through `rho22v10` assignments in
`LALSimIMRSpinEOBFactorizedWaveform.c` agree term by term with the native
`_rho22` expression when the diagnostic retains the cubic Kerr-spin term.
The source supplies `a = sigmaKerr_z = x1² chi1 + x2² chi2`, equal to native
`(1-2 nu) chiS + delta chiA`. This coefficient inspection does not validate
the full `(2,2)` mode or its NQC correction. In the tested aligned states,
the native `(2,2)` term contributes about 97.8–99.1% of total flux, so even
small amplitude or state-convention differences matter. The
[frozen public-torque sampling plan](public_torque_sampling_plan.json) and
[result](public_torque_sampling.json) exactly replay both earlier public
dynamics grids and add a third. At a shared time near `M Omega=0.12`, the
five-point public torque differs by up to `9.6 ppm` between adjacent output
steps, while the source-aligned native torque differs from the finest sampled
derivative by `-22.6 ppm`. The derivative estimate is grid-sensitive and is
not direct compiled-flux parity; it cannot by itself identify a `(2,2)`
coefficient error.

The [frozen NQC-coefficient plan](nqc_coefficients_plan.json) and
[result](nqc_coefficients.json) use the public v4 entry point's coefficient
output and replay the exact refined native waveform hashes before comparing
the five active `(2,2)` NQC coefficients. On the aligned row, the native
coefficient magnitudes differ from the public ones by `0.16–0.34%`; on the
antialigned row, by `0.36–1.55%`. The other five slots in the public v4 NQC
vector are zero, consistent with the three-amplitude/two-phase basis used
natively. This is a direct comparison of fitted coefficients, but each fit
uses its own trajectory and pre-NQC mode. The mismatch therefore cannot yet
be assigned to the NQC targets, spline differentiation, or preceding dynamics.
The same result transcribes the pinned source's full-precision `(2,2)` peak
fits and substitutes their five targets into the **recorded native matching
matrices**, holding the native trajectory, pre-NQC mode, match time, and
derivatives fixed. The largest target-relative change is about `4.9e-5` (the
frequency slope). The resulting coefficient shifts have at most `1.1%` of
the original aligned public gap and `6.0%` of the antialigned gap in absolute
magnitude; some shifts increase the gap. Thus printed-fit rounding cannot
account for the main coefficient difference on these native inputs. This
counterfactual does not isolate differences in the public matching matrices,
which use full-series GSL cubic splines, versus the tested native diagnostic's
local phase-basis polynomial, nor differences in trajectory and pre-NQC mode.
It does not establish coefficient parity or qualify attachment. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_nqc_coefficients --json-out
/tmp/nqc-coefficients.json` in the pinned development environment.

The [frozen phase-matrix plan](nqc_phase_matrix_plan.json) and
[result](nqc_phase_matrix.json) replay those same native waveform hashes and
hold each native phase matching right-hand side fixed. Replacing only the
local degree-six phase-basis derivative matrix with the full-series natural
cubic-spline matrix changes the phase-coefficient distance from the public
values from `3.7437` to `3.7995` on the aligned row, and from `0.06563` to
`0.06621` on the antialigned row (absolute coefficient L2). Thus this matrix
choice alone does not close either gap. The public routine sees different
trajectory and pre-NQC samples; this is a native-input counterfactual, not a
claim of compiled NQC parity. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_nqc_phase_matrix --json-out /tmp/nqc-phase-matrix.json`.

The [same-input GSL plan](nqc_gsl_same_input_plan.json) and
[result](nqc_gsl_same_input.json) replay the same hashes again and pass the
native NQC amplitude, product, phase-basis, and unwrapped phase samples
through the GSL natural-cubic library shipped with the pinned LAL wheel.
They apply the source's full-precision targets and phase sign convention to
the resulting matching systems. GSL and native full-series phase-basis
derivative matrices agree at the recorded precision on both rows; the first
two amplitude-product matrix rows also agree exactly, while the native
diagnostic's local second-derivative row differs by at most `1.8e-5`
relatively. The five-coefficient distance from the public fit is `5.6238`
versus `5.5635` for native on the aligned row, and `0.66555` versus `0.66502`
on the antialigned row. This shows that using the source's GSL interpolation
operations on the **native** samples does not close the public coefficient
gap. The private compiled LAL NQC function was not called on those samples,
and the public fit has different upstream inputs. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_nqc_gsl_same_input
--json-out /tmp/nqc-gsl-same-input.json`.

The [source-compiled C NQC plan](nqc_c_same_input_plan.json) and
[result](nqc_c_same_input.json) replay those two native waveforms once more,
check their frozen hashes, and call the exact pinned-source V4 C matching
routine through a small [development wrapper](../../reference/nqc_v4_source_wrapper.c).
The radius, radial momentum, and orbital frequency inputs are recovered
algebraically from the native NQC basis samples; the defining basis identity
holds to `5.6e-17` or better. The source-compiled C coefficients differ from
the pinned-wheel GSL replay by at most `1.33e-6` absolute on the aligned row
and `9.3e-9` on the antialigned row. Their distances from the public fitted
coefficients remain `5.6238` and `0.66555` in five-coefficient L2. This
validates the C matching operation on the native inputs but does not call
the private wheel routine, establish its exact build flags, or equate the
public and native pre-NQC inputs. Reproduce on macOS with the pinned source
checkout and headers using `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_nqc_c_same_input --source-root
/tmp/diffgw-lalsuite-pinned-53acc0c --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/nqc-c-same-input.json`.

The [public C-binding plan](nqc_public_c_binding_plan.json) and
[result](nqc_public_c_binding.json) compile a small
[public-API probe](../../reference/seobnrv4_public_c_probe.c) against the
same pinned wheel. Direct C calls reproduce the frozen Python-binding NQC
coefficients to `5.37e-11` absolute on the aligned row and `7.11e-13` on the
antialigned row. Python argument marshalling therefore does not account for
the observed coefficient gap. This public call still supplies the wheel's
own upstream evolution and does not expose its pre-NQC matching arrays.
The [direct source component check](source_flux_initial.json) now pins the
wheel's own clean [`lalsuite-v7.26.15` source commit](https://git.ligo.org/lscsoft/lalsuite/-/tree/lalsuite-v7.26.15/lalsimulation/lib).
The inspected SEOBNRv4 source and public-header files have the same bytes at
the earlier `lalsimulation-v6.2.1` tag. The source-compiled *full generator*
returns the direct-source legacy initial state exactly on two rows, but that
state differs slightly from the wheel's public first state. A `-ffast-math`
control demonstrates numerical sensitivity without reproducing the wheel or
establishing its actual build flags. Direct source and native raw pre-NQC
`(2,2)` modes agree within `1.8e-10` complex relative at eight fixed states,
including strong aligned spin through `M Omega=0.12`; this is source-component
parity, not a callable private wheel-mode check. The independently built
source waveform is not a substitute for the pinned wheel oracle. The same
adapter computes direct evolution flux and all 35 positive-m mode
contributions at ten frozen states. The earlier native choice differs by up
to `27.8 ppm` in total flux; `(4,3)` and `(8,8)` dominate. Moving the
higher `(4,3)` Kerr terms into `rho43^4` and omitting the generic `ell=8`
spin term, as in the pinned non-HM v4 source, reduces the largest native/source
flux difference to `3.6e-10` relative; all ten states pass the predeclared
`1e-8` source-component gate. The [waveform plan](source_placement_whole_waveform_plan.json)
then applies precisely these placements to the frozen aligned-spin waveform
for a separate [whole-waveform and numerical check](source_placement_whole_waveform.json).
The frozen baseline reproduces exactly. The candidate passes all numerical
gates, with `3.64e-7` refined complex relative L2 and `5.65e-9` refined
amplitude relative L2. Peak-fitted public-waveform complex relative L2
decreases from `0.00299999` to `0.00280674`; the inspiral window decreases
from `0.00333358` to `0.00311712`, whereas the post-peak window stays near
`0.0008139`. Direct source flux parity therefore does not close the whole
waveform residual or qualify the generator. Reproduce the waveform result in
the pinned environment with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_source_placement_whole_waveform --json-out
/tmp/source-placement-whole-waveform.json`. Reproduce the
C-binding control on macOS with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_nqc_public_c_binding --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --json-out
/tmp/nqc-public-c-binding.json`.

The [source/wheel dynamics plan](source_wheel_dynamics_plan.json) and
[result](source_wheel_dynamics.json) compare the exact pinned-source full v4
generator with its pinned wheel on three `0.125 M` low-rate grids. Their
initial `pr_star` differs by `1.1e-7` to `2.5e-7`; by `M Omega=0.12`, the
source-minus-wheel orbital-phase span is only `7.0e-6` to `2.1e-5 rad`.
The [optimized-source-field plan](optimized_source_field_plan.json) and
[result](optimized_source_field.json) call the optimized RHS actually selected
by the non-HM v4 generator at six fixed states. Native/source torque differs
by at most `1.5e-11` relative; the source's regular and optimized flux paths
differ by at most `2.5e-10` relative at these states.

The [full-mode plan](source_wheel_mode_plan.json) and
[result](source_wheel_mode.json) compare both complete generated `(2,2)`
modes and all ten NQC coefficients. Their source/wheel peak-fitted complex
relative L2 is `0.000648`, `0.000654`, and `0.005796` on the nonspinning,
mild mixed-spin, and strong aligned-spin rows. On the strong-spin row, the
same-index phase difference grows from `-0.00130 rad` to `-0.00728 rad` by
20 M before peak, while the NQC phase-coefficient differences are `+0.0833`
and `-4.6884`. A coefficient-only counterfactual multiplies the wheel mode
by the ratio of source and wheel NQC factors on the *same wheel trajectory*.
On the pre-attachment interval ending 20 M before peak, it reduces the
source/wheel complex relative L2 from `0.00348` to `6.21e-6`; the analogous
nonspinning and mild-spin residuals fall from `2.28e-4` to `1.77e-5` and
`2.44e-4` to `1.04e-5`. This isolates the NQC-fit difference as the dominant
source/wheel mode discrepancy on those windows. The coefficient difference's
build/numerical origin, native matching parity, attachment, broader domain,
and ROM training/holdouts remain open.
The [compiler-control plan](source_nqc_build_controls_plan.json) and
[result](source_nqc_build_controls.json) rebuild the same source adapter at
`-O0`, `-O2`, `-O3`, and `-O2 -ffast-math`. The first three produce identical
NQC coefficients at all three rows. Fast-math changes them and reduces the
active-coefficient L2 distance to the wheel from `0.2195` to `0.0133`
(nonspinning) and from `8.1388` to `4.4684` (strong aligned spin), without
reproducing the wheel. This control establishes sensitivity to floating-point
semantics; it does not reveal the wheel's compiler flags or justify adopting
fast-math as a physical or native-model choice.

The [native/source/wheel NQC plan](native_source_nqc_plan.json) and
[result](native_source_nqc.json) replay the frozen strong-spin native
source-placement waveform with coefficient capture. Both waveform array
hashes reproduce exactly. The five active coefficients have L2 distances
`2.77` native/source, `5.45` native/wheel, and `8.14` source/wheel. Thus the
native fit lies between the two independently generated C references on this
row, but neither matches it. The local source adapter and wheel link the
same hash-pinned wheel GSL library; the precise upstream fit inputs, build
floating-point behavior, and attachment remain unresolved. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_native_source_nqc --json-out
/tmp/native-source-nqc.json`.

The [source-input capture plan](source_nqc_inputs_plan.json) and
[result](source_nqc_inputs.json) intercept the pinned full generator's V4 NQC
call in a private, hash-verified source snapshot. Its [array asset](source_nqc_inputs.npz)
contains the exact pre-fit amplitude, phase, radius, radial momentum, and
orbital-frequency samples on the strong aligned-spin row. Instrumentation
leaves all ten coefficients bitwise unchanged from the uninstrumented source.
The [native-input plan](native_source_nqc_inputs_plan.json) and
[result](native_source_nqc_inputs.json) capture the same five fields from the
bitwise-replayed native source-placement waveform in a separate
[array asset](native_source_nqc_inputs.npz). At their respective NQC matches,
native minus source relative amplitude, radius, radial momentum, and orbital
frequency are `-54.1`, `+67.7`, `-229.7`, and `-75.2 ppm`. The pinned-GSL
reconstruction on those inputs yields a five-coefficient source/native gap of
`2.742` (L2), near the actual `2.768`; the reconstructed source coefficients
are within `1.30e-5` absolute of its captured C fit. This establishes that
small upstream input differences are materially amplified by this fit, but
does not yet close every C/native matching-method difference.

The [crossed-input plan](mixed_nqc_inputs_plan.json) and
[result](mixed_nqc_inputs.json) index-align the two high-rate grids and fit
mode samples against either trajectory's orbital scalars. It finds the
amplitude-coefficient difference mainly in pre-NQC mode samples and the
phase-coefficient difference mainly in orbital-state samples; the crossed
fits are counterfactual, not self-consistent waveforms. Aligning the match
by orbital frequency requires a `0.004768 M` shift and makes the amplitude
and radius values agree within `0.4 ppm`, yet the five-coefficient gap grows
from `2.742` to `3.481` (L2). A simple time offset does not explain the
derivative and shape differences.

The [source/wheel grid plan](source_wheel_nqc_grid_plan.json) and
[result](source_wheel_nqc_grid.json) replay both five-coefficient fits at
requested output steps `0.125`, `0.0625`, `0.03125`, and `0.015625 M`.
Their L2 gaps are `8.139`, `1.800`, `3.284`, and `1.318`, respectively.
The [finer-grid plan](source_wheel_nqc_fine_grid_plan.json) and
[result](source_wheel_nqc_fine_grid.json) add `0.0078125` and `0.00390625 M`,
where the gaps are `19.582` and `6.518`. Both builds' own coefficients also
move strongly and non-monotonically with output step; these six grids show no
converged NQC fit. Build flags alone cannot be inferred as the sole cause of
the original `0.125 M` discrepancy. None of these diagnostics qualifies the
physical NQC targets, complete waveform, model domain, or ROM holdouts.
The [source peak-grid plan](source_nqc_peak_grid_plan.json) and
[result](source_nqc_peak_grid.json) intercept three of those source runs to
compare their actual pre-fit inputs. All three use the same `0.015625 M`
internal step and the same match position within its grid cell. The peak
moves by `0.0625 M` between requested steps `0.125` and `0.0625 M`, then
stays fixed at `0.03125 M`. Even in the latter pair, the four nonphase input
values at matching change by at most `0.15 ppm`, while amplitude spline
curvature changes by `0.413%` and active NQC coefficients change by `3.937`
(L2). The earlier pair has at most `0.031 ppm` value drift but `0.205%`
curvature drift and `3.236` coefficient drift. Thus both peak placement and
derivative sensitivity contribute; the identical peak in the latter pair
excludes a peak-time-only explanation. This is a source-internal numerical
diagnostic, not an alternate NQC prescription.

The [source local-window plan](source_nqc_local_windows_plan.json) and
[result](source_nqc_local_windows.json) replay the exact captured source arrays
and targets with natural-cubic splines truncated symmetrically around the
match. A `0.5 M` half-width (64 samples) and every wider tested window give
bitwise-identical fitted coefficients and amplitude curvature to the
full-series pinned-GSL replay on all three requested output grids. A `0.25 M`
half-width differs by at most `3.9e-6` in any coefficient; `0.125 M` introduces
up to `0.148` and `0.0625 M` up to `28.5`. Distant spline boundary conditions
therefore do not explain the observed output-grid instability, while very
short local windows introduce their own bias. This is an input-level replay
on one row, not a production-window selection.

The [fixed-coarse high-rate plan](source_nqc_high_rate_plan.json) and
[result](source_nqc_high_rate.json) keep the source's requested output and
coarse evolution step at `0.125 M`, overriding only the private snapshot's
high-rate resampling factor to 4, 8, 16, and 32. The factor-8 control
reproduces the unmodified captured input hash and source fit. The high-rate
steps are `0.03125`, `0.015625`, `0.0078125`, and `0.00390625 M`. Consecutive
active source coefficient distances are `8.390`, `18.929`, and `25.054` (L2),
so finer high-rate sampling alone does not converge the source cubic fit on
this row. Factors 16 and 32 have the same detected peak and match time, and
their nonphase input values at matching differ by no more than `0.30 ppm`;
their amplitude second derivatives differ by `2.69%`. A counterfactual
degree-four polynomial derivative fit within `±2 M` reduces that coefficient
distance to `0.078`. Replaying all captures at the factor-32 match time gives
successive polynomial-fit distances `0.076`, `0.121`, and `0.078`, versus
`4.523`, `21.144`, and `25.054` for GSL cubic fits. The polynomial result
identifies derivative sensitivity; it has neither an independent physical
accuracy bound nor domain and whole-waveform validation, and cannot be
promoted as the NQC prescription.

The [additional-spin high-rate plan](source_nqc_high_rate_domain_plan.json)
and [result](source_nqc_high_rate_domain.json) repeat factors 8, 16, and 32
with the same fixed `0.125 M` coarse output on the nonspinning and mild
mixed-spin source rows. Factor 8 reproduces the previously captured source
coefficients on each row, and every compiled-source fit replays against pinned
GSL to better than `2.4e-6` maximum absolute coefficient error. At each row's
factor-32 match time, consecutive degree-four `±2 M` polynomial coefficient
distances decrease from `0.023` to `0.011` (nonspinning) and from `0.026` to
`0.012` (mild mixed spin). The corresponding natural-cubic distances increase
from `0.040` to `2.225` and from `0.202` to `4.298`. This sampled-domain
result supports a source derivative-instability hypothesis but is not an
accuracy proof for a replacement estimator or a whole-waveform validation.

The [current-native dense plan](source_placement_nqc_dense_plan.json) and
[result](source_placement_nqc_dense.json) replay the frozen source-placement
strong-spin waveform and NQC amplitude array, then independently integrate
the same provisional vector field with DOP853 from a state about `4 M` before
matching. Five-point derivatives at `0.025 M` are stable against DOP853
tolerance and stencil changes: the largest nonphase second-derivative spreads
are `1.3e-11` and `2.3e-13`. Against that reference, the largest nonphase
second-derivative errors are `1.83e-8` for the native GSL cubic, `8.27e-8`
for a degree-four polynomial over `±2 M`, and `4.20e-10` for a degree-six
polynomial over `±0.5 M`. This demonstrates smooth local native inputs on one
row; it does not establish an accurate source derivative or validate the
provisional physics. The project Python 3.13 environment matches the frozen
LAL extension and GSL binary hashes; the system Python 3.12 environment has
a different wheel build. This diagnostic uses the pinned GSL library and
does not refresh the source-wheel comparison. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_source_placement_nqc_dense
--json-out /tmp/source-placement-nqc-dense.json`.

The [source shared-grid plan](source_nqc_shared_grid_plan.json) and
[result](source_nqc_shared_grid.json) replay the final two pinned-source
high-rate captures exactly (factors 16 and 32) in the pinned Python 3.13
environment. At the 512 shared timestamps within `±2 M` of matching, the
maximum amplitude difference is `2.97e-9`. Their common-match natural-cubic
NQC coefficients differ by `25.05` (L2). Fitting the factor-32 samples at
every other timestamp, with the same time span as factor 16, changes its own
coefficients by `42.44` (L2); dropping only the extra endpoint samples changes
them by `0.0` at stored precision. Factor 16 versus decimated factor 32 still
differs by `18.16` (L2). Thus both a small change in shared inputs and the
cubic's grid-density response contribute to the observed nonconvergence;
these counterfactual fits are not waveform validations or replacement rules.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_source_nqc_shared_grid --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/source-nqc-shared-grid.json`.

The [source dense-field plan](source_nqc_dense_field_plan.json) and nine
results compile the pinned regular-v4 generator's explicit vector field and
pre-NQC `(2,2)`
mode. A private source capture adds the missing orbital phase and angular
momentum arrays, while the original five-column input arrays and active NQC
coefficients replay the frozen source runs. For each row, the explicit mode
agrees with all 1,024 nearby captured amplitude samples within `1.0e-10`,
and centered DOP853 runs from the source's interpolated matching state
measure the continuous-field derivatives. The source cubic minus dense-field
amplitude curvature, divided by dense-field curvature, varies as follows:

| Row | Factor 8 (source default) | Factor 16 | Factor 32 |
| --- | ---: | ---: | ---: |
| Strong aligned spin | [-0.326%](source_nqc_dense_field_strong_aligned_spin_f8.json) | [-2.175%](source_nqc_dense_field_strong_aligned_spin_f16.json) | [-4.811%](source_nqc_dense_field_strong_aligned_spin.json) |
| Nonspinning | [-0.0011%](source_nqc_dense_field_nonspinning_f8.json) | [+0.190%](source_nqc_dense_field_nonspinning_f16.json) | [+3.233%](source_nqc_dense_field_nonspinning.json) |
| Mild mixed spin | [+0.071%](source_nqc_dense_field_mild_mixed_spin_f8.json) | [+0.248%](source_nqc_dense_field_mild_mixed_spin_f16.json) | [-4.922%](source_nqc_dense_field_mild_mixed_spin.json) |

The factor-32 gaps are far above the tested DOP853 tolerance and stencil
spreads. The source orbit's cubic first derivatives differ from its compiled regular vector
field by up to `4.1e-7`; the generator's regular integration path already
resamples adaptive states with GSL cubic before the NQC cubic fit. This is a
diagnostic of the existing numerical path, not an accuracy proof for new NQC
coefficients or a waveform qualification. Reproduce each row with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_source_nqc_dense_field --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --row-name ROW --resample-factor FACTOR --json-out
/tmp/ROW-source-nqc-dense.json`, where `ROW` is one of the three names above
and `FACTOR` is `8`, `16`, or `32`.

The [coefficient-field plan](source_nqc_coefficient_field_plan.json) and
[result](source_nqc_coefficient_field.json) propagate each captured cubic
derivative and each regular-field DOP853 derivative through the pinned v4 NQC
linear systems. The replay of all nine source fits has maximum absolute
coefficient error `2.5e-4`; the amplitude-system condition number reaches
`5.6e4` on the strong-spin row. At factor 32, the largest relative change in
an individual coefficient under this derivative substitution is `2.03%`
(strong aligned), `3.62%` (nonspinning), and `3.45%` (mild mixed). From factors
16 to 32 the strong-spin source coefficients move by `25.05` in L2, versus
`0.0082` for the dense-field counterfactual on the **same matching time**.
The other rows move their matching times by `0.0098 M` and `0.0045 M`, so
their factor-to-factor changes do not isolate differentiation. These fitted
counterfactuals are not source-generator outputs or waveform qualifications.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_source_nqc_coefficient_field --json-out
/tmp/source-nqc-coefficient-field.json`.

The [factor-impact plan](source_nqc_factor_impact_plan.json) and
[result](source_nqc_factor_impact.json) recapture the pinned source's
pre-NQC high-rate arrays at factors 8 and 32, checking their exact array
hashes and original coefficients. Applying the source-cubic and dense-field
counterfactual NQC coefficients to **the same** arrays gives the following
maximum relative complex `(2,2)` correction difference within the `2 M`
*before* matching:

| Row | Factor 8 | Factor 32 |
| --- | ---: | ---: |
| Strong aligned spin | 0.419% | 0.689% |
| Nonspinning | 0.103% | 0.252% |
| Mild mixed spin | 0.0158% | 0.558% |

The factor ratio is the ratio of corrected modes on identical pre-NQC
samples, so it isolates the NQC coefficient choice. The larger differences
after matching in the full captured high-rate arrays are not public-waveform
errors: the source subsequently replaces the mode with attached ringdown.
At the source's default factor 8, a separate comparison applies the pinned
wheel's **public** NQC coefficients to those same source states. The maximum
relative complex factor difference in the pre-match `2 M` window is:

| Row | Source cubic versus wheel | Dense-field counterfactual versus wheel |
| --- | ---: | ---: |
| Strong aligned spin | 0.931% | 0.512% |
| Nonspinning | 0.0999% | 0.00326% |
| Mild mixed spin | 0.0973% | 0.0816% |

The wheel uses its default factor-8 algorithm; this comparison does not
identify the wheel's internal numerical choices or prove that the dense-field
counterfactual is physically more accurate. It shows that a source-side
derivative change explains part, but not all, of the source/wheel NQC gap.
This is a controlled source-mode counterfactual, not a validation of the
dense-field coefficients or the full native waveform. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_source_nqc_factor_impact --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/source-nqc-factor-impact.json`.

The [whole-mode override plan](source_nqc_whole_mode_override_plan.json),
[development-only adapter](../../reference/full_v4_source_nqc_override_adapter.c),
and [result](source_nqc_whole_mode_override.json) run the **complete** pinned
source generator with only its factor-8 `(2,2)` NQC coefficients overridden
by the regular-field counterfactual. The unmodified source and wheel modes
first reproduce their frozen byte hashes. Peak-fitted whole-mode complex
relative L2 against the pinned wheel changes as follows:

| Row | Source fit | Dense-field NQC override |
| --- | ---: | ---: |
| Nonspinning | `6.48e-4` | `2.97e-5` |
| Mild mixed spin | `6.54e-4` | `5.33e-4` |
| Strong aligned spin | `5.80e-3` | `3.19e-3` |

The source and wheel trajectories, pre-NQC modes, and ringdown machinery
remain fixed within each row. The override reduces the pre-attachment
source/wheel gap and establishes that NQC differentiation is a substantial
part of the complete-mode difference. It does not identify a unique or
physically validated derivative rule; post-peak residuals increase slightly
and the strong-spin whole-mode error remains `3.19e-3`. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_source_nqc_whole_mode_override --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/source-nqc-whole-mode-override.json`.

The [wheel-coefficient override plan](source_nqc_wheel_override_plan.json)
and [result](source_nqc_wheel_override.json) apply a second control to the
same complete pinned source generator: inject only the five active `(2,2)`
NQC coefficients reported by the pinned public wheel at the same `0.125 M`
requested output step. Both baseline mode byte hashes and the previous
whole-mode comparison replay exactly. Peak-fitted source/wheel complex L2
falls from `6.48e-4` to `1.13e-5` on the nonspinning row, from `6.54e-4`
to `2.46e-5` on the mild mixed row, and from `5.80e-3` to `8.58e-6` on
the strong aligned row. The resulting nonzero residual bounds the combined
effect of source/wheel dynamics, raw modes, attachment, and remaining numerical
differences on these rows. The larger baseline difference is overwhelmingly
caused by the NQC coefficients, but this experiment does not identify why
the two builds fit different coefficients. The wheel's reported coefficients
are black-box reference outputs and cannot be used as native release data.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_source_nqc_wheel_override --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/source-nqc-wheel-override.json`.

The [wheel-state fit plan](source_nqc_wheel_state_fit_plan.json) and
[result](source_nqc_wheel_state_fit.json) compile the pinned source field and
raw-mode functions, then apply the source NQC equations to the wheel's
hash-replayed public high-rate states for strong aligned spin. A separately
compiled pinned generator supplies its own public high-rate states as a
reconstruction control. Its radius and radial momentum match the captured
private source NQC inputs exactly. An adapter using algebraically equivalent
mass and spin formulas reconstructs raw amplitude and orbital frequency to
`3.53e-10` relative and `1.85e-11` absolute, yet moves the sensitive fit by
`0.215` L2. A second adapter reproduces the generator's exact parameter
construction and per-state calls. It matches the captured source orbital
frequency exactly and raw amplitude to `3.33e-16` relative. Its pinned C NQC
fit is within `2e-6` per coefficient of the captured source coefficients;
the Python/GSL fit has `2.5e-5` L2 distance.
The source and wheel high-rate arrays both start at `786.875 M`, while the
wheel has four more samples at the end. Near the source match their public
state differences are at most `(3.65e-6, 8.89e-6, 3.91e-7, 1.04e-6)` in
`(r, phi, pr*, pphi)`.

The source peak-search rule exactly reproduces its captured `149.70311737 M` peak;
its initial spline derivatives are both negative, so its bisection interval
does not bracket the actual cubic-spline maximum at `149.68100027 M`. This
is a source algorithm and grid effect, not a validated peak prescription.
Applying that same rule to source-field frequencies on the wheel states also
returns `149.70311737 M`, so this rule does not account for the observed
source/wheel coefficient difference on this row. The exact-source-call fit
on the wheel states reduces the L2 distance from the wheel's five reported
coefficients from `8.14` to `1.524`, without closing it; its pinned C and
Python/GSL fits agree within `2.48e-5` per coefficient. The earlier
algebraically equivalent adapter yielded `3.66` L2 distance, and its frozen
nine-point match-time shift scan over `±0.0625 M` reached only `3.09` at
`+0.015625 M`. Wheel trajectory differences and parameter construction
explain parts of the coefficient gap. The remaining private wheel fit inputs
or implementation are unresolved. The scan is a diagnostic sensitivity
check, never a fitted model choice. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_source_nqc_wheel_state_fit --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/source-nqc-wheel-state-fit.json`.

The [three-row exact-input plan](source_nqc_exact_rows_plan.json) and
[result](source_nqc_exact_rows.json) extend that source construction and
the pinned C NQC fit to the frozen nonspinning, mild mixed-spin, and strong
aligned-spin rows. Each compiled-source reconstruction reproduces its
captured five coefficients to at most `1.9e-6` L2. On the wheel's
hash-frozen public high-rate states, the corresponding fits remain `0.0687`,
`0.220`, and `1.524` L2 from the wheel-reported coefficients. The original
source-versus-wheel coefficient gaps are `0.220`, `0.113`, and `8.139` L2,
respectively. Public-state substitution helps the nonspinning and strong
aligned rows but does not close any row; on mild mixed spin it increases the
coefficient gap. These fits retain the pinned source equations and target
values. The wheel's private raw-mode and fit arrays are still needed to
attribute the residual precisely. Reproduce with `PYTHONPATH=src:.
/Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_source_nqc_exact_rows --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/source-nqc-exact-rows.json`.

The [public-mode boundary plan](public_nqc_mode_boundary_plan.json) and
[result](public_nqc_mode_boundary.json) use both public v4 modes API paths at
`0.015625 M`: the coefficient-only call and the call returning modes plus low-
and high-rate dynamics. The high-rate dynamics and returned 22 mode can be
mapped to the same grid using the mode epoch and integer output cut. The mode
amplitude peaks `0.00517 M` before the inferred NQC match on the aligned row
and `0.00729 M` before it on the antialigned row, each within one output
sample. The pinned source's ringdown routine overwrites the waveform starting
at its amplitude-maximum index. Consequently the returned mode straddles the
ringdown boundary at the NQC match: its amplitude value nearly equals the
target, but its second-derivative residual is `2.91e-5` and `-1.46e-4`, and
the frequency-slope residual is `-0.00618` and `-0.00479`. These are
properties of the **post-attachment public output**, not evidence that the
pre-attachment NQC fit missed its targets. The public API does not expose the
unmodified mode samples on both sides of matching, so exact upstream NQC
input comparison requires a separately instrumented pinned source build.
Reproduce in the pinned development environment with `PYTHONPATH=src:.
python -m dev.native_rom.diagnose_public_nqc_mode_boundary --json-out
/tmp/public-nqc-mode-boundary.json`.

The [frozen pre-NQC dynamics plan](public_nqc_dynamics_plan.json) and
[result](public_nqc_dynamics.json) replay both native waveform hashes, then
recover native radius, tortoise radial momentum, and orbital frequency from
the sampled NQC basis. At fixed times relative to the inferred orbital peaks,
the antialigned radius, radial momentum, and frequency differ by `−0.0548%`,
`+0.0893%`, and `+0.0581%` at `−12 M`; all three differences remain nonzero
under a one-sample shift of the inferred public peak. This comparison alone
does not establish different phase-space trajectories. At six fixed orbital
frequencies from `M Omega=0.07` to `0.16`, the same rows agree in radius to
within `3.4 ppm` and radial momentum to within `47 ppm`. Their peak-relative
time differences stay near `−0.011 M` (aligned) and `−0.0195 M`
(antialigned), varying across the frequency range by only `0.00113 M` and
`0.00037 M`, respectively. The public peak inferred from the output epoch has
one-sample (`0.015625 M`) uncertainty, so its absolute offset is not itself a
measured evolution error. The fixed-frequency result points to alignment or
timing convention as the dominant source of the fixed-time differences; the
small residuals and their physical cause remain unresolved.

The same plan scans native NQC match-time shifts from `−0.03` to `+0.03 M`
while holding the source's full-precision targets fixed. On the antialigned
row, a `−0.015 M` shift reduces the five-coefficient distance to the public
fit from `0.66555` to `0.03438`. On the aligned row, that shift increases the
distance from `5.62377` to `8.51492`; the smallest distance in the scan is
still `2.59534` at `+0.025 M`. These are post hoc sensitivities of a
counterfactual matching system, not an approved timing correction or compiled
LAL input-parity result. The public pre-NQC mode samples remain unavailable
through the public API. Run
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_public_nqc_dynamics
--json-out /tmp/public-nqc-dynamics.json` in the pinned development environment.

The [pre-NQC mode plan](public_pre_nqc_mode_plan.json) and
[result](public_pre_nqc_mode.json) divide the pinned public `(2,2)` inspiral
mode by the NQC factor formed from its returned coefficients and high-rate
dynamics. All selected fixed-frequency samples are at least `1 M` before both
the inferred NQC match and the public mode peak, so ringdown replacement does
not enter. With the nominal epoch-derived mode-to-dynamics grid mapping, the
antialigned raw amplitude differs from native by at most `3.1 ppm`; the aligned
raw amplitude differs by `38–87 ppm`. The reconstructed raw amplitude slopes
and waveform frequencies are also recorded, with native and public phase-sign
conventions reconciled. These nominal-grid residuals require an alignment
control before they can be assigned to the mode formula.

The [public grid-alignment plan](public_mode_grid_alignment_plan.json) and
[result](public_mode_grid_alignment.json) hold the same frozen native mode
values fixed, repeat the public reconstruction at output steps `0.015625`,
`0.0078125`, and `0.00390625 M`, and bracket the epoch-derived mapping by one
mode sample in each direction. On the aligned row, the `−1`, `0`, and `+1`
sample mappings on the respective grids each reduce the maximum raw-amplitude
residual to below `1 ppm`; the nominal coarse-grid `38–87 ppm` residual is
therefore not evidence of a physical mode-amplitude error. The antialigned
nominal mapping remains within `3.1 ppm` on all grids. Even with those
amplitude-consistent mappings, raw amplitude-slope and waveform-frequency
residuals reach roughly `85` and `92 ppm` on the aligned row, and `60` and
`75 ppm` on the antialigned row. The one-sample choices are a sensitivity
bracket, not fitted corrections. This public API does not expose unmodified
pre-NQC samples on both sides of the matching time, so these pre-attachment
checks do not establish equality of the centered NQC matching systems or
release qualification. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_public_pre_nqc_mode --json-out
/tmp/public-pre-nqc-mode.json` and `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_public_mode_grid_alignment --json-out
/tmp/public-mode-grid-alignment.json`.

The [exact public mode mapping plan](public_mode_exact_mapping_plan.json) and
[result](public_mode_exact_mapping.json) resolve that bracket for the frozen
`0.015625 M` outputs without selecting a shift from native residuals. In the
pinned non-PA v4 source, the uncut mode length is the low-dynamics length plus
the high-dynamics length and a QNM-dependent ringdown allocation; the returned
mode loses its first `kMin` samples. The public
`SimIMREOBGenerateQNMFreqV2` call and returned lengths give `kMin=26236` on
the aligned row and `12918` on the antialigned row. The exact high-rate mode
starts at returned indices `24117` and `2166`, respectively: the aligned
epoch-derived index was one sample late, while the antialigned index was
correct. The source-backed index changes the aligned maximum raw-amplitude
residual over the five frozen `M Omega=0.07–0.13` comparisons from `87.0` to
`0.923 ppm`, and its raw gravitational-wave-frequency residual from `671.6`
to `92.0 ppm`. The antialigned maxima remain `3.03` and `74.2 ppm`. These
figures still divide an attached public mode by an inferred NQC factor on a
separately evolved public trajectory; they are not identical-state mode
parity, a centered NQC-input comparison, or release qualification. Reproduce
with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_public_mode_exact_mapping --source-repo
/path/to/pinned/lalsuite-git --json-out /tmp/public-mode-exact-mapping.json`.

**Output-step correction.** The frozen native whole-waveform and direct NQC
coefficient checks request `0.125 M` output; the public run internally returns
high dynamics at `0.015625 M`. The preceding mode and dynamics diagnostics
instead request `0.015625 M` output (and finer steps for the mode bracket).
Equal high-rate spacing does not make these the same public evolution. The
[matched-output plan](matched_output_reference_plan.json) and
[result](matched_output_reference.json) call the pinned public binary at
`0.125 M`, exactly reproduce all five frozen public coefficients on both rows,
and compare high dynamics with the same native frozen values at six fixed
orbital frequencies. Radius differs by at most `4.0 ppm` and tortoise radial
momentum by at most `49.6 ppm`. The change from `0.015625 M` to `0.125 M`
public output changes these phase-space values by at most `2.6` and `6.5 ppm`
at those frequencies; this supports the earlier small phase-space result,
but does not validate the earlier fixed-time epoch inference or mode alignment
as matching-input comparisons for the `0.125 M` waveform. An exploratory
near-peak follow-up at three additional frequencies per row, selected after
inspecting the public peak range, finds the two public high-rate trajectories
within `2.1 ppm` in radius, `5.8 ppm` in radial momentum, and `0.6 ppm` in
angular momentum at equal orbital frequency. This does not measure their
pre-NQC mode or the differentiated matching matrices.

The same pinned calls show a substantially larger **coefficient** output-step
effect. Relative to the frozen `0.125 M` public coefficients, at least one of
the `0.015625`, `0.0078125`, and `0.00390625 M` calls differs by up to
`1.52%` across the aligned row's five coefficients and `2.98%` across the
antialigned row's five. The original native-versus-public gaps at `0.125 M`
reach `0.34%` and `1.55%`, respectively. The NQC coefficients do not converge
monotonically over these output steps. This is a measured sensitivity, not an
explanation of which integration, interpolation, peak, or matching operation
causes it. No fine-output public mode result should be used as if it supplied
the centered inputs of the frozen `0.125 M` coefficient fit. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_matched_output_reference
--json-out /tmp/matched-output-reference.json`.

The [native output-grid plan](native_nqc_output_grid_plan.json) and
[result](native_nqc_output_grid.json) replay the frozen `0.125 M` native
waveform and public coefficients exactly, then request `0.0625`, `0.03125`,
and `0.015625 M` output on both strong-spin rows. Every native run uses the
same `0.015625 M` internal high-rate step, fixed coarse initial step, and
fixed restart time; the native peak, match time, and all five NQC coefficients
are bitwise identical across the four requested output steps. The pinned
public coefficients vary by at most `0.27%` on the aligned row and `1.69%`
on the antialigned row over those four steps. At the finest of those steps,
the maximum native/public coefficient gaps are `0.22%` and `3.00%`;
output-step sensitivity does not remove the discrepancy. The native control
does not reproduce the public low-rate evolution rule: pinned source
`LALSimIMRSpinAlignedEOB.c` at revision `53acc0c2cd7ea2d3c83fc71a7bd88035c2861d6c`
(SHA-256 `936e95ee47422a3986f84f71b81c6917d6d164b73fd6421ee60ecfbe502595dd`)
uses requested `deltaT` for the non-postadiabatic coarse ODE step and chooses
the high-rate restart from that coarse trajectory. This supplies a plausible
path for the measured public sensitivity, not a causal isolation of the NQC
fit or a historical same-input parity result. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_native_nqc_output_grid
--json-out /tmp/native-nqc-output-grid.json` in the pinned development
environment.

Reproduce the new waveform checks from the repository root with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_mode_32_whole_waveform
--json-out /tmp/mode32-waveform.json` and
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_mode_32_antialigned
--json-out /tmp/mode32-antialigned.json`; the combined source-placement check
is `PYTHONPATH=src:. python -m dev.native_rom.diagnose_source_flux_whole_waveform
--json-out /tmp/source-flux-waveform.json` in the pinned development environment.
The additional antialigned and public-torque sampling checks use the same
module form with `diagnose_source_flux_antialigned` and
`diagnose_public_torque_sampling`, respectively.

# Public SEOBNRv4 output-grid control

The [predeclared plan](reference_ultrafine_plan.json) and
[pinned result](reference_ultrafine.json) continue the aligned-spin public
binary self-comparison from [the prior campaign](../2026-09-23/README.md).
The same `lal=7.7.1`, `lalsimulation=6.2.1` binary was called at output steps
`0.03125`, `0.0078125`, and `0.001953125 M`. The second run reproduced every
comparison field exactly. These are public-reference diagnostics, not native
waveform validations.

| Public grid refinement | Raw shared-epoch complex L2 | Independent peak-anchor complex L2 |
| --- | ---: | ---: |
| `0.03125 → 0.0078125 M` | 0.00091904 | 0.00021263 |
| `0.0078125 → 0.001953125 M` | 0.00042932 | 0.00048157 |

Neither anchor convention decreases monotonically over the full set of
fourfold refinements, including the two coarser comparisons in the prior
campaign. The public implementation's
[source](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_aligned_e_o_b_8c_source.html)
passes the requested `deltaT` into the low-rate integration/interpolation,
uses it to select the step-back index, and derives the high-rate step from it.
Thus changing `deltaT` changes more than output interpolation. This source
path provides a plausible mechanism for the observed self-differences, but
the diagnostic has not apportioned them among integration, restart, NQC, and
ringdown operations. A single public output step cannot yet be treated as a
numerically converged reference for a tight native parity gate.

Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_reference_ultrafine --json-out
/tmp/seobnrv4-reference-ultrafine.json`.

# Historical (3,3) flux placement sensitivity

The [source-formula plan](mode_33_flux_placement_plan.json) and
[same-state result](mode_33_flux_placement.json) isolate a difference between
the native Pan-style Taylor `f33` spin series and the historical non-HM v4
implementation. The latter places the `v^4` through `v^7` Kerr-spin
coefficients inside `rho33`, then cubes it. The local LAL source revision,
file hash, and [published source](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_e_o_b_factorized_waveform_8c_source.html)
are pinned in the plan. This is a source-formula distinction, not a direct
compiled mode-flux comparison.

At four frozen public states (`M Omega=0.032–0.05`), replacing only this
mode's spin residual changes the native torque discrepancy from `13–49 ppm`
to within about `1.5 ppm`. The [extended plan](mode_33_late_flux_plan.json)
and [two-grid result](mode_33_late_flux.json) sample the same public dynamics
at `M Omega=0.06–0.15`. Through `0.12`, the historical choice changes an
approximately `80–812 ppm` discrepancy to at most `13.3 ppm` on both grids.
At `0.15`, about `242–246 ppm` remains. Public torque here is inferred from
a five-point derivative of sampled `p_phi`; these numbers do not establish
historical flux parity or explain the late residual. The same-state native
orbital-frequency derivative differs from the sampled public value by about
`75–76 ppm` at `0.15`, versus at most about `6 ppm` through `0.12`; the native
radial-momentum derivative differs by about `1732 ppm` there. Those late
vector-field gaps need their own Hamiltonian and sampling analysis before
the residual can be assigned to a flux mode.

The first [same-state result](mode_33_flux_placement.json) froze the baseline
source before the diagnostic option was added. Replaying its script with the
current source reproduces every numerical point exactly; the later-flux and
whole-waveform reports record hashes of the implemented choice.

Torch and JAX now expose the historical `(3,3)` placement as an explicit
development choice, with the previous choice retained as the default. The
[frozen whole-waveform plan](mode_33_whole_waveform_plan.json) and
[result](mode_33_whole_waveform.json) use the same 30+20 solar-mass aligned-spin
row, pinned binary, fixed numerical settings, and exact baseline replay.
The sampled-peak complex relative L2 improves from `0.0133609` to
`0.0107870`. With independent quadratic sub-sample peak fits it instead
changes from `0.00807196` to `0.00827564`: the late-inspiral and post-peak
windows improve, but the longer inspiral window worsens. The reversal shows
that this full-waveform comparison is sensitive to peak anchoring; the
candidate is not promoted to the default or to a release model.

Reproduce the latter diagnostics with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_mode_33_late_flux --json-out
/tmp/seobnrv4-mode33-late-flux.json` and `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_mode_33_whole_waveform --plan
dev/native_rom/evidence/2026-09-24/mode_33_whole_waveform_plan.json
--json-out /tmp/seobnrv4-mode33-whole-waveform.json`. The evidence records
the source and public binary identities. Both commands are diagnostics only.

# Historical Hamiltonian logarithmic derivative

The pinned [historical LAL Hamiltonian source](https://git.ligo.org/lscsoft/lalsuite/-/raw/5154f04e470d1cc48d6f61193af728b5692474ab/lalsimulation/lib/LALSimIMRSpinEOBHamiltonian.c)
omits the derivative of the logarithm in its internal `deltaU_u` expression.
The native Hamiltonian differentiates the potential mathematically, including
`k5l u^4`; its potential value is unchanged. The frozen
[plan](hamiltonian_log_derivative_plan.json) pins the source revision and hash,
and the [same-state result](hamiltonian_log_derivative.json) checks a
development-only historical variant against sampled public dynamics at two
output grids. At `M Omega≈0.15`, the orbital-frequency derivative discrepancy
falls from about `75–76 ppm` to `0.19–0.37 ppm`, and the radial-momentum
derivative discrepancy from about `1732–1734 ppm` to `193–203 ppm`.
The [four-grid public sampling control](hamiltonian_public_grid.json) shows
nonmonotonic sampled derivative residuals at the finest grid, so these are
diagnostic sensitivities, not direct compiled-Hamiltonian parity.

The [compiled-component capability plan](hamiltonian_component_capability_plan.json)
and [result](hamiltonian_component_capability.json) pin four public states,
the historical source, and the wheel before checking callable entry points.
The wheel exports a post-adiabatic Hamiltonian wrapper, but this public-API
check finds no exported SEOBNRv4 coefficient initializer for that wrapper.
It does not call the wrapper. The source-backed compiled adapter below
resolves this particular capability gap for the tested states. Reproduce the
public-API audit with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_hamiltonian_component_capability --source-root
<pinned-lalsuite-source-root> --json-out /tmp/hamiltonian-capability.json`,
using the annotated tag object, peeled commit, and file hashes in the plan.

The [compiled same-state plan](hamiltonian_same_state_plan.json),
[result](hamiltonian_same_state.json), and
[provenance and limits](HAMILTONIAN_SAME_STATE.md) compare eight identical
states. An adapter compiles the exact pinned source v4 coefficient initializer
and Hamiltonian, then passes those coefficients to the wheel's exported
Hamiltonian wrapper. Source and wheel values of `H_real/nu` agree bitwise at
all eight states. The paper-derived native value differs by at most
`1.02e-8` relative in four moderate-spin early states and `7.18e-6` in four
strong aligned-spin states. The [two-factor plan](hamiltonian_source_conventions_plan.json)
and [result](hamiltonian_source_conventions.json) then compare native variants
with the pinned header's shorter calibration decimals and the historical
logarithmic-derivative convention, individually and together. The maximum
absolute native-minus-wheel gap changes from `2.904e-5` (published decimals
and mathematical derivative), to `2.918e-5` (source decimals only), to
`1.426e-7` (historical derivative only), to `8.9e-16` (both). This attributes
the measured eight-state gap to those two conventions. It does not establish
the wheel's private generator initialization, dynamics convergence, or
whole-waveform parity. Reproduce the two-factor result with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_hamiltonian_source_conventions
--source-root /path/to/pinned/lalsuite-git --json-out
/tmp/hamiltonian-source-conventions.json`.

The [source-calibration waveform plan](source_calibration_waveform_plan.json)
and [result](source_calibration_waveform.json) replay the preceding
source-backed flux waveform bitwise, then change only the Hamiltonian's
calibration decimals to those in the pinned v4 header. The historical
logarithmic derivative, 35-mode flux choices, native early start, and `2 M`
NQC phase window stay fixed. All three ODE levels complete and the refined
complex relative L2 is `2.96e-7`, below the frozen `1e-6` budget. Against
the unchanged public binary, the peak-fitted full-waveform complex L2 falls
from `0.00299749` to `0.00271301`; the inspiral part falls from `0.00333081`
to `0.00301216`, while the post-peak value remains about `0.0008125`. The
source decimals therefore explain part, not all, of the aligned inspiral gap.
They are confined to a development-only diagnostic class, not promoted as
independent production calibration. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_source_calibration_waveform --source-root
/path/to/pinned/lalsuite-git --json-out
/tmp/source-calibration-waveform.json`.

The [same-public-state early-field plan](source_calibration_early_field_plan.json)
and [result](source_calibration_early_field.json) reuse eight frozen public
states on two requested output grids. With the source decimals, the maximum
native-versus-sampled-public radial-momentum derivative gap over four early
frequencies falls from `4.70e-4` to `7.47e-5` relative on the `0.0625 M`
grid, but the maximum torque gap changes from `6.42e-7` to `6.55e-7`.
The torque oracle is a finite difference of public angular momentum, not a
direct compiled-flux call; its output-grid sensitivity limits attribution.
Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_source_calibration_early_field --source-root
/path/to/pinned/lalsuite-git --json-out
/tmp/source-calibration-early-field.json`.

The [frozen initial-state plan](source_calibration_initial_plan.json) and
[result](source_calibration_initial.json) compare the published and pinned-source
calibration choices with the same source-backed 35-mode flux at two earlier
public starting states. The rounded choice reduces the circular-radius and
angular-momentum gaps to at most `6.36e-9` and `2.96e-9`, respectively, but the
inward tortoise-momentum gap remains `2.09e-7` and `2.41e-7`. Replacing the
analytic radial response by an idealized `0.001` ordinary-momentum finite-step
response, as motivated by the pinned initial-data source, explains less than
1% of that gap. This is not a direct replication of the compiled finite
derivative or flux; the remaining initial-data prescription is unresolved.
The public starting frequency here is inferred from sampled output and is
slightly grid-dependent. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_source_calibration_initial --source-root
/path/to/pinned/lalsuite-git --json-out /tmp/source-calibration-initial.json`.

The [frozen two-row domain plan](source_calibration_domain_plan.json) and
[result](source_calibration_domain.json) extend the rounded calibration to the
mass-ratio-four nonspinning and strong antialigned-spin boundary rows, holding
the source-backed flux, `1 M` NQC window, and each row's restart fixed. Both
pass all three-tolerance numerical gates: refined complex relative L2 is
`1.42e-8` and `5.84e-7`. Their independently peak-fitted public waveform
differences, however, rise from `0.000410287` to `0.000411544` and from
`0.000212727` to `0.000212906`. The aligned-spin improvement therefore does
not generalize across these samples. These source decimals remain a diagnostic
for historical convention attribution, not an independently sourced release
calibration. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_source_calibration_domain --source-root
/path/to/pinned/lalsuite-git --json-out /tmp/source-calibration-domain.json`.

The [combined late-field result](combined_late_field.json) applies both the
historical Hamiltonian convention and `(3,3)` flux placement at the same public
states. At `M Omega≈0.15`, the orbital-frequency residual is below `0.4 ppm`,
while radial-momentum and torque residuals remain roughly `25–35 ppm` and
`22–23 ppm`, respectively, against finite-difference sampled dynamics.

The [frozen full-waveform plan](hamiltonian_whole_waveform_plan.json) and
[result](hamiltonian_whole_waveform.json) replay the earlier `(3,3)`-only
candidate exactly, then change only the Hamiltonian convention on a 30+20
solar-mass, aligned-spin row. The sampled-peak complex relative L2 improves
from `0.01078696` to `0.01006589`; with independently fitted quadratic peaks,
it improves from `0.00827564` to `0.00760014`. The fitted inspiral difference
falls from `0.00955401` to `0.00877661`, but a substantial phase residual
remains. Neither this comparison nor same-state derivatives qualify the
physical generator or the ROM. The diagnostic convention is confined to the
development driver; the production Hamiltonian still uses the mathematical
derivative.

The [predeclared three-grid check](hamiltonian_grid_convergence_plan.json) and
[result](hamiltonian_grid_convergence.json) hold the combined candidate and ODE
tolerances fixed at output steps `0.25`, `0.125`, and `0.0625 M`. Nested native
complex differences are `4.31e-9` and `1.26e-8`, both far below the `1e-6`
budget, with unchanged event times. The strict gate nevertheless **fails**
its required decrease in error. The fitted-peak public comparison changes
`0.00758983 → 0.00760014 → 0.00683810`; the public call changes its own
integration behavior with output step, so this sequence is not a converged
physical-reference study. We do not waive the failed numerical gate.

The [frozen native-warm-start plan](combined_warmstart_waveform_plan.json) and
[four-choice result](combined_warmstart_waveform.json) then combine that prior
fully native frequency warm start with the two historical conventions, changing
one choice at a time on the same aligned-spin row and output step. The baseline
waveform exactly replays the earlier warm-start hash. Independently fitted-peak
complex relative L2 is `0.00168589` at baseline, `0.00166839` with `(3,3)`
alone, `0.00131487` with the Hamiltonian convention alone, and `0.00131140`
with both. The combined inspiral, late-inspiral, and post-peak values are
`0.00150094`, `0.00041128`, and `0.00035279`. This identifies a useful
historical source sensitivity after native initial-state preparation, but
does not close the residual, numerical gates, or parameter-domain validation.

The [frozen initial-state ablation](combined_initial_state_ablation_plan.json)
and [result](combined_initial_state_ablation.json) replay the combined native
waveform exactly, then replace only the early warm-start state with the pinned
public state at `M Omega≈0.03162`. The native and public states differ by
`1.57e-8` in radius, `2.43e-7` in tortoise radial momentum, and `1.35e-8` in
angular momentum. Fitted whole-waveform complex relative L2 decreases from
`0.00131140` to `0.00116397`; the inspiral-window value decreases from
`0.00150094` to `0.00133120`, while post-peak changes from `0.00035279` to
`0.00035668`. The intervention diagnoses sensitivity to the initial state;
the public state is never a native runtime input. Most of the tested
whole-waveform residual remains after the substitution.

The [combined-candidate numerical plan](combined_warmstart_convergence_plan.json)
and [result](combined_warmstart_convergence.json) hold the native early-start
frequency, coarse first ODE step, restart, NQC rule, and historical conventions
fixed. The three nested output grids produce identical sampled waveforms and
event times. The refined ODE tolerance pair decreases in error but **fails**
both `1e-6` budgets: complex relative L2 is `3.53e-6` and amplitude relative
L2 is `1.51e-6`. Event times agree. Thus neither the native warm start nor
the two source-motivated conventions closes numerical qualification for this
aligned-spin row. The exact-hash tolerance trace is recorded separately.

The [frozen tolerance trace](combined_tolerance_trace_plan.json) and
[result](combined_tolerance_trace.json) replay the medium/tight pair byte for
byte. Their raw pre-attachment modes differ by only `1.57e-9` complex relative
L2, and the largest orbital-state difference before attachment is
`3.55e-9` in phase. The final `3.53e-6` error is therefore introduced by NQC
calibration and attachment, not by a comparably large raw inspiral difference.
The amplitude NQC coefficients shift by about `(0.00150, -0.01339, 0.01541)`
between the two solves. These observations motivate the separate, predeclared
local second-derivative intervention; they do not certify that rule as
historically correct.

The [predeclared local second-derivative plan](combined_matching_second_plan.json)
and [result](combined_matching_second.json) change only the NQC matching
second-derivative estimate. On the same aligned-spin row, the refined ODE
complex and amplitude differences fall to `1.05e-7` and `2.87e-9`, passing
the `1e-6` numerical budgets with decreasing error and unchanged event times.
The independently fitted public-binary difference moves from `0.00131140`
to `0.00131155`. This establishes one-row numerical stability for this
diagnostic rule, not historical NQC parity, physical accuracy over the
domain, or release qualification.

# Combined-candidate boundary rows

The [frozen three-row plan](combined_domain_numerics_plan.json) and
[result](combined_domain_numerics.json) apply the native warm start, historical
Hamiltonian derivative, historical `(3,3)` placement, and local NQC derivative
choices to a mass-ratio-four nonspinning row and strong aligned/antialigned
spin rows. The mass-ratio-four row passes the refined ODE check (`3.39e-8`
complex relative L2); its peak-fitted public-binary difference is `0.0004103`.
The strong aligned row fails the `1e-6` ODE complex budget at `2.74e-6`,
despite an amplitude difference of `3.39e-9`; its fitted public difference is
`0.01985`. The fixed `165 M` restart lies *after* the antialigned row's
`163.49 M` match, so that row raises a match-grid error and the frozen
three-row campaign fails.

A [follow-up common-restart plan](combined_domain_numerics_early_restart_plan.json)
and [result](combined_domain_numerics_early_restart.json) repeat those three
rows at `150 M`, chosen after the `165 M` failure so the antialigned row's
`163.49 M` match is inside the fine trajectory. The refined ODE complex
relative L2 values are `1.82e-7` (mass-ratio-four nonspinning), `8.76e-7`
(strong aligned), and `9.82e-7` (strong antialigned); all three pass the
existing `1e-6` budget, amplitude, decreasing-error, and event-time checks.
The independently peak-fitted public differences remain `0.000410`, `0.01985`,
and `0.000204`, respectively. This is a post-hoc restart diagnostic on three
rows, with the antialigned numerical result close to the budget. It neither
establishes a domain-wide numerical bound nor resolves the aligned-spin
physical mismatch. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_combined_domain_numerics --plan
dev/native_rom/evidence/2026-09-24/combined_domain_numerics_early_restart_plan.json
--json-out /tmp/combined-domain-early-restart.json` in the pinned development
environment.

A separately frozen [earlier-restart plan](antialigned_restart_plan.json) and
[result](antialigned_restart.json) move only that row's fixed restart to
`100 M`. Its refined ODE check passes (`7.16e-7` complex, `2.39e-9`
amplitude), with a fitted public difference of `0.0002039`. This is a
diagnostic intervention, not a pass for the original campaign. The
[nested-grid plan](boundary_grid_plan.json) and [result](boundary_grid.json)
then hold the tight ODE tolerance, first step, warm start, and each passing
row's restart fixed over `0.25`, `0.125`, and `0.0625 M` output grids. Both
rows pass all listed grid gates with zero complex and amplitude differences
and unchanged peak, match, and boundary times. These grids sample the same
fine trajectory and establish an identity control; they do not supply an
independent trajectory-error estimate.

The [exact-hash aligned-spin trace](boundary_tolerance_trace.json) shows a
`2.21e-10` raw pre-NQC mode difference between the two tight ODE solves,
versus `2.74e-6` after NQC and attachment. Its
[matching-system plan](combined_strong_spin_nqc_inputs_plan.json) and
[result](combined_strong_spin_nqc_inputs.json) reconstruct both stored phase
systems and close the coefficient-drift attribution to `1.03e-12`. The phase
coefficient changes by `(4.45e-5, -0.002558)`; a matrix change contributes
`(4.41e-5, -0.002530)`, mostly through its second-derivative row, while the
right-hand-side contribution is `(3.69e-7, -2.81e-5)`. This identifies the
numerically sensitive operation. It does not validate an alternative NQC
rule or clear the physical residual.

The [frozen window study](combined_nqc_windows_plan.json) and
[exact-hash result](combined_nqc_windows.json) hold each matching
right-hand side fixed and reconstruct the phase-basis matrix with local
windows of `0.5`, `1`, and `2 M`. The two-solve phase-coefficient drift falls
from `0.002558` to `0.000598` and `0.000182`, respectively. These are
matrix-input sensitivities, not whole-waveform convergence results. Wider
windows may also bias the true local derivative; the `0.5 M` default remains
unchanged.

The [predeclared whole-waveform intervention](strong_spin_nqc_window_plan.json)
and [result](strong_spin_nqc_window.json) replay the `0.5 M` baseline waveform
hash exactly, then change only the diagnostic phase-basis window to `2 M` on
the strong aligned-spin row. The refined ODE complex difference falls from
`2.74e-6` to `2.70e-7`, with `3.39e-9` amplitude difference, decreasing
error, and unchanged event times; all frozen numerical gates pass. The
peak-fitted public-binary complex difference changes only from `0.0198521`
to `0.0198485`. The wider window is opt-in and is not a physical-model
correction.

The [frozen dense-reference plan](strong_spin_nqc_dense_plan.json) and
[result](strong_spin_nqc_dense.json) independently integrate the same
provisional vector field with DOP853 from a native state about `4 M` before
matching. Centered five-point differentiation of the dense trajectory gives
stable phase-basis derivatives: changing DOP853 tolerance alters each matrix
by at most `1.31e-11` at the finest `0.025 M` stencil. At that stencil, the
native `0.5`, `1`, and `2 M` local windows differ from DOP853 by at most
`4.59e-10`, `3.90e-11`, and `1.63e-9`, respectively. Thus the `2 M` window
suppresses ODE-to-ODE waveform drift on this row but increases local derivative
bias; the `1 M` window is closest to the dense estimate here. None is qualified
across the physical domain. This check shares the provisional Hamiltonian and
flux, so it does not validate their physics.

Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_strong_spin_nqc_dense --json-out
/tmp/strong-spin-nqc-dense.json`.

The [predeclared 1 M waveform plan](strong_spin_nqc_one_M_plan.json) and
[result](strong_spin_nqc_one_M.json) then test that middle window on the same
aligned-spin row at the three frozen ODE tolerances. Its refined complex
relative difference is `6.23e-7`, below the existing `1e-6` numerical limit,
with unchanged event times and decreasing error. The peak-fitted difference
from the pinned public waveform remains `0.0198508`, essentially unchanged.
This is one-row numerical evidence only; the `1 M` window remains opt-in and
is not a release choice. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_strong_spin_nqc_one_M --json-out
/tmp/strong-spin-nqc-one-M.json`.

The [frozen two-row 1 M plan](nqc_one_M_domain_plan.json) and
[result](nqc_one_M_domain.json) extend that numerical check to the
mass-ratio-four nonspinning row and strong antialigned-spin row. The latter
retains the separately frozen `100 M` restart; all other numerical and
physical candidate choices match the prior boundary campaign. Refined ODE
complex relative L2 changes from `3.39e-8` to `1.41e-8` for the nonspinning
row and from `7.16e-7` to `5.85e-7` for the antialigned row when the phase
derivative window changes from `0.5` to `1 M`. Both pass the existing `1e-6`
complex and amplitude budgets, decreasing-error requirement, and event-time
gates. Independently peak-fitted differences from the pinned public waveform
change only from `0.000410279` to `0.000410287` and from `0.000203914` to
`0.000203842`, respectively. Across these three sampled rows, the opt-in
`1 M` window has numerical support; this is neither a physical-domain bound
nor release qualification. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_nqc_one_M_domain --json-out
/tmp/nqc-one-M-domain.json` in the pinned development environment.

The [frozen source-flux 1 M domain plan](source_flux_one_M_domain_plan.json)
and [result](source_flux_one_M_domain.json) add the existing source-backed
`(2,1)`, `(3,1)`, and `(3,2)` flux placements to those same two rows while
holding the `1 M` NQC window, three ODE levels, and row-specific restarts
fixed. All three nonspinning waveforms are bitwise unchanged. For the strong
antialigned row, refined complex relative L2 changes only from `5.853e-7`
to `5.849e-7`; the fitted difference from the pinned public waveform rises
from `0.000203842` to `0.000212727`. Both rows pass the frozen `1e-6`
complex and amplitude budgets, decreasing-error requirement, and event-time
gates. These two samples test numerical stability of the combined diagnostic
choices, not physical accuracy or domain-wide release qualification. Reproduce
with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_source_flux_one_M_domain --json-out
/tmp/source-flux-one-M-domain.json` in the pinned development environment.

**Dense signed-spin QNM.** The [predeclared grid](qnm_dense_spin_plan.json)
and [result](qnm_dense_spin.json) compare both independently generated native
frontends with the hash-pinned public `SEOBNRv4` QNM routine at 131 signed
spins from `-0.9` to `0.9996`. Torch and JAX agree to `1.7e-16`, but both
miss the previous `6e-7` historical-binary proximity gate: the worst
frequency difference is `1.7154e-5` at `j=0.999540793009`. Four post-hoc
strict runs of our original Kerr equation solver, including that point,
match the native spline to at most `7.1e-16`. These solver controls verify
native interpolation but are not independent author data. The sparse Berti
and Cook author checks remain the external physical anchors; historical
interpolation and the whole waveform remain unqualified. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_qnm_dense_spin --json-out
/tmp/diffgw-qnm-dense-spin.json`; exit status 1 records the failed wheel
proximity gate in the JSON.

**QNM reachability from binary inputs.** The [post-hoc witness plan](qnm_binary_reachability_plan.json)
and [result](qnm_binary_reachability.json) feed ten binary configurations to
both the published Hofmann `(3,4)` remnant fit and the pinned public
`SEOBNRv4` remnant helper. The final spins agree to `2.3e-16`. The
`(q,chi1,chi2)=(8,0.85,0.85)` row used for NR calibration in
[Bohé et al. (2017), Sec. III](https://arxiv.org/html/1611.03703) gives
`j=0.8941271984` and a native/wheel QNM frequency difference of `1.20e-6`,
already above the previous `6e-7` proximity threshold. In extrapolated
high-mass-ratio rows, `(76,1,-1)` gives `j=0.9995469946` and a frequency
difference of `1.91e-5`; `(80,1,-1)` gives `j=0.9996053273`, beyond the
pinned QNM helper's `0.9996` upper limit. Strict native Kerr equation solves
at the `q=8` calibration and `q=76` witness spins match the native spline to
at most `3.4e-16`. The high-`q` rows are not evidence of NR-calibrated model
accuracy, and this finite witness set is not a domain-wide bound. The failed
historical parity/domain gates remain in the JSON. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_qnm_binary_reachability
--json-out /tmp/diffgw-qnm-binary-reachability.json`; exit status 1 records
the failed gates.

**Historical QNM source interpolation.** The [frozen source-audit plan](qnm_source_interpolation_plan.json)
and [result](qnm_source_interpolation.json) identify the 107 printed spin
knots and `gsl_interp_cspline` rule in pinned LAL source revision
`53acc0c2cd7ea2d3c83fc71a7bd88035c2861d6c`. The source file SHA-256 is
`99430b73bd8f67aa72bfbe49b3a86051908d4bae83324dc5c5d6410b304cde95`.
A development-only SciPy natural cubic replay matches the pinned public QNM
outputs to `1.2e-16` at five witness spins. At `j=0.999540793009`, using
native solved values at those same source knots still yields a `1.751e-5`
frequency interpolation error; the source's printed values contribute only
`-3.55e-7` to the interpolated frequency. The historical knot spacing and
curvature explain most of the positive near-extremal wheel/native gap. The
negative endpoint shows a separate discrepancy: at `j=-0.9996` its printed
frequency differs from the native solved mode by `0.02135`. This audit
establishes historical implementation provenance, not physical accuracy or
waveform qualification. No third-party QNM table is copied into production
code. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_qnm_source_interpolation --source-root
/Users/xangma/repos/lalsuite-git --json-out
/tmp/diffgw-qnm-source-interpolation.json`.

**Counterrotating Cook author check.** The [post-hoc sparse plan](qnm_negative_author_plan.json)
and [result](qnm_negative_author.json) read five rows from Cook's hashed
`KerrQNM_00.h5` dataset `/n00/m-02/{2,-2,0}` using `h5dump`; the archive
SHA-256 is `d5853acabc1a07b66b4fae437c02e1435a9d2b3561bd7e4752b9721c9cd48923`.
Native positive-frequency counterrotating roots agree with author values
within `6.97e-13` at signed spins `-0.99` through `-0.999625`. The pinned
wheel's source table differs from Cook by `0.015019` in frequency at
`j=-0.9995`, inside its public `|j|<=0.9996` domain. Three sparse author
rows also test Torch and JAX. These rows were chosen after inspecting the
negative-spin discrepancy; they do not provide a domain-wide error bound or
validate a waveform. The author dataset remains a development-only
validation source. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_qnm_negative_author --author-archive
/tmp/diffgw-KerrQNM_00.h5 --json-out /tmp/diffgw-qnm-negative-author.json`.

**Pinned-wheel private NQC inputs.** The [frozen capture plan](wheel_nqc_spline_capture_plan.json)
and [three-row result](wheel_nqc_spline_capture.json) use a development-only
macOS interposer to record the ten GSL spline input arrays passed by the
unmodified, hash-pinned wheel. The interposer calls the original GSL function;
the recorded arrays are diagnostic outputs, not native model data. The
source C fit reproduces the wheel's five active coefficients to at most
`1.01e-12` L2. Algebraic recovery from the wheel's Q3, Q4, and P3 spline
inputs shows its private fit radius and radial momentum agree with its public
high-rate dynamics to at most `1.78e-15` and `1.11e-16`, respectively.

On those identical public states, the pinned source's explicit-state calls
differ from the wheel's captured NQC input amplitudes by at most
`1.61e-10`, orbital frequencies by `2.65e-11`, and phases by `1.21e-10`
within `4 M` of matching. Replacing only the captured amplitude with the
source-evaluated amplitude changes the fitted coefficient vector by `0.0603`,
`0.206`, and `1.527` L2 on the nonspinning, mild mixed-spin, and strong
aligned-spin rows. Replacing only frequency changes it by `0.00882`,
`0.0152`, and `0.0312`; replacing only phase changes it by `0.00121`,
`0.00207`, and `0.00207`. Replacing all three yields the earlier source
reconstruction gaps of `0.0687`, `0.220`, and `1.524`. A separate source
generator control confirms the explicit-state adapter exactly reproduces its
own captured amplitude and frequency at all 512 strong-spin samples around
matching. The residual therefore lies between the pinned source evaluation
and wheel binary on the same states; this experiment does not isolate its
compiler or source-code origin. Compiling the explicit-state source probe at
`-O0`, `-O2`, and `-O3` leaves the checked amplitudes unchanged to
`1.11e-16`. The `-O2 -ffast-math` control changes them by up to `1.99e-10`
but raises the amplitude RMS error against the wheel on all three rows.
These controls exclude a simple optimization-level fix; they do not reveal
the wheel's build settings. Natural-cubic differentiation amplifies the
small amplitude differences into much larger coefficient changes. This
attribution does not establish a native NQC rule, physical accuracy, or
release qualification. Reproduce on the pinned macOS environment with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_wheel_nqc_spline_capture
--source-root /Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/diffgw-wheel-nqc-spline-capture.json`.

A further controlled ablation evaluates the identical source mode with the
wheel's captured orbital frequency as the cube-root argument, while retaining
the source state, Hamiltonian, and coefficients. This removes the same-state
mode phase difference to at most `3.55e-15` in the three matching windows.
The amplitude difference falls to at most `8.38e-11` but does not vanish;
the remaining RMS difference is `1.98e-11` to `2.53e-11`. With only this
amplitude substituted into the captured wheel fit, the coefficient distances
are `0.00851`, `0.0618`, and `1.291` (L2) for nonspinning, mild mixed-spin,
and strong aligned-spin rows. The wheel frequency input therefore explains
the phase discrepancy and part of the amplitude discrepancy, but the source
mode still differs in amplitude. A separate same-state call to the wheel's
exported Hamiltonian wrapper, initialized with pinned source coefficients,
agrees with the source mode's Hamiltonian to at most `8.88e-16` in `H/nu`
over the same 1,536 samples. This strongly disfavors that Hamiltonian as the
residual's cause under the controlled initialization; the wheel generator's
private Hamiltonian argument was not captured directly. Mode prefactors or
build-level floating-point behavior remain candidates, and this diagnostic
does not identify one. Compiling the wheel-frequency source-mode ablation with
`-O2 -ffast-math` raises the amplitude RMS difference on every row (from
`1.98e-11`–`2.53e-11` to `3.01e-11`–`3.12e-11`), so that simple build switch
does not close the residual.

**Untouched q8 and mixed-spin stress rows.** The [frozen plan](source_candidate_new_rows_plan.json)
and [result](source_candidate_new_rows.json) apply the same source-convention
candidate to `(q, chi1, chi2)=(8, 0.85, 0.85)` and `(4, 0.8, -0.6)`, both
starting at `M Omega=0.04`. The q8 refined complex numerical relative L2 is
`1.2687e-6`, just above the frozen `1e-6` gate; the mixed-spin row passes at
`5.9242e-7`. Against the pinned public SEOBNRv4 mode, their peak-fitted
whole-mode complex relative L2 differences are `0.40226` and `0.01154`.
The q8 discrepancy is mostly phase (`0.42191` inspiral complex relative L2,
versus `8.73e-4` inspiral amplitude relative L2). These failures preclude
generator qualification or independent ROM training. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_source_candidate_new_rows
--json-out /tmp/source-candidate-new-rows.json` in the pinned environment.

The [q8 same-state field plan](q8_inspiral_field_plan.json) and
[result](q8_inspiral_field.json) compare the native source-convention vector
field at exact public states through `M Omega=0.12`. Differences from sampled
public derivatives are at most about `1.9e-6` relative in angular-momentum
rate on the finest `0.0625 M` output grid; these are finite-difference
estimates, not direct compiled right-hand sides. The
[equal-frequency start plan](q8_initial_alignment_plan.json) and
[result](q8_initial_alignment.json) find native-minus-public differences at
`M Omega=0.04` of about `-3.7e-8` in radius, `-3.2e-8` in angular momentum,
and `-3.6e-9` in radial momentum using the native-field frequency crossing.
The two public output grids agree at this scale. The start state cannot by
itself explain the large peak-anchored mode phase gap.

The [q8 orbital-phase plan](q8_orbital_phase_plan.json) and
[result](q8_orbital_phase.json) independently integrate the native field from
that frozen warm-start state and compare orbital phase spans at equal orbital
frequencies on both public grids. Through `M Omega=0.13`, the largest
native-minus-public phase-span difference is `2.03e-5 rad` and the largest
elapsed-time-span difference is `1.21e-4 M`. This localizes the large q8
whole-mode gap downstream of the tested initial state and pre-peak orbital
dynamics. The comparison does not establish raw-mode, NQC, merger, ringdown,
or complete waveform parity. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_q8_orbital_phase --json-out
/tmp/q8-orbital-phase.json`.

The [frozen q8 mode-phase plan](q8_mode_phase_plan.json) and
[result](q8_mode_phase.json) replay the fine native waveform hash and the
pinned public waveform, then fit each amplitude peak locally without fitting
their relative time or phase. The peak-anchored native-minus-public phase
residual is `0.494 rad` at `-1000 M`, `0.335 rad` at `-100 M`, and about zero at
the peaks. Reanchoring phase at `-1000 M` is a diagnostic counterfactual: it
reduces the inspiral complex relative L2 from `0.42191` to `0.11026`, but the
whole-mode L2 remains `0.19976`. The residual varies in time, so a constant
phase convention alone does not explain it. The next frozen comparison uses
each mode's own early gravitational-wave frequency crossing as a physical time
and phase origin; no fitted correction is accepted as release evidence.

The [predeclared frequency-anchor plan](q8_frequency_anchor_plan.json) and
[result](q8_frequency_anchor.json) use each mode's first prepeak
`M omega22=0.1` crossing as its own time and phase origin. They do not optimize
relative time, phase, amplitude, or model coefficients. The q8 whole-mode
complex relative L2 falls from `0.40226` under independent amplitude-peak
alignment to `0.0036573` under frequency anchoring. The corresponding inspiral
and postpeak values are `0.0010316` and `0.0146391`; the independently located
amplitude peaks occur `1.33785 M` apart relative to their frequency anchors.
The large peak-aligned gap therefore principally reflects a different peak
time, not a half-radian prepeak phase error at equal frequency. This is a
diagnostic alignment, not a release fidelity gate or parameter fit. Reproduce
with `PYTHONPATH=src:. python -m dev.native_rom.diagnose_q8_frequency_anchor
--json-out /tmp/q8-frequency-anchor.json` in the pinned environment.

For the same q8 inputs, the fitted merger-ringdown amplitude shape in
[Bohé et al. Eq. (2.12)](https://arxiv.org/pdf/1611.03703v3) has positive
curvature at its nominal zero-slope attachment: writing the QNM damping as
`gamma`, the C1-matched ansatz gives
`A''(0)/A(0)=-gamma*(gamma+2*c1f*tanh(c2f))`. A true local amplitude maximum
requires the expression to be nonpositive. The published polynomial fit for
`c1f` can violate this condition at q8, allowing the native ringdown amplitude
to rise after attachment. The [frozen counterfactual
plan](q8_ringdown_peak_bound_plan.json) tests a cap derived solely from this
condition on the development branch; it does not change the production
implementation or establish that the cap is the complete model prescription.
The [counterfactual result](q8_ringdown_peak_bound.json) replays the frozen
fine q8 time-grid hash, changes only this ringdown parameter, and retains the
same physical frequency anchor. The unbounded value `c1f=0.0617872` has
`A''(0)/A(0)=+2.93277e-4`; the bound `c1f=0.0580202` makes the curvature zero.
The native-minus-public amplitude-peak lag falls from `1.33785 M` to
`0.03442 M`. Peak-fitted whole-mode complex relative L2 falls from `0.40226`
to `0.01137`; frequency-anchored whole-mode L2 falls from `0.003657` to
`0.001865`, with postpeak L2 from `0.01464` to `0.005837`. The latter residual
and the untested parameter domain prevent a physical parity claim. No
production code changes or ROM training follow from this one-row ablation.
Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_q8_ringdown_peak_bound --json-out
/tmp/q8-ringdown-peak-bound.json` in the pinned environment.

The separate [q4 frequency-anchor plan](q4_frequency_anchor_plan.json) and
[result](q4_frequency_anchor.json) replay the untouched mixed-spin fine hash
and apply the same prepeak `M omega22=0.1` anchor. The native-minus-public
amplitude-peak lag is `-0.05094 M`. Whole-mode complex relative L2 is
`0.003162` at this anchor, versus `0.01154` with independent peak fitting;
the inspiral and postpeak frequency-anchored values are `0.000942` and
`0.009976`. The mixed-spin postpeak discrepancy requires separate diagnosis. Neither
anchor is a fitted validation metric or a domain-wide qualification. Reproduce
with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_q4_frequency_anchor --json-out
/tmp/q4-frequency-anchor.json` in the pinned environment.

The [frozen q4 postpeak-profile plan](q4_postpeak_profile_plan.json) and
[result](q4_postpeak_profile.json) sample the same waveform relative to the
public amplitude peak, with no new alignment or fitted correction. The native
amplitude divided by the public amplitude changes from `0.999618` at the peak
to `0.997402` at `80 M`, then remains at `0.997402` at `120 M`. The phase
residual similarly settles from `-0.002535` to `-0.006844 rad`. Over
`70–120 M`, native/public log-amplitude slopes are `-0.0721019270` and
`-0.0721019754` per `M`; phase slopes are `-0.6641033113` and
`-0.6641033840 rad/M`. The asymptotic slopes agree closely, while the
postpeak transient leaves a persistent normalization and phase offset. This
localizes the residual; it does not establish which attachment ingredient
causes it or qualify the waveform. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_q4_postpeak_profile --json-out
/tmp/q4-postpeak-profile.json` in the pinned environment.

The [frozen q8 tighter-ODE plan](q8_tighter_ode_plan.json) and
[result](q8_tighter_ode.json) replay the original fine q8 waveform hash, then
hold the `0.5 M` output grid and all scientific choices fixed while tightening
the ODE tolerances from `(atol, rtol)=(1e-14, 1e-13)` to
`(3e-15, 3e-14)`. The direct complex relative L2 is `3.4126e-7` and the
amplitude relative L2 is `2.5957e-9`; peak, match, and boundary time
differences are zero at this grid. This additional step passes the frozen
`1e-6` source-stability gate after the prior tolerance step missed it at
`1.2687e-6`. It qualifies neither the q8 physical waveform nor the parameter
domain; this is pre-peak-condition generator evidence. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_q8_tighter_ode
--json-out /tmp/q8-tighter-ode.json` in the pinned environment.

The [frozen integrated peak-condition plan](peak_condition_integrated_plan.json)
and [result](peak_condition_integrated.json) rerun the Torch q8 and q4 `ode_2`
adaptive paths with the ansatz-derived continuation. The same continuation is
wired into the JAX attachment path and checked against the independent scalar
reference, but no full JAX adaptive waveform is qualified. The q8 mode hash matches the
previous independent one-row counterfactual exactly; the q4 hash matches its
unchanged control exactly. At the predeclared `M omega22=0.1` anchor, q8 has
whole-mode complex relative L2 `0.00186519`, postpeak L2 `0.00583740`, and
native-minus-public peak lag `0.0344231 M`. Q4 has whole-mode L2 `0.00316192`,
postpeak L2 `0.00997582`, and peak lag `-0.0509413 M`; all are identical to
their frozen comparators. The paper ansatz supplies the curvature condition,
but the continuation is our uncalibrated choice, not a coefficient fit printed
by Bohé et al. These two replays verify integration and preserve the prior
physical residuals; they do not qualify the model or ROM. The evidence records
the input, script, and source hashes. Reproduce with `PYTHONPATH=src:. python
-m dev.native_rom.diagnose_peak_condition_integrated --json-out
/tmp/seobnrv4-peak-condition-integrated.json` in the pinned environment.
