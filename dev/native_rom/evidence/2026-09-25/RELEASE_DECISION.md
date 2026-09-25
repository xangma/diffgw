# Native SEOBNRv4 and proposed ROM: release decision (2026-09-25)

**Verdict: unqualified.** Keep the native generator and proposed independent ROM
on the development branch and outside the release catalog. Selected Torch/JAX
backend comparisons pass, but they do not establish physical validity or the
whole-waveform compiled-gradient contract.

The smallest existing prospectively fixed physical pass/fail check is the
[third five-case SXS plan](../2026-09-24/sxs_third_holdout_plan.json) with its
[weighted gate](../2026-09-24/sxs_third_holdout_weighted_plan.json): a complex
dominant-mode `h22` comparison with independent NR at total masses 10–200
solar masses in steps of one. Each selected case must have detector-weighted
overlap at least 0.99 after one bounded global time and phase shift, with no
amplitude or intrinsic-parameter fit. LAL supplies only the pinned noise PSD,
not waveform targets. This is a finite selected-case proxy, not a whole-domain
or fixed-chart full-complex-waveform release guarantee. The existing fixed-chart
complex waveform comparisons predeclare no physical pass threshold.

The [original frozen result](../2026-09-24/sxs_third_holdout_weighted.json)
fails one of five cases: SXS:BBH:2569 has minimum overlap
`0.9891417104357128` at 22 solar masses, against `0.9999241175571614` for
the low/high-resolution NR control. The numerical-convergence diagnostic
[for that case](../2026-09-24/sxs_2569_convergence.json) reports a
`6.65e-8` complex-waveform relative L2 change under tenfold tighter ODE
tolerances; this does not explain the overlap deficit. A fresh
[stored-array replay](sxs_third_holdout_weighted_replay.json) uses the same
frozen plans and input array hashes. Its five minimum overlaps agree with the
original within `3.4e-16`; it also fails 2569 at `0.9891417104357124`.
The replay ran on RONI2 using the local Python 3.13 environment rather than
the original Python 3.12 environment, so its binary fingerprints and runner
hash differ; it is a reproducibility check of stored data, not a new holdout.

**Source-revision limitation:** the original anchored waveforms identify
generator source-manifest hash
`8ca6d152b4b07bb89bbbd97734d29814023148b79f0af448fda67079957ca532`,
whereas the source snapshot at the current-candidate replay hashed to
`2062f70072d9953a7f9b968336c80512b8034497eb2e4cb0fea7ad2633b7d2cf`.
Generator files changed between those snapshots. A fresh
[current-source execution of the known 2569 case](sxs_2569_current_regression.json)
now removes the ambiguity for this case: the anchored native array has exactly
the same SHA-256 as the original, `4180cbd3aaee9eb492d776d291ecd603c1ba19483777f0e326d18d525d7527c7`.
The minimum detector-weighted overlap is again `0.9891417104357124` at 22
solar masses, below the frozen 0.99 gate. This is a current-source failure on
an already inspected case, not an untouched release holdout. The other four
cases have not been rerun on the current source; this one failure is sufficient
to keep the candidate unqualified under the selected-case gate.
The subsequent catalog/API-only commit `a2fd3fe` changed the broad source
snapshot hash without changing the generator physics. The replay's exact
array hash and its recorded source snapshot remain the evidence for this case;
the broad hash is not a claim about later repository revisions.
The [pinned public SEOBNRv4 control](../2026-09-24/sxs_third_public_parity.json)
scores `0.9891409499192219` at the same mass, while native and public complex
waveforms differ by only `0.0001174` relative L2 on their shared chart. Thus
this miss is already present in the historical public model within the measured
parity error; adjusting the native port to pass by changing its physics would
need a distinct successor-model identity and a new validation campaign.

A [fresh five-case SXS reservation](fresh_sxs_metadata_reservation.json) records
the identifiers, exact archive checksums, and 0.99 gate intent selected from a
pinned public metadata catalog. None of those five waveforms has been opened.
The complete source-specific plan and numerical/fixed-chart thresholds must be
frozen before intake; this reservation does not turn the failing current
candidate into a qualified one.

The [source specification](../../reference/SPEC.md) remains incomplete: six
of 48 manifest components are `pending`, including the full flux aggregate,
QNM interpolation, ODE initial conditions, TD/FD conventions, and the ROM
construction. The
published model does not supply independently derived ROM bases and spline
tables, and no qualified generator/training set or independent ROM holdout
exists. Full-waveform gradients and domain-wide convergence remain open.
The [frozen nonzero-reference TD/FD plan](td_fd_nonzero_reference_plan.json)
and [result](td_fd_nonzero_reference.json) extend the identical-input
public-binary adapter check to two new selected rows at `f_ref=40 Hz`, phase
`0.41 rad`, inclination `1.1 rad`, and distance `173 Mpc`. Both pass their
predeclared `1e-10` conditioning/FFT/complex-FD and `1 ns` epoch gates. The
largest public complex FD relative L2 error is `1.18e-12`, FFT-only error
is below `3.43e-16`, and epoch error is `0.25 ns`. The public generator
supplies both TD inputs, so this checks the adapter, not native generator
fidelity or general TD/FD conventions; the aggregate remains pending. It ran
on RONI2 in this worktree as PID `85647`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_td_fd_nonzero_reference`,
log `/tmp/diffgw-td-fd-nonzero-reference.log`, and completed in 1.5 seconds
with exit code 0. No process or next check remains.
The new conditional native-QNM spline, zero-radial ODE, and nonzero-radial ODE
vector-field leaves make their arithmetic and paper-derived equations
independently executable.
The former checks selected Berti/Cook author rows and the native coefficient
rule, but has no continuous bound or historical-spline claim. The latter
checks four supplied states with an external torque and zero radial force,
not the physical flux, initial-data root, or integrated trajectory. Neither
leaf closes its pending aggregate component.
The nonzero-radial leaf checks four supplied states in each frontend with
caller-supplied radial and angular forces. It verifies the local tortoise
Hamilton equations, but not the physical flux or an integrated trajectory.
The newly executable tail-modulus leaf independently checks the scalar, Torch,
and JAX supplied-input factor against the published complex-gamma expression;
it does not construct the modes or close the pending full-flux aggregate.
The conditional paper-flux ODE leaf joins all 35 independently computed
pre-NQC amplitudes to a scalar-differentiated Hamiltonian at two supplied
nonzero-radial states per frontend. Its four-component Torch/JAX comparisons
pass at `3e-10` relative / `2e-12` absolute. Historical mode selection,
initial data, integration, convergence, and physical overlap gates remain open.
The initial-data source audit corrects the paper trail: [Bohé et al. Sec. II.2](https://arxiv.org/html/1611.03703)
cites [Buonanno, Chen, and Damour (2006), Sec. IV.1](https://arxiv.org/html/gr-qc/0508067) for the quasicircular
prescription. A separate two-point scalar check recomputes neighboring
circular-orbit energies and verifies the leading-adiabatic energy balance and
tortoise radial velocity within `2e-9` relative using supplied torque. The
existing Torch/JAX initial-state checks remain conditional; the historical
Cartesian root solve, physical torque, and integrated waveform are unqualified.
The existing [public/NR chirp diagnostic](sxs_public_chirp.json) locates a
waveform-phase discrepancy on the already inspected 2569 case: at mode
frequency `0.05/M`, relative to a `0.09/M` anchor and smoothing window 20,
public versus high-resolution NR differs by `0.5733` rad, while public versus
native differs by `0.0001914` rad. The
[cross-resolution control](sxs_cross_run_chirp.json) has low/high NR phase
difference `0.002706` rad at that frequency. These post-hoc measurements
support a shared historical-model discrepancy over a porting or NR-resolution
explanation on this chart; they do not identify which internal physical term
causes it or qualify a changed model.
The [frozen post-hoc pre-peak PSD plan](sxs_2569_prepeak_psd_plan.json) and
[result](sxs_2569_prepeak_psd.json) reproduce the original 22-solar-mass
full-waveform overlaps before truncation. Cutting all waves at the high-NR
`Mω22=0.09` crossing, before their amplitude peaks, gives native/high-NR
overlap `0.9976975`, public/high-NR `0.9976966`, and low/high-NR `0.9999546`
with a 50M end taper. The native/high-NR value spans `0.9972783` to
`0.9982056` across predeclared 25–100M end tapers; the 0.08/0.10 frequency
brackets give `0.9979298`/`0.9974554`. Thus the mismatch is already present
before peak, while including the later waveform worsens the globally aligned
overlap to `0.9891417`. The optimized time shift also changes substantially
between truncated and full waveforms, so the pre-peak comparison does not
isolate one physical term. Its Fourier-bin residual split is algebraic, not a
unique time-domain amplitude/phase or NQC attribution. These inspected-case
results do not replace the frozen release gate or qualify a new model.
The [frozen JAX pathwise whole-mode gradient diagnostic](jax_pathwise_whole_gradient.json)
passes its finite-difference and accepted-mesh replay checks, but fails its
eager/JIT waveform gate: relative L2 is `1.547138e-9` against `1e-10`.
Trajectories and the pre-NQC mode agree to roundoff; matching-derivative
roundoff is amplified by the NQC solve. This one-row derivative path is itself
narrower than a production physical-parameter gradient claim. The separately
[frozen spline-barrier experiment](jax_nqc_spline_barrier_plan.json) applied a
JAX compiler barrier immediately before that dev-only NQC spline. Its
[result](jax_nqc_spline_barrier.json) changed none of the measured eager or JIT
stages: the fixed-input amplitude coefficients still differ by `2.78e-7`,
above the experiment's `1e-8` gate. The fixed-chart attached waveform differs
by `3.37e-11` relative L2, below its `1e-10` gate, but that narrower pass does
not resolve the original whole-path `1.547138e-9` failure. Keep both failures
and investigate numerical conditioning without weakening their frozen gates.
The subsequent [frozen amplitude-solve experiment](jax_nqc_amplitude_solve_plan.json)
tested local match-cell derivative-row scaling and residual-centered iterative
refinement. Its [result](jax_nqc_amplitude_solve.json) also fails: the
coefficient gap remains about `2.78e-7` and the fixed-input corrected-mode
eager/JIT relative L2 remains about `1.30e-10`. The existing solve's row
equilibration cancels the extra scale; refinement barely changes the eager
answer. The next numerical target was upstream spline derivative formation.
The [frozen spline-normalization experiment](jax_nqc_spline_normalization_plan.json)
then shifted and scaled time around the match cell, centered spline ordinates,
and combined both changes. Its [result](jax_nqc_spline_normalization.json)
preserves the eager corrected mode exactly, but all three variants reproduce
the baseline eager/JIT differences: `2.783505e-7` in amplitude coefficients
against `1e-8`, and `1.295776e-10` corrected-mode relative L2 against
`1e-10`. The matching second derivative still differs by `2.613556e-14`.
This algebraic normalization does not repair the compiled-gradient contract;
the failed result remains part of the release evidence.

The [frozen continuous Lie-derivative amplitude experiment](jax_nqc_ode_lie_plan.json)
was a development-only attempt to replace the three spline amplitude-matching
rows. Its [partial result](jax_nqc_ode_lie.json) passes the fixed-input
eager/JIT amplitude coefficient and corrected-mode checks, but the whole-mode
gap is still `1.539901e-9`, above `1e-10`. The experiment was stopped after
the automatic-differentiation compilation reached about 64 GiB RSS; its AD/FD
check is **incomplete**, not a pass. The intervention also changes the eager
waveform relative to the original spline method. The
[fixed-row readout](jax_nqc_fixed_rows.json) shows exact eager/JIT agreement
for phase matching on identical inputs, while the
[frozen two-by-two whole attribution](jax_nqc_whole_attribution_plan.json) and
[result](jax_nqc_whole_attribution.json) isolate the remaining discrepancy:
attachment assembly on identical replay arrays agrees within `1.1e-15`
relative L2, but replay states differing by at most `1.78e-15` shift phase
NQC coefficients by `1.21e-7` and the whole mode by `1.54e-9`. The Lie
amplitude coefficients shift by only `3.41e-12`. The Lie intervention changes
the eager whole mode by `8.36e-6` relative L2 compared with the source spline
method, so it is not source preserving and cannot be promoted. The original
whole-waveform JAX gate remains failed. The completed no-AD attribution ran
on RONI2 in this worktree as PID `40048`, with log
`/tmp/diffgw-jax-nqc-whole-attribution.log`; it finished in 143 seconds,
requires no next check, and has no running process. Its command was
`env PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_whole_attribution > /tmp/diffgw-jax-nqc-whole-attribution.log 2>&1`.

The [current-source replay of the earlier JAX stage diagnostic](jax_nqc_jit_stages_current_replay.json)
locates the phase sensitivity more narrowly. Eager/JIT unwrapped phase differs
by at most `7.11e-15`; its spline-derived phase right-hand side differs by
`3.99e-11`, while the phase matching matrix is identical. Solving that same
matrix (condition number about 98) yields the observed `1.21e-7` coefficient
gap. This is a measured replay-to-phase-spline derivative amplification, not
evidence that the matrix solve itself changes under JIT. The replay used the
unchanged frozen stage plan and current script hash on RONI2 in this worktree,
PID `47462`, and completed in 17 seconds. Its log is
`/tmp/diffgw-jax-nqc-jit-stages-current.log`; no process remains or next check
is needed. It was invoked with `DIFFGW_DIAGNOSTIC_OUTPUT=/tmp/diffgw-jax-nqc-jit-stages-current.json`
and `python -m dev.native_rom.diagnose_jax_nqc_jit_stages`.

The subsequent [frozen direct-phase experiment](jax_nqc_direct_phase_plan.json)
replaced the development NQC phase-unwrapping input with the continuous
published phase computed directly from orbital state. Its
[failed result](jax_nqc_direct_phase.json) preserves the eager source whole
mode to `4.49e-16` relative L2, but whole eager/JIT L2 is `1.547e-9`
against `1e-10` and the phase-coefficient gap is `1.212e-7` against `1e-8`.
Thus cumulative phase unwrapping is not the cause on this chart. Fixed-input
corrected-mode eager/JIT L2 is `1.296e-10`, also above its `1e-10` gate.
AD/FD was not rerun after the failed whole-mode gates; this experiment is a
negative diagnostic, not release evidence. It ran on RONI2 in this worktree
as PID `56543` with log `/tmp/diffgw-jax-nqc-direct-phase.log`, completed in
30 seconds with exit code 1 for frozen gate failure, and has no remaining
process or next check. The command was
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_direct_phase`.

The [frozen phase-centering experiment](jax_nqc_phase_centering_plan.json)
subtracted the phase at the left NQC match knot before constructing the
development spline and restored it in the final mode. The
[failed result](jax_nqc_phase_centering.json) preserves the eager source mode
to `4.84e-16` relative L2, but whole eager/JIT L2 remains `1.547138e-9`
against `1e-10`, and the phase-coefficient gap remains `1.212475e-7`
against `1e-8`. Its fixed-input corrected mode is `1.296e-10` against
`1e-10`. Thus a constant phase offset does not explain the derivative
amplification. AD/FD was not rerun after the failed whole-mode gates and the
previous AD resource limit. The diagnostic ran on RONI2 in this worktree as
PID `80116`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_phase_centering`,
with log `/tmp/diffgw-jax-nqc-phase-centering.log`. It finished in 29 seconds
with exit code 1 for the frozen gate failure; no process or next check remains.

The [frozen continuous-phase-derivative control](jax_nqc_continuous_phase_plan.json)
then replaced only the phase NQC matching right-hand side with the published
frequency and its Lie derivative at the same fixed-chart EOB state. Its
[result](jax_nqc_continuous_phase.json) reduces the replay-induced phase
coefficient gap from `1.21e-7` to `2.13e-14`, but the eager waveform changes
by `1.907405e-5` relative L2 against the original spline match. The remaining
amplitude-coefficient gap is `1.291122e-6`; whole eager/JIT relative L2 is
`1.494895e-10`, still above the unchanged `1e-10` gate. This is a failed,
no-AD development control, not source-preserving or release-qualified. It
ran on RONI2 in this worktree as PID `43085`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_continuous_phase`,
with log `/tmp/diffgw-jax-nqc-continuous-phase.log`. It finished in 35
seconds with exit code 1 for the frozen gate failures; no process or next
check remains. The next numerical investigation should isolate accepted-knot
versus spline-resampling eager/JIT drift, since replacing matching semantics
has not preserved the historical waveform.

The [frozen replay-stage attribution](jax_replay_stage_attribution_plan.json)
and [result](jax_replay_stage_attribution.json) now locate the first eager/JIT
discrepancy in natural-spline resampling. Both coarse and fine accepted RKF45
knots are bit-identical between eager and JIT; each uniform state series
differs by at most `1.776357e-15`. Compiling only the resampler on identical
eager knots reproduces that difference. This excludes accepted-step replay
as the initiating cause on this chart. It ran on RONI2 in this worktree as
PID `47674`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_replay_stage_attribution`,
log `/tmp/diffgw-jax-replay-stage-attribution.log`, and completed in 10
seconds with exit code 0. No process or next check remains.

The [frozen resampler-barrier control](jax_resample_barrier_plan.json) and
[failed result](jax_resample_barrier.json) insert optimization barriers at
either the accepted-knot input or the four cubic-polynomial coefficient
arrays. Both preserve the eager state exactly but leave the same
`1.776357e-15` eager/JIT state gap, so neither is a repair. An exploratory
fixed-input decomposition found eager/JIT differences in second-derivative
construction (about `1.1e-16`), coefficient construction from identical
second derivatives (up to `2.8e-17`), and spline evaluation from identical
coefficients (up to `1.8e-15`). A stable implementation must therefore
address the arithmetic throughout the resampler, then recheck source parity,
whole-waveform and AD/FD gates. The barrier control ran on RONI2 in this
worktree as PID `49258`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_resample_barrier`,
log `/tmp/diffgw-jax-resample-barrier.log`, and completed in 24 seconds with
exit code 1 for the frozen bitwise-equality gates. No process or next check
remains.

The [frozen fixed-coefficient Horner control](jax_fixed_spline_horner_plan.json)
and [result](jax_fixed_spline_horner.json) isolate one repairable part of the
resampler: with identical precomputed coefficients, optimization barriers
between each polynomial multiply and add make coarse and fine eager/JIT
samples bit-identical while reproducing the original eager samples exactly.
This is a positive fixed-coefficient control only. Coefficient construction
itself still shows eager/JIT roundoff, so the full resampler and whole waveform
remain unqualified. It ran on RONI2 in this worktree as PID `53995`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_fixed_spline_horner`,
log `/tmp/diffgw-jax-fixed-spline-horner.log`, and completed in 3.3 seconds
with exit code 0. No process or next check remains.

The [frozen full-spline arithmetic control](jax_full_spline_arithmetic_plan.json)
and [result](jax_full_spline_arithmetic.json) applied barriers through the
Thomas solve, cubic coefficient construction, and evaluation. It preserved
the eager resampled state to `2.78e-17` maximum absolute error and reduced
fine-grid eager/JIT state drift to `1.39e-17`, but missed the prospectively
frozen bitwise-equality gate. It ran on RONI2 in this worktree as PID `56108`,
command `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_full_spline_arithmetic`,
log `/tmp/diffgw-jax-full-spline-arithmetic.log`, and completed in 14 seconds
with exit code 1. No process or next check remains.

The follow-on [frozen whole-waveform control](jax_stable_resample_whole_plan.json)
and [failed result](jax_stable_resample_whole.json) used that resampler only
for accepted-knot state sampling. The eager whole mode matches the unchanged
baseline exactly, but eager/JIT whole relative L2 is `1.547139e-9` against
`1e-10`; phase and amplitude NQC coefficient gaps remain `1.21e-7` and
`1.29e-6`. Thus eliminating almost all state-resampling drift does not
stabilize the model's mode/NQC path. This diagnostic had no AD/FD check and
must not be promoted. It ran on RONI2 in this worktree as PID `58118`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_stable_resample_whole`,
log `/tmp/diffgw-jax-stable-resample-whole.log`, and completed in 28 seconds
with exit code 1. No process or next check remains. The next numerical
attribution should hold resampled states identical and compare eager/JIT
published-mode evaluation before revisiting the NQC spline.

The [frozen fixed-state mode/NQC control](jax_fixed_state_mode_plan.json)
and [failed result](jax_fixed_state_mode.json) complete that attribution on
the same chart. Holding state arrays identical, separate eager/JIT pre-NQC
mode evaluations differ by only `1.02e-15` relative L2 on the fine grid;
orbital and continuous frequencies also meet their `1e-12` diagnostic
gates. Nevertheless the NQC fit on identical state, frequency, and mode
inputs changes the amplitude coefficients by `1.39e-7`, missing its frozen
`1e-9` gate. Substituting the separately computed JIT mode adds `1.15e-6`
of amplitude coefficient difference and `1.21e-7` of phase coefficient
difference; substituting the JIT resampled state after that adds zero at
reported precision. This isolates both an intrinsic eager/JIT NQC solve/
spline sensitivity and amplification of pre-NQC mode roundoff. It does not
identify a source-preserving repair; any new numerical formulation still
needs independently frozen whole-waveform, source, AD/FD, and multi-case
checks. The diagnostic ran on RONI2 in this worktree as PID `62961`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_fixed_state_mode`,
log `/tmp/diffgw-jax-fixed-state-mode.log`, and completed in 23 seconds with
exit code 1 for the frozen NQC gate. No process or next check remains.

The next [frozen NQC stable-spline control](jax_nqc_stable_spline_plan.json)
and [failed result](jax_nqc_stable_spline.json) put arithmetic barriers
through the development-only natural-spline Thomas solve, coefficient
construction, and matching-row evaluation while retaining the physical
targets and original equilibrated linear solves. It preserved the eager
corrected mode to `1.08e-15` relative L2 and eager coefficients to `3.58e-12`
maximum absolute error. On identical input state, orbital frequency, and
base mode arrays, however, the eager/JIT amplitude coefficients still differ
by `2.78e-7` against the frozen `1e-9` gate. Hence this spline-only arithmetic
barrier is not a source-preserving numerical repair. It ran on RONI2 in this
worktree as PID `66594`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_stable_spline`,
log `/tmp/diffgw-jax-nqc-stable-spline.log`, and completed in 8.8 seconds
with exit code 1 for the frozen fixed-input gate. No process or next check
remains. The follow-on control below supplied an identical precomputed NQC
basis/product array directly to both spline executions; no whole-path claim
follows from this negative control.

The [frozen fixed-combined attribution](jax_nqc_fixed_combined_plan.json)
and [result](jax_nqc_fixed_combined.json) now isolate the input to that
spline. Passing the exact same precomputed combined amplitude/basis-product
array into eager and JIT spline calls gives bit-identical matching value,
first-derivative, and second-derivative rows; the unchanged amplitude solve
also returns bit-identical coefficients. Separately constructing the combined
array from identical state, frequency, and base-mode inputs changes it by at
most `1.39e-17` absolute (`2.73e-18` relative L2), and this produces the
`2.78e-7` amplitude-coefficient change. Therefore the fixed-input NQC failure
enters during the basis/product-array construction and is amplified by the
matching problem; the data do not yet isolate which individual basis
operation differs. This control is an attribution pass, not a source-
preserving whole-waveform repair or release pass. It ran on RONI2 in this
worktree as PID `70178`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_fixed_combined`,
log `/tmp/diffgw-jax-nqc-fixed-combined.log`, and completed in 8.3 seconds
with exit code 0. No process or next check remains.

The [frozen basis-arithmetic control](jax_nqc_basis_barrier_plan.json)
and [passing fixed-chart result](jax_nqc_basis_barrier.json) place explicit
rounding barriers in the development-only NQC basis and amplitude-product
construction. It reproduces the unchanged eager combined array and amplitude
coefficients exactly. The separately evaluated eager/JIT combined arrays are
bit-identical, and the amplitude coefficient gap falls from `2.78e-7` to
`4.26e-14`, inside the prospective `1e-9` gate. This result only covers one
fixed NQC chart; it has not passed the complete eager/JIT waveform, source,
AD/FD, multi-case, or NR gates. It ran on RONI2 in this worktree as PID
`72796`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_basis_barrier`,
log `/tmp/diffgw-jax-nqc-basis-barrier.log`, and completed in 8.8 seconds
with exit code 0. No process or next check remains. The next prospective
control should thread the stable amplitude rows through the unchanged whole-
waveform attachment while measuring any remaining phase and pre-NQC drift.

The follow-on [frozen whole-waveform control](jax_nqc_basis_whole_plan.json)
and [failed result](jax_nqc_basis_whole.json) did exactly that. The eager
candidate preserves the original full complex waveform to `2.61e-16`
relative L2, but eager/JIT whole-mode L2 remains `1.545667e-9` against the
prospective `1e-10` gate. Its amplitude and phase coefficient gaps are
`1.15e-6` and `1.21e-7`. The fixed-input basis repair is therefore
insufficient once the pre-NQC mode is separately generated by eager and JIT;
the upstream `~1e-15` mode difference is still amplified by matching. This
control had no AD/FD or independent source/NR check and must remain in dev.
It ran on RONI2 in this worktree as PID `75055`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_nqc_basis_whole`,
log `/tmp/diffgw-jax-nqc-basis-whole.log`, and completed in 27.6 seconds
with exit code 1 for the frozen whole-waveform and amplitude gates. No
process or next check remains.

The [frozen pre-NQC stage attribution](jax_mode_stage_plan.json) and
[result](jax_mode_stages.json) reproduce the published eager waveform mode
exactly through an instrumented decomposition, in both eager and JIT paths.
On an identical fine state, the published mode changes by `1.0184e-15`
relative L2 between eager and JIT. Hamiltonian-derived frequencies and
energies already differ at about `1e-16` relative L2, residual phase by
`5.08e-16`, and amplitude by `6.11e-16`. Even when the compiled amplitude,
phase, and mode receive the identical eager quantities and residual phase,
the mode still differs by `3.97e-16` relative L2. This is distributed
floating-point arithmetic drift, not a single bad pre-NQC component; the
subsequent NQC fitting magnifies it. The attribution passes its reproduction
checks, but proposes no waveform repair and does not qualify a release.
It ran on RONI2 in this worktree as PID `78894`, command
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_jax_mode_stages`,
log `/tmp/diffgw-jax-mode-stages.log`, and completed in 7.7 seconds with
exit code 0. No process or next check remains.

Two different release claims must stay separate:

- **Historical SEOBNRv4 fidelity:** a port may be described as faithful only
  after a frozen, independently checked full-waveform source-fidelity and
  backend contract is met. It would need to disclose the observed sub-0.99 NR
  proxy result rather than imply that historical parity guarantees that gate.
- **Physical NR accuracy at the 0.99 gate:** a release under this claim needs a
  prospectively frozen current candidate, a new untouched independent NR
  holdout with valid start frequency and cap, and full-complex waveform,
  convergence, and applicable gradient checks. If a faithful historical model
  cannot pass the gate, a calibrated successor needs its own model identity.

The next step depends on the release claim. For a faithful historical port,
finish missing clean-room component provenance and freeze the source-fidelity,
numerical, and gradient contracts while disclosing the observed NR limitation.
For a model required to clear the 0.99 NR gate, diagnose the known 2569 miss
using independent physical evidence, develop a separately named successor,
and freeze its generator revision and complete numerical/gradient contract.
Then predeclare a fixed-chart complex-waveform threshold and the full weighted
gate on the reserved untouched NR rows before running them.
Only after that generator passes should independent ROM data be trained and
tested on reserved holdouts. Neither criterion currently permits release.

The current-source 2569 execution ran on host RONI2, cwd
`/Users/xangma/.codex/worktrees/native-seobnr-rom/diffgw`, PID `55820`
(managed session `6479`), and exited zero in 53 seconds. Its log is
`/tmp/diffgw-sxs-2569-current-regression.log`; no process remains or next
check is needed. The command was:

```sh
env PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python \
  -m dev.native_rom.diagnose_current_2569_regression \
  --plan dev/native_rom/evidence/2026-09-25/sxs_2569_current_regression_plan.json \
  --array-out /tmp/diffgw-sxs-2569-current-anchored.npz \
  --json-out dev/native_rom/evidence/2026-09-25/sxs_2569_current_regression.json \
  > /tmp/diffgw-sxs-2569-current-regression.log 2>&1
```

The completed replay ran on host RONI2, cwd
`/Users/xangma/.codex/worktrees/native-seobnr-rom/diffgw`, shell job/session
`67491`, and exited zero. Its log is
`/tmp/diffgw-sxs-third-holdout-weighted-replay.log`; no process remains or next
check is needed. The command was:

```sh
env PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python \
  -m dev.native_rom.measure_sxs_third_holdout_weighted \
  --case-plan dev/native_rom/evidence/2026-09-24/sxs_third_holdout_plan.json \
  --weighted-plan dev/native_rom/evidence/2026-09-24/sxs_third_holdout_weighted_plan.json \
  --comparison dev/native_rom/evidence/2026-09-24/sxs_third_holdout_comparison.json \
  --decoded-dir /tmp/diffgw-sxs-third-holdout/decoded \
  --archive-dir /tmp/diffgw-sxs-third-holdout/archive \
  --json-out dev/native_rom/evidence/2026-09-25/sxs_third_holdout_weighted_replay.json \
  > /tmp/diffgw-sxs-third-holdout-weighted-replay.log 2>&1
```
