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
of 46 manifest components are `pending`, including flux, QNM interpolation,
ODE initial conditions, TD/FD conventions, and the ROM construction. The
published model does not supply independently derived ROM bases and spline
tables, and no qualified generator/training set or independent ROM holdout
exists. Full-waveform gradients and domain-wide convergence remain open.
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
