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
to `0.0198485`. The wider window is opt-in, has no independent derivative
accuracy proof over the domain, and is not a physical-model correction.
