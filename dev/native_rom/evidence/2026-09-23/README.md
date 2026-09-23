# SEOBNRv4 native-generator diagnostics: numerical gate failed

This development-only evidence extends the [first campaign](../2026-09-22/README.md).
The recorded source snapshot was unchanged during both runs. The numerical
campaign uses scalar CPU `float64`, a 30+20 solar-mass nonspinning binary, and
starting orbital frequency `M Omega=0.04`. Exact inputs, source hashes, solver
costs, installed LAL binary identities, and per-check gates are in
[qualification.json](qualification.json). Its `git_head` is the parent commit
at run time; the source hash records the edited generator under test.

| Refined comparison | Coarse to medium complex L2 | Medium to fine complex L2 | Medium to fine amplitude L2 | `1e-6` gate |
| --- | ---: | ---: | ---: | --- |
| ODE tolerances `rtol=1e-11,1e-12,1e-13` | 1.902e-5 | 8.748e-6 | 3.331e-7 | **Failed** |
| Output grid `dt/M=0.5,0.25,0.125` | 1.648e-7 | 6.082e-6 | 2.585e-7 | **Failed** |

The paired absolute tolerances were `1e-12,1e-13,1e-14`. All five bounded
solves completed. Peak/match time changes at the tightest ODE pair were
`1.526e-5 M`, within their separate `0.05 M` gate. The output-grid error rose
at the last refinement, so these runs show sensitivity rather than demonstrated
convergence. Changing `dt` also changes the current adaptive initial step and
coarse/fine restart sampling. Event-peak location uses an explicit `1e-5 M`
bisection tolerance. No threshold or waveform coefficient was adjusted to
make the gate pass.

The subsequent [grid-coupling diagnostic](grid_coupling.json) held these two
numerical choices fixed separately. It used the same 30+20 nonspinning row
with `atol=1e-14`, `rtol=1e-13`, and exact nested waveform samples. The
default `dt/M=0.125` solve restarted at `164.875 M`, whereas the `0.5 M`
baseline restarted at `165 M`. Relative to that baseline:

| `dt/M=0.125` choice | Complex relative L2 |
| --- | ---: |
| Default first step and restart | 5.916e-6 |
| First step fixed at `0.5 M` only | 6.365e-6 |
| Restart fixed at `165 M` only | 9.817e-8 |
| Both fixed | 0 |

A `0.25 M` run with both fixed was also identical on nested samples. The
recorded source hash was unchanged during all six solves. Thus the restart
sample shift accounts for most of the measured grid discrepancy in this row;
the diagnostic overrides do **not** change the default published-convention
route or rescue its failed qualification gate. In particular, the ODE
tolerance error in the first campaign was `8.748e-6`, and no fixed-restart
run establishes correctness of the equations or full waveform. Reproduce
this diagnostic with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_grid_coupling --json-out
/tmp/seobnrv4-grid-coupling.json`.

An additional [tighter qualification run](tight_qualification.json) used
`rtol=(1e-13,1e-14,1e-15)` and paired
`atol=(1e-14,1e-15,1e-16)`, retaining the same default restart rule and
declared `1e-6` complex-error gate. All five solves completed and the recorded
source hash stayed unchanged. The ODE complex differences were `2.159e-6`
then `1.158e-5`; the grid differences were `1.789e-7` then `2.613e-5`.
Both checks failed, including their required decreasing-error condition. The
peak times differed by at most `1.526e-4 M` across the refined grid pair,
still within the separate `0.05 M` event-time limit. The nonmonotonic results
do not establish an asymptotic error regime or identify a precision floor.
This run did not change the waveform or promotion status. Reproduce it with
the `qualify_generator` command below, replacing both tolerance triples with
the tighter values and writing a separate JSON output.

The [regional ODE diagnostic](ode_regions.json) repeated its three ODE levels
at `dt/M=0.5`, partitioning each unaligned complex waveform into fixed time
windows. In the finest pair, the complex relative L2 was `1.158e-5`, versus
an amplitude relative L2 of `3.657e-7`; phase RMS was `1.488e-5 rad`.
Approximately 1% of squared complex error was before the `165 M` restart,
21% from `165–300 M`, 21% from `300–310 M`, 33% from `310–320 M`, and 24%
thereafter. The earlier tolerance pair had the same broad distribution, though
smaller errors. Thus the discrepancy is phase dominated and already present
in the inspiral, not confined to attachment or ringdown. These are descriptive
windows, not qualification gates or a determination of the numerical cause.
The source hash remained fixed. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_ode_tolerance --json-out
/tmp/seobnrv4-ode-regions.json`.

The [orbital-phase diagnostic](orbital_phase.json) integrates the same
provisional Hamiltonian, flux, and initial state to `165 M` using the native
adaptive solver and SciPy DOP853. This is an independent **integrator** check,
not an independent physics reference. At native `rtol=1e-13`, accepted orbital
phase states differ from the tighter DOP853 solution by at most `1.31e-11 rad`;
cubic Hermite samples built from those states and derivatives differ by at most
`3.93e-11 rad`. The current natural-cubic output samples differ by up to
`1.30e-6 rad`. The two DOP853 tolerance levels differ by up to `1.18e-10 rad`,
so the exact last digits of the smaller comparisons are not established.
The several-order gap nonetheless identifies natural-cubic **resampling** as
a major source of early waveform phase sensitivity. It does not determine
whether changing the published-convention sampler preserves LAL parity, nor
does it establish full-waveform convergence. All source hashes stayed fixed.
Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_orbital_phase --json-out
/tmp/seobnrv4-orbital-phase.json`.

The [310 M](orbital_phase_310.json) and [314 M](orbital_phase_314.json)
extensions use the same vector field and initial state, continuing almost to
the orbital-frequency peak. At all three native tolerances (`rtol=1e-13` to
`1e-15`), accepted orbital-phase states differ from DOP853 `rtol=1e-12` by at
most `1.22e-10 rad`; the two DOP853 settings differ by as much as
`7.47e-10 rad`. Hermite output samples differ by at most `5.48e-10 rad`,
while natural-cubic output samples reach `7.58e-8 rad` at `rtol=1e-14`.
This checks the same provisional ODE through `314 M`, not its physical
Hamiltonian or flux. Reproduce by adding `--t-end 310` or `--t-end 314`
to the preceding command.

An opt-in [Hermite resampling campaign](hermite_campaign.json) then replaced
only the accepted-state output interpolant in the development generator. It
retained the original `rtol=(1e-11,1e-12,1e-13)`, paired absolute tolerances,
default restart and initial-step rules, and all declared gates. It completed
all five solves and the pinned LAL diagnostic, with unchanged source hashes.

| Hermite comparison | Coarse to medium complex L2 | Medium to fine complex L2 | `1e-6` gate |
| --- | ---: | ---: | --- |
| ODE tolerances | 1.236e-5 | 6.833e-6 | **Failed** |
| Output grid `dt/M=0.5,0.25,0.125` | 2.940e-7 | 6.553e-6 | **Failed** |

The ODE error decreased but exceeded the budget; the grid error increased.
The moving coarse/fine restart is still present in the grid check. The raw
full-support LAL complex error was `1.987`, with no fitted alignment; that
metric is not a model-parity test. Thus derivative-aware resampling removes
the early interpolation discrepancy but does **not** qualify the full
waveform. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_hermite_campaign --json-out
/tmp/seobnrv4-hermite-campaign.json`; its nonzero exit is expected while the
gate fails.

The [Hermite error attribution](hermite_error.json) retained all three ODE
levels at `dt/M=0.5`, with the restart fixed by the default rule at `165 M`
and the reported peak at `314.7991867 M` in every run. For the refined
`rtol=1e-12` to `1e-13` pair, the coarse/fine sampled orbital phases differ
by at most `3.52e-9`/`4.64e-9 rad`, but the solved NQC phase coefficients
change by up to `4.68e-4`. At the attachment boundary, the *raw* fine-grid
waveform phases differ by `4.91e-9 rad`; applying a single shared coefficient
vector to both runs gives a `6.65e-9 rad` difference, while applying each
run's own coefficients gives `1.34e-5 rad`. This exact series-level
counterfactual identifies the NQC coefficient change as the dominant phase
amplifier for this row. Re-evaluating the same two-by-two matching system
separates its inputs: in the refined pair the phase-basis derivative matrix
changes by at most `1.53e-7`, and the waveform-phase derivative right-hand
side by at most `2.15e-8`. Holding the medium matrix fixed, the right-hand
side change contributes at most `4.23e-5` to the coefficients; changing the
matrix contributes up to `5.11e-4` and dominates the observed `4.68e-4`
coefficient change. This is an exact local decomposition for this pair, not
a universal error budget. It points to the sampled NQC basis derivatives as
the next numerical target; it does not establish whether the calibration
matches the historical SEOBNRv4 algorithm. Reproduce with `PYTHONPATH=src:.
python -m dev.native_rom.diagnose_hermite_error --json-out
/tmp/seobnrv4-hermite-error.json`.

The [local NQC derivative diagnostic](nqc_derivatives.json) differentiates
the **same** Hermite-sampled phase-basis series with degree-four and degree-six
polynomials over symmetric `0.25–2 M` windows around the match. It keeps the
phase right-hand side unchanged. Between `rtol=1e-12` and `1e-13`, the
natural-cubic phase matrix differs by `1.53e-7` and its fitted coefficients
by `4.68e-4`. The local-polynomial matrices differ by `1.96e-10–2.18e-9`,
and their fitted coefficients by `3.89e-5–4.50e-5`. This supports derivative
sensitivity as the source of the natural-cubic coefficient spread. It does
**not** establish a replacement: absolute polynomial coefficients depend
substantially on window and degree (for example, the second coefficient at
`rtol=1e-12`, degree four, moves from `-21.9769` at `0.25 M` to `-22.0914`
at `2 M`). The polynomial fit is a diagnostic on the provisional trajectory,
not an independent physical model or a qualification gate. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_nqc_derivatives --json-out
/tmp/seobnrv4-nqc-derivatives.json`.

The [DOP853 dense-trajectory check](nqc_dense_reference.json) independently
integrates the same provisional vector field from the same initial state to
the frozen `310.4268139 M` NQC match. Five-point derivatives of its NQC phase
basis at steps `0.025–0.2 M` are compared with the two Hermite-run matrices.
At `0.025 M`, changing DOP853 tolerance from `1e-11` to `1e-12` changes the
matrix by at most `1.32e-10`; the `0.025` and `0.05 M` derivative estimates
at the tighter tolerance agree within `2.24e-10`. The tight Hermite run's
natural-cubic matrix differs from DOP853 by `1.40e-7`, whereas its local
degree-six `0.25 M` polynomial matrix differs by `2.99e-9`. This identifies
the full-series natural-cubic **basis derivative** as a numerical outlier for
this row. The DOP853 check validates integration and differentiation of the
same provisional equations, not the equations' physical correctness or an
alternative historical NQC rule. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_nqc_dense_reference --json-out
/tmp/seobnrv4-nqc-dense-reference.json`.

The [whole-waveform diagnostic](whole_waveform.json) used the tightest
`dt/M=0.125`, `rtol=1e-13`, `atol=1e-14` settings and the pinned public
LALSuite `lal=7.7.1`, `lalsimulation=6.2.1` `SEOBNRv4` binary. The
[comparison inputs](whole_waveform_inputs.json) add one aligned-spin row
(`chi1=+0.2`, `chi2=-0.1`) to the nonspinning campaign row. Each complex `(2,2)`
mode is placed at its own measured amplitude peak (`t=0`, peak phase zero);
no shift, amplitude scale, or coefficients are optimized. Metrics use common
support; the JSON retains raw full-reference-support metrics too.

| 30+20 solar masses | Common peak-relative support | Complex relative L2 | Amplitude relative L2 | Phase RMS | Real overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chi1=chi2=0` | `[-310,150] M` | 0.02171 | 0.002219 | 0.02420 rad | 0.999764 |
| `chi1=+0.2`, `chi2=-0.1` | `[-330,150] M` | 0.01337 | 0.001665 | 0.01298 rad | 0.999911 |

Raw full-support complex errors were 1.987 and 1.864, respectively, before
the stated time/phase anchoring. The aligned diagnostics isolate shape
differences under that convention; they are not an optimized match, a
detector-weighted mismatch, a physical-fidelity test, or a parity claim.
Numerical convergence was **not** prechecked for either row. The second row
has no dedicated convergence campaign, so it is exploratory. These results
do not qualify the generator or ROM training.

Reproduce from the repository root with the recorded runtime and binary
versions (the first command is expected to exit 1 because its gate fails):

```sh
PYTHONPATH=src python -m dev.native_rom.qualify_generator \
  --atol-levels 1e-12 1e-13 1e-14 --rtol-levels 1e-11 1e-12 1e-13 \
  --compare-lal --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out /tmp/seobnrv4-qualification.json
PYTHONPATH=src python -m dev.native_rom.compare_whole_waveforms \
  --qualification /tmp/seobnrv4-qualification.json \
  --inputs dev/native_rom/evidence/2026-09-23/whole_waveform_inputs.json \
  --json-out /tmp/seobnrv4-whole-waveform.json
```

The [opt-in local-NQC campaign](hermite_local_nqc_campaign.json) carries out
that derivative experiment. A degree-six polynomial over ±`0.5 M` supplies
only the two NQC phase-basis derivatives; the waveform-phase RHS and all
other matching terms retain their original full-series splines. This window
was chosen **after** inspecting the one-row dense DOP853 comparison: its
tight-run phase-matrix difference was `1.92e-10` versus `2.99e-9` for the
previously reported ±`0.25 M` degree-six estimate. It is a post-hoc
numerical choice for diagnosis, not a published SEOBNRv4 prescription.
With Hermite trajectory sampling and the original tolerances, the ODE gate
passed: medium-to-fine complex relative L2 fell to `6.52e-7`, below `1e-6`.
The grid gate **failed** monotonicity: coarse-to-medium `2.46e-8`,
medium-to-fine `6.61e-7`. The default restart and initial-step rules were
unchanged, and no generator or ROM is qualified. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_hermite_campaign
--local-nqc --json-out /tmp/seobnrv4-hermite-local-nqc.json`.

The [matching whole-waveform diagnostic](local_nqc_whole_waveform.json) used
that failed campaign as its prerequisite, the same opt-in numerical choices,
and the pinned public LAL binary. With the unchanged peak-only alignment,
complex relative L2 was `0.02171096` on the nonspinning row and `0.01335888`
on the aligned-spin row. The aligned row still has no convergence campaign;
neither comparison is a parity or physical-validation claim. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.compare_whole_waveforms
--diagnostic-campaign /tmp/seobnrv4-hermite-local-nqc.json --inputs
dev/native_rom/evidence/2026-09-23/whole_waveform_inputs.json --json-out
/tmp/seobnrv4-local-nqc-whole.json`.

The [fixed-restart campaign](hermite_local_nqc_fixed_restart_campaign.json)
repeated the same five solves and pinned LAL diagnostic with the local NQC
derivative choice, but held the coarse/fine restart at `165 M` on every grid.
The original `dt/M=0.5,0.25,0.125` rule had restarted at `165,165,164.875 M`.
The initial-step rule, tolerances, error budgets, and gates were unchanged;
the source snapshot stayed fixed during the run. All five recorded restart
epochs are `165 M`.

| Fixed-restart comparison | Coarse to medium complex L2 | Medium to fine complex L2 | `1e-6` gate |
| --- | ---: | ---: | --- |
| ODE tolerances | 1.071e-6 | 6.523e-7 | **Passed** |
| Output grid `dt/M=0.5,0.25,0.125` | 2.463e-8 | 4.781e-9 | **Passed** |

This controlled comparison attributes the earlier grid nonmonotonicity in
this **one nonspinning row** to the grid-dependent restart epoch. It is a
post-hoc diagnostic choice, as is the local NQC derivative window; neither is
established as the historical SEOBNRv4 prescription. The default generator's
grid gate remains failed. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_hermite_campaign --local-nqc
--fixed-restart-time-M 165 --json-out /tmp/seobnrv4-fixed-restart.json`.

The [matching fixed-restart whole-waveform diagnostic](local_nqc_fixed_restart_whole_waveform.json)
uses the same pinned public LAL binary and peak-only alignment. Complex relative
L2 is `0.02171172` over `[-310,150] M` for the numerically checked nonspinning
row and `0.01336093` over `[-330,150] M` for the aligned-spin row. The
aligned-spin row's dedicated numerical campaign below **failed**. These
percent-scale shape errors are descriptive, not a parity pass or a
physical-fidelity test. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.compare_whole_waveforms
--diagnostic-campaign /tmp/seobnrv4-fixed-restart.json --inputs
dev/native_rom/evidence/2026-09-23/whole_waveform_inputs.json --json-out
/tmp/seobnrv4-fixed-restart-whole-waveform.json`.

The [dedicated aligned-spin campaign](aligned_spin_fixed_restart_campaign.json)
uses the [frozen row](aligned_spin_convergence_input.json), the same Hermite,
local NQC, and `165 M` fixed-restart choices, and unchanged gates. All five
solves completed with the same restart epoch. The ODE complex relative L2
changes from `3.576e-6` to `3.387e-6` with refinement, exceeding the `1e-6`
budget; the refined-pair amplitude error is `1.466e-6`. Grid complex error
increases from `1.946e-8` to `1.114e-7`, so its monotonicity gate also fails
despite remaining below the absolute budget. Event-time gates pass. This shows
that the nonspinning fixed-restart pass does not transfer to this spin row.
Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_hermite_campaign --inputs
dev/native_rom/evidence/2026-09-23/aligned_spin_convergence_input.json
--local-nqc --fixed-restart-time-M 165 --json-out
/tmp/seobnrv4-aligned-fixed-restart.json` (exit status 1 is the expected gate
failure).

The [paired aligned-spin whole-waveform diagnostic](aligned_spin_fixed_restart_whole_waveform.json)
records a `0.01336093` peak-anchored complex relative L2 difference over
`[-330,150] M`, matching the earlier two-row comparison, and explicitly marks
its convergence prerequisite as failed. Reproduce with `PYTHONPATH=src:. python
-m dev.native_rom.compare_whole_waveforms --diagnostic-campaign
/tmp/seobnrv4-aligned-fixed-restart.json --inputs
dev/native_rom/evidence/2026-09-23/aligned_spin_convergence_input.json
--json-out /tmp/seobnrv4-aligned-fixed-restart-whole-waveform.json`.

The [ODE](aligned_spin_trajectory_ode.json) and
[grid](aligned_spin_trajectory_grid.json) trajectory-sensitivity replays verify
the exact frozen waveform hashes before comparing stages. On the refined ODE
pair, pre-NQC `(2,2)` modes agree to `2.608e-9` complex relative L2 on the
common pre-attachment samples, whereas the attached waveform differs by
`3.387e-6`. The corresponding grid pair is `3.620e-10` before NQC versus
`1.114e-7` after attachment. Orbital-frequency maximum differences are
`8.576e-11` and `1.509e-11`, respectively; event times are identical within
the recorded precision. Even the coarse prefix, which uses the fine-run NQC
coefficients, differs by `7.209e-7` and `2.408e-8` locally. The amplitude
NQC coefficients change by up to `0.0149` across the ODE pair and `0.000483`
across the grid pair. This localizes the observed amplification to NQC
calibration or its application, before the final ringdown attachment; it does
not yet identify which matching-system input drives the coefficient drift.
The replay records amplitude matching condition numbers near `833` and phase
matching condition numbers near `8.52` on both pairs. Linear-system residuals
are at floating-point roundoff, so a failed solve is not the observed cause;
the sampled matching inputs and their derivative estimates remain to be
isolated. These condition numbers are diagnostics of the chosen normalization,
not physical validation criteria.

These are diagnostic, post-hoc replays of one aligned-spin row, not numerical
or physical validation. Reproduce each with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_trajectory_sensitivity --campaign
dev/native_rom/evidence/2026-09-23/aligned_spin_fixed_restart_campaign.json
--pair ode --json-out /tmp/seobnrv4-aligned-trajectory-ode.json` (substitute
`grid` for both occurrences of `ode` for the grid pair).

The [peak-resolution diagnostic](peak_anchor_diagnostic.json) found a
measurement bias in the earlier whole-waveform comparisons: each mode was
anchored to its highest **sample** on a `0.125 M` grid, although its actual
amplitude peak lies between samples. Re-running the *same* native waveforms
and pinned public binary with local sub-sample peak fits changes the complex
relative L2 as follows:

| Row | Sampled-peak anchor | Quadratic power peak | Quartic power peak |
| --- | ---: | ---: | ---: |
| Nonspinning | 0.021712 | 0.006949 | 0.006832 |
| `chi1=+0.2`, `chi2=-0.1` | 0.013361 | 0.008072 | 0.007817 |

Quadratic fits to amplitude and amplitude squared agree to about `1e-8` in
the whole-waveform metric. The fits locate each waveform's *own* peak; they
do not optimize a relative shift, phase, amplitude, or model coefficient.
The older campaign predates a CLI-only input-row edit to its diagnostic
runner; all generator source hashes and both replayed native waveform hashes
match the frozen comparison.
The prior sampled-anchor numbers remain correct for their stated convention,
but substantially overstate the shape difference due to peak quantization.
With quadratic-power anchoring, the residual regional complex error is
`0.00808`/`0.00931` in inspiral and `0.000587`/`0.000630` post-peak
(nonspinning/aligned). Its physical or implementation cause remains
unidentified. This re-anchoring does not change the failed aligned-spin
numerical gates or qualify either waveform. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_peak_anchor --campaign
dev/native_rom/evidence/2026-09-23/hermite_local_nqc_fixed_restart_campaign.json
--whole-waveform
dev/native_rom/evidence/2026-09-23/local_nqc_fixed_restart_whole_waveform.json
--json-out /tmp/seobnrv4-peak-anchor.json`.

The [inspiral residual profile](inspiral_residual_diagnostic.json) replays those
same two native waveforms and the pinned public binary, verifying their hashes
and the quadratic-peak metrics before partitioning the complex squared error.
For `t<-20 M`, phase accounts for 94.07% of the nonspinning error and 96.19%
of the aligned-spin error; the corresponding amplitude fractions are 5.93%
and 3.81%. Ten-`M` bins show a time-dependent phase residual: the nonspinning
mean moves from `-0.01442 rad` near `-225 M` to `+0.00761 rad` near `-75 M`;
the aligned-spin mean rises to `+0.01471 rad` near `-100 M` and falls toward
zero at the peak. Mode-frequency residuals change sign in both rows. This
localizes the measured shape difference to inspiral phase evolution but does
not identify whether initial conditions, Hamiltonian, flux, waveform phase,
or numerical treatment is responsible. It remains a two-row diagnostic, with
the aligned-spin numerical gate failed. Reproduce with `PYTHONPATH=src:. python
-m dev.native_rom.diagnose_inspiral_residual --campaign
dev/native_rom/evidence/2026-09-23/hermite_local_nqc_fixed_restart_campaign.json
--whole-waveform
dev/native_rom/evidence/2026-09-23/local_nqc_fixed_restart_whole_waveform.json
--peak-anchor dev/native_rom/evidence/2026-09-23/peak_anchor_diagnostic.json
--json-out /tmp/seobnrv4-inspiral-residual.json`.

A separate audit of the [LAL SEOBNRv4 driver](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_aligned_e_o_b_8c_source.html)
and [factorized-flux source](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_e_o_b_factorized_flux_8c_source.html)
narrows one hypothesis: the flux routine can multiply its `(2,2)` mode by
NQC, but the v4 driver initializes those coefficients to zero before orbital
evolution and calculates waveform NQC afterward. Omission of a *nonzero*
NQC factor in the native inspiral flux therefore does not explain this early
v4 phase residual. This is a source audit, not a copied implementation rule
or proof that the remaining 35-mode flux agrees. The
[LAL initial-condition source](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_e_o_b_initial_conditions_8c_source.html)
also uses a particular circular-root and radial-momentum calculation; the
paper-derived native initial state has not yet been checked against the
historical LAL state with a controlled waveform ablation.

The [initial-momentum ablation](initial_momentum_ablation.json) replays the
nonspinning frozen row and verifies both its native waveform hash and the
pinned public-binary comparison before changing one initial-state component.
Setting the solved `p_r*=-0.00781014` to zero while retaining `r` and `p_phi`
raises the quadratic-peak complex relative L2 from `0.006949` to `0.087420`.
Thus a missing initial radial momentum does not explain this row's residual;
the test does not validate the exact radial-momentum prescription or identify
the remaining phase cause. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m dev.native_rom.diagnose_initial_momentum
--whole-waveform dev/native_rom/evidence/2026-09-23/local_nqc_fixed_restart_whole_waveform.json
--peak-anchor dev/native_rom/evidence/2026-09-23/peak_anchor_diagnostic.json
--json-out /tmp/seobnrv4-initial-momentum-ablation.json` on RONI2 with the
frozen Python environment; the diagnostic rejects a different LAL binary.

The [independent leading-adiabatic initial-state check](initial_state_scalar_check.json)
solves the published circular conditions with the high-precision scalar
Hamiltonian and converts a **supplied** torque to inward `p_r*`. For the two
frozen rows, native and scalar `r`, `p_phi`, and `p_r*` differ by at most
`1.8e-14`, `1.3e-14`, and `3.1e-16`, respectively. This rules out a
transcription error in those native equations at these points. It does not
validate the native 35-mode torque or reproduce LAL's historical Cartesian
root and radial-momentum procedure. A finite `0.001` radial derivative step
changes the native Hamiltonian's local linear response by roughly `5.2e-7`
relative; that local calculation is not a measured LAL initial-state or
waveform difference. Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_initial_state --json-out
/tmp/seobnrv4-initial-state-scalar-check.json` in the frozen environment.

The [flux subset ablation](flux_subset_ablation.json) and single-mode omissions
([`(2,1)`](flux_ablation_omit-21.json), [`(3,2)`](flux_ablation_omit-32.json),
[`(5,5)`](flux_ablation_omit-55.json)) replay the same frozen nonspinning row
and pinned public binary. Each intervention changes radiation reaction in both
the initial-state solve and evolution; retained modes keep their original sum
order. At the native starting state, the `(2,2)`, `(3,3)`, and `(4,4)` modes
provide `99.96623%` of the **native model's** total flux. The omitted modes
individually contribute `(2,1): 0.01636%`, `(3,2): 0.01079%`, and
`(5,5): 0.00526%` at that state.

| Radiation-reaction flux | Retained initial flux | Quadratic-peak complex relative L2 |
| --- | ---: | ---: |
| All 35 modes (baseline) | 100% | 0.00694939 |
| Omit `(2,1)` | 99.98364% | 0.00649327 |
| Omit `(3,2)` | 99.98921% | 0.00643442 |
| Omit `(5,5)` | 99.99474% | 0.00673227 |
| Keep only `(2,2)`, `(3,3)`, `(4,4)` | 99.96623% | 0.00612824 |

Amplitude relative L2 stays near `0.0017` in these runs. Small initial mode
fractions can therefore accumulate into a measurable full-waveform difference.
The lower L2 values do not validate mode removal: compensating errors and
unresolved numerical or implementation differences could produce the same
pattern. The fractions describe the provisional native implementation at one
state, not independently verified mode fluxes or phase-error bounds. Reproduce
each result with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_flux_subset --whole-waveform
dev/native_rom/evidence/2026-09-23/local_nqc_fixed_restart_whole_waveform.json
--peak-anchor dev/native_rom/evidence/2026-09-23/peak_anchor_diagnostic.json
--selection SELECTION --json-out /tmp/seobnrv4-flux-ablation.json`, where
`SELECTION` is `three-mode`, `omit-21`, `omit-32`, or `omit-55`.

Reference specification 0.8.4 adds independent high-precision `(3,1)`,
`(6,4)`, and `(6,6)` pre-NQC moduli from [Pan et al.](https://arxiv.org/html/1006.0431),
[Damour--Iyer--Nagar](https://arxiv.org/html/0811.2069), and
[Taracchini et al.](https://arxiv.org/html/1202.0790). Their Newtonian and
mass-exchange limits pass; `(3,1)` retains its equal-mass spin limit. Four
frozen Torch points agree for each. This checks 11 of 35 individual flux
amplitudes conditionally; it does not establish historical hybrid choices or
a converged full flux.

Reference specification 0.8.5 adds the `(6,2)` and four even `ℓ=8` pre-NQC
moduli from the same primary papers. Their Newtonian and mass-exchange limits
pass, and four frozen Torch points per mode agree within floating precision.
The `ℓ=8` comparison explicitly covers both the tabulated and generic-leading
spin selections; the source papers do not identify which belongs to historical
SEOBNRv4. Conditional scalar coverage is now 16 of 35 amplitudes. Nineteen
remain without scalar references, and this does not close the full-flux,
numerical-convergence, or whole-waveform qualification gates.

Reference specification 0.8.6 adds J-source scalar checks for `(5,4)`,
`(5,2)`, `(7,6)`, `(7,4)`, and `(7,2)` from the same pinned primary papers.
Their Newtonian and mass-exchange limits and four frozen Torch rows per mode
agree within floating precision. The `(5,4)` and `(5,2)` finite-mass spin
completion is explicitly hybrid; the three `ℓ=7` checks use only the printed
1PN truncation. Conditional scalar coverage is 21 of 35 amplitudes; fourteen
remain, in addition to unresolved historical assembly and both-backend gates.

Reference specification 0.8.7 covers the remaining fourteen declared
pre-NQC flux moduli: `(4,1)`, `(5,3)`, `(5,1)`, the three odd-`m` `ℓ=6`
modes, the four odd-`m` `ℓ=7` modes, and the four odd-`m` `ℓ=8` modes.
Independent paper-derived scalar values agree with frozen Torch rows and
Newtonian, body-exchange, and equal-mass limits. The `(4,1)` finite
equal-mass unequal-spin contribution is retained. This completes
**conditional scalar coverage of 35/35 modes** for the declared hybrid
reconstruction. It does not identify historical v4 spin or high-`ℓ`
truncations, establish the JAX torque, or close convergence and
whole-waveform gates.

Reference specification 0.8.8 adds an independent high-precision Eq. (7)
NQC solve at a fixed matching time. Analytic polynomial trajectories verify
all five corrected peak conditions and recover the same coefficients through
the Torch and JAX sampled frontends. This establishes the matching algebra
under exact-sampling conditions. It does not establish the historical peak
selection or a convergent derivative rule for numerical trajectories; the
strong-spin convergence failure below remains a release blocker.

The exact-hash [ODE](aligned_spin_nqc_matching_inputs_ode.json) and
[grid](aligned_spin_nqc_matching_inputs_grid.json) NQC matching-input replays
reconstruct the frozen amplitude and phase matrices and right-hand sides from
the replayed fine trajectory. Their solved coefficients agree with the stored
calibration to at most `3.0e-12`; the exact matrix/RHS decomposition closes
each observed coefficient difference to at most `4.1e-12`. The unscaled
amplitude and phase matrix condition numbers are about `1.06e4` and `97.4`;
the generator reports smaller, row/column-equilibrated conditions (`833` and
`8.52`).

| Refined pair | Amplitude matrix / total coefficient-difference norm | Phase RHS / total coefficient-difference norm | Dominant changed input |
| --- | ---: | ---: | --- |
| ODE | `0.01705 / 0.01982` (`86%`) | `0.000296 / 0.000278` (`107%`) | Amplitude-basis second derivative; waveform-phase second derivative |
| Grid | `0.000552 / 0.000641` (`86%`) | `9.30e-6 / 9.38e-6` (`99%`) | Same two second-derivative inputs |

The `107%` reflects partial cancellation by a smaller phase-matrix change;
these vector contributions are algebraic, not independently measured error
budgets. The amplitude target itself is identical across runs. These results
identify numerical sensitivity in the sampled second derivatives, but do not
select a validated derivative rule or make the aligned-spin convergence gates
pass. Reproduce either replay with `PYTHONPATH=src:.
/Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_nqc_matching_inputs --campaign
dev/native_rom/evidence/2026-09-23/aligned_spin_fixed_restart_campaign.json
--pair ode --json-out /tmp/seobnrv4-nqc-matching-inputs.json`, replacing `ode`
with `grid` for the grid pair.

The [two-row dense-output check](nqc_second_derivatives_two_rows.json) replays
the frozen nonspinning and aligned-spin waveforms with exact array hashes, then
differentiates the same provisional equations on independent DOP853 dense
trajectories. At the match, the natural-spline errors in the four amplitude
second derivatives are `7.1e-9–1.0e-7`; fixed ±`0.5 M` degree-six local-fit
errors are `1.9e-11–1.8e-10`. Waveform-phase second-derivative errors change
from `1.7e-8–4.0e-8` to `3.5e-11–2.0e-10`, with the aligned-spin local error
comparable to the DOP853 tolerance spread. This is a numerical comparison
under shared provisional physics, not a validation of a published NQC rule.

The [candidate rule](nqc_second_derivative_candidate.md) was committed before
the five-level runs. It replaces only second derivatives of `A`, `A q_1`,
`A q_2`, `A q_3`, and unwrapped waveform phase at the NQC match. It keeps the
existing local phase-basis matrix, the first derivatives, and all other
choices. The [campaign](nqc_second_derivative_campaign.json) uses the original
`1e-6` waveform budgets, refinement monotonicity, and `0.05 M` event gates:

| Row | Refined ODE complex L2 | Refined grid complex L2 | Numerical gates |
| --- | ---: | ---: | --- |
| Nonspinning | `6.47e-8` | `4.72e-11` | Both pass |
| Aligned spin | `1.39e-7` | `4.89e-10` | ODE passes; grid monotonicity fails (`5.69e-11` to `4.89e-10`) |

The aligned-spin grid values are well below the error budget, but the
predeclared monotonicity gate still fails. No post-run tuning or gate relaxation
was made. The [hash-verified whole-waveform replay](nqc_second_derivative_waveforms.json)
uses each waveform's independently fitted amplitude peak, without optimizing
a relative shift, phase, scale, or coefficients. Its quadratic-peak complex
relative L2 is `0.00694919` nonspinning and `0.00807026` aligned spin, versus
the frozen baseline's `0.00694939` and `0.00807196`. The remaining difference
is essentially unchanged. This opt-in experiment neither qualifies the
generator nor licenses ROM training or release promotion.

The [aligned-spin grid attribution](nqc_grid_floor.json) replays the three
frozen grid waveforms with exact array hashes and reconstructs their NQC
matching systems. In the refined grid pair, the raw pre-NQC inspiral already
differs by `3.62e-10` complex relative L2, versus `4.89e-10` for the finished
waveform. Holding the finer run's NQC coefficients common gives `4.29e-10`
before attachment, while changing coefficients at the same coarse trajectory
gives `7.02e-10`; these are interacting counterfactuals, not additive error
shares. The reconstructed amplitude and phase coefficient changes have L2
norms `3.55e-6` and `6.44e-8`, respectively. Both trajectory and matching
vary with the grid, so the local second-derivative fit alone does not explain
the failed monotonicity. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_nqc_grid_floor --campaign
dev/native_rom/evidence/2026-09-23/nqc_second_derivative_campaign.json
--json-out /tmp/seobnrv4-nqc-grid-floor.json`.

The [fixed-first-step control](nqc_grid_initial_step.json) keeps the same
candidate, tolerances, `165 M` restart, and three output grids, while setting
the coarse ODE's first trial step to `0.5 M` for every grid. Its baseline has
the exact frozen waveform hash. All three waveforms then agree **exactly on
nested samples** (`0` complex relative L2), and their coarse and fine accepted
step counts are identically `596` and `424`. Thus the frozen aligned-spin grid
failure is caused by coupling the initial ODE step to `dt_geom` in this row.
This control makes the grids sample the same trajectory; zero error is not a
convergence-order measurement, validation of the equations, or a reason to
retroactively mark the predeclared grid gate as passed. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_nqc_grid_initial_step
--campaign dev/native_rom/evidence/2026-09-23/nqc_second_derivative_campaign.json
--json-out /tmp/seobnrv4-nqc-grid-initial-step.json`.

The [fixed-trajectory output-sampling check](output_sampling_aligned_spin.json)
holds the first trial step and provisional vector field fixed through `165 M`
for the aligned-spin row. All three output grids (`0.5`, `0.25`, `0.125 M`)
have identical SHA-256 hashes for accepted times, states, and derivatives;
their nested Hermite samples agree exactly. Against an independent DOP853
integration of the *same* equations, the maximum Hermite-state differences
are `3.32e-11` in radius and `3.77e-11 rad` in orbital phase, including a
fixed off-grid probe. The corresponding natural-cubic differences reach
`6.86e-7` and `1.26e-6 rad`. Two DOP853 tolerance settings differ by as much
as `1.90e-10` in radius, so the Hermite comparison is not an absolute error
bound at its last digits. It supports the output-sampling diagnosis without
establishing full-waveform convergence or physical model fidelity. Reproduce
with `PYTHONPATH=src:. python -m dev.native_rom.diagnose_output_sampling
--json-out /tmp/seobnrv4-output-sampling.json`.

The [parameter-boundary plan](parameter_boundary_plan.json) was committed
before its [six-row campaign](parameter_boundary_campaign.json). These are
exploratory one-factor stress points, **not** asserted calibration-domain
boundaries. All 30 scalar solves (five per row) completed without exceptions;
the source hashes stayed fixed. The table gives the finer comparison's direct,
unaligned, whole-waveform complex relative L2. The same `1e-6` complex and
amplitude budgets, decreasing-error rule, and `0.05 M` event gates apply to
each row. Both start-frequency rows pass; four other rows fail at least one
declared gate.

| Changed parameter | Refined ODE L2 | Refined grid L2 | Failed gate |
| --- | ---: | ---: | --- |
| Equal masses, unequal spins (`25+25`, `+0.2,-0.2`) | `5.80e-8` | `3.71e-10` | Grid error increased from `3.46e-11` |
| Mass ratio four, nonspinning (`40+10`) | `3.11e-7` | `6.44e-11` | ODE error increased from `1.53e-7`; grid error from `2.99e-11` |
| Strong aligned spin (`+0.8,+0.8`) | `6.57e-6` | `1.17e-9` | ODE complex-error budget |
| Strong antialigned spin (`-0.8,-0.8`) | `5.85e-7` | `2.39e-10` | Grid error increased from `1.98e-10` |
| Lower start frequency (`M Omega=0.03`) | `5.87e-8` | `3.75e-10` | None |
| Higher start frequency (`M Omega=0.05`) | `5.76e-8` | `8.14e-11` | None |

The campaign preserves the earlier aligned-spin failure; passing a different
row cannot qualify the generator. It also does not compare these boundary
waveforms with an independent physical reference. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_parameter_boundaries
--json-out /tmp/seobnrv4-parameter-boundaries.json`; exit status 1 is expected
while the frozen aggregate gate fails.

The [strong-spin ODE attribution plan](boundary_strong_spin_ode_plan.json) was
committed before its [two-run replay](boundary_strong_spin_ode.json). Both
waveform hashes, the frozen whole-waveform comparison, stage statistics, and
source manifest match the boundary campaign. In the `rtol=1e-12` to `1e-13`
pair, the unaligned complex relative L2 is `6.566e-6`, while amplitude is
`1.682e-8`; an exact squared-error decomposition attributes `99.99934%` to
phase. The common pre-attachment **raw** mode differs by only `3.255e-10`
complex relative L2. At the attachment boundary, raw phase differs by
`-2.90e-10 rad`, whereas the change in NQC phase coefficients contributes
`1.229e-5 rad` to the corrected phase difference. The squared complex error
is spread across the waveform: `1.6%` before the `165 M` restart, `30.1%`
from restart to boundary minus `30 M`, `36.0%` over the last `30 M` before
the boundary, `19.7%` from boundary to peak, and `12.6%` in ringdown. Thus
NQC matching amplifies a much smaller trajectory/raw-mode difference in this
row. This does not validate the provisional equations. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_boundary_strong_spin_ode
--json-out /tmp/seobnrv4-boundary-strong-spin-ode.json`.

The [strong-spin NQC matching-input plan](boundary_strong_spin_nqc_inputs_plan.json)
was committed before its [two-run replay and dense check](boundary_strong_spin_nqc_inputs.json).
Both waveform hashes and stage statistics still match the frozen campaign.
Reconstructed phase systems reproduce their stored coefficients within
`4.6e-13`; the exact matrix/RHS decomposition closes within `1.8e-13`.
From medium to fine ODE tolerance, the phase coefficients change by
`[-1.09346e-4, +6.09512e-3]`. The change in the matching matrix accounts
for `97.6%` of the coefficient-change norm, and its second-derivative row
alone contributes `+5.90457e-3` to the second phase coefficient. The RHS
accounts for `2.4%` of the norm. The matrix's first phase-basis second
derivative changes by `+2.57147e-9`; compared with dense DOP853 of the
same provisional vector field, its medium/fine errors are `-2.925e-9` and
`-3.534e-10`. The dense tolerance and stencil spreads for that derivative
are `2.12e-12` and `4.59e-13`, respectively. This identifies the dominant
numerical matching input, while the remaining fine error and the underlying
physics still need independent validation. No coefficients or gates were
retuned. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_boundary_strong_spin_nqc_inputs --json-out
/tmp/seobnrv4-boundary-strong-spin-nqc-inputs.json`.

The [phase-basis error-trace plan](boundary_strong_spin_basis_trace_plan.json)
was committed before its [frozen replay and numerical partition](boundary_strong_spin_basis_trace.json).
The plan was amended before the final replay to cover the entire local fit
window and its bracketing accepted knots with dense DOP853; the first attempt
had extrapolated the far-side dense output and is not evidence. Both final
replays match their frozen waveform hashes, and cubic Hermite reconstruction
of the native output from accepted states agrees within `3.6e-15`. For
`p1=pr*/(r Ω)`, the medium-to-fine change in `p1''` is `+2.57147e-9`.
Evaluating the same cubic Hermite map on dense states at the native accepted
times attributes `+2.57169e-9` to output interpolation; accepted-state
evolution, the degree-six derivative fit, and the restart state together
contribute about `-2.14e-13`. The medium and fine local windows use six and
ten accepted knots, respectively. At the looser dense tolerance, the
interpolation contribution is `+2.57274e-9`, so its dominance is stable;
the tiny restart-coordinate contributions are below the dense numerical
spread and should not be interpreted individually. This traces the frozen
NQC sensitivity to resampling the fine trajectory with cubic Hermite, not
to a physics correction. The qualification failure is unchanged. Reproduce
with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_boundary_strong_spin_basis_trace --json-out
/tmp/seobnrv4-boundary-strong-spin-basis-trace.json`.

The [boundary first-step plan](boundary_grid_first_step_plan.json) was also
committed before its [nine-run control](boundary_grid_first_step.json). It
replays the three rows whose default grid errors increased, holding only the
coarse ODE's first trial step at `0.5 M` across `dt/M=0.5,0.25,0.125`. Each
`dt=0.5 M` baseline exactly matches the frozen waveform hash. Within each
row, all three coarse and fine accepted histories have identical time, state,
and derivative hashes, and the complete waveforms agree exactly at nested
samples: both adjacent-grid complex and amplitude relative L2 values are
`0`. This isolates the default grid variation in these rows to coupling the
first trial step to the output spacing. It does not demonstrate a convergence
order; in particular, mass-ratio-four's separate ODE monotonicity failure
and strong aligned spin's ODE error-budget failure remain. The original grid
gates remain failed. Reproduce with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_boundary_grid_first_step --json-out
/tmp/seobnrv4-boundary-grid-first-step.json`.

The [fine accepted-step cap plan](boundary_fine_step_cap_plan.json) was
committed before its [four-row control](boundary_fine_step_cap.json). It
replays an uncapped strong aligned-spin run exactly, then caps only the fine
stage's accepted steps at `0.125 M` for the three frozen ODE tolerance levels
in each of the four failing boundary rows. All 12 capped solves complete,
their accepted steps obey the cap, the recorded sources stay unchanged, and
all medium-to-fine event-time differences are zero. In strong aligned spin,
the medium-to-fine complex relative L2 falls from `6.56623e-6` to
`3.30022e-10`, passing its ODE error budget; the phase-basis `p1''` change
falls from `+2.57147e-9` to `-2.30e-13`. This supports the cubic Hermite
output interpolation diagnosis for that row. In mass ratio four, the complex
error falls from `3.10513e-7` to `4.76183e-8`, but its monotonicity gate
still fails: the capped coarse-to-medium error is `1.60292e-9`. The
equal-mass and antialigned-spin ODE gates pass, as they already did before
this control. Their separate frozen output-grid failures were not re-run
here. These are numerical controls on the same provisional equations, not a
production interpolation choice or independent SEOBNRv4 validation.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_boundary_fine_step_cap --json-out
/tmp/seobnrv4-boundary-fine-step-cap.json`.

The [mass-ratio-four attribution plan](boundary_q4_capped_attribution_plan.json)
preceded its [exact capped replay](boundary_q4_capped_attribution.json). All
three waveform hashes, accepted trajectories, and phase matching systems
reproduce the cap control exactly. From medium to fine, the pre-attachment raw
mode changes by only `5.78270e-10` relative L2, but the full waveform changes
by `4.76183e-8`, with `99.93%` of squared error in phase. The NQC phase-basis
second-derivative matching row changes by `-5.78227e-10` in its first column;
its exact matrix-row contribution to the phase-coefficient change is about
`[+3.97e-7, -3.91e-6]`. The corresponding raw boundary phase difference is
`-1.39e-9` rad, while the NQC coefficient-change contribution is
`-9.83e-8` rad. The coarse-to-medium pair has a larger raw-mode difference
(`2.60130e-9`) yet a much smaller full-waveform difference (`1.60292e-9`).
This localizes the nonmonotonicity to the NQC phase matching input, not to
an event-time shift. This first replay did not distinguish accepted-state error,
Hermite output interpolation, and local derivative-fit error in that input.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_boundary_q4_capped_attribution --json-out
/tmp/seobnrv4-boundary-q4-capped-attribution.json`.

The subsequent [phase-basis trace plan](boundary_q4_capped_basis_trace_plan.json)
preceded its [dense numerical control](boundary_q4_capped_basis_trace.json).
It again replays both capped waveform hashes and reconstructs the exact
native local-fit derivative. The first-column medium-to-fine second-derivative
change is `-5.78227e-10`; the exact between-run partition attributes
`-5.75542e-10` (about `99.5%`) to cubic Hermite interpolation on the two
accepted-step meshes. Fine accepted-state evolution contributes
`-1.72e-13`, local derivative fitting `+1.24e-12`, and the restart state
`-3.75e-12`. The partition closes exactly at recorded precision; the tight
and loose DOP853 continuous references differ by `1.81e-11` in that column.
This explains why a `0.125 M` cap reduces the waveform error but still leaves
a nonmonotone NQC phase response. The DOP853 control uses the same provisional
vector field and therefore does not validate SEOBNRv4 physics. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_boundary_q4_capped_basis_trace --json-out
/tmp/seobnrv4-boundary-q4-capped-basis-trace.json`.

The [Fehlberg output plan](rkf45_output_boundary_plan.json) froze a
development-only replacement for cubic Hermite output sampling: one fifth-order
Fehlberg substep from the preceding accepted state, with the `0.125 M` fine
accepted-step cap. Its [mass-ratio-four pilot](rkf45_output_q4_pilot.json)
reduced medium-to-fine complex waveform error from the capped Hermite
`4.76183e-8` to `4.00764e-11` and passed all ODE gates, but still failed the
output-grid monotonicity gate. The [fixed-first-step plan](rkf45_fixed_start_boundary_plan.json)
then froze the same sampler with a common `0.5 M` coarse initial trial step.
The [combined six-row report](rkf45_fixed_start_boundary_combined.json)
verifies one plan, unchanged source manifest, five levels per row, and all
frozen numerical gates. Refined ODE complex errors range from `1.01036e-11`
to `4.00764e-11`. Every nested-grid waveform difference is exactly zero
because those grids reuse identical accepted trajectories and sample times;
this is an identity control, not independent accuracy or convergence evidence.
The deliberately stopped serial component and three completed per-row
components are retained in the combined report.

The predeclared [off-grid reference plan](rkf45_offgrid_reference_plan.json)
tests the mass-ratio-four and strong aligned-spin tight-tolerance runs at
fixed probes around the NQC match. The [report](rkf45_offgrid_reference.json)
replays both frozen waveform hashes and starts each DOP853 reference from the
preceding accepted fine-stage state, isolating local output-sampling error.
Fehlberg's maximum absolute state difference is `3.55e-15` in each row,
versus Hermite's `2.62e-11` and `9.00e-12`, respectively. The first NQC
phase-basis second-derivative difference from the reference is `-1.06e-14`
and `+3.97e-15` for Fehlberg, versus `+2.82e-10` and `-5.27e-11` for
Hermite. Two tight DOP853 tolerances agree at recorded floating precision
on these short local solves. These results support the interpolation-error
diagnosis for these probes only. The candidate remains development-only and
does not establish physical SEOBNRv4 fidelity or ROM training readiness.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_rkf45_offgrid_reference --json-out
/tmp/seobnrv4-rkf45-offgrid-reference.json`.

The [six-row off-grid plan](rkf45_domain_offgrid_plan.json) extends that local
sampler check to every row in the frozen numerical campaign. It selects fine
accepted-step brackets at 10%, 50%, and 90% of each trajectory and the NQC
match bracket, with duplicate brackets evaluated once. Three interior points
per bracket are compared with two tight DOP853 solves seeded from the preceding
accepted state. The [completed report](rkf45_domain_offgrid.json) replayed all
six frozen waveform hashes and verified that generator source hashes stayed
unchanged during the run. Across the finite probes, the largest absolute
Fehlberg state difference from the tighter reference is `3.55e-15`; the
largest Hermite difference is `5.95e-11`, and the largest difference between
the two reference tolerances is `3.55e-15`. An [initial stopped
attempt](rkf45_domain_offgrid_attempt1_failed.json) found that the high-start
row's match bracket duplicates a selected fractional bracket; the duplicate
rule was frozen before the complete rerun. These are local output-sampling
checks of the same provisional ODE, not independent trajectory-convergence,
historical-physics, waveform-parity, or domain-wide accuracy evidence.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_rkf45_domain_offgrid --json-out
/tmp/seobnrv4-rkf45-domain-offgrid.json`.

The [full-trajectory plan](rkf45_full_trajectory_plan.json) extends the
cross-integrator check from local interpolation probes to every accepted
fine-stage knot in the six frozen `ode_2` rows. Each DOP853 run starts once
from that row's first accepted fine state; it is not reseeded at later knots.
The [completed report](rkf45_full_trajectory.json) replays all six waveform
hashes, verifies unchanged generator sources, and passes both predeclared
diagnostic ceilings (`1e-6` native state error and `1e-7` reference spread).
There are 205–4,902 compared knots per row. The largest absolute native-state
difference from the tighter reference is `1.61e-11`; the largest spread
between the two reference tolerances is `2.47e-10`. The latter exceeds the
native difference, so these data do not resolve error at the last reported
digits. This checks whole fine-stage trajectories of the same provisional
ODE from frozen restart states. It does not establish historical equations,
initial-state accuracy, full-waveform parity, or ROM release qualification.
Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_rkf45_full_trajectory --json-out
/tmp/seobnrv4-rkf45-full-trajectory.json`.

At this earlier stage, the next checks were the historical initial-state
prescription and remaining Hamiltonian/flux components, without fitting
correction coefficients to the LAL binary. The later component gate below
records subsequent flux and remnant checks. The current component status is
tracked in the [reference specification](../../reference/SPEC.md).

A separate [read-only audit of pinned lalsimulation-v6.2.1
source](historical_v4_component_audit.json) identifies the active ordinary-v4
finite-mass-ratio tortoise quartic as the 3PN-truncated choice used by our
paper-derived scalar reference. It finds that the active `(2,1)` flux path
retains comparable-mass spin terms through `v**3`, while the complex-waveform
path retains only the leading `v` term. Both keep the published Eq. (29b)
spin terms from `v**4` to `v**7` in resummed `rho21`, while omitting its
lower-order spin terms, unlike our conditional Taylor-residual reconstruction.
The audit records behavior only; it contains no upstream source excerpts or
copied coefficients. No compiled component comparison was performed, so the
Hamiltonian and `(2,1)` findings do not establish numerical parity.

The [frozen `(2,1)` rho-hybrid diagnostic](mode_21_rho_hybrid_diagnostic.json)
uses a separate scalar candidate derived from Pan Eq. (29b)'s higher-spin
`rho21` terms, with the flux and complex-mode auxiliary truncations from the
audit. At shared circular surrogate inputs for mass ratio 4:1 and spins
`(+0.7, -0.3)`, the flux-path rho hybrid exceeds our Taylor-`f` amplitude by
0.20%, 0.67%, and 2.36% at `M*Omega` 0.01, 0.02, and 0.04. Zero-spin and
equal-mass unequal-spin controls pass. This isolates a finite-frequency
prescription difference; it does not establish which candidate matches the
compiled historical component. Reproduce with
`PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python -m
dev.native_rom.diagnose_mode_21_rho_hybrid --plan
dev/native_rom/evidence/2026-09-23/mode_21_rho_hybrid_plan.json --json-out
/tmp/seobnrv4-mode-21-rho-hybrid.json`.

The [frozen whole-waveform sensitivity plan](mode_21_whole_waveform_plan.json)
replays the aligned-spin row above with the original Taylor-`f` `(2,1)` flux
and the paper-derived higher-spin `rho21` hybrid. The baseline native waveform
hash and sampled-peak metric exactly reproduce the prior comparison; both runs
use the same pinned binary, fixed-restart numerical choices, and independent
quadratic-power peak fits. The [result](mode_21_whole_waveform_diagnostic.json)
changes fitted complex relative L2 from `0.0080719610` to `0.0080611104`.
The fitted inspiral-window value changes from `0.00931421` to `0.00930171`;
post-peak is essentially unchanged. This supports the `(2,1)` prescription
as a small contributor to this row's residual, while leaving the historical
compiled component and overall waveform parity unverified. Reproduce with
`PYTHONPATH=src python -m
dev.native_rom.diagnose_mode_21_whole_waveform --plan
dev/native_rom/evidence/2026-09-23/mode_21_whole_waveform_plan.json --json-out
/tmp/seobnrv4-mode-21-whole-waveform.json`.

## Later component gate: remnant and QNM (reference 0.8.3)

The independent scalar [reference](../../reference/SPEC.md) now also checks
the published hybrid pre-NQC `(4,2)`, `(4,3)`, and `(5,5)` amplitudes. Their Newtonian
limits and body-exchange checks pass, and four frozen Torch comparisons per
mode pass. Twenty-seven mode amplitudes still lack scalar references, while
the complete historical `(2,1)` spin choice remains unresolved.

The explicit [public-binary component probe](../../compare_remnant_qnm_lal.py)
was run on the local host with `lal=7.7.1` and `lalsimulation=6.2.1`. Its
[JSON result](remnant_qnm_binary.json) records binary hashes, inputs, values,
thresholds, and gates. The `SEOBNRv4` remnant call selects Hofmann's `(3,4)`
variant at four discriminating rows; the largest absolute remnant-mass and
selected-spin differences from the independent scalar fit are `1.2e-16` and
`2.3e-16`. Both native QNM frontends differ from the public-binary QNM call
by at most `4.8e-7` in dimensionless real frequency or damping across nine
signed-spin points. This is a bounded component comparison, not a
whole-waveform or domain-wide qualification.

The reference suite also checks three sparse `n1l2m2.dat` values from the
[authors' l=2 archive](https://blackholes.tecnico.ulisboa.pt/Webpagecodes/l2.tar.gz)
to `1e-9` absolute for each native frontend. The archive is not packaged;
its download SHA-256 and the copied validation values are recorded in the
[reference specification](../../reference/SPEC.md).
A second independent author check uses [Cook's high-accuracy Kerr QNM
dataset](https://zenodo.org/records/2650358), file `KerrQNM_00.h5`, MD5
`34c04f3b03bbb99935d1f91f7296bd02`, SHA-256
`d5853acabc1a07b66b4fae437c02e1435a9d2b3561bd7e4752b9721c9cd48923`.
Dataset `/n00/m+02/{2,2,0}` has `(j,Re Mω,Im Mω)` in its first three
columns. At `j=0.999`, both native frontends agree within `3e-16`; at
`j=0.99990625`, the raw Torch spline differs by `2.7e-13` in frequency and
`6.8e-13` in damping. The older Berti archive's `j=0.9999` row differs
from our independently solved value by `1.0152e-7` and `3.3590e-7`,
respectively. A fresh strict Kerr root reproduces the native value, so
denser spline interpolation would not resolve this discrepancy. These samples
are validation-only; neither author archive is bundled or consumed by the
generator. The signed-spin frontends clamp at `|j|=0.9999`, and the pinned
public LAL QNM call rejects `j>0.9996`.
Reproduce the public-binary probe from the repository root with:

```sh
PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python \
  -m dev.native_rom.compare_remnant_qnm_lal \
  --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out /tmp/seobnrv4-remnant-qnm-binary.json
```

## Fixed-step peak-event regression

The Torch and JAX fixed-step integrators now default to the seven-sample
local-polynomial orbital-frequency peak locator. For an analytic Gaussian
frequency with peak at `1.37 M`, Torch's former natural-cubic default located
the peak `0.00104074 M` late at `dt=0.05 M` and gave a center derivative of
`0.933764` instead of `1`. The local-polynomial locator has a
`5.41e-12 M` time error and `3.47e-10` derivative error on that same case.
JAX's batched local-polynomial solve and prefix-mask checks now have direct
analytic tests. Explicit continuous-spline matching remains available. The
nearby Torch/JAX event, attachment, published-driver, and release-contract
tests pass with the worktree source on `PYTHONPATH`. This checks event
arithmetic on synthetic trajectories; it does not establish convergence or
historical SEOBNRv4 matching on physical trajectories. The earlier frozen
campaigns above retain their own source hashes and results.
