# Frozen tabulated SEOBNRv4 numerical gate diagnosis

Scope: scalar CPU float64 `(2,2)` generator on the two frozen rows only. This
record does not qualify physics, a ROM, or a whole waveform. No gate or
production generator setting was changed.

## Reproduction

The unmodified source at `aa1f365fb6f284ac788a3f7378e5c3bc11933140`
was archived to host `len`, cwd
`/home/xangma/tmp/diffgw-seobnr-ode-gate-aa1f365f`. All jobs used
`PYTHONPATH=src:.`, one OpenMP/OpenBLAS thread, Torch CPU float64, and
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`. The source manifest
SHA-256 is `ba2c668401077505afd7b095eff1b81e2f13d03aba5bcf4a6e16df52383b8427`.
The LAL binary versions were pinned only for the existing black-box diagnostic;
no other implementation's source was used in this slice.

Commands for the exact frozen reruns, from that cwd:

```sh
python -m dev.native_rom.qualify_generator --spin-truncation tabulated \
  --atol-levels 1e-12 1e-13 1e-14 --rtol-levels 1e-11 1e-12 1e-13 \
  --compare-lal --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out rerun-nonspin.json
python -m dev.native_rom.qualify_generator --inputs spin_row.json \
  --spin-truncation tabulated --atol-levels 1e-12 1e-13 1e-14 \
  --rtol-levels 1e-11 1e-12 1e-13 --compare-lal \
  --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out rerun-spin.json
```

Job ledger on `len` (the cwd above applies to every entry):

| Run | Command suffix after the common Python executable | PID | Log in cwd |
| --- | --- | ---: | --- |
| Frozen nonspin | `-m dev.native_rom.qualify_generator` with first command above | 2156350 | `rerun-nonspin.log` |
| Frozen spin | `-m dev.native_rom.qualify_generator` with second command above | 2156351 | `rerun-spin.log` |
| Grid restart | `grid_probe.py` | 2249195 | `grid-restart-probe.log` |
| ODE decomposition, nonspin | `-m dev.native_rom.diagnose_current_trajectory --campaign dev/native_rom/evidence/2026-09-29/qualification_tabulated_nonspin.json --pair ode --json-out ode-decomposition-nonspin.json` | 2285843 | `ode-decomposition-nonspin.log` |
| ODE decomposition, spin | same with `qualification_tabulated_spin.json` and `ode-decomposition-spin.json` | 2285845 | `ode-decomposition-spin.log` |
| Spin Hermite | `ode_control_probe.py --row spin --control hermite` | 2423158 | `ode-control-spin-hermite.log` |
| Spin second derivatives | `ode_control_probe.py --row spin --control second` | 2423160 | `ode-control-spin-second.log` |
| Spin phase matrix | `ode_control_probe_phase.py --row spin --control phase` | 2624694 | `ode-control-spin-phase.log` |
| Grid initial step | `grid_initial_probe.py` | 2603990 | `grid-initial-step-probe.log` |

The expected next check for each job was its per-run log and JSON output. All
listed PIDs have exited after writing their results; while running, the stop
command was `kill <PID>` on `len`.

`spin_row.json` is a one-element array containing
`{"mass1":30,"mass2":20,"chi1":0.2,"chi2":-0.1,"omega_start":0.04}`.
The nonspin row changes both spins to zero. Each rerun reproduced all five
waveform hashes, the source manifest, and the complete frozen check object in
the corresponding [prior nonspin](../qualification_tabulated_nonspin.json) and
[prior spin](../qualification_tabulated_spin.json) receipts.

The immutable budget is complex and amplitude relative L2 `<=1e-6`, refined
error no larger than the preceding level, and peak/match/boundary shifts
`<=0.05 M`. The metric compares exact nested, unaligned complex samples and
zero-fills only unequal tails.

| Row | ODE complex coarse→medium | ODE complex medium→fine | ODE amplitude medium→fine | Grid complex coarse→medium | Grid complex medium→fine | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Nonspin | `2.135988e-5` | `9.0792058666e-6` | `3.47176949e-7` | `1.33339639e-7` | `6.14985548e-6` | ODE and grid fail |
| Spin | `6.314091e-5` | `4.9384488011e-5` | `1.47015986e-6` | `5.76186573e-7` | `5.50714987e-7` | ODE fails; grid passes |

## Competing-cause checks

The two ODE decomposition replays verified the exact frozen `ode_1` and
`ode_2` waveform hashes before evaluating the trajectory's raw pre-NQC mode
on common valid samples before attachment. Raw complex relative L2 is
`1.35465068e-8` nonspin and `1.29133375e-8` spin. The complete attached
waveforms are respectively `9.07920587e-6` and `4.93844880e-5`. Changes in
the NQC matching coefficients and the attached waveform dominate this
observed discrepancy. This comparison does not prove the continuous ODE's
physical correctness. The largest common pre-attachment orbital-phase state
differences are `2.01658e-7` and `1.93733e-7` radians.
The NQC phase matching matrices have condition numbers about `8.79` and
`8.52`, with solve residuals at float64 roundoff; the amplitude matrices are
about `746` and `833`. The visible coefficient drift is therefore associated
with changes to matching inputs rather than a failed linear solve.

Peak and match differences in the refined ODE pair are at most
`1.52587890625e-5 M` nonspin and zero spin; boundary differences are zero.
Thus the frozen event gate passes. The waveform discrepancy is phase dominated:
refined phase RMS is `1.08242e-5` and `5.52206e-5` radians, while the amplitude
metrics are shown above. Changing the norm or fitting away phase would change
the frozen observable, so neither is a correction.

The default fine-grid nonspin restart occurs at `164.875 M`, compared with
`165.0 M` on the medium grid. Holding it at `165.0 M` reduces the refined grid
complex error from about `6.15e-6` to `1.84e-7`. It still exceeds the frozen
coarse-to-medium `1.33e-7`, so the monotonicity gate still fails. A row-specific
restart time is a diagnostic control, not a model rule. Holding both that
restart and the coarse integrator's first trial step at `0.5 M` on all three
output grids makes both nested-sample complex errors exactly zero and yields
identical NQC coefficients. This attributes the residual fixed-restart grid
sensitivity to the output-spacing-dependent initial step. The control reuses
one trajectory and provides no independent convergence or accuracy evidence.

On the spin refined ODE pair, using Hermite output resampling gives complex
`5.68975004e-5` and amplitude `1.69563692e-6`: that full resampling route
does not resolve the gate. Its coarse accepted trajectory hashes match the
other controls, but its sampled restart state changes the fine accepted
trajectory, so this is not an isolated change to final waveform interpolation.
Replacing only the five sampled NQC second derivatives with the diagnostic
local polynomial gives complex `5.25835014e-5` but amplitude
`1.75371344e-8`. Replacing only the two NQC phase-basis derivative rows with
the diagnostic local polynomial gives complex `3.69792597e-6` while amplitude
stays at `1.47015986e-6`. The latter two controls have identical accepted
coarse and fine trajectory hashes at each tolerance. Thus the amplitude and
phase sensitivities occur in distinct matching inputs, after trajectory
integration. None of these individual controls passes the ODE gate. The local
`0.5 M`, degree-six fit is a previously selected diagnostic window, not a
published differentiation prescription.

## Equation and blocker

[Bohé et al., Eq. (7)](https://arxiv.org/html/1611.03703) defines the NQC
amplitude and phase basis and fixes their coefficients through amplitude,
curvature, frequency, and frequency-slope conditions at the calibrated match
time. The current implementation differentiates natural-cubic splines of
sampled matching inputs to form those systems. The paper does not specify a
trajectory sampling or numerical differentiation rule. The controls above do
not identify a paper-derived replacement that passes the frozen gates; the
tabulated generator remains numerically unqualified.

The next decisive experiment is to form all derivatives required by the five
matching conditions at the continuous match time from the published ODE and
waveform equations
(for example, chain-rule/automatic differentiation), cross-check their values
against an independently converged dense trajectory, then predeclare that
single rule and rerun both rows over all frozen ODE and grid levels. The
derivative vectors, solved coefficients, and complete gate outcomes must all
be reported. A fixed row-specific restart or a tuned local-fit window alone
would not establish that rule.

## Artifacts

- `rerun_nonspin.json`, `rerun_spin.json`: exact frozen qualification reruns.
- `ode_decomposition_nonspin.json`, `ode_decomposition_spin.json`: hash-verified
  raw trajectory versus attached mode comparisons.
- `grid_restart_control.json`: moving-versus-fixed nonspin restart control.
- `grid_initial_step_control.json`: fixed restart and first-step control on all
  three nonspin grids.
- `ode_spin_hermite_control.json`, `ode_spin_second_control.json`: separate
  interpolation and sampled second-derivative controls.
- `ode_spin_phase_control.json`: sampled phase-basis derivative control.
- `probes/`: exact one-off scripts used for the controls. To replay the
  trajectory decomposition, copy `probes/diagnose_current_trajectory.py` into
  `dev/native_rom/` in the same source archive and run it as
  `python -m dev.native_rom.diagnose_current_trajectory` with `--campaign`,
  `--pair ode`, and `--json-out`. Other probes run from the repository root
  with `PYTHONPATH=src:.`.

No deployment, PR, CI edit, or production asset change was made.
