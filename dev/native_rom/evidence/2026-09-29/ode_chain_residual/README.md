# Residual nonspinning SEOBNR frozen-gate diagnosis

This development-only probe follows the opt-in ODE chain-rule NQC derivative
candidate in [`../ode_chain_derivatives/`](../ode_chain_derivatives/README.md).
The matching inputs follow [Bohé et al. (2017), Eq. (7)](https://arxiv.org/html/1611.03703),
as implemented by diffgw's published-equation mode and ODE field. No LAL
implementation source or private DCC text was used. The frozen `(30,20)`
nonspinning row, `tabulated` spin truncation, three ODE tolerance levels,
three output grids, events, limits, and waveform hashes were replayed
unchanged. The source manifest is
`2caa5b4a3bb20e01cd6e0b0e8891e76b5d49345e3a254c9cc2395d7304d06a13`,
the same as the prior chain candidate gate. The preregistered plan is
[PLAN.md](PLAN.md), SHA-256
`0a223c0f467dd0b26760118ca5aa2582bed1e30d53da085461ac1873699fd35a`.

## Decisive comparisons

[`probe.py`](probe.py) evaluates `(A,Aq1,Aq2,Aq3,p1,p2)` directly from the
published mode and NQC basis at the existing continuous match state. It
compares those values with the independently splined sample values. It then
re-solves the amplitude system with only its value row and RHS amplitude
replaced; both chain-rule derivative rows, targets, and the phase system stay
fixed. This is a coefficient counterfactual, not a new generator rule.

| Frozen level | Maximum direct minus spline value | Maximum amplitude coefficient change |
| --- | ---: | ---: |
| `ode_0` | `8.550105068e-14` | `9.691092373e-11` |
| `ode_1` | `8.434919430e-14` | `9.828227121e-11` |
| `ode_2` | `8.397449403e-14` | `9.685408031e-11` |
| `grid_1` | `8.389122730e-14` | `9.777778587e-11` |
| `grid_2` | `8.244793737e-14` | `9.864464801e-11` |

Phase values are not matching conditions, so this value-row change leaves
both phase coefficients unchanged. Its effect is much smaller than the
observed coefficient drift; it does not offer a credible correction to the
complex gate failures. The [level receipt](levels.json) gives all values,
counterfactual coefficients, match states, exact accepted-array hashes, and
replayed waveform hashes. All five waveform hashes and accepted-array hashes
match the prior [frozen chain gate](../ode_chain_derivatives/gate_nonspin.json).

The `ode_1` to `ode_2` pair changes match time by `-1.52587890625e-5 M`,
continuous match state by up to `2.290274931e-6`, and a phase NQC coefficient
by up to `9.951707868e-5`. Both coarse and fine accepted-array hashes differ.
The prior baseline [trajectory decomposition](../ode_gate_slice/ode_decomposition_nonspin.json)
measured pre-NQC complex relative L2 `1.354650681e-8` on common
pre-attachment samples, versus `9.079205867e-6` for its full baseline
waveform. The chain diagnostic first acts after the fine integration, so its
pre-NQC trajectory and raw mode are unchanged by that flag; this is an
inference from the source path, not a separately measured candidate raw-mode
receipt. The current full chain waveform error is `2.614471228e-6`.
Together these locate the remaining ODE sensitivity in the
tolerance-dependent trajectory/match-state and subsequent phase matching
path. The evidence does not apportion it between integration error and the
peak/match locator.

Default `ode_2` to `grid_1` and `grid_1` to `grid_2` have the same match time
but different coarse and fine accepted-array hashes. The finest grid changes
the default restart from `165 M` to `164.875 M`; the initial trial step also
depends on output spacing. Match-state changes are at most `1.694893115e-10`
and `2.027995549e-10` across those pairs, respectively, much smaller than
the ODE pair. The existing opt-in controls then held restart at `165 M` and
coarse first step at `0.5 M` across all three grid levels. In the
[control receipt](control.json), every coarse and fine accepted-array hash,
match time, and NQC coefficient matches; both nested waveform comparisons
are exactly equal with complex error `0.0`. This locates the default grid
sensitivity in the trajectory path. Because both controls were fixed
together, the experiment does not assign separate shares to restart and
first step or justify a row-specific default.

## Frozen gates and boundary

The [level receipt](levels.json) reproduces the previous chain-candidate
checks exactly. Each cell below shows coarse-to-medium then medium-to-fine
complex relative L2. The refined complex budget is `1e-6`; decreasing error
requires the latter no greater than the former plus `1e-12`.

| Pair | Complex relative L2 | Complex budget | Decreasing error | Amplitude budget | Peak, match, boundary |
| --- | --- | --- | --- | --- | --- |
| Nonspin ODE | `5.100139018e-6 → 2.614471228e-6` | **fail** | pass | pass | pass |
| Nonspin grid | `6.220681283e-9 → 8.178640339e-9` | pass | **fail** | pass | pass |
| Fixed-path grid control | `0.0 → 0.0` | pass | pass | pass | pass |

The [previous frozen two-row gate](../ode_chain_derivatives/README.md)
already shows that the spinning row passes both ODE and grid gates under the
same opt-in chain candidate. No numerical correction emerged from this
diagnosis, so no new candidate or production change was made and the spin
row was not rerun. The maximal supported finding is that the direct
continuous matching values agree with sampled splines to float64 precision,
the residual nonspinning ODE error follows the tolerance-dependent
trajectory/match/phase path, and the tiny nonmonotone grid error disappears
when its trajectory path is held constant. This does not validate the
physical model or qualify a training generator.

## Execution and verification

The jobs ran on host `len`, cwd
`/home/xangma/tmp/diffgw-seobnr-ode-gate-aa1f365f`, with
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`, `PYTHONPATH=src:.`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `JAX_ENABLE_X64=1`, and
`JAX_PLATFORMS=cpu`. Commands after the Python executable were:

- `dev/native_rom/evidence/2026-09-29/ode_chain_residual/probe.py --part levels --json-out dev/native_rom/evidence/2026-09-29/ode_chain_residual/levels.json` (PID `1515369`, remote log `levels.log`).
- `dev/native_rom/evidence/2026-09-29/ode_chain_residual/probe.py --part control --json-out dev/native_rom/evidence/2026-09-29/ode_chain_residual/control.json` (PID `1516002`, remote log `control.log`).

The expected next check was each JSON checkpoint and corresponding log. Both
PIDs have exited; while running, either could be stopped with `kill <PID>` on
`len`. The first attempt reached its initial numerical checkpoint but the
evidence writer rejected omitted development-only scope flags. The probe was
fixed before these complete runs. `python -m py_compile` passed for the final
probe. Both receipts report `diagnostic_complete`, identical before/after
generator source manifests, exact frozen-level hash replay, and finite
counterfactual systems. No deployment, production promotion, or PR occurred.
