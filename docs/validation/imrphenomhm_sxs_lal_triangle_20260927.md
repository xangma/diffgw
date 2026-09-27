# Frozen SXS–Torch–JAX–LAL HM triangle (non-gating)

The public default-D six-mode IMRPhenomHM implementations in Torch, JAX, and
pinned LALSuite 7.26.15 have essentially the **same complex waveforms and the
same NR residuals** on the frozen SXS:BBH:2265 and SXS:BBH:3631 systems.
Each passes only the original `(2,2)` observable gate: the other nine of ten
mode/projection gates remain red on both systems. This experiment does not
change a model formula, fixture, conditioning choice, or acceptance gate.

## Reproducible scope

The reusable [diagnostic](https://github.com/xangma/diffgw/blob/codex/waveform-evidence-integration/scripts/diagnose_imrphenomhm_sxs_lal_triangle.py)
requires the already frozen fixture and source directories from the
[2265](imrphenomhm_sxs_2265_protocol_20260927.md) and
[3631](imrphenomhm_sxs_3631_protocol_20260927.md) protocols. It verifies the
source MD5 records and fixture SHA256 before calculating anything. The
fixture SHA256 values are `d7c53e4450e59ddf21711fff481d4ba152d48be4693050450489bed04c62e7ba`
and `f51ce5cb5613ca5bb1d61e67876f81fc3ef3a4e7a2f1feeae965505e3a8e8bed`,
respectively. The pinned public LAL binary is accessed only through
`SimInspiralChooseFDModes` and `SphHarmFrequencySeriesGetMode`, with the
already maintained positive-frequency mode adapter `(-i)^m conj(H_lm(-f))`.
The clean-room diagnostic worker inspected no LAL source code or other
external waveform implementation.

On `len`, the run used `/tmp/diffgw-hm-sxs-3631-20260927/repo` as cwd and:

```sh
env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=/home/xangma/diffgw-pr45-318b37f7/.oracle-deps:/tmp/diffgw-hm-sxs-3631-20260927/repo/src:/tmp/diffgw-hm-sxs-3631-20260927/repo \
  /tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  /tmp/diffgw-hm-triangle-20260927/diagnose_imrphenomhm_sxs_lal_triangle.py \
  --data-2265 /tmp/diffgw-hm-sxs-2265-20260927/data \
  --data-3631 /tmp/diffgw-hm-sxs-3631-20260927/data \
  --output /tmp/diffgw-hm-triangle-20260927/triangle.json
```

The completed final run was PID `2452952`, with log
`/tmp/diffgw-hm-triangle-20260927/run.log` and JSON receipt
`/tmp/diffgw-hm-triangle-20260927/triangle.json`, all outside Git. A live
run can be stopped with `ssh len 'kill 2452952'` (the final PID has finished).
Versions were LALSuite `7.26.15`, NumPy `2.5.3`, SciPy `1.18.1`, Torch
`2.14.0+cpu`, JAX/JAXlib `0.11.2`. The full per-mode and per-projection
full/early/high statistics, resolution checks, phase checks, and invariant
summaries are in that JSON file. The local and len script SHA256 both equal
`1f001ea51a20559642c3ed4ee7c458710a51b6d0ffdd1dedd0dc3e7ce1acc0c7`;
the copied JSON SHA256 on both hosts is
`6422de1adc5cc70012d95fa1a6e8b3b748a1ef089691e7b4bf1c227fafabedaa`.

## Frozen-gate and parity result

Each backend was evaluated with the fixture's exact masses, zero spins,
distance, frequency grid, and finite reference frequency (`30 Hz` for 2265,
`35 Hz` for 3631). Each was aligned only by the protocol's single `(2,2)`
time/orbital-phase fit. The real normalized inner product and norm ratio
then reproduced the original frozen failures. Values below are Torch;
JAX and LAL agree at the shown precision.

| Observable | 2265 match / ratio | 3631 match / ratio |
| --- | ---: | ---: |
| `(2,2)` | `0.99126 / 1.00366` | `0.99329 / 1.00528` |
| `(2,1)` | `-0.44133 / 0.67132` | `0.23978 / 0.64801` |
| `(3,3)` | `-0.72848 / 1.07080` | `-0.07121 / 1.07714` |
| `(3,2)` | `0.67792 / 1.11139` | `0.79742 / 1.14904` |
| `(4,4)` | `-0.19198 / 1.16016` | `-0.96133 / 1.18245` |
| `(4,3)` | `0.04525 / 0.98477` | `0.38883 / 1.07392` |
| plus/cross at angle 0 | `0.92870 / 0.92955` | `0.92358 / 0.92544` |
| plus/cross at angle 1 | `0.84997 / 0.85897` | `0.82639 / 0.84102` |

The projection row entries are plus/cross **matches**; their ratios and all
resolution-gate values are in JSON. All ten Lev2–Lev3 resolution gates pass.
Across all six raw complex modes, the largest Torch–JAX relative L2 is
`7.37e-13` (2265) or `1.25e-12` (3631); the largest Torch–LAL value is
`1.68e-6` or `2.01e-6`. The agreement answers the first question: these red
residuals are not specific to the native frontends.

## Phase and frequency covariance

For a normalized mode `u_lm=H_lm/|H_lm|`, the doubled relative-mode invariant

`T_lm(f)=u_lm(f)^2 conj[u_22(2f/m)]^m`

uses the inspiral correspondence `f_22=2f/m` discussed around Eqs. (1)–(3)
of [London et al. (2018)](https://arxiv.org/html/1708.00404v4). The carrier
phase is **unwrapped before interpolation** at `2f/m`, matching the original
SXS receipts. A common time shift cancels because
`2 f = m (2f/m)`; an orbital azimuth shift `u_lm → exp(±i m δ)u_lm` cancels
for the same reason. The odd-`m` branch is removed by squaring.

Changing native `phic` by `+0.2 rad` gives the independently predicted
`exp(+i m·0.2)` mode ratio to `3.1e-15` and leaves `T_lm` invariant to
`5.5e-15`. In contrast, the pinned public LAL `SimInspiralChooseFDModes`
output shows **no response** to `phi_ref+0.2` (mode-ratio error from unity
below `1.5e-14`). This is a black-box observation about that mode API, not
an orbital-covariance test of LAL. In all three backends, amplitudes change
by at most `7e-16` and `T_lm` changes by at most `2.4e-14` in complex chord
distance. The separate identity formed from measured mode and carrier
ratios is only an algebraic consistency check; its error is at most
`1.7e-15`.

Changing finite `f_ref` by `+5 Hz` also preserves each mode's amplitude to
`6e-16` and produces a frequency-independent phase ratio within each mode:
maximum full-band phase scatter is `2.5e-14` in complex chord distance.
The resulting invariant shift is reproduced by inserting those measured
mode/carrier ratios, to `1.6e-15`; this is again a consistency check, not an
independent physical prediction. Native finite-`f_ref` to native `f_ref=None`
likewise preserves amplitudes to `5e-16` with per-mode phase scatter at most
`6e-14`. These shifts differ across 2265 and 3631, so the finite-reference
invariant has a case-dependent gauge. The original finite-reference gates
remain the acceptance result.

## One frame factor, one held-out case

For a diagnostic unit factor `c=exp(iθ)` applied uniformly to all model
modes, direct substitution gives `T_lm → c^(2−m) T_lm`. Thus `(3,2)` is
unchanged. One `θ` was fitted across the four affected modes `(2,1)`,
`(3,3)`, `(4,4)`, `(4,3)` and **only the 2265 early band** (`30–80 Hz`),
minimizing mean circular chord loss. It was then applied unchanged to 3631;
no per-mode or per-case factor was fitted, and no production correction was
made.

| Invariant setting | 2265 fit `θ` | 3631 held-out early residuals after same `c`: 21 / 33 / 32 / 44 / 43 (rad) |
| --- | ---: | --- |
| Frozen finite `f_ref` | `+1.66646` | `−1.964 / +1.254 / −0.190 / +2.767 / +1.233` |
| Diagnostic native `f_ref=None` | `−2.36206` | `−0.513 / −0.196 / −0.190 / −0.134 / −0.217` |

The reference-free factor is close to the previously examined algebraic
`−exp(iπ/4)` convention candidate, but is **empirically fitted here** and
cannot establish a universal convention. Its held-out transfer is
qualitative: `(2,1)` retains about `0.51 rad`, `(3,2)` cannot be changed by
any common factor, and the high-frequency invariant remains broad. For
example, on 2265 the reference-free `(3,2)` high-band resultant length is
`0.355` and `(4,3)` is `0.209`; on 3631 they are `0.596` and `0.267`.
The receipt's `lal_gauge_mapped` view multiplies finite-reference LAL modes
by the unit ratio of native `f_ref=None` to finite-reference modes. It uses
no NR data and agrees with native reference-free modes to the same ~`2e-6`
L2 level, but **is not an independent LAL `f_ref=None` oracle**.

The next validation should retain every frozen red gate and predeclare a
separate unseen-NR comparison with the paper's aLIGO-weighted statistic and
optimized extrinsic parameters, alongside mode-shape diagnostics. That
would test the narrower published performance claim without rewriting this
holdout. If accurate `(3,2)`/`(4,3)` merger and ringdown modes are required,
a separate mixing-capable model is needed; London et al. explicitly identify
omitted spherical–spheroidal mixing as a limitation of this model. The
remaining `(2,1)` and high-band residuals prevent interpreting all red
gates as a phase convention alone.
