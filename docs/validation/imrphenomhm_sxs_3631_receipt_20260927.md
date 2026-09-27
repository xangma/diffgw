# IMRPhenomHM SXS:BBH:3631v3.0 holdout — red receipt

The [second protocol](imrphenomhm_sxs_3631_protocol_20260927.md) was committed
as `6e59ae685414afac388caf0281e57a9459bca572` **before downloading or
decoding either 3631 waveform**. Its source identities, 35–350 Hz band,
conditioning, single `(2,2)` alignment, and gates were not changed after
seeing the data. Both default-D-carrier native frontends fail the same **nine
of ten** observable gates: only `(2,2)` passes. All Lev2–Lev3 resolution gates
pass. This independent q≈4 case does not qualify physical six-mode accuracy;
it corroborates the red [q≈3 case](imrphenomhm_sxs_2265_receipt_20260927.md)
without fitting to either case.

The 35 Hz lower edge was chosen from the metadata's initial orbital frequency
and the short 15-orbit length, not from decoded strain. It is about 2.1 times
the inferred initial `(2,2)` frequency at the frozen `80 M_sun` total mass.

## Source and reproduction

The SXS Collaboration's [SXS:BBH:3631v3.0](https://doi.org/10.26138/SXS:BBH:3631v3.0)
comes from the [official CaltechDATA record](https://data.caltech.edu/records/fxz1a-agp58).
The Lev3 metadata lists `date_run_earliest=2022-05-04` and
`date_run_latest=2022-12-12`, after the 2018
[London et al. model paper](https://arxiv.org/html/1708.00404). The
[SXS Catalog 3 paper](https://arxiv.org/html/2505.13378v2) states that
center-of-mass and memory corrections were applied to all v3.0 waveforms;
both pinned sidecars record space translations, boosts, and memory additions.
The six official Lev3/Lev2 N2 source files passed the protocol's size and MD5
checks before decoding. Their SHA256 digests, in protocol order, are:

| Source | SHA256 |
| --- | --- |
| `Lev3:Strain_N2.h5` | `07257bb0beab708546bad6e0bee7be0d48f3ea8aeb1c29545bd9cf70c6163956` |
| `Lev3:Strain_N2.json` | `174d80754603c9ac13c3709ee1c432ee6f0eb922e8658ada5dc092bb521833c5` |
| `Lev3:metadata.json` | `f3da914c891e3ddaea0a6c6d042db03e79ac26e111a06524df24f78d00a0baba` |
| `Lev2:Strain_N2.h5` | `42452f9e5a9e72b289e938c334c09cd45fe7b6d2e9e03b5e241192e53b3d984a` |
| `Lev2:Strain_N2.json` | `597144cd6400a83e9d685e10867b063c6a1f837327a665cdf3cb38b658c1ddea` |
| `Lev2:metadata.json` | `09d0f7333e879705a1323495ebc4cc6759c371238b0fafe39dc623ddd7e1b8f0` |

The run used host `len`, cwd `/tmp/diffgw-hm-sxs-3631-20260927/repo`, and
external data directory `/tmp/diffgw-hm-sxs-3631-20260927/data`. The existing
isolated Python 3.12.3 environment contained `sxs 2026.0.6`, NumPy 2.5.3,
SciPy 1.18.1, JAX/JAXlib 0.11.2, PyTorch 2.14.0+cpu, SymPy 1.14.0, mpmath
1.3.0, and pytest 9.1.1. Preparation used:

```sh
PYTHONPATH=repo/src:repo/tests \
  /tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  repo/scripts/prepare_imrphenomhm_sxs_3631_nr_holdout.py --data-dir data
```

From the parent directory above, preparation PID `105471` finished with exit
0; log `prepare.log`, exit marker `prepare.exit`. The derived fixture
`data/hm_sxs_3631_reference.npz` has SHA256
`f51ce5cb5613ca5bb1d61e67876f81fc3ef3a4e7a2f1feeae965505e3a8e8bed`.
The strict test command, from `repo`, was:

```sh
env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 DIFFGW_REQUIRE_HM_SXS_3631_NR=1 \
  DIFFGW_HM_SXS_3631_NR_DIR=/tmp/diffgw-hm-sxs-3631-20260927/data \
  PYTHONPATH=src:tests /tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  -m pytest -q -s tests/test_imrphenomhm_sxs_3631_nr_holdout.py -ra
```

Test PID `123423` finished with exit 1 in 12.88 seconds: **2 failed, zero
skipped**. Log `holdout-test.log` and exit marker `holdout-test.exit` are in
the parent directory. Complete machine-readable rows are in the external
`data/hm_sxs_3631_{torch,jax}_metrics.json` files. No raw or derived waveform
array is committed. Both jobs are finished; no next check or stop is pending.
During a rerun, stop an active job with `kill <PID>` on `len`.
The normal source-only invocation without the strict environment variables
reports two expected skips. Local `uvx ruff check` on the two new Python files
and `uvx --from mkdocs-material --with mkdocstrings-python mkdocs build
--strict` both passed.

## Frozen-gate results

The Torch values below expose every gate; JAX differs by at most
`4.15e-13` in any reported statistic. `M` is the phase-sensitive model/Lev3
full-band match, `A` its norm ratio, `R` the Lev2/Lev3 match after a separate
single `(2,2)` alignment, and `A₂₃` its norm ratio. The model-to-Lev3
alignment was `Δt=0.00546875 s`, `φ=-0.12824058061658977 rad`; Lev2-to-Lev3
was `Δt=-0.0007568359375 s`, `φ=0.0676591273151671 rad`.

| Observable | M | A | R | A₂₃ | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `(2,2)` | 0.993291 | 1.005279 | 0.999989 | 1.000262 | pass |
| `(2,1)` | 0.239782 | 0.648009 | 0.999948 | 0.999748 | **fail** |
| `(3,3)` | -0.071209 | 1.077139 | 0.999891 | 1.000309 | **fail** |
| `(3,2)` | 0.797421 | 1.149037 | 0.999960 | 0.999885 | **fail** |
| `(4,4)` | -0.961330 | 1.182446 | 0.999484 | 1.000136 | **fail** |
| `(4,3)` | 0.388829 | 1.073919 | 0.999886 | 1.000986 | **fail** |
| plus `(0.7,0.3)` | 0.923581 | 1.018358 | 0.999981 | 1.000535 | **fail** |
| cross `(0.7,0.3)` | 0.925445 | 1.017304 | 0.999983 | 1.000488 | **fail** |
| plus `(1.2,-0.2)` | 0.826393 | 1.048928 | 0.999969 | 1.000760 | **fail** |
| cross `(1.2,-0.2)` | 0.841022 | 1.034684 | 0.999972 | 1.000591 | **fail** |

The `(2,1)` norm ratio also violates its fixed 0.80 lower bound; the other
nine ratios lie within their respective fixed intervals. The failed signed
matches are not explained by Lev2–Lev3 numerical resolution at these gates.

## Interpretation after the frozen test

These results do not identify whether model calibration, inter-mode phase
convention, or the more restrictive flat-band statistic dominates the model
disagreement. The gate remains red pending a source-derived diagnosis and a
future, separately predeclared holdout after any correction.

## Cross-case phase-convention diagnostic (not an acceptance result)

The [2265 receipt](imrphenomhm_sxs_2265_receipt_20260927.md) defines the
doubled relative-phase invariant
`T_lm(f) = u_lm(f)^2 conj[u_22(2f/m)]^m`, with `u = H/|H|`. It cancels a
common time and orbital-phase shift and removes the odd-`m` orbital branch.
For this case, a read-only Torch calculation on `len` used the fixture's
`35–85 Hz` mode-frequency grid, retaining only bins with `2f/m ≥ 35 Hz`.
The comparison took the circular mean of `T_model conj(T_NR)` with the native
`f_ref=None` option; the 2265 calculation used `30–80 Hz` and
`2f/m ≥ 30 Hz`. Both calculations used the frozen NR conditioning, and neither
changed a gate or model output.

| Mode | 2265 residual (rad) | 3631 residual (rad) | After diagnostic `c` on 2265 | After diagnostic `c` on 3631 |
| --- | ---: | ---: | ---: | ---: |
| `(2,1)` | +1.894 | +1.849 | -0.462 | -0.507 |
| `(3,3)` | -2.544 | -2.558 | -0.188 | -0.202 |
| `(3,2)` | -0.101 | -0.190 | -0.101 | -0.190 |
| `(4,4)` | +1.495 | +1.425 | -0.076 | -0.146 |
| `(4,3)` | -2.499 | -2.579 | -0.143 | -0.223 |

Here `c = -exp(iπ/4)` is an **algebraic diagnostic factor**, uniformly
applied to all model modes, so the invariant changes by `c^(2-m)`. It was
not fitted per mode or applied to a production waveform. The
[2019 SXS catalog convention paper](https://arxiv.org/abs/1904.04831)
documents an overall strain-sign revision, and the
[SXS third-catalog paper](https://arxiv.org/html/2505.13378v2)
confirms it was retroactively applied to earlier simulations. The `π/4`
factor has the form of a stationary-phase Fourier term. A direct
time-domain-to-frequency-domain stationary-phase check of both pinned SXS
files at `f_22=40 Hz` reproduced the `(3,3)`, `(3,2)`, `(4,4)`, and `(4,3)`
NR invariant phases within `0.05 rad`. These observations make a convention
explanation plausible for much of the *early-band* constant offset, but they
do not establish which exact NR convention the
[2018 HM model paper](https://arxiv.org/html/1708.00404v4) used.

The native finite-`f_ref` assembly adds a mode-dependent phase gauge
`(m/2-1) B`, with `B = slope_peak M_seconds f_ref`. This changes the doubled
invariant by `(m-2) B`: measured `B` was `4.028 rad` for 2265 and
`5.479 rad` for 3631. Thus the smaller offsets above at `f_ref=None` do
not repair the **frozen finite-`f_ref`** comparisons. A constant phase factor
also cannot fix the high-band `(3,2)` and `(4,3)` mode-shape losses in the
2265 receipt. A production phase change requires a source-derived convention
map and a new independent holdout; the current physical gates remain red.
