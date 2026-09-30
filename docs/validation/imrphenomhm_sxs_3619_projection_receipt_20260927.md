# IMRPhenomHM SXS:BBH:3619 projected-strain holdout — red receipt

**This case is infeasible for qualification under its frozen protocol.** Both
Torch and JAX meet the two model/NR projected-match and amplitude gates, but
the Lev2–Lev3 numerical-resolution screen fails at both source angles. The
[protocol](imrphenomhm_sxs_3619_projection_protocol_20260927.md) was committed
as `d3a2ef4b313c582efea896a44a512fea55284c15` before either strain file
was downloaded or decoded; its clarification of the same-simulation
resolution screen, `3427ebfa554c16fcd0b33d7de3afc4846694506e`, was also
committed before download. Neither gate nor conditioning was altered after
the data were seen. The earlier [2265](imrphenomhm_sxs_2265_receipt_20260927.md)
and [3631](imrphenomhm_sxs_3631_receipt_20260927.md) phase-locked six-mode
gates remain red.

## Source and metric audit

The numerical source is the SXS Collaboration's
[SXS:BBH:3619v3.0](https://doi.org/10.26138/SXS:BBH:3619v3.0) from the
[official CaltechDATA record](https://data.caltech.edu/records/6fw56-9q785).
The case was selected from the official version-3 metadata before strain
retrieval. The Lev3 metadata give q≈5, negligible spins, and a 2021 run date,
after the 2018 [London et al. model paper](https://arxiv.org/html/1708.00404v4).
The six pinned Lev3/Lev2 strain, sidecar, and metadata files passed their
official size and MD5 checks before the SXS decoder read them. Their SHA-256
digests, in protocol order, are:

| Source | SHA-256 |
| --- | --- |
| `Lev3:Strain_N2.h5` | `274557fb43434ed93c0a5f2c989be152216ec256bb8888a23fbfbfd8f20feaf9` |
| `Lev3:Strain_N2.json` | `eb86644dead5a85a6af12250ad2d4ed1bca2845bae083a20877af57987320bc8` |
| `Lev3:metadata.json` | `9226d3622cfc51e96a4841724168157c5a6b6c22db8d08bfeaa45fb95fbfd085` |
| `Lev2:Strain_N2.h5` | `251ae2c0e69159d59d94ff9e1dace833b81bdab639525d7eb67d6ab0dec7947f` |
| `Lev2:Strain_N2.json` | `5d7ad5314930b12965b6faa4f8ccfabf76609cc813ee38757fe3f46040d75c9c` |
| `Lev2:metadata.json` | `8b1305d0c7f1f5fece402b0e1c65f3ac00461b1cc41534ca08ae6202a4901988` |

Both sidecars record space translations, boosts, and memory additions; no
correction was reversed. The memory integration starts differ: Lev3 `342.5 M`
and Lev2 `200.5 M`. This version-3 correction context follows the
[SXS Catalog 3 description](https://arxiv.org/html/2505.13378v2). The SXS
decoder reported inertial spin-`-2` strain and all required `2≤ell≤5` modes.
The prepared external fixture SHA-256 is
`21b6e75f133e4ddc3b3c8e55e52fae6afa99c5e4103c38245ed283e2a587bdcb`.

The match follows the paper's detector-projection comparison: fixed intrinsic
parameters and inclination, with template arrival time, orbital phase, and
polarization optimized against full NR. It uses the [documented PyCBC aLIGO
design PSD generator](https://pycbc.org/pycbc/latest/html/psd.html) at
validation time, rather than copying a LAL array or implementation. The PSD
samples used here have SHA-256
`0d20d2092bfaeb6073b0d656f4a3eb495bc3adbeb350ef279420cb4ffd0d67ea`.
The committed known-shift check recovers a synthetic arrival exactly. At each
reported optimum, a direct `4 Re Σ a* b/S_n` evaluation agrees with the FFT
maximizer to `1.2e-16`. All four differential-evolution seeds agree within
`4.5e-11` or better. Thus neither the FFT sign/scale nor a missed optimizer
mode explains the red result. This is a paper-like one-case test, not a
reproduction of London's full ensemble.

## Frozen-gate result

All values below are Torch; JAX model matches differ by at most `2.9e-15` and
share the same NR resolution statistics. `M` is the optimized model/Lev3
detector match (gate `≥0.980`); `A` is its PSD-weighted norm ratio (gate
`0.85–1.15`); `R` is the Lev2–Lev3 match with frozen time/polarization
optimization but no orbital-phase retuning (gate `≥0.995`).

| Source `(iota,psi)` | M | A | R | Model gates | Resolution gate |
| --- | ---: | ---: | ---: | --- | --- |
| `(0.7,0.0)` | 0.996195851379 | 0.961627051 | 0.991880051 | pass | **fail** |
| `(1.2,0.4)` | 0.985979162286 | 0.850460134 | 0.980142976 | pass | **fail** |

The resolution optima have candidate/Lev3 norm ratios `0.983814632` and
`0.773284683`; their arrival shifts are `-0.0106201171875` and
`-0.010498046875 s`. The resolutions' `(2,2)` phases agree within about
`5e-5 rad` at common `t=342.5 M`, but differ by about `0.60 rad` at common
`t=7000 M`; their peak times are about `21.9 M` apart.

After seeing the red result, a **separate exploratory diagnostic** also
optimized one common orbital phase for *all* Lev2 NR modes, in addition to
arrival time and polarization. Its full-mode basis reconstructed the frozen
Lev2 plus/cross fixture to relative maximum errors below `4.2e-13`. With
that added freedom the best Lev2–Lev3 matches become `0.999884070` and
`0.999842333`, with norm ratios `0.999932` and `0.995088`; the best common
orbital phases are about `-0.408` and `-0.399 rad` modulo `2π`. Three of four
seeds at the second angle found the best value; the fourth was lower by
`2.56e-5`. This strongly supports an accumulated orbital phase difference as
the main reason the *frozen no-orbital-retuning* screen fails. It also shows
that the NR shapes can agree closely under a standard alignment. It is not a
predeclared gate and **cannot turn this red holdout into a qualified pass**.
The complete external exploratory record is
`data/hm_sxs_3619_resolution_orbital_diagnostic.json`, with basis SHA-256
`25a4271ff5c4442a2e19c97178c56e54960d4344c330026d25b31a5a1329b170`.

## Reproduction and disposition

Host `len`; cwd `/tmp/diffgw-hm-sxs-3619-20260927/repo`; external data in
`/tmp/diffgw-hm-sxs-3619-20260927/data`. Preparation used Python 3.12.3,
NumPy 2.5.3, SciPy 1.18.1, SXS 2026.0.6, and SymPy 1.14.0. Evaluation used
NumPy 2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch 2.14.0, and JAX 0.11.2.
The raw files and derived arrays stay outside Git. Preparation command:

```sh
PYTHONPATH=src:tests /tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  scripts/prepare_imrphenomhm_sxs_3619_projection.py \
  --data-dir /tmp/diffgw-hm-sxs-3619-20260927/data
```

Preparation PID `688517` completed successfully; its log is
`/tmp/diffgw-hm-sxs-3619-20260927/prepare.log`. The final strict command was:

```sh
env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 DIFFGW_REQUIRE_HM_SXS_3619_PROJECTION=1 \
  DIFFGW_HM_SXS_3619_PROJECTION_DIR=/tmp/diffgw-hm-sxs-3619-20260927/data \
  PYTHONPATH=src:tests /tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  -m pytest -q -s tests/test_imrphenomhm_sxs_3619_projection.py -ra
```

Runner PID `810339` finished with exit 1: **2 failed, 1 passed** in 17.69 s.
The log and exit marker are `holdout-test.log` and `holdout-test.exit` beside
the remote `repo`; complete machine-readable metrics are the external
`data/hm_sxs_3619_{torch,jax}_projection_metrics.json` files. No job remains
active; there is no next check or stop command. During a rerun, stop an active
PID with `kill <PID>` on `len`. The source-only invocation yielded 1 pass and
2 expected skips. Ruff passed for the new Python files.

The historical exploratory resolution diagnostic used
`/tmp/diffgw-hm-sxs-3619-20260927/data`, first preparing alternatives in
the SXS environment and then scoring them in the PyCBC environment, each
with `PYTHONPATH=.:src:tests`. PIDs `931927` and `939587` both completed
with exit 0; logs and exit markers are
`diagnostic-{prepare,score}.{log,exit}` beside the remote `repo`. Its
one-off replay script is not maintained. It did not alter the fixture or
strict test; the maintained preparer and holdout test above remain the
frozen workflow.

This red receipt is retained without gate changes. A future numerical
reference or resolution-screen design must be predeclared against an unseen
case before any new qualification claim; the 3619 result cannot be rescored
as an independent pass after inspecting it.
