# IMRPhenomHM independent SXS:BBH:2265v3.0 holdout — red receipt

The [protocol](imrphenomhm_sxs_2265_protocol_20260927.md) was committed as
`860ebd2dd379c387214c7d01f4e7fe92a4518cb7` **before either waveform
was decoded**. Its fixed case, conditioning, one `(2,2)`-derived alignment,
and acceptance gates were applied without changes. Both public default-D
`IMRPhenomHM` frontends fail **nine of ten** observable gates. The `(2,2)`
mode passes; the other five modes and plus/cross at both orientations do not.
This case therefore does **not** qualify physical six-mode accuracy, despite
the separate same-model [LAL binary agreement](imrphenomhm_lal_six_modes_20260927.md).

## Source and reproduction

The independent numerical source is the SXS Collaboration's
[SXS:BBH:2265v3.0](https://doi.org/10.26138/SXS:BBH:2265v3.0), via the
[official CaltechDATA record](https://data.caltech.edu/records/258wv-a5t45).
See the [SXS catalog paper](https://doi.org/10.1103/PhysRevD.99.064045),
[SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3), and
[London et al.](https://arxiv.org/html/1708.00404) for the model's published
NR validation context. The six official Lev3/Lev2 N2 source size/MD5 pairs
are fixed in the protocol and checked before decoding. Their SHA256 digests,
in protocol order (Lev3 strain, sidecar, metadata; then Lev2 equivalents), are:

| Source | SHA256 |
| --- | --- |
| `Lev3:Strain_N2.h5` | `47b99bca2825f3659103e3e08ee926c7749ddfefbcdb679ca6b0dbe5cc550b07` |
| `Lev3:Strain_N2.json` | `e5e41ab62c43aba9555d47e7a7dc255b66bb55b5739870a1eea0389fcf4ecd30` |
| `Lev3:metadata.json` | `f564d8563f285e62409278cc1b9d1596f86125a5e92ad2fb5f61d152833026ca` |
| `Lev2:Strain_N2.h5` | `0331c1cef89cbaf61717a7f0a61982a8ff052607c2b16ceaf40dd376522af022` |
| `Lev2:Strain_N2.json` | `bd78e384599e7f76cfb74c8a617a33d8b1961aff2378aad001fbc6868a94bc58` |
| `Lev2:metadata.json` | `050bf1ab76f2481247441272930cda6d1b9c96aeeec0a00b98a707254f9d8722` |

The `sxs 2026.0.6` decoder reports `frame_type='inertial'` and a single
identity quaternion `[1,0,0,0]` for **both** resolutions (77 stored modes
each); preparation requires these decoded frame properties. No rotating
frame conversion or frame alignment is applied.

Host `len`; cwd `/tmp/diffgw-hm-sxs-2265-20260927/repo`; external data
directory `/tmp/diffgw-hm-sxs-2265-20260927/data`. The isolated Python
3.12.3 environment used `sxs 2026.0.6`, NumPy 2.5.3, SciPy 1.18.1,
JAX/JAXlib 0.11.2, PyTorch 2.14.0+cpu, SymPy 1.14.0, mpmath 1.3.0,
and pytest 9.1.1. The preparation command was:

```sh
PYTHONPATH=repo/src:repo/tests .venv/bin/python \
  repo/scripts/prepare_imrphenomhm_sxs_nr_holdout.py --data-dir data
```

Run it from `/tmp/diffgw-hm-sxs-2265-20260927`, or from any checkout with
an absolute script path and external `--data-dir`. The resulting compressed
derived fixture has SHA256
`d7c53e4450e59ddf21711fff481d4ba152d48be4693050450489bed04c62e7ba`.
The strict test command, from `repo`, is:

```sh
env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 DIFFGW_REQUIRE_HM_SXS_NR=1 \
  DIFFGW_HM_SXS_NR_DIR=/tmp/diffgw-hm-sxs-2265-20260927/data \
  PYTHONPATH=src:tests ../.venv/bin/python -m pytest -q -s \
  tests/test_imrphenomhm_sxs_nr_holdout.py -ra
```

Preparation PID `503993` completed and logged to
`/tmp/diffgw-hm-sxs-2265-20260927/prepare.log`. Test PID `535844`
completed in 12.82 seconds and logged to
`/tmp/diffgw-hm-sxs-2265-20260927/holdout-test.log`:
**2 failed** (Torch and JAX), with no skipped strict tests. The full
per-backend machine-readable rows remain outside Git beside the fixture as
`hm_sxs_2265_torch_metrics.json` and `hm_sxs_2265_jax_metrics.json`; neither
raw nor derived SXS waveform arrays are redistributed here. The two backends'
reported statistics differ by at most `2.04e-13` in absolute value.
After adding the explicit decoded inertial-frame assertion, the same prepare
command ran again as PID `730203` (`prepare-frame.log`) and produced the
**identical** fixture SHA256. The same strict command ran as PID `739943`
(`holdout-frame-test.log`, `holdout-frame-test.exit=1`) and again reported
**2 failed in 11.80 s** with the same nine failed observable names. Both jobs
have finished; no further check or stop is pending. During a rerun, the stop
command on `len` is `kill <active PID>`.

## Frozen-gate results

The Torch values below are printed to sufficient precision to expose each
failure; JAX is numerically the same at this precision. `M` is the
phase-sensitive model/Lev3 full-band match, `A` the model/Lev3 norm ratio,
`R` the Lev2/Lev3 match after the separate single `(2,2)` alignment, and
`A₂₃` its norm ratio. Each row's applicable fixed gate is in the protocol and
in the external metrics JSON. The model-to-Lev3 `(2,2)` alignment was
`Δt=0.0055908203124999944 s`, `φ=0.8943899694847643 rad`; the independent
Lev2-to-Lev3 alignment was `Δt=0.017553710937499994 s`,
`φ=-0.4611200253325575 rad`.

| Observable | M | A | R | A₂₃ | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `(2,2)` | 0.991262 | 1.003659 | 0.999998 | 1.000129 | pass |
| `(2,1)` | -0.441332 | 0.671323 | 0.999863 | 0.998437 | **fail** |
| `(3,3)` | -0.728477 | 1.070802 | 0.999947 | 1.000275 | **fail** |
| `(3,2)` | 0.677924 | 1.111389 | 0.999937 | 1.000525 | **fail** |
| `(4,4)` | -0.191982 | 1.160160 | 0.999875 | 1.000410 | **fail** |
| `(4,3)` | 0.045247 | 0.984775 | 0.999936 | 1.000973 | **fail** |
| plus `(0.7,0.3)` | 0.928701 | 0.994322 | 0.999998 | 0.999871 | **fail** |
| cross `(0.7,0.3)` | 0.929553 | 0.996547 | 0.999998 | 0.999947 | **fail** |
| plus `(1.2,-0.2)` | 0.849969 | 0.989296 | 0.999994 | 0.999485 | **fail** |
| cross `(1.2,-0.2)` | 0.858970 | 0.991440 | 0.999997 | 0.999283 | **fail** |

Every Lev2–Lev3 resolution gate passes by a wide margin. Thus numerical
resolution is not the observed reason for the model failures in this fixed
test. The `(2,1)` amplitude ratio also violates its frozen 0.80 lower bound;
the other nine amplitude ratios are within their respective frozen intervals.

## Post-hoc, non-acceptance diagnostics

The `(2,2)` alignment fixes orbital phase only modulo `π`. A bounded
diagnostic reapplied the **alternate global** branch `φ+π` to all modes and
their projections, without per-mode fitting or changing the primary result.
Odd-`m` matches become `(2,1)=0.441332`, `(3,3)=0.728477`, and
`(4,3)=-0.045247`, still failing. Even-`m` matches are unchanged, including
`(4,4)=-0.191982` and `(3,2)=0.677924`. At the first orientation, plus/cross
rise to `0.976620/0.977563`; at the second, they are
`0.946392/0.955391`, so its plus result remains below the frozen 0.95
threshold. These are diagnostic numbers, **not alternate acceptance results**.

The magnitudes of the complex mode coherences, which independently remove
an arbitrary *per-mode* phase only for diagnosis, are `(2,2)=0.992547`,
`(2,1)=0.951469`, `(3,3)=0.998154`, `(3,2)=0.724978`,
`(4,4)=0.972260`, `(4,3)=0.576375`. They suggest that convention or relative
phase accounts for much of the `(2,1)`, `(3,3)`, `(4,4)` signed-match failure,
while `(3,2)` and `(4,3)` also have larger shape disagreement. This is an
inference, not a correction or proof of cause. Direct six-pair SXS
time-domain polarization FFT and symmetry projection of the six positive-m
targets agree to matches `0.9999997–0.99999995` across the four projections,
which checks the target-side truncation/projection convention internally.

The next investigation should derive the native-versus-SXS **relative mode
phase convention** from published equations and test it on a separate case,
without fitting this holdout or revising its gate. If a convention correction
is justified, it needs a new predeclared protocol and independent holdout;
the red result here remains the historical result.
