# HM orbital convention: amplitude/phase mapping diagnostic

With the corrected orbital reference convention, the **phase mapping** accounts
for almost all of the optimized improvement from switching mapping options on
these two seen NR cases. Changing amplitude mapping alone has little effect.
This identifies a sensitivity, not a newly qualified waveform: 2374 remains
below the unchanged 0.98 match floor, and no diagnostic hybrid is promoted.
The [original failures](imrphenomhm_published_nr_receipt_20260927.md) and
[earlier source-option results](imrphenomhm_published_options_20260927.md)
retain their original scope.

## Controlled calculation

Both native frontends use `hm_reference_phase="orbital"`, `f_ref=30 Hz`,
published calibration, and the three named Mishra PN options. Mass, distance,
raw NR data, conditioning, frequencies and PSD are unchanged. Only the failing
second view is evaluated: inclination 1.2 rad and source polarization 0.4 rad.
The six modes and 30–1024 Hz comparison band are those of the frozen protocol.

`L` denotes `london_2018`; `R` denotes `phenomd_runtime`. For each mode the
diagnostic combines the magnitude of one complete native output with the phase
of the other, as $|P_A|P_\varphi/|P_\varphi|$. Thus `RL` changes only amplitude
mapping, `LR` only phase mapping, and `RR` both. Undefined phase at nonzero
amplitude is rejected. All sampled support masks agree. The 22 mode is unchanged
between native mapping choices to relative complex L2 below 1.21e-15.

All four variants and both backends first use one common alignment per case:
the prior Torch diagnostic's corrected-30-Hz optimum. The new native `LL`
reproduces its match and norm. Each variant is then optimized over the original
time, orbital phase and polarization coordinates with seeds 17, 23, 41 and 67.
Neither intrinsic parameters nor distance are fitted.

## Results and interpretation

Torch results are shown; JAX gives the same conclusions. Norm means template
norm divided by NR norm. Fixed columns retain the common alignment; optimized
columns report the best observed seed.

| Case | Magnitude / phase | Fixed match | Fixed norm | Optimized match | Optimized norm |
| --- | --- | ---: | ---: | ---: | ---: |
| 2374 | LL | 0.971789397 | 0.811099946 | 0.971789397 | 0.811099946 |
| 2374 | RL | 0.971872814 | 0.811411449 | 0.971872915 | 0.811053789 |
| 2374 | LR | 0.968848827 | 0.821376685 | 0.979692352 | 0.985909672 |
| 2374 | RR | 0.968828022 | 0.821716068 | 0.979583195 | 0.984981421 |
| 2163 | LL | 0.977392313 | 0.824196407 | 0.977392313 | 0.824196407 |
| 2163 | RL | 0.977346877 | 0.824538000 | 0.977346893 | 0.824566389 |
| 2163 | LR | 0.973738921 | 0.827737131 | 0.984557692 | 0.932958379 |
| 2163 | RR | 0.973661775 | 0.827976788 | 0.984504786 | 0.933009278 |

Amplitude-only optimized match changes are +0.000083519 and −0.000045420.
Phase-only changes are +0.007902955 and +0.007165378. Combining both changes
slightly lowers the best match relative to phase-only in each case. Runtime
phase worsens the match at the common alignment; its benefit appears after
physical realignment. It therefore does not uniformly reduce the fixed-spectrum
residual.

The normalized residual power after optimization falls from 0.081446468 to
0.040241551 for 2374 and from 0.068173251 to 0.033308641 for 2163 when changing
`LL` to `LR`. The corresponding projected magnitude terms fall from 0.059379633
to 0.020504358 and from 0.047089906 to 0.022208479, despite retaining the same
intrinsic mode magnitudes. This demonstrates why a large detector-magnitude
residual cannot by itself diagnose an intrinsic amplitude defect: phase and
polarization alter coherent modal interference. The retained output includes
the exact magnitude/phase split and additive frequency-band budgets.

Both `LR` and `RR` put 2163 within the original match/norm numerical bounds.
Neither puts 2374 above the match floor; its best `LR` result misses by
0.000307648. The best observed result is not a certified global optimum.
No run has a flagged angular boundary seed. The largest four-seed spread is
5.627e-6 for 2374 `RR`, below the original 1e-5 stability bound; all other
spreads are below 1.08e-10. This does not erase the earlier `f_ref=None`
boundary/seed instability. Maximum Torch/JAX relative complex mode L2 across
all four variants and both cases is 6.28e-13.

[London et al., Methods](https://arxiv.org/html/1708.00404v4) separately defines
the amplitude and phase maps and their printed transition constants. These
results do not justify changing those constants, selecting a production
variant by its performance on seen data, or retuning PN coefficients. The next
scientific check is to compare the mapped intrinsic mode phase slopes and
relative phase offsets with NR, especially 33 and 44, after one common 22-based
time/orbital alignment. Evaluate the two existing phase mappings across their
physical joins without fitting independent phase offsets for every mode. This
can distinguish interpolation-shape error from relative-mode phase error;
the carrier and 22 residual must remain explicit controls.

## Evidence and reproduction

The job completed on `len` on 2026-09-28, 01:18:21–01:19:04 UTC, PID 270706.
It evaluated CPU FP64 native modes using Python 3.12.3, NumPy 2.3.5, SciPy
1.16.3, PyCBC 2.11.0, Torch 2.14.0, JAX/JAXlib 0.11.2 and SymPy 1.14.0.
The runner separately verifies the frozen repository against the unchanged
manifests, then records the new native source tree. It does not require the
modified native tree to have the old hash. The frozen protocol and base runner
remain identical. Raw-file checksums, sidecars, fixture identities, PSD and
prior covariance receipts are checked before waveform evaluation.

External artifacts remain in `len:/tmp/diffgw-hm-orbital-20260928/`:

| Artifact / source | SHA-256 |
| --- | --- |
| `hm_orbital_mapping_ablation.py` | `fc9e67aabd89275a6d28710bb8f98196cc6a7bbaad8b3f544e49aa2ccd65264e` |
| `mapping_ablation.json` | `7cebdfd3d4050b5f91cf8839458a20f43f551d98adbb7aebfe465d3c1492dc5f` |
| New native Python source tree | `8ce50a34cc42e4b246b05356d3a3d4c41a2e8343272cfc0afaffc43fdc00702f` |
| Frozen Python source tree | `15d843ade209bbd36a65b8c1b850a415182d4296ab3994dd1dcdd304a9cd446a` |
| Frozen protocol runner | `1cba5bb618c1994cf4851c6dbcf3a2a7c46b95d9ccc3706805432b5e297c6be2` |
| Frozen base runner | `89856b9b94af8cf546620d2f07928016a35f8ddafa4277e9c479b0638d18b2c0` |

The source-tree hash uses the frozen runner's ordered relative-path/content
framing and describes the run snapshot before the final catalog claim metadata.
The unchanged native waveform file hashes are recorded in the
[implementation receipt](imrphenomhm_orbital_reference_20260928.md). Full input identities, all seeds and both backends are retained in
the output; the principal input hashes are:

| Case | Input | SHA-256 |
| --- | --- | --- |
| 2374 | Manifest | `4381ddf5537e328c0708ce9f77152ea985c075302a5d7f1f09e62fd2cb13fccd` |
| 2374 | Conditioned fixture | `e163d466ac0f56f6506d09ce438b4c52cd919fefab7197884038e038f085d0b1` |
| 2374 | Prior covariance output | `a6a6f0a8d372e6a5f9f9df382860a06f23714f69052fd46f7ff5d75ef98d2a1f` |
| 2163 | Manifest | `477083ba0fa8ba6346008d946cd41504abc1111d9f727e82ff8bdbc42ed80b58` |
| 2163 | Conditioned fixture | `bb6cc4dc43f726aca4c9209c76d7ba615167aaa89e9fe06e6bbe841a366ae707` |
| 2163 | Prior covariance output | `1cd7a0299730d93abe6697076c136b4e638292b6df16eca76cfab04b1ca07f36` |

Replay into a new external output path using the recorded native source tree:

```sh
cd /tmp/diffgw-hm-orbital-20260928/repo
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  /tmp/diffgw-hm-orbital-20260928/hm_orbital_mapping_ablation.py \
  --repo "$PWD" --case all \
  --output /tmp/diffgw-hm-orbital-20260928/mapping_ablation_replay.json
```

The hash-pinned helpers and prior covariance outputs are loaded from
`/tmp/diffgw-hm-published-options-20260927/`; frozen source and NR fixtures are
read from `/tmp/diffgw-hm-published-nr-20260927/`. Their identities are retained
in the output and [prior receipt](imrphenomhm_published_options_20260927.md).
Scratch code and the large output are not shipped as maintained library code.
No new holdout was consumed and qualification remains open.
