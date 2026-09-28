# HM amplitude factors and conditioning audit — 2026-09-28

The [intrinsic-mode diagnostic](imrphenomhm_intrinsic_projection_20260928.md)
found a large 2374 (2,1) deficit and 2163 (4,4) excess. This follow-up checks
the amplitude construction independently and tests sensitivity to NR
conditioning on those same seen cases. The checked factors agree with both
frontends; the published alternative factor does not resolve these discrepancies.
No production waveform, frozen fixture, threshold or qualification claim changes.

## Source equation and controlled alternative

Let $x=Mf$, $g=f_{22}^{A}(x)$, and
$H_{\ell m}(y)=V_m(y)^{-7/2}\hat H_{\ell m}(V_m(y))$, with
$V_m(y)=(2\pi y/m)^{1/3}$. The common dimensional SPA prefactor cancels.
[Mishra et al., Eqs. (11), (12)](https://arxiv.org/html/1601.05588v1)
supply the envelopes; existing references specify the retained
[21](../math/imrphenomhm-amplitude-ratio-21.md),
[33](../math/imrphenomhm-amplitude-ratio.md),
[32](../math/imrphenomhm-amplitude-ratio-32.md),
[44](../math/imrphenomhm-amplitude-ratio-44.md) and
[43](../math/imrphenomhm-amplitude-ratio-43.md) terms.
In particular, the 21 coefficients agree with Eq. (12b), and the leading
44 envelope is proportional to $(1-3\eta)V_4^{-3/2}$, including the SPA power.

[London et al., Eq. (4) and its following paragraph](https://arxiv.org/pdf/1708.00404v4)
give two distinct amplitude prescriptions:

\[
B=\left|\frac{H_{\ell m}(g)}{H_{22}(g)}
             \frac{H_{\ell m}(x)}{H_{\ell m}(2x/m)}\right|,
\qquad B_{\rm alt}=\left|\frac{H_{\ell m}(x)}{H_{22}(g)}\right|.
\]

The scratch alternative multiplies each existing complex mode by

\[
\frac{B_{\rm alt}}B=
\left|\frac{H_{\ell m}(2x/m)}{H_{\ell m}(g)}\right|.
\]

The positive ratio preserves phase and support. Symbolic checks establish the
ratio identity, equality when $g=2x/m$, and unchanged 22. All denominators
are nonzero and mapped frequencies positive on these grids. This is not a
general prescription at amplitude zeros or symmetric odd-mode limits.

The Eq. (5) amplitude map, 22 carrier, QNMs, cutoffs and mode phases remain
fixed within each comparison. Both `london_2018` and `phenomd_runtime` maps
are tested separately; only the former uses the paper's amplitude transition.
No detector alignment, amplitude normalization, optimizer or coefficient fit
enters this comparison. No external implementation source was inspected.

## Factor checks

The audit evaluates restricted symbolic envelopes on all 995 frequency bins,
30–1024 Hz at 1 Hz spacing, for all six modes, both cases and both maps.
Separate 50-digit scalar evaluations check the map, factor, carrier and
composed amplitude at fixed frequencies and near mapped joins. The scalar
composition independently derives the published carrier/remnant and uses the
declared QNM data. This reuses the maintained reference components; it is
not an independent rederivation of their calibration assets.

| Check | Maximum error |
| --- | ---: |
| Full symbolic SPA envelopes versus native factors, relative | $9.94\times10^{-16}$ |
| Eq. (4) amplitude factor, relative | $1.81\times10^{-15}$ |
| Scalar map frequency, absolute in $Mf$ | $5.56\times10^{-17}$ |
| Scalar QNM frequency, absolute in $Mf$ | $2.78\times10^{-17}$ |
| Sampled independent 22 carrier amplitude, relative | $1.24\times10^{-14}$ |
| Sampled independent composed amplitude, relative | $1.36\times10^{-14}$ |
| Native amplitude recomposed from its factors, relative | $2.09\times10^{-15}$ |
| Torch/JAX complex modes, relative L2 | $6.28\times10^{-13}$ |

Relative factor errors use maximum absolute difference divided by maximum
reference magnitude over the checked bins, not the maximum pointwise relative
error. Phase-convention amplitude agreement is within $4.12\times10^{-16}$;
the legacy and SPA conventions consequently share the amplitude table below.
All assertions passed. These sampled checks found no transcription discrepancy;
they do not establish physical accuracy or complete parameter-domain coverage.

## NR amplitude results

Use the prior independently recovered intrinsic NR carriers, fixed masses,
distance, signed-mode conventions and aLIGO PSD. Define
$E_A=\|\,|P|-|R|\,\|/\|R\|$ with
$\|a\|^2=4\sum_f |a(f)|^2\Delta f/S_n(f)$.
This measures amplitude shape and scale without phase alignment. It is not
a detector mismatch or an existing qualification threshold.

| Case | Mode | London baseline $E_A$ | London alternative $E_A$ | Runtime baseline $E_A$ | Runtime alternative $E_A$ |
| --- | --- | ---: | ---: | ---: | ---: |
| 2374 | 21 | 0.5193 | 0.5126 | 0.5104 | 0.5185 |
| 2374 | 33 | 0.0866 | 0.0667 | 0.0933 | 0.0679 |
| 2374 | 32 | 0.3764 | 0.2504 | 0.3761 | 0.2482 |
| 2374 | 44 | 0.2312 | 0.2053 | 0.2344 | 0.2056 |
| 2374 | 43 | 0.4185 | 0.3830 | 0.4186 | 0.3832 |
| 2163 | 21 | 0.1557 | 0.2203 | 0.1499 | 0.2245 |
| 2163 | 33 | 0.0928 | 0.0795 | 0.0964 | 0.0794 |
| 2163 | 32 | 0.6374 | 0.5552 | 0.6385 | 0.5553 |
| 2163 | 44 | 0.4303 | 0.4041 | 0.4318 | 0.4037 |
| 2163 | 43 | 0.4339 | 0.4465 | 0.4335 | 0.4481 |

22 is unchanged: $E_A=0.009113$ and $0.013835$ for 2374 and 2163.
For London mapping, the 2374 21 norm ratio changes from 0.5294 to 0.5044;
the 2163 44 ratio changes from 1.3613 to 1.3386. Neither discrepancy is
removed. The 2163 21 ratio improves from 0.9339 to 0.9881 even though its
amplitude error rises from 0.1557 to 0.2203. A norm close to one can hide an
incorrect frequency dependence. Runtime mapping gives the same qualitative
outcome for these three comparisons.

The alternative improves several other mode amplitudes, but worsens 2163 21
and 43 for both mappings. It cannot justify a general replacement or a
qualification claim. Phase residuals are untouched; total projected overlap
was deliberately not reoptimized for this amplitude-only diagnostic.

## Independent conditioning sensitivity

The raw-reader calculation decodes the higher-resolution NR files and
reconstructs the 22, 21 and 44 carriers using the frozen time interval and
22-peak clock. It first reproduces the frozen carriers to below $10^{-10}$
relative complex L2. It then changes one setting at a time:

- Start taper: 250M or 1000M instead of 500M.
- End taper: 50M or 200M instead of 100M.
- Sampling: 16384 Hz instead of 8192 Hz.
- Interpolation: real/imaginary PCHIP instead of cubic splines.

All other settings, raw files, PSD and comparison frequencies stay fixed.
The taper is the same product of squared-sine ramps used by the frozen runner.
There is no extrapolation or clock realignment. The baseline end taper begins
45.46M after the 22 peak in 2374 and 50.40M after it in 2163. The 200M
stress test starts at **−54.54M and −49.60M**, respectively, attenuating merger.
Its large effect must not be presented as ordinary late-ringdown uncertainty.

Let $\delta=\|\,|R'|-|R|\,\|/\|R\|$ and
$\rho=\|R'\|/\|R\|$. The triangle inequalities give

\[
\frac{\max(0,E_A-\delta)}{\rho}
\le \frac{\|\,|P|-|R'|\,\|}{\|R'\|}
\le \frac{E_A+\delta}{\rho}.
\]

These bounds require no new model evaluation. Across the five variations
other than the merger-attenuating 200M stress test:

| Case | Mode | Maximum $\delta$ | Enclosing London baseline $E_A$ bounds |
| --- | --- | ---: | --- |
| 2374 | 22 | 0.000371 | [0.00874, 0.00949] |
| 2374 | 21 | 0.000800 | [0.51852, 0.52013] |
| 2374 | 44 | 0.000883 | [0.23027, 0.23205] |
| 2163 | 22 | 0.000488 | [0.01334, 0.01433] |
| 2163 | 21 | 0.000724 | [0.15496, 0.15642] |
| 2163 | 44 | 0.000725 | [0.42957, 0.43103] |

Norm changes are below $8.05\times10^{-5}$; complex spectral changes can
reach 0.00232, so the sub-0.09% statement applies only to amplitudes.
The targeted deficits/excesses
therefore survive these conditioning variations. For completeness, the 200M
stress-test deltas for (22,21,44) are (0.1262, 0.2219, 0.1226) for 2374 and
(0.1040, 0.1680, 0.0958) for 2163. Even this test leaves lower London error
bounds of 0.3759 for 2374 21 and 0.3530 for 2163 44. Its changed waveform
is not substituted into any historical score.

This is a bounded sensitivity study, not a full NR systematic-error budget.
It does not vary extrapolation order, extraction method, simulation physics
or resolution. The historical resolution screens are reused. 32/43 conditioning
has not been tested here; low-power frequency bands can have much larger
relative changes than the integrated values above.

## Decision and remaining work

Retain the current declared amplitude prescription. The independently checked
formula and these conditioning variations do not explain away the two targeted
discrepancies. The results support investigating source-model approximation
error, without proving it is the unique cause. The paper's alternative remains
a scratch diagnostic, not a new production option or fitted correction.

The next discriminating amplitude check is to replace only the mapped
$|D_{22}(g)|$ carrier by recovered $|NR_{22}(g)|$ for 21/44, keeping each
factor and map fixed. It must compare identical mapped-frequency support,
report excluded modal power, and avoid extrapolation or a fitted gain. The
small integrated 22 error does not bound its error under the different
frequency weighting of a higher mode. This separates mapped-carrier error
from the combined higher-mode factor/map approximation.

The frequency-dependent 32/43 phase residual remains separate: a source-backed
spherical/spheroidal and phase-map comparison is needed before attributing it
to mixing. If a different model is pursued to
improve physical accuracy, it needs a separate mathematical specification and
prospectively frozen validation on untouched data. Repeated changes on these
seen cases cannot close qualification. All prior failed gates remain visible.

## Execution and reproducibility

Both jobs completed on len in `/tmp/diffgw-hm-amplitude-audit-20260928`.
The amplitude job used source revision
`1db24fb1417e7b4f8a84de53498ec2d039496e00`, archived in `repo/`, with unchanged
native source hashes from the prior intrinsic audit. PID 4126673 ran from
14:02:45 to 14:03:44 UTC on 2026-09-28; conditioning PID 4046395 ran from
13:59:48 to 13:59:53 UTC. Both receipts say `completed` and
`qualification_promoted=false`; no scientific job remains active.

The amplitude environment was
`/tmp/diffgw-waveform-library-1315f6e/venv-latest`: Python 3.12.3,
NumPy 2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch 2.14.0, JAX/jaxlib 0.11.2,
SymPy 1.14.0. The raw-reader environment was
`/tmp/diffgw-hm-sxs-2265-20260927/.venv`: Python 3.12.3, NumPy 2.5.3,
SciPy 1.18.1, SXS 2026.0.6. Existing environments were retained for comparison;
no packages changed.

Replay requires the prior intrinsic driver and its three unchanged helpers,
the frozen source/data directories, and `psd-2374.npy`/`psd-2163.npy` extracted
with the frozen runner. The scripts verify PSD content hashes, manifests,
raw files, sidecars, fixtures and prior evidence. Use fresh output paths:

```sh
cd /tmp/diffgw-hm-amplitude-audit-20260928
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  hm_amplitude_audit.py --repo "$PWD/repo" \
  --source-revision 1db24fb1417e7b4f8a84de53498ec2d039496e00 \
  --output /tmp/new-hm-amplitude.json
/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  hm_amplitude_conditioning.py /tmp/new-hm-conditioning.json
```

| Artifact | SHA-256 |
| --- | --- |
| `amplitude.json` | `11e236d9a39ed64022b0d9824ca1c8b44e77057baa9d8ed786fbef6d9a7d1daa` |
| `hm_amplitude_audit.py` | `48959e61b42af601a3ad14d3062df3318802b24fad521805c60a7725292da567` |
| `conditioning.json` | `ca6a48aaedd7d948ba80bf915f13073f38d68f95bbbd59c14126ca344c6ab2f0` |
| `hm_amplitude_conditioning.py` | `5158a05290aba8d7770301b1c10d393eaf69449c380857f5f908c41d55d0ac51` |
| Prior intrinsic JSON | `12114e33b00c2f68b18002bd0710db5ab4b3789652d8a484c7263ca340aee85f` |

Logs are `amplitude.log` and `conditioning.log`; complete per-band and
component results remain in the external JSON artifacts. This maintained
receipt preserves the derivation and findings, not permanent availability of
the scratch files. No numerical regression suite is required by this
documentation-only change. The first generated-library check imported the
environment's older installed catalog; it was rerun with the archived
checkout's `src` on `PYTHONPATH`. Both generated-document checks and strict
MkDocs passed on len against that checkout.
