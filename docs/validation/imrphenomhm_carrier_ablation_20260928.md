# HM mapped NR22-carrier ablation — 2026-09-28

Replacing only the quadrupole amplitude does not resolve the targeted higher-mode
amplitude errors on shared frequency support. The 2374 (2,1) deficit survives;
the 2163 (4,4) excess improves modestly but remains large. This completes the
carrier check proposed by the [amplitude audit](imrphenomhm_amplitude_audit_20260928.md).
It is a controlled diagnostic on two already-seen cases, not a qualification
run. Production waveforms, frozen gates and earlier failures are unchanged.

## Controlled substitution

Use the previous audit's independently checked factors $B$ and $B_{\rm alt}$,
and its two fixed amplitude maps $g=f_{22}^{A}(Mf)$. Replace

\[
A_{\ell m}(f)=B(Mf,g)|D_{22}(g)|
\quad\longrightarrow\quad
A^{\rm NR22}_{\ell m}(f)=B(Mf,g)|R_{22}(g/M_{\rm sec})|.
\]

Repeat separately with $B_{\rm alt}$. The amplitude equations and sources are
specified in the [preceding receipt](imrphenomhm_amplitude_audit_20260928.md#source-equation-and-controlled-alternative).
Each comparison fixes the factor, frequency map, native phase, QNMs and cutoff.
There is no fitted gain, detector alignment or optimizer. Both Torch and JAX
are checked against the same component construction; their redundant result
columns are omitted. The ablated amplitudes use the Torch phase after that check.

The common mask is $30\leq g/M_{\rm sec}\leq1024$ Hz and nonzero native mode
amplitude. Both original and substituted amplitudes are scored against the same
intrinsic NR mode on that mask. No extrapolation is allowed. Mode 22 is an
identity control. Mass, distance, PSD, signed modes, NR resolutions and
conditioning are inherited unchanged from the frozen 2374/2163 evidence.

For masked PSD norm $\|a\|^2=4\sum |a|^2\Delta f/S_n$, report
$E_A=\||A|-|R_{\ell m}|\|/\|R_{\ell m}\|$ and
$N=\|A\|/\|R_{\ell m}\|$. These are amplitude diagnostics, not detector
mismatches or newly introduced pass thresholds.

## Direct Fourier control

Linear and PCHIP interpolation of the recovered NR22 magnitude provide pilot
values. The reported final result instead evaluates its discrete Fourier
transform directly at every supported mapped frequency:

\[
R_{22}(f)=\frac{s}{8192}\sum_j w_j h_{2,-2}(t_j)
                    e^{-2\pi i f(t_j-t_0)M_{\rm sec}}.
\]

Here $s$ is the frozen amplitude scale, $t_j,t_0$ are in units of $M$, and the
samples use the frozen common interval, cubic interpolation of raw strain,
8192 Hz sampling, and 500M/100M sine-squared start/end ramps. The signed
carrier factor $(-1)^\ell$ is +1 for 22. The same time origin is retained.
This evaluates the same finite discrete transform without frequency
interpolation; it is not a new continuum or conditioning-uncertainty estimate.

The direct calculation reproduces the frozen integer-frequency complex NR22
with relative PSD L2 errors $2.43\times10^{-13}$ and $2.60\times10^{-13}$.
The largest substituted 22 amplitude error is $4.62\times10^{-14}$.
The maximum assembled linear/direct amplitude difference, normalized by the
direct ablation's norm, is $1.85\times10^{-4}$; PCHIP/direct is
$8.75\times10^{-5}$. Interpolation does not explain the residuals below.

All previous fullband native/alternative amplitude scores reproduce within
$2\times10^{-10}$. Native Torch/JAX complex-mode relative L2 is at most
$4.55\times10^{-13}$. Component checks pass, substituted unit phases agree
within $4.97\times10^{-16}$, and active support is unchanged on the common mask.
Hash checks pin the prior amplitude/intrinsic/periodic evidence, helper scripts,
source files, manifests, raw NR files, sidecars, fixtures, PSD and mapped queries.
The final direct reader also checks intrinsic evidence, helper, frozen-source
and PSD identities independently of the main driver.

## Results on identical support

“NR22” below means direct DFT substitution. Each row compares the same mask;
44 values differ from the earlier **fullband** receipt for that reason.

| Case | Map | Mode | Factor | Native $E_A$ | NR22 $E_A$ | Native $N$ | NR22 $N$ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 2374 | London | 21 | baseline | 0.519334 | 0.522957 | 0.529441 | 0.526408 |
| 2374 | London | 21 | alternative | 0.512598 | 0.516364 | 0.504381 | 0.501377 |
| 2374 | London | 44 | baseline | 0.257960 | 0.248861 | 1.238467 | 1.229746 |
| 2374 | London | 44 | alternative | 0.227475 | 0.218625 | 1.211259 | 1.202785 |
| 2374 | Runtime | 21 | baseline | 0.510424 | 0.513796 | 0.554882 | 0.552567 |
| 2374 | Runtime | 21 | alternative | 0.518473 | 0.521829 | 0.497357 | 0.494708 |
| 2374 | Runtime | 44 | baseline | 0.261710 | 0.252536 | 1.244143 | 1.235424 |
| 2374 | Runtime | 44 | alternative | 0.227912 | 0.219052 | 1.212316 | 1.203880 |
| 2163 | London | 21 | baseline | 0.155690 | 0.163077 | 0.933938 | 0.919290 |
| 2163 | London | 21 | alternative | 0.220269 | 0.220260 | 0.988100 | 0.972042 |
| 2163 | London | 44 | baseline | 0.469600 | 0.449649 | 1.405811 | 1.387275 |
| 2163 | London | 44 | alternative | 0.440594 | 0.421262 | 1.379177 | 1.361179 |
| 2163 | Runtime | 21 | baseline | 0.149856 | 0.156713 | 0.943669 | 0.929580 |
| 2163 | Runtime | 21 | alternative | 0.224543 | 0.224235 | 0.983037 | 0.967431 |
| 2163 | Runtime | 44 | baseline | 0.471253 | 0.451354 | 1.408853 | 1.390404 |
| 2163 | Runtime | 44 | alternative | 0.440179 | 0.420911 | 1.378792 | 1.360890 |

Excluded powers are fractions of each original fullband mode's own PSD power.
The native/alternative columns account for model power separately from NR.

| Case | Map | Mode | Retained bins | Excluded NR power | Excluded native power | Excluded alternative power |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2374 | London | 21 | 366 | 2.410309e-11 | 0 | 0 |
| 2374 | London | 44 | 517 | 0.23549743 | 0.19676028 | 0.20387658 |
| 2374 | Runtime | 21 | 366 | 2.410309e-11 | 0 | 0 |
| 2374 | Runtime | 44 | 517 | 0.23549743 | 0.19531894 | 0.20359342 |
| 2163 | London | 21 | 338 | 5.6332122e-10 | 0 | 0 |
| 2163 | London | 44 | 574 | 0.17256342 | 0.11762496 | 0.12165394 |
| 2163 | Runtime | 21 | 338 | 5.6332122e-10 | 0 | 0 |
| 2163 | Runtime | 44 | 574 | 0.17256342 | 0.11717711 | 0.12171358 |

For 44, the mapped NR22 band excludes the 30–59 Hz target bins; the mask also
excludes the native zero tail. Thus **23.55% / 17.26% of NR44 power is outside
this comparison**. The result does not establish a fullband NR22 ablation.
For 21, excluded NR power is below $5.64\times10^{-10}$ and excluded model
power is zero, so the masked result is effectively fullband for these cases.

The fixed-band London baseline comparison locates the retained error. Errors
use each band's own retained NR norm; the last column uses fullband NR power.
Empty bands are omitted, not interpreted as zero error.

| Case | Mode | Target band (Hz) | Native $E_A$ | NR22 $E_A$ | Retained NR power / fullband |
| --- | --- | --- | ---: | ---: | ---: |
| 2374 | 21 | 30-60 | 0.254447 | 0.253821 | 0.21702066 |
| 2374 | 21 | 60-100 | 0.483807 | 0.490911 | 0.38782871 |
| 2374 | 21 | 100-200 | 0.646005 | 0.648281 | 0.3949686 |
| 2374 | 21 | 200-400 | 0.520741 | 0.499972 | 0.00018202723 |
| 2374 | 44 | 60-100 | 0.132430 | 0.124024 | 0.23017583 |
| 2374 | 44 | 100-200 | 0.196806 | 0.193695 | 0.29873346 |
| 2374 | 44 | 200-400 | 0.386892 | 0.371983 | 0.23558578 |
| 2374 | 44 | 400-1025 | 0.382905 | 0.277132 | 7.504107e-06 |
| 2163 | 21 | 30-60 | 0.131712 | 0.138481 | 0.31509027 |
| 2163 | 21 | 60-100 | 0.173174 | 0.189510 | 0.3363169 |
| 2163 | 21 | 100-200 | 0.156876 | 0.154713 | 0.34806144 |
| 2163 | 21 | 200-400 | 0.478451 | 0.516751 | 0.00053139275 |
| 2163 | 44 | 60-100 | 0.162794 | 0.159798 | 0.2168764 |
| 2163 | 44 | 100-200 | 0.254271 | 0.244875 | 0.3152647 |
| 2163 | 44 | 200-400 | 0.686946 | 0.653367 | 0.27549367 |
| 2163 | 44 | 400-1025 | 1.153226 | 1.129159 | 0.019801811 |

## Interpretation and remaining work

The native mapped quadrupole magnitude is not sufficient to explain either
targeted discrepancy. The 2374 21 error slightly increases with NR22, for
both factors and maps. For 2163 44, the London baseline error falls from
0.469600 to 0.449649 and its norm from 1.405811 to 1.387275 on common support;
the alternative still leaves error 0.421262 and norm 1.361179. Runtime gives
the same qualitative result. Errors remain substantial in the bands carrying
most retained NR power, rather than only in weak tails.

Together with the prior independent factor checks and conditioning audit,
this localizes the remaining amplitude discrepancy to the **combined
higher-mode scaling/map approximation relative to conditioned NR** in this
scope. It does not separate a factor error from a map error, prove global
model inadequacy, or exclude every NR systematic. The unweighted
`carrier_only` statistics in the external JSON must not replace the assembled,
beta-weighted results above when making this inference.

There is no evidence here for refitting the D22 carrier or changing production
coefficients. Close this carrier hypothesis for these two seen cases and
retain the scoped mathematical passes and physical failures. The separate
32/43 frequency-dependent phase/mixing discrepancy is the next research
question: compare published spherical/spheroidal and phase-map relations,
with an explicit source-derived specification before any implementation.
Any physically improved variant requires its own declared model scope and a
prospectively frozen test on untouched data. Reusing these cases or weakening
their gates would not qualify the family.

## Execution and replay

All computation ran on `len` in `/tmp/diffgw-hm-carrier-ablation-20260928`.
The `repo/` source archive is revision
`48302b01111ff95b134441bf3780640c5c147bb3`; native source hashes match the
preceding audit. Pilot PID 3103267 completed 16:00:45–16:01:15 UTC;
final PID 3143490 completed 16:02:13–16:02:41 UTC on 2026-09-28.
Direct-reader PID 3138091 completed 16:02:02–16:02:11 UTC. All final receipts
say `completed`; `qualification_promoted=false`. No scientific job remains.

An initial direct calculation also completed successfully. Review then added
explicit intrinsic-prior, helper, frozen-source and PSD hash assertions;
the final run uses the new `direct-pinned.npz`. The earlier `direct.npz` is
retained as exploratory evidence and is not the input to the final result.
The two direct runs reproduce the same integer-bin controls.

Backend environment: `/tmp/diffgw-waveform-library-1315f6e/venv-latest`,
Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch 2.14.0,
JAX/jaxlib 0.11.2, SymPy 1.14.0. Raw-reader environment:
`/tmp/diffgw-hm-sxs-2265-20260927/.venv`, Python 3.12.3, NumPy 2.5.3,
SciPy 1.18.1, SXS 2026.0.6. Packages were retained for direct comparison.

Replay requires the preceding amplitude driver and unchanged intrinsic/periodic
helpers, frozen source/data directories and PSD arrays documented by the
[amplitude receipt](imrphenomhm_amplitude_audit_20260928.md#execution-and-reproducibility).
Use fresh external output paths (the following names illustrate a new run):

```sh
cd /tmp/diffgw-hm-carrier-ablation-20260928
export PYTHONPATH="$PWD/repo/src" JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
BACKEND=/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python
READER=/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python
REV=48302b01111ff95b134441bf3780640c5c147bb3
"$BACKEND" hm_carrier_ablation.py --repo "$PWD/repo" \
  --source-revision "$REV" --output replay-pilot.json \
  --export-queries replay-queries.npz
"$READER" hm_carrier_direct_dft.py replay-queries.npz replay-direct.npz
"$BACKEND" hm_carrier_ablation.py --repo "$PWD/repo" \
  --source-revision "$REV" --output replay-final.json \
  --direct-carrier replay-direct.npz --queries-reference replay-queries.npz
```

| Artifact | SHA-256 |
| --- | --- |
| `carrier-final.json` | `3d38b4e203d362762c69525ad7359e498e691f8b6bccada92a1b3d773c7e421c` |
| `carrier-pilot.json` | `065ed6027a59248c781e52611dfb330e1c1c23cfb49c9556576ac30d41e616d1` |
| `queries.npz` | `83086097a20b6179f31c98a1a8c9cf6606988603b52dea31519b23cb4049d815` |
| `direct-pinned.npz` | `9b32cde9e1ae921e020451637beca84090c0f8445fbfa7a8a08373f3e896ff19` |
| `hm_carrier_ablation.py` | `c02e285ab254b3ac5a8c84ca93da7c3c72395cd37570e595d948151ef945a8db` |
| `hm_carrier_direct_dft.py` | `da7688d7d8b1f5b3d30367f5529a33b48dbb4adc78c75e8f3749675edc946dbf` |
| Prior amplitude JSON | `11e236d9a39ed64022b0d9824ca1c8b44e77057baa9d8ed786fbef6d9a7d1daa` |

Logs are `pilot.log`, `direct-pinned.log` and `final.log`. Complete per-band,
interpolation and component results remain outside Git. This curated receipt
preserves findings and provenance, not permanent availability of scratch files.
No external implementation source was inspected. This documentation-only
change passed both generated-document consistency checks and strict MkDocs
on len, with the updated documentation overlaid on that source archive.
No production regression suite is added.
