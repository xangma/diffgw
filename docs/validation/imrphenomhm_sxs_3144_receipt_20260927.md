# IMRPhenomHM SXS:BBH:3144 prospective qualification — green receipt

**The precommitted one-case qualification passes.** Both available SXS
resolutions pass the two-view plus/cross screen under one shared orbital
phase and arrival shift. Native Torch and JAX IMRPhenomHM then pass the
separate match and norm-ratio gates at both fixed source views. The
[protocol](imrphenomhm_sxs_3144_prospective_protocol_20260927.md) was
committed as `3b3d50d` before any 3144 strain was retrieved or decoded.
Its conditioning, optimizer, modes, and thresholds were unchanged after
inspection. The prior [2265](imrphenomhm_sxs_2265_receipt_20260927.md),
[3631](imrphenomhm_sxs_3631_receipt_20260927.md),
[3619](imrphenomhm_sxs_3619_projection_receipt_20260927.md), and
[3630](imrphenomhm_sxs_3630_projection_receipt_20260927.md) red results
and their gates remain unchanged. The [complete numeric metrics](imrphenomhm_sxs_3144_metrics_20260927.json)
include every alignment candidate, optimizer seed, norm, mode, and diagnostic.

## Source identity and conditioning

The metadata-only selection used the official SXS v3.0.0 catalog JSON
(32,618,880 bytes, SHA-256
`132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6`;
4,170 entries), with the exact filter and sort in the protocol. The first
eligible case after excluding the four earlier cases was SXS:BBH:3144v3.0;
the remaining eligible IDs in order were 2374, 2499, 2164, and 2498.
Its reference mass ratio is `4.499621846563142`, with negligible but
nonzero NR spin vectors. All six files came from the
[official SXS:BBH:3144v3.0 record](https://doi.org/10.26138/SXS:BBH:3144v3.0).
Each published byte size and MD5 in the protocol was verified **before**
decoding. The local SHA-256 values, in protocol order, are:

| Official file | SHA-256 |
| --- | --- |
| `Lev3:Strain_N2.h5` | `59ce19bbed32e257c0bcb307ada7adf5b75c297ffc8d3ae4cfe6a7762eb4440a` |
| `Lev3:Strain_N2.json` | `d7a74026b07f0dff57be6f9a437e52f7610847773342fcb5fd8531c4da3da95f` |
| `Lev3:metadata.json` | `16360fe120d177b1daefba42387d666a552aa03332e0eebd2a4c9b2140ba303c` |
| `Lev2:Strain_N2.h5` | `2c1dac19a419aceda36af366516a9807b6f1e2880da77f10ecb227d6359b9787` |
| `Lev2:Strain_N2.json` | `f0e31fcfca7ab124f8792aca295dfa5b50375979d6119d64adc2edee71b99d1d` |
| `Lev2:metadata.json` | `9aa7fdac4d35f6b52066e8bdc3aa23a267ab36c9014ab57958ba34a660d9106d` |

The public SXS loader reported inertial, spin-weight `-2`, `h` strain with
`m` and radius scaled out, and every signed-`m` mode for `2≤ell≤5`,
including `m=0`. The sidecars record extrapolation, center-of-mass
translation and boost, memory addition, and truncation; no modification
was reversed. Memory integration starts at `1580.5 M` for Lev3 and
`692 M` for Lev2. The common interval was `1580.5–10092.011834583338 M`;
the 8192 Hz grid contained 34,344 samples and its Lev3 `|h_22|` peak set
time zero at `9933.490303373206 M`. The nominal time conversion was
`0.0004925490947641267 s/M`; the loader's scaled-out strain was
multiplied by `9.570831683175683e-21` for `100 M_sun` at `500 Mpc`.
No amplitude scale was fitted. The external fixture SHA-256 is
`cf127c18a2211f0994ba0a03d406c360b8230ef27609920b5c14d11f49745cf2`.

The direct Riemann plus/cross spectra and independently computed mode
Fourier sums agree in every resolution/view/channel to relative maximum
errors `3.46e-13–5.54e-13`, below the frozen `1e-10` tolerance. A synthetic
two-channel shift of `+100/8192 s` was recovered exactly, checking the
inverse-FFT sign. Direct inner products at all reported optima agree with
FFT matches within `3.33e-16`. The separately generated
[PyCBC design PSD](https://pycbc.org/pycbc/latest/html/psd.html) has
comparison-bin SHA-256
`0d20d2092bfaeb6073b0d656f4a3eb495bc3adbeb350ef279420cb4ffd0d67ea`.

## Frozen gates

The **one** Lev2-to-Lev3 alignment was `phi_NR=5.743179471018169 rad`,
`tau_NR=-0.014892578125 s`, selected from the 2048-point grid and the
eight highest of 14 periodic local phase maxima. Its combined two-view
network match was `0.999939629340` with Lev2/Lev3 norm ratio
`0.999601745405`. This same phase and shift gave:

| Fixed `(iota,psi_s)` | Lev2/Lev3 match, floor `0.995` | Norm ratio, bounds `0.95–1.05` |
| --- | ---: | ---: |
| `(0.7,0.0)` | `0.999944177938` | `0.999708145957` |
| `(1.2,0.4)` | `0.999926931964` | `0.999303375647` |

Each native model comparison held mass, spins, distance, inclination,
source polarization, and the six mode carriers fixed. It optimized only
template orbital phase, polarization, and the shared discrete arrival
shift. Each row below passed match `≥0.980` and norm ratio in
`[0.85,1.15]`; all four seeds `17,23,41,67` passed the `1e-5`
optimizer-agreement check.

| Backend; view | Match | Norm ratio | Orbital phase (rad) | Polarization (rad) | Shift (s) | Seed spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch; `(0.7,0.0)` | `0.992168374557` | `0.995166487244` | `5.296249409445` | `1.022772935771` | `0.0062255859375` | `7.92e-12` |
| JAX; `(0.7,0.0)` | `0.992168374557` | `0.995166487214` | `5.296249407960` | `1.022772935466` | `0.0062255859375` | `7.92e-12` |
| Torch; `(1.2,0.4)` | `0.982689246447` | `1.128589480708` | `5.253069413264` | `1.287188873549` | `0.0062255859375` | `2.70e-11` |
| JAX; `(1.2,0.4)` | `0.982689246447` | `1.128589480708` | `5.253069413264` | `1.287188873549` | `0.0062255859375` | `2.70e-11` |

The full NR comparison used 32 modes, all `2≤ell≤5` and `-ell≤m≤ell`.
The model used positive-`m` carriers `(2,2), (2,1), (3,3), (3,2),
(4,4), (4,3)` and their symmetry partners. At the two views, the norm
of the NR modes outside those pairs was respectively `0.026614` and
`0.074496` of the full NR network norm. The six-pair NR subset, without
new alignment, matched full NR at `0.999646` and `0.997231`. Individual
mode norms are coherent contributions and do not add in quadrature to
the full norm; the full list is in the metrics JSON. For example, the
`(5,0)` and `(4,0)` network norm fractions were `1.34e-5` and `8.83e-5`.
These diagnostics neither remove modes nor alter the primary gates.

This is an **operational** pass for one nearly nonspinning `q≈4.5` case,
`100 M_sun`, the two stated views, 30–1024 Hz, this PSD and conditioning.
The [London et al. comparison](https://arxiv.org/html/1708.00404v4)
motivates optimizing arrival time, template polarization, and orbital
phase at fixed intrinsic parameters; its typical `>0.99` population
statement is not a result of this one-case test. The
[SXS 2019 Appendix B](https://arxiv.org/html/1904.04831#S.app2)
motivates a common time/azimuth shift and plus/cross network, while this
fixed taper and band are protocol choices. The resolution screen does
not establish continuum NR accuracy, and the pass does not imply
all-sky, other-mass, spin, or independent-implementation performance.

## Reproduction and checks

Host `len`; cwd `/tmp/diffgw-hm-sxs-3144-20260927/repo`; all raw and
derived strain in `/tmp/diffgw-hm-sxs-3144-20260927/data`, outside Git.
The final runner SHA-256 was
`89856b9b94af8cf546620d2f07928016a35f8ddafa4277e9c479b0638d18b2c0`.
Evaluation requires this exact preparation-script hash as well as the
raw-source and fixture hashes. The complete tracked metrics JSON SHA-256
is `cfd71dc820ab3ff6210de1591de1ffb35baf4cc63ecfe8e309e6eea5fe751dee`.
The exact final commands from that cwd were:

```sh
/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  -m scripts.run_imrphenomhm_sxs_3144 \
  --data-dir /tmp/diffgw-hm-sxs-3144-20260927/data \
  --prepare-only --source-revision 3b3d50d
env PYTHONPATH=.:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  -m scripts.run_imrphenomhm_sxs_3144 \
  --data-dir /tmp/diffgw-hm-sxs-3144-20260927/data \
  --evaluate-only --source-revision 3b3d50d
```

The preparation environment was Python 3.12.3, NumPy 2.5.3, SciPy
1.18.1, SXS 2026.0.6, and SymPy 1.14.0. The evaluation environment
was Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch
2.14.0, JAX/JAXLIB 0.11.2, and SymPy 1.14.0. These were the latest
available compatible versions on `len` for their respective stages.
Preparation launcher PID `3155412` completed; log
`/tmp/diffgw-hm-sxs-3144-20260927/prepare-final.log`. Evaluation
launcher PID `3170169` completed with `qualified: true`; log
`/tmp/diffgw-hm-sxs-3144-20260927/evaluate-final.log`. No job remains
active, hence no next check or stop command applies; on a rerun, stop
its active PID with `kill <PID>` on `len`. Data-free tests covering
two-channel shift, direct network norm/inner product, signed-`m`
time-domain azimuth rotation, and preparation-script hash enforcement
passed `4/4`. Ruff and `python -m mkdocs build --strict` passed on `len`;
the latter log is `/tmp/diffgw-hm-sxs-3144-20260927/mkdocs-strict.log`.
