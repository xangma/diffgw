# IMRPhenomHM six-mode pinned-binary holdout — 27 September 2026

The maintained [`tests/test_imrphenomhm_lal_modes_binary.py`](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomhm_lal_modes_binary.py) compares the **complex, unprojected** `(2,2)`, `(2,1)`, `(3,3)`, `(3,2)`, `(4,4)`, and `(4,3)` HM/D modes from Torch and JAX with the public LALSuite 7.26.15 binary API. All six tests (three systems × two backends), covering all 36 mode/backend/system comparisons, pass the criteria frozen before the holdout systems were run. The test is included in the CI maintained binary-oracle step with `DIFFGW_REQUIRE_LAL_BINARY=1`.

## Source and fixed comparison contract

[London et al., Eqs. (1)–(9)](https://arxiv.org/pdf/1708.00404v4) give the six-mode decomposition and frequency/phase constructions; [García-Quirós et al., Sec. II, Eqs. (2), (4)–(9)](https://arxiv.org/html/2001.10914v2) state the Fourier sign and positive-frequency mode-pair convention used by the native projection specification. The oracle call uses only the documented public [`SimInspiralChooseFDModes` and `SphHarmFrequencySeriesGetMode` API](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_inspiral_8h.html). No LAL source or numeric coefficient was copied into the test.

For this public IMRPhenomHM call, the binary returns all twelve `±m` labels. Its **positive-`m`** complex series have nonzero samples at negative Fourier frequency and zero samples at positive frequency. An exploratory system outside the holdout (`60, 20 M_sun`, spins `0.5, -0.3`, `f_ref=31 Hz`) established the single fixed adapter

\[
 h^{\mathrm{native}}_{\ell m}(f)
 \quad\longleftrightarrow\quad
 (-i)^m\,\overline{h^{\mathrm{LAL}}_{\ell m}(-f)},\qquad f>0.
\]

The conjugation and factor are an **empirical public-API convention inference**, not claimed as a derivation of LAL's internal orbital-phase origin from the paper. They were frozen before any holdout was evaluated. All systems use LAL `phiRef=0`, native `phic=0`, `tc=0`, inclination `0`, aligned spins, default native D carrier, and no per-system or per-mode amplitude, constant-phase, or time alignment. Input masses are solar masses converted with `lal.MSUN_SI`; distances are Mpc converted with `lal.PC_SI`. LAL uses `deltaF=1 Hz`, `f_min=20 Hz`, `f_max=512 Hz`; both native backends and the comparison sample `20–400 Hz` inclusive at `1 Hz` spacing in CPU FP64. The unnormalized complex values retain the absolute distance and mode scale.

The three holdouts were declared in the test before their first binary run:

| System | `(m1,m2)` in solar masses | `(chi1,chi2)` | `f_ref` (Hz) | Distance (Mpc) |
| --- | ---: | ---: | ---: | ---: |
| Moderate mass ratio, mixed spins | `(38,27)` | `(0.15,-0.25)` | 29 | 530 |
| High mass ratio, negative primary spin | `(58,14)` | `(-0.45,0.27)` | 33 | 760 |
| Positive spins | `(70,35)` | `(0.61,0.11)` | 23 | 850 |

For each mode, the predeclared bounds are complex relative `L2 = ||native−reference||₂/||reference||₂ ≤ 10⁻³` over **all 381 bins**, and maximum complex pointwise relative residual `≤ 10⁻²` where the reference amplitude is at least `1%` of its per-mode peak. No residual is phase-aligned or magnitude-normalized before subtraction.

## Observed complex residuals

Each pair below is `relative L2 / maximum supported-bin pointwise residual`; the latter evaluates 209–381 bins depending on system and mode. Values are rounded from the complete log.

| System | Mode | Torch | JAX |
| --- | --- | ---: | ---: |
| Moderate | 22 | `7.72e-14 / 5.04e-13` | `3.27e-14 / 1.81e-13` |
| Moderate | 21 | `3.07e-7 / 8.75e-7` | `3.07e-7 / 8.75e-7` |
| Moderate | 33 | `4.24e-7 / 1.93e-6` | `4.24e-7 / 1.93e-6` |
| Moderate | 32 | `9.91e-7 / 2.75e-6` | `9.91e-7 / 2.75e-6` |
| Moderate | 44 | `1.53e-7 / 6.73e-7` | `1.53e-7 / 6.73e-7` |
| Moderate | 43 | `2.05e-7 / 8.20e-7` | `2.05e-7 / 8.20e-7` |
| High mass ratio | 22 | `6.05e-13 / 3.86e-12` | `3.97e-13 / 2.51e-12` |
| High mass ratio | 21 | `2.56e-6 / 2.34e-5` | `2.56e-6 / 2.34e-5` |
| High mass ratio | 33 | `1.85e-6 / 2.20e-5` | `1.85e-6 / 2.20e-5` |
| High mass ratio | 32 | `8.71e-7 / 4.40e-6` | `8.71e-7 / 4.40e-6` |
| High mass ratio | 44 | `6.70e-7 / 6.25e-6` | `6.70e-7 / 6.25e-6` |
| High mass ratio | 43 | `2.39e-6 / 1.47e-5` | `2.39e-6 / 1.47e-5` |
| Positive spins | 22 | `2.71e-6 / 1.29e-12` | `2.71e-6 / 2.35e-14` |
| Positive spins | 21 | `3.80e-5 / 9.43e-6` | `3.80e-5 / 9.43e-6` |
| Positive spins | 33 | `9.46e-7 / 1.01e-5` | `9.46e-7 / 1.01e-5` |
| Positive spins | 32 | `8.44e-7 / 2.56e-6` | `8.44e-7 / 2.56e-6` |
| Positive spins | 44 | `8.44e-7 / 3.18e-6` | `8.44e-7 / 3.18e-6` |
| Positive spins | 43 | `1.78e-6 / 1.16e-5` | `1.78e-6 / 1.16e-5` |

In the positive-spin system, native 22 is exactly zero in the final 14 bins (`387–400 Hz`) and native 21 is exactly zero in the final 50 bins (`351–400 Hz`), while the LAL binary returns small nonzero tails. Their reference amplitudes at the first omitted bins are respectively `3.51e-6` and `6.35e-5` of the mode peak, below the frozen 1%-peak pointwise support. Those bins **remain included in the full-band L2**, explaining the larger 22/21 L2 values. All other system/mode comparisons have zero such bins. This is a real cutoff difference, not a phase or scale fit; the acceptance bounds were not changed after observing it.

## Reproduce and scope

The run used host `len`, cwd `/home/xangma/tmp/diffgw-hm-sixmode-holdout-20260927`, Python `3.12.3`, LALSuite `7.26.15`, NumPy `2.5.3`, SciPy `1.18.1`, Torch `2.14.0+cpu`, JAX/JAXlib `0.11.2`, and pytest `9.1.1`:

```sh
cd /home/xangma/tmp/diffgw-hm-sixmode-holdout-20260927
env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu DIFFGW_REQUIRE_LAL_BINARY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:tests \
  /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
  tests/test_imrphenomhm_lal_modes_binary.py -ra
```

The full second-run log is `hm_sixmode_binary.log` in that cwd; it reports **6 passed in 16.56 s** and every per-mode residual, support count, and native-zero count. The first run, before adding diagnostic-only output, is retained there as `hm_sixmode_binary_initial.log` (6 passed in 16.99 s). LALSuite is pinned so CI repeats this exact binary oracle; the len package index had DNS errors during a latest-version query, so no claim of package-index currentness is made.

This result supports sampled **implementation agreement** with the public same-model binary for the named HM/D contract. It does not establish the independently sourced fit coefficients, derivatives, all spin/mass/frequency limits, negative-frequency reconstruction, polarization projection under arbitrary orientation, or agreement with numerical relativity. The separate equation and projection receipts cover bounded pieces of those questions.
