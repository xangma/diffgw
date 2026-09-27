# IMRPhenomHM plus/cross pinned-binary holdout — 27 September 2026

## Frozen comparison contract

The maintained `tests/test_imrphenomhm_lal_polarizations_binary.py` compares
complex plus and cross from public Torch/JAX `IMRPhenomHM` with LALSuite
**7.26.15** `SimInspiralChooseFDWaveform` for the same aligned-spin D-carrier
model. The test, cases, and bounds in this section were committed **before**
running these holdouts. Only the public LAL binary is queried; no LAL source or
coefficient is copied.

[London et al., Eq. (1)](https://arxiv.org/pdf/1708.00404v4) state the
spin-weighted-harmonic polarization sum.
[García-Quirós et al., Sec. II](https://arxiv.org/html/2001.10914v2) fix
the positive-frequency mode-pair convention. The separate
[projection specification](../math/imrphenomhm-projection.md) derives and checks
the six angular coefficients. The public LAL
[waveform API](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_inspiral_8h.html)
names the inclination and reference phase arguments.

One **non-holdout** exploratory system (`60,20 M_sun`, spins `0.5,-0.3`,
`f_ref=31 Hz`) established that LAL `phiRef` corresponds to native
`phi0 + phic` for this API/model; the opposite sign had order-one complex
residuals. This fixed convention was selected before the new cases below.
LAL has no `tc` argument in this call, so its output is multiplied by the
analytic Fourier shift `exp(-2πiftc)`; no shift is estimated from the data.
No per-case, polarization-specific, phase, time, or amplitude alignment is
allowed. Native settings use default D carrier and all six modes.

| Unseen system | `(m1,m2)` $M_\odot$ | `(chi1,chi2)` | `f_ref` Hz | Distance Mpc | Inclination | `phi0`, `phic` rad | `tc` s |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| Mixed generic | `(42,23)` | `(0.31,-0.18)` | 31 | 610 | 0.73 | `(0.21,-0.08)` | 0.011 |
| Asymmetric generic | `(55,17)` | `(-0.32,0.41)` | 27 | 930 | 1.29 | `(-0.28,0.37)` | -0.007 |
| Positive-spin backside | `(78,31)` | `(0.52,0.07)` | 37 | 740 | 2.15 | `(0.43,-0.19)` | 0.019 |

LAL uses `deltaF=1 Hz`, `f_min=20 Hz`, `f_max=512 Hz`; both native frontends
sample **20–400 Hz inclusive**, 381 bins, CPU FP64. The unnormalized complex
plus/cross retain absolute distance and polarization scale. For each
polarization separately, the predeclared bounds are complex relative
`L2 = ||native−binary||₂/||binary||₂ ≤ 10⁻³` across all bins and maximum
pointwise complex relative residual `≤ 10⁻²` where the binary amplitude is
at least 1% of that polarization's peak. No overlap maximization is used.

## Result and scope

Pending the first run on `len`. This is a sampled same-model public-binary
polarization gate, distinct from the existing
[six-mode binary holdout](imrphenomhm_lal_six_modes_20260927.md) and
[source projection gate](imrphenomhm_projection_20260927.md). A pass will not
establish exact paper fit provenance, derivatives, all parameter/frequency
limits, negative-frequency reconstruction, or physical accuracy against NR.
