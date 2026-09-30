# NRTidal v1 nonspinning merger taper and carrier composition

## Scope and status

Specification **`nrtidalv1-nonspinning-merger-taper/v1`** composes the
[independently checked v1 tidal phase](nrtidalv1-phase.md) with the published
merger-frequency fit and descending Planck taper for zero-spin,
dominant-mode `IMRPhenomD_NRTidal` on CPU FP64. The paper-to-Torch/JAX
factor and the premerger full carrier agree on six declared systems. The
pinned LALSuite 7.26.15 binary has a nonzero-tide taper-boundary discrepancy,
so this specification does **not** qualify its tapered full waveform.

## Inputs, conventions, and options

Component masses are in solar masses for Torch/JAX and in seconds
$M_{\rm sec}=G(m_A+m_B)/c^3$ for the scalar reference. Frequency is in Hz;
$\Lambda_A,\Lambda_B\ge0$, $\chi_A=\chi_B=0$, $f_{\rm ref}=30$ Hz,
$D=100$ Mpc, and inclination is $0.4$ rad. The v1 phase uses the
documented `carrier_rounded=True` realization of the published fit.
The native option is `apply_merger_taper=True`. The binary test uses the
public `SimInspiralChooseFDWaveformSequence` API without phase or time
alignment. Both polarizations have the same tidal multiplicative factor;
the numerical gates use $h_+$.

The six declared $(m_A,m_B;\Lambda_A,\Lambda_B)$ systems are
$(1.35,1.35;400,400)$, $(1.6,1.2;200,800)$, its body-exchanged copy,
$(1.35,1.35;0,0)$, $(1.48,1.22;600,250)$, and
$(1.25,1.25;900,900)$. The merger-window fractions are
$f/f_{\rm mrg}=0.9,1,1.025,1.05,1.1,1.15,1.195,1.2,1.25$.

## Mathematics and components

[Dietrich et al. (2018), Eq. (11)](https://arxiv.org/pdf/1804.02235)
fit the merger angular frequency and explicitly state that the previous
NRTidal signal receives a Planck taper from merger to 1.2 times merger.
The fit uses $\kappa_{\rm eff}^{T}$ in place of $\kappa_2^T$ for the LAL
implementation, as the paragraph before Eq. (11) explains. With
$X_A=\max(m_A,m_B)/(m_A+m_B)$, $X_B=\min(m_A,m_B)/(m_A+m_B)$, and
$\kappa_{\rm eff}^{T}=3\widetilde\Lambda/16$, the printed fit is

\[
 f_{\rm mrg}=\frac{0.3586}{2\pi M_{\rm sec}}\sqrt{\frac{X_B}{X_A}}
 \frac{1+0.03354\kappa_{\rm eff}^{T}
            +0.00004315(\kappa_{\rm eff}^{T})^2}
      {1+0.07542\kappa_{\rm eff}^{T}
            +0.0002236(\kappa_{\rm eff}^{T})^2}.
\]

[Dietrich et al. (2019), Eq. (25) and adjacent text](https://arxiv.org/pdf/1905.06011v2)
confirm that the 1-to-1.2 merger taper is unchanged from the previous
NRTidal implementation. The descending branch is
[McKechan et al. (2010), Eq. (7)](https://arxiv.org/pdf/1003.2939),
which the 2018 paper cites as its Planck-taper source. Setting its endpoints
to the stated frequencies and writing
$z=(f-f_{\rm mrg})/(0.2f_{\rm mrg})$ gives

\[
 W(f)=\begin{cases}
 1,&f\le f_{\rm mrg},\\
 [1+\exp(-1/z-1/(z-1))]^{-1},&f_{\rm mrg}<f<1.2f_{\rm mrg},\\
 0,&f\ge1.2f_{\rm mrg}.
 \end{cases}
\]

The independently specified v1 phase $\psi_T$ gives the paper-derived
tidal factor $F_T(f)=\exp[-i\psi_T(f)]W(f)$, so that the native nonspinning
carrier is $h_+^{\rm BNS}=h_+^{\rm IMRPhenomD}F_T$.

| Component ID/version | Primary source | Scalar reference | Frontends | Evidence |
| --- | --- | --- | --- | --- |
| `nrtidal.shared.merger_frequency.eq11.v1` | Dietrich 2018 Eq. (11) | `nrtidalv2_amplitude.merger_frequency` | `_bns_merger_frequency` | six-case frequency and body-exchange |
| `nrtidal.shared.planck_taper.v1` | Dietrich 2018 after Eq. (11), McKechan 2010 Eq. (7); 2019 Eq. (25) | `nrtidalv2_amplitude.taper` | `_planck_taper_window` | endpoints, monotonicity, native factor |
| `nrtidalv1.composed_tidal_factor.v1` | Dietrich 2017 Eq. (7) plus 2018 Eq. (11) | `nrtidalv1_taper.factor` | `IMRPhenomD_NRTidal` | Torch/JAX complex quotient, pinned binary premerger |

## Implementation mapping

The [v1 scalar composition](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/nrtidalv1_taper.py)
reuses the existing independent merger/taper expressions in
`diffgw_reference.nrtidalv2_amplitude`: these are shared published components,
not NRTidalv2-specific fits. A symbolic test verifies their effective tidal
coupling is exactly the v1 paper coupling. The [math tests](https://github.com/xangma/diffgw/blob/main/tests/test_nrtidalv1_taper_math_reference.py)
check Torch and JAX's complete complex BNS/BBH quotient against $F_T$,
including both window endpoints. The [binary tests](https://github.com/xangma/diffgw/blob/main/tests/test_nrtidalv1_taper_lal_binary.py)
compare the full unaligned $h_+$ through merger, the premerger quotient,
and the zero-tide taper against the named public binary.

## Evidence and limitations

The predeclared merger-frequency relative gate is $10^{-11}$ and the
Torch/JAX complex factor absolute gate is $10^{-10}$. Premerger binary
frequencies are 30–800 Hz in 5 Hz steps, plus $0.9f_{\rm mrg}$ and
$f_{\rm mrg}$; the full $h_+$ relative complex $L_2$ gate is $10^{-8}$.
The binary factor maximum absolute gate is $5\times10^{-5}$. The zero-tide
binary taper has the same $5\times10^{-5}$ absolute gate.

The [27 September 2026 receipt](../validation/nrtidalv1_taper_20260927.md)
records the strict passing tests and a separate, frozen nonzero-tide taper
failure diagnostic. For the five nonzero-tide systems, the pinned binary
amplitude quotient differs from the published window by
$2.77\times10^{-4}$ to $5.36\times10^{-4}$ at the declared fractions.
Those values exceed the $5\times10^{-5}$ boundary gate. No coefficient
was inferred or adjusted from the binary. The premerger success does not
establish agreement within the nonzero-tide taper, nor physical agreement
with independent numerical-relativity data. Spins, self-spin, other modes,
FP32, accelerators, and parameter-domain coverage remain separate.
The full tapered unaligned $h_+$ comparison also exceeds its frozen
relative complex $L_2\le10^{-6}$ gate on both holdouts.

The [convention diagnostic](../validation/nrtidalv1_taper_conventions_20260927.md)
finds the published Planck shape to numerical precision after inferring one
merger frequency from the binary's central taper sample. The inferred
frequencies are $2.5\times10^{-5}$ to $4.9\times10^{-5}$ below the printed
fit for nonzero tides. The paper's printed precision admits this size of
shift, but does not identify the binary's full-precision coefficients.
Substituting $\kappa_2^T$ for $\kappa_{\rm eff}^T$ does not account for the
discrepancy. The inferred frequencies are diagnostic data, not inputs to
the scalar reference or a passing binary gate.

The historical boundary diagnostic measured the failed gates outside the
required strict test selection without qualifying the mismatch. A separate
historical convention diagnostic tested window shape, merger frequency,
and the explicit alternative coupling. Their one-off replay scripts are
not maintained; their methods and numerical conclusions are recorded in
the linked receipts. The maintained boundary gate remains
`tests/test_nrtidalv1_taper_lal_binary.py`.

## Reproduce the checks

On CPU FP64 with pinned LALSuite 7.26.15, run
`DIFFGW_REQUIRE_LAL_BINARY=1 JAX_ENABLE_X64=1 PYTHONPATH=src:tests
python -m pytest -q tests/test_nrtidalv1_phase_math_reference.py
tests/test_nrtidalv1_phase_lal_binary.py
tests/test_nrtidalv1_taper_math_reference.py
tests/test_nrtidalv1_taper_lal_binary.py`.
Strict mode makes missing reference or binary dependencies fatal. The
published merger/taper expression and its binary discrepancy require
separate review before any broader qualification claim.
