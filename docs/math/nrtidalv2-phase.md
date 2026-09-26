# NRTidalv2 nonspinning tidal phase

## Scope and identity

Specification **`nrtidalv2-nonspinning-tidal-phase/v1`** covers only the additive
frequency-domain NRTidalv2 tidal phase of the dominant (2,2) mode, with zero
component spins, CPU FP64 and the declared 30–800 Hz binaries below the merger
taper. The independent [scalar reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/nrtidalv2_phase.py)
has component IDs `nrtidalv2.phase.lambda_tilde.v1`, `.pade.v1`, `.tidal.v1`
and `.frequency_derivative.v1`. It imports no Torch, JAX or LAL code.

## Primary sources and transcription

[Dietrich et al., *Phys. Rev. D* **100**, 044003 (2019),
arXiv:1905.06011v2, Eqs. (17)–(21)](https://arxiv.org/pdf/1905.06011v2)
prints the NRTidalv2 phase, Padé form, PN constraints and four calibrated
decimals. [Wade et al., *Phys. Rev. D* **89**, 103012 (2014), Eq. (5)](https://arxiv.org/pdf/1402.5156v1)
defines the mass-weighted dimensionless deformability. The four free decimals
are kept exactly as printed: $n_{5/2}=90.550822$, $n_3=-60.253578$,
$d_1=-15.111208$, and $d_2=8.0641096$. They are empirical fit inputs,
not values derived from PN theory. The reference asserts the published PN
expansion independently of either native implementation.

## Mathematics and units

Let $M=m_1+m_2$, $\delta=(m_1-m_2)/M$,
$\eta=(1-\delta^2)/4$, $M_{\rm sec}=GM/c^3$, and
$x=(\pi M_{\rm sec}f)^{2/3}$, where $f$ is positive GW frequency in Hz.
The inputs Λ₁ and Λ₂ are nonnegative *dimensionless* quadrupolar
deformabilities, with no equation-of-state relation imposed. Wade's Eq. (5),
using signed δ so that either body order is allowed, is

\[
\widetilde\Lambda=\frac8{13}\left[(1+7\eta-31\eta^2)(\Lambda_1+\Lambda_2)
+\delta(1+9\eta-11\eta^2)(\Lambda_1-\Lambda_2)\right].
\]

Dietrich's Eqs. (17)–(18) give

\[
\psi_T(f)=-\frac{39}{16\eta}\frac3{16}\widetilde\Lambda\,x^{5/2}
\frac{1+n_1x+n_{3/2}x^{3/2}+n_2x^2+n_{5/2}x^{5/2}+n_3x^3}
{1+d_1x+d_{3/2}x^{3/2}+d_2x^2}.
\]

The remaining coefficients follow Eqs. (19)–(20), with
$c_1=3115/1248$, $c_{3/2}=-\pi$,
$c_2=28024205/3302208$, $c_{5/2}=-4283\pi/1092$:
$n_1=c_1+d_1$,
$n_{3/2}=(c_1c_{3/2}-c_{5/2}-c_{3/2}d_1+n_{5/2})/c_1$,
$n_2=c_2+c_1d_1+d_2$, and
$d_{3/2}=-(c_{5/2}+c_{3/2}d_1-n_{5/2})/c_1$.
The waveform multiplier is $\exp(-i\psi_T)$. The phase is not shifted at
$f_{\rm ref}$; with zero spins, the finite-reference point-particle carrier
cancels in the BNS/BBH quotient. Symbolic differentiation defines
$d\psi_T/df$ with masses and deformabilities fixed.

## Frontend and binary boundary

The maintained [math tests](https://github.com/xangma/diffgw/blob/main/tests/test_nrtidalv2_phase_math_reference.py)
compare the unit complex quotient `IMRPhenomD_NRTidalv2 / IMRPhenomD` from
Torch and JAX with the independent high-precision phase. The denominator
removes the inherited point-particle carrier; normalization removes the
tidal amplitude. The same quotient is built from the installed **LALSuite
7.26.15 binary** by its public waveform-sequence API in the
[binary test](https://github.com/xangma/diffgw/blob/main/tests/test_nrtidalv2_phase_lal_binary.py). There is no
phase/time alignment or fitting to that binary. The oracle is used only for
agreement, never to choose coefficients.

## Predeclared qualification gate

The frozen systems $(m_1,m_2;\Lambda_1,\Lambda_2)$, in solar masses,
are $(1.35,1.35;400,400)$, $(1.6,1.2;200,800)$, its body-exchanged
copy, and $(1.35,1.35;0,0)$. Paper-value frequencies are 30, 100,
300, 500 and 800 Hz. At every point each frontend's unit-complex residual
must be at most $10^{-10}+10^{-11}|\psi_T|$. At 300 Hz, Torch/JAX
autodifferentiated $d\psi_T/df$ must agree with the independent symbolic
derivative within relative $10^{-8}$ and absolute $10^{-10}$ rad/Hz.

The LAL grid is 30–800 Hz in 5 Hz steps. For each case, the unaligned unit
complex quotient must have relative complex $L_2\le10^{-6}$ and maximum
pointwise phase residual $\le10^{-5}$ rad against the paper reference;
Torch and JAX quotients must meet the same limits against the LAL quotient.
Run with `DIFFGW_REQUIRE_LAL_BINARY=1` to fail instead of skip when the pinned
binary is unavailable. These limits are fixed before the binary run and are
not adjusted in response to failures.

## Evidence and limits

The [qualification receipt](../validation/nrtidalv2_phase_20260926.md) records
the exact host, checkout, command, binary versions, metrics and outcome.
Passing this gate establishes only literature transcription, native
implementation and named-binary agreement for the declared slice. It does
not establish physical accuracy, an equation-of-state relation, amplitude
Eq. (24), merger frequency and taper Eq. (25), self-spin Eqs. (26)–(29),
spinning binaries, other modes, FP32/GPU behavior or the full family.
