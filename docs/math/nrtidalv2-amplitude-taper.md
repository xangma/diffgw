# NRTidalv2 nonspinning tidal amplitude and merger taper

## Scope and status

Specification **nrtidalv2-nonspinning-tidal-amplitude-taper/v1** covers
the signed additive Fourier tidal amplitude, published merger-frequency fit
and descending Planck taper for the dominant mode, zero component spins,
CPU FP64, the four frozen binaries below, and 100 Mpc distance. Its amplitude
gate is currently a strict expected-failure diagnostic.

## Inputs, conventions, and options

With $M_{\rm sec}=GM/c^3$, $D_{\rm sec}=D_L/c$,
$\eta=m_1m_2/M^2$, $x=(\pi M_{\rm sec}f)^{2/3}$ and
$\kappa_{\rm eff}^{T}=3\widetilde\Lambda/16$, the frozen systems
$(m_1,m_2;\Lambda_1,\Lambda_2)$ in solar masses are
$(1.35,1.35;400,400)$, $(1.6,1.2;200,800)$, the latter with both
bodies exchanged, and $(1.35,1.35;0,0)$. Inclination is 0.4 rad.
Amplitude frequencies are 100, 300, 800 and 1200 Hz. Taper frequencies
are $f/f_{\rm mrg}=$ 0.9, 1, 1.025, 1.05, 1.1, 1.15, 1.195, 1.2 and
1.25. The zero-tide case probes the taper independently of Eq. (24).

## Mathematics and components

[Dietrich et al. (2019), Eq. (24) and text around Eq. (25)](https://arxiv.org/pdf/1905.06011v2)
give the amplitude and its multiplication by a Planck taper.
[Dietrich et al. (2018), Eq. (11)](https://arxiv.org/pdf/1804.02235)
give the merger-frequency fit, substituting $\kappa_{\rm eff}^{T}$ for
$\kappa_2^{T}$ as the paper states for its LAL implementation.
The mass-weighted $\widetilde\Lambda$ follows
[Wade et al. (2014), Eq. (5)](https://arxiv.org/pdf/1402.5156v1).
The signed face-on amplitude is

\[
\widetilde A_T=-\sqrt{\frac{5\pi\eta}{24}}\,
\frac{9M_{\rm sec}^2}{D_{\rm sec}}\kappa_{\rm eff}^{T}x^{13/4}
\frac{1+(449/108)x+(22672/9)x^{2.89}}{1+13477.8x^4}.
\]

The square-root normalization is part of Eq. (24); the decimal fit
inputs are used exactly as printed. For $X_A$ the heavier mass fraction,
$X_B$ the lighter, the quoted merger fit is

\[
f_{\rm mrg}=\frac{0.3586}{2\pi M_{\rm sec}}\sqrt{\frac{X_B}{X_A}}\,
\frac{1+0.03354\kappa_{\rm eff}^{T}
      +0.00004315(\kappa_{\rm eff}^{T})^2}
     {1+0.07542\kappa_{\rm eff}^{T}
      +0.0002236(\kappa_{\rm eff}^{T})^2}.
\]

Equation (25) gives $(\widetilde A_{\rm BBH}+\widetilde A_T)W(f)$.
The descending Planck window is 1 through $f_{\rm mrg}$, 0 from
$1.2f_{\rm mrg}$ onward, and for
$z=(f-f_{\rm mrg})/(0.2f_{\rm mrg})\in(0,1)$ is
$W=[1+\exp(-1/z-1/(z-1))]^{-1}$. Inclined $h_+$ applies the positive
projection $(1+\cos^2\iota)/2$ to $\widetilde A_T$.

## Implementation mapping

The independent scalar reference `src/diffgw_reference/nrtidalv2_amplitude.py`
uses SymPy/mpmath and no native waveform or LAL implementation. The tests
below compare it separately with the Torch/JAX frontends and the installed
LALSuite binary.

## Evidence and limitations

The green paper-math tests in `tests/test_nrtidalv2_amplitude_math_reference.py`
require merger-frequency relative error $\le10^{-11}$ and scalar/window
amplitude error $\le10^{-10}$. The strict amplitude diagnostics in
`tests/test_nrtidalv2_amplitude_diagnostic.py`
retain the frozen native Torch/JAX scalar amplitude relative limit $10^{-11}$
and untapered plus-polarization amplitude-ratio absolute limit $10^{-10}$.
The pinned-binary tests in `tests/test_nrtidalv2_amplitude_lal_binary.py`
compare the *unaligned* modulus ratios
$|h_+^{\rm BNS}|/|h_+^{\rm BBH}|$ with the paper result using each
backend's own BBH amplitude, then compare native and LAL ratios. Maximum
absolute ratio error must be $\le5\times10^{-5}$ before and within the
taper, including its endpoints. LALSuite 7.26.15 is required under
DIFFGW_REQUIRE_LAL_BINARY=1. These limits are not retuned on failure.

**Disposition:** the amplitude gate fails. The published Eq. (24) absolute
correction and the Torch/JAX plus the pinned-LAL carrier quotients differ in
normalization; the cause is unresolved. The native merger fit and Planck
window pass the scalar gate, while the nonzero-tide native/LAL quotient
exceeds its binary limit inside the taper. Strict expected-failure markers
retain these diagnostics outside the required green math-reference CI job
and fail on a future unexpected pass.

This gate tests only literature transcription and named-binary agreement
for this slice. It does not validate the NR calibration, self-spin,
nonzero spins, other modes, other distances/inclinations, FP32/GPU or
full-family behavior.

## Reproduce the checks

Run `PYTHONPATH=src:tests DIFFGW_REQUIRE_LAL_BINARY=1 JAX_ENABLE_X64=1
python -m pytest -q tests/test_nrtidalv2_amplitude_math_reference.py
tests/test_nrtidalv2_amplitude_diagnostic.py
tests/test_nrtidalv2_amplitude_lal_binary.py` on CPU FP64 with the pinned
binary installed. Exact measurements and commands are in the
[26 September 2026 receipt](../validation/nrtidalv2_amplitude_20260926.md).
