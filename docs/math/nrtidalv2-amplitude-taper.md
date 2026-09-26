# NRTidalv2 nonspinning tidal amplitude and merger taper

## Scope and status

Specification **nrtidalv2-nonspinning-tidal-amplitude-taper/v1** covers
the signed additive Fourier tidal amplitude, published merger-frequency fit
and descending Planck taper for the dominant mode, zero component spins,
CPU FP64, the four frozen binaries below, and 100 Mpc distance. The native
paper-amplitude gate passes; the pinned-LAL binary gate remains a strict
expected-failure diagnostic.

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

The square-root normalization is part of Eq. (24). It also follows
independently from the leading term of Dietrich et al. (2019), Eq. (23):
$(27/16)\sqrt{160\pi\eta/27}=9\sqrt{5\pi\eta/24}$, while
$x^{-7/4}x^5=x^{13/4}$. Thus Eqs. (23) and (24) have the same
low-frequency normalization. The physical Newtonian IMRPhenomD carrier is
$\sqrt{5\eta/24}\,M_{\rm sec}^{5/6}/(\pi^{2/3}D_{\rm sec}f^{7/6})$
([Cutler–Flanagan, Eqs. (62)–(63)](https://arxiv.org/pdf/gr-qc/9402014v1)).
In the same $x$ variable this is exactly
$\sqrt{5\pi\eta/24}\,M_{\rm sec}^2 x^{-7/4}/D_{\rm sec}$.
The published additive correction therefore has the same physical amplitude
scale as the carrier; the earlier native $M_{\rm sec}^2/(2D_{\rm sec})$
scale was inconsistent with both equations. Equation (24) specifies the
**absolute additive tidal amplitude** $\widetilde A_T$, rather than a
dimensionless multiplier or the full BNS amplitude. Equation (25) adds it
once to $\widetilde A_{\rm BBH}$ before tapering. The frontend's quotient
$1+\widetilde A_T/\widetilde A_{\rm BBH}$ is only an algebraic way to apply
that sum to the complex BBH carrier; it introduces no second Newtonian
normalization. The decimal fit inputs are used
exactly as printed. For $X_A$ the heavier mass fraction,
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

The paper-math tests in `tests/test_nrtidalv2_amplitude_math_reference.py`
require merger-frequency relative error $\le10^{-11}$ and scalar/window
amplitude error $\le10^{-10}$. The now-green native amplitude checks in
`tests/test_nrtidalv2_amplitude_math_reference_native.py`
retain the frozen Torch/JAX scalar amplitude relative limit $10^{-11}$
and untapered plus-polarization amplitude-ratio absolute limit $10^{-10}$.
The pinned-binary tests in `tests/test_nrtidalv2_amplitude_lal_binary.py`
compare the *unaligned* modulus ratios
$|h_+^{\rm BNS}|/|h_+^{\rm BBH}|$ with the paper result using each
backend's own BBH amplitude, then compare native and LAL ratios. Maximum
absolute ratio error must be $\le5\times10^{-5}$ before and within the
taper, including its endpoints. LALSuite 7.26.15 is required under
DIFFGW_REQUIRE_LAL_BINARY=1. These limits are not retuned on failure.

**Disposition:** Torch/JAX now agree with the independent Eq. (24) scalar
reference and composed untapered waveform at the frozen limits. The pinned
LALSuite 7.26.15 binary outputs match the earlier native amplitude
normalization at the measured frequencies: its quotient differs from Eq. (24),
so both paper-versus-LAL and corrected-native-
versus-LAL binary gates fail before merger. The native merger fit and Planck
window pass the scalar gate. Nonzero-tide binary quotients also exceed the
frozen limit inside the taper. Strict expected-failure markers remain on the
binary comparisons and fail on a future unexpected pass.
Public binary quotients imply an approximately 25 ppm lower effective
taper boundary for the nonzero-tide cases if their untapered amplitudes
continue to agree inside the window. This is a diagnostic inference, not
an identification of the pinned binary's internal merger calculation.

This gate tests only literature transcription and named-binary agreement
for this slice. It does not validate the NR calibration, self-spin,
nonzero spins, other modes, other distances/inclinations, FP32/GPU or
full-family behavior.

## Reproduce the checks

Run `PYTHONPATH=src:tests DIFFGW_REQUIRE_LAL_BINARY=1 JAX_ENABLE_X64=1
python -m pytest -q tests/test_nrtidalv2_amplitude_math_reference.py
tests/test_nrtidalv2_amplitude_math_reference_native.py
tests/test_nrtidalv2_amplitude_lal_binary.py` on CPU FP64 with the pinned
binary installed. Exact measurements and commands are in the
[26 September 2026 receipt](../validation/nrtidalv2_amplitude_20260926.md).
