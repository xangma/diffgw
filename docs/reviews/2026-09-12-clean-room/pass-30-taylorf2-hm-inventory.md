# Pass 30: TaylorF2_HM leading-PN mode inventory

This pass audits the nonspinning leading terms against García-Quirós et al.,
“Frequency-domain gravitational-waveforms for non-precessing black-hole
binaries,” Phys. Rev. D 102, 064002 (2020), arXiv:2001.10914v2, Appendix FPN.
Only equations printed in that primary source were used.

## Source-complete additions

The paper defines the Fourier-domain harmonic velocity

\[
v_m=(2\pi M f/m)^{1/3},\qquad
\mathcal A^{\rm SPA}_{\ell m}=\pi\sqrt{2\eta/3}\,v_m^{-7/2}\hat H_{\ell m}.
\]

Its explicit leading nonspinning terms are

\[
\hat H_{31}=\frac{i}{12\sqrt 7}\,\delta v_1+O(v_1^3),
\]

and

\[
\hat H_{43}=\frac{3i}{4}\sqrt{\frac{3}{35}}
  \left[\delta(2\eta-1)v_3^3+O(v_3^5)\right].
\]

Here \(\delta=(m_1-m_2)/M\).  Converting the paper's \(v_m\) factors to
TorchWave's \(v_{22}=(\pi M f)^{1/3}\), and dividing by the existing common
\(22\) mode normalization \(\sqrt{16\pi/5}\,A_0\), gives the real-amplitude
ratios used by the implementation:

\[
R_{31}=\frac{\delta v_{22}}{12\sqrt7\,2^{5/6}},
\]

\[
R_{43}=\frac34\sqrt{\frac3{35}}\left(\frac32\right)^{1/6}
  \delta(1-2\eta)v_{22}^3.
\]

The sign in the second expression is the existing TaylorF2_HM real-amplitude
phase convention: the common \(-i\) phase is represented by the mode phase
connection, so the source's \(\delta(2\eta-1)\) is written as the equivalent
positive \(\delta(1-2\eta)\) coefficient.  The exact implementation ratios are
also checked without an absolute-value fit: \(R_{31}/R_{21}=1/(4\sqrt{14})\),
and

\[
R_{43}/R_{33}=\frac{\sqrt2}{5}\left(\frac32\right)^{1/6}
  (1-2\eta)v_{22}^2.
\]

No explicit leading Fourier-domain equation for \((4,2)\) is printed in this
source, so that mode remains rejected per-call and is not inferred.

## Projection convention

The \((3,1)\) and \((4,3)\) projections use the standard spin-weight \(-2\)
Wigner-d harmonics, with the same convention already used for the existing
modes.  The positive-m polynomials are

\[
{}_{-2}Y_{31}=\sqrt{\frac{70}{256\pi}}\sin\iota(1+\cos\iota)
  (3\cos\iota-1)e^{i\phi_0},
\]

\[
{}_{-2}Y_{43}=3\sqrt{\frac{14}{256\pi}}\sin\iota(1+\cos\iota)^2
  (1-2\cos\iota)e^{3i\phi_0}.
\]

The negative-m partners and \(d_+,d_\times\) are formed with the existing
García-Quirós projection equation.  Their signs and normalizations agree with
the existing \((2,1),(3,3),(3,2),(4,4)\) formulas.

## Verification

Focused command:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest \
  tests/test_taylorf2_hm.py tests/test_taylorf2_hm_mode_inventory.py -q
```

Result: **13 passed**.  Coverage includes exact published ratios, direct
projection polynomials, equal-mass odd-mode zeros, unsupported-mode validation,
batched float32 CPU execution (and CUDA when available), and the existing
autograd path over the expanded default mode set.
