# HM signed-mode projection and phase transport

## Scope and status

`imrphenomhm-signed-projection/v1`, component
`hm.conventions.signed-pair-projection.v1`, extends the
[one-sided projection](imrphenomhm-projection.md)
and [Gaussian SPA convention reference](imrphenomhm-spa-conventions.md)
without replacing either specification. It derives the exact Fourier transform
of a signed TD mode pair, including its opposite-frequency contribution.
It is a research component, not a production convention or physical
qualification claim. Numerical execution status is recorded in the
[validation receipt](../validation/imrphenomhm_signed_projection_20260928.md).

## Inputs, conventions, and options

Use $h=h_+-ih_\times$, $\mathcal F_-[h](f)=\int h(t)e^{-2\pi ift}dt$,
and $h_{\ell,-m}=p\overline{h_{\ell m}}$, where $p=(-1)^\ell$ and $m>0$.
These are [García-Quirós et al., arXiv:2001.10914v2, Sec. II.1,
Eqs. (2), (4)–(6)](https://arxiv.org/html/2001.10914v2).
The supported pairs have positive labels 22, 21, 33, 32, 44 and 43.
Physical observer azimuth is minus native `phi0`, as in the existing
projection adapter. Harmonic normalization and the Wigner construction are
inherited from `hm.projection.london-eq1.v1`.

Define $P(f)=\mathcal F_-[\overline{h_{\ell m}}](f)$ and
$Q(f)=\mathcal F_-[h_{\ell m}](f)=\overline{P(-f)}$.
Both are retained. A finite damped quadratic chirp has nonzero spectral
leakage; one-sided support must not be assumed in an exact transform test.

## Mathematics and components

### Exact signed-pair projection

Evaluate $Y_\pm={}_{-2}Y_{\ell,\pm m}$ at native `phi0`, and reuse

$$D_+=\frac{Y_++p\overline{Y_-}}2,\qquad
D_\times=-\frac{i}{2}(Y_+-p\overline{Y_-}).$$

At physical azimuth `-phi0`, the TD strain of the pair is
$H=\overline{Y_+}h_{\ell m}+p\overline{Y_-}\overline{h_{\ell m}}$.
Taking $h_+=\operatorname{Re}H$ and $h_\times=-\operatorname{Im}H$
before Fourier transformation gives the exact expressions

$$\widetilde h_+=D_+P+\overline{D_+}Q,\qquad
\widetilde h_\times=D_\times P+\overline{D_\times}Q.$$

Conjugation at negative frequency follows directly from
$Q(f)=\overline{P(-f)}$. Setting $Q(f)=0$ at positive frequency recovers
the native assembly and the paper's Eqs. (8)–(9), after its
$\widetilde h_{\ell,-m}=pP$ carrier conversion. This reduction verifies
the assembly, not the accuracy of discarding leakage in physical data.

### Transport of the fixed SPA phase

For a unit phase $q_m$, a consistent TD transformation is

$$h_{\ell m}'=\overline{q_m}h_{\ell m},\qquad
h_{\ell,-m}'=q_mh_{\ell,-m}.$$

It preserves equatorial symmetry and gives $P'=q_mP$ and
$Q'=\overline{q_m}Q$. Consequently

$$\widetilde h_A'=D_Aq_mP+\overline{D_Aq_m}Q,
\qquad A\in\{+,\times\}.$$

The source-derived SXS-basis candidate is
$q_m=\exp[i(m/2-1)K]$, $K=3\pi/4$, with one common orbital branch.
It leaves both $m=2$ modes unchanged. Applying $q_m$ to both $P$ and $Q$
would generally violate real-polarization conjugate symmetry.
The [SPA derivation](imrphenomhm-spa-conventions.md) fixes this candidate
from the leading PN coefficient phases and the common 22 phase alignment;
it is not fitted to NR residuals.

### Why this is not generally a polarization rotation

The identity $q_m=e^{-iK}e^{imK/2}$ separates an orbital shift from a
common analytic carrier phase. In the one-sided approximation, the latter
multiplies both Fourier polarizations by the same complex scalar. A physical
basis change instead mixes the two real polarizations through a real rotation
matrix. At edge-on, a 22 signal has $\widetilde h_\times=0$: a nonreal common
Fourier factor cannot be reproduced by a real matrix while keeping the cross
channel zero. Circular polarization at a pole hides this distinction.
This counterexample concerns the common factor in the decomposition; the
complete $q_2$ equals one.

Equivalently, after removing the orbital shift, the TD transformation has
opposite phases for opposite signs of $m$. It is not a uniform rotation of
the complex strain. The conversion therefore defines a distinct carrier
convention; the improved primitive phase residual alone cannot identify it
as a harmless observer-basis change or a complete waveform repair.

## Implementation mapping

`src/diffgw_reference/imrphenomhm_signed_projection.py` composes the independent
Wigner projection and Gaussian transform references. The maintained checks are
in `tests/test_imrphenomhm_signed_projection_math_reference.py`.
Native Torch/JAX waveform evaluation is unchanged. Their existing
`test_imrphenomhm_projection_math_reference.py` gate separately checks the
one-sided coefficients against public per-mode and complete polarizations.

## Evidence and limitations

The checks cover exact Fourier conjugacy, TD signed-pair symmetry, direct
quadrature of the real TD polarizations, all six modes, nonzero azimuth,
both hemispheres, and conjugate phase transport. A controlled Gaussian
sequence tests suppression of the omitted term. This is evidence about the
specified analytic chirps and projection algebra; it does not establish a
bound on NR leakage or waveform accuracy.

This component alone does not qualify a production waveform. The subsequent
[native convention extension](../validation/imrphenomhm_spa_native_20260928.md)
applies its phase factor under an explicit option and separately checks the
complete composition; full-family qualification remains open. The remaining frequency-dependent phase and amplitude errors in
the [seen-NR diagnostic](../validation/imrphenomhm_spa_conventions_20260928.md)
remain separate from this convention question.

## Reproduce the checks

On len, with the optional reference and both frontend dependencies installed:

```sh
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_imrphenomhm_signed_projection_math_reference.py \
  tests/test_imrphenomhm_spa_conventions_math_reference.py \
  tests/test_imrphenomhm_projection_math_reference.py
```

The new filename joins the strict mathematical reference inventory automatically.
