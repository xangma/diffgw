# IMRPhenomNSBH torus ratio and Kerr ISCO radius

## Scope and status

Specification **nsbh-torus-isco/v1** covers the Kerr ISCO radius and baryonic
torus-mass ratio that follow the already gated compactness/disruption root.
It does not qualify remnant/QNM fits, amplitude branches, phase, or the whole
waveform.

## Inputs, conventions, and options

The black hole is body 1, the nonspinning neutron star body 2,
$q=m_1/m_2\ge1$, $\chi$ is the aligned black-hole spin, $C$ is neutron-star
compactness, and $\xi_{\rm tide}$ is the largest positive root of Thompson
Appendix A Eq. (10). The ISCO radius is in units of the black-hole mass. The
ratio is dimensionless and is relative to the initial neutron-star **baryonic**
mass, not gravitational mass.

The frozen Cartesian gate has $q\in\{1,2,5,10,15\}$,
$\chi\in\{-0.5,0,0.5\}$, and
$\Lambda\in\{0,1,10,300,800,2600,5000\}$: 105 cases.
The compactness and root are the published scalar expressions specified in
[nsbh-compactness-disruption-root/v1](nsbh-disruption-root.md).
Both native backends are checked on CPU FP64.

## Mathematics and components

[Thompson et al. (2020), Appendix A Eqs. (11)–(12)](https://arxiv.org/pdf/2002.08383v1)
give, at the paper's printed decimal precision,

\[
Z_1=1+(1-\chi^2)^{1/3}\left[(1+\chi)^{1/3}+(1-\chi)^{1/3}\right],
\quad Z_2=\sqrt{3\chi^2+Z_1^2},
\]
\[
\bar r_{\rm ISCO}=3+Z_2-\operatorname{sign}(\chi)
\sqrt{(3-Z_1)(3+Z_1+2Z_2)},
\]
\[
T_{\rm raw}=0.296\,\xi_{\rm tide}(1-2C)
-0.171\,qC\bar r_{\rm ISCO}.
\]

The paper prints $T_{\rm raw}=M_{b,\rm torus}/M_{b,\rm NS}$, but this fit can
be negative. The native carrier's physical-mass adapter is
$T_{\rm carrier}=\max(0,T_{\rm raw})$ and its torus flag is
$T_{\rm carrier}>0$. The gate reports the unclipped equation and checks the
adapter separately; it does not attribute clipping to Eq. (11). For $\chi=0$,
Eq. (12) gives $\bar r_{\rm ISCO}=6$ and its local slope is
$-4\sqrt{2/3}$; the native implementations use that linear expansion only
for $|\chi|<10^{-4}$ to keep autodiff regular.

## Implementation mapping

The backend-free, high-precision scalar reference is
`src/diffgw_reference/nsbh_torus.py`, sharing only the independent scalar
compactness/root reference. Torch and JAX use their respective
`conversions.kerr_isco_radius` and
`approximants.imrphenom_nsbh.nsbh_torus_mass_ratio` helpers. The latter is
the exact component used in native amplitude classification.

## Evidence and limitations

Before native comparison, the gate fixes maximum absolute errors
$2\times10^{-13}$ for ISCO radius and $2\times10^{-12}$ for raw and clipped
torus ratios. It checks the sign/clip relation and both positive- and
zero-torus cases. A $70$- versus $90$-digit scalar check must differ by at most
$10^{-30}$ for all values. These are equation/component checks, not a
public-binary or physical-accuracy claim. A source error common to both
implementations remains possible despite the independent scalar evaluation.

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src pytest -q
tests/test_nsbh_torus_reference.py` with the `reference`, `torch`, and `jax`
extras on CPU. The measured result is in the
[27 September 2026 validation receipt](../validation/nsbh_torus_isco_20260927.md).
