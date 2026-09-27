# IMRPhenomHM (2,1) 1.5PN amplitude ratio

## Scope and status

`hm.amp.beta21.1p5pn.eq4.v1` is the (2,1) amplitude factor in
[London et al., arXiv:1708.00404v4, Eqs. (3)–(4) and the paragraph after
Eq. (4)](https://arxiv.org/pdf/1708.00404v4). London retains terms through
relative 1.5PN for $|m|=\ell-1$ modes. [Mishra et al.,
arXiv:1601.05588v1, Eqs. (11), (12a), (12b)](https://arxiv.org/pdf/1601.05588v1)
print the corresponding complete SPA polynomial through $V_1^4$.
This source-first version froze the equation and gate before the scalar,
production correction, and public checks. The sampled CPU FP64 gate passed
on `len`; see the [validation receipt](../validation/imrphenomhm_amplitude_ratio_21_20260927.md).
The printed equation is available through the public
`hm_21_pn_variant="mishra_eq12b"` option. The default
`hm_21_pn_variant="binary_compatible"` preserves the established HM binary
behavior; the source gate does not assert that the printed equation is the
binary model. This is not a whole-HM qualification.

## Inputs, conventions, and options

Let $x=Mf_{21}>0$, $g(x)>0$ be the supplied (2,2) amplitude-map frequency,
$m_1\ge m_2>0$, $\delta=(m_1-m_2)/(m_1+m_2)$, and
$\eta=m_1m_2/(m_1+m_2)^2=(1-\delta^2)/4$. Aligned component spins obey
$|\chi_i|\le1$, with $\chi_s=(\chi_1+\chi_2)/2$ and
$\chi_a=(\chi_1-\chi_2)/2$, as in Mishra Eq. (7). Eq. (3) uses the
**magnitude** of the complex ratio. At $\delta=\chi_a=0$, the (2,1)
polynomial vanishes identically and literal Eq. (4) is $0/0$; this scalar
component excludes that point. Equal masses with unequal spins are valid.

The public `hm_mapping_transitions` setting chooses `phenomd_runtime`
(default) or `london_2018` outer amplitude-map cutoff. The D carrier,
QNM frequencies, phase, distance, and projection remain supplied by the
public assembly. The `mishra_eq12b` option requires that D carrier.
The two Eq. (5) joins are $f_0^A$ and $f_{21}^{\rm RD}$;
one-sided slopes are used there.

## Mathematics and components

Set $V_m(y)=(2\pi y/m)^{1/3}$ and $v_y=V_1(y)$. Mishra Eq. (12b) gives
$\hat H_{21}(y)=-(\sqrt2/3)P(v_y)$ through $v_y^4$, where

$$
\begin{aligned}
P(v)={}&\delta v-\frac32(\chi_a+\delta\chi_s)v^2
 +\delta\left(\frac{335}{672}+\frac{117}{56}\eta\right)v^3\\
&+\left[\chi_a\left(\frac{4771}{1344}-\frac{11941}{336}\eta\right)
 +\delta\chi_s\left(\frac{4771}{1344}-\frac{2549}{336}\eta\right)
 +\delta\left(-\pi-i\left(\frac12+2\log2\right)\right)\right]v^4.
\end{aligned}
$$

Mishra Eqs. (11), (12a) give the full SPA envelopes
$\mathcal H_{21}(y)=-(\sqrt2/3)V_1(y)^{-7/2}P(v_y)$ and
$\mathcal H_{22}(y)=-V_2(y)^{-7/2}$ at the retained order. Their
$V_m^{-7/2}$ factors cancel between the two ratios in London Eq. (4):

$$
B_{21}(x;g)=\frac{\sqrt2}{3}
 P(v_{g(x)})\frac{P(v_x)}{P(v_{2x})},\qquad
\beta_{21}=|B_{21}|,\qquad
A_{21}(x)\simeq\beta_{21}A_{22}(g(x)).
$$

The nonspinning leading reduction is
$(\sqrt2/3)\delta[\pi g(x)]^{1/3}$, but it is **not** the prescribed
1.5PN public factor. Eq. (5) supplies $g'=2$ below $f_0^A$,
$g'=(f_{22}^{\rm RD}-2f_0^A)/(f_{21}^{\rm RD}-f_0^A)$ between joins,
and $g'=1$ above ringdown. For $B\ne0$, its one-sided slope is
$\operatorname{Re}(\overline B\,B')/|B|$, using the corresponding $g'$.
At $\delta=\chi_a=0$, the public mode takes the finite zero-mode branch;
the literal scalar Eq. (4) quotient is undefined there. No limiting
amplitude ratio independent of the path toward this point is claimed.

| Component | Source | Scalar | Native production | Gate |
| --- | --- | --- | --- | --- |
| `hm.amp.beta21.1p5pn.eq4.v1` | London Eqs. (3)–(4); Mishra Eqs. (11), (12a), (12b) | `diffgw_reference.imrphenomhm_amplitude_ratio` | Torch/JAX public `IMRPhenomHM` D-carrier (2,1) with `hm_21_pn_variant="mishra_eq12b"` | `test_imrphenomhm_amplitude_ratio_math_reference.py` |

## Implementation mapping

An independent SymPy reduction must begin with both complete SPA
envelopes, and separately reduce the printed $v^4$ spin coefficients.
An mpmath scalar evaluates $P$, $B$, $|B|$, the Eq. (5) composition, and
the analytic one-sided magnitude slopes. It imports no native helper or
calibration data. The public test divides the unprojected Torch/JAX (2,1)
mode magnitude with `hm_21_pn_variant="mishra_eq12b"` by the public PhenomD (2,2) carrier magnitude at the
independently mapped frequency. This checks the production-called ratio,
including its complex polynomial, through Eq. (3). It does not use the
older public `h_hat_multipole` helper as a substitute for production.

## Evidence and limitations

Frozen `len` CPU FP64 grid: $(m_1,m_2,\chi_1,\chi_2)$ equal to
$(35,20,0.2,-0.1)$, $(35,20,0,0)$, and $(30,30,0.2,-0.1)$, masses in
solar units, distance 100 Mpc, and both outer-map choices. Sample below,
at, between, and above both joins. Require public quotient versus mpmath
relative error $\le10^{-10}$ with absolute floor $10^{-12}$. At each join,
compare one-sided secants with $h=10^{-7}$ in geometric frequency to
analytic slopes, relative error $\le5\times10^{-5}$ and absolute floor
$5\times10^{-8}$. Check exact SymPy Eq. (4) reduction, the printed $v^4$
spin coefficients, a hand-reduced nonspinning scalar, equal-mass unequal-spin
nonzero behavior, and rejection of the $0/0$ equal-mass equal-spin point.
Capture the pre-correction public failure before changing production.
The [receipt](../validation/imrphenomhm_amplitude_ratio_21_20260927.md)
records that failure, the source-option implementation, and the passing
source/public and neighboring HM checks. The strict binary oracle separately
qualifies the default `binary_compatible` option.

Passing this gate establishes only the sampled (2,1) 1.5PN magnitude
factor. Higher PN orders, independent QNM and D-amplitude correctness,
odd-mode phase and parity, projection, LAL agreement, physical accuracy,
and the complete HM waveform remain separate. No LAL code, constants,
or binary output enters this gate.

## Reproduce the checks

On `len`, with the repository's reference, Torch, JAX, and test
dependencies, run `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
python -m pytest -q tests/test_imrphenomhm_amplitude_ratio_math_reference.py
--reference-device=cpu -ra`. The linked receipt records the exact checkout,
environment, command, pre-correction failure, final result, and skips.
A changed source equation, truncation, map convention,
or tolerance requires a new component version and reviewed evidence.
