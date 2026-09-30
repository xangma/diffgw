# IMRPhenomHM (4,3) 1.5PN amplitude ratio

## Scope and status

`hm.amp.beta43.1p5pn.eq4.v1` is the restricted (4,3) amplitude factor
from [London et al., arXiv:1708.00404v4, Eqs. (3)–(4) and the paragraph
following Eq. (4)](https://arxiv.org/pdf/1708.00404v4). London keeps terms
through relative 1.5PN for $|m|=\ell-1$ modes. [Mishra et al.,
arXiv:1601.05588v1, Eqs. (7), (11), (12a), (12g)](https://arxiv.org/pdf/1601.05588v1)
print the SPA polynomial used here. This source-first version freezes the
equation and validation gate before native changes. The printed formula is
selected by `hm_43_pn_variant="mishra_eq12g"`; the
`"binary_compatible"` default retains the established native carrier.
This is a component claim, not whole-HM qualification.

## Inputs, conventions, and options

Let $x=Mf_{43}>0$ and $g(x)>0$ be the supplied (2,2) amplitude-map
frequency in geometric units. Require $m_1\ge m_2>0$,
$\delta=(m_1-m_2)/(m_1+m_2)$,
$\eta=m_1m_2/(m_1+m_2)^2=(1-\delta^2)/4$,
$\chi_s=(\chi_1+\chi_2)/2$, and $\chi_a=(\chi_1-\chi_2)/2$ with
$|\chi_i|\le1$. The aligned spins are the projections on the Newtonian
orbital angular momentum in Mishra Eq. (7). London Eq. (3) takes the
**magnitude** of the complex Eq. (4) ratio. The option requires the public
PhenomD (2,2) carrier. `hm_mapping_transitions` selects the default
`phenomd_runtime` or source `london_2018` outer-map cutoff.

At $\delta=\chi_a=0$, the (4,3) SPA polynomial is identically zero and
literal Eq. (4) is $0/0$; the scalar ratio excludes that point. Equal mass
with unequal spins is valid. The public zero-mode branch is checked
separately. This component neither supplies QNM frequencies nor validates
the PhenomD amplitude, phase, distance, or projection.

## Mathematics and components

Set $V_m(y)=(2\pi y/m)^{1/3}$ and

$$P_{43}(v)=\delta(1-2\eta)v^3+\frac52\eta(\chi_a-\delta\chi_s)v^4.$$

At the retained order Mishra Eqs. (11), (12a), (12g) give
$\mathcal H_{43}(y)=-(3/4)\sqrt{3/35}\,V_3(y)^{-7/2}P_{43}(V_3(y))$
and $\mathcal H_{22}(y)=-V_2(y)^{-7/2}$. In London Eq. (4), the SPA
$V_m^{-7/2}$ factors cancel between the two ratios, yielding

$$
B_{43}(x;g)=\frac34\sqrt{\frac3{35}}\,
P_{43}(V_3(g(x)))
\frac{P_{43}(V_3(x))}{P_{43}(V_3(2x/3))},\qquad
\beta_{43}=|B_{43}|,\qquad
A_{43}(x)\simeq\beta_{43}A_{22}(g(x)).
$$

The nonspinning reduction is
$(3/4)\sqrt{3/35}\,\delta(1-2\eta)[\pi g(x)]$;
it does not replace the printed spin term. The Eq. (5) amplitude map has
$g'=2/3$ below $f_0^A$, an affine slope between $f_0^A$ and
$f_{43}^{\rm RD}$, and $g'=1$ above ringdown. At a nonzero ratio,
$d|B|/dx=\operatorname{Re}(\overline B B')/|B|$, with the appropriate
one-sided map slope at each join.

| Component | Source | Scalar | Native public option | Gate |
| --- | --- | --- | --- | --- |
| `hm.amp.beta43.1p5pn.eq4.v1` | London Eqs. (3)–(5); Mishra Eqs. (7), (11), (12a), (12g) | `diffgw_reference.imrphenomhm_amplitude_ratio_43` | Torch/JAX `IMRPhenomHM`, D carrier, `hm_43_pn_variant="mishra_eq12g"` | `test_imrphenomhm_amplitude_ratio_43_math_reference.py` |

## Implementation mapping

Independent SymPy reduction starts with both complete SPA envelopes and
checks the printed $v^4$ coefficient. The mpmath scalar evaluates the
unreduced polynomial, the Eq. (4) magnitude, and analytic one-sided Eq. (5)
slopes without native imports or calibration data. The public test removes
the separately evaluated PhenomD (2,2) carrier magnitude at $g(x)$ from
the unprojected (4,3) mode and compares the remaining factor with mpmath.
It does not substitute the public `h_hat_multipole` helper for production.

## Evidence and limitations

Frozen `len` CPU FP64 cases are $(m_1,m_2,\chi_1,\chi_2)$ equal to
$(35,20,0.2,-0.1)$, $(35,20,0,0)$, and $(30,30,0.2,-0.1)$, in solar
masses; distance is 100 Mpc. Both map settings sample below, at, between,
and above $f_0^A$ and $f_{43}^{\rm RD}$. The public quotient must agree
with mpmath to `rtol=1e-10, atol=1e-12`. One-sided secants at geometric
step $10^{-7}$ use `rtol=5e-5, atol=5e-8`. Check the exact symbolic
reduction, a hand-reduced nonspinning scalar, equal-mass unequal-spin
nonzero behavior, and rejection of the scalar $0/0$ point. Capture the
pre-change public failure. The default must equal an explicit
`binary_compatible` request exactly; the paper option must be observable.

These checks establish sampled source and implementation agreement for
the (4,3) amplitude factor only. QNM and D-carrier correctness, phase,
projection, same-model LAL parity, physical accuracy, and the complete
HM waveform remain separate. No LAL code, constants, or binary output
enters this gate.

## Reproduce the checks

On `len` with the repository's reference, Torch, JAX, and test
dependencies, run `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
python -m pytest -q tests/test_imrphenomhm_amplitude_ratio_43_math_reference.py
--reference-device=cpu -ra`. The validation receipt records the exact
checkout, environment, red/green commands, results, and skips.
A changed source equation, truncation, map convention, or tolerance
requires a new component version and reviewed evidence.
