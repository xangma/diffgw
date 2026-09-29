# IMRPhenomHM leading (4,4) amplitude ratio

## Scope and status

`hm.amp.beta44.leading.eq4.v1` is the restricted stationary-phase (4,4)
amplitude factor in [London et al., arXiv:1708.00404v4, Eqs. (3)–(4) and
the paragraph after Eq. (4)](https://arxiv.org/pdf/1708.00404v4).
[Mishra et al., arXiv:1601.05588v1, Eqs. (11), (12a),
(12f)](https://arxiv.org/pdf/1601.05588v1) supply the complete leading
terms used here. This is a component specification, not a qualification of
the complete HM model or its independent PhenomD carrier.

The equation, domain, and numerical gate were frozen first in commit
`87d1898`. The independent scalar and public Torch/JAX checks passed the
declared CPU FP64 gate on `len`; the [run receipt](../validation/imrphenomhm_amplitude_ratio_44_20260927.md)
records the command and result. This is an executable component only.

## Inputs, conventions, and options

Let $x=Mf_{\rm physical}>0$ be the (4,4) Fourier frequency, $g(x)>0$ the
supplied (2,2) amplitude-map frequency in the same geometric units, and
$\eta=m_1m_2/(m_1+m_2)^2$ with positive masses, hence $0<\eta\le1/4$.
The leading (4,4) term is positive in the amplitude ratio because
$1-3\eta\ge1/4$. It has no explicit spin dependence. The public
`hm_mapping_transitions` option chooses `phenomd_runtime` (default) or
`london_2018` and changes the supplied map's first cutoff. The QNM
frequencies, distance, phase, and PhenomD amplitude are supplied by the
public assembly; none is calibrated by this component.

The map joins are $f_0^A$ and $f_{44}^{\rm RD}$. The ratio is continuous,
but its frequency slope is generally discontinuous at each join. The gate
therefore uses one-sided secants. Nonpositive frequencies and nonphysical
$\eta$ are outside the scalar domain.

## Mathematics and components

Mishra defines $V_m(y)=(2\pi y/m)^{1/3}$. Their Eq. (11) gives a common
SPA factor times $V_m^{-7/2}\hat H_{\ell m}$. At London's restricted order,

$$
\hat H_{22}=-1,\qquad
\hat H_{44}=-\frac49\sqrt{\frac{10}{7}}
                 (1-3\eta)V_4^2.
$$

The *full* leading SPA amplitudes are consequently
$\mathcal H_{22}(y)=-V_2(y)^{-7/2}$ and
$\mathcal H_{44}(y)=-(4/9)\sqrt{10/7}(1-3\eta)V_4(y)^{-3/2}$.
In London's Eq. (4), their ratio at $g$ is multiplied by
$\mathcal H_{44}(x)/\mathcal H_{44}(x/2)=2^{-1/2}$. The factors of two
cancel, giving

$$
\beta_{44}(x;g,\eta)=\frac49\sqrt{\frac{10}{7}}(1-3\eta)
                        [\pi g(x)]^{2/3},\qquad
\frac{d\beta_{44}}{dx}=\frac{2\beta_{44}}{3g(x)}g'(x).
$$

London's Eq. (3) supplies
$A_{44}(x)\simeq|\beta_{44}(x)|A_{22}(g(x))$.
The Eq. (5) amplitude map has $g'=1/2$ below $f_0^A$,
$g'=(f_{22}^{\rm RD}-f_0^A/2)/(f_{44}^{\rm RD}-f_0^A)$ between joins,
and $g'=1$ above ringdown. The independent frequency-map component
specifies the boundary values and both outer-cutoff choices.

| Component | Source | Scalar | Native production | Gate |
| --- | --- | --- | --- | --- |
| `hm.amp.beta44.leading.eq4.v1` | London Eqs. (3)–(4); Mishra Eqs. (11), (12a), (12f) | `diffgw_reference.imrphenomhm_amplitude_ratio` | Torch/JAX public `IMRPhenomHM` D-carrier (4,4) | `test_imrphenomhm_amplitude_ratio_math_reference.py` |

## Implementation mapping

The independent SymPy expression must reduce the ratio of full leading SPA
amplitudes, including $V_m^{-7/2}$, to the formula above. A standalone
mpmath function evaluates the ratio and composes the existing scalar Eq. (5)
map. It does not import native helpers, fit tables, or QNM data.

The test calls the public Torch and JAX unprojected (4,4) mode with the D
carrier and divides its magnitude by the public PhenomD (2,2) carrier
magnitude evaluated at the independently mapped frequency. This quotient
checks the production-called amplitude factor through the public route.
The public (4,4) phase and projection are outside this quotient claim.

## Evidence and limitations

Frozen CPU FP64 gate on `len`: use $m_1,m_2=(35,20)$ and $(30,30)$ in
solar masses, $(\chi_1,\chi_2)=(0.2,-0.1)$, distance 100 Mpc, and both
named map choices. Sample below, at, between, and above the amplitude-map
joins. Require relative quotient error at most $5\times10^{-11}$ with
absolute floor $5\times10^{-13}$. At each join compare one-sided secants
with step $h=10^{-7}$ in geometric frequency to analytic slopes, with
relative error at most $3\times10^{-5}$ and absolute floor
$3\times10^{-8}$. Check an exact SymPy reduction, a hand-reduced mpmath
value, and scalar domain rejection. No LAL implementation or binary output
is an input to this gate.

The passing checks establish only the leading (4,4) factor at sampled
public D-carrier points. Higher PN terms, independent QNM and D-amplitude
correctness, phase, projection, full modes, LAL agreement, and physical
accuracy remain separate. This component is retained in the provisional
research ledger and does not create a whole-model verification claim.

## Reproduce the checks

On `len`, with the repository's reference, Torch, JAX, and test dependencies,
run `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q
tests/test_imrphenomhm_amplitude_ratio_math_reference.py
--reference-device=cpu -ra`. Record the exact checkout, environment,
command, result, and any skip before changing the status above. A changed
source equation, truncation, map convention, or tolerance requires a new
component version and reviewed evidence.
