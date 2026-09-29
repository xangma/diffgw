# IMRPhenomHM (3,2) 1.5PN amplitude ratio

## Scope and status

`hm.amp.beta32.1p5pn.eq4.v1` is the aligned-spin (3,2) amplitude factor of
[London et al., arXiv:1708.00404v4, Eqs. (3)–(4) and the paragraph after
Eq. (4)](https://arxiv.org/pdf/1708.00404v4). For $|m|=\ell-1$, London
retains the spin-dependent 1.5PN expression.
[Mishra et al., arXiv:1601.05588v1, Eqs. (7), (11), (12a), (12d)](https://arxiv.org/pdf/1601.05588v1)
provide its SPA envelope. This paper equation and the numerical gate are
frozen here before implementation. The existing native default uses only
the leading term; a separately named paper option is required to preserve
the pinned binary-compatible default. This component alone does not qualify
the complete HM waveform.

## Inputs, conventions, and options

Let $x=Mf_{32}>0$ and $g(x)>0$ be the supplied (2,2) amplitude-map
frequency. The masses obey $m_1\ge m_2>0$, with
$\eta=m_1m_2/(m_1+m_2)^2\in(0,1/4]$. Aligned spins obey
$|\chi_i|\le1$ and $\chi_s=(\chi_1+\chi_2)/2$. Frequencies are
dimensionless geometric $Mf$. The public amplitude map has the existing
`phenomd_runtime` and `london_2018` cutoff choices; the printed PN option
will be `hm_32_pn_variant="mishra_eq12d"`, with
`"binary_compatible"` as the unchanged default. Both paths use the D
carrier. This component compares *magnitudes*, as London Eq. (3) prescribes.

## Mathematics and components

Set $v_y=V_2(y)=(\pi y)^{1/3}$ and
$P_{32}(v)=(1-3\eta)v^2+4\eta\chi_s v^3$.
Mishra Eq. (12d), truncated as London specifies, is
$\hat H_{32}(y)=-(1/3)\sqrt{5/7}\,P_{32}(v_y)$;
Eqs. (11) and (12a) give the complete SPA envelopes
$\mathcal H_{32}(y)=V_2(y)^{-7/2}\hat H_{32}(y)$ and
$\mathcal H_{22}(y)=-V_2(y)^{-7/2}$ at the retained order.
Since $m=2$, London's Eq. (4) second quotient is
$\mathcal H_{32}(x)/\mathcal H_{32}(2x/m)=1$ wherever it is defined.
The common SPA power cancels in the first quotient, leaving

$$
\beta_{32}(x;g)=\frac13\sqrt{\frac57}\,
\left|P_{32}\bigl((\pi g(x))^{1/3}\bigr)\right|.
$$

The leading nonspinning reduction is
$(\sqrt{5/7}/3)(1-3\eta)(\pi g)^{2/3}$. For $P\ne0$,
the one-sided derivative at either Eq. (5) map join is
$\operatorname{sgn}(P)(\sqrt{5/7}/3)
[2(1-3\eta)v+12\eta\chi_s v^2]v\,g'_{\pm}/(3g)$.
At a root of $P$, the magnitude has a cusp; that point is excluded from
the slope gate. Equal masses are valid and generally have nonzero (3,2).

| Component | Source | Scalar | Native production | Gate |
| --- | --- | --- | --- | --- |
| `hm.amp.beta32.1p5pn.eq4.v1` | London Eqs. (3)–(4); Mishra Eqs. (7), (11), (12a), (12d) | `diffgw_reference.imrphenomhm_amplitude_ratio` | Public Torch/JAX `IMRPhenomHM` D-carrier (3,2) with `hm_32_pn_variant="mishra_eq12d"` | `test_imrphenomhm_amplitude_ratio_math_reference.py` |

## Implementation mapping

The independent SymPy reference reduces the complete SPA quotient, then
checks the printed $4\eta\chi_s$ coefficient. An mpmath scalar evaluates
$P_{32}$, the Eq. (5) composition, and analytic one-sided slopes without
native helpers or fit data. The public gate divides the unprojected (3,2)
mode magnitude by a separately evaluated public PhenomD (2,2) carrier
magnitude at $g(x)$. It thus exercises the production-called ratio, not the
older public `h_hat_multipole` helper. Tests check the paper option and exact
default preservation in both frontends.

## Evidence and limitations

Use CPU FP64 on `len` at three predeclared rows:
$(m_1,m_2,\chi_1,\chi_2)=(35,20,0.2,-0.1)$,
$(35,20,0,0)$, and $(30,30,0.2,-0.1)$ (solar masses and dimensionless
spins), distance 100 Mpc, both map choices, and points below, at, between,
and above both Eq. (5) joins. Require public quotient versus mpmath
`rtol=1e-10, atol=1e-12`; for secants at geometric step $10^{-7}$,
require analytic one-sided slopes within `rtol=5e-5, atol=5e-8`.
Capture a pre-change red gate and post-change green gate. Independently
check exact SymPy reduction, the spin coefficient, the nonspinning limit,
and scalar domain. The pinned binary gate separately protects the default.

The public quotient takes the D carrier, amplitude map, and QNM frequencies
as supplied inputs. Their correctness, higher PN orders, phase/parity,
projection, LAL agreement, physical accuracy, and whole-HM qualification
are separate. No LAL source or output enters this equation gate.

## Reproduce the checks

On `len`, with reference, Torch, JAX, and test dependencies installed, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q
tests/test_imrphenomhm_amplitude_ratio_math_reference.py
--reference-device=cpu -ra -k beta32`. The validation receipt records the
exact host, checkout, environment, command, outcomes, and skips. A changed
equation, truncation, map convention, or tolerance requires versioned
review.
