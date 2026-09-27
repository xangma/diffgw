# IMRPhenomNSBH compactness and disruption root

## Scope and status

Specification **nsbh-compactness-disruption-root/v1** covers only the
neutron-star compactness fit, the largest positive relativistic tidal root,
and the tidal radius/frequency derived from that root. It does not qualify
the NSBH amplitude branches, remnant or ringdown fits, phase gauge, or full
waveform. The independent paper gate below is separate from any comparison
with a public LALSuite waveform binary.

## Inputs, conventions, and options

The black hole is body 1, the nonspinning neutron star is body 2,
$q=m_1/m_2$, $\chi$ is the aligned black-hole spin, and $\Lambda$ is the
neutron-star dimensionless tidal deformability. CPU FP64 is required for both
native backends. The frozen Cartesian gate is
$q\in\{1,2,5,10,15\}$,
$\chi\in\{-0.5,0,0.5\}$, and
$\Lambda\in\{0,1,10,300,800,2600,5000\}$.
The compactness and root use all 105 cases. Radius and frequency use the
60 cases with $\Lambda\ge300$ and $m_2=1.4M_\odot$. The paper's calibration
domain is $1\le q\le15$, $|\chi|\le0.5$, and $0\le\Lambda\le5000$.
The frequency gate requires a positive, finite denominator in Eq. (8);
all 60 frozen cases must satisfy that precondition.

## Mathematics and components

[Thompson et al. (2020), Appendix B Eq. (32)](https://arxiv.org/pdf/2002.08383v1)
define $C(\Lambda)$ by the cubic below for $\Lambda\le1$ and the logarithmic
fit for $\Lambda>1$, with $(a_0,a_1,a_2)=(0.360,-0.0355,0.000705)$:

\[
C(\Lambda)=\begin{cases}
\tfrac12+(3a_0-a_1-\tfrac32)\Lambda^2+
(a_1-2a_0+1)\Lambda^3,&\Lambda\le1,\\
a_0+a_1\ln\Lambda+a_2(\ln\Lambda)^2,&\Lambda>1.
\end{cases}
\]

With $\mu=qC$, Appendix A Eq. (10) defines $\xi$ as the **largest positive
real root** of

\[
P(\xi)=\xi^5-3\mu\xi^4+2\chi\sqrt{\mu^3\xi^7}
-3q\xi^2+6q\mu\xi-3q\mu^2\chi^2=0.
\]

The independent scalar reference uses $y=\sqrt\xi$ and brackets the
rightmost crossing of the resulting degree-ten polynomial from a Cauchy
upper bound, then bisects at high precision. This does not share the native
fixed-iteration Newton solver. Appendix A Eqs. (8)–(9) give
$\widetilde r_{\rm tide}=\xi m_{2,\rm sec}(1-2C)/C$ and
$f_{\rm tide}=[\pi(\chi m_{1,\rm sec}+
\sqrt{\widetilde r_{\rm tide}^3/m_{1,\rm sec}})]^{-1}$.
The positive-frequency native adapter is immaterial on the frozen
positive-denominator frequency grid.

## Implementation mapping

`src/diffgw_reference/nsbh_disruption.py` evaluates the published scalar
expressions without importing a production backend or LALSuite. The Torch
and JAX paths are `ns_compactness_from_lambda`, `solve_xi_tide`, and
`nsbh_frequencies` in their respective `imrphenom_nsbh.py` modules. Tests
call these components directly, without an IMRPhenomD carrier or assembled
NSBH waveform.

## Evidence and limitations

The gate requires maximum absolute compactness error $\le2\times10^{-13}$,
maximum relative root error $\le2\times10^{-11}$, and normalized native
polynomial residual $\le2\times10^{-12}$. The normalization is one plus
the sum of absolute polynomial terms at the returned root. On the 60-case
frequency subset, maximum relative radius and frequency errors must each be
$\le2\times10^{-10}$. A separate degree-ten polynomial root enumeration
checks that the scalar bracket selected the largest positive root. These
thresholds and cases were fixed before any public-binary evaluation.

An assembled LAL NSBH waveform does not expose $C$, $\xi$, or
$f_{\rm tide}$ as a public observable, so waveform matching cannot serve as
an independent oracle for this component. Binary comparison belongs to a
separate composition gate after the remnant, amplitude and phase components
have their own references. Numerical agreement here does not validate the
underlying NR calibration or imply whole-waveform qualification.

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src pytest -q
tests/test_nsbh_disruption_reference.py` with the `reference`, `torch`, and
`jax` extras on CPU. The measured result and environment are recorded in
the linked validation receipt once the gate is run.
