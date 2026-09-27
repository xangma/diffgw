# IMRPhenomNSBH branch amplitude factors

## Scope and status

Specification **nsbh-branch-amplitude-fits/v1** covers the adjusted ringdown
frequency, the four fit coordinates, and the three dimensionless branch
factors from Thompson et al. (2020), Appendix A Eqs. (20)–(24), (30), and
Table I. It excludes the Table I joining frequencies, Eq. (26) and (29)
corrections, PhenomC carrier, amplitude windows, remnant/QNM provider, phase,
and assembled waveform. This specification fixes the grid and limits before
native comparison.

## Inputs, conventions, and options

$q=m_1/m_2\ge1$, $\eta=q/(1+q)^2$, $\chi$ is the black-hole spin,
$C$ is neutron-star compactness, $T=M_{b,\rm torus}/M_{b,\rm NS}$ is the
nonnegative carrier ratio, and $f_{\rm tide},f_{\rm RD}>0$ use the same units.
$\Lambda\ge0$ is the neutron-star tidal deformability. The paper's fitted
domain is $1\le q\le15$, $|\chi|\le0.5$, $0\le\Lambda\le5000$.

The synthetic grid fixes $q=2$, $\chi=0.2$, $C=0.17$,
$f_{\rm tide}=1000$ Hz, $f_{\rm RD}\in\{800,1000,1200\}$ Hz,
$T\in\{0,0.05\}$ and $\Lambda\in\{0,0.5,1,2,800\}$: 30 cases. It
exercises all four Table I classes, equality, and both Eq. (20) branches.
The composition grid uses the 60 already specified physical inputs
$q\in\{1,2,5,10,15\}$, $\chi\in\{-0.5,0,0.5\}$,
$\Lambda\in\{300,800,2600,5000\}$, independent scalar $C$,
$f_{\rm tide}$ and clipped $T$, and controlled
$f_{\rm RD}/f_{\rm tide}\in\{0.8,1,1.2\}$: 180 cases. This is not a
test of the remnant-derived ringdown frequency. Both native gates use CPU FP64.

## Mathematics and components

[Thompson et al. (2020), Appendix A Eqs. (20)–(24), (30) and Table I](https://arxiv.org/pdf/2002.08383v1)
print

\[
\widetilde f_{\rm RD}=0.98f_{\rm RD}
\begin{cases}1-0.02\Lambda+0.01\Lambda^2,&\Lambda\le1,\\0.99,&\Lambda>1,\end{cases}
\quad u=\left(\frac{f_{\rm tide}-\widetilde f_{\rm RD}}
{\widetilde f_{\rm RD}}\right)^2,
\]
\[
x_{\rm ND}=u-0.571505C-0.00508451\chi,\quad
x'_{\rm ND}=u-0.657424C-0.0259977\chi,
\]
\[
x_{\rm D}=T+0.424912C+0.363604\sqrt\eta-0.060559\chi,
\]
\[
x'_{\rm D}=T-0.132754C+0.576669\sqrt\eta-0.0603749\chi
-0.0601185\chi^2-0.0729134\chi^3.
\]

With $\omega^\pm_{x_0,d}(x)=[1\pm\tanh(4(x-x_0)/d)]/2$,
Table I assigns

| Class | $\epsilon_{\rm tide}$ | $\epsilon_{\rm ins}$ | $\sigma_{\rm tide}$ |
| --- | --- | --- | --- |
| non-disruptive, no torus | $\omega^+_{-0.0796251,0.0801192}(x_{\rm ND})$ | 1 | $\omega^-_{-0.206465,0.226844}(x'_{\rm ND})$ |
| mildly disruptive, torus | same | $1.29971-1.61724x_{\rm D}$ | same |
| mildly disruptive, no torus | 0 | $1.29971-1.61724x_{\rm D}$ | half the sum of the high-branch value and $0.137722-0.293237x'_{\rm D}$ |
| disruptive, torus | 0 | $1.29971-1.61724x_{\rm D}$ | $0.137722-0.293237x'_{\rm D}$ |

The Table I note and Eq. (30) require the half factor in each window. The
torus ratio's nonnegative clip is an existing carrier adapter, not Eq. (23).

## Implementation mapping

An independent high-precision scalar evaluator will implement the printed
equations. Torch and JAX `approximants.imrphenom_nsbh` will expose a helper
used by the production amplitude assembly. It consumes the already gated
branch predicates and numeric inputs; neither native module imports the
scalar reference.

## Evidence and limitations

Before comparison, the gate fixes maximum relative error
$2\times10^{-13}$ for $\widetilde f_{\rm RD}$ and maximum absolute error
$2\times10^{-12}$ for each of $x_{\rm ND},x'_{\rm ND},x_{\rm D},
x'_{\rm D},\epsilon_{\rm tide},\epsilon_{\rm ins},\sigma_{\rm tide}$.
The 70- versus 90-digit scalar results must agree within $10^{-30}$
absolutely. No tolerance may be enlarged after observing the native output.
The existing disruption and torus gates support the composition inputs; this
gate does not certify those fits' physical calibration, the remnant/QNM
model, joining frequencies, amplitude assembly, or a binary waveform.

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src pytest -q
tests/test_nsbh_branch_amplitude_fits.py` with reference, Torch and JAX extras
on CPU. A validation receipt will record measured results and versions.
