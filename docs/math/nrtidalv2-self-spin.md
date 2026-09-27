# NRTidalv2 self-spin in the published D composition

## Scope and status

Specification **`nrtidalv2-self-spin-d/v1`** covers the positive-deformability,
aligned-spin, dominant-mode Fourier self-spin component in the *printed*
`IMRPhenomD_NRTidalv2` composition. It covers CPU FP64 scalar phase components
and Torch/JAX complex quotient agreement on the frozen systems below. It is a
**conditional equation-level gate**: the paper's Fig. 7 example is inconsistent
with its printed Eqs. (28)–(29), and this gate does not validate the universal
fit against neutron-star data or the complete waveform against a binary oracle.

## Inputs, conventions, and options

[Dietrich et al. (2019), arXiv:1905.06011v2, Eqs. (26)–(29) and Table II](https://arxiv.org/html/1905.06011)
are the primary source. $m_A,m_B>0$ may be passed in either order;
$X_i=m_i/M$, $M=m_A+m_B$, $\nu=X_AX_B$, $x=(\pi M_{\rm sec}f)^{2/3}$,
$f>0$, and aligned dimensionless spins $\chi_i$ use the native signed
convention. The printed Love–Q relation requires $\Lambda_i>0$. The gate tests
$\Lambda_i\ge180$; zero and tiny values use native clamps outside this paper
reference. The paper D composition retains the 2PN and 3PN EOS self-spin terms,
but not the Pv2 3.5PN quadrupole tail/cubic-spin terms (Table II). Its tidal
amplitude is omitted. The historical `binary_compatible` default is preserved.

The existing API calls this choice `amplitude_variant`. For v2,
`amplitude_variant="paper_2019"` now selects the entire printed **D** finite-size
composition: Eq. (28) quadrupole fit, 2PN+3PN self-spin, and no Eq. (24) tidal
amplitude. `binary_compatible` retains the inherited earlier Love–Q fit,
3.5PN self-spin and historical tidal amplitude. This option name predates the
self-spin qualification and remains for compatibility.

## Mathematics and components

For each body, let $\widehat C_Q=C_Q-1$ and $\widehat C_{Oc}=C_{Oc}-1$.
Dietrich Eq. (28) prints

\[
\log C_Q=0.1940+0.09163\ell+0.04812\ell^2
-0.004286\ell^3+0.00012450\ell^4,\qquad \ell=\log\Lambda.
\]

Equation (29) prints

\[
\log C_{Oc}=0.003131+2.071q-0.7152q^2+0.2458q^3-0.03309q^4,
\qquad q=\log C_Q.
\]

Equation (27) gives the per-body coefficients

\[
P_{2,i}=-50\widehat C_{Q,i}X_i^2\chi_i^2,\quad
P_{3,i}=\frac5{84}(9407+8218X_i-2016X_i^2)
\widehat C_{Q,i}X_i^2\chi_i^2,
\]

\[
P_{3.5,i}=10\left[(X_i^2+\frac{308}{3}X_i)\chi_i+
(X_j^2-\frac{89}{3}X_j)\chi_j-40\pi\right]
\widehat C_{Q,i}X_i^2\chi_i^2
-440\widehat C_{Oc,i}X_i^3\chi_i^3.
\]

The independent scalar reference evaluates each order separately with
$K=3/(128\nu x^{5/2})$: $S_2=Kx^2\sum_iP_{2,i}$,
$S_3=Kx^3\sum_iP_{3,i}$, and $S_{3.5}=Kx^{7/2}\sum_iP_{3.5,i}$.
`nrtidalv2.self-spin.eq26-29.printed/v1` comprises these components. The
paper D quotient uses $\exp[-i(\psi_T+S_2+S_3)]$ before taper; $S_{3.5}$ is
retained only as a separate Eq. (27) component for the published Pv2 scope.
At `phasing_convention="reference_frequency"`, the native D adapter instead
uses $S_2(f)+S_3(f)-S_2(f_{ref})-S_3(f_{ref})$; this subtraction is a
project phase convention, not Eq. (26).

## Implementation mapping

The independent mpmath scalar reference is
`src/diffgw_reference/nrtidalv2_self_spin.py`. Its printed fit and separated
Eq. (27) terms are compared with
`src/torchwave/approximants/imrphenomd_nrtidal.py` and
`src/jaxwave/approximants/imrphenomd_nrtidal.py`. The maintained gate
`tests/test_nrtidalv2_self_spin_math_reference.py` also compares the full
pre-taper complex BNS/BBH quotient using the separate
`diffgw_reference.nrtidalv2_phase` component. Neither native implementation
imports the reference. `tests/test_nrtidal_reference.py` remains a regression
for the historical default and is labelled accordingly.

## Evidence and limitations

The frozen cases $(m_1,m_2;\chi_1,\chi_2;\Lambda_1,\Lambda_2)$ are
$(1.35,1.35;0.20,0.20;350,350)$,
$(1.6,1.2;0.04,-0.03;250,800)$, its exchanged copy, and
$(1.45,1.25;0,0;300,650)$. Frequencies are 30, 100, 300 and 800 Hz.
The absolute component, unaligned complex quotient, and spin-derivative limits
are $10^{-12}$, $10^{-10}$, and $10^{-10}$ respectively. The quotient is tested
in coalescence and 30 Hz reference-frequency conventions, without fitting.
The [execution receipt](../validation/nrtidalv2_self_spin_20260927.md)
records the actual result.

At $\Lambda=350$, direct evaluation of the **printed** Eqs. (28)–(29) gives
$(C_Q,C_{Oc})\simeq(5.29659,10.49832)$; the Fig. 7 description instead
quotes $(4.30,7.28)$. This discrepancy is recorded by a diagnostic test and
is unresolved. It limits source-fidelity interpretation beyond the equations
as printed. A passed scalar gate does not certify the EOS-insensitive relation,
physical BNS accuracy, zero-$\Lambda$ behavior, binary agreement, taper, or
the full waveform. The historical default remains separately qualified where
its existing evidence supports it.

## Reproduce the checks

With CPU FP64 Torch, JAX, mpmath and pytest installed, run
`PYTHONPATH=src:tests python -m pytest -q
tests/test_nrtidalv2_self_spin_math_reference.py tests/test_nrtidal_reference.py`.
