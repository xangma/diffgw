# Clean-room TEOBResumS NQC matching

This component is derived from the rendered, public papers [arXiv:1406.6913](https://arxiv.org/abs/1406.6913), v1, Eqs. (96)--(97) and (100)--(106), and [arXiv:1806.01772](https://arxiv.org/abs/1806.01772), v2, Eq. (14) and Appendix F.  It uses equations and rendered prose only; it contains no copied implementation, LAL/PyCBC/TEOB source, legacy model body, or fitted constants from LAL.

The basis is

\[
n_1=\left(\frac{p_r^\ast}{r\Omega}\right)^2,\qquad
n_2=\frac{\ddot r^{(0)}}{r\Omega^2},\qquad
n'_1=\frac{p_r^\ast}{r\Omega},\qquad
n'_2=p_r^\ast r\Omega,
\]

and the factor is

\[
\hat h_{\rm NQC}=(1+a_1n_1+a_2n_2)
\exp\!\left[i(b_1n'_1+b_2n'_2)\right].
\]

The solver imposes amplitude, amplitude derivative, frequency, and frequency
derivative osculation as two separate 2x2 linear systems.  It uses
`h=A exp(-i phase)`, so the frequency equations solve
`b dot(d n_prime/dt) = omega - omega_target`; this is the sign implied by the
paper's `+i` factor.  The basis derivatives are explicit inputs.  No
interpolation, finite-difference stencil, trajectory evolution, peak search,
or NQC feedback into radiation reaction is hidden here.

All amplitude conditions must use one common convention.  Appendix F of
1806.01772 defines its nonspinning extraction amplitude as the RW-normalized
quantity `|Psi_hat_22| = |h_hat_22|/sqrt(24)`.  The module provides
`rw_normalize_22` and `rw_denormalize_22` as explicit conversions and inserts
no `nu`, distance, or `sqrt(24)` factor into matching.  The caller supplies
base and target conditions in the same convention.

For the related fresh `teob_postmerger.nqc_targets` contract, Appendix F
Eqs. (F24)--(F25) and Table X give

\[
\dot{\hat A}_{22}^{\rm NQC}=\omega_{22}^{\rm NQC}
\left[-\frac{-0.00421428-0.0847947\nu}{1+16.1559\nu}\right].
\]

Thus its positive nonspinning value near \(\nu=1/4\) follows the two minus
signs in the rendered fit; it is not a source-level sign defect.  Table X's
derivative heading includes ``/\nu``; we treat that as the paper's annotation
for the already \(\nu\)-normalized hatted waveform used by Eqs. (F24)--(F25),
so the target helper applies no additional \(\nu\) factor.  A caller using
raw h22 amplitudes must convert both amplitude and derivative to the
documented hatted/RW convention first.  The nonspinning timing convention is
`t_NQC=t_peak,Omega_orb-1M`, with the chart merger label `t_merger=t_NQC-2M`
when the NR extraction point is merger+2M.
