# Closed `P^1_5` orbital potential derivation

The printed PN series is

\[
A(u)=\sum_{k=0}^{6}c_k u^k,
\quad(c_0,c_1,c_2,c_3,c_4,c_5,c_6)
=(1,-2,0,2\nu,\nu a_4,\nu L_5,\nu L_6),
\]

where `L_5 = a_5^c + (64/5) log(u)` and `L_6 = a_6^c + a_6^log log(u)`
use the published coefficient expressions in `teobresums.py`. Write

\[
P^1_5[A](u)=\frac{1+n_1u}{1+d_1u+d_2u^2+d_3u^3+d_4u^4+d_5u^5}.
\]

Matching the coefficients of `D(u)A(u)-N(u)` at powers `u^2,...,u^6` gives

\[
\begin{aligned}
d_2&=2d_1,\\
d_3&=4d_1-c_3,\\
d_4&=(8-c_3)d_1-2c_3-c_4,\\
d_5&=(16-4c_3-c_4)d_1-4c_3-2c_4-c_5,\\
d_1&=\frac{c_6-c_3^2+8c_3+4c_4+2c_5}{32-12c_3-4c_4-c_5},\qquad n_1=-2+d_1.
\end{aligned}
\]

The module evaluates these rational expressions and both polynomials with
Horner's rule. For the logarithmic coefficients, `dc5/du = nu(64/5)/u`
and `dc6/du = nu a6^log/u`; quotient differentiation then gives `dA/du`.
The radial API applies the required coordinate identity
`dA/dr = -u^2 dA/du` for `u=1/r`. At `nu=0`, the closed expressions reduce
algebraically to `A=1-2u`, `dA/dr=2u^2`; this also retains the one-sided
mass-ratio derivative at the boundary.

`tests/test_teob_orbital_potential_closed_clean.py` independently solves the
five matching equations with 80-digit `mpmath` arithmetic and checks both
the value and radial derivative at representative radii and mass ratios,
including the logarithmic terms.
