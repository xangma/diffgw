# Independent MSA PN coefficient derivation

Sources: Chatziioannou et al. [1703.03967v3](https://arxiv.org/abs/1703.03967)
Appendices A–C and [1307.4418v2](https://arxiv.org/abs/1307.4418) Appendix A.
This specifies equations for a fresh implementation, without reference to
inherited or external implementation code.

Use total mass one, q=m2/m1<1, eta=q/(1+q)^2, delta=(1-q)/(1+q),
and squared dimensionful-normalized spins s1=|S1|^2, s2=|S2|^2.
The precession averages in 1703 Appendix A provide mean projections u1,u2,
their squares/cross product, and mean S1 dot S2=(Sav-s1-s2)/2.
Insert these directly in beta3, beta5, sigma4 of 1307 Appendix A, then
construct a0,a2,a3,a4,a5 and reciprocal coefficients g0..g5 from 1703
Appendix A. g1=0. These are flux coefficients, without phenomenological fits.

## Spin phase and a printed coefficient discrepancy

Write J^2=eta^2/v^2+2c1/v+Sav and L=eta/v in the Appendix B cubic
x^3+B(v)x^2+C(v)x+D(v). Its leading coefficients are

```
bminus2 = eta*delta^2
bminus1 = 2*eta*xi - 4*c1
b0 = q*s1+s2/q-s1-s2-2*Sav
cminus2 = 4*c1^2-4*eta*c1*xi+4*eta^3*xi^2
          -2*eta*delta^2*(s1+s2)
dminus2 = 4*(1-q)/q*(s2-q*s1)*c1^2
          +eta*delta^2*(s1-s2)^2+4*delta*eta*c1*xi*(s1-s2)
```

The finite limiting roots p>=n solve
bminus2*x^2+cminus2*x+dminus2=0. Vieta's identity then gives

```
sqrt(Splus^2-Sthird^2)
 = sqrt(eta)*delta/v * (1+r1*v+r2*v^2+O(v^3))
r1 = bminus1/(2*bminus2)
r2 = (b0+2*p+n)/(2*bminus2)-bminus1^2/(8*bminus2^2)
psi1 = 3*(r1-xi)/2
psi2 = 3*(r2-xi*r1+g2/g0)
raw_psi(v) = -3*g0*delta/4 * v^-3*(1+psi1*v+psi2*v^2)
psi(v) = psi_initial + raw_psi(v)-raw_psi(v0)
```

The sign is fixed by the positive derivative of the displayed PN phase and
the initial triple-product condition, checked against the conservative vector
spin equations. The printed negative A in Appendix B is used in magnitude
for this phase convention.

An independent symbolic reduction shows this psi2 equals Appendix C if the
first term of the second factor in Delta is c1^2*eta*q/delta^4. The arXiv v3
source prints c1^2*eta^2/delta^4. This correction follows from the quadratic
discriminant: disc=16*delta^8/eta^2 * Delta^2. No public waveform output was
used to determine it. Implementing the quadratic expansion above avoids
transcribing that questionable factor. Public compatibility remains a
separate validation question.

Required checks include reciprocal-series multiplication through v^5,
the independent leading-cubic expansion, phase anchoring, and float64
batch/gradient comparisons. Singular equal-mass and collinear limits must
be handled explicitly before production use.
