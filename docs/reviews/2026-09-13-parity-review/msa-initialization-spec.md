# Clean-room MSA initialization specification

This note transcribes the initialization contract from Chatziioannou et al.,
arXiv:1703.03967v3, §§II–III and Eqs. (13), (16), (18), (21), (22),
(25)–(27), and (35).  It is a mathematical input specification for a fresh
implementation; it does not describe or rely on an inherited implementation.

## Normalized inputs and invariants

Use geometric units and set the total mass to one while constructing the
dimensionless dynamics.  Order the bodies so that

\[
 q=m_2/m_1\leq1,\qquad \eta=m_1m_2,
 \qquad \delta m=m_1-m_2,
\]

with the ordered masses rescaled by `m1+m2`.  The dimensionless spin
vectors are \(\mathbf S_A=m_A^2\boldsymbol\chi_A\), and
\(\mathbf J=\mathbf L+\mathbf S_1+\mathbf S_2\).  The seven instantaneous
invariants used to initialize the conservative solution are

\[
 (S_1,S_2,L,J,\hat{\mathbf J},\xi),\qquad
 \xi=(1+q)\mathbf S_1\!\cdot\!\hat{\mathbf L}
 +(1+q^{-1})\mathbf S_2\!\cdot\!\hat{\mathbf L}.
\]

For the radiation-reaction extension, \(S_1,S_2,\xi\) remain fixed (at the
order retained and neglecting horizon absorption), \(L=\eta/v\), and
\(\hat{\mathbf J}\) is taken along the inertial-frame z axis.  The supplied
initial vectors must satisfy their stated magnitudes and these vector sums;
degenerate zero-spin or collinear configurations require separate limiting
branches because the angular constructions divide by spin and triangle
factors.

## Spin cubic and root selection

At fixed conservative parameters, the total-spin magnitude obeys Eq. (21):

\[
 \left(\frac{dS^2}{dt}\right)^2=-A^2(S^6+B S^4+C S^2+D).
\]

Factor the cubic in \(S^2\) as Eq. (22),

\[
 -A^2(S^2-S_+^2)(S^2-S_-^2)(S^2-S_3^2).
\]

The physical oscillation is bounded by the two turning roots \(S_+^2\) and
\(S_-^2\), with \(S_+^2\ge S_-^2\); the third root is normally negative.
Select roots by the physical interval and require
\(m=(S_+^2-S_-^2)/(S_+^2-S_3^2)\in[0,1]\), rather than sorting numerical
roots without checking the interval.  The paper calls \(S_+\) and \(S_-\)
the maximum and minimum physical values; its later “largest/smallest” wording
refers to the separated third root and should not replace this bounded-interval
check.

## Initial Jacobi phase and oscillation

The leading solution is Eq. (23):

\[
 S^2=S_+^2+(S_-^2-S_+^2)\operatorname{sn}^2(\psi,m),\quad
 \dot\psi=\frac A2\sqrt{S_+^2-S_3^2},\quad
 m=\frac{S_+^2-S_-^2}{S_+^2-S_3^2}.
\]

Determine \(\psi_0\) from the supplied initial \(S_0^2\) by inverting the
`sn²` relation.  Choose the inverse branch using Eq. (24):

\[
 \operatorname{sign}(\sin\phi'_0)
 =\operatorname{sign}[(\hat{\mathbf L}_0\times\mathbf S_{1,0})\cdot\mathbf S_{2,0}]
 =\operatorname{sign}[-\dot S_0^2]
 =\operatorname{sign}[\operatorname{sn}(\psi_0,m)\operatorname{cn}(\psi_0,m)].
\]

Thus the initial triple product fixes the sign branch; at a turning point it
is zero and the continuation direction must be supplied by the limiting
derivative.  In the no-radiation-reaction solution the period of \(S^2\) is
\(2K(m)\).  Under MSA, promote the roots and \(A\) to slow-time quantities,
retain the same elliptic form, and use the paper's slow-time phase equation
(Eq. (35)).

There is a sign-convention mismatch between the paper's displayed negative
coefficient in the precession-rate discussion and the positive derivative of
the printed phase function.  The clean initializer uses the positive rate

\[
 |A|\sqrt{S_+^2-S_3^2}/2,
\]

on the ordinary branch, then applies Eq. (24)'s triple-product sign to the
initial inverse.  This fixes a convention without changing the invariant
cubic or fitting a time shift; callers using the opposite Fourier convention
should negate the phase consistently.

## Precession averages required at initialization

The leading precession average used for the slow evolution is Eq. (27),

\[
 S^2_{\rm av}=\frac12(S_{+,0}^2+S_{-,0}^2)+O(v^2),
\]

while the exact elliptic average is Eq. (26), involving complete elliptic
integrals \(E(m)\) and \(K(m)\).  For the inertial precession phase, retain
the oscillatory \(\operatorname{sn}^2\) and \(\operatorname{sn}^4\) terms in
Eq. (25); the paper defines the corresponding normalized averages \(D_2\)
and \(D_4\) in Eqs. (49)–(50).  Collinear limits, equal masses, vanishing
spin, repeated roots, and \(m\to0,1\) need explicit continuous limiting
tests before use in an MSA initializer.

## First tests for a fresh module

1. Random valid ordered masses and nondegenerate vectors preserve the seven
   invariants and reconstruct \(\mathbf S_2=\mathbf J-\mathbf L-\mathbf S_1\).
2. The cubic factorization reproduces Eq. (21); selected roots produce
   \(0\le m\le1\) and bounded \(S^2\).
3. Recovered \(\psi_0\) reproduces \(S_0^2\), and reversing the initial
   triple-product sign selects the opposite `sn cn` branch.
4. Numerical period averages converge to Eq. (26), with the leading result
   approaching Eq. (27) as \(v\to0\); test the repeated-root and collinear
   limits separately.
