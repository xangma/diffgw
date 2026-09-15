# Clean TEOB remnant compatibility contract

This contract covers the opt in, nonspinning remnant helper.  It is derived
from the public [Healy, Lousto and Zlochower paper](https://arxiv.org/pdf/1406.7295),
arXiv:1406.7295v1, Eqs. (14) and (16), Table VI, and the public
[Jiménez-Forteza paper](https://arxiv.org/pdf/1611.00332),
arXiv:1611.00332v2, Eq. (7), Table I.  It contains no implementation source,
fit reconstruction, or output-derived coefficient.

For positive component masses, define

\[
  \eta = \frac{m_1m_2}{(m_1+m_2)^2},\qquad
  \delta_m = \frac{m_1-m_2}{m_1+m_2}.
\]

The zero-spin reductions of Healy et al. use the printed Table VI values

\[
 (M_0,K_{2d},K_{4f})=(0.951507,1.995246,2.980990),
 \quad
 (L_0,L_{2d},L_{4f})=(0.686710,0.801838,0.953458).
\]

With the final spin \(a\), radius \(r\), specific energy \(E\), and angular
momentum \(J\), the equations evaluated by the helper are

\[
 a=(4\eta)^2(L_0+L_{2d}\delta_m^2+L_{4f}\delta_m^4)
    +\eta J(r,a)\delta_m^6,
\]

\[
 \frac{M_f}{M}=(4\eta)^2(M_0+K_{2d}\delta_m^2+K_{4f}\delta_m^4)
    +[1+\eta(E(r,a)+11)]\delta_m^6.
\]

The implicit spin equation is solved by a bounded fixed point iteration and
fails explicitly when its convergence budget is exhausted.

The paper branch, named `paper-prograde`, uses the signed Bardeen expression

\[
 r=3+Z_2-\operatorname{sgn}(a)\sqrt{(3-Z_1)(3+Z_1+2Z_2)}.
\]

The TEOB compatibility branch, named `teob-compatibility`, fixes the branch
sign to zero and uses \(r=3+Z_2\).  This radius is retained as a named
compatibility convention; it is not described as the physical prograde ISCO.
For that branch, `E(r,a)` is the general circular Kerr energy and retains its
spin argument.  The angular momentum is the published ISCO expression

\[
 J(r,a)=\frac{2(3\sqrt r-2a)}{\sqrt{3r}},
\]

evaluated at the named radius.  At the physical paper ISCO these expressions
reduce to the standard Kerr ISCO quantities.

The circular energy used by the compatibility branch is

\[
 E(r,a)=\frac{r^{3/2}-2\sqrt r+a}
 {r^{3/4}\sqrt{r^{3/2}-3\sqrt r+2a}}.
\]

The direct Healy API returns its internally solved spin.  The explicit
`teob_hybrid_nonspinning_remnant` helper returns the compatibility-branch
Healy mass fraction together with the printed Table I Jiménez-Forteza spin,
using `a2=3.833`, `a3=-9.49`, and `a5=2.513` in Eq. (7).  This printed-table
function is separate from the existing full-precision standalone UIB2016
helper and does not change that helper's behaviour.

The implementation is float64, supports broadcastable real tensor inputs on a
shared device, preserves autograd paths, and rejects nonpositive or nonfinite
masses.  It is an opt in component; production dispatch and existing remnant
defaults remain unchanged.
