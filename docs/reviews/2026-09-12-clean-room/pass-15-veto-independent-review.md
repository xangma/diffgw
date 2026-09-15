# Independent primary-source review: IMRPhenomXHM intermediate amplitude vetoes

Source reviewed: arXiv:2001.10914v2, Sec. V A, Eqs. (4.5), (4.6), (5.3), (6.2), Tables III–VI, pp. 8 and 11–15. This review uses the paper only.

## Amplitude normalization and units

The paper uses two related normalizations:

- Eq. (4.5) defines
  \[
  A_{22}^{0}(f)=A_0(\pi f)^{-7/6},\qquad A_0=\pi\sqrt{2\eta/3}.
  \]
- Eq. (4.6) defines the nonnegative, low-frequency-regular quantity
  \(H_{\ell m}=|A^{\rm SPA}_{\ell m}|/A_{22}^{0}(f)\).
- The intermediate ansatz in Eq. (5.3) is instead explicitly
  \(A^{\rm Inter}_{\ell m}/A_0\), with the frequency-independent \(A_0\). Table IV gives every collocation value as \(v_i=A^{\rm Inter}_{\ell m}(f_i)/A_0\) and every derivative as \(d_i=(A^{\rm Inter}_{\ell m}/A_0)'(f_i)\).
- Eq. (6.2) writes the ringdown ansatz as \(A^{\ell m}_{\rm RD}/A_{22}^{0}\).

Therefore the source-supported conversion at a frequency \(f\) is
\[
A_{\ell m}(f)=A_0\,v_{\rm int}(f)
 =A_0(\pi f)^{-7/6}H_{\ell m}(f)
\]
when \(H_{\ell m}\) denotes the Eq. (4.6)/(6.2) normalized quantity. The paper does not explicitly state this conversion in the veto prose, but it follows directly from its displayed definitions.

The numeric veto thresholds in Table VI remain unit-ambiguous. The surrounding section reconstructs \(A/A_0\), but the prose calls the 0.2 quantity the “strain Fourier domain amplitude \(|\tilde h_{21}|\)” and never writes \(A\), \(A/A_0\), or \(H_{21}\) beside either threshold. Thus the primary source does not settle raw \(A\) versus \(A/A_0\), and it does not justify silently treating 0.2 or 0.01 as thresholds on \(H_{\ell m}\). If a stored value is \(H\), the algebraic conversion to the intermediate normalization would be \(A/A_0=H(\pi f)^{-7/6}\), but choosing which quantity the constants apply to requires an implementation decision beyond this paper. The source also does not specify a distance-scaled physical strain convention for these checks.

## Literal ordered veto table

Table VI says the vetoes are sorted in order of application. A collocation point is “still alive” exactly when a previous veto has not removed it. The inequalities below are literal strict inequalities from the tables.

| Order | Check / parameter region | Modes | Action on surviving intermediate reconstruction |
|---|---|---|---|
| 1 | For each of \(f^{\ell m}_{\rm Int1}\), \(f^{\ell m}_{\rm Int2}\), test whether the amplitude is \(<0.2\); Table VI region \(q<8\) | 21 | Remove each collocation point that fails. The prose says “at which”, and Table VI’s \(f_{\rm Int1,2}\) is shorthand for the individually failing points; it is not an all-or-nothing pair test. |
| 2 | Amplitude at \(f^{\ell m}_{\rm RD}<0.01\) | 21 | Remove both \(f^{21}_{\rm Int1}\), \(f^{21}_{\rm Int2}\). |
| 2 | Same strict check, \(A_{32}(f^{32}_{\rm RD})<0.01\) | 32 | Remove both intermediate points **and the derivatives at the boundaries**. |
| 3 | Monotonicity, only if both \(f^{\ell m}_{\rm Int1}\) and \(f^{\ell m}_{\rm Int2}\) survived the preceding checks | 21, 32 | If the source's unspecified amplitude sequence is judged non-monotonic, remove only \(f^{\ell m}_{\rm Int2}\). |
| 4 | “\(a_\lambda\) badly behaved” in \(q>7\), \(\chi_1>0.95\) | 33, 44 | Remove both intermediate points; the text says to connect inspiral to ringdown with a third-order polynomial. |
| 5 | Parameter-space rows in Table V (below) | 21, 33, 32, 44 | Remove \(f_{\rm Int1}\) and \(f_{\rm Int2}\) that are still alive. |
| 6 | Resulting inverse-polynomial denominator crosses zero anywhere in the intermediate-frequency interval | all affected modes | Lower polynomial order by iteratively relaxing boundary conditions until a defined ansatz is obtained; the minimum order stated is one. Table VI summarizes this as removing boundary derivatives. |

Table V’s parameter-space rows are:

| Mode | Region | “Veto applied if” |
|---|---|---|
| 21 | \(\eta<0.23\), \(\chi_1>0.7\), \(\chi_2<-0.5\) | Always |
| 21 | \(q>40\), \(\chi_1>0.9\) | \(f_{\rm Int1,2}\) still alive |
| 33 | \(q>40\), \(\chi_1>0.9\) | \(f_{\rm Int1,2}\) still alive |
| 32 | \(q>2.5\), \(\chi_1<-0.6\), \(\chi_2>0\) | Always |
| 32 | \(\chi_1<-0.9\), \(\chi_2<-0.9\) | Always |
| 32 | \(q>40\), \(\chi_1>0.9\) | \(f_{\rm Int1,2}\) still alive |
| 44 | \(q>40\), \(\chi_1>0.9\) | \(f_{\rm Int1,2}\) still alive |

“Always” is the table’s distinction from “still alive”: it does not condition the parameter-space veto on survival from earlier checks. The target is still the pair of intermediate points; if one was already removed, only the remaining target can be removed in practice.

## Partial-node and derivative details

- The 21 amplitude check can remove either intermediate point independently (the prose says “collocation points at which” the amplitude is below 0.2). The 21/32 monotonicity check is reached only if both points remain; the paper does not define the sequence being tested or its required direction. Two unconstrained values alone are always monotonic in one of the two directions, so the intended test may include the boundary amplitudes at \(f_{\rm Ins}\) and \(f_{\rm RD}\) (yielding a four-value sequence), but that is an inference. The stated action for a failed test is to remove the higher-frequency point \(f_{\rm Int2}\). A later Table V row marked “still alive” must then remove any remaining point; it must not resurrect an already removed point.
- The \(32\) ringdown-amplitude failure is the one table row that explicitly removes both points **and boundary derivatives** together.
- Table V does not separately specify derivative removal for its parameter-space vetoes. The general post-veto zero-denominator fallback is the only later derivative instruction: relax boundary conditions/remove derivatives while lowering order. The paper does not define whether the two boundary derivatives are dropped together, which boundary is dropped first, or whether an individual derivative is tried before changing polynomial order.
- The prose says the \(32\) ringdown amplitude must be “above” 0.01, while Table VI expresses the veto as amplitude “\(<0.01\)”. Literal table semantics preserve equality at exactly 0.01; the prose leaves equality ambiguous. The same strict-table reading applies to the 21 0.2 and 0.01 rows.
- The monotonicity row is underspecified: the paper names only \(f_{\rm Int1}\) and \(f_{\rm Int2}\) as having passed, but does not say whether endpoint values at \(f_{\rm Ins}\) and \(f_{\rm RD}\) participate or which monotonic direction is required.
- Table VI explicitly restricts the 21 intermediate 0.2 check to \(q<8\), whereas the nearby paragraph says simply “For the (2,1) mode” and omits that qualifier. The table is the only ordered summary and should control this rule; the prose omission is an editorial ambiguity.
- “\(a_\lambda\) badly behaved” has no numerical test in this paper; only its mode/region and pair-removal action are specified.
