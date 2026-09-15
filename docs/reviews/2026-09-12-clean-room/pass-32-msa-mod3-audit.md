# Pass 32: MSA FinalSpinMod3 source audit

Date: 2026-09-13  
Scope: clean-room audit of an explicit `final_spin_mod=3` path for
IMRPhenomXP/XPHM. No LAL, PyCBC, or TEOBResumS source was inspected or used.

## Primary-source evidence

The governing primary source is Pratten et al., *Computationally efficient
 model of precessing binary black holes*, arXiv:2004.06503v2:
 [HTML version](https://arxiv.org/html/2004.06503v2).

* Its final-spin construction first defines
  \(\lvert S_{\rm fin}\rvert=M_{\rm fin}^2\lvert a_{\rm fin}\rvert\) and
  \(S_{\parallel}+L_{\rm fin}=M_{\rm fin}^2a_{\rm fin}^{\parallel}\)
  (Eqs. 30--31). Thus the orbital angular momentum in this construction is
  tied to the **final** mass.
* The MSA prescription is the averaged algebra
  \[
  \langle S_{\rm fin}^2\rangle_{\rm pr}
   =S_{\rm av}^2+L_{\rm fin}^2
    +2L_{\rm fin}(\langle S_1\!\cdot\!\hat L\rangle_{\rm pr}
                  +\langle S_2\!\cdot\!\hat L\rangle_{\rm pr}),
  \]
  Eq. 33, and the paper labels the corresponding magnitude formula
  \(\lvert a_{\rm fin}\rvert=\sqrt{\langle S_{\rm fin}^2\rangle_{\rm pr}}/M^2\)
  as the MSA default (Eq. 42). Earlier in the same construction, however,
  \(M\) is defined as the initial total mass. The source does not state that
  (M=M_{\rm fin}), nor provide a conversion or erratum resolving this
  normalization mismatch.
* The source explicitly warns that the equal-mass \(q=1\) MSA expressions
  have singular precession-averaged spin couplings and says they “must be
  regularized in the equal-mass limit”; it gives no regularization formula.
  It also says the MSA system can be numerically unstable near misalignment,
  with fallback or error behavior selected by implementation. This is not an
  analytic continuation prescription.
* The source specifies the final-spin sign by agreement with the
  nonprecessing final spin, but gives no tie-break for an exactly zero
  nonprecessing final spin. Its single-spin/aligned alternatives do not define
  a unique replacement for the MSA averaged couplings in singular rows.
* Table 3 distinguishes MSA versions that fall back to NNLO from versions
  that raise on initialization failure. That policy covers the MSA angle
  evolution; it does not define whether a hypothetical FinalSpinMod3 path
  should use fallback couplings, a separately computed remnant, or a limiting
  branch.

## Decision

The generic Eq. 33 algebra is already represented by TorchWave's
`compute_msa_final_spin_squared` helper, but a source-complete public
`final_spin_mod=3` implementation is **not** uniquely defined. In particular,
the source leaves all of the following unresolved:

1. whether Eq. 42 must use the initial (M^2) as printed or the (M_{\rm fin}^2)
   implied by Eqs. 30--31;
2. how to regularize the equal-mass singular couplings;
3. which continuation to use for one-spin, aligned-root, and other degenerate
   MSA rows; and
4. how sign and fallback/error policy should be coupled to those continuations.

Consequently this pass makes **no runtime change** and preserves the current
API behavior: `final_spin_mod=0` remains the explicit NNLO mode, while an
explicit `final_spin_mod=3` remains rejected rather than silently selecting an
unsupported normalization or degenerate-row guess. The current default is
unchanged.

The existing clean-room tests cover the implemented Eq. 33 helper, batching,
gradients, degenerate initialization/fallback behavior, and the wrapper's
explicit Mod3 rejection. No new runtime test is warranted for an undefined
continuation.
