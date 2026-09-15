# Published 2017 NQC target audit

**Scope.** Static comparison of `published_2017_nqc_targets` in
`src/torchwave/approximants/eob/teob_merger_nqc_clean.py` with Appendix A of
[Nagar et al. 2017, arXiv:1703.06814v1](https://arxiv.org/pdf/1703.06814v1),
and a version check against [Nagar et al. 2018,
arXiv:1806.01772v2](https://arxiv.org/html/1806.01772v2). No runtime or
oracle code was changed.

| Item | Primary-source evidence | Local status |
|---|---|---|
| Equal-mass targets | 2017 Appendix A, Table II: fourth-order polynomials in \(\hat a_0\) for \(A,\dot A,\omega,\dot\omega\). | Coefficients match the printed Table II values exactly to the shown digits. |
| Unequal-mass targets | 2017 Appendix A, Table IV gives \(A,\dot A,\dot\omega\); Eq. (A3) gives \(\omega\). | All local coefficients match Table IV and Eq. (A3) to the shown digits. Table III/Eq. (A2) is a different 2017 fit and is not the cited branch. |
| Branch choice | 2017 Appendix A states Table II for \(q=1\), Table IV alternative for \(q>1\). | `nu == .25` selects Table II; otherwise Table IV. This is source-consistent. |
| NQC event | 2017 Appendix A and Sec. IV: \(t^\mathrm{NR}_\mathrm{NQC}=t^\mathrm{NR}_\mathrm{mrg}+2M\), with merger at the peak of the \((2,2)\) modulus. | The helper returns target scalars only; callers must define the event and +2M offset. |
| Conventions | 2017 Sec. II/Appendix A defines positive \(\omega_{22}=M\dot\phi_{22}\) and \(A_{22}=|\Psi_{22}|/\nu\). | The positive frequency and documented `A_RW=A/(nu*sqrt(24))` are compatible when the caller's `A` is the strain-mode amplitude and \(\Psi=h/\sqrt{24}\). |

## Findings

There is no printed-coefficient transcription error in the 2017 target
formulas. The meaningful source discrepancy is the local domain guard:
`0.16 < nu <= 0.25`, equivalently `1 <= q < 4` (and it excludes the exact
`q=4` boundary). The 2017 discussion calls the Table IV fit the `q>1`
branch; its fit/sample tables include mass ratios through about `q=9.99`,
and the paper discusses reliability through `q=10`. Thus the `q<4` guard is
a compatibility-selection policy aligned with the 2018 replacement at
`q >= 4`, not a published 2017 fit-validity domain. It should be documented
as such if retained. Widening it would retain the legacy 2017 selection; it
would not by itself establish extrapolation accuracy.

The printed coefficients have eight decimal places. A coefficient-rounding
scale of at most `5e-9` per displayed coefficient follows from
round-to-nearest. The quoted `5e-9` (or `(1+nu)*5e-9` for affine-in-`nu`
terms) applies to zero-spin evaluation; at nonzero spin, the bound sums the
absolute polynomial basis terms times their coefficient bounds. These are
target-evaluation estimates, not rigorous nonlinear waveform error bounds.

## Version boundary

The 2018 paper's Appendix F.2 introduces a different factorized target-fit
representation and explicitly says those fits replace the 2017 fits for
`q >= 4` (Sec. II.B.4). Its fit representation and variables are therefore
not drop-in replacements for these 2017 constants; that difference alone
does not indicate an amplitude-normalization error in this helper. The local
function is correctly labeled as a 2017 helper. No source supports mixing
the 2018 coefficients into it while retaining the 2017 provenance.
