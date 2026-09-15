# Pass 12: XHM parameter-space amplitude vetoes

García-Quirós et al., [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2),
Sec. V.A.3 and Table V specify regions where both intermediate amplitude
collocation points are removed. This pass implements the unconditional subset:

- 21: eta < 0.23, chi1 > 0.7, chi2 < -0.5.
- 32: q > 2.5, chi1 < -0.6, chi2 > 0; or both spins below -0.9.
- 33 and 44: q > 7, chi1 > 0.95.

The production reconstruction selects its existing cubic Hermite polynomial,
retaining the two boundary amplitudes and derivatives. Canonical body ordering
feeds the masks; selection broadcasts across batches. Existing pole guards,
carrier normalization, phases, calibration tables and mode mixing are retained.

The Luna candidate also applied the paper's 0.2 and 0.01 amplitude thresholds
directly to the native H_lm ratios, implemented quartic/linear reconstructions,
and combined conditional q > 40 checks with the unconditional masks. Parent
review removed those additions. Sec. V.A.3 describes a strain-amplitude
threshold; Eq. 4.6's ratio definition alone does not establish the required
conversion. The conditional rows also depend on which nodes survived preceding
checks. Those ordered checks remain incomplete and require source clarification.
This pass does not claim full Table VI implementation.

Tests cover strict parameter boundaries, irrelevant modes and the omitted
conditional region. A production mixed batch perturbs the actual 32 inner-fit
value: the vetoed row is unchanged, the non-vetoed row changes, and disabling
only the new selection changes the vetoed waveform. This prevents a pre-existing
pole fallback from making the test vacuous. Spin gradients are finite and
nonzero away from selection boundaries. Comparisons use zero absolute tolerance
or normalized relative changes so strain scale cannot hide a failure.

PDF SHA256:
`00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee`.
The updated amplitude model in Colleoni et al., arXiv:2412.16721v1, Sec. II.1,
requires different fits and is not substituted for this 2020 prescription.
Remaining phase-node targets, amplitude-unit metadata and spheroidal 32
conventions remain independent gaps. See the pass-12 summary for validation.
