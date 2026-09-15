# Pass 19: IMRPhenomNSBH low-Λ amplitude factor

## Scope and source boundary

This is a bounded, source-first audit of the low-Λ pre-merger/ringdown amplitude
factor. I used the primary paper and its cited primary amplitude papers before
opening the native implementation. I did not use LAL/PyCBC, ports, generated C,
notebooks, git history, verification outputs, binaries, or residuals. No runtime
or test file was edited.

Primary URLs and local provenance:

* Thompson et al., *Modeling the gravitational wave signature of neutron star
  black hole coalescences: PhenomNSBH*, arXiv:2002.08383v2,
  [record](https://arxiv.org/abs/2002.08383v2),
  [source archive](https://arxiv.org/src/2002.08383v2). The downloaded archive is
  `/tmp/nsbh-primary/thompson.tar`, SHA-256
  `bf81b62a2397ca291e4847e979c62f83f2435e29f470c12baa111041f7377855`; its
  `paper.tex` is SHA-256
  `8afc8c4ddfa0f5b9cad5856590f5cea7edba3c317f0e84fbad6a022e96ee96b5`.
* Pannarale et al., *Aligned spin neutron star-black hole mergers: a
  gravitational waveform amplitude model*, arXiv:1509.00512v1,
  [record](https://arxiv.org/abs/1509.00512v1),
  [source archive](https://arxiv.org/src/1509.00512v1). The downloaded archive is
  `/tmp/nsbh-primary/p15v1.tar`, SHA-256
  `d770e5022805b00ac037534634cdbb6de0199b147961185983b114e41f7ee94f`; its
  `PhenoMixedSpinGWs.tex` is SHA-256
  `99a98ca04e3417f8dcab7b48f8680808b9f86b92ed41a4653108e776c286b86f`.
* Pannarale et al., *Nonspinning black hole-neutron star mergers: a model for
  the amplitude of gravitational waveforms*, arXiv:1307.5111v1,
  [record](https://arxiv.org/abs/1307.5111v1),
  [source archive](https://arxiv.org/src/1307.5111v1). The downloaded archive is
  `/tmp/nsbh-primary/p13v1.tar`, SHA-256
  `56aa31f45bc14107d965da5e5b73347c0b9f96213fbe2d5fe70369a1507b3109`; its
  `PhenoMixedGWs.tex` is SHA-256
  `6bbb1046f091a722d39b94056db268712224ec9d673fe76c4a2e42f2f8f19d8d`.

The arXiv source archives contain TeX, bibliography, and plot PDFs only. No
official `.m`, `.mat`, `.dat`, or `.csv` asset was present. The corresponding
published Pannarale record is [Phys. Rev. D 92, 084050](https://doi.org/10.1103/PhysRevD.92.084050).
I found no official erratum or supplementary mathematical asset that resolves
the Thompson notation conflict.

## Primary derivation

Thompson v2 Appendix A gives the amplitude assembly (paper.tex lines
300--314):

\[
A=A_{\rm PN}\,\omega^-_{\tilde f_0,d}(f)
  +\gamma'_1 f^{5/6}\,\omega^-_{\tilde f_1,d}(f)
  +A_{\rm RD}\,\omega^+_{\tilde f_2,d}(f).
\]

The same appendix defines the purported low-Λ correction (lines 740--749)
as the literal piecewise quantity

\[
\gamma'_1(\Lambda)=
\begin{cases}
1.25, & \Lambda>1,\\
1-0.5\Lambda-0.25\Lambda^2, & \Lambda\le 1.
\end{cases}
\]

Thus the exact source values are

\[
\gamma'_1(0)=1,
\qquad
\lim_{\Lambda\to1^-}\gamma'_1=0.25,
\qquad
\lim_{\Lambda\to1^+}\gamma'_1=1.25.
\]

There is no branch equality at Λ=1: the jump is exactly `1.00`. The lower
branch is continuous to the BBH-limit endpoint Λ=0, and its derivative there is
`-0.5`; neither fact makes the split at one continuous. The sentence immediately
following the equation says that the split is used to “smoothly match to the BBH
limit where \(\Lambda=0\)” (line 749). It does not state continuity at the
piecewise boundary, and the displayed formula does not provide it.

The same paper explicitly computes the PhenomC parameters \(\gamma_1,\delta_1,
\delta_2\) in workflow step 7 (lines 296--298), but the displayed amplitude
uses \(\gamma'_1\) alone and the displayed \(\gamma'_1\) formula contains no
\(\gamma_1\). Taken literally, Thompson therefore replaces the mass-ratio and
spin-dependent PhenomC coefficient by a scalar depending only on Λ. That is
internally inconsistent with the stated step-7 input and with the cited
construction; it is not possible to infer from this paper alone whether the
prime was intended as an absolute coefficient or as a multiplicative factor.

The cited Pannarale sources resolve the historical convention, but not the
Thompson typo/inconsistency:

* Pannarale 2013, lines 303--316, defines the PhenomC pre-merger term as
  \(\gamma_1 f^{5/6}\). Its generalized NSBH ansatz (lines 365--402 and
  765--775) is \(\kappa\gamma_1 f^{5/6}\), explicitly describing \(\kappa\) as
  “a correction ... to the BH-BH coefficient \(\gamma_1\)”. The paper chooses
  \(\kappa=1.25\) for the calibrated model (lines 984--1001).
* Pannarale 2015 Eq. (PhenoMixedAmp), source lines 585--600, and the
  non-disruptive Eq. (PhenoMixedAmpND), lines 693--724, both use
  \(1.25\gamma_1 f^{5/6}\). It explains that \(\gamma_1\) is supplied by the
  PhenomC fit (lines 597--600, 636--645).

Therefore, the primary-source-supported historical semantics are “correction
factor times PhenomC \(\gamma_1\)”. The exact Thompson v2 amplitude text instead
spells the product as a standalone \(\gamma'_1\). Multiplying the Thompson
piecewise factor by \(\gamma_1\) is the interpretation used by the native code
and is consistent with Pannarale and the BBH limit, but it is an interpretation,
not a source-complete transcription of Thompson Eq. (ampansatz).

For completeness, Thompson’s neighboring low-Λ ringdown fit (lines 758--766)
has the same endpoint design for a different quantity:

\[
\delta'_2=\begin{cases}
\frac{A}{2}\omega^-_{x_3,d_3}((f_{\rm tide}-\tilde f_{\rm RD})/\tilde f_{\rm RD}),&\Lambda>1,\\
\delta_2-2(\delta_2-b_0)\Lambda+(\delta_2-b_0)\Lambda^2,&\Lambda\le1,
\end{cases}
\]

with \(b_0=0.81248=A/2\). This branch is continuous at Λ=0 to \(\delta_2\),
but its Λ=1 equality also depends on the high branch's frequency/window value;
it does not repair the independent \(\gamma'_1\) discontinuity.

## Native comparison (after derivation)

The native branch is `src/torchwave/approximants/imrphenom_nsbh.py` lines
620--646 and 744--746. It computes the Santamaria/PhenomC \(\gamma_1\), then
uses

```python
gamma1_prime = torch.where(
    lam2 > 1.0,
    1.25 * gamma1,
    (1.0 - 0.5 * lam_safe - 0.25 * lam_safe**2) * gamma1,
)
```

and inserts that product as the coefficient of \(f^{5/6}\). This is
source-consistent with Pannarale’s \(\kappa\gamma_1\) semantics and the
\(\Lambda=0\) BBH limit. It is not literal Thompson Eq. (ampansatz), which
would use the scalar \(\gamma'_1\). The native branch also inherits the exact
source discontinuity: its coefficient jumps from \(0.25\gamma_1\) to
\(1.25\gamma_1\) at Λ=1 (unless \(\gamma_1=0\)).

## Decision, blocker, and implementation implications

The exact decision is:

1. The factor is supported as multiplying PhenomC \(\gamma_1\) by Pannarale
   2013/2015 and by Thompson’s workflow/BBH-limit context, so the native
   product interpretation is the defensible implementation reading.
2. Thompson v2 does not provide a source-complete, continuous low-Λ branch:
   its literal amplitude omits \(\gamma_1\), while its polynomial has an exact
   unit jump at Λ=1. No permitted primary asset supplies a corrected
   coefficient or a boundary convention. This is a blocker to claiming an
   exact source-derived “smooth” formula at Λ=1.
3. Do not silently change the polynomial to force continuity. Any such change
   (for example, changing the sign or adding a factor) would be a new model
   choice unsupported by the permitted sources.

The independent implementation review should decide whether to preserve the
published discontinuity for fidelity or introduce an explicitly documented
regularization after obtaining an upstream clarification. Neither choice can be
validated from the primary source alone.

## Test plan for the implementer

No tests were run because this pass forbids runtime/test edits. Focused tests
should cover:

* exact branch values at Λ=0, just below 1, exactly 1, and just above 1;
* the algebraic one-unit jump in the scalar factor, and the corresponding
  \(\gamma_1\)-scaled jump in the assembled pre-merger amplitude;
* batch and autograd behavior across the `torch.where` branch, including the
  BBH-limit coefficient \(\gamma_1\) at Λ=0;
* an independent equation-level check that the implementation uses the
  Pannarale-supported product `factor * gamma1`, while documenting that this is
  an interpretation of Thompson’s inconsistent notation.

