# Pass 20: SEOBNRv4 NQC and merger/ringdown attachment

## Verdict

**RETAIN** the native `nqc.py` equations and `attachment.py` matching
semantics as an isolated published-equation component. **REJECT** the current
`SEOBNR_ODE` last-sample ringdown as a SEOBNRv4 implementation. No runtime
change is safe in the three owned files without adding the missing model/API
metadata; this pass is therefore documentation-only.

## Primary-source derivation

Bohé et al., [arXiv:1611.03703v3](https://arxiv.org/abs/1611.03703), Eq. (2.7)
uses

\[
N_{22}=\left[1+\left({p_{r^*}\over rM\Omega}\right)^2
 (a_1+a_2/r+a_3/r^{3/2})\right]
\exp\!\left[i{p_{r^*}\over rM\Omega}(b_1+b_2p_{r^*}^2)\right].
\]

The five conditions at
\(t_{\rm peak}^{22}=t_{\rm peak}^{\Omega}+\Delta t_{\rm peak}^{22}\) are:
zero amplitude derivative; peak amplitude; amplitude curvature; GW frequency;
and GW-frequency derivative. The Appendix-A fits provide the latter four
inputs (with the zero first derivative). The native basis, target fits, and
offset in `nqc.py` reproduce these expressions, including the paper's
different \(\nu=0\) versus \(\nu=10^{-3}\) endpoint conventions.

The SEOBNRv4 merger model is not the older comb construction. Its
[Sec. 2.3](https://arxiv.org/abs/1611.03703) ansatz is attached at the NQC
peak, uses only the least-damped fundamental \((2,2,0)\) QNM, and imposes C1
continuity of amplitude and phase (therefore C0 continuity of GW frequency).
The constrained amplitude and phase coefficients are fixed by those C1
conditions; the four free coefficients are the printed \((\nu,\chi)\) fits.

For provenance, the older [Taracchini et al., arXiv:1202.0790](https://arxiv.org/abs/1202.0790)
model used a linear QNM sum with eight terms (the highest replaced by a
pseudo-QNM), derivative/value matching over a comb, and the calibrated comb
size \(\Delta t_{\rm match}^{22}=7.5M\). That interval and overtone content
must not be imported into SEOBNRv4. The intermediate [SEOBNRv2 paper,
arXiv:1311.2544](https://arxiv.org/abs/1311.2544) likewise describes the
multi-QNM hybrid matching, whereas SEOBNRv4 replaces it with the
phenomenological single-fundamental ansatz.

## Native comparison and blocker

`attachment.py` correctly locates the orbital-frequency peak, applies the
published \(\Delta t_{\rm peak}^{22}\), solves the five NQC conditions, and
passes the fundamental QNM and matching data to the C1 merger ansatz. Its
five-sample quartic interpolation is explicitly a numerical reconstruction
choice; the paper specifies the equations and fitted inputs, not this
stencil. The focused direct tests verify all five conditions, complex C1
joining, batch/scalar equivalence, and gradients.

`waveform.py` still builds a Newtonian/prototype (or uncorrected factorized)
mode, matches at the final trajectory sample, and appends a single PhenomD
exponential. That path does not locate \(t_{\rm peak}^{\Omega}\), apply
NQC, use the SEOBNRv4 amplitude/phase ansatz, or select the published remnant
mass/spin/QNM convention. A source-complete wiring correction requires the
factorized mode with continuous tail/residual phase and explicit final-spin
fit/QNM choices. Those choices are absent from `SEOBNR_ODE`'s API, and the
available published driver requires them as explicit metadata. Guessing a
default or routing the prototype through one convention would silently change
the model, so no runtime edit is made in this pass.

## Verification

`pytest -q tests/test_clean_room_eob_nqc.py tests/test_clean_room_eob_attachment.py tests/test_clean_room_eob_merger_ringdown.py`

Result: **30 passed**. No public-binary verification was run.
