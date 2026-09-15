# Pass 20 independent review: EOB NQC and attachment

## Decision

**RETAIN** `nqc.py` and `attachment.py` as explicitly scoped, equation-level
components. **REJECT** the claim that the default `SEOBNR_ODE` waveform path is
a source-complete SEOBNRv4 waveform. No runtime or test changes are made.

The first decision is conditional on the caller supplying a continuous complex
inspiral mode, its positive GW frequency, and the chosen remnant fundamental
QNM. The second follows directly from `waveform.py`: it attaches at the last
trajectory sample and appends one exponential using the PhenomD remnant
properties. That path does not execute the SEOBNRv4 peak, NQC, or merger ansatz.

## Clean-room sources and derivation

The primary sources used here were:

* Bohé et al., *Improved effective-one-body model for coalescing binary-black-
  hole binaries*, [arXiv:1611.03703v3](https://arxiv.org/abs/1611.03703),
  SHA-256 `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba`.
* Damour and Nagar, *Faithful effective-one-body waveforms of equal-mass
  coalescing black-hole binaries*, [arXiv:0906.1769](https://arxiv.org/abs/0906.1769),
  SHA-256 `516f63eeb99f80e33c245e2290e8b97ce252e9fbc07adf76a003b20b132e6350`.
* Taracchini et al., *Prototype effective-one-body model for nonprecessing
  spinning binary black-hole coalescence*, [arXiv:1202.0790](https://arxiv.org/abs/1202.0790),
  SHA-256 `d6d126be90bc789835ec363451f4ec5172a8d6ddc09eff078620fdc46860745`.

The SEOBNRv4 NQC factor is Eq. (2.7) of 1611.03703:

\[
 N_{22}=\left[1+q^2(a_1+a_2/r+a_3/r^{3/2})\right]
 \exp\{i q(b_1+b_2p_{r_*}^2)\},\qquad
 q={p_{r_*}\over rM\Omega}.
\]

Its five matching conditions at
\(t_{\rm peak}^{22}=t_{\rm peak}^{\Omega}+\Delta t_{\rm peak}^{22}\)
are the amplitude, zero amplitude slope, amplitude curvature, GW frequency,
and GW-frequency slope. The Appendix-A fits supply those targets and the
calibrated time offset. `nqc.py` has the same three amplitude basis terms and
two phase basis terms, uses the same positive-frequency convention, and solves
the two corresponding linear systems without clipping or regularization.
The printed fits are reproduced, including the paper's distinction between
the \(\nu=10^{-3}\) endpoint used by the global amplitude/curvature/slope fits
and the \(\nu=0\) test-particle endpoint in the direct frequency fit.

SEOBNRv4 then sets the attachment time to the NQC-corrected amplitude peak,
\(t_{\rm match}^{22}=t_{\rm peak}^{22}\), and uses the merger ansatz in
Eqs. (2.10)--(2.17) of 1611.03703. It contains the least-damped remnant
\((2,2,0)\) QNM and constrained amplitude and phase terms. The constraints
make the complex mode C1 at the attachment: amplitude and phase are C1, hence
the GW frequency is C0. The remnant mass/spin fit and QNM convention are model
inputs; the paper does not license inferring them from an arbitrary mode.

The older construction is materially different. Damour--Nagar 0906.1769
describes a \((2p+1)\)-tooth matching comb and, for \(p=2\), five positive-
frequency Kerr QNMs. Taracchini et al. 1202.0790 specifies the calibrated
spinning construction: eight linear modes with the highest replaced by a
pseudo-QNM, values on \([t_{\rm match}-\Delta t_{\rm match},t_{\rm match}]\),
endpoint derivative conditions, and \(\Delta t_{\rm match}=7.5M\). This is
not the SEOBNRv4 attachment. The five samples in `attachment.py` are a local
quartic interpolation used to estimate NQC values and derivatives; they are
not a 7.5M comb and do not introduce eight QNM coefficients.

## Findings

`nqc.py` is algebraically consistent with Eq. (2.7) and Appendix A, including
mass-exchange invariance, dimensionless \(t/M\) derivatives, broadcasting,
and autograd dependence through the matching solve. One boundary caveat is
visible in the API: it accepts \(\nu=0\) for all target fits. For the global
amplitude, curvature, and frequency-slope fits, evaluating exactly at zero is
an extrapolation from the published \(10^{-3}\) endpoint; only the direct
frequency fit explicitly prints a \(\nu=0\) endpoint. This does not alter the
printed algebra on the calibrated domain, but it should not be described as a
published zero-mass-ratio calibration.

`attachment.py` correctly forms the orbital-frequency peak plus the published
\(\Delta t_{\rm peak}^{22}\), solves the five NQC conditions, and supplies the
fundamental QNM and phase anchor to the C1 merger ansatz. Its quartic local
stencil is a declared numerical reconstruction rather than a published
algorithm. Therefore exactness is conditional on a resolved, well-conditioned
stencil; the implementation correctly retains gradients inside a selected
stencil and fails on insufficient history instead of silently extrapolating.
The direct tests also exercise the complex value and derivative at the join,
the five targets, phase covariance, batch/scalar agreement, and stencil
convergence.

`waveform.py` does not wire those components into `SEOBNR_ODE`. It constructs a
prototype or uncorrected factorized mode, chooses the final input sample as
the match, and appends a single PhenomD exponential. It therefore lacks the
published orbital-peak time shift, NQC-corrected amplitude/frequency inputs,
SEOBNRv4 amplitude/phase shape, and explicit final-remnant/QNM metadata. The
existing API does not provide a source-complete choice for the continuous
residual phase, remnant fit, mode convention, and fundamental QNM. Replacing
the endpoint exponential by guessed values would change the model silently,
so no runtime correction is justified in this pass.

## Inspected material and validation

Inspected file hashes:

* `nqc.py`: `f023b84fc0a48e59eb6f543d234635df965e04ee9aff05672ea3ff984d0cf85`
* `attachment.py`: `09941c1cc1aa371898205a38719f13d359655f0b9cad76828ea86cfb09df0521`
* `waveform.py`: `03a559c59f1aef83037ba0fdea61cc070c1afa4478f082d4068b67fe1fc27ced`
* primary review: `1f65c2c101c32ed94558335a6327dd3383094a187f5e6761dfa051883f03ddb2`

Focused source-derived validation:

```text
pytest -q tests/test_clean_room_eob_nqc.py \
  tests/test_clean_room_eob_attachment.py \
  tests/test_clean_room_eob_merger_ringdown.py \
  tests/test_clean_room_eob_waveform.py
42 passed in 2.45s
```

No frozen public-binary verification, residual inspection, implementation
source comparison, or runtime/test edit was performed.
