# Pass 29: TEOB NQC extraction and postmerger bridge

**Date:** 2026-09-13  
**Scope:** clean-room audit of the standalone TEOB conservative trajectory, NQC targets, and postmerger components.  
**Verdict:** documentation only; no runtime bridge is source-complete enough to add.

## Source ledger

This audit used the primary TEOB paper, [Nagar et al., arXiv:1806.01772v2](https://arxiv.org/abs/1806.01772), and the conservative EOB dynamics source, [Damour and Nagar, arXiv:0704.3550v2](https://arxiv.org/abs/0704.3550).  The repository's existing TEOB modules and focused tests were also inspected.  No LAL or PyCBC implementation source was inspected, read, translated, or copied.

The 2018 TEOB paper gives the NQC structure

\[
 \hat h^{\rm NQC}_{22} = (1+a_1 n_1+a_2 n_2)
   \exp\{i(b_1 n'_1+b_2 n'_2)\},
\]

and defines the model NQC time by

\[
 t_{\rm NQC}=t^{\Omega_{\rm orb}}_{\rm peak}-\Delta t_{\rm NQC},
 \qquad
 \Omega_{\rm orb}=H_{\rm EOB}^{-1}
   \frac{\partial H^{\rm orb}_{\rm eff}}{\partial p_\phi}.
\]

The paper explicitly defers the basis functions \(n_i,n'_i\) to Eq. (96) of its Ref. [17].  It also states that \(\Delta t_{\rm NQC}\) is calibrated with NR information; the quoted default is one mass unit, with a separate limited high-spin/high-mass-ratio prescription.  The merger time for the postmerger waveform is instead

\[
 t_{\rm mrg}=t_{\rm peak}^{|h_{22}|},
\]

while NR NQC target data are measured at

\[
 t_{\rm NQC}^{\rm NR}=t_{\rm mrg}+2M.
\]

Those are distinct event and time conventions.  The paper's postmerger template also assumes an explicitly supplied remnant mass, remnant spin, phase anchor, and QNM convention.

## What the repository can support

`src/torchwave/approximants/eob/teob_trajectory.py` provides the source-derived conservative canonical equations and a differentiable, batched diagnostic for \(r,\phi,p_{r*},p_\phi\).  It does not produce a complex TEOB strain mode, select an amplitude peak, or supply the NQC basis variables.

`src/torchwave/approximants/eob/teob_postmerger.py` provides the published nonspinning merger/postmerger fits and `nqc_targets(nu)`.  The target fit returns the four fitted values \((\hat A,\dot{\hat A},\omega,\dot\omega)\) at the paper's NR NQC point; it does not extract coefficients from a model mode or attach those coefficients to a trajectory.

The existing `src/torchwave/approximants/eob/nqc.py` and `attachment.py` implement the separately sourced SEOBNRv4 construction.  Reusing that bridge for TEOB would silently substitute a different model's basis, event timing, target conventions, and attachment rules.

## Exact missing inputs

An explicit TEOB bridge still requires all of the following source-resolved inputs:

1. The exact Eq. (96) basis definitions and derivative conventions from Ref. [17], including the normalization and sign of the radial-momentum and radial-acceleration terms.
2. The complete \(\Delta t_{\rm NQC}(q,\chi_A,\chi_B,\ldots)\) prescription, including its domain and interpolation at the stated exceptional corners.
3. A complete factorized TEOB \(h_{22}^{\rm base}\) and its normalization, together with the orbital frequency and the quantities needed to evaluate the basis at the NQC time.  The current conservative trajectory is insufficient for this.
4. A source-defined amplitude-peak event and alignment between that event, the orbital-frequency peak, and the NR target time.  The repository's generic local peak interpolation is a numerical utility, not the TEOB calibration convention.
5. Spinning and tidal NQC target fits.  The current target implementation is explicitly the nonspinning branch and cannot establish general aligned-spin parity.
6. The remnant mass/spin, QNM table/convention, phase anchor, and continuation policy needed to attach the postmerger template to the calibrated mode.

Adding a generic interpolation from the trajectory, or wiring the SEOBNRv4 NQC solver to the TEOB postmerger fits, would therefore add undocumented model choices rather than implement a uniquely published TEOB construction.  It would not be a clean-room parity result, so this pass intentionally makes no runtime change.

## Focused validation

The existing source-complete components remain covered by:

```text
tests/test_clean_room_teob_trajectory.py
tests/test_clean_room_teob_postmerger.py
```

These tests cover the conservative equations, batching/device/dtype behavior, gradients, fit continuity, and the published nonspinning target/ringdown relations.  They do not claim a TEOB NQC extraction or full waveform parity.
