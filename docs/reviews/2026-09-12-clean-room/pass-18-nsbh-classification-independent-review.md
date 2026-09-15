# Independent review: Pass 18 NSBH remnant-frequency classification

## Decision

**RETAIN.** The patch is source-consistent for the classification path. No
runtime or test changes are required by this review.

The conclusion is about the published construction in Thompson et al.,
arXiv:2002.08383v2, not about the physical accuracy of the fitted model
outside its stated calibration domain.

## Clean-room basis

This review used the Thompson et al. v2 paper and its cited primary equations
for the Zappa remnant fit and London Kerr QNM fit, together with direct
inspection of the three files named in the task. I did not inspect LAL/PyCBC
source or ports, generated C, author notebooks, git history, verification
directories/results, public binaries, parity outputs, or residuals.

## Findings

1. **Published ordering and remnant choice are correct.** Thompson Appendix A
   steps 4--6 first evaluate the NSBH fit
   \[
   F(\eta,\chi,\Lambda)=F_{\rm BBH}(\eta,\chi)
   \frac{1+p_1\Lambda+p_2\Lambda^2}{(1+p_3^2\Lambda)^2},
   \]
   separately for final mass and final spin; then evaluate the Kerr QNM
   frequency and quality factor; then apply the table conditions. The runtime
   does exactly this at `imrphenom_nsbh.py:546-553`, with one Zappa remnant and
   one London frequency for every class. The BBH remnant is only the baseline
   inside the fit. At `Lambda=0`, the multiplicative correction is exactly one
   (up to floating-point roundoff), so the BBH limit is preserved.

2. **The boundary and units are correct.** The paper's table and prose use
   the closed condition `f_tide >= f_RD`; the runtime uses that exact operator
   at `imrphenom_nsbh.py:553`. It does not use the adjusted
   `f_RD_tilde` for classification; the paper reserves that quantity for the
   non-disruptive fitting variables. `f_tide` and the London result are both in
   Hz: the former uses component masses converted with `MSUN_SEC`, and the
   latter divides the dimensionless QNM frequency by
   `2*pi*(M_f*MSUN_SEC)` at `imrphenom_nsbh.py:325-327`.

3. **The NSBH spin argument is aligned with the source model.** The source
   defines `chi` as the black-hole spin. The Zappa tidal correction therefore
   uses `chi1` directly at `imrphenom_nsbh.py:279-281`, rather than a
   mass-weighted effective spin. The published model assumes a non-spinning
   neutron star; retaining `chi2` in the BBH baseline preserves the existing
   callable API but is outside that calibration when nonzero.

4. **Tensor behavior is preserved.** The remnant and QNM functions use tensor
   operations elementwise, and the forward path computes classification masks
   before unsqueezing parameters onto a frequency grid. This supports scalar
   parameters, batched binaries, and batched binaries with a shared frequency
   vector. Hard classification itself is necessarily nondifferentiable at the
   boundary, while the tested off-boundary paths have finite gradients.

5. **Test coverage has a small non-blocking gap.** The supplied boundary test
   demonstrates a configuration separating the NSBH and BBH ringdown choices
   and checks the intended inequality, but it does not instrument the waveform
   to prove which table column was selected, nor construct an exact floating
   point equality. Source inspection verifies the operator and branch order;
   the gap does not undermine the patch decision.

## Validation

Focused test command:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_pass18_nsbh_classification.py
```

Result: **2 passed**.

Additional source-derived checks used only the published formulas and local
functions:

- For `(m1,m2,chi1,Lambda)=(5.6,1.4,0,800)`, the values are
  `f_tide=2152.2118314983463 Hz`, `f_RD(NSBH)=2145.9982816199113 Hz`, and
  `f_RD(BBH)=2155.3979262783932 Hz`; hence the closed NSBH comparison is true
  while the old BBH comparison would be false.
- At `Lambda=0`, the Zappa remnant mass ratio and spin agree with the BBH
  baseline to float64 roundoff (`1.2e-16` maximum mass-ratio difference and
  zero measured spin difference in the checked batch).
- The checked batched frequency path returned shape `(3,)`, finite ringdown
  frequencies, and finite waveform gradients with respect to both masses,
  black-hole spin, and deformability.

No parity or external implementation was used for validation.

## Sources

- Thompson et al., *Modeling the gravitational wave signature of neutron
  star-black hole coalescences: PhenomNSBH*, arXiv:2002.08383v2,
  especially Sec. II.1, Table 1, Eqs. (13)--(20), and Appendix A:
  https://arxiv.org/abs/2002.08383v2
- The cited Zappa et al. NSBH remnant fit and London et al. Kerr QNM fit are
  used only through the equations reproduced or cited by Thompson et al.
