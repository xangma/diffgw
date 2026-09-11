# Independent Investigation: IMRPhenomD Phasing Conventions, Author Sources, and Parity Residuals

**Date**: 2026-09-11  
**Approximant**: `IMRPhenomD`  
**Focus**: Primary literature and thesis search, Table V raw $\alpha_1$ fit, intrinsic global time gauge, and measured complex $L_2$ error reduction from $\sim 1.04$ to $\sim 0.003 - 0.011$.

---

## 1. Primary-Source Search & Document Status

We investigated author repositories, published literature, university archives, and DCC records under strict clean-room boundaries (no inspection or copying of LAL source code).

### A. Published Papers & LaTeX Source
- **Khan et al. (2016), PRD 93, 044007 (Paper II)**:
  - arXiv source: `https://arxiv.org/src/1508.07253v2` (archive SHA256: `e70cde9d5e9ee92e4592c4c93152f6eb2410264358486f6a68064fb03b39b1cf`).
  - Table V in `PhenomPaper2.tex` (line 2638) contains the raw fit coefficients for $\alpha_1$:
    $$(43.3151, 638.633, -32.8577, 2415.89, -5766.88, -61.8546, 2953.97, -8986.29, -21.5714, 981.216, -3239.57)$$
    While $C^1$ continuity matching at the intermediate-to-merger boundary ($f_2 = 0.5 f_{\rm RD}$) analytically fixes the piecewise connection constant $\alpha_1$, the raw fit $\alpha_1^{\rm raw}$ provides the author-calibrated phase derivative slope in the merger-ringdown regime.
- **Husa et al. (2016), PRD 93, 044006 (Paper I)**:
  - Title: *Frequency-domain gravitational waves from non-precessing black-hole binaries. I. New numerical waveforms and anatomy of the signal*.
  - arXiv source: `https://arxiv.org/src/1508.07250v2` (archive SHA256: `79dd5b277fd99d8ff3576800fb274b296bea8ae1553f134a015205f40fae5b4a`).
  - Establishes final spin $a_f$ (Eq. 16) and radiated energy $E_{\rm rad}/M$ (Eq. 17).

### B. Inaccessible Author Assets & Inquiries
- **Sebastian Khan PhD Thesis**:
  - URL: `https://orca.cardiff.ac.uk/93841/1/2016KhanSPhD.pdf` (Cardiff University ORCA).
  - Observed status: The thesis exists as an identified 19MB PDF in the repository catalog, but is inaccessible via automated retrieval methods.
- **LIGO Document Control Center (DCC)**:
  - Targeted identifiers: `LIGO-P1500142-v2` (Paper II) and `LIGO-P1500143-v2` (Paper I).
  - Observed status: The endpoints checked did not yield an accessible calibration download. This does not establish that no public author asset exists.
- **Author Mathematica Notebooks & Calibration Assets**:
  - Documented in independent literature as obtained via private communications (specifically acknowledged by Bhat et al., PRD 107, 024009 [arXiv:2207.13761v2], Appendix A, and Datta et al., PRD 109, 044036 [Cardiff ORCA 167427], acknowledgments). No public download repository, supplementary tarball, or open data release of the unrounded double-precision coefficients has been identified.

---

## 2. Mathematical Parity Analysis: Phase Anchor and Global Time Gauge

### A. The Reference Phase Anchor $\phi_c$
At the declared reference frequency $f_{\rm ref}$, the convention is:
$$\text{arg}(h_+(f_{\rm ref})) = 2 \phi_c \pmod{2\pi}$$
When $t_c = 0$, both LAL and TorchWave yield $\text{arg}(h_+(f_{\rm ref})) = 2\phi_c$ to within $10^{-14}$.

### B. Intrinsic Global Time Gauge at $f_{\rm peak}$
The merger-ringdown phase derivative (Paper II, Eq. 14) evaluated at the amplitude peak frequency $f_{\rm peak}$ using the raw fitted $\alpha_1^{\rm raw}$ is:
$$\text{slope}_{\rm peak} = \left. \frac{d\Psi_{\rm MR}}{d(Mf)} \right|_{f_{\rm peak}} = \frac{1}{\eta} \left[ \alpha_1^{\rm raw} + \alpha_2 f_{\rm peak}^{-2} + \alpha_3 f_{\rm peak}^{-1/4} + \frac{\alpha_4 f_{\rm damp}}{f_{\rm damp}^2 + (f_{\rm peak} - \alpha_5 f_{\rm RD})^2} \right]$$
where $f_{\rm peak}$, $f_{\rm RD}$, and $f_{\rm damp}$ are dimensionless ($Mf$).

By defining an intrinsic global time gauge that subtracts $\text{slope}_{\rm peak} \cdot (Mf - Mf_{\rm ref})$ from the reference-anchored phase:
$$\Phi(f) = \psi_{\rm raw}(f) - \psi_{\rm ref} - \text{slope}_{\rm peak} \cdot M_{\rm sec} (f - f_{\rm ref}) + 2\pi f t_c - 2\phi_c$$
and for unanchored generation:
$$\Phi(f) = \psi_{\rm raw}(f) - \text{slope}_{\rm peak} \cdot M_{\rm sec} f + 2\pi f t_c - 2\phi_c$$
we achieve phenomenological alignment across frequency bands.

**Important Note on Gauge Provenance**: This intrinsic time gauge slope at $f_{\rm peak}$ is an independent phenomenological compatibility convention supported by black-box binary oracle measurements, rather than a proven physical identity for the time-domain IFFT peak. It strictly preserves:
1. $C^1$ continuity across all frequency boundaries (as $\text{slope}_{\rm peak} \cdot Mf$ is a globally smooth linear term).
2. The reference phase anchor $\Phi(f_{\rm ref}) = -2\phi_c$.
3. Exact Fourier time-shift linearity $e^{-i 2\pi f t_c}$.

### C. Measured Parity Improvement
External binary oracle measurements performed by Codex across 7 representative mass and spin configurations ($df = 0.25$ Hz, $f \in [20, 1024]$ Hz, $d = 400$ Mpc, $\iota = 0.5$, $\phi_c = 0.3$, $f_{\rm ref} = 20$ Hz):
- **Raw complex relative $L_2$ before**: $0.9442 - 1.2567$ (baseline $\sim 1.04$).
- **Raw complex relative $L_2$ after**: $0.00313 - 0.01110$.
These historical measurements showed a substantial reduction; they did not pass strict parity and do not validate the intrinsic gauge across the full parameter domain.

---

## 3. Remaining Parity Gaps

Achieving strict catalog acceptance ($L_2 \le 1.0 \times 10^{-6}$, mismatch $\le 1.0 \times 10^{-8}$) requires separating the known precision limits from potential implementation discrepancies:
1. **Sensitivity to Printed Coefficient Precision**:
   - The printed fit coefficients in Paper II Table V are rounded to 5–7 significant figures.
   - Codex executed [`check_table_v_sensitivity.py`](./check_table_v_sensitivity.py) after verifying that the printed centers equal every runtime coefficient. With no taper, float64, and four binaries on 20–1024 Hz at 0.25 Hz spacing, constructed per-row perturbations produced complex relative $L_2$ changes up to 0.004054; simultaneous perturbations reached 0.01153. [Measured results](./table-v-sensitivity-results.json).
   - The intervals assume independent rounding to the nearest printed last digit. These examples show that numbers compatible with that assumption can produce differences larger than $10^{-6}$. They are not an actual error estimate, a statistical interval, a worst-case bound, or evidence that rounding explains every observed discrepancy. Exact recovery would benefit from more precise author calibration information; this analysis does not prove that double precision is necessary or that parity is impossible.
2. **Unresolved Implementation Conventions**:
   - Beyond coefficient truncation, any residual differences may reflect internal choices in author implementation scripts (such as higher-order PN truncation nuances or intermediate-region connection subtleties) that cannot be deduced solely from the published journal text.

---

## 4. Verification Suite

The regression test suite [`tests/test_phenomd_phase_conventions.py`](../../../../../tests/test_phenomd_phase_conventions.py) validates:
- The exact phase anchor $\text{arg}(h_+(f_{\rm ref})) = 2\phi_c \pmod{2\pi}$ within $10^{-12}$.
- Exact Fourier time-shift linearity with zero amplitude distortion.
- Batched vs scalar equivalence using static call `IMRPhenomD.__call__`.
- Differentiability through the global time gauge slope via PyTorch autograd.

A separate coefficient sensitivity diagnostic is [`check_table_v_sensitivity.py`](./check_table_v_sensitivity.py); it is not part of the phase regression suite. Reproduce from the repository root with `PYTHONPATH=src python docs/reviews/2026-09-11-independent/phenomd/check_table_v_sensitivity.py --json-out /tmp/table-v-sensitivity.json`.
