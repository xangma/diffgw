# Audit Record: IMRPhenomXAS Primary Literature, Parameter Space Fits, and Collocation Numerical Stability

**Date**: 2026-09-11  
**Approximant**: `IMRPhenomXAS`  
**Primary Published Reference**: Pratten et al. (2020), Phys. Rev. D 102, 064001 ([arXiv:2001.11412v1](https://arxiv.org/abs/2001.11412))  
**Published Paper Title**: *Setting the cornerstone for a family of models for gravitational waves from compact binaries: The dominant harmonic for nonprecessing quasicircular black holes*  
*(Note: Author arXiv preprint title differs slightly: "Setting the cornerstone for the IMRPhenomX family of models...")*  
**Primary Author Source File**: `src/torchwave/data/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m` (G. Pratten 2019, SHA256: `94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52`)  
**Clean-Room Boundary & Access Scoping**: The Agy implementation received only primary mathematical literature (Pratten et al. 2020), author parameter fits (`.m`), and public-output diagnostics. Codex previously inspected LAL source for an independent separate review, but supplied no LAL implementation, internal coefficients, or source-derived code to Agy in this phase. No LAL source code, internal tables, or third-party ports were accessed or inspected by Agy.

---

## 1. Scope of Audit & Verified Formulas

This audit specifically compared the concrete formulas implemented in `imrphenom_xas.py` against the corresponding definitions in `IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`. Only the specific formulas enumerated below were directly checked:

### A. Effective Spin Parameterizations
1. **`sTotR`** (`.m` lines 21–30):
   $$S_{\rm tot,R} = \frac{m_1^2 \chi_1 + m_2^2 \chi_2}{m_1^2 + m_2^2}$$
   Implemented in `imrphenom_xas.py` line 223. Formula transcription matches.
2. **`chiPN`** (`.m` lines 33–46):
   $$\chi_{\rm PN} = \frac{1}{2}(1+\delta)\chi_1 + \frac{1}{2}(1-\delta)\chi_2 - \frac{76}{113}\frac{1}{2}(\chi_1+\chi_2)\eta$$
   Implemented in `imrphenom_xas.py` line 225. Formula transcription matches.
3. **`chiPNhat`** (`.m` lines 49–56):
   $$\hat{\chi}_{\rm PN} = \frac{\chi_{\rm PN}}{1 - 76\eta/113}$$
   Implemented in `imrphenom_xas.py` line 226. Formula transcription matches.

### B. Amplitude Parameter Space Fits
1. **`fit_f_meco`** (`.m` lines 64–76, `fMECOFit`):
   Matches Version 103 rational function coefficients and spin difference terms.
2. **`fit_final_spin`** (`.m` lines 83–93, `FinalSpin2017`):
   Matches rational function and asymmetric spin terms.
3. **`fit_erad`** (`.m` lines 100–115, `Erad2017`):
   Matches Version 2017 fits.
4. **`fit_rd_amp_v1`** (`.m` lines 469–480, `RDAmpv1Fit`):
   Matches Version 103 with $S = S_{\rm tot,R}$.
5. **`fit_rd_amp_gamma2`** (`.m` lines 486–496, `RDAmpgamma2Fit`):
   Matches Version 103 with $S = S_{\rm tot,R}$.
6. **`fit_rd_amp_gamma3`** (`.m` lines 503–514, `RDAmpgamma3Fit`):
   Matches Version 103 with $S = S_{\rm tot,R}$.
7. **`fit_int_amp_va`** (`.m` lines 524–538, `IntAmpvAFit`):
   - Numerical fit coefficients match Version 104 rational function and asymmetric spin terms.
   - **Discrepancy Note**: The author release candidate `.m` specifies $S = \hat{\chi}_{\rm PN}$ (`PhenAux'chiPNhat`). However, compatibility tests against public waveform outputs across multiple configurations ((35, 30, 0.2, -0.3), (30, 30, 0, 0), (70, 10, -0.5, 0.2)) demonstrate that the intermediate midpoint amplitude evaluates with the other named paper effective spin, namely the mass-weighted total spin $\hat{S} = S_{\rm tot,R} = (m_1^2 \chi_1 + m_2^2 \chi_2)/(m_1^2 + m_2^2)$ (`compute_spin_variables` output `s_tot_r`).
   - Prior to this choice, midpoint amplitude ratios were 1.00155439, 1.0, 0.99774819; with $S = S_{\rm tot,R}$, midpoint ratios match public binary outputs at floating precision.
   - This observed compatibility choice is documented explicitly; all numerical coefficients remain the independently sourced author decimals, and no other fit definitions are changed.
8. **`fit_ins_amp_r2`** (`.m` lines 589–604, `InsAmpR2Fit`):
   Matches Version 103 evaluated at $0.5 f_{\rm endpoint}$ with $S = \hat{\chi}_{\rm PN}$.
9. **`fit_ins_amp_r3`** (`.m` lines 611–624, `InsAmpR3Fit`):
   Matches Version 103 evaluated at $0.75 f_{\rm endpoint}$ with $S = \hat{\chi}_{\rm PN}$.
10. **`fit_ins_amp_r4`** (`.m` lines 631–645, `InsAmpR4Fit`):
    Matches Version 103 evaluated at $1.0 f_{\rm endpoint}$ with $S = \hat{\chi}_{\rm PN}$.

### C. 3PN TaylorF2 Amplitude Expansion
- `PNAmplitude` (`.m` lines 656–667) compared against `IMRPhenomXAS.pn_amp_func`:
  Matches the 6 PN terms ($x^0, x^{2/3}, x^1, x^{4/3}, x^{5/3}, x^2$ for $x = \pi M f$).
- Analytic frequency derivative $\frac{d A_{\rm PN}}{d(Mf)} = \pi \frac{d A_{\rm PN}}{dx}$ verified by term-by-term power rule.

---

## 2. Inspiral Residual Collocation Endpoint: Primary Literature Discrepancy

### A. Published Equation vs. Public Binary Behavior
- In the published paper (Pratten et al. 2020 Eq. 6.5 / manuscript Eq. 70), the inspiral residual collocation nodes are printed as $(0.5, 0.75, 1.0) \times f_{\rm MECO}$.
- However, external public binary tests conducted across multiple mass and spin configurations ($df = 0.25$ Hz, sampled at $Mf \in \{0.004, 0.008, 0.015, 0.020\}$) revealed that using $f_{\rm MECO}$ produces an inspiral amplitude deviation growing up to $+34\%$ near merger.
- When using the named original-paper frequency:
  $$f_{\rm Tamp} = f_{\rm MECO} + 0.25 (f_{\rm ISCO} - f_{\rm MECO})$$
  (Eq. 52 / Eq. 6.2) as the collocation endpoint instead of $f_{\rm MECO}$, the resulting inspiral amplitude matches public LAL binary outputs to $\sim 10^{-14}$ across test cases ((35, 30, 0.2, -0.3), (30, 30, 0.0, 0.0), (70, 10, -0.5, 0.2)).
- **Documentation of Discrepancy**: This endpoint choice is a behavioral compatibility convention confirmed by black-box binary waveform measurements. It is NOT claimed that the published paper text explicitly states $f_{\rm Tamp}$ for Eq. 6.5.
- The collocation helper argument is named neutrally as `f_endpoint`.

### B. Algebraic Resolution of Small Determinant Fallback
- The previous implementation used dimensionful Cramer's rule where the 3x3 matrix determinant scaled as $(\pi M f_{\rm endpoint})^8 \det K$. When $f_{\rm endpoint} \le 0.007$, $\det M < 10^{-18}$, which triggered an arbitrary fallback replacing $\det M$ with 1.0, suppressing $\rho_i$ by up to $10^{18}$.
- **Dimensionless Collocation Formulation**: Factoring out $(\pi M f_{\rm endpoint})^p$ maps the system to:
  $$K \tilde{\boldsymbol{\rho}} = \mathbf{r}, \quad K_{ij} = k_i^{p_j}, \quad k = (0.5, 0.75, 1.0), \quad p = (7/3, 8/3, 3)$$
  Because $k$ and $p$ are constant, $K$ is a constant 3x3 matrix with non-vanishing determinant:
  $$\det K = k_1^{7/3}(k_2^{8/3} - k_2^3) - k_1^{8/3}(k_2^{7/3} - k_2^3) + k_1^3(k_2^{7/3} - k_2^{8/3}) \neq 0$$
- Dimensionless weights $\tilde{\rho}_j$ are solved directly without dimensionful determinant collapse, and $\rho_j = \tilde{\rho}_j (\pi M f_{\rm endpoint})^{-p_j}$ recovers node residuals exactly without any threshold fallback.
- Note on scale invariance: Dimensionless $M f_{\rm MECO}$ and $M f_{\rm Tamp}$ depend solely on mass ratio $\eta$ and dimensionless spins $\chi_1, \chi_2$, and are invariant under total mass scaling.

---

## 3. Intermediate Scaled Amplitude: Physical Normalization Convention & Quartic 104

### A. Model Variant Mismatch & Default Quartic Ansatz
- Public LAL waveform-parameter inspection identified that the default model variant is `IntermediateAmpVersion104`, corresponding to the quartic inverse polynomial ansatz in published Table II.
- The previous implementation contained only the quintic (Version 105) ansatz with an unjustified $\pi^{1/6}$ scaling factor and $\sim 0.3\%$ residual discrepancies. Default quartic 104 inverse polynomial collocation has replaced it.

### B. Physical Normalization Convention ($B(f)$)
- The global waveform amplitude in `imrphenom_xas.py` is structured as:
  $$A_{\rm full}(f) = A_{0,\rm phys}(f) \cdot \text{scaled\_amp}(f)$$
  where $A_{0,\rm phys}(f) \propto f^{-7/6}$.
- In the intermediate region, compatible behavior reconstructs the inverse amplitude with only the constant Newtonian prefactor removed:
  $$B(f) = \text{scaled\_amp}(f) \cdot f^{-7/6}$$
  The inverse polynomial $P(u) = 1/B(f)$ is interpolated on dimensionless coordinate $u = (f - f_1)/(f_4 - f_1)$ with $f_1 = f_{\rm Tamp}$ and $f_4 = f_{\rm peak}$.
- **Boundary & Midpoint Values**:
  - At $u = 0$: $B_1 = \text{val}_1 / f_1^{7/6}$, where $\text{val}_1 = \text{scaled\_amp}_{\rm ins}(f_1)$.
  - At $u = 0.5$: $B_{\rm mid} = v_A$ directly from `fit_int_amp_va`.
  - At $u = 1$: $B_4 = \text{val}_4 / f_4^{7/6}$, where $\text{val}_4 = v_{1,\rm RD}$.
- **Frequency Derivatives ($C^1$ Matching)**:
  $$\frac{d B}{df} = \frac{\frac{d}{df}[\text{scaled\_amp}] - \frac{7}{6}\frac{\text{scaled\_amp}}{f}}{f^{7/6}}$$
  - At $u = 0$: $B'_1 = (d_1 - (7/6) \text{val}_1 / f_1) / f_1^{7/6}$ with exact analytic $d_1 = \left.\frac{d A_{\rm ins}}{df}\right|_{f_1}$.
  - At $u = 1$: $B'_4 = (d_4 - (7/6) \text{val}_4 / f_4) / f_4^{7/6}$ with analytic ringdown slope $d_4$.
- **Reconstruction**:
  The intermediate scaled amplitude is evaluated as:
  $$\text{scaled\_amp}_{\rm int}(f) = \frac{f^{7/6}}{P(u)}$$
  When multiplied by $A_{0,\rm phys}(f) \propto f^{-7/6}$, the $f^{\pm 7/6}$ factors cancel exactly, enforcing exact physical $C^1$ Hermite continuity across boundaries.
- **Discrepancy with Printed Literature**:
  Eq. (6.7) read together with Eq. (6.2) in Pratten et al. (2020) omits this explicit $f^{-7/6}$ factor extraction. This discrepancy is documented here as the independently reconstructed normalization that matches public-output behavior, with $C^1$ boundary matching enforced by construction across the physical amplitude.
- Generic solver `solve_intermediate_amplitude_quartic` retains its explicit value/derivative contract on dimensionless $u$, preserving numerical stability and clean autograd differentiation.

---

## 4. Verification Suite (Executed by Codex)

All numerical tests are executed externally by Codex in its dedicated environment; Agy performs static code maintenance and verification only.

1. **`tests/test_xas_collocation.py`**:
   - Node recovery across wide range of $f_{\rm meco}$ values ($< 10^{-12}$).
   - Batched vs scalar consistency.
   - Autograd differentiability and `torch.autograd.gradcheck` using fixed independent physical evaluation frequency $f_{\rm fixed} = 0.015$.
   - Float32 stability without determinant collapse.
   - High-mass binary waveform generation.
   - Removed unused imports (`math`, `pytest`, `XASParameterFits`).

2. **`tests/test_xas_intermediate_amplitude.py`**:
   - Exact boundary and midpoint value matching ($< 10^{-12}$).
   - Exact derivative continuity at boundaries verified using autograd ($< 10^{-12}$) and second-order one-sided finite-difference stencils ($O(\epsilon^2)$ error $< 10^{-7}$, well within the $5 \times 10^{-5}$ tolerance).
   - Batched vs scalar consistency.
   - Float32 stability with relative comparison of evaluated amplitudes against float64 ($< 10^{-4}$).
   - Autograd gradient flow and `torch.autograd.gradcheck`.
   - Physical transition boundary coverage ($f_{\rm trans} \pm \epsilon$, $f_{\rm peak} \pm \epsilon$) checking relative jump $< 10^{-5}$ and one-sided slope matching across boundaries $< 5 \times 10^{-3}$, plus broadband positivity.
   - Removed unused imports (`math`, `pytest`).

---

## 5. Clean-Room Boundary & Independence Note

- All numerical coefficients implemented in `XASParameterFits` are taken directly from the author's public release candidate file `IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`.
- Discrepancies between printed paper equations / release candidate conventions and public waveform behavior ($f_{\rm Tamp}$ endpoint, $B(f)$ normalization, and $S_{\rm tot,R}$ spin in `fit_int_amp_va`) were diagnosed exclusively via mathematical alternatives and black-box public waveform outputs.
- Scoping: Agy implementation received only primary mathematical sources and public-output diagnostics; Codex previously inspected LAL source for a separate review, but supplied no LAL implementation, internal coefficients, or source-derived code to Agy in this phase.
- Full model parity and broader parameter-space validation are reserved for independent Codex verification.

