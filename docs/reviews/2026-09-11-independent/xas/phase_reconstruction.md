# IMRPhenomXAS Phase Reconstruction: Mathematical Formulation & Clean-Room Compatibility Record

**Date**: 2026-09-11  
**Approximant**: `IMRPhenomXAS`  
**Primary Reference**: Pratten et al. (2020), Phys. Rev. D 102, 064001 ([arXiv:2001.11412v1](https://arxiv.org/abs/2001.11412)) Eqs. (5.8), (5.10), (5.11), (7.6), (7.8)  
**Clean-Room Boundary & Access Scoping**: The Agy implementation received only primary mathematical literature (Pratten et al. 2020) and public-output diagnostics. Codex previously inspected LAL source for an independent separate review, but supplied no LAL implementation, internal coefficients, or source-derived code to Agy in this phase. Zero access by Agy to LAL source code, internal tables, third-party ports, or fitted residual oracle parameters.

---

## 1. Mathematical Framework & 3-Region Phase Engine

The native 3-region $C^1$ phasing engine models the frequency derivative of the phase $\phi'(f) = \frac{d\phi}{df}$ across three regimes:
1. **Inspiral** ($f \le f_{\rm trans\_phase}$): TaylorF2 3.5PN expansion plus 4-node pseudo-PN collocation ($\sigma_1, \sigma_2, \sigma_3, \sigma_4$).
2. **Intermediate** ($f_{\rm trans\_phase} < f \le f_{H,\rm int}$): 5-node collocation ($b_0, b_1, b_2, b_3, b_4$) on Chebyshev nodes with Lorentzian damping.
3. **Merger-Ringdown** ($f > f_{H,\rm int}$): 5-node collocation ($c_{\rm rd}, c_1, c_2, c_4, c_0$) on ringdown Chebyshev nodes with QNM Lorentzian resonance.

---

## 2. Transition Frequencies and Calibration Intervals

### A. Transition Boundaries
- Calibration reference start frequency:
  $$f_{\rm T\_phase} = 0.6 \times (0.5 f_{\rm rd} + f_{\rm isco})$$
- Ringdown transition interval width:
  $$\delta_R = 0.03 \times (f_{\rm T\_phase} - f_{\rm meco})$$
- Inspiral-to-intermediate boundary:
  $$f_{\rm trans\_phase} = f_{\rm meco} - \delta_R$$
- Intermediate-to-ringdown boundary (and collocation upper endpoint):
  $$f_{H,\rm int} = f_{\rm T\_phase} + 0.5 \delta_R$$

### B. Intermediate Collocation Grid
The five Chebyshev nodes for intermediate collocation are placed on $[f_{L,\rm int}, f_{H,\rm int}] = [f_{\rm trans\_phase}, f_{H,\rm int}]$:
$$\begin{aligned}
f_{n,1} &= f_{L,\rm int} \\
f_{n,2} &= f_{L,\rm int} + 0.5 \left(1 - \frac{1}{\sqrt{2}}\right) \Delta_{\rm int} \\
f_{n,3} &= f_{L,\rm int} + 0.5 \Delta_{\rm int} \\
f_{n,4} &= f_{L,\rm int} + 0.5 \left(1 + \frac{1}{\sqrt{2}}\right) \Delta_{\rm int} \\
f_{n,5} &= f_{H,\rm int}
\end{aligned}$$
where $\Delta_{\rm int} = f_{H,\rm int} - f_{L,\rm int}$.

---

## 3. Key Compatibility Conventions & Literature Discrepancies

### A. $v_{5,\rm int}$ Calibration Frequency vs. Collocation Node
- In the intermediate collocation target vector:
  $$v_{5,\rm int} = \eta \left.\frac{d\phi_{\rm rd}}{df}\right|_{f_{\rm T\_phase}}$$
  is evaluated at $f_{\rm T\_phase}$ rather than the upper collocation node $f_{H,\rm int} = f_{\rm T\_phase} + 0.5\delta_R$.
- **Discrepancy Note**: The published paper (Eq. 7.8) does not explicitly distinguish the calibration evaluation point from the interval endpoint. This distinction is an observed compatibility convention verified by black-box waveform output analysis.

### B. Intermediate Lorentzian Sign Convention (Eq. 7.6)
- The intermediate frequency derivative ansatz adds a Lorentzian term:
  $$\eta \frac{d\phi_{\rm int}}{df} = \sum_{j=0}^4 b_j f^{-j} + \frac{4 c_0}{(f - f_{\rm rd})^2 + (2 f_{\rm damp})^2}$$
- Because the ringdown parameter fits $v_{\rm fit}$ are negated ($v_{\rm fit} = -\eta \phi'$), the collocation target vector removes the Lorentzian contribution via:
  $$b_i = v_{i,\rm int} - \frac{4 c_0}{(f_{n,i} - f_{\rm rd})^2 + (2 f_{\rm damp})^2}$$
- The analytic integral of this Lorentzian term in $\phi_{\rm int}(f)$ is:
  $$\int \frac{4 c_0}{(f - f_{\rm rd})^2 + (2 f_{\rm damp})^2} df = \frac{2 c_0}{f_{\rm damp}} \arctan\left(\frac{f - f_{\rm rd}}{2 f_{\rm damp}}\right)$$
  which enters with a **positive sign** in $\phi_{\rm int}(f)$.
- **Discrepancy Note**: Printed Eq. (7.6) and the fit convention leave a compatibility discrepancy regarding the sign of the Lorentzian term. Rather than claiming that inverting fit signs mathematically forces this sign choice by itself, we document that this positive-sign convention was resolved and confirmed by public-output diagnostic phase shapes.

### C. $C^1$ Boundary Matching at $f_{H,\rm int}$
- Exact $C^1$ continuity is enforced mathematically by construction through the matching equations:
  $$\begin{aligned}
  \Delta t_{\rm rd} &= \left.\frac{d\phi_{\rm int}}{df}\right|_{f_{H,\rm int}} - \left.\frac{d\phi_{\rm rd}}{df}\right|_{f_{H,\rm int}} \\
  \Delta \phi_{\rm rd} &= \phi_{\rm int}(f_{H,\rm int}) - \left(\phi_{\rm rd}(f_{H,\rm int}) + \Delta t_{\rm rd} \cdot f_{H,\rm int}\right)
  \end{aligned}$$
- Ringdown phase evaluation is then:
  $$\phi_{\rm rd,matched}(f) = \phi_{\rm rd}(f) + \Delta t_{\rm rd} \cdot f + \Delta \phi_{\rm rd}$$
  ensuring continuous phase value ($C^0$) and frequency derivative ($C^1$) across $f_{H,\rm int}$.
- Piecewise phase selection for both full grid and reference phase uses $f_{\rm join\_rd} = f_{H,\rm int}$.
- Note on verification: Finite-tolerance numerical checks in regression tests verify the code implementation, but exact $C^1$ continuity holds analytically from these boundary matching relations.

---

## 4. Phase Shape Residuals and Status of Global Parity

- Diagnostic tests comparing this formulation against public waveform outputs after diagnostic affine alignment ($\Delta t_{\rm aff} \cdot f + \Delta \phi_{\rm aff}$) yield phase shape residuals below $3 \times 10^{-11}$ rad across 16 equal/unequal-spin configurations ($q \in [1, 100]$).
- **Unresolved Global Time Origin**:
  While the intrinsic phase shape $\phi''(f)$ and $C^1$ derivative matching agree to high precision, the global linear phase/time gauge $(\Delta t_{\rm global}, \Delta \phi_{\rm global})$ remains unresolved.
- In accordance with pair programming guidelines, no ad-hoc or fitted time corrections have been introduced. Full waveform parity is NOT claimed (raw waveform parity is not yet achieved), and further parity validation is deferred to Codex's external test harness.

---

## 5. Verification Coverage (Executed by Codex)

All numerical tests are executed externally by Codex in its dedicated environment. (The initial 32-test suite passed in 2.58s on an earlier revision; on the strengthened test suite, Codex reported 272/273 tests passing across the broader suite, with the single continuity slope test failure now resolved via one-sided linear extrapolation to the boundary, pending rerun). Agy performs static code maintenance only and runs no tests locally.

The suite in `tests/test_xas_phase_reconstruction.py` validates:
1. $C^0$ phase continuity ($< 10^{-3}$ rad) and autograd frequency derivative continuity on normalized complex phase using one-sided linear extrapolation to the boundary ($< 5 \times 10^{-5}$ relative difference, with analytical matching $< 1.84 \times 10^{-11}$) across both transition joins ($f_{\rm trans\_phase}$ and $f_{\rm join\_rd}$).
2. Phase reference calibration ($f_{\rm ref}$) and exact gauge rotations under time shift $t_c$ ($< 10^{-12}$) and orbital phase $\phi_c$ ($< 10^{-12}$).
3. Batched vs. scalar numerical consistency via relative L2 norm difference ($< 10^{-12}$) with positive reference norm assertion.
4. Clean autograd gradient flow and `torch.autograd.gradcheck` on normalized full complex waveform across mass and spin parameters.

## Final external verification

Codex reran the complete repository test suite after the extrapolation correction: **273 passed in 16.19s**. See [test log](../verification/full-pytest-final.log). The direct strict catalog remains **6/16 cases passing**, with unchanged gates; passing unit tests is not evidence of catalog parity. The production XAS holdout diagnostics and source limitations are summarized in [the current report](../completion_report.md).
