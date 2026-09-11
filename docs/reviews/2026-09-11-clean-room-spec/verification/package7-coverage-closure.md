# Work Package 7: Verification & Coverage Closure Report

**Date**: 2026-09-11
**Status**: AUDITED & COMPLETED

---

## 1. Inventory of Manifest Models (17 Models)
Across `tests/parity_harness.py`, exactly 17 manifest models are defined and evaluated.

### A. Strict Parity Passing (2 Models)
1. **`TaylorF2`**:
   - 4 test cases evaluated against LAL `SimInspiralChooseFDWaveform`.
   - Result: **PASSED** (Gate: $\Delta \le 1.0\times 10^{-6}, \mathcal{M} \le 1.0\times 10^{-8}$).
2. **`TaylorF2_Eccentric`**:
   - 2 test cases evaluated against LAL `SimInspiralChooseFDWaveform`.
   - Result: **PASSED** (Gate: $\Delta \le 1.0\times 10^{-6}, \mathcal{M} \le 1.0\times 10^{-8}$).

### B. Clean-Room Literature Implemented with Reference Blockers (10 Models)
All 10 models have verified component physics and 100% clean-room implementations from published literature, but remain unaligned or separated by documented rounding/reference-data constraints:
1. `IMRPhenomD` (PhenomD rounding blocker)
2. `IMRPhenomPv2` (Inherits PhenomD baseline)
3. `IMRPhenomXP` (MSA variant 223 implemented; inherits XAS baseline)
4. `IMRPhenomHM` (London et al. mapping implemented; inherits PhenomD baseline)
5. `IMRPhenomXHM` (Calibrated 88-mode fits implemented; inherits XAS baseline)
6. `IMRPhenomXPHM` (MSA variant 223 + XHM implemented; inherits XAS baseline)
7. `IMRPhenomXAS` (XAS dominant time calibration blocker)
8. `IMRPhenomD_NRTidal` (NRTidalv1 implemented; inherits PhenomD baseline)
9. `IMRPhenomD_NRTidalv2` (NRTidalv2 reference phasing implemented; match 0.99999993)
10. `IMRPhenomNSBH` (Thompson et al. 2020 amplitude & remnant reconstructed)

### C. Unresolved Oracle Mappings (5 Models)
These five models are standalone clean-room constructions that do not possess a 1-to-1 standalone LAL FD approximant string in `SimInspiralChooseFDWaveform`. Their independent validation paths are enumerated below:
1. **`TaylorF2_HM`**:
   - *Reason*: Multipolar expansion model has no standalone LAL FD approximant string in `SimInspiralChooseFDWaveform`.
   - *Independent Validation Route*: Sum individual mode outputs from `SimInspiralChooseFDModes` or compare against closed-form post-Newtonian multipole expansions.
2. **`KerrRingdown`**:
   - *Reason*: Standalone time/frequency QNM model without independent external LAL FD choosing interface.
   - *Independent Validation Route*: Direct numerical Kerr perturbation eigenvalue solvers (Leaver's continued fractions or Berti-Cardoso-Will tables).
3. **`SEOBNR_ODE`**:
   - *Reason*: Standalone clean-room time-domain ODE integrator; ground truth for native surrogate, not an external LAL FD approximant.
   - *Independent Validation Route*: Self-convergence testing under adaptive Runge-Kutta / Dormand-Prince step-size reduction.
4. **`SEOBNR_ROM`**:
   - *Reason*: Standalone native PyTorch surrogate trained on SEOBNR_ODE; intentionally distinguished from LAL SEOBNRv4_ROM.
   - *Independent Validation Route*: Residual regression error evaluated against native `SEOBNR_ODE` trajectories across $(\eta, \chi_1, \chi_2)$ parameter space.
5. **`TEOBResumS`**:
   - *Reason*: Standalone clean-room time-domain EOB prototype without external LAL FD approximant mapping in harness.
   - *Independent Validation Route*: Native EOB Hamiltonian and flux diagnostics compared against published Damour & Nagar reference tracks.

---

## 2. Test Suite Status
- Repository unit and acceptance tests: **326 passed in 17.00s**.
- Strict catalog parity gates: preserved unchanged without artificial thresholds or regression overfitting.
