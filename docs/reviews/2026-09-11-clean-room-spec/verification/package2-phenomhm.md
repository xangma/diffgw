# Work Package 2: PhenomHM Mapping Reconstruction — Verification Report

**Date**: 2026-09-11
**Clean-Room Implementer Subagent**: `30e1995d-f09f-45ef-ab59-e8a3ebac8086`
**Isolated Worktree**: `/Users/xangma/.gemini/antigravity/brain/ab8503fe-b41f-4387-a2d9-4234a84753d1/.system_generated/worktrees/subagent-PhenomHM-Clean-Room-Implementer-self-a3272e8c`

---

## 1. Clean-Room Boundary Confirmation
- Implementation derived purely from published literature:
  - London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
  - Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
  - Mishra et al. (2016), Phys. Rev. D 93, 084054 (arXiv:1601.05588)
- No LAL/LALSimulation/PyCBC code, internal tables, or empirical oracle fits were used.
- Subagent executed within an isolated branched worktree with a clean-room specification barrier.

---

## 2. Key Components Implemented
1. **Mode (4, 3) QNM Fit & SWSH**:
   - Fundamental QNM fit coefficients from Berti et al. Table VIII:
     $f = (1.6869, -0.8862, 0.2822)$, $q = (1.4812, 2.8096, -0.4271)$.
   - Spin-weighted spherical harmonic projection:
     $c_{\rm norm} = \frac{3\sqrt{14}}{16\sqrt{\pi}}$,
     $Y_{4,3} = c_{\rm norm} \sin\iota (1+\cos\iota)^2 (1-2\cos\iota) e^{3i\phi_0}$,
     $Y_{4,-3} = c_{\rm norm} \sin\iota (1-\cos\iota)^2 (1+2\cos\iota) e^{-3i\phi_0}$.
2. **Piecewise Frequency Mappings**:
   - $f_{22}^A(f)$ and $f_{22}^\varphi(f)$ with transition frequencies $f_0^A = 0.018 \frac{f_{\ell m}^{\rm RD}}{f_{22}^{\rm RD}}$ and $f_0^\varphi = 0.014 \frac{f_{\ell m}^{\rm RD}}{f_{22}^{\rm RD}}$.
3. **Phase Derivative Integration & $C^0$ Continuity**:
   - $\varphi_{\ell m}(f) = \kappa \varphi_{22}[f_{22}^\varphi(f)] + \Delta_{\ell m}$ with exact $C^0$ continuity offsets at $f_0^\varphi$ and $f_{\ell m}^{\rm RD}$.
4. **Amplitude Rescaling with Spin Asymmetry**:
   - $A_{\ell m}(f) = |\beta_{\ell m}(f)| A_{22}[f_{22}^A(f)]$ with published $\hat{H}_{\ell m}$ expansions.
   - Non-zero odd modes for equal-mass unequal-spin binaries ($\delta = 0$, $\chi_{\rm a} \neq 0$).
5. **Architectural Separation**:
   - `IMRPhenomHM` and `IMRPhenomXHM` separated into distinct classes.

---

## 3. Test & Verification Results
- **Acceptance Tests** (`tests/test_phenomhm_reconstruction.py`):
  9 passed in 1.10s (mapping continuity, phase continuity, derivative connection, finite strains, body swap, spin asymmetry, autograd differentiability).
- **Existing HM Tests** (`tests/test_imrphenom_hm.py`, `tests/test_imrphenom_xphm.py`):
  11 passed in 2.18s.
- **Full Repository Test Suite**:
  303 passed in 16.12s.
- **Strict Catalog Parity Suite**:
  6/16 cases pass strict criteria (TaylorF2, TaylorF2_Eccentric); remaining catalog models documented in parity log.
