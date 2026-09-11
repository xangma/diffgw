# Work Package 4: PhenomXP / XPHM Precession Prescriptions (MSA Variant 223) — Verification Report

**Date**: 2026-09-11
**Clean-Room Implementer Subagent**: `95bce3e5-25b2-4775-afeb-985fc9b44115`
**Status**: COMPLETED & COMMITTED (Commit `2da9cb7`)

---

## 1. Clean-Room Boundary Confirmation
- Implementation derived purely from published literature:
  - Pratten et al. (2020), Phys. Rev. D 103, 104056 (arXiv:2004.06503)
  - Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
  - Chatziioannou et al. (2013), Phys. Rev. D 88, 063011 (arXiv:1307.4418)
- No LAL/LALSimulation/PyCBC code, internal tables, or empirical oracle fits were used.
- No coefficients or phase/time offsets were fitted to oracle residuals.
- Subagent executed within an isolated branched worktree with a clean-room specification barrier.

---

## 2. Key Components Implemented (`src/torchwave/approximants/precession_msa.py`, `imrphenom_xp.py`, `imrphenom_xphm.py`)
1. **Orbital Angular Momentum & Depressed Cubic Roots**:
   - 3PN spinning orbital angular momentum $L(v)$.
   - Depressed cubic root solver for spin magnitude roots $S_3^2 \le S_-^2 \le S_+^2$.
2. **Multiple-Scale Analysis Precession Angles**:
   - Azimuthal angle $\alpha(v) = \alpha_{-1}(v) + \alpha_0(v)$ with closed-form $\phi_z^{(n)}(v)$ antiderivatives.
   - Coprecessing angle $\gamma(v) = -\epsilon(v)$ via $\langle\Omega_\zeta\rangle^{(n)}$.
3. **Reference Frame Alignment & Conventions**:
   - Convention 1 (default) from Pratten et al. Table 4 and Appendix B ($\alpha_{\rm offset}, \epsilon_{\rm offset}, \zeta$).
   - Exact polarization rotation by $\exp(2i\zeta)$.
4. **Graceful Fallback**:
   - Fallback to NNLO if roots are unphysical or if $q - 1 < 10^{-4}$.
5. **Architectural Integration**:
   - Created `IMRPhenomXP` dominant-mode precessing approximant twisting up `IMRPhenomXAS` carrier with MSA angles.
   - Updated `IMRPhenomXPHM` with default `precession_method="msa"` and `prec_version=223`.

---

## 3. Test & Verification Results
- **Acceptance Tests** (`tests/test_phenomxp_msa_reconstruction.py`):
  8 passed in 1.95s (aligned-spin limit matches `IMRPhenomXAS`, two-spin independence, body exchange symmetry, reference frequency changes, fallback behavior, autograd differentiability).
- **Full Test Suite**:
  All 319 repository unit tests pass.
