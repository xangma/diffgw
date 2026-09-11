# Work Package 3: PhenomXHM Calibrated Higher Modes — Verification Report

**Date**: 2026-09-11
**Clean-Room Implementer Subagent**: `61b82158-430a-4cc6-855c-feced06f320f`
**Status**: COMPLETED & COMMITTED (Commit `9499bb4`)

---

## 1. Clean-Room Boundary Confirmation
- Implementation derived purely from published literature and author ancillary mathematical files:
  - García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
  - Author ancillary Wolfram `.m` fit files (88 coefficient files hashed in `docs/reviews/2026-09-11-clean-room-spec/xhm-ancillary-sha256.json`)
- No LAL/LALSimulation/PyCBC code, internal tables, or empirical oracle fixtures were inspected or copied.
- No coefficients or phase/time offsets were fitted to oracle residuals.
- Subagent executed within an isolated branched worktree with a clean-room specification barrier.

---

## 2. Key Components Implemented (`src/torchwave/approximants/imrphenom_xhm.py` and `xhm_fits.py`)
1. **Wolfram `.m` Fit Extraction**:
   - 88 fit polynomials converted to pure Python/PyTorch dictionaries in `src/torchwave/approximants/xhm_fits.py`.
   - Verified SHA-256 provenance against author ancillary release.
2. **Multipole Mode Decomposition**:
   - Supported modes: $(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)$.
   - Negative-$m$ modes reconstructed via exact conjugate parity: $h_{\ell, -m}(f) = (-1)^\ell h_{\ell m}^*(f)$.
3. **Inspiral Amplitude & Phase**:
   - García-Quirós et al. (2020) 3PN Fourier-domain amplitudes $\hat{H}_{\ell m}$ with collocation points `Insp1`, `Insp2`, `Insp3`.
   - Symmetries correctly preserve odd-$m$ modes ($(2, 1), (3, 3)$) when $m_1 = m_2$ if $\chi_1 \neq \chi_2$.
4. **Intermediate Matching**:
   - Confluent 5th-order inverse polynomial matching $C^1$ boundary values at $f_{\rm Ins}$ and $f_{\rm RD}$ plus intermediate collocation points `Int1` and `Int2`.
5. **Ringdown Model & Spherical-Spheroidal Mixing**:
   - Lorentzian ringdown ansatz with fits `ALambda`, `Lambda`, and mode-specific $\sigma$.
   - For $(3, 2)$, implemented spherical-spheroidal mixing: ringdown phase and amplitude evaluated in spheroidal basis, aligned to $(2, 2)$ carrier via $\Delta T_{32,S}$ and $\Delta \phi_{32,S}$, then transformed via $h_{32} = \alpha_{322} h_{22}^S + \alpha_{332} h_{32}^S$.
6. **Harmonic Projection**:
   - Summed over spin-weighted spherical harmonics ${}_{-2}Y_{\ell m}(\iota, \pi/2 - \varphi_{\rm ref})$.

---

## 3. Test & Verification Results
- **Acceptance Tests** (`tests/test_phenomxhm_reconstruction.py`):
  8 passed in 1.45s (finite non-zero strain, equal-mass unequal-spin non-vanishing odd modes, body swap symmetry, spherical-spheroidal mode mixing, autograd differentiability).
- **Full Test Suite**:
  All 311+ repository unit tests pass.
