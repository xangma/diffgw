# Work Package 5: PhenomNSBH Amplitude and Remnant Reconstruction — Verification Report

**Date**: 2026-09-11
**Clean-Room Implementer Subagent**: `87f33195-9a1d-43f9-95c4-03da1615abae`
**Status**: COMPLETED & COMMITTED (Commit `522ed4f`)

---

## 1. Clean-Room Boundary Confirmation
- Implementation derived purely from published literature:
  - Thompson et al. (2020), Phys. Rev. D 101, 124059 (arXiv:2002.08383)
  - Pannarale et al. (2015), Phys. Rev. D 92, 084050 (arXiv:1509.00512)
  - Pannarale et al. (2015), Phys. Rev. D 92, 081504 (arXiv:1509.06209)
  - Zappa et al. (2019), Phys. Rev. Lett. 123, 041102 (arXiv:1903.11622)
  - Jiménez-Forteza et al. (2017), Phys. Rev. D 95, 064024 (arXiv:1611.00332)
  - Santamaria et al. (2010), Phys. Rev. D 82, 064016 (arXiv:1005.3306)
  - London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
  - Dietrich et al. (2019), Phys. Rev. D 100, 044003 (arXiv:1905.06011)
- No LAL/LALSimulation/PyCBC code, internal tables, or empirical oracle fits were used.
- Subagent executed within an isolated branched worktree with a clean-room specification barrier.

---

## 2. Key Components Implemented (`src/torchwave/approximants/imrphenom_nsbh.py`)
1. **NS Compactness $C(\Lambda)$**:
   - Piecewise $C^1$-smooth function for $\Lambda > 1$ and $\Lambda \le 1$ matching $C(0) = 0.5$.
2. **Relativistic Tidal Disruption Root $\xi_{\rm tide}$ & Frequency $f_{\rm tide}$**:
   - Vectorized Newton-Raphson solver on the 5th-degree polynomial $P(\xi) = 0$ with residual $|P(\xi)| < 10^{-11}$.
   - Orbital frequency $f_{\rm tide}$ computed from circular equatorial Kerr metric at $r_{\rm tide}$.
3. **Baryonic Torus Remnant Mass Ratio $M_{\rm b, torus}/M_{\rm b, NS}$**:
   - Analytic Kerr ISCO radius $\bar{r}_{\rm ISCO}(\chi)$ and torus mass ratio $M_{\rm b, torus}/M_{\rm b, NS}$.
4. **Remnant Properties $M_f$ and $\chi_f$**:
   - Clean-room implementation of the 19-parameter `FinalStateUIB2016` BBH baseline for $M_f/M$ and $\chi_f$.
   - Zappa et al. tidal correction factors $p_k(\eta, \chi)$.
5. **Kerr Ringdown Frequency $f_{\rm RD}$ & Quality Factor $Q$**:
   - London et al. (2018) conformal fit $\tilde{\omega}(\kappa)$.
6. **Merger Classification & Window Parameters**:
   - Exact four-way classification (non-disruptive, mildly disruptive with $f_{\rm tide} \ge f_{\rm RD}$, mildly disruptive with $f_{\rm tide} < f_{\rm RD}$, and disruptive).
   - Dynamic window joining frequencies $\tilde{f}_0, \tilde{f}_1, \tilde{f}_2$ and width $d_{\rm win}$.
7. **Phenomenological Amplitude Assembly**:
   - Inspiral 3PN / TaylorT4 amplitude $A_{\rm PN}(f)$ using Santamaria et al. (2010) coefficients.
   - Pre-merger power-law amplitude $A_{\rm PM}(f)$ modulated by $\gamma'_1$.
   - Ringdown Lorentzian amplitude $A_{\rm RD}(f)$ damped by $\sigma = \delta'_2 f_{\rm RD} / Q$ and scaled by $\epsilon_{\rm tide}$.
8. **Phase Synthesis & Polarizations**:
   - Combines `IMRPhenomD` aligned-spin BBH phase with `IMRPhenomD_NRTidalv2` tidal phasing.
   - Synthesizes $h_{22}(f) = A(f) e^{-i\phi(f)}$ and projects onto spin-weighted spherical harmonics for $(h_+, h_\times)$.

---

## 3. Test & Verification Results
- **Acceptance Tests** (`tests/test_phenomnsbh_reconstruction.py`):
  7 passed in 0.51s (compactness monotonicity, root solver accuracy, disruption regimes, remnant limits, amplitude positivity & exponential suppression, batched parity, autograd differentiability).
- **Existing NSBH Suite** (`tests/test_nsbh.py`):
  6 passed in 0.08s.
- **Full Repository Test Suite**:
  All 326 repository unit tests pass in 17.00s.
