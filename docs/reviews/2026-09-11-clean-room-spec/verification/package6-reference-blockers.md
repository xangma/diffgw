# Work Package 6: Reference-Data Blockers & Residual Tidal Audit

**Date**: 2026-09-11
**Status**: AUDITED & DOCUMENTED

This audit details the external reference-data blockers preventing strict machine-precision catalog parity across higher-order and legacy aligned/precessing models, while upholding the zero-contamination clean-room protocol.

---

## 1. PhenomD Baseline Calibration Precision
- **Identified Blocker**:
  - The published coefficients in Table V of Husa et al. (2016) [PRD 93, 044006 / arXiv:1508.07250] and Khan et al. (2016) [PRD 93, 044007 / arXiv:1508.07253] are rounded to 5–7 decimal places.
  - Due to extreme sensitivity in high-order polynomial evaluation for the intermediate collocation and ringdown ansatz, rounded coefficients lead to small phase discrepancies relative to double-precision LAL simulations.
- **Clean-Room Policy Enforced**:
  - Do **NOT** copy or transplant LAL-only extra unrounded digits directly from `LALSimIMRPhenomD.c`.
  - Unless the original unrounded author calibration release is published in a verifiable independent repository (e.g. Zenodo, arXiv ancillary release), the model retains its pure literature-based parameters.

---

## 2. IMRPhenomXAS Dominant-Mode Time/Peak Calibration
- **Identified Blocker**:
  - As documented in Appendix C of García-Quirós et al. (2020) [PRD 102, 064002 / arXiv:2001.10914], the dominant (2, 2) mode in PhenomXAS employs an unrounded time/phase shift calibration $(t_0, \phi_0)$ matching the peak of the waveform.
  - The ancillary mathematical release for arXiv:2001.10914v2 includes 88 higher-mode fit files (`xhm_fits.py`), but does not contain the standalone dominant-mode time calibration table.
- **Clean-Room Policy Enforced**:
  - Do **NOT** regress a linear phase slope $\Delta t \cdot f$ from oracle residuals.
  - The model continues to evaluate from the published analytical closed-form specifications without empirical residual fitting.

---

## 3. NRTidal Amplitude Normalization & Strain Mode Convention
- **Identified Blocker**:
  - Eq. 24 of Dietrich et al. (2019) [PRD 100, 044003 / arXiv:1905.06011] specifies an additive face-on strain correction $A_{\rm tidal}(f)$ designed for binary neutron stars.
  - When projected to polarizations or transformed between multipolar bases, differing convention choices (e.g. mode prefactor vs face-on strain prefactor) can introduce global constant scaling factors.
- **Clean-Room Policy Enforced**:
  - Maintain the derived mathematical convention without arbitrary scalar multipliers.
  - All tidal phase equations ($2\text{PN}/3\text{PN}$ and $3.5\text{PN}$ self-spin) remain strictly anchored to published literature.
