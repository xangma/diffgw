# Pass 30: TEOBResumS factorized `(2,2)` prerequisite audit

Date: 2026-09-13. This is a clean-room literature audit of the
nonspinning point-mass factorized mode that would be the input to a TEOBResumS
NQC extraction. It changes documentation only. No LAL, PyCBC, or TEOBResumS
implementation source was inspected, and no oracle or deferred NR data was
used.

## Primary sources

- Nagar et al., [arXiv:1904.09550v2](https://arxiv.org/abs/1904.09550),
  *A Multipolar Effective One Body Model for Non-Spinning Black Hole
  Binaries*. The locally audited PDF has SHA-256
  `9e0b00c9622834515fe37ff88cb2c85e8ef6927af4fea3949e17f334e5815014`.
- Nagar et al., [arXiv:1806.01772v2](https://arxiv.org/abs/1806.01772),
  *Time-domain effective-one-body gravitational waveforms for coalescing
  compact binaries with nonprecessing spins, tides and self-spin effects*.
  The locally audited PDF has SHA-256
  `d5efb16e4bfc3852aa443879dba5c49dd0460b4812fee7f58e26b50b82bc8ebc`.
- Nagar et al., [arXiv:2104.07533v2](https://arxiv.org/abs/2104.07533),
  *TEOBResumS: assessment of consistent next-to-quasicircular corrections and
  post-adiabatic approximation in multipolar BBH waveforms*.

## What is source-defined

The 1904 paper gives the factorized structure in its Eq. (3.16):

\[
h_{\ell m}=h^{(N,\epsilon)}_{\ell m}\,S^{(\epsilon)}_{\rm eff}\,
\hat h^{\rm tail}_{\ell m}(y)\,[\rho^{\rm orb}_{\ell m}(x)]^\ell\,
\hat h^{\rm NQC}_{\ell m}.
\]

Its Eqs. (3.17)--(3.18) define the tail as
\(\hat h^{\rm tail}_{\ell m}=T_{\ell m}\exp(i\delta_{\ell m})\) and give the
Newtonian multipole prefactor. For the improved Taylor-expanded nonspinning
model, Eq. (3.21) fixes the carrier

\[
h^{(N,0)}_{22}=-8\sqrt{\pi/5}\,\nu\,v_\phi^2 e^{-2i\phi}.
\]

This carrier, including its sign gauge, is a complete source-defined
subcomponent when `nu`, `v_phi`, and `phi` are supplied.

The 1806 paper independently presents the same factorized/NQC architecture
in Eq. (13), with the four-parameter `(2,2)` NQC form in Eq. (14), and fixes
the extraction time relative to the orbital-frequency peak in Eq. (15). The
2104 paper gives the iterative NQC matching procedure and identifies the
nonspinning `(2,2)` amplitude coefficients as NR-derived fits. These sources
therefore define the architecture and matching conditions, but not a
calibration-free waveform function from trajectory state alone.

## Why the full TEOBResumS h22 is not uniquely specified here

The papers leave several choices outside a single self-contained equation:

1. The 1904 paper distinguishes a fully Padé-resummed
   `TEOBiResumSMultipoles` option from the improved Taylor-expanded
   `EOBResumMultipoles+` option. The former hybridizes the finite-mass 3PN
   terms with test-particle information through 6PN and then applies the
   mode-specific Padé choices in Table I of an external reference. Those
   hybridization and Padé choices are not printed in the audited paper as a
   single `rho_22` definition. The latter uses a different Taylor-expanded
   truncation. Calling both “TEOBResumS” does not select one unique residual
   amplitude.
2. The h22 Newtonian factor uses `v_phi`, while the generic DIN factorized
   utility in this repository uses the Keplerian `v = omega**(1/3)` for its
   Newtonian factor and residual series. A silent substitution would change
   the model during plunge. The effective source, tail-radius convention,
   residual phase, and orbital variable must be selected together.
3. The NQC basis quantities are referred to earlier equations in the 1806
   paper rather than being fully restated there. They depend on radial motion
   and acceleration. The 2104 extraction procedure also uses calibrated
   matching data and an iterative flux/NQC update; the published coefficient
   fits are NR-inferred. They cannot be recovered from the factorized PN
   equations alone.
4. The conservative/radiation-reaction calibration, including the effective
   potential coefficient choices, is model-version dependent. Existing
   `TEOBResumSHamiltonian` and `teobresums.py` code is explicitly a prototype,
   so it cannot serve as an implicit convention selector for the waveform.

## Mapping to current TorchWave code

`torchwave.approximants.eob.factorized_modes.FactorizedModes` is a source-
complete DIN circular factorization. Its `(2,2)` mode is useful when the DIN
Keplerian convention is requested: it supplies the Newtonian factor, effective
source, tail, DIN residual amplitude, and residual phase, with batching,
autograd, dtype, and device support. Its module documentation explicitly
excludes non-Keplerian velocity replacement and NQC.

That mapping is not sufficient for TEOBResumS: the current utility has no
TEOB-specific hybrid/Padé `rho_22`, no `v_phi` trajectory prescription, and no
TEOB NQC calibration or extraction-time policy. The existing trajectory code
supplies radial derivatives, but those derivatives do not resolve the missing
waveform-version choices. Consequently no runtime wiring or coefficient
translation is justified in this pass.

## Decision and verification

No runtime change is made. The complete source-defined Newtonian h22 carrier
and the independent DIN factorized `(2,2)` implementation remain available in
their existing, explicitly scoped APIs. A TEOBResumS-specific h22 API should
wait for a declared target version plus the exact residual-amplitude table,
tail/phase conventions, `v_phi` definition, dynamics calibration, and NQC
matching inputs.

Focused source-derived tests were run with the repository's no-plugin test
environment:

```
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_factorized_modes.py \
  tests/test_clean_room_teob_trajectory.py \
  tests/test_clean_room_eob_nqc.py \
  tests/test_teobresums.py
```

These tests cover the existing DIN factorized mode, TEOB trajectory derivatives,
isolated NQC equations, and prototype TEOBResumS path. This pass adds no
calibration, parity threshold, oracle reference, or deferred NR dependency.
