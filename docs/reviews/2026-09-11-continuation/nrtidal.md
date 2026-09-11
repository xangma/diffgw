# NRTidal continuation — 2026-09-11

This phase repairs the merger taper and adds the published NRTidalv2 amplitude correction. All 285 tests pass, but strict catalog parity remains **6/16 cases**. Neither tidal case passes the unchanged acceptance gates (raw complex relative L2 ≤ 1e-6 and mismatch ≤ 1e-8).

## Changes and sources

- The merger fit uses the effective tidal coupling κ_eff = 3 Λ_tilde / 16. The paragraph preceding Eq. 11 in [Dietrich et al., *Matter imprints in waveform models for neutron star binaries: tidal and self-spin effects*](https://arxiv.org/html/1804.02235v1) explicitly specifies this substitution for κ_2. The mass-ratio factor uses the lighter/heavier mass ratio; the coupling retains the association between each mass and its deformability. Swapping both bodies gives the same merger frequency.
- NRTidalv2 adds the amplitude from Eq. 24 of [Dietrich et al., *Improving the NRTidal model for binary neutron star systems*](https://arxiv.org/html/1905.06011v2):

  `A_T = -9 sqrt(5 π η / 24) M²/D κ_eff x^(13/4) (1 + 449/108 x + 22672/9 x^2.89)/(1 + 13477.8 x⁴)`.

  Here M and D are geometric seconds and x = (π M f)^(2/3). Eq. 25 tapers the sum of the BBH and tidal amplitudes. The implementation removes the plus-polarization inclination factor to obtain the common BBH amplitude and applies one amplitude factor to both polarizations. Safe denominators preserve gradients at zero bins and beyond the BBH cutoff. Nonpositive frequencies return exact zeros.
- The existing v1 phase coefficients retain the published relation `d1 = n1 - 3115/1248`, from Eq. 7 of [Dietrich, Bernuzzi and Tichy, *Closed-form tidal approximants for binary neutron star gravitational waveforms constructed from high-resolution numerical relativity simulations*](https://arxiv.org/html/1706.02969v2). They were not rounded or fitted to oracle outputs. The v2 phase and self-spin formulas were not changed.

Historical scope matters: Table II of the 2019 paper omits amplitude and 3.5PN self-spin corrections from the historical IMRPhenomD_NRTidalv2 combination. Current public waveform outputs show additional amplitude structure. Applying the published amplitude equation improves agreement, but does not establish the current oracle's internal normalization or conventions.

This continuation consulted primary papers and public LALSimulation waveform outputs only. No LAL source, internal coefficient tables, third-party ports, or oracle-fitted coefficients were used in this phase. This is a scoped provenance statement, not a claim that all earlier contributors had zero source exposure. Native generation requires neither LAL/PyCBC imports nor LAL_DATA_PATH.

## Review and verification

Agy was used for three sequential attempts: Gemini 3.8 Flash Low, Medium, and High. Each returned a network-error result after saving partial edits, despite a zero CLI return code. Review found incomplete changes and, in the last patch, undefined variables, incorrect polarization normalization, and inadequate tests. Codex took over within the skill's three-quality-attempt cap, repaired the code and rewrote the reconstruction tests. [Delegation record](verification/delegation.json).

The tests exercise the published effective coupling, independent leading and next-to-leading PN amplitude limits, four inclinations, zero-tide recovery, dtype contracts, taper bounds, scalar/batched agreement, and finite-difference gradients of both complex polarizations. Gradcheck uses a fixed 1e-23 amplitude scale and includes taper, DC, negative-frequency and cutoff bins.

- [Full suite](verification/pytest.log): **285 passed** in 17.66 seconds.
- [Native runtime check](verification/native-runtime.json): **16/16** finite, nonzero waveforms with LAL/PyCBC imports blocked and LAL_DATA_PATH unset; this is not a parity claim.
- [Frozen strict catalog](verification/strict-catalog.json): **6/16 passed**, valid inventory, no manifest drift. No harness, case, gate, or manifest was changed.

| Frozen case, plus polarization | Raw complex relative L2 | PSD mismatch | Result |
|---|---:|---:|---|
| IMRPhenomD_NRTidal | 1.1403102e-4 | 2.1769497e-9 | Fail |
| IMRPhenomD_NRTidalv2 | 1.1813034e-1 | 6.9785460e-3 | Fail |

Separate [before](verification/tidal-before.json) and [after](verification/tidal-after.json) diagnostics use four systems on 30–1800 Hz, spacing 0.25 Hz, distance 100 Mpc, inclination 0.4, and reference frequency 30 Hz. For the nonspinning 1.4/1.3 solar-mass system with deformabilities 350/420, v2 raw complex L2 improves from 7.06e-3 to 1.36e-3. For equal 1.4 solar-mass bodies with deformability 1000, it improves from 1.85e-2 to 3.52e-3. Tidal-to-BBH ratios in these files isolate corrections for diagnosis; they do not replace full-waveform acceptance. No phase/time alignment was fitted.

## Remaining gaps

The spinning v2 diagnostic retains an approximately 0.118 rad phase offset. Its reference-phase/self-spin convention is unresolved; no empirical offset was subtracted. Nonspinning amplitude residuals also remain, and the v1 phase has a small unresolved difference despite using published coefficients. Baseline IMRPhenomD errors contribute to raw waveform errors. These gaps require further source-backed derivation or published precision data. The work does not establish strict parity for either tidal model or the wider catalog.

## Reproduction

Run from the repository root with a Python environment containing the recorded dependencies in the strict report, and pytest available:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider tests/ --tb=short
PYTHONPATH=src python tests/parity_harness.py --strict --json-out strict-catalog.json
PYTHONPATH=src python docs/reviews/2026-09-11-independent/verification/check_native_runtime.py --json-out native-runtime.json
PYTHONPATH=src python docs/reviews/2026-09-11-continuation/verification/check_tidal_factors.py --json-out tidal-after.json
```

The strict command exits nonzero while cases fail. To reproduce the before diagnostics, export `src/torchwave/approximants/imrphenomd_nrtidal.py` from commit `e8620f7ea2031cf21d1bb213c9b565051ca3fad7` to a temporary file and pass its path with `--baseline-file` to the diagnostic script. The diagnostic script imports oracle libraries for verification only.
