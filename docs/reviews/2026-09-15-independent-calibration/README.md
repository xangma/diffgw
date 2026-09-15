# Independent (3,3) amplitude calibration pilot

## Erratum: input quality and metadata

The [follow-up diagnosis](extraction-diagnosis/README.md) found that six of
seven simulations were paired with metadata from a different resolution,
and all seven selected simulations are deprecated. The original q≈3 NR
target also shows a large resolution-dependent ringdown extraction discrepancy:
N2/N3 disagreement falls from 30.14% at Lev3 to 3.40% at Lev5.

The frozen artifacts and results below are retained as historical evidence.
They require a corrected data-selection and metadata pipeline before a new
calibration can be assessed. Correcting the q≈3 relaxation time alone has a
negligible effect; changing resolution substantially reduces the apparent
NR/LAL disagreement without changing the fit.

## Scope

This experiment independently fits the **ringdown-frequency amplitude** of the
nonspinning `(3,3)` mode to public numerical-relativity (NR) simulations. It
uses the published XHM functional form and selected settings; it is not a
reproduction of the authors' complete calibration procedure or a complete
waveform model.

- Published choices: [XHM paper](https://arxiv.org/html/2001.10914v2),
  Eq. (42), Table 8 (`sigma33=1.3`), and Table 2 / Eq. (41)
  (`0.95*f_ring33 <= Mf <= 0.3`).
- Independent choices: dataset selection, finite-NR preprocessing, fitting
  objective, parameter-space representation, and validation split.
- Ringdown frequencies: TorchWave's independently written Kerr eigenproblem
  solver, supplied with each NR simulation's remnant mass and spin.
- Training simulations: SXS:BBH:0008, 0169, 0182, 0056, 0063
  (mass ratios approximately 1.5, 2, 4, 5, 8).
- Held-out simulations: SXS:BBH:0030 and 0166 (approximately 3 and 6).
  Additional extraction orders of these cases are validation data only.

## License and provenance

The original pilot code remains MIT. Each of the seven exact NR dataset
records states CC-BY-4.0; the separate
[data attribution notice](DATA_ATTRIBUTION.md) and
[independent record verification](verification/dataset-license-records.json)
preserve those terms and creator information. No upstream contact is needed
to use the permissions those records already grant.

LAL output is reserved for validation after the independently calculated
coefficients have been frozen. It does not enter training. This experiment
does not resolve provenance questions for inherited calibration arrays.

## Interpretation

Physical agreement with held-out NR and numerical agreement with LAL are
separate measurements. An independent calibration can improve the former
while moving away from LAL's particular calibration. Amplitude errors on a
limited frequency band do not establish complex-waveform parity or mismatch.
The NR inputs also have finite numerical and extraction errors.

The runtime inputs were identical to the prior strict-parity baseline at
[pilot startup](verification/runtime-identity-start.json). The baseline was
28/34 passing cases. This pilot is a review artifact and has not been promoted
into the waveform runtime.

Final verification: **15 focused tests passed**, all five pilot scripts
compiled, and [all 139 baseline inputs remained identical](verification/runtime-identity-end.json).
The prior 28/34 result is reused on that basis; the full parity suite was not
rerun. Tests cover FFT sign, direct-fit recovery, holdout exclusion, changed
input rejection, units, matching grids, and calibration integrity checks.

## Results

The [independent calibration](nr33-calibration.json) was frozen at
2026-09-15 09:32:53 UTC, before LAL validation. Its SHA256 is
`f447acda86f64f1d13745c8a5d725acdf295b050a2638d1731d4b6a139e1358d`.
The [freeze record](nr33-calibration-freeze.json) records the command,
environment, input identities and split. The fitter opened only five training
waveforms and verified their declared hashes. All five independent Kerr QNM
solves converged, with radial residual magnitudes below `2e-14`.

Direct training-fit relative L2 residuals, in the paper's normalized amplitude
`A33/A022(f)`, were 14.51%, 1.35%, 0.66%, 4.29%, and 1.05% for q approximately
1.5, 2, 4, 5, and 8. These describe the individual direct fits, before the
parameter-space polynomial is applied. The fixed quadratic fit has RMS
coefficient residuals 0.0692 in `log(|a_lambda|/delta)` and 0.2554 in `lambda`.

### Held-out validation

The [validation report](nr33-validation.json) and
[comparison plot](nr33-heldout-comparison.png) use the frozen parameter-space
fit, with no refitting, scale adjustment, or alignment. Each row is an
amplitude-only relative L2 error, `||prediction-target|| / ||target||`, on the
same native FFT frequency bins within the prescribed ringdown band.

| Held-out mass ratio | Fit versus NR | Fit versus LAL | NR versus LAL |
| --- | ---: | ---: | ---: |
| q≈3 | 27.70% | 8.60% | 42.21% |
| q≈6 | 3.18% | 3.41% | 0.479% |

LALSimulation 6.2.0 / LAL 7.7.0 was called only after freeze, with XHM release
**122022 explicitly selected** through the public setter. The paper's v2
formula and this LAL release are distinct version references. The public
one-mode API uses `(3,-3)` for positive frequencies; the NR FFT conjugates
the SXS `(3,3)` convention. Physical Fourier amplitudes use
`Msec*(Msec*c/D)`, with total mass 50 solar masses and distance 100 Mpc. The
small q≈6 NR/LAL discrepancy is a useful normalization cross-check, not a
fit target. Actual calibration-file hashes match before and after validation.

### NR data sensitivity

The [post-freeze audit](nr33-preprocessing-audit.json) varies the start/end
tapers and halves the resampling time step across all seven baseline cases.
The largest amplitude L2 change is **0.0846%**; no coefficients are refitted.
The [extraction comparison](nr33-extraction-comparison.png) shows a larger
limitation:

| Held-out case | N2 versus N3 amplitude L2 | N2 / N3 amplitude norm |
| --- | ---: | ---: |
| SXS:BBH:0030, q≈3 | 30.14% | 0.79733 |
| SXS:BBH:0166, q≈6 | 0.0591% | 0.99969 |

The q≈3 extraction discrepancy spans the main ringdown band. It prevents
treating that single N3 spectrum as a precise physical target. These two
extraction orders do not provide a complete NR error bound; a numerical
resolution study is reported in the [follow-up diagnosis](extraction-diagnosis/README.md).
Small endpoint/sampling changes do not explain the large extraction discrepancy.

### Scope limits and next step

The pilot demonstrates independent generation of calibration coefficients,
but does not establish LAL parity or production accuracy. Before extending
the fit, compare NR resolutions and extraction orders using a predefined
quality rule, then reserve fresh validation simulations. This pilot's held-out
results must not become training targets for a claimed independent retest.

The simulations are near-nonspinning (reference spin magnitudes up to about
`1.5e-5`); the parameter fit treats them as nonspinning. It covers q≈1.5–8,
uses NR remnant metadata, and fits amplitude in a limited frequency band.
Runtime use would also require a validated remnant model, spin dependence,
phase, and attachment to the remaining waveform. **Units correction:** SXS
waveforms use total reference-mass units, as specified in the
[SXS catalog documentation](https://arxiv.org/html/2505.13378). The original
pilot treated catalog remnant masses as fractions directly; a corrected
pipeline must divide the matching resolution's remnant mass by its total
reference mass. These mass-scale differences are at the parts-per-million
level and do not explain the large extraction gap.

## Reproduce

Raw waveform inputs and decoded NPZs are cached outside the repository in
`/tmp/torchwave-nr33-pilot/`. The manifest pins public URLs and checksums.
The preparation helper uses a separate environment with `sxs==2026.0.3`;
the fitting/validation environment is recorded in the freeze and run records.
See [the detailed method](nr33-fit-method.md).

From the repository root, using the recorded environment:

```sh
python -m scripts.prepare_nr33_calibration_data --download \
  --manifest /tmp/nr33-replay-manifest.json
python -m scripts.fit_nr33_amplitude_pilot \
  /tmp/nr33-replay-manifest.json \
  --output /tmp/nr33-replay/calibration.json
PYTHONPATH=src python -m scripts.run_nr33_frozen_validation \
  --calibration docs/reviews/2026-09-15-independent-calibration/nr33-calibration.json \
  --manifest docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json \
  --expected-sha256 f447acda86f64f1d13745c8a5d725acdf295b050a2638d1731d4b6a139e1358d \
  --output-dir /tmp/nr33-validation-replay --with-lal
```

The first command requires the SXS environment. Regenerating the manifest with
different paths or software can change its hash; preserve the original
manifest for validation of this frozen artifact. Fresh fits should be written
to a separate output directory and treated as separate experiments.
