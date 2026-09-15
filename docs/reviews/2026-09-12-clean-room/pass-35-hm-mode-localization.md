# Pass 35: clean-room PhenomHM mode localization

Date: 2026-09-13. This pass measures the remaining `IMRPhenomHM` residual
mode by mode through the public LALSimulation API and checks the current
runtime against the primary PhenomHM paper. No LAL, PyCBC, TEOBResumS, or
other waveform implementation source was inspected. No residual was used to
fit a runtime coefficient.

## Method

The public `lalsimulation.SimIMRPhenomHMGethlmModes` API was called on the
same frequency grid as TorchWave. Its positive-m mode series was compared to
`IMRPhenomHM(..., return_unprojected_modes=True)`. A single complex scale was
first obtained from the `(2,2)` mode, which agrees with the carrier to about
`1.1e-6` relative L2 in the diagnostic and `4.8e-7` in the holdout. This
removes the public mode-series normalization while preserving all
frequency-dependent differences. The reported `optL2` additionally allows a
constant complex scale for that mode; it is diagnostic only.

The diagnostic is `(m1,m2,chi1,chi2,phic,f_ref)=(40,15,0.3,-0.2,0.5,20)`
and the holdout is `(55,11,-0.4,0.2,0.71,27)`, with `f=20..512 Hz` and
`df=0.25 Hz`.

| mode | diagnostic amp 20--100 | diagnostic amp 400--512 | diagnostic optL2 | holdout amp 20--100 | holdout amp 400--512 | holdout optL2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 1.078 | 7.949 | 0.150 | 1.016 | 0.832 | 0.167 |
| 33 | 1.000 | 1.046 | 0.0140 | 1.000 | 0.929 | 0.0148 |
| 32 | 0.970 | 1.116 | 0.0601 | 1.039 | 1.248 | 0.100 |
| 43 | 0.882 | 0.827 | 0.0303 | 1.062 | 1.189 | 0.0490 |
| 44 | 1.000 | 1.034 | 0.0174 | 1.000 | 1.044 | 0.0171 |

The high-frequency 21 amplitude ratio is ill-conditioned where the small
odd mode approaches a zero; its optimized full-band residual remains the more
stable measure. The largest non-(3,2) residual is 21, while 33 and 44 are at
the percent level after removing a constant phase/normalization. The 32 and
43 results are consistent with the already documented spherical/spheroidal
mixing limitation.

## Source audit

London et al., *First higher-multipole model of gravitational waves from
spinning and coalescing black-hole binaries*, arXiv:1708.00404, gives the
following construction:

* Eq. (3) maps the carrier amplitude and phase to each mode.
* Eq. (4) defines the SPA rescaling through `H_hat_lm`; the paper also states
  the alternative `H_hat_lm(f)/H_hat_22(f_22^A)` form used for aligned-spin
  behavior.
* The restricted leading-PN and 1.5PN `H_hat_lm` expressions used by the
  runtime are the published construction for the supported modes.
* The piecewise frequency mappings and their continuity conventions are
  implemented in `frequency_mapping_amplitude` and
  `frequency_mapping_phase`. The distinct London transition constants and
  the PhenomD carrier joins are documented in the pass-30 audit; the primary
  sources do not uniquely authorize swapping them in this runtime.
* The paper explicitly identifies spherical/spheroidal mixing as absent from
  PhenomHM and particularly important for `(3,2)` and `(4,3)`.

Reviewing these equations against
`src/torchwave/approximants/imrphenom_hm.py` found no single source-backed
defect that explains the measured 21/33/44 residuals. The remaining
frequency-dependent errors can arise from the paper's approximate mapping and
its unresolved phase/time-gauge choices; changing a PN factor, transition,
QNM fit, or mode sign from the black-box residual would be oracle fitting.
Adding mixing would define an extended model rather than repair the published
PhenomHM construction.

Therefore this pass makes **no runtime change**. The quantitative blocker is
model-level approximation and incomplete authoritative calibration metadata,
not a uniquely established implementation error.

## Validation

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_phenomhm_reconstruction.py tests/test_clean_higher_modes.py \
  tests/test_imrphenom_hm.py
```

Result: **50 passed in 9.47 s**. `git diff --check` was run after adding this
report. No deferred NR run was started.
