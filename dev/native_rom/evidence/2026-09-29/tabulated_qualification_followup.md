# Native SEOBNRv4 tabulated qualification follow-up

This branch starts from `b8d33abc1fcd803d2ab2ce07b6622d5daec91af6` and
uses generator/receipt revision `311215021d4d6afa9c278ea5af8ba6c6dd6f17f8`.
The [frozen specification](../../reference/SPEC.md),
[manifest](../../reference/manifest.json),
[ell8 disposition](ell8_source_disposition.md), and
[release ledger](../2026-09-27/historical_fidelity_release.json) retain their
acceptance rules. No equation, coefficient, model asset, catalog, or release
gate changed. The historical source-eligibility ledger remains **31/35**.

## Question and discriminating checks

The selected compatibility scope is `spin_truncation="tabulated"` for the
`(8,2)`, `(8,4)`, `(8,6)`, and `(8,8)` residuals. The qualification CLI now
records that selector in `config.scientific_choices` and passes it to each
native solve. The whole-waveform diagnostic requires a recorded selector from
the prerequisite numerical campaign, passes it to its native solves, and
records it in its own receipt. `generic-leading-ell8` remains available.

The frozen numerical row has zero spins. At a representative scalar point
`(m1,m2,omega,omega_circ,H_eff,v_phi)=(30,20,.08,.077,.91,.975)`, the
independent reference's four ell8 even amplitudes are exactly equal under
the two selectors for `(chi1,chi2)=(0,0)`. For `(.2,-.1)`, their largest
relative difference is `0.0289960263506817`. Therefore the nonspinning
campaign checks the numerical gate but cannot discriminate this selector;
the separate spinning-row campaign checks its selected path. These component
values do not establish whole-waveform agreement.

Both campaigns retain the `1e-6` complex and amplitude relative-L2 budget,
the `0.05 M` event-time budget, and the decreasing-error requirement. The
ODE levels use `(atol,rtol)=(1e-12,1e-11),(1e-13,1e-12),(1e-14,1e-13)`;
the grid steps remain `0.5,0.25,0.125 M`. LAL comparisons use the public
binary API pinned to `lal==7.7.1`, `lalsimulation==6.2.1`. A whole-waveform
receipt is diagnostic even if its numerical prerequisite fails.

## Versioned public capability comparison

| Code and version | Checked option and actual interface | Qualification implication |
| --- | --- | --- |
| LALSuite binary on `len`, LAL 7.7.1 / LALSimulation 6.2.1 | `GetApproximantFromString` and implemented-domain queries return `SEOBNRv4`: TD yes, FD no; `SEOBNRv4_ROM`: TD no, FD yes; `SEOBNRv4HM`: TD yes, FD no; `SEOBNRv4HM_ROM`: TD no, FD yes. The [versioned LALSimulation header documentation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_inspiral__h.html) lists these distinct approximants. | The public `SEOBNRv4` TD binary is the pinned whole-waveform diagnostic. An exposed ROM approximant does not provide diffgw with independent training data or a validated native ROM. |
| Same LALSuite binary and host | An actual `SimInspiralChooseFDWaveform` call for `SEOBNRv4_ROM` at `(30,20,.2,-.1)`, `52–1024 Hz`, `deltaF=.25 Hz` fails because `SEOBNRv4ROM_v3.0.hdf5` is absent from `LAL_DATA_PATH`. | This host verifies the FD interface but cannot execute that ROM without its separate public data asset. No data asset was fetched or copied. |
| pySEOBNR documentation snapshot `0.3.7.dev32+g4fef6d1c0` | The [documented interface](https://waveforms.docs.ligo.org/software/pyseobnr/source/basic_usage.html) demonstrates `SEOBNRv5HM` time-domain modes and frequency-domain polarizations, with v5PHM and v5EHM examples. The [frozen ell8 audit](ell8_source_disposition.md) separately observes tabulated residuals at that commit. | The documented v5 examples and the limited ell8 observation are not a v4 whole-waveform or ROM parity certificate. No implementation code was used as an equation source here. |
| diffgw at `3112150` | Independent paper-derived scalar reference plus Torch and JAX component paths; experimental native TD generator and explicit ell8 selector. No independently generated v4 basis, spline tensors, complete knots, or qualified training generator. | The native [Bohé Sec. VII ROM architecture](https://arxiv.org/abs/1611.03703) remains pending until source, numerical, waveform, and holdout gates close. |

## Execution

The committed snapshot was transferred by `git archive` to isolated `len`
directory `/home/xangma/tmp/diffgw-native-seobnr-qualification-3112150`;
`source_revision.txt` records the exact commit because the archive has no
`.git` directory. The numerical commands use
`PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu`
and `/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`:

```sh
python -m dev.native_rom.qualify_generator --spin-truncation tabulated \
  --atol-levels 1e-12 1e-13 1e-14 --rtol-levels 1e-11 1e-12 1e-13 \
  --compare-lal --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out qualification-tabulated.json
python -m dev.native_rom.qualify_generator --inputs spin_row.json \
  --spin-truncation tabulated --atol-levels 1e-12 1e-13 1e-14 \
  --rtol-levels 1e-11 1e-12 1e-13 --compare-lal \
  --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out qualification-tabulated-spin.json
```

`spin_row.json` is the second row of the frozen
[whole-waveform inputs](../2026-09-23/whole_waveform_inputs.json).
The first run's PID/log were `708062`/`qualification-tabulated.log`;
the spinning run's were `778569`/`qualification-tabulated-spin.log`.
The unmodified pre-receipt-change source was independently exercised with an
in-memory tabulated selector in a separate archive at
`/home/xangma/tmp/diffgw-native-seobnr-qualification-20260929`, PID `460550`,
log `qualification-tabulated.log`. That baseline cannot serve as the final
versioned CLI receipt.

## Baseline result and hypothesis check

The [pre-change tabulated receipt](qualification_tabulated_baseline.json)
finished `failed`, with one complete five-solve row, unchanged source hashes,
and no runtime error. The wrapper selected `tabulated` in memory before
calling the unchanged qualifier. The final ODE comparison has complex L2
`9.079205866631182e-6` and amplitude L2 `3.4717694919773975e-7`;
the final grid comparison has complex L2 `6.1498554800270005e-6` and
amplitude L2 `2.6143134705476666e-7`. Event-time gates pass. The ODE
error decreases but exceeds `1e-6`; the grid error rises from
`1.3333963887868187e-7` to `6.1498554800270005e-6` and exceeds
`1e-6`. The old [generic qualification](../2026-09-23/qualification.json)
has the same gate pattern, with final complex L2 `8.748496280033223e-6`
and `6.081529368739555e-6`. This is a reproduction of the numerical
block, not evidence that the selector caused it: both options give the same
ell8 scalar values for this zero-spin row, and the source revisions differ.
The versioned CLI rerun and spinning-row campaign below were the next
decisive checks. No physics patch or threshold change was made.

## Committed-revision numerical results and closure

The committed [nonspinning receipt](qualification_tabulated_nonspin.json)
and [spinning receipt](qualification_tabulated_spin.json) both record
`spin_truncation="tabulated"`, LAL 7.7.1 / LALSimulation 6.2.1, unchanged
source hashes, one complete five-solve row, no runtime errors, and final
status `failed`. The archive's `source_revision.txt` is
`311215021d4d6afa9c278ea5af8ba6c6dd6f17f8`. The receipts' source
manifest hash is
`ba2c668401077505afd7b095eff1b81e2f13d03aba5bcf4a6e16df52383b8427`.
The pre-change and committed nonspinning numerical metrics agree exactly;
the receipt-wiring change did not alter that run's outputs.

| Frozen input row | ODE coarse/medium → medium/fine complex L2 | ODE final amplitude L2 | Grid coarse/medium → medium/fine complex L2 | Grid final amplitude L2 | Failed gates |
| --- | ---: | ---: | ---: | ---: | --- |
| `(30,20,0,0,.04)` | `2.135988e-5 → 9.079206e-6` | `3.471769e-7` | `1.333396e-7 → 6.149855e-6` | `2.614313e-7` | ODE complex budget; grid complex budget and decreasing error |
| `(30,20,.2,-.1,.04)` | `6.314091e-5 → 4.938449e-5` | `1.470160e-6` | `5.761866e-7 → 5.507150e-7` | `1.609781e-8` | ODE complex and amplitude budgets |

All event-time gates pass in both rows. The spinning grid passes all its
numerical gates, but its ODE tolerance gate fails. The public-binary LAL
comparison in each receipt is an unaligned convention diagnostic, with
complex L2 `1.98735` and `1.86359`, respectively; it is not a calibrated
whole-waveform parity score. The earlier [peak-anchored whole-waveform
diagnostic](../2026-09-23/whole_waveform.json) remains at complex L2
`0.021709` nonspinning and `0.013367` spinning on an earlier source
revision. No new whole-waveform comparison was run on this committed
tabulated revision, and no parity or physical-accuracy claim follows from
these numerical receipts.

The selected ell8 option is recorded and exercised, but the frozen numerical
gate remains failed. The historical source gate remains **31/35**; the
selected ell8 disposition does not close the other four source items.
Whole-waveform parity, independent physical/NR holdouts, and the native ROM
training assets and validation remain outstanding. No ROM training, selector
pin for production, release promotion, deployment, or PR was performed.

The archived receipt SHA-256 values are `82a54fcc060125f907f2683e1bfe97c1733842fbcb947945a571b001c43798a5`
for nonspinning, `995fe0751800d8a315f7c5d8e44a2a39bb2e375590e3fcb57e99a6d8dff01472`
for spinning, and `619404d8934f9a62bf5b0906773c697ae9fd0e826d258c7e5b0c28f8bdb3679f`
for the pre-change baseline. On the committed branch, the focused
`test_qualification.py` and `test_whole_waveform_diagnostic.py` suites pass
21/21 and 2/2 tests, respectively. The execution archive was used only for
these qualification runs; it was not deployed as a production waveform.
