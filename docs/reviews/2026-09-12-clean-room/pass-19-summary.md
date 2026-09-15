# Pass 19: source-first parity continuation

Date: 2026-09-13. Passes 18 and 19 retain one source-derived runtime
correction and close four bounded source audits. Full LAL parity remains
unfinished.

## Retained NSBH correction

Thompson et al. Appendix A computes the Zappa NSBH final mass and spin before
obtaining the London ringdown frequency and applying the closed
`f_tide >= f_RD` merger-class boundary. The native path previously used a BBH
remnant frequency conditionally for one finite-Lambda class. It now uses the
Zappa remnant and its ringdown frequency for every class. The Lambda-zero BBH
limit, batching and off-boundary gradients remain intact. See the
[primary derivation](pass-18-nsbh-classification.md) and
[independent review](pass-18-nsbh-classification-independent-review.md).

The low-Lambda pre-merger coefficient remains unchanged. Thompson prints
`1 - 0.5*Lambda - 0.25*Lambda^2` for `Lambda <= 1` and `1.25` above one, so
the source itself has an exact unit jump at the boundary. The cited Pannarale
papers support multiplying that correction by the PhenomC `gamma1`; they do
not resolve Thompson's omitted `gamma1` notation or supply a continuity fix.
See the [primary audit](pass-19-nsbh-low-lambda-primary.md) and
[independent review](pass-19-nsbh-low-lambda-independent-review.md).

## Source audits without runtime changes

- The official PhenomD archives do not contain full-precision calibration
  coefficients or QNM metadata. The native Table V ordering, joins and remnant
  scaling are source-supported, while absolute affine phase anchors and the
  hard `Mf=0.2` cutoff remain unresolved. See the
  [PhenomD asset audit](pass-18-phenomd-assets-primary.md).
- Printed H1/L1 detector geometry agrees with the native definitions at the
  authoritative precision. The permitted sources do not determine exact Virgo
  arm-axis conventions, KAGRA geometry, or UT1 without Earth-orientation data.
  See the [detector audit](pass-18-detector-primary.md).
- TaylorF2_HM is a leading-order four-mode subset. The sources publish more
  modes, PN amplitude and odd-mode spin-difference terms, but do not select a
  truncation or reconcile the complete harmonic-frequency SPA normalization
  with the native stored-mode convention. See the
  [primary audit](pass-19-taylorf2-hm-primary.md) and
  [independent review](pass-19-taylorf2-hm-independent-review.md).
- KerrRingdown is a valid causal equal-twin, amplitude-supplied primitive. A
  complete physical waveform requires `M/D_L`, independent opposite-m
  excitations, overtone excitation metadata and a declared complex spheroidal
  convention. The papers do not determine those merger-dependent inputs. See
  the [primary audit](pass-19-kerr-ringdown-primary.md) and
  [independent review](pass-19-kerr-ringdown-independent-review.md).

## Verification

With `PYTHONPATH=src:tests`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`LAL_DATA_PATH=/tmp/torchwave-oracle-data`, and
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python`, the ordinary suite
passed **1,142 tests with 9 skipped in 78.49 s**.

The unchanged strict verifiers were run separately after the runtime settled.
They remain **11/21 catalog**, **4/13 holdouts**, **3/34 explicit components**,
and **4/4 mode-sum controls**. The retained classifier does not add a passing
case. Its catalog NSBH maximum polarization/projection complex relative L2
errors change from `2.9541899e-06`/`3.0959595e-06` to
`3.6745664e-04`/`3.6758860e-04`. The NSBH holdout remains effectively
unchanged at `1.6230897e-03`/`1.6232098e-03`. This regression is recorded
rather than used to tune a source-derived branch.

The Pass 19 outputs are in `/tmp/torchwave-pass19/`. No acceptance threshold,
calibration coefficient, reference case, or parity harness was changed.

Final retained hashes:

| File | SHA-256 |
| --- | --- |
| `src/torchwave/approximants/imrphenom_nsbh.py` | `e35473938e92d1e3fef50de8dfbd9d41ff6c8787fe39e8472db6e21c3ca974c7` |
| `tests/test_pass18_nsbh_classification.py` | `52ced6776882b8fe36a42db61e1dff5e4ec28c6a5897ef982b2765fe4a2998ac` |

The user-deferred 48-worker numerical-relativity run remains stopped and is
not required for these source-derived changes.
