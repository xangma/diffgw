# Round-five review

The implementation remains incomplete. This review independently reproduced **215 passing tests in 15.77 seconds** and **6/6 strict TaylorF2/TaylorF2Ecc cases**. The newly introduced XAS mathematics substantially worsens oracle agreement, and the current model tests accept much lower overlaps.

Verified progress: previous DC/support, bad metric, projection, missing-PyCBC and empty-CLI-suite controls now reject the injected failures. Omitted XPHM spins and supported direct spin aliases work. XAS no longer obtains its phase from D; HM and XHM now select different default carriers. The XAS batch test uses a meaningful strain-scaled tolerance. D/XAS/XHM manifest statuses are more cautious.

## Current numerical results

CPU float64/complex128, LAL 7.7.0, LALSimulation 6.2.0, PyCBC 2.10.0, Torch 2.9.1. XAS: distance 500 Mpc, f_ref=20 Hz on both sides, df=.125 Hz, band `[20,512)`. These are flat-noise diagnostic matches, independently maximized over time/phase for each polarization; they are not strict certification. Magnitude relative L2 is unaffected by these alignment freedoms.

| XAS masses; spins | Previous optimized plus match | Current match | Current magnitude relative L2 |
|---|---:|---:|---:|
| 35,30; .2,-.3 | .997625015 | .808545065 | .097438750 |
| 50,15; .6,.4 | .999395333 | .818339692 | .050668823 |
| 60,7.5; -.4,.1 | .998313923 | .863503383 | .047600059 |
| 75,5; .5,-.2 | .993811984 | .851934020 | .032064562 |
| 30,30; .8,.8 | .999143657 | .778521135 | .054221550 |

All four original XAS magnitude errors also increased; the first went from 7.93% to 9.74%. At masses 40,20, spins .3,-.1 and inclination 1 rad, the XHM plus match fell from .983573322 to .842129648. HM gives .967444014. D's high-spin magnitude error remains .004308618.

## Material remaining problems

- The XAS oracle tests now accept .80/.85 (`tests/test_imrphenom_xas.py:45`), XHM .75 (`tests/test_imrphenom_hm.py:229`), and XPHM .80 (`tests/test_imrphenom_xphm.py:153`). The strict harness contains no cases for these models, despite the XAS test docstring referring readers there for strict validation.
- XAS, XHM and XPHM raise `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` under both `torch.no_grad()` and `torch.inference_mode()`. The XAS forward path unconditionally calls `autograd.grad` at lines 812–814. D and HM pass these controls.
- Strict pytest still passes an empty suite. Six copies of one case pass the suite's count check. An all-NaN aLIGO PSD array passes with reported PSD mismatch zero; a negative-infinite PSD metric also passes. These are separate from the repaired previous controls.
- Detector comparison in the harness uses only H1 at synthetic GMST zero. At actual GPS 1470000000, public-oracle comparisons give H1/L1 delay errors of about 31.6/36.0 microseconds. The linear GMST approximation needs validation over the supported time range.
- XAS silently ignores unknown keywords. XHM still shares HM's higher-mode construction; changing the carrier alone does not establish XHM parity.
- `docs/reviews/2026-09-10-parity/completion_report.md` retains incompatible VERIFIED PARITY claims and stale numbers. The previous round's JSON evidence was overwritten without refreshing its logs and hash manifest. New measurements require a separate run bundle.

## Reproduction

The new `round-5-evidence/` directory contains the scripts, numerical results, full test log, strict output and an input/evidence hash manifest. `controls.py` preserves the prior control script; its old absolute-tolerance demonstration is not the current batch test. `additional_checks.py` reproduces the new execution-context and harness failures. No implementation files were edited during this review.
