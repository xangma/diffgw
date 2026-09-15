# Pass 15: ordered amplitude selection and interval denominator checks

Completed with three fresh-context **GPT-5.6 Luna** subagents, parent integration,
and separate parent-only public-binary verification. **No additional strict case passes; full LAL parity remains unfinished.**

XHM now removes the higher-frequency inner amplitude node when the four
collocation amplitudes are nonmonotonic in common `A/A0` units. The four-value
sequence is an explicitly documented interpretation of an underspecified
paper rule, not a uniquely established prescription. The `q > 40`, `chi1 > 0.9`
rows remove each remaining inner node. Removing one node retains the other
and both boundary derivatives through a quartic inverse polynomial.

The old eleven-sample, five-percent denominator heuristic is replaced by a
native Torch numerical check of endpoints and stationary points throughout
the interval. Derivative-root isolation uses affine interval normalization
and batched bisection. The selected value-only fallback preserves surviving
nodes; a still-invalid fallback reduces to the endpoint line. Only the active
branch can trigger a fallback. Discrete decisions are detached while selected
polynomial gradients remain available.

The [independent source review](pass-15-veto-independent-review.md) found that
Table VI does not resolve raw `A` versus `A/A0` units for the `0.2`/`0.01`
thresholds. They remain inactive, including the `(3,2)` special derivative
veto. Both boundary derivatives are removed together for pole fallback; the
paper does not specify a one-at-a-time removal order. See the
[veto implementation](pass-15-vetoes.md) and [numerical method](pass-15-root-detection.md).

## Verification

| Check | Result |
| --- | --- |
| Focused reconstruction and new tests | 71 passed, 1 skipped |
| Independent parent polynomial extrema comparison | 600/600 matched |
| Full ordinary suite | **1,140 passed, 9 skipped, 81.76 seconds** |
| Frozen strict catalog | **11/21 passed** |
| Frozen strict holdouts | **4/13 passed** |
| Separate explicit components | **3/34 passed** |
| Mode-sum closure controls | **4/4 passed** |

Inventories, grids and acceptance gates are unchanged. The full ordinary suite
includes a new CUDA-conditional detector test; seven skips require CUDA and
two require generation-only gmpy2. MPS tests pass. Two older unit tests now
explicitly retain nodes to isolate their original normalization/fallback
contract; no external parity test was altered.

| Changed frozen case | Complex strain | Magnitude only |
| --- | ---: | ---: |
| IMRPhenomXHM_diagnostic | 0.32766557 → 0.38200117 | 0.25884171 → 0.28669083 |
| IMRPhenomXPHM_diagnostic | 0.10160197 → 0.16092642 | 0.065554273 → 0.10998196 |
| IMRPhenomXHM_holdout_6 | 0.65967621 → 0.65865751 | 0.44463141 → 0.44363111 |
| IMRPhenomXPHM_holdout | 0.56006875 → 1.1851135 | 0.44846581 → 1.1401227 |

The [comparison](verification/pass-15/comparison.json) retains complete before
and after component diagnostics, including regressions. These overlapping
controls add no frozen passing cases. No waveform binary residuals were given
to implementers or used for calibration fitting.

The verifier ran on **RONI2**, cwd `/Users/xangma/repos/torchwave`, driver PID
**70751**, from `2026-09-13T10:54:53.212305+00:00` to `2026-09-13T10:56:35.327583+00:00` with
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` and the
[driver](verification/pass-15/verify.py). All children finished. Exact commands,
PIDs, logs and exits are in the [run record](verification/pass-15/verification-run.json).
All **189** runtime/test/script/fixture/manifest hashes
were stable during verification; the later manifest change only records results.
See the [incremental patch](verification/pass-15/changes-from-pass14.patch) and
[parent review](verification/pass-15/integration-review.json).

## Remaining work and limits

Resolve threshold normalization and the monotonicity/fallback interpretations
from additional independent primary material; finish sparse phase-node
selection and calibration-target metadata. Broader MSA, EOB, ROM and complete
Kerr waveform work remains in the [assessment](pass-12-remaining-work.md).
The interval helper is a floating-point guard, not an algebraic certificate:
near-roundoff minima, clustered critical roots, and extreme overflow/underflow
limit classification. The random comparison excludes minima within `1e-9`
of zero; focused tests separately cover tangencies and narrow negative dips.

Implementers used primary papers and native TorchWave files. LAL/PyCBC source,
translated ports, generated notebooks, the earlier source-derived specification,
and verification artifacts were excluded. Primary source:
[García-Quirós et al., arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2).
No new source assets or runtime oracle dependencies were added. Separation is
an instruction boundary, not OS isolation; inherited provenance is uncertified.
Existing calibration tables remain in use. The deferred NR job was not restarted
or scheduled. No commit or pull request was created.
