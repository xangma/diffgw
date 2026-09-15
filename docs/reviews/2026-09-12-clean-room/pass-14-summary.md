# Pass 14: odd-mode amplitude correction and independent fit checks

Completed 13 September 2026 with three fresh-context **GPT-5.6 Luna** subagents
and parent review/verification. The odd-mode inspiral and intermediate amplitude
consumers now take the absolute value of signed calibration fits, as required
by the paper. Previously, negative intermediate values were clamped almost to
zero before inversion. Raw fit coefficients are unchanged.

The [source audit](pass-14-amplitude-assets.md) exposed negative raw `(2,1)`
fits at a chosen equal-mass, unequal-spin point. The
[runtime correction](pass-14-odd-amplitude.md) includes production-path target
checks and finite, nonzero gradients. Independent numerical fixtures also
check eight intermediate amplitude fits and three nonmixing p6 phase fits;
normal tests need no downloaded source cache. Parent review verifies their
arithmetic correspondence and reproducible generation. Matching expressions
does not establish production candidate selection or all calibration preprocessing.

**No additional strict case passes; full LAL parity remains unfinished.**
This is a source-supported behavior correction, not a full-waveform parity result.

## Verification

| Check | Result |
| --- | --- |
| Full ordinary suite | **1,126 passed, 8 skipped, 75.37 seconds** |
| Frozen strict catalog | **11/21 passed** |
| Frozen strict holdouts | **4/13 passed** |
| Separate explicit components | **3/34 passed** |
| Mode-sum closure controls | **4/4 passed** |

All suites were rerun after integration. The pass-13 measurements are used only
as the before baseline. Inventories, grids and gates are unchanged. Six skips
require CUDA and two require generation-only gmpy2; MPS tests pass. The catalog
still contains nine numerical failures and one unresolved complete reference.

All frozen catalog and holdout case metrics are unchanged from pass 13.

The [comparison](verification/pass-14/comparison.json) retains complete before
and after component metrics. These overlapping controls add no frozen passing
case. No binary residuals were supplied to implementers or used to fit coefficients.

The verifier ran on **RONI2**, cwd `/Users/xangma/repos/torchwave`, driver PID
**48054**, from `2026-09-13T10:33:37.441616+00:00` to `2026-09-13T10:35:12.512389+00:00` using
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` and the
[driver](verification/pass-14/verify.py). All children finished; exact commands,
PIDs, logs and exit codes are in the [run record](verification/pass-14/verification-run.json).
All **185** runtime/test/script/fixture/manifest hashes
remained stable during verification. The later manifest edit only records results.
See the [patch](verification/pass-14/changes-from-pass13.patch) and
[parent integration review](verification/pass-14/integration-review.json).

## Remaining work and source boundary

Complete ordered amplitude vetoes, replace the inherited 11-point denominator
heuristic with complete interval root detection, and resolve sparse phase-node
selection and calibration-target metadata. The [phase audit](pass-14-phase-assets.md)
resolves p6 expression correspondence; repeating that lookup is unnecessary.
MSA, EOB, ROM and complete Kerr waveform work also remains, as described in the
[remaining-work assessment](pass-12-remaining-work.md).

Implementers used primary papers, author mathematical `.m` assets and native
TorchWave files. LAL/PyCBC source, translated ports, generation notebooks and
verifier outputs were excluded. Primary assets came from
[García-Quirós et al., arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2).
Complete downloaded records are ignored local review copies; tracked records
retain URLs, hashes and independently evaluated numerical fixtures. This does
not assert new redistribution rights or certify inherited provenance. The
clean-room separation is an instruction boundary, not OS isolation.

Existing calibration tables remain in use; the deferred NR job was not restarted
or scheduled. No commit or pull request was created.
