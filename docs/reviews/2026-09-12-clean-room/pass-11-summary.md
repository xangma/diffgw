# Pass 11: source-derived components and independent review

Completed on 13 September 2026 using fresh-context **GPT-5.6 Luna**
implementers and reviewers. **Full LAL parity remains unfinished.** This pass
adds one NSBH correction, reusable MSA components and XHM regression coverage;
it adds no strict passing waveform case.

## Retained implementation

| Area | Change | Limit |
| --- | --- | --- |
| NSBH | Zappa tidal remnant polynomials use the initial BH spin specified in the paper. | Rounded coefficients and the BBH baseline are unchanged. Calibration assumes zero NS spin; nonzero NS spin remains an uncalibrated API extension. |
| MSA precession | Expose existing precession-averaged spin couplings and compute the published squared final angular momentum for explicit orbital angular momentum. | No automatic normalized or signed MSA Kerr remnant is selected. Existing XP/XPHM waveform behavior is preserved. |
| XHM | Extract and document the existing carrier ratio, with independent amplitude, matrix and production-flow checks. | The common mode normalization cancels. Waveform normalization is unchanged. |

The delta from the pass-10 snapshot is exactly three runtime files and three
new test files; see the [patch](verification/pass-11/changes-from-pass10.patch).
XAS, XP and XPHM were restored byte-for-byte to that snapshot after narrowing
the candidates, preserving the pass-10 NNLO implementation.

Two proposals were rejected during source review, before post-change binary
verification. An extra XHM factor double-counted a conversion common to both
numerator and denominator. The MSA proposal chose a final-mass denominator
despite conflicting initial/final-mass formulas in the primary paper; setting
initial total mass to one does not remove that difference. Its exact-zero
sign case and eager fallback also needed resolution. The retained squared
angular momentum helper avoids those choices. Initial independent reviewers
accepted aspects of both proposals; the parent challenged the algebra and
the reviewers corrected their conclusions. See the
[independent review](pass-11-independent-review.md).

## Verification

| Check | Result |
| --- | --- |
| Final focused integration | 55 passed |
| Complete ordinary suite | **1,087 passed, 8 skipped, 72.63 seconds** |
| Frozen strict catalog | **11/21 passed**, 9 numerical failures, 1 unresolved reference |
| Frozen strict holdouts | **4/13 passed** |
| Separate explicit-component probes | **3/34 passed**, unchanged before/after |
| Native/reference mode-sum controls | **4/4 passed** before and after |

The 12 new tests cover published spin variables, limiting identities,
normalization cancellation, actual mixing inputs, batching, invalid MSA rows
and gradients. Six full-suite skips require CUDA; two require generation-only
gmpy2 in the Torch test environment. MPS tests passed.

Every catalog metric is unchanged from pass 10. The only changed frozen
holdout is NSBH: maximum polarization complex relative L2 rises from
`0.0013294683` to `0.0016230897` (about 22% worse); maximum detector-projection
error rises from `0.0013295885` to `0.0016232098`. It fails before and after.
The source-defined BH spin is retained without tuning coefficients to this
residual. Tests passing establish implementation consistency, not parity.

The separate 34-case probe comprises eight explicit precession comparisons
(NNLO 102/FinalSpinMod0 and MSA 223/FinalSpinMod3), fourteen HM mode checks and
twelve XHM mode checks. All before/after metrics are identical. The three
passing cases are XHM's two 22 carriers and HM's holdout 22 carrier; all higher
modes still fail. These overlapping checks add no frozen passing cases.
XHM's maximum carrier polarization errors are `5.78e-14` and `7.15e-14`.

The public XHM reference selects both signed modes and disables multibanding;
native positive mode entries include their physical pairs. Four mode-sum
controls close to at most `1.71e-16` in the reference and exactly in native
arithmetic. Initial positive-mode-only and default-multibanding setup probes
are archived separately; neither is used as the final baseline. Public
dictionary lookups report defaults 300/4 for the installed binary, which does
not establish its internally resolved dispatch.

Frozen case inventories, parameters, frequency grids, scopes and acceptance
gates are unchanged. Gates remain unaligned complex relative L2 and relative
norm errors at most `1e-6`, with flat/named-PSD mismatches at most `1e-8` for
polarizations and projections. The comparison script verifies these invariants.

## Evidence and next work

- [NSBH derivation](pass-11-baseline.md), [XHM derivation](pass-11-higher-modes.md),
  [MSA derivation and ambiguity](pass-11-precession.md).
- [Primary source audit and hashes](verification/pass-11/source-audit.json),
  [before/after comparison](verification/pass-11/comparison.json),
  [run record](verification/pass-11/verification-run.json),
  [full test log](verification/pass-11/pytest.log).
- [Current full inventory and remaining work](summary.md),
  [updated manifest](../../model_manifest.json).

The next bounded implementation work is to resolve the MSA mass convention
from independent primary evidence, then integrate the selected remnant with
strict/fallback validity behavior. Higher-mode phase/collocation conventions
remain another independent track. The verified carrier normalization should
be preserved. Existing calibration tables remain in use; the user-deferred
NR run was not restarted or scheduled.

The verifier ran on `RONI2`, cwd `/Users/xangma/repos/torchwave`, with
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python`, using the
[driver](verification/pass-11/verify.py). Driver PID 36202 ran from
08:50:06 to 08:51:38 UTC; all child processes finished. Test/catalog/holdout/
component exit codes were 0/1/1/1. All 169 runtime, test, script and manifest
hashes were stable during verification. A separately hashed manifest update
then recorded the results. No commit or pull request was created.

Implementers and reviewers were instructed to use primary papers and inherited
native TorchWave code, excluding LAL/PyCBC implementation source, translated
ports and binary residuals. The parent used public binary APIs separately.
This instruction boundary does not certify inherited code provenance or
constitute OS-enforced isolation.
