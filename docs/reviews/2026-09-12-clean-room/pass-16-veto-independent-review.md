# Pass 16 independent review: intermediate amplitude vetoes

Date: 2026-09-13. This review used the permitted primary source, García-Quiros
et al., [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914), PDF SHA-256
`00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee`, and the
archived excerpt `source-assets/pass-14-amplitude/hmc2-relevant-excerpts.tex`
(SHA-256 `f3b5bf04413e3602b7b0db264c3c1cb3ce5e392c372849a560030c4637d32897`).
No runtime file was changed.

## Findings

1. **The implementation omits two source-specified preliminary veto stages.**
   In `src/torchwave/approximants/imrphenom_xhm.py:142-163`, the `0.01`
   ringdown-amplitude check and the monotonicity check are explicitly left
   inactive. Section V.A.3 of v2 states that, after the per-node `(2,1)`
   check, a ringdown amplitude below `0.01` removes both intermediate points;
   it then removes `f_Int2` when the surviving `(2,1)` points are not
   monotonic. The same paragraph specifies the ringdown and monotonicity
   checks for `(3,2)`, and Table VI specifies that the `(3,2)` ringdown veto
   also removes the boundary derivatives. The existence and action of these
   stages are source behavior; only the exact sequence inputs for the
   monotonicity test need a documented implementation choice. Suggested fix:
   implement the ordered checks with detached boolean masks, preserving the
   source order and the `still alive` condition, and set
   `remove_derivatives` only for the `(3,2)` ringdown rule.

   The stale-test change in `tests/test_pass15_xhm_vetoes.py:28-35` now
   asserts that a deliberately non-monotone `(2,1)` sequence removes no node,
   and `:48-55` asserts that a below-threshold ringdown value remains active.
   Those assertions document the omission rather than test the published
   model. They should be replaced with source-derived ordering tests when the
   stages are implemented. The focused current tests pass (`12 passed` for
   `test_pass16_xhm_vetoes.py` and `test_pass15_xhm_vetoes.py`), but this does
   not validate the omitted rules.

2. **The raw-amplitude normalization is asserted more strongly than the
   source establishes.**  The production call at
   `src/torchwave/approximants/imrphenom_xhm.py:1202-1208` passes the
   intermediate fits `A_\{lm\}^{Inter}/A_0` directly, while
   `_ordered_amplitude_vetoes` compares `A_0 |h_int|` to `0.2` at
   `:151-158`. The paper gives the fit normalization in Eq. (5.3), but
   describes the veto as the strain Fourier amplitude `|\tilde h_21|` in
   geometric units and does not state that multiplying the fitted ratio only
   by the frequency-independent `A_0` is the complete conversion to that
   quantity. The test comments in `tests/test_pass16_xhm_vetoes.py:24` and
   `tests/test_pass15_xhm_vetoes.py:49-51` therefore treat an implementation
   convention as source-resolved. Suggested fix: either derive and document
   the full conversion from the native `A/A_0` representation (including the
   common geometric normalization and frequency convention), or keep the
   threshold decision explicitly marked as an implementation assumption and
   avoid calling it “raw” in code/tests.

The parameter-space regions, strict inequalities, per-node `(2,1)` handling,
and batched boolean/autograd structure otherwise agree with Table V and the
runtime selection logic; no additional finding was made there.
