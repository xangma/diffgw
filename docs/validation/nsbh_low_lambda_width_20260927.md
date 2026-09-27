# NSBH low-tidal ringdown-width gate — 27 September 2026

The [six-section v1 specification](../math/nsbh-low-lambda-ringdown-width.md)
was frozen in `a78e646` before adding the independent scalar reference,
production-called Torch/JAX helpers, or native tests. It transcribes the
closed $0\leq\Lambda\leq1$ branch of [Thompson et al. (2020), Eq. (29),
and Eq. (28)](https://arxiv.org/html/2002.08383v2). The scalar code uses
mpmath and imports neither native backend nor LAL. The native refactor
keeps the original polynomial operation order and the historical public
branch selection.

**Validation pending on len.** No numerical tests or scientific compute
were run on the Mac for this slice. The frozen test spans 112 supplied
input tuples, 70/90-digit scalar stability, both endpoint identities,
Torch/JAX CPU FP64 values, six conditional partials, and domain rejection.
The expected limits are $10^{-30}$ for scalar stability, $2\times10^{-13}$
for native values, and $2\times10^{-12}$ for native partials, all absolute.

From a clean len checkout containing both commits, use the existing
Torch/JAX/mpmath environment:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src /home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python -m pytest -q -s tests/test_nsbh_low_lambda_width_math_reference.py tests/test_nsbh_amplitude_variant_provenance_math_reference.py tests/test_nsbh_amplitude_window_sum_math_reference.py tests/test_nsbh_clean_amplitude.py tests/test_clean_room_nsbh.py tests/parity/test_nsbh_parity.py
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python scripts/generate_waveform_library_table.py --check
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python -m ruff check src/diffgw_reference/nsbh_low_lambda_width.py src/torchwave/approximants/nsbh_clean_amplitude.py src/torchwave/approximants/imrphenom_nsbh.py src/jaxwave/approximants/imrphenom_nsbh.py tests/test_nsbh_low_lambda_width_math_reference.py
```

Record the len host, cwd, environment versions, pass counts, and measured
maxima after execution. A passing component gate will cover only the
source-unambiguous low-$\Lambda$ correction and width on supplied inputs.
The [large-$\Lambda$ Eq. (29) sign and Eq. (26) provenance](../math/nsbh-amplitude-variant-provenance.md),
joining-center discrepancy, upstream provider guards, full amplitude, and
whole waveform remain unqualified.
