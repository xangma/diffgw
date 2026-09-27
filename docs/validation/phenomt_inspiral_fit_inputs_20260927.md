# IMRPhenomT inspiral fit-input transcription, 2026-09-27

`T-PN-FIT-INPUT-v1` checks the aligned `(2,2)` TaylorT3 time shift and five
inspiral frequency collocation targets against the pinned MIT phenomxpy
author source. It is a finite-grid transcription and assembly check, not
independent reconstruction of the numerical calibration or complete waveform
qualification. The [component specification](../math/phenomt-inspiral-fit-inputs.md)
freezes the grid and thresholds.

## Source and results

The author asset is `phenomxpy/phenomt/fits.py` at revision
`4121fafa0484f3db2a7352f4eea697faaaf29f65`, SHA-256
`2dfcca659cd50d927b9e164867913a586e9b1372f58ffde424f8bfbd74b56e21`.
The source-derived fixture has SHA-256
`729963d920439c7741c631fbc9982704ec1a785007d7b15911a146cd1b7936c3`.
On `len`, the source verifier reproduced all six outputs in five frozen cases
within `2e-13` relative or absolute error. The backend gate passed **37 tests,
no skips**: 15 fit-input cases (Python, Torch, JAX), six collocation cases
(Torch and JAX), and 16 TaylorT3 checks. The second gate ran without
`JAX_ENABLE_X64` set; the maintained tests explicitly enabled JAX float64.

The source-backed fit results agreed with the shared native fit functions at
`rtol=2e-11, atol=5e-10` for `t0` and `rtol=2e-11, atol=2e-12` for the five
frequencies. Scalar pseudo-PN coefficients assembled from source-derived fit
targets and independent TaylorT3 coefficients agreed with both native
carriers at `rtol=2e-8, atol=1e-8`, including on-node and off-node frequency
checks. No tolerances were changed after the runs.

## Reproduction

Host: `len`, CPU. Scratch cwd:
`/home/xangma/tmp/diffgw-phenomt-fit-inputs-20260927`. The pinned author
asset must be supplied separately at `author-fits.py` with the exact hash
above. Python is
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python` (3.12.3); package
versions: NumPy 2.5.3, PyTorch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0,
pytest 9.1.1.

```sh
PYTHONPATH=src /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python scripts/verify_phenomt_author_inspiral_fits.py --author-fits author-fits.py
JAX_PLATFORMS=cpu PYTHONPATH=src:tests /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_phenomt_inspiral_fit_inputs_math_reference.py tests/test_phenomt_inspiral_collocation_math_reference.py tests/test_phenomt_taylor_t3_math_reference.py --reference-device=cpu -ra
```

Outputs: `verified 5 cases and six fit outputs per case`; `37 passed in
10.13s`. The second command's log is `gate-no-env-x64.log` in the scratch
cwd. The published THM paper defines the collocation construction but does
not print these fit coefficients or the full NR/EOB target tables; hence
physical calibration provenance remains conditional in the research ledger.
