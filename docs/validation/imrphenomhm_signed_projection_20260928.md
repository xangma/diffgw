# HM signed-mode projection validation

The [signed-pair specification](../math/imrphenomhm-signed-projection.md)
extends the Gaussian SPA convention reference to real plus/cross polarizations.
It derives the exact pair as $D_A P+\overline{D_A}Q$, where
$Q(f)=\overline{P(-f)}$. A unit carrier correction requires conjugate
transport: $P\mapsto qP$, $Q\mapsto\overline qQ$. The common analytic carrier
phase in the proposed SPA conversion is generally distinct from a physical
polarization-basis rotation.

## Status

The focused CPU FP64 run on len passed **85 tests in 62.37 seconds**, with
no failures, skips or expected failures: 36 new signed-projection cases,
36 existing SPA-convention cases, and 13 existing Torch/JAX projection cases.
Ruff, the strict MkDocs build and both generated-table checks also passed.
Production implementations, previous receipts, frozen NR gates, and
qualification status are unchanged. No new NR holdout was used.

Source review covers the exact Fourier definitions and conjugate transport.
The new test module passes 36 parametrized cases, including direct
TD quadrature with deliberately nonzero leakage, both hemispheres and all six
modes, symbolic reconstruction and a polarization-rotation counterexample.
The direct quadrature uses 65-digit arithmetic and a separately bounded
Gaussian tail; it checks absolute errors below $10^{-48}$ while deliberately
retaining leakage that makes the one-sided answer fail. These are analytic
chirp and projection checks, not physical-accuracy tests of the intrinsic modes.

The preceding commit `e34fa4a` passed its GitHub workflow
[36388726280](https://github.com/xangma/diffgw/actions/runs/36388726280).
That workflow predates this extension and is not evidence for its new tests.

## Execution and replay

The run used `/tmp/diffgw-hm-signed-20260928/repo` on len, an archive of
`e34fa4af89a683db9c23d0519512b10f3e0470fc` with the reviewed new reference,
tests and documentation overlaid. Test PID 175385 completed on 2026-09-28
at approximately 09:02 UTC. Logs are `tests.log` and `docs.log` in the parent
directory; `tests.exit` records zero. No job remains running.

| Tested artifact | SHA-256 |
| --- | --- |
| `src/diffgw_reference/imrphenomhm_signed_projection.py` | `ba6c2ea2259f3228b0b75ede183310f6863afc19c962da126409d343be10b9fd` |
| `tests/test_imrphenomhm_signed_projection_math_reference.py` | `48e59c83eda958f2fb29d9ccee9c9747d56df95a637cd7acbebac17048f21af4` |

Dependencies were Python 3.12.3, SymPy 1.14.0, mpmath 1.3.0, pytest 9.1.1,
Torch 2.14.0, JAX/JAXlib 0.11.2, NumPy 2.3.5, MkDocs 1.6.1 and Ruff 0.16.8.
The environment is `/tmp/diffgw-waveform-library-1315f6e/venv-latest`.
Use its `bin` directory on `PATH` for this replay:

```sh
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q \
  tests/test_imrphenomhm_signed_projection_math_reference.py \
  tests/test_imrphenomhm_spa_conventions_math_reference.py \
  tests/test_imrphenomhm_projection_math_reference.py
ruff check src/diffgw_reference/imrphenomhm_signed_projection.py \
  tests/test_imrphenomhm_signed_projection_math_reference.py
python -m mkdocs build --strict
python scripts/generate_waveform_library_table.py --check
python scripts/generate_waveform_coverage.py --check
```

An initial generated-table invocation omitted `PYTHONPATH=src` and imported
an older installed catalog, producing a mismatch. Repeating it against the
snapshot source passed without modifying any generated document. The test
command used `PYTHONPATH=src` from its first invocation. The earlier SSH
authentication block was resolved before any numerical execution.

## Remaining waveform work

With this bounded projection gate passed, assess an explicitly versioned native convention
against the independent reference in both frontends, including derivatives
and complete polarizations. A leading-PN convention correction does not
address the measured merger/ringdown phase drift or amplitude errors. Those
require separate source-backed investigations before another frozen physical
qualification run.
