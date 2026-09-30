# TaylorF2 eccentric 3PN phase qualification (2026-09-27)

## Claim and source

The executable claim is restricted to Moore, Favata, Arun and Mishra,
[arXiv:1605.00304v2, Eq. (6.26)](https://arxiv.org/pdf/1605.00304v2):
the dominant-harmonic SPA eccentric phase at order (e_0^2), through relative
3PN. The separate native phase-origin convention subtracts that paper term
at an explicitly supplied positive `f_ref`; (e_0) remains defined at
`f0_ecc`. The [component specification](../math/taylorf2-eccentric-phase.md)
records the equations, inputs and exclusions.

`src/diffgw_reference/taylorf2_eccentric.py` transcribes Eq. (6.26) as SymPy
rationals, π, Euler's constant and logarithms, then evaluates it using
mpmath at 80 decimal digits. It imports no Torch, JAX or LAL coefficients.
The only shared frontend value is the solar-mass-to-seconds unit conversion.
The paper polynomial includes mixed (v,v_0) powers and both 3PN logarithms;
the reference-frequency subtraction is a project convention, not a term of
Eq. (6.26).

## Execution

Base commit: `ab1faa158304b13c1651eb66ab3b26507f3b16c3`.
Host: `len` (Linux x86_64); working directory:
`/home/xangma/repos/diffgw-worktrees/eccentric-phase-reference`.
Python 3.12.8; Torch 2.8.0+cu128, JAX 0.11.2, SymPy 1.13.3,
mpmath 1.3.0, NumPy 2.2.6 and pytest 8.3.4. JAX was forced to CPU
with 64-bit arithmetic; both backend test fixtures pass float64 inputs.

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  /home/xangma/repos/diffgw-worktrees/kerr-angular-qualification/.venv/bin/python \
  -m pytest -q tests/test_taylorf2_eccentric_math_reference.py \
  tests/test_eccentric.py tests/parity/test_eccentric_parity.py \
  --reference-device=cpu
```

Result: **162 passed in 22.37 s**. The 44 new reference cases exercise
Torch and JAX separately over all six truncation orders, three mass and
eccentricity configurations, each PN increment, (e_0^2) scaling and zero,
and both complex polarizations at three distinct `f0_ecc`/`f_ref` relations.
No phase or time alignment is fitted. The existing eccentric behavior and
cross-frontend parity cases remain in the focused command as regressions.

The full CI math-reference invocation includes this new module through the
`test_*math_reference*.py` inventory in `tests/conftest.py`; with
`--require-math-reference` it rejects a skipped, expected-failure, missing,
or deselected reference case.

## Boundary

This is sampled coefficient and gauge evidence, not independent full-model
waveform verification or a physical accuracy claim. The circular carrier,
spin-eccentric coupling, oscillatory eccentric terms, cutoffs, GPU/FP32
arithmetic, and optional amplitude multiplier remain outside this gate.
