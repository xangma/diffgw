# ppE supported equations v2 evidence

Date: 2026-09-29. Host: `len`. Remote test copy: `/home/xangma/.cache/diffgw-ppe-qualification-20260929`. Source checkout: `codex/ppe-equation-qualification` at the commit containing this receipt. Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2 with x64, SymPy 1.14.0, mpmath 1.3.0, NumPy 2.5.3, pytest 9.1.1.

The independent mathematical source is `src/diffgw_reference/ppe_equations.py`, specification `ppe-supported-equations/v2`. It records versioned expressions and citations for inspiral amplitude and phase, PN exponent conversion, massive-graviton dispersion, the two cosmological integrals, direct and luminosity-distance-derived $D_0$, and the project's negative-Fourier-sign carrier multiplier. Tests evaluate the symbolic expression with 70-digit mpmath and independent quadrature. Neither SymPy map nor scalar oracle imports a native frontend.

## Focused result

From the remote test copy, with `PYTHONPATH=src:tests`:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -ra --reference-device=cpu tests/test_ppe_supported_equations_math_reference.py tests/test_ppe_taylorf2_carrier_math_reference.py tests/test_ppe_amplitude_math_reference.py tests/test_ppe_math_reference.py tests/test_ppe.py
```

Result: **34 passed in 27.23 s**, exit 0. The newly added file alone passed **10/10**. It checks symbolic identities and limits; TaylorF2, IMRPhenomD and IMRPhenomXAS carrier selectors; both polarizations; supplied and derived $D_0$; zero-modifier GR limits; per-waveform batches with both $B=N$ and $B\ne N$; Torch autograd and `compile` with the eager backend; and JAX JIT and gradients. It does not compare absolute carrier strain with an external implementation.

Before the batch alignment edit, the actual `B=2,N=3` wrapper call failed on both frontends at phase and dispersion broadcasting. After the edit, batched ratios match independently evaluated per-waveform scalars in both shape cases. The first new XAS GR-ratio assertion found an approximately $3\times10^{-12}$ complex-division residual in otherwise matching small strains; the ratio tolerance is now $5\times10^{-12}$ absolute for this GR-limit check. The modified non-GR ratio gate remains $5\times10^{-12}$ relative plus $5\times10^{-13}$ absolute.

## Boundary

The cited ppE inspiral expression does not uniquely fix merger/ringdown branches and transitions. The current adapter evaluates its multiplier at every carrier frequency, but only its inspiral algebra and the stated carrier composition are qualified. Public LALInference and ppE Bilby interfaces expose different option sets; no external waveform parity, generic full-IMR ppE model, or astrophysical accuracy is asserted. See [the math page](../math/ppe-phase-dispersion.md) for the option-specific source map and citations.
