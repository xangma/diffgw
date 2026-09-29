# IMRPhenomHM complex intrinsic-mode assembly — 2026-09-27

The conditional `hm.intrinsic.complex-assembly.eq3.v1` gate passed on `len`.
The standalone SymPy/mpmath reference composes London's amplitude and phase
equations with the previously specified frequency map, PN ratio, and phase
offset components. Its public adapter accounts for the native Fourier sign,
odd-mode sign, and nonzero orbital-phase and time gauges. The test compares
full complex unprojected modes, including their amplitude scale, with both
Torch and JAX. It does not compare polarizations or claim NR accuracy.

The CPU-FP64 fixture uses masses `(35,20)` solar masses and aligned spins
`(0.2,-0.1)`, all six positive-frequency HM modes, and both
`london_2018` and `phenomd_runtime` transition selections. Each mode uses
11 frequencies spanning the low-frequency amplitude and phase joins and the
mode ringdown join, including one point on each side. It fixes
`f_ref=None`, `phic=0.37`, and `tc=0.008` seconds, and explicitly selects
the Mishra-source 1.5PN variants for `(2,1)`, `(3,2)`, and `(4,3)`.
The test checks each complex sample against its own nonzero expected
amplitude, with maximum relative error required below `3e-8`; this is
264 frontend samples. Scalar checks separately verify the symbolic Eq. (3),
the paper's positive-exponent convention, the exact symmetric odd-mode
undefined ratio, and the public symmetry-zero/time-gauge contract.

Focused run: host `len`; cwd
`/tmp/diffgw-hm-intrinsic-20260927/repo`; command
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python -m pytest -q
tests/test_imrphenomhm_intrinsic_modes_math_reference.py
--reference-device=cpu -ra`; result **6 passed in 21.49 s**. The foreground
command completed, with no running PID or log to poll. Environment: Python
3.12.3, Torch 2.14.0+cu130, JAX 0.11.2, SymPy 1.14.0, mpmath 1.3.0,
NumPy 2.3.5, SciPy 1.16.3, pytest 9.1.1; both frontends ran on CPU.

The adjacent HM math-reference suite passed **289 tests in 155.34 s** on
`len` from the same cwd and environment. Command: `JAX_ENABLE_X64=1
JAX_PLATFORMS=cpu PYTHONPATH=src
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python -m pytest -q
tests/test_imrphenomhm_*math_reference.py --reference-device=cpu -ra`.
PID `2392236` has completed; log:
`/tmp/diffgw-hm-intrinsic-20260927/all-hm.log`; exit-status file:
`/tmp/diffgw-hm-intrinsic-20260927/all-hm.exit` contains `0`.
The next check is complete and there is no scheduled poll. If that job were
still running, its stop command would be `ssh len 'kill 2392236'`.
The new source and test pass Ruff, and the docs pass `mkdocs build --strict`.

The D carrier amplitude and phase, alpha fit coefficients, remnant and QNM
frequencies come from the public frontends and are **supplied inputs** to the
scalar reference. This gate can catch a wrong composition of those inputs,
but shared upstream errors remain possible. It does not cover another
mass/spin region, a finite `f_ref`, negative-frequency reconstruction,
polarization projection, binary-library agreement, or physical mismatch
against NR. See the [mathematical specification](../math/imrphenomhm-intrinsic-modes.md).
