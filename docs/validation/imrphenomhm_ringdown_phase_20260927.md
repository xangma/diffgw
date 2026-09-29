# HM ringdown phase-shape gate — 2026-09-27

The bounded `hm.phase.ringdown.eq9-c1.v1` gate passed on `len`, in `/home/xangma/tmp/diffgw-hm-eq14-phase-20260927`, using a copy of the isolated worktree based on `9cbb2a2`. The maintained test is `tests/test_imrphenomhm_ringdown_phase_math_reference.py`; it calls the production Torch/JAX helpers against the independent mpmath scalar.

Command: `PYTHONPATH=$PWD/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_imrphenomhm_ringdown_phase_math_reference.py --reference-device=cpu -ra`

Result: **14 passed in 4.13 s**. A final combined run adding `tests/test_waveform_definitions.py` passed **26 tests in 8.51 s**. Environment: Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, pytest 9.1.1. The gate covers the paper limit, distinct native D/HM QNM input convention, values, frequency and $\alpha_4$ gradients, C1 join, and 22 control. The `--require-math-reference` pytest option belongs to the full CI suite: its focused-file invocation exits at collection because the other required math-reference files are absent from that invocation; it does not indicate a test failure.

No whole-model accuracy claim follows. QNM and alpha fits, gauge/attachment beyond this correction, and LAL/NR comparisons remain separate gates. See the [component specification](../math/imrphenomhm-ringdown-phase.md) for the source and precise boundary.
