# IMRPhenomT dominant inspiral time adapter: len receipt (2026-09-27)

`T-INSP-TIME-ADAPTER-v1` passed its strict bounded gate: **16 passed in
60.15 s**. No native implementation, defaults, or frozen failure list changed.
The [six-section component spec](../math/phenomt-inspiral-time-adapter.md)
was written before the scalar implementation and test.

The run used host `len`, cwd
`/home/xangma/tmp/diffgw-phenomt-fit-inputs-20260927`, Python 3.12.3,
Torch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, and pytest 9.1.1. JAX x64 and
CPU were explicitly enabled. The exact command was:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests \
  /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_phenomt_inspiral_time_adapter_math_reference.py \
  --reference-device=cpu -ra
```

The test PID was `3874494` and has exited. Output is preserved at
`/home/xangma/tmp/diffgw-phenomt-fit-inputs-20260927/phenomt-time-adapter-pytest.log`.
The stop command while it ran was `kill 3874494` on `len`; no further check
is pending. The transferred scalar and test files had matching local/len
SHA-256 values `4e2b67c9afd764c1485e10093014658b6f49ac8d215532dc686fc908b6c03225`
and `e1517db5080c0248bf0b5d429d562f2ca370c052da15bfb09cce9952ee66b489`.

Three source-backed intrinsic points and three inspiral frequencies per point
were independently inverted in the scalar theta domain and compared with
both native backends' time roots and forward frequencies. A separate
integrated-frequency phase check covered evolution, reference-frequency
rotation, and `2*phic`; public checks covered centered and uncentered grids,
the lower-frequency mask, and omitted/zero `f_ref` resolution. Thresholds
were frozen in the spec and unchanged after the first passing run.

This is a component-level agreement receipt, not physical calibration or
full-waveform qualification. The pinned MIT author
[`internals.py`](https://gitlab.com/imrphenom-dev/phenomxpy/-/blob/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt/internals.py)
uses a `t0` shift when inverting before `tEarly`; the native code does not.
The test targets lie after `tEarly` and before the inspiral join. The
[THM paper](https://arxiv.org/pdf/2012.11923v1) specifies the frequency
series but not the public sampling adapter. Pre-`tEarly` inversion,
merger/ringdown, amplitude, auto-duration, HM/precessing adapters, and
physical-error validation remain open.
