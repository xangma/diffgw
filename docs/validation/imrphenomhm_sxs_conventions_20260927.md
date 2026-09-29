# IMRPhenomHM–SXS convention bridge — 2026-09-27

The `hm.conventions.sxs-bridge.v1` source and tests establish the Fourier
mode-pair algebra, the SXS positive-carrier target, and declared orbital,
time, common-sign, and finite-reference gauge effects. They do not select an
NR-derived phase factor or change a production waveform.

The independent SymPy/mpmath reference checks
$N_{\ell m}(f)=(-1)^\ell\overline{\mathcal F_-[h_{\ell m}](-f)}$ and
$P_{\ell m}(f)=(-1)^\ell N_{\ell m}(f)$. A sampled complex chirp checks
both Fourier signs, time translation, physical orbital rotation, and even/odd
$\ell$ pair signs by direct numerical integration. The doubled relative
phase check proves cancellation of common time and orbital shifts and the
explicit effects of a common strain sign and the existing finite-reference
adapter. Sources and derivation are in the
[component specification](../math/imrphenomhm-sxs-conventions.md).

The conditional public gate compares all six full complex unprojected HM
modes with the scalar assembly on Torch and JAX CPU FP64. It uses
$(m_1,m_2)=(35,20)M_\odot$, $(\chi_1,\chi_2)=(0.2,-0.1)$,
$\phi_c=0.37$, $t_c=0.008$ s, both `london_2018` and
`phenomd_runtime` transition variants, 11 frequencies around the map and
ringdown joins per mode, and $f_\mathrm{ref}=\mathrm{None}$ or $40$ Hz.
The maximum permitted full-complex relative error at each nonzero sample is
`3e-8`. D amplitude/phase, fit and QNM inputs, and the D peak slope for the
finite gauge are supplied by the frontend. A shared upstream error could
therefore pass this conditional check.

The check ran on host `len`, cwd
`/tmp/diffgw-hm-conventions-20260927/repo`, with Python 3.12.3,
Torch 2.14.0+cpu, JAX 0.11.2, NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0,
mpmath 1.3.0, and pytest 9.1.1. The maintained commands are:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src \
  /tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python -m pytest -q \
  tests/test_imrphenomhm_sxs_conventions_math_reference.py \
  tests/test_imrphenomhm_intrinsic_modes_math_reference.py \
  --reference-device=cpu -ra

JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src \
  /tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python -m pytest -q \
  tests/test_imrphenomhm_*math_reference.py --reference-device=cpu -ra
```

The focused final-snapshot run passed **17/17 tests in 31.87 s**. The
adjacent HM math-reference suite passed **300/300 in 164.11 s**. That suite
started just before one extra physical-rotation assertion was added to the
synthetic convention test; the focused final-snapshot run covers the added
assertion. The full suite ran via
`/tmp/diffgw-hm-conventions-20260927/run-all.sh`, PID `374923`, and completed
with exit code `0`. Its log and exit marker are
`/tmp/diffgw-hm-conventions-20260927/all-hm.log` and `all-hm.exit`.
There is no pending check or running job. If it were still running, the stop
command would be `ssh len 'kill 374923'`. Ruff passed on all changed Python
files, `mkdocs build --strict` passed, and `git diff --check` passed.

This component gate leaves the frozen
[2265](imrphenomhm_sxs_2265_receipt_20260927.md),
[3631](imrphenomhm_sxs_3631_receipt_20260927.md), and
[3630 projection](imrphenomhm_sxs_3630_projection_receipt_20260927.md)
physical holdouts red. The earlier diagnostic $-e^{i\pi/4}$ is still an
unproved HM–SXS conversion. No acceptance threshold, data, or status in
those receipts changed.
