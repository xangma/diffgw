# HM public intrinsic phase composition gate — 2026-09-27

The conditional `hm.phase.public-composition.eq3-14.v1` gate passed on `len`.
The independent mpmath composer uses London's three-branch phase map and
continuity offsets with the already checked D-carrier C1 ringdown correction.
It compares the public Torch/JAX unprojected mode's unit phasor with the
source-composed phase, plus left/right one-sided slopes at the low-frequency
phase join, mapped PhenomD ringdown attachment, and mode ringdown join.
The D phase, alpha coefficients and HM/D QNM pairs are supplied by the
frontends; the scalar composer does not independently validate these inputs.

The frozen CPU-FP64 sample covers `(2,1)`, `(3,3)`, `(3,2)`, `(4,4)` and
`(4,3)` for `(35,20,0.2,-0.1)` and `(50,10,0.7,-0.3)` in solar-mass and
dimensionless-spin units, with both `phenomd_runtime` and `london_2018`
transition settings. It fixes `f_ref=None`, `phic=tc=0`, positive
frequencies and `m1>m2`; the latter fixes the signed odd-mode amplitude
factor for unit-phasor comparison. Each combination samples 13 frequencies
and three joins. Phasors use `rtol=atol=2e-9`; one-sided slopes use
`rtol=3e-5`, `atol=3e-4`.

Host: `len`. Cwd: `/home/xangma/tmp/diffgw-hm-phase-composition-20260927`.
Command: `PYTHONPATH=src:tests JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_imrphenomhm_phase_composition_math_reference.py --reference-device=cpu -ra`.
PID: `149502` (completed). Log: `/home/xangma/tmp/diffgw-hm-phase-composition-20260927/focused.log`.
Next check: completed, no scheduled poll. Stop command if still running:
`kill 149502`. Result: **9 passed in 26.57 s**. Environment: Python 3.12.3,
Torch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, NumPy 2.5.3,
pytest 9.1.1.

An adjacent len run of this file, the frequency-map, phase-offset and
ringdown-phase reference gates, HM reconstruction and clean higher-mode
tests, and waveform definitions passed **183 tests in 70.58 s** (one
Torch JIT deprecation warning). Host and cwd are as above; PID `305544`
(completed), log `adjacent.log`, next check complete, stop command
`kill 305544` if still running. Command: `PYTHONPATH=src:tests
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q
tests/test_imrphenomhm_phase_composition_math_reference.py
tests/test_imrphenomhm_phase_offsets_math_reference.py
tests/test_imrphenomhm_ringdown_phase_math_reference.py
tests/test_imrphenomhm_frequency_map_math_reference.py
tests/test_phenomhm_reconstruction.py tests/test_clean_higher_modes.py
tests/test_waveform_definitions.py --reference-device=cpu -ra`.
The waveform-library and coverage generators passed `--check` on len.
The new reference and test files passed Ruff 0.16.9. Strict MkDocs built
successfully in the isolated local checkout using current `mkdocs-material`
and `mkdocstrings-python`; len lacked those packages and could not resolve
its package index during an attempted install. That len job used the cwd above,
command `/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m venv
docs-venv && docs-venv/bin/python -m pip install -q
--upgrade mkdocs-material mkdocstrings-python && docs-venv/bin/python -m
mkdocs build --strict`, PID `332002` (pip child `333554`), log
`docs-build.log`; it was stopped after the DNS failure (`kill 332002 333554`),
with no further poll scheduled.

This is a phase-composition implementation gate, not a whole-mode or
whole-family accuracy result. D phase and alpha fit provenance, remnant/QNM
accuracy, amplitude sign and zero cases, nonzero reference/time gauges,
negative-frequency conventions, external binary agreement and NR physical
accuracy remain separate. See the [component specification](../math/imrphenomhm-phase-composition.md).
