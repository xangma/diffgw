# NSBH conditional polarization composition execution receipt

Specification: [nsbh.polarization-composition/v1](../math/nsbh-polarization-composition.md).
Host: `len` (Linux x86_64 CPU). Working directory:
`/home/xangma/tmp/diffgw-nsbh-qualification-20260929`, an isolated synced
copy of branch `codex/nsbh-qualification`. Runtime:
`/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python`.
Torch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, pytest 9.1.1;
`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, `OMP_NUM_THREADS=1`.

Focused command:

```sh
PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  /home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python \
  -m pytest -q -s tests/test_nsbh_polarization_composition_math_reference.py
```

Result: **16 passed in 13.76 s**. The maximum absolute complex error for
both polarizations over four inclinations and four frequencies was
$2.987\times10^{-15}$ (Torch) and $2.987\times10^{-15}$ (JAX), below the
predeclared $2\times10^{-13}$ limit. Both public-path polarization ratio
checks passed at each inclination, including edge-on and face-off.

The pre-change focused NSBH baseline on the same host and isolated copy
passed **127/127** in 65.71 s. It included amplitude variant and window
references, explicit phase, remnant guard, clean-room, parity, and strict
binary regression tests. The post-change adjacent run added this gate to
the same set and passed **143/143** in 71.43 s:

```sh
PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  /home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python \
  -m pytest -q -ra \
  tests/test_nsbh_amplitude_variant_provenance_math_reference.py \
  tests/test_nsbh_amplitude_window_sum_math_reference.py \
  tests/test_nsbh_explicit_phase_math_reference.py \
  tests/test_nsbh_remnant_guard_source_boundary.py \
  tests/test_clean_room_nsbh.py tests/parity/test_nsbh_parity.py \
  tests/test_nsbh_phase_mass_regression.py \
  tests/test_nsbh_polarization_composition_math_reference.py
```

This is a conditional assembly result. It does not resolve the printed/native
amplitude variants, remnant guard provenance, scale, or whole-waveform claim.
