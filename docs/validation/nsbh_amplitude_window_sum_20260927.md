# NSBH conditional amplitude window-and-sum gate — 27 September 2026

The [six-section v1 specification](../math/nsbh-amplitude-window-sum.md)
was frozen in commit `be85d40` before native comparison. It transcribes
[Thompson et al. (2020), Eqs. (4) and (30)](https://arxiv.org/html/2002.08383v2)
with independently supplied centers, width, total mass, and three carrier
amplitudes. The scalar reference uses mpmath and imports neither native
backend nor LAL code. The native helpers are called by the public Torch and
JAX amplitude assembly. No fit coefficients or historical default changed.

On local Darwin arm64 with Python 3.13.7, Torch 2.14.0, JAX 0.11.2,
NumPy 2.5.3, mpmath 1.3.0, and pytest 9.1.1, the command
`JAX_ENABLE_X64=1 PYTHONPATH=src /Users/xangma/repos/diffgw/.venv/bin/python
-m pytest -q -s tests/test_nsbh_amplitude_window_sum_math_reference.py
tests/test_nsbh_amplitude_variant_provenance_math_reference.py
tests/test_nsbh_joining_frequencies_math_reference.py
tests/test_nsbh_branch_amplitude_fits_math_reference.py
tests/test_nsbh_clean_amplitude.py tests/test_clean_room_nsbh.py
tests/parity/test_nsbh_parity.py` passed **102/102** in 16.84 s. The 14 new
tests cover four frozen center/width patterns, 70-versus-90-digit scalar
stability, window centers and neighborhoods, Torch/JAX values and selected
frequency/carrier derivatives, and two fixed pre-refactor public amplitude
sample grids. The scalar difference stayed below its frozen `1e-30`
absolute limit. The maximum native value error was `2.220e-16` and maximum
selected partial error was `1.110e-16`, below `2e-13` and `2e-12`
respectively. The fixed public amplitude samples passed at `2e-13`
absolute after scaling by `1e22`.

The adjacent NSBH phase and reconstruction suite passed **31/31** in
3.83 s. The documentation generator's `--check` and Ruff on all changed
Python files passed. An additional local strict LAL-binary suite produced
**33 passed, 22 failed**: this Mac environment lacks PyCBC, so the harness
returns `optimized_match=None` for each strict comparison, which its
existing strict gate rejects. This is an environment limitation of that
suite, not a changed tolerance or evidence for a waveform discrepancy.
On `len` with CPU FP64, the new gate, its NSBH amplitude neighbors, and
the HM mapping reference passed **142/142** in 72.03 s from the integrated
checkout at `6ad7663`. The command used `JAX_ENABLE_X64=1`,
`JAX_PLATFORMS=cpu`, and `PYTHONPATH=src:tests`, with
`--reference-device=cpu -ra`; its test files were
`test_nsbh_amplitude_window_sum_math_reference.py`,
`test_nsbh_amplitude_variant_provenance_math_reference.py`,
`test_nsbh_joining_frequencies_math_reference.py`,
`test_nsbh_branch_amplitude_fits_math_reference.py`,
`test_nsbh_clean_amplitude.py`, `test_clean_room_nsbh.py`,
`parity/test_nsbh_parity.py`, and
`test_imrphenomhm_frequency_map_math_reference.py` under `tests/`.

The strict public-binary NSBH remnant-mass regression was then rerun on
`len` from that same checkout with PyCBC 2.11.0 installed in a separate
package directory (NumPy 2.3.5, SciPy 1.16.3), without changing the shared
Torch/JAX environment. All **24/24** tests passed in 6.87 s, including the
22 cases that could not obtain a match statistic on the Mac without PyCBC.
The command was `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests
OMP_NUM_THREADS=1 python -m pytest -q
tests/test_nsbh_phase_mass_regression.py -ra`, using the existing len venv's
Python. This resolves the environment limitation above; it does not expand
the conditional amplitude-window gate into whole-waveform qualification.

This gate verifies only the three-window algebra with supplied inputs. The
[low/no-torus joining-center discrepancy](../math/nsbh-joining-frequencies.md),
[Eq. (26)/(29) variant discrepancies](../math/nsbh-amplitude-variant-provenance.md),
upstream PhenomC carriers and distance normalization, and the complete
NSBH waveform remain unqualified.
