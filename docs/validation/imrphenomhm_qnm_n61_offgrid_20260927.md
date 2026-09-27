# IMRPhenomHM N61 six-mode QNM off-grid receipt — 27 September 2026

The [`hm.qnm.n61.offgrid-six-mode/v1` specification](../math/imrphenomhm-qnm-n61-offgrid.md)
was evaluated for the **`n61_main` clean-room refit**, whose coefficient
payload SHA-256 is `afcb21b634efacbc1ed0b44f24fda5df5ab08e236f3b7cbfdc7e71d3452d6525`.
This is the explicitly hypothesized 61-node variant; it is not identified
with historical author training data or coefficients.

On `len`, the independent Cook--Zalutskiy solver generated 18 strict roots:
the six fundamental modes `(2,2,0)`, `(2,1,0)`, `(3,3,0)`, `(3,2,0)`, `(4,4,0)`,
`(4,3,0)` at oriented remnant spins `-0.71`, `0.37`, and `0.79`. The minimum
distance to the hypothesized training grid is `0.00643` in spin. No native
fit coefficient or waveform function entered root generation. The solver
source SHA-256 is `36792fc5ad7201cfd5e4cac5221d48ca0db70f38b9d6fec4c3232caf60f84717`;
the [frozen roots](https://github.com/xangma/diffgw/blob/main/tests/data/hm_qnm_n61_offgrid_roots.json)
have SHA-256 `febbad662efb5209691d3e26fa311eaf03d855510470019b2201d18b7761222a`.
Every root reports the large-`ell` WKB/separation/spin continuation branch,
continued-fraction depth `1024`, maximum estimated frequency error
`1.54×10⁻¹⁵`, and maximum angular truncation error `7.11×10⁻¹⁴`.

The maximum `|omega_N61 - omega_root|/|omega_root|` across the three spins
for each mode was:

| Mode | Max complex relative error | Max real-part relative error | Max damping relative error |
| --- | ---: | ---: | ---: |
| (2,2,0) | 0.000676 | 0.000637 | 0.001423 |
| (2,1,0) | 0.000810 | 0.000503 | 0.002661 |
| (3,3,0) | 0.000468 | 0.000448 | 0.001093 |
| (3,2,0) | 0.000163 | 0.000106 | 0.000739 |
| (4,4,0) | 0.000429 | 0.000412 | 0.001214 |
| (4,3,0) | 0.000106 | 0.000103 | 0.000546 |

All 18 points meet the predeclared `0.01` complex relative-error bound.
Torch and JAX returned identical FP64 fit values at these points to the
displayed precision (maximum complex relative difference `0` in this run).
The public `IMRPhenomHM.mode_qnm_frequencies` total-mass conversion passed
on all six modes in both frontends with supplied `M_f=57.6`, `a=0.37`, and
`M_total=60`. The supplied remnant state isolates the QNM adapter; it is not
a test of the remnant provider.

The CPU FP64 focused test on `len` passed **67/67** in **3.33 s**:

```bash
cd /home/xangma/tmp/diffgw-hm-qnm-offgrid-20260927
env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_imrphenomhm_qnm_n61_offgrid_math_reference.py --reference-device=cpu -ra
```

The root-generation command was `PYTHONPATH=src OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 .../python -u
scripts/generate_hm_qnm_n61_offgrid_reference.py --output
tests/data/hm_qnm_n61_offgrid_roots.json`; its job PID was `3328567` and
the complete log is `hm_qnm_n61_roots.log` in the same remote working
directory. It finished successfully before the focused test. Environment:
Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, Torch 2.14.0+cpu, JAX 0.11.2,
pytest 9.1.1. JAX used `CpuDevice(id=0)`.

This receipt establishes selected off-grid QNM **input** agreement and
Torch/JAX evaluation and scaling for the named N61 artifact. It does not
establish historical London fit provenance, full-spin or near-extremal
accuracy, remnant-fit accuracy, mode assembly, polarizations, or whole-waveform
physical validity.
