# Clean TEOB alternating post-adiabatic verification

This verification covers the bounded scalar, nonspinning, zero-tide vacuum
post-adiabatic component specified in
`docs/reviews/2026-09-13-parity-review/teob-postadiabatic-contract.md`.
The implementation uses the existing Torch TEOB Hamiltonian and reviewed
vacuum radiation helper as dependencies. It does not route a waveform,
calibrate a model, or continue through merger.

Physical ``f_start`` inputs require an explicit total mass in solar masses or
both component masses; the reusable PA API does not infer a catalog mass.

## Checks

The new unit tests in `tests/test_teob_postadiabatic_clean.py` cover:

- quartic derivative exactness on descending grids, including both boundary
  stencils and vectorized radial values;
- four-node cubic interval and cumulative quadrature exactness, including
  descending orientation;
- independent radial and angular balance diagnostics;
- PA order 2/4/8 refinement and radial-grid refinement;
- finite gradients through the intrinsic symmetric mass ratio and caller time
  and phase offsets;
- physical start-frequency mapping, explicit short-grid skipping, endpoint
  inheritance, and invalid input/failure handling.

The test command completed with 11 passing tests:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_teob_postadiabatic_clean.py
11 passed in 1.09s
```

## Bounded probe

The diagnostic command was run on host `RONI2` in
`/Users/xangma/repos/torchwave`:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/diagnose_teob_postadiabatic_clean.py \
  --output docs/reviews/2026-09-13-parity-review/verification/teob-postadiabatic-clean/diagnostic.json
```

The probe used `nu=0.24`, zero aligned spins, zero tides, `r0=20`, and
`r_min=14`. All runs completed with finite inward profiles. Results are in
`diagnostic.json`:

| PA order | points | elapsed (s) | endpoint time | endpoint phase | max relative angular residual |
|---:|---:|---:|---:|---:|---:|
| 2 | 61 | 0.0374 | 9606.1659 | 133.7478837 | 8.00e-4 |
| 4 | 61 | 0.0384 | 9608.5600 | 133.7845041 | 4.69e-6 |
| 6 | 61 | 0.0530 | 9608.5537 | 133.7843999 | 4.80e-8 |
| 8 | 61 | 0.0667 | 9608.5537 | 133.7844007 | 2.81e-9 |

For PA order 8, changing `dr` from `0.2` (31 points) to `0.1` (61 points)
changed the endpoint time by `3.45e-05` and phase by `4.94e-07`. The largest
absolute radial balance residual in these runs was `1.96e-18`.

These numbers are bounded numerical and refinement diagnostics; they are not
a waveform or external-library parity claim.
