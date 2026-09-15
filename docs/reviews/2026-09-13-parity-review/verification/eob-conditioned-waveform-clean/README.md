# Clean SEOBNRv4 TD-to-FD adapter verification

This verification bundle covers the bounded clean-room adapter in
`src/torchwave/approximants/eob/conditioned_waveform_clean.py`.  The adapter
uses the reviewed interface contract in
[`td-fd-conditioning-contract.md`](../../td-fd-conditioning-contract.md):

1. `fft_length_and_sampling(f_max, delta_f)` fixes the common one-sided grid
   and its physical `dt`.
2. `plan_td_support` supplies the scalar TD start frequency.  The native
   adaptive generator receives `omega_start = pi * M_seconds * f_start` and
   `dt_geom = dt / M_seconds`.
3. The independent mode pair is projected with the spin-weighted `l=2,
   |m|=2` coefficients, scaled to physical distance, and conditioned by
   `condition_td_to_fd`.
4. `tc` is the only FD phase shift.  The generated clock's location relative
   to the continuous peak is returned as `physical_epoch` metadata.

The reviewed adaptive choices are fixed to:

```text
spin_truncation       = generic-leading-ell8
spin_v6               = bohe_total_energy
include_pan_v8        = False
include_positive_chi_540 = False
final_spin_fit_order  = (3, 4)
ringdown_duration     = 150 M (default)
```

`phic` rotates the generated modes as `h22 -> exp(-2 i phic) h22` and
`h2m2 -> exp(+2 i phic) h2m2`.  No `fref` phase reanchoring is defined; the
adapter rejects `fref` explicitly.  The physical epoch is metadata and is not
multiplied into the transform.

The focused tests inject a short differentiable TD generator and verify the
start frequency, sample spacing, fixed grid, reviewed choices, two-mode
projection, distance and phase semantics, epoch metadata, batching, dtype,
device, gradients, and invalid-input behavior.  They do not invoke an
external implementation or compare against an oracle.

Run them with the repository's clean environment:

```text
PYTHONPATH=src:tests:scripts:. \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
-m pytest -q tests/test_eob_conditioned_waveform_clean.py
```

The native cost probe is intentionally separate from this synthetic suite.
The script prints the proposed high-mass/high-start configuration without
running it; pass `--run` only after that probe is requested:

```text
PYTHONPATH=src:tests:scripts:. \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
scripts/diagnose_eob_conditioned_waveform_clean.py
```

The implementation is MIT clean-room work.  Equation-level provenance is
limited to public references: [Berti, Cardoso and Will,
gr-qc/0512160](https://arxiv.org/abs/gr-qc/0512160) for the fundamental-mode
damping fit and [Tichy and Marronetti,
0807.2985](https://arxiv.org/abs/0807.2985) for the aligned-spin remnant
estimate.  The projection convention follows the standard spin-weighted
spherical-harmonic decomposition; the numerical conditioning and sampling
choices are those recorded in the local contract above.

