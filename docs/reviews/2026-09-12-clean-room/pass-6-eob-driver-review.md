# Independent published EOB driver review

Review date: 2026-09-12. Scope: `eob/published_waveform.py`, the updated
`integrate_to_frequency_peak(post_peak_duration=...)`, their immediate adapters,
and the `get_fd_waveform('SEOBNRv4Published')` dispatch. Source inspection and
short independent CPU checks only; no external implementation, oracle output,
fitting, or repeated long integration was used. Only this report was written
by the reviewer.

**Result:** one public-wrapper batch defect was found, fixed by the author,
and verified. No actionable defect remains in the reviewed snapshot. This
assesses the stated experimental reconstruction, not SEOBNRv4 implementation
parity or the unresolved publication-level choices.

## Resolved finding

The final mask in `get_fd_waveform` flattened a batched frequency grid when
the output had additional parameter axes. With masses shaped `(2, 1)` and
`(1, 3)`, and frequencies shaped `(3, 4)`, the model correctly returned
`(2, 3, 4)`, but the public wrapper reshaped the grid to `(1, 12)` and raised
a dimension-mismatch error. This was reproduced with a synthetic reconstruction,
without integrating an orbit. The author now preserves frequency dimensions,
converts the mask grid to the output device/dtype, and uses ordinary broadcasting.
The added Tensor/list regression verifies both dimensions and cutoffs.

## Mathematical and interface checks

- For the stated mode convention `h22=A exp(-i Phi22)`, the directional
  derivative is `grad(Phi22) dot F`, where `F` includes radiation reaction.
  The local shift differentiates the full continuous orbital, tail, and
  residual phase. It does not replace the result by `2 Omega`. The outer
  graph retains derivatives of both the phase and the vector field.
- Parameter axes precede the sample axis in Hamiltonian/mode evaluation.
  The trajectory conversion `(N, ..., 4) -> (..., N, 4)` and validity-mask
  conversion preserve those axes. All five unresolved choices remain required.
- With `M_sec=G M_initial/c^3`, the start is `M_initial Omega=pi M_sec f_lower`.
  The author QNM supplies `M_final omega`; division by `M_final/M_initial`
  therefore gives the initial-mass frequency required by merger attachment.
  Physical mode amplitude is multiplied by `M_sec c / D`, and the Fourier
  integration measure is `dt_geom M_sec`.
- Pairing spherical `m=+/-2` gives
  `hplus=sqrt(5/(16 pi)) (1+cos(i)^2) Re(h22)` and
  `hcross=-sqrt(5/(4 pi)) cos(i) Im(h22)`.
  The phase rotation `exp(-2i phic)` and Fourier shift
  `exp[-2 pi i f (tc-t_match M_sec)]` agree with the documented negative-sign
  Fourier transform and put the imposed NQC amplitude peak at `tc`.
- The extra duration is `max(Delta_t22_peak,0)+dt_geom`. Combined with the
  integrator's two post-maximum samples, it covers sub-sample peak refinement
  and the two samples after the nearest matching sample. Completed rows remain
  frozen and are excluded by `valid`. Each output row is zero after its own
  requested ringdown duration; no taper or extrapolated peak is introduced.

These contracts are consistent with the primary-equation component reviews:
[phase](pass-6-eob-phase22.md), [remnant](pass-6-eob-remnant.md), and
[attachment](pass-6-eob-attachment-review.md). In particular, the continuous
phase is the declared reading of Taracchini/Pan's factorized mode, and the
matching time remains Bohé's `t_Omega_peak + Delta_t22_peak`.

## Independent verification

- Six distinct binaries on a `(2, 3)` parameter batch, with two state samples
  each: continuous-phase central directional differences at steps
  `1e-2, 1e-3, 1e-4` agreed within `3.1e-12` absolute. Every scalar row agreed
  with its batch result at `2e-13` tolerance. State, mass-ratio, and both spin
  gradients were finite. Locally created tensors under `no_grad` and
  `inference_mode` reproduced the same detached outputs.
- Independent `gradcheck` passed for mode real/imaginary parts and frequency
  with respect to state, mass ratio, and both spins; a separate check with
  parameter gradients alone also passed. This used the alternative explicit
  `bohe_plus_pan`, positive-chi, and generic-leading-ell8 selections.
- An analytic six-row flow with distinct peaks and positive offsets through
  `3.2 M` retained the entire five-point matching stencil in every row.
  Peak-time errors were below `2.5e-8 M`; the minimum extra stencil margin was
  one sample. This specifically checks the newly added post-peak duration.
- A direct finite Fourier sum at exact FFT bins, using physical sample times,
  tested `(2, 3)` mass/extrinsic batches and row-dependent frequency grids.
  Relative errors were below `9.1e-15` for both polarizations. This jointly
  checks projection normalization, physical amplitude/time units, `phic`, and
  the sign and origin of the coalescence-time shift.
- A separate synthetic reconstruction-boundary probe checked the QNM mass
  conversion, state/mask axis moves, requested post-peak duration, and per-row
  ringdown truncation. These probes validate wiring, not orbit accuracy.
- Focused author tests rerun after the mask repair:
  `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest tests/test_clean_room_eob_published_driver.py tests/test_clean_room_eob_events.py -q`
  returned **21 passed in 1.24 s**.

The author's separate full reconstruction convergence record was inspected,
not rerun: [result.json](verification/pass-6/eob-convergence/result.json).
It reports two binaries at `dt=0.5,0.25,0.125 M`, finite outputs, unchanged
source hashes, and decreasing successive complex-mode L2 differences. This
limited evidence does not establish convergence across the full parameter
space. Float32 conditioning, GPU execution, long-trajectory derivatives, and
derivatives across discrete event/stencil/sample-count changes are not newly
certified by this review. The public contract retains the signed-spin QNM
domain restriction and the need for adequate pre-peak history.

## Reviewed snapshot

SHA-256:

| File | Digest |
| --- | --- |
| `eob/published_waveform.py` | `306bcedb98abee3241b4c4b386cd8e17c104834fbf28eb8bd11386fc3d868ffd` |
| `eob/integrator.py` | `60657589b70a3dc6ae75abbb26093e65cae63f2de9fb2a5b0110c07c15d88879` |
| `eob/waveform.py` | `03a559c59f1aef83037ba0fdea61cc070c1afa4478f082d4068b67fe1fc27ced` |
| `waveforms.py` | `8a00dfe83fc46a257d77d5055356c409412da248c6a602f57bcd798f25f550ac` |
