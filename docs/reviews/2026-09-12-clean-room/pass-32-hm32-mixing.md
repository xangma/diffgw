# Pass 32: IMRPhenomHM (3,2) mixing audit

## Scope and clean-room boundary

This pass isolates the strict `IMRPhenomHM` residual to the proposed
spheroidal/spherical mode-mixing path.  The implementation was checked against
the published model description and its equations.  No waveform
implementation source was read or copied.  Public LAL executable/API output
was used only as a black-box comparison; no runtime coefficient was fitted to
that output.

## Source-backed finding

The published PhenomHM construction starts from a spherical-harmonic
decomposition and maps the calibrated `(2,2)` PhenomD mode to higher modes.
The paper explicitly lists spherical/spheroidal mixing between the
`|m|=l` and `|m|=l-1` modes as a physical effect absent from PhenomHM, and
identifies that omission as important for `(3,2)` and `(4,3)` accuracy:

* London et al., *First higher-multipole model of gravitational waves from
  spinning and coalescing black-hole binaries*, arXiv:1708.00404, Secs. II–III,
  especially the discussion around Eqs. (3)–(9) and the model limitations:
  [arXiv PDF](https://arxiv.org/abs/1708.00404).

TorchWave follows that published architecture.  In
[`imrphenom_hm.py`](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_hm.py:65),
`_swsh_projection_coefficients` computes the spin-weighted **spherical**
harmonic projection.  The mode loop then projects each mapped `(l,m)` directly
at [`imrphenom_hm.py`](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_hm.py:698);
there is no spheroidal-to-spherical mixing term.  This is therefore a known
model limitation, not a uniquely determined implementation defect.  Adding a
mixing correction would define a different model and would require an
independent published prescription and coefficients that are not part of the
PhenomHM source description.

## Black-box narrowing result

For the strict diagnostic configuration

```text
m1=40, m2=15, chi1=+0.3, chi2=-0.2, f_ref=20 Hz,
distance=500 Mpc, inclination=0.8 rad, phase=0.2 rad,
grid=20..512 Hz, df=0.25 Hz, approximant=IMRPhenomHM
```

the parity harness against the public LAL binary reports:

```text
plus complex relative L2       4.540949061216069e-03
plus flat mismatch             1.0105917114033858e-05
cross complex relative L2      3.638032292787460e-03
cross flat mismatch            6.542809111276249e-06
```

The strict gates are `1e-6` for complex relative L2 and `1e-8` for mismatch,
so the residual is real and remains outside the gate.  A direct public
`SimInspiralChooseFDModes` comparison for the same masses/spins and grid,
with the positive-frequency negative-`m` branch selected to match the Fourier
convention, gives a `(3,2)` mode/carrier relative L2 of approximately
`1.0542385181811884e-01` (median magnitude ratio `0.913376692`, median phase
offset `-7.66e-03` rad).  This quantifies the external binary difference; it
does not identify a source-backed correction and was not used to fit one.

## Disposition

No runtime change is made in this pass.  The narrowed blocker is the absence
of a published, independently specified spheroidal-mixing prescription for
the PhenomHM model itself.  The strict `(3,2)` residual should be tracked as
model-version/API behavior until such a prescription or authoritative asset
is available.

## Verification

The focused suite passed:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_phenomhm_reconstruction.py \
  tests/test_clean_higher_modes.py tests/test_imrphenom_hm.py
50 passed in 8.31s
```

`git diff --check` and bytecode compilation of `imrphenom_hm.py` also passed.
No deferred NR job was started.
