# Pass 11: higher-mode normalization and `(3,2)` mixing

## Source-audited decision

The initially proposed multiplication of the XHM `(3,2)` carrier probe by
`N = sqrt(16*pi/5)` was rejected by the source algebra before any oracle or
parity testing. It would double-count the normalization already applied by
the production reconstruction. The frozen waveform is therefore unchanged by
this pass.

García-Quirós et al. (arXiv:2001.10914v2) define the paper's normalized
quantity as

```text
A^0_22 = pi sqrt(2 eta / 3) (pi f)^(-7/6),                 Eq. (4.5)
H_lm   = |A^SPA_lm| / A^0_22,                              Eq. (4.6)
```

Their Appendix E Eq. (E1) gives the physical spherical mode normalization

```text
h_lm = A_lm exp(-i m phi),
A_lm = 2 eta x sqrt(16*pi/5) hhat_lm.
```

At the mixing probe distance `D=M`, the native XAS leading amplitude is

```text
a0_native = sqrt(5 eta / 24) * pi^(-2/3) * M^(-1/6) * f^(-7/6).
```

Direct simplification of Eq. (4.5) gives
`A^0_22 = N * a0_native`, with `N = sqrt(16*pi/5)`. The physical face-on
quadrupole carrier is likewise `h22_mode = N * h22_native`. Thus the ratio
fed to Eq. (A3) is

```text
H_22 = h22_mode / A^0_22
      = (N * h22_native) / (N * a0_native)
      = h22_native / a0_native.
```

`_carrier_h22_ratio` now documents and implements this cancellation. The
production reconstruction retains its existing single `N*a0*h_ratio`
conversion, and the existing `_final_spin_override` forwarding remains in
the carrier, derivative, and mixing probes.

The `(3,2)` spherical/spheroidal conversion remains the two-mode elimination
of García-Quirós et al. Appendix A Eq. (A3):

```text
h_32 = (alpha_322 / alpha_222) h_22
      + (alpha_332 - alpha_322 alpha_232 / alpha_222) h_32^S.
```

## Source and local asset hashes

Primary sources:

* García-Quirós et al., “Multimode frequency-domain model for the
  gravitational wave signal from non-precessing black-hole binaries:
  IMRPhenomXHM,” [arXiv:2001.10914](https://arxiv.org/abs/2001.10914),
  v2, Eqs. (4.5), (4.6), (E1)–(E3), and Appendix A Eq. (A3).
* London et al., “First higher-multipole model of gravitational waves from
  nonprecessing black-hole binaries,” [arXiv:1708.00404](https://arxiv.org/abs/1708.00404),
  v4, Eqs. (3)–(5), for the inherited HM frequency mapping.

The exact public paper assets used for the source audit are retained locally:

```text
/tmp/torchwave-paper-assets/garcia-quiros-2020-imrphenomxhm-v2.pdf
00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee  (SHA256)

/tmp/torchwave-paper-assets/london-2018-phenomhm-v4.pdf
079521aebaccec86c4079901c7f5e56b2ed4712cf79320327d903f46d565a8d7  (SHA256)
```

Working-tree hashes for the owned source/test files at handoff are:

```text
3f733034030004aa964ff19aadb11800f893f5bf3efcaa87c8b1c317bd716dcc  src/torchwave/approximants/imrphenom_xhm.py
cfe44564b1f5ca8546f3194ee0ca1166f96375945f90857c5e7aa99115ce4e42  src/torchwave/approximants/imrphenom_hm.py
6704f0c59a1e1527f5c31063d1bac8ab84f7349207acc3f1c24532c370167548  src/torchwave/approximants/xhm_fits.py
db7d5e07135ca044918dbeee92385703017352bc97aaefb27dfdf66475aef6c3  tests/test_pass11_higher_modes.py
```

## Independent checks and limits

`tests/test_pass11_higher_modes.py` checks all of the following:

* an actual XAS face-on carrier against the paper `A^0_22` expression,
  showing exact cancellation of `N`;
* the production XHM `(3,2)` call, by capturing the normalized carrier value
  passed into the mixing function;
* exact reconstruction of an arbitrary complex Appendix A matrix; and
* the published mixing-fit zero-spin limit.

Focused run:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_pass11_higher_modes.py
4 passed
```

The bounded issue remains subject to the published two-mode truncation in
Appendix A; couplings with `l > 3` are omitted. No fitted-target convention,
new calibration data, NR data, LAL/PyCBC implementation, binary oracle,
parity harness, or residual report was used.
