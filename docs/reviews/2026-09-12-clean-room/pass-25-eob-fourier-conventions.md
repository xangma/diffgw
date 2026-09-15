# Pass 25: SEOBNRv4 phase and Fourier conventions

Date: 2026-09-13

This pass audited the sufficient-horizon `SEOBNRv4Published` waveform path from
the published time-domain (h_{22}) mode through polarization projection and
finite Fourier sampling. The audit used the primary SEOBNRv4 paper
([Bohé et al., arXiv:1611.03703](https://arxiv.org/abs/1611.03703)) and the
factorized-mode convention in
([Damour, Iyer & Nagar, arXiv:0811.2069](https://arxiv.org/abs/0811.2069)).
Public binary output was used only as a validation oracle.

## Convention checks

The factorized waveform writes the Newtonian quadrupole with the scalar
spherical harmonic (Y^{2,-2}(pi/2,Phi)), so the 22 mode has the orbital
factor (e^{-2 iPhi}). The residual phase is a multiplicative
(e^{idelta_{22}}); with TorchWave's mode representation
(h_{22}=Aexp[-iphi_{22}]), this gives the implemented

\[
\phi_{22}=2\Phi-\phi_{\rm tail}-\delta_{22}+\pi.
\]

The (+\pi) is the negative Newtonian prefactor, rather than a fitted phase
offset. Differentiating the complex mode gives the positive GW frequency

\[
\omega_{22}=-\operatorname{Im}(\dot h_{22}/h_{22}),
\]

which is the frequency used by the published driver and remains valid through
phase wraps.

For the spin-weighted 22 pair,

\[
h_+-i h_\times=\sum_{m=\pm2}h_{2m}\,{}_{-2}Y_{2m}(\iota,\varphi),
\]

reduces in the chosen basis to the implemented real time-domain projections

\[
h_+=\sqrt{\frac{5}{16\pi}}(1+\cos^2\iota)\operatorname{Re}h_{22},\qquad
h_\times=-\sqrt{\frac{5}{4\pi}}\cos\iota\operatorname{Im}h_{22}.
\]

Finally, `torch.fft.rfft` uses the kernel (e^{-2\pi i f t}), and the driver
translates the waveform by the attachment time and requested `tc` with the
same sign. No sign reversal, phase-origin change, or polarization swap is
forced by the published equations.

## Independent tests

Added `test_quadrupole_fd_matches_finite_fourier_sum_for_general_mode_phase`.
It compares the FFT helper against a direct finite Fourier sum for three
inclinations and three nontrivial mode phase offsets, with the mode placed on
an exact positive-frequency bin. Together with the existing periodic-mode,
complex-frequency, orbital-rotation, and public-driver time-origin tests,
this checks the conventions without using an oracle residual.

Command:

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_clean_room_eob_waveform.py \
  tests/test_clean_room_eob_published_driver.py \
  tests/test_clean_room_seobnrv4_mode22.py
```

The new six-case analytic subset and the full three-file focused run pass:
`55 passed`.

## Decision and metric

No code correction is justified: all audited phase, mode, projection, Fourier
sign, and time-origin identities pass independently. The frozen strict
SEOBNR_ODE diagnostic therefore remains unchanged at relative L2
`1.0206987416256423` (polarization) and `1.0205836988469421` (projection).
Because this pass changes only tests and documentation, these are before and
after values; the remaining discrepancy cannot be attributed to a convention
error established by this audit. Its precise source requires a separate audit
of model physics or calibration choices.
