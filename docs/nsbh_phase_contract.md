# NSBH phase contract (clean-room)

The permitted primary sources are Dietrich et al., arXiv:1905.06011v2
(NRTidalv2; downloaded 2026-09-13, SHA-256
`692ccb5a3c9c8098c9e389d9f118e2709b72b95076bb53600d5e0b2b93f60f6f`) and
Thompson et al., arXiv:2002.08383v2 (PhenomNSBH; SHA-256
`7e8bf4ba34a7fb98d54fa1a405267fdb4d06a78e1b164f57361b0b2abebd34e2`).

Dietrich Eq. (18) defines the frequency-domain tidal phase

\[
\psi_T(x)=-\frac{39}{16\nu}\kappa_T^{\rm eff}x^{5/2}
 \frac{1+\tilde n_1x+\tilde n_{3/2}x^{3/2}+\tilde n_2x^2+
 \tilde n_{5/2}x^{5/2}+\tilde n_3x^3}
 {1+\tilde d_1x+\tilde d_{3/2}x^{3/2}+\tilde d_2x^2},
\quad x=(\pi M f)^{2/3},
\]

with \(\kappa_T^{\rm eff}=3\tilde\Lambda/16\), equivalent to the existing
`-117/(256 eta) * lambda_tilde` prefactor. Its Eq. (16) gives the SPA
relation \(d^2\psi_T/d\omega^2=(1/\omega)d\phi_T/d\omega\); this does not
specify an integration constant or an affine gauge. Thompson Sec. II.B says
the model is PhenomD plus NRTidal, but defines no merger, taper, or peak phase
anchor. Therefore no reference-frequency or endpoint correction is justified
by these sources.

`nrtidalv2_frequency_phase_derivative` is the direct analytic derivative of
Eq. (18), in rad/Hz. It applies no affine shift and is not integrated into the
production NSBH waveform. Tests compare it against PyTorch differentiation of
the same equation and verify linearity in `lambda_tilde`.

Reproducible diagnostic command (black-box public binary comparison):

```text
PYTHONPATH=src:tests /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/nsbh_endpoint_diagnostic.py
```

At the canonical `df=0.25, fmax=1024` settings, the unchanged waveform
diagnostic reports catalog phase slope `-2.943478878657425e-7` rad/Hz and RMS
`3.1770964386772236e-9`; holdout slope `-1.958664693801455e-5` rad/Hz and RMS
`1.0142996731092146e-5`. Since this change only adds an unused helper, its
before/after waveform output is identical; these slopes remain observations,
never correction constants.
