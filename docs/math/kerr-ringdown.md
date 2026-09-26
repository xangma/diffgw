# Kerr 220 energy-normalized carrier

Specification ID: `kerr-220-energy-carrier-conditional/v1`.

## Scope and status

This specification covers the causal, physical-energy fundamental 220
polarizations and their Fourier-minus transform in `kerr220_td`/`kerr220_fd`.
It conditions on supplied QNM frequency/damping and normalized angular factors.
The separate [angular projection specification](kerr-angular-projection.md)
now checks selected spherical and spheroidal factors independently. This
carrier check itself does not verify the QNM eigenvalue problem; a separate
[scalar 220 QNM input check](kerr-qnm-scalar-qualification.md) covers three
moderate-spin frequencies and angular eigenvalues. Neither check qualifies
excitation, the inferred public-compatible angular-series stopping rule, or
other modes and overtones.

## Inputs, conventions, and options

The remnant mass $M$ and luminosity distance $D$ enter in geometric seconds,
$M_s=GM/c^3$ and $D_s=D/c$. Let $\epsilon$ be the fraction of remnant rest
energy emitted by one mode, $\omega>0$ its angular frequency in rad/s,
$\lambda=1/\tau>0$ its damping rate, $t_c$ the onset, and $S_+$ and $S_-$
the supplied spin-weighted angular values at $\iota$ and $\pi-\iota$.
The frontend inputs use solar masses, Mpc, dimensionless QNM frequency and
damping, and phase $\phi_c$ included in the angular factors. The two sides of
the onset use $H(t-t_c)$ with $H(0)=1$. The Fourier convention is
$\widetilde h(f)=\int h(t)e^{-2\pi ift}\,dt$.

## Mathematics and components

The finite-quality-factor amplitude for a single complex mode is

$$
A=\frac{\sqrt{32\pi\epsilon M_s\tau}}
        {D_s\sqrt{1+(\omega\tau)^2}}.
$$

This follows by integrating the mode energy flux for $t>t_c$, retaining both
the oscillatory and damping derivatives. The energy-flux convention is from
Ruiz et al., [arXiv:0707.4654v3](https://arxiv.org/abs/0707.4654), Eq. (2.8),
with unit angular norm as in Flanagan and Hughes,
[arXiv:gr-qc/9701039v2](https://arxiv.org/abs/gr-qc/9701039), Eq. (3.22).
For comparison, Berti, Cardoso, and Will,
[arXiv:gr-qc/0512160v2](https://arxiv.org/abs/gr-qc/0512160), Eq. (3.18),
write a dimensionless coefficient $A_{lmn}^{\rm FH}$ using
$Q=\omega\tau/2$ and $f_{lmn}=\omega/(2\pi)$. Identifying their
$\epsilon_{\rm rd}$ with $\epsilon$ gives the algebraic mapping
$A=(M_s/D_s)A_{lmn}^{\rm FH}$. Their equation assumes the FH doubled-time
waveform and equal-polarization convention; this coefficient identity does
not establish equivalence of that waveform to the causal projection below.

For $u=t-t_c\geq0$, $z=e^{-(\lambda+i\omega)u}$, the selected projection is

$$
h_+(t)=-\frac{A}{\sqrt2}\operatorname{Re}[z(S_++S_-)],\qquad
h_\times(t)=\frac{A}{\sqrt2}\operatorname{Im}[z(S_+-S_-)].
$$

Both are zero for $u<0$. Define $P=S_++S_-$, $C=S_+-S_-$, and
$L_\pm=[\lambda+i(2\pi f\pm\omega)]^{-1}$. Direct Laplace integration gives

$$
\widetilde h_+=-\frac{A e^{-2\pi ift_c}}{2\sqrt2}
  (P L_+ + P^*L_-),\qquad
\widetilde h_\times=\frac{A e^{-2\pi ift_c}}{2i\sqrt2}
  (C L_+ - C^*L_-).
$$

## Implementation mapping

`diffgw_reference.kerr_ringdown` evaluates these expressions with scalar
mpmath arithmetic and also integrates the time-domain polarizations by
quadrature as a separate Fourier check. The Torch and JAX production paths
are `torchwave.approximants.kerr_ringdown_public_clean` and
`jaxwave.approximants.kerr_ringdown_public_clean`. The maintained
`test_kerr_ringdown_math_reference.py` supplies each frontend's QNM and
angular values explicitly to the reference and compares both polarizations
before, at, and after onset and at positive, zero, and negative frequencies.

## Evidence and limitations

The scalar checks isolate energy normalization, causal projection, Fourier
signs, mirror contribution, and time shift on two off-canonical parameter
sets. Since they share QNM and angular inputs with each tested frontend,
they do not independently establish those inputs. A fixed off-canonical
full-waveform holdout uses the public `SimBlackHoleRingdown` TD callable,
converged numerical Fourier integration, and the unchanged strict parity
gate. It tests one additional point, not a parameter domain.

## Reproduce the checks

Run `PYTHONPATH=src:tests python -m pytest -q
tests/test_kerr_ringdown_math_reference.py` in a reference-enabled environment.
The independent public-binary holdout is the
`KerrRingdown_holdout_off_axis_energy_onset` case in `tests/parity_harness.py`.
