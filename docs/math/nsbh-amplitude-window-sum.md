# IMRPhenomNSBH amplitude windows and sum

## Scope and status

Specification **nsbh.amplitude.window-sum/v1** is a conditional component
gate for the windowing and three-term sum in [Thompson et al. (2020),
arXiv v2, Eq. (4) and Eq. (30)](https://arxiv.org/html/2002.08383v2).
It uses supplied carrier amplitudes, joining centers, width, and total mass.
It does not qualify their upstream formulas or the full waveform.

## Inputs, conventions, and options

The frequency $f>0$ and centers $f_0,f_1,f_2>0$ are in Hz. The total mass
$M>0$ is in seconds, so $Mf$ is dimensionless. The window width
$d=0.015+\sigma_{\rm tide}>0$ is dimensionless in $Mf$ units. The three
nonnegative, dimensionless carrier amplitudes $P(f)$, $B(f)$, and $R(f)$ are
supplied independently at each sample; they correspond to $A_{\rm PN}$,
$\gamma'_1 f^{5/6}$, and $A_{\rm RD}$. The gate does not assume their
derivatives with respect to frequency. It checks the smooth positive-frequency
formula, not the native zero-frequency adapter or distance normalization.

## Mathematics and components

With $z_j=4M(f-f_j)/d$ and the paper's
$W_j^\pm=[1\pm\tanh(z_j)]/2$, Eq. (4) is

\[
S(f)=P(f)W_0^-+B(f)W_1^-+R(f)W_2^+.
\]

[Pannarale et al. (2015), Eq. (13) and its following discussion](https://arxiv.org/pdf/1509.00512v1)
independently states the three-term structure and $d=0.015+\sigma_{\rm tide}$.
For *constant supplied carriers* in a local derivative check,

\[
\frac{dS}{df}=\frac{2M}{d}\left[-P\,\operatorname{sech}^2(z_0)
-B\,\operatorname{sech}^2(z_1)+R\,\operatorname{sech}^2(z_2)\right],
\quad
\frac{\partial S}{\partial(P,B,R)}=(W_0^-,W_1^-,W_2^+).
\]

These derivatives are of the conditional window-and-sum component, not of
the physical waveform through its frequency-dependent carrier terms.

## Implementation mapping

`diffgw_reference.nsbh_amplitude_window_sum` evaluates Eq. (4) independently
with mpmath. Production-called Torch and JAX helpers evaluate the same
conditional component using their native tensor operations. The public NSBH
forward path supplies its existing three carriers, centers, width, and total
mass to those helpers. No reference code or LAL implementation is imported
into either native backend.

## Evidence and limitations

The test grid is frozen before native comparison: $M=5\times10^{-5}$ s;
four controlled center/width/carrier rows
$(f_0,f_1,f_2,d;P,B,R)$ are
$(800,800,800,0.015;0.8,0.3,0.5)$,
$(700,700,850,0.040;0.8,0.3,0.5)$,
$(650,750,850,0.040;0.8,0.3,0)$, and
$(500,600,850,0.080;0.8,0.3,0)$, with Hz centers.
They resemble the four Table I center patterns but are **supplied numerical
conditioning inputs**, not a check of class selection or joining fits.
For each row, sample $f=f_j+k d/(8M)$ for all three centers and
$k\in\{-2,-1,0,1,2\}$, plus $f=50,1500$ Hz, retaining positive values.
The centers and their neighborhoods test the signs, half values, width and
unit conversion; the zero-ringdown rows test inert third terms.

At every point, independent 70- and 90-digit scalar evaluations of $S$,
$dS/df$, and the three carrier partials must agree within $10^{-30}$
absolutely. Torch and JAX CPU FP64 values must match the 70-digit scalar
within $2\times10^{-13}$ absolute; selected $dS/df$ and carrier partials
at each center and the two adjacent offsets must match within
$2\times10^{-12}$ absolute. These are fixed limits, not tunable against
native output. The test also compares the public waveform before and after
the production-helper refactor on a fixed, positive-frequency sample grid
to guard against an output change.

The [joining-frequency gate](nsbh-joining-frequencies.md) still finds a
low/no-torus paper/native difference. The [variant audit](nsbh-amplitude-variant-provenance.md)
still blocks paper qualification of Eq. (26)/(29) corrections. The PhenomC
carrier, remnant extrapolation guards, distance normalization, and whole
waveform are outside this component gate.

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src pytest -q
tests/test_nsbh_amplitude_window_sum_math_reference.py` with mpmath, Torch,
JAX, and CPU FP64 support. The validation receipt records measured errors,
versions, and the neighboring checks.
