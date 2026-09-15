# Pass 3: published merger-ringdown matching

`src/torchwave/approximants/eob/merger_ringdown.py` implements the standalone
SEOBNRv4 ansatz in Bohé et al., **arXiv:1611.03703v3**, Eqs. (2.10),
(2.12)–(2.17). It accepts a caller-supplied QNM and matching data. It does not
choose a remnant fit, locate a trajectory peak, or replace the prototype's
existing attachment.

## Source and reconstruction

The primary agent read Sec. II C directly from a fresh
[versioned paper PDF](https://arxiv.org/pdf/1611.03703v3), downloaded by the
clean-room baseline implementer. SHA-256:
`db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba`.
The four free shape coefficients are the printed polynomial fits in
Eqs. (2.13), (2.14), (2.16), (2.17). No binary output or external implementation
supplied a coefficient.

All times are relative to matching and measured in the initial total mass.
Write the supplied QNM as `sigma=omega_qnm-i*gamma`, and use
`h22=A*exp(-i*phi)`. If the supplied matching amplitude, amplitude derivative,
positive GW frequency and phase are `A0`, `Adot0`, `omega0`, `phi0`, imposing
C1 matching on the printed ansatz gives

```
C1 = (Adot0 + gamma*A0)*cosh(c2f)^2/c1f
C2 = A0 - C1*tanh(c2f)
D1 = (omega_qnm - omega0)*(1+d2f)/(d1f*d2f)
```

The paper writes an explicit overall symmetric-mass-ratio factor. Here the
caller supplies its actual amplitude normalization; that factor cancels in
the constrained coefficients and is not multiplied in a second time.

The runtime evaluates equivalent stable forms. The tanh addition identity
gives the QNM-rescaled amplitude

```
A0 + (Adot0 + gamma*A0)/c1f
     * tanh(c1f*t)/(1+tanh(c1f*t)*tanh(c2f))
```

The positive accumulated phase is
`phi0+omega_qnm*t+D1*log((1+d2f*exp(-d1f*t))/(1+d2f))`.
The `d2f=0` limit is removable: `log1p(x)/x` is evaluated with its analytic
Taylor continuation, where `x=d2f/(1+d2f)*expm1(-d1f*t)`. This keeps values
and gradients finite at zero and avoids subtracting close logarithms at
matching. Multiplication by `exp(-gamma*t-i*phi)` restores the QNM.

## Validation and limits

The 12 focused tests check the printed nonspinning test-particle constants,
complex value and derivative at matching for three amplitude slopes, QNM
asymptotics and frequency monotonicity, two-dimensional parameter batching,
amplitude-unit scaling, the zero-`d2f` analytic limit, gradients in intrinsic
parameters/QNM/matching data, and explicit invalid-input rejection. Combined
with the 13 NQC tests: **25 passed in 0.63 s**.

A separate clean-room implementer checked the four fits and independently
derived the constrained coefficients from the same versioned primary paper.
Its literal-coefficient evaluation over 16 parameter configurations and seven
times agreed to `1.42e-14` relative error; no actionable finding was reported.

The calibrated paper attaches at an amplitude peak (`Adot0=0`). General
slopes expose the mathematical C1 matching conditions; they do not establish
the calibrated attachment prescription. Inputs must provide a damped QNM
in **initial-mass units**, not final-mass units without conversion. The
published calibration uses the fundamental **(2,2,0)** QNM. The
nonlinear amplitude factor is not clipped. A poorly chosen nonpeak slope
can therefore produce an unphysical signed amplitude later.

This is a component implementation, with no full-waveform or parameter-domain
parity claim. The incomplete trajectory peak and remnant conventions must be
resolved before it can support a calibrated end-to-end generator.
