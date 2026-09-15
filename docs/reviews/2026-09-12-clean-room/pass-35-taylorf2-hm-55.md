# Pass 35: clean-room TaylorF2_HM `(5,5)` audit

## Decision

No runtime or test change is made.  The primary literature fixes a leading
Fourier-domain `(5,5)` coefficient, but this bounded audit does not establish
the final identification of that source-frame mode with TorchWave's native
stored negative-`m` mode convention independently of an already-implemented
mode.  Adding `(5,5)` would therefore rely on convention extrapolation.  The
supported-mode inventory remains

```text
((2,2), (2,1), (3,1), (3,2), (3,3), (4,2), (4,3), (4,4))
```

and `(5,5)` remains unsupported.

## Primary source evidence

Mishra et al., *Ready-to-use post-Newtonian gravitational waveforms for
compact binary coalescences*, Phys. Rev. D **96**, 024010 (2017),
[arXiv:1601.05588](https://arxiv.org/abs/1601.05588), gives the individual-mode
SPA construction in Eq. (11) and the leading coefficient in Eq. (12j):

```text
Hhat_55 = -(125/96) sqrt(5/33) delta (1 - 2 eta) V_5^3 + O(5).
```

The same equation set gives `Hhat_22 = -1 + O(V_2^2)`.  Algebraically, if
that source convention is identified with the native shared quadrupole
normalization, the candidate leading ratio would be

```text
A_55 / (sqrt(16*pi/5) A0)
  = (25/48) sqrt(5/33) (5/2)^(7/6)
    * delta * (1 - 2 eta) * v22^3.
```

Here `V_5=(2/5)^(1/3) v22`, and the extra `(5/2)^(7/6)` is the SPA Jacobian
from the common `V_m^(-7/2)` prefactor.  The coefficient consequently has
the expected equal-mass zero and is differentiable as a polynomial in the
existing variables.

## Exact blocker at the native API boundary

The paper's Eq. (11) is written for `\tilde h_lm` in its own source-frame
mode convention, while TorchWave stores a positive-frequency mode as the
native `h_{l,-m}` contribution and then combines it with the negative-`m`
SWSH partner.  The source also documents an `i^m` frame relation.  The
allowed source evidence does not explicitly state the complete mapping of
the `(5,5)` source-frame coefficient, its `i^m` factor, and the native
negative-`m` storage phase used by this API.  The existing `(4,3)` branch
demonstrates an internal convention, but using it to infer `(5,5)` is an
implementation convention rather than an independently established
source-level identity.

The standard `s=-2` angular functions themselves are unambiguous.  In the
same convention as the existing branches they would be

```text
Y_5,5  = -sqrt(165/(512*pi)) sin(iota)^3 (1 + cos(iota))^2 exp(5 i phi0)
Y_5,-5 =  sqrt(165/(512*pi)) sin(iota)^3 (1 - cos(iota))^2 exp(-5 i phi0).
```

Those formulas do not resolve the source-mode-to-native-mode phase boundary.
No fitted constants, LAL/PyCBC/TEOBResumS implementation, or waveform oracle
was consulted.

## Validation

The existing focused TaylorF2_HM suite was run unchanged:

```text
15 passed in 2.17s
```

`py_compile` passed for the TaylorF2_HM implementation and focused tests, and
`git diff --check` passed.  The primary source TeX used for the equation text
had SHA-256
`95c9784ca1d07374d7526f234592e77d5296459df7081f343d1aaf7b8e128000`.
