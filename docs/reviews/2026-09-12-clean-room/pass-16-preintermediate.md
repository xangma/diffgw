# Pass 16: IMRPhenomXHM pre-intermediate amplitude branch

## Disposition

No runtime implementation or focused test is added.  The permitted primary
sources specify the shape of the branch, but do not provide the numerical fit
needed to construct its activation frequency.  Adding a branch from the paper
description alone would invent calibration behavior.

## What the paper fixes

The source is Garcia-Quiros et al., arXiv:2001.10914v2 (paper PDF SHA-256
`00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee`, extracted
text SHA-256
`8041291797cf4facb95801762fbb80a59eaedf6211c354347d7e8a3392f17c86`).  Section
V.A.2 and Fig. 6 state that, for the `(2,1)` extreme-mass-ratio branch, the
usual inspiral cutoff is replaced when `q > 70` by the frequency of a fitted
local maximum of the amplitude of `psi4`.  They define the extra collocation
point

```text
fInt0 = fIns + (fInt1 - fIns)/3
```

and calibrate amplitude value and derivative there.  The pre-intermediate
segment uses the inverse of a fourth-order polynomial (five free
coefficients), with the five conditions in Table III: value and derivative at
the fitted `fIns`, value and derivative at `fInt0`, and value at `fInt1`.  The
default inverse-fifth-order intermediate reconstruction then starts at
`fInt0` (Table IV conditions with `fIns` replaced by `fInt0`).  This gives the
intended value/derivative continuity at the join, subject to numerical
conditioning.

The paper does not publish the function, coefficients, independent variables,
validity domain, or extrapolation rule for the fitted local-maximum frequency.
It only says that a fit was made over the EMR parameter space and used for
`q > 70`.  Thus the paper supplies the construction but not a reproducible
`fIns(eta, spins, ...)` definition.

## Ancillary audit

The official ancillary index is
[`arXiv ancillary files`](https://arxiv.org/src/2001.10914v2/anc/).  The
`ParspaceFits/AmpFits/21` mathematical files are the ordinary amplitude
collocation and ringdown fits (`Insp1/2/3`, `Int0/1/2`, `DInt0`, `ALambda`,
`Lambda`, and `Sigma`).  For example, the downloaded `CoefficientStatsDInt0.m`
has SHA-256
`24f4ddaa2f17ce7aedc8dc7b46e2b6bae83870858ad2cbef9c71504213ea296c`, and
`CoefficientStatsInt0.m` has SHA-256
`38a3f472e53733d45ef5a7e3a56de0981b920122146d1dc2233a08160c77b38a`.
Those files contain fits for amplitude values/derivatives and ringdown shape
parameters; none defines the fitted `psi4` local-maximum frequency.  The
ancillary index contains no separate `fIns`, `fPeak`, `psi4`, or local-maximum
`.m` asset.  The only listed artifact that could contain the omitted fitting
workflow is `ParspaceFits/IMRPhenomXHM_AmplitudeFits.nb`, which is explicitly
excluded by the clean-room rules.

Consequently the permitted evidence leaves the following required runtime
inputs unresolved: the local-maximum fit formula and coefficient set, its
independent spin/mass-ratio variables and fitted domain, and the behavior at
the `q = 70` boundary or outside that domain.  The inverse-quartic solve can
be written once those quantities are supplied, but doing so now would not
reproduce the published model.  No files outside this report were changed for
pass 16; in particular `imrphenom_xhm.py` remains untouched.

