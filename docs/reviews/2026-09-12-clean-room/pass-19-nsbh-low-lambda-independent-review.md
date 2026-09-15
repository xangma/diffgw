# Pass 19 independent review: IMRPhenomNSBH low-Λ amplitude factor

## Decision

**RETAIN**, with a source-notation blocker on any claim of literal Thompson
Eq. (ampansatz) transcription. The current native behavior preserves the
published piecewise values and uses the historically supported
`correction * PhenomC gamma1` interpretation.

## Source and scope

I checked Thompson et al., [arXiv:2002.08383v2](https://arxiv.org/abs/2002.08383v2),
and the cited primary amplitude papers Pannarale et al.,
[arXiv:1307.5111v1](https://arxiv.org/abs/1307.5111v1) and
[arXiv:1509.00512v1](https://arxiv.org/abs/1509.00512v1), then inspected only
the relevant current code in
[`imrphenom_nsbh.py`](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_nsbh.py).
I excluded implementation ports, generated code, notebooks, history,
verification artifacts, binaries, and residuals. No runtime or test file was
edited.

## Findings

Thompson Appendix A, Eq. (gamma-1-prime), gives the exact scalar factor

```text
g(Lambda) = 1 - 0.5*Lambda - 0.25*Lambda^2,  Lambda <= 1
            1.25,                              Lambda > 1.
```

Therefore `g(0) = 1`, `g(1) = 0.25` (the closed lower branch), and
`lim(Lambda -> 1+) g = 1.25`. The source has an exact unit jump at Lambda=1;
its statement that the split smoothly matches the BBH limit at Lambda=0 does
not establish equality at the split.

Thompson’s displayed amplitude writes the pre-merger term as
`gamma1_prime * f^(5/6)` and does not show PhenomC `gamma1`. That is
inconsistent with the same workflow’s calculation of `gamma1` and leaves the
literal coefficient ambiguous. Pannarale 2013 first defines the PhenomC term
as `gamma1*f^(5/6)` and introduces `kappa*gamma1` as its NSBH correction;
Pannarale 2015 retains `1.25*gamma1*f^(5/6)` in the mixed-spin and
non-disruptive ansätze. Those primary sources support interpreting Thompson’s
piecewise quantity as the correction multiplying PhenomC `gamma1`.

The native code at lines 633--637 applies exactly that interpretation:

```python
torch.where(lam2 > 1.0,
            1.25 * gamma1,
            (1.0 - 0.5 * lam_safe - 0.25 * lam_safe**2) * gamma1)
```

and uses the result as the coefficient of `f_bar**(5/6)` at lines 649 and
744--745. Thus it selects the lower branch at equality and has the same
`0.25*gamma1` to `1.25*gamma1` jump (unless `gamma1` is zero).

## Blocker

No permitted primary source resolves whether Thompson’s `gamma1_prime` is an
absolute coefficient or a correction factor, and no permitted source supplies
a continuity repair at Lambda=1. The native product is the defensible
Pannarale/BBH-limit interpretation, so retain it and preserve the published
jump. Calling it an exact literal Thompson transcription, or silently
regularizing the boundary, would require an unsupported additional choice.
