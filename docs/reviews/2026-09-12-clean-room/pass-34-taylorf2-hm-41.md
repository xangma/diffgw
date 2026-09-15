# Pass 34: clean-room TaylorF2_HM `(4,1)` audit

## Decision

No runtime change is made.  The primary source supplies a leading circular
`(4,1)` time-domain mode, but the source set does not uniquely determine its
translation into the native `TaylorF2_HM` frequency-domain mode convention.
Adding a ratio or phase by analogy would be speculative.

## Primary source evidence

Chakraborty et al., *Spin effects in Spherical Harmonic Modes of Gravitational
Waves from Eccentric Compact Binary Inspirals*, Phys. Rev. D **108**, 024023
(2023), [arXiv:2211.04155](https://arxiv.org/abs/2211.04155), gives the general
orbit `(4,1)` expressions in Eqs. (3.8) and (3.9) of the source TeX (lines
420--425 of the downloaded source):

```text
H41_NS = -2/(7*sqrt(70)) (GM/r) delta (1-2*nu) (r*phidot)
         [GM/r - rdot^2/2 + (5 i/6) rdot (r*phidot)
          - 11 (r*phidot)^2/12] + O(c^-5)
```

Their mode convention is Eq. (3.10):

```text
h_lm = -4 i^(2m) sqrt(pi/5) (GM nu/(c^4 R)) exp(-i m phi) H_lm.
```

Setting `rdot=0` and `GM/r=(r*phidot)^2=x` gives
`H41_NS = -delta*(1-2*nu)*x^(5/2)/(42*sqrt(70))` in that source-frame
convention.  The paper also gives the leading spin-orbit `(4,1)` term, but it
uses the source-frame spin combinations `s_c` and `sigma_c`; translating it to
the scalar aligned-spin API would require selecting and documenting a complete
spin-amplitude truncation.

The paper explicitly says that its source frame differs from the Blanchet
frame by an `i^m` factor (source line 344).  Its circular-limit `(4,1)` result
is therefore a time-domain spherical-harmonic mode, not the native stored
frequency-domain quantity.

## Exact blocker at the native API boundary

`src/torchwave/approximants/taylorf2_hm.py` stores each positive-frequency
mode as a ratio to one shared quadrupole SPA amplitude `amp0`, evaluates the
base TaylorF2 phase at `2*f/m`, and applies the native
`delta_phi=(3*pi/4)*(1-m/2)` connection.  The allowed primary papers provide
the `(4,1)` time-domain coefficient above and, separately, the generic SPA
harmonic construction, but do not identify that source-frame mode with the
native stored `(h_{l,-m})` quantity or establish the native phase connection
for `m=1`.

In particular, the mapping must resolve all of the following together:

1. the `m=1` stationary-phase Jacobian and the frequency argument
   (`V_1=(2*pi*M*f)^(1/3)` versus the native shared `v22`);
2. the `i^(2m)` source-frame factor and the negative-`m` storage/projection
   convention; and
3. the native `m=1` phase offset relative to the paper's common `-pi/4` SPA
   term.

The existing pass-19 audit records this same unresolved normalization and
phase-convention boundary.  The `(4,2)` pass was source-complete because
`m=2` removes the frequency and phase conversion ambiguity; that argument
does not apply to `(4,1)`.  Consequently no amplitude constant, phase offset,
SWSH branch, or mode-inventory entry is added in this pass.

## Clean-room scope and validation

Only the two primary paper sources were consulted for this pass.  No LAL,
PyCBC, TEOBResumS, waveform implementation, generated implementation asset,
or oracle fit was inspected.  The downloaded source hashes were:

```text
2211.04155 source archive: f709063c2f9afa500f57eb4fb89183b272fd1f9f4bfbd9710ea8ca9bb8
2211.04155 draft.tex:      8655a1550f987cfe2ece16ab54f49d2d721cc813721b983e35c39d494a923509
2001.10914 hmc2.tex:       5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e
```

The IMRPhenomXHM primary paper (`2001.10914`) lists its PN mode set without
`(4,1)`, so it supplies no independent `(4,1)` frequency-domain convention.
No runtime or test files were changed.  The existing TaylorF2_HM focused suite
was run unchanged and remains green; the working-tree diff was checked for
whitespace errors.
