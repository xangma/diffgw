# Pass 17 independent review: high-q inspiral phase

Date: 2026-09-13. This review uses only García-Quirós et al.,
[arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2), its v2 source
archive, and the official mathematical ancillary files. It excludes
implementation ports, generated C, notebooks, git history, verification
outputs, residuals, and public binaries. No runtime or test file was edited.

## Source identity and asset scope

The v2 source archive is
[the arXiv source](https://arxiv.org/src/2001.10914v2), SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`.
`hmc2.tex` is SHA-256
`5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.

The project archive at
`docs/reviews/2026-09-12-clean-room/source-assets/pass-17-phase/README`
contains only the `3Dfit_p1`--`p5` intermediate derivative assets; it
explicitly excludes the `Lambda.m` files. The official source archive does
contain those files. Their hashes, recorded here for provenance, are:

| mode | official source asset | SHA-256 |
|---|---|---|
| 21 | `anc/ParspaceFits/PhaseFits/21/Lambda.m` | `d8868c0d038bf69b3c9d3852930b1a8d635a85ff10cce25e7ad9999a280c4906` |
| 32 | `anc/ParspaceFits/PhaseFits/32/Lambda.m` | `da5e09900972cb53b8ea3e419533a3e9058a8ab80b619b6f3a001b32bfcf6716` |
| 33 | `anc/ParspaceFits/PhaseFits/33/Lambda.m` | `5e7da0c943cab189cb6a3d49a6db81e8b561ac98654c30e67e84baf334fd5461` |
| 44 | `anc/ParspaceFits/PhaseFits/44/Lambda.m` | `c701d46ab1c7825582ef870747aec8b55c8113ee05f8416f5b0935891767be2d` |

The ancillary README says that the phase notebook exports functional forms
and gives examples for obtaining them, but that notebook is excluded here.

## Branch selection

The paper defines \`q = m1/m2 >= 1\` (source lines 207--209). Equation
(4.10), `eq:Lambda_lm`, source lines 759--765, is unambiguous:

```text
Lambda_lm = Lambda_lm^PN   if q < 100
            Lambda_lm^fit  if q >= 100.
```

Thus the equality belongs to the fit branch. If a caller uses only
If \`eta = q/(1+q)^2\`, the equivalent boundary is

```text
eta <= 100/101^2 = 100/10201 = 0.009802960494069209...
```

for the fit branch; \`eta > 100/10201\` is the PN branch. This conversion is
an algebraic consequence of the paper's q definition and the usual
\`q >= 1\` domain. It is not a separate paper convention.

## Is the first expanded `Lambda.m` expression the intended fit?

No source statement makes that selector unambiguous. Each official
`Lambda.m` is a Wolfram Language export containing many nested fit records,
not one named production function. For example, the first record begins at
line 2 in each file, but additional records begin at lines 47, 133, 227, ...
for mode 21; 53, 154, 264, ... for mode 32; 31, 105, 187, ... for mode 33;
and 44, 141, 243, ... for mode 44. The records contain different expanded
expressions and diagnostics; some later records include an explicit
\`chi1 - chi2\` term (for example mode 33 around source-asset line 105).

The paper says only that parameter-space fits are computed and that the
high-q ansatz uses \`Lambda_fit\` (source lines 742--764).
The ancillary README delegates the export/selection details to the excluded
notebook. Therefore the first expanded expression is a plausible import
candidate, but it is not a source-proven canonical selector. Importing it
would guess among records.

## Low-frequency phase constant and odd-m convention

The paper's phase ansatz is Eq. (4.10), source lines 752--756:

```text
phi_lm(f) = (m/2) phi_22^X(2f/m) + Lambda_lm(f)
            + dphi_lm^Ins f + phi_lm^Ins.
```

The carrier rescaling is also stated in Appendix `app:inspiral_hm`, source
lines 1677--1685. The derivative coefficient is fixed by continuity with the
intermediate ansatz (source line 767), but the paper does not provide a
closed numerical selector for that operation.

Equation (4.11), `phase_alignment`, source lines 769--774, fixes the relative
Fourier-phase constant:

```text
lim_{f->0} (Psi_lm - (m/2) Psi_22) = (3*pi/4) (1 - m/2).
```

Using the paper's Eq. (4.3) definition
(Psi_{ell m}=2\pi f t_c-phi_{ell m}-\pi/4+\psi_0) (source lines
556--567) and (psi_0=\pi), the target values are:

| mode | m | target relative Ψ constant |
|---|---:|---:|
| 21 | 1 | (3\pi/8) |
| 32 | 2 | (0) |
| 33 | 3 | (-3\pi/8) |
| 44 | 4 | (-3\pi/4) |

For a finite selected (Lambda_{ell m}(0)), the ansatz therefore requires
(phi_{ell m}^{\rm Ins}=-\Lambda_{ell m}(0)) after the common scaled
22 carrier is used. A frequency-independent fitted Lambda contributes no
derivative, but it does change this additive gauge constant.

The source also records the remaining odd-m convention. The tetrad condition
is (Phi_{ell m}\simeq(m/2)\Phi_{22}) at large separation (source lines
243--266), while Appendix `HMconventions`, source lines 1512--1519, states

```text
mod(2 Phi_lm - m Phi_22, 2*pi) = pi  for odd m,
                               0    for even m.
```

The paper explains that odd-m amplitudes change sign under a pi rotation,
that fits may use signed amplitudes, and that reconstructed amplitudes are
made non-negative by taking absolute values (source lines 468--485). It also
says that odd-mode phase boundary values near equal mass are complicated and
numerically noisy (source lines 481--485). Consequently the universal
Eq. (4.11) target fixes the phase gauge only once the signed-versus-absolute
amplitude convention and the selected Lambda record are fixed. It does not
authorize an additional hard-coded odd-m pi shift.

## Native comparison and decision

Only after deriving the source facts above, I inspected the native phase
path. The default non-(3,2) branch uses the scaled 22 carrier plus a
hard-coded `delta_target` at
`src/torchwave/approximants/imrphenom_xhm.py:1307-1315`; the 32 branch uses
`delta_target_32 = PI` at `:1458`. The `phase_lambda_*` fit definitions are
not selected in the default path. The optional
`experimental_phase_connections` path calls `_pn_phase_lambda` for every
mass ratio, rather than applying the published q boundary, and it does not
consume a `Lambda.m` fit record.

The q<100/q>=100 selector itself is fully specified and testable, but a
source-complete runtime patch for the full high-q phase branch is **not
implementable now without guessing**. Two independent pieces are unresolved:
the official assets do not identify which repeated `Lambda.m` record is the
production (Lambda^{\rm fit}), and the paper does not specify the exact
mapping of that record's constant/sign convention into the native odd-m
phase while preserving Eq. (4.11). Do not implement this slice until an
allowed author asset or erratum resolves both.

## Test plan if the blockers are resolved

Add focused batched, dtype/device, and gradient tests that:

1. assert PN selection at `q < 100`, fit selection at `q == 100` and
   `q > 100`, and the equivalent (eta=100/10201) boundary;
2. evaluate the author-selected Lambda expression for 21, 32, 33, and 44 in
   the fit branch and verify the carrier uses ((m/2)\phi_{22}(2f/m));
3. verify derivative continuity determines the linear coefficient and that
   the low-frequency relative phase constants are (3\pi/8,0,-3\pi/8,-3\pi/4)
   for 21, 32, 33, and 44 respectively;
4. check the odd-m tetrad relation modulo (2\pi) under the resolved
   signed/absolute-amplitude convention, including the equal-mass vanishing
   limit; and
5. run the same assertions for float32/float64 with spin and mass inputs
   requiring gradients.

No runtime or test file was modified in this review.
