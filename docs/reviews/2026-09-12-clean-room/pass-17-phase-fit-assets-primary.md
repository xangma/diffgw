# Pass 17: intermediate phase-fit assets (primary-source audit)

## Source and derivation

The primary source is arXiv:2001.10914v2, source archive URL
[`https://arxiv.org/e-print/2001.10914v2`](https://arxiv.org/e-print/2001.10914v2).
The archived source tarball is SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`; the
extracted `hmc2.tex` used here is SHA-256
`5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.

Section “Phase” defines the intermediate derivative as

\[
 \phi'_{\ell m}=a_\lambda\frac{f_\mathrm{damp}}
 {f_\mathrm{damp}^2+(f-f_\mathrm{ring})^2}
 +a_0+a_1/f+a_2/f^2+a_3/f^3+a_4/f^4.
\]

For the nonmixing modes `(2,1)`, `(3,3)`, and `(4,4)`, the paper sets
`a3=0` and solves five collocation equations. Table `inter_phase_cpoints`
defines six frequencies `f1` through `f6`, with `f1=beta*fIns` and the
remaining expressions given explicitly in the TeX. Figure 8 names these
points `[C1,C2,E2,E3,E4,E5]`: the standard five-point subset is
`[C1,C2,E2,E3,E5]`, while the fallback subset is `[C1,C2,E3,E4,E5]`.
Thus, with the p1...p6 asset ordering, the required RHS assets are

* default: p1, p2, p3, p4, p6;
* alternate: p1, p2, p4, p5, p6.

For `(3,2)`, the paper explicitly uses only four intermediate collocation
equations (`i=1..4`) and retains `a3`. Two additional equations at the
ringdown join impose `phi'` and `phi''` continuity. Its `fEnd` is
`fRing22 - 0.5*fDamp22`; the four intermediate RHS fits are therefore p1
through p4. The source tarball also contains `3Dfit_p5_lm_32.m` and
`3Dfit_p6_lm_32.m`, but the published `(3,2)` intermediate system does not
use them, so they are intentionally not archived here.

## Archived assets and native comparison

The directly used missing files are archived under
[`source-assets/pass-17-phase`](source-assets/pass-17-phase), with a
provenance README and SHA-256 manifest:

* `3Dfit_p1...p5_lm_21.m`, `3Dfit_p1...p5_lm_33.m`, and
  `3Dfit_p1...p5_lm_44.m`;
* `3Dfit_p1...p4_lm_32.m`.

The existing pass-14 archive supplies p6 for the three nonmixing modes.
Each archived `.m` file is a Wolfram Language expression whose first
expression is the parameter-space fit; the remaining entries are fit
diagnostics. After recording source hashes, I compared those first
expressions with the native definitions in `src/torchwave/approximants/xhm_fits.py`.
The required p1...p5 nonmixing definitions and p1...p4 `(3,2)` definitions
have matching coefficients and spin/eta monomials after the mechanical
notation conversion `\\[Eta] -> eta`, `S -> S`, and Wolfram `*^n -> 10^n`.
The native p5/p6 `(3,2)` definitions are present but are not used by the
published intermediate solve.

## Recommendation and blocker

No fit-function runtime change is required: the complete source-backed RHS
set is already represented by the native fit functions. The implementation
should keep the five-equation nonmixing solve on p1,p2,p3,p4,p6 and may add
the paper's alternate p1,p2,p4,p5,p6 solve only when its own dispatch rule
is established. The paper says this switch is for “regions with fewer
calibration waveforms” (high spin and/or low eta), but gives no numerical
parameter-space boundary or predicate. Enabling that branch by inventing a
threshold would exceed the primary source.

The `(3,2)` solve is source-complete with p1...p4 plus the two ringdown
boundary equations. Focused future tests should assert those four RHS fits,
the `fRing22 - 0.5*fDamp22` endpoint, and six-equation (`phi'`/`phi''`)
assembly; a sparse nonmixing dispatch test must remain blocked until its
source-defined region is available. No runtime or test files were edited in
this audit.
