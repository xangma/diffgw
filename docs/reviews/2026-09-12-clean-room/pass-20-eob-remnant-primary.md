# Pass 20 — SEOBNRv4 remnant fit selection (primary-source audit)

**Decision: RETAIN the current remnant equations; do not add a default Hofmann block.**
The final-mass prescription is source-complete and agrees algebraically with the
published SEOBNRv4 lineage.  The final-spin ansatz and all three Hofmann coefficient
blocks are source-complete, but the SEOBNRv4 paper does not identify which block is
its target.  A runtime default would therefore be an unsupported calibration choice.
The current required `fit_order` interface preserves that uncertainty explicitly.
No runtime or test file was changed.

## Permitted primary material and provenance

The source archives were fetched directly from arXiv during this pass.  Hashes are
SHA-256 of the downloaded archives:

| material | URL | local archive/hash |
|---|---|---|
| SEOBNRv4, Bohé et al., arXiv:1611.03703v2 (TeX) | <https://export.arxiv.org/e-print/1611.03703v2> | `/tmp/pass20/seobnrv4.tar`, `2ace0a82f0009fd03a516b29d3a652486f2eac39ea81f33a4d4d740c09304e21` |
| SEOBNRv2 lineage, Taracchini et al., arXiv:1311.2544 (TeX) | <https://export.arxiv.org/e-print/1311.2544> | `/tmp/pass20/taracchini.tar`, `d36532ad40b3354ac3e6ea0cb9880f762c0ad3d4667b57ea1c91032ca89e1650` |
| Barausse, Morozova & Rezzolla, arXiv:1206.3803 (TeX) | <https://export.arxiv.org/e-print/1206.3803> | `/tmp/pass20/barausse.tar`, `7646ab9c33b01805ad25091ca9551a454824e9cb9a8439b3cc3d41120225dbe1` |
| Hemberger et al., arXiv:1305.5991 (TeX) | <https://export.arxiv.org/e-print/1305.5991> | `/tmp/pass20/hemberger.tar`, `79ee7d6ba69a524e3114d4b38ec3ed601cd744a9be205d4ce4153dbea62fb2c1` |
| Hofmann, Barausse & Rezzolla, arXiv:1605.01938v2 (TeX) | <https://export.arxiv.org/e-print/1605.01938v2> | `/tmp/pass20/hofmann.tar`, `316476cf38a0bd0d8e44fc0870cb042a706fb33e0033bb5563160eec33fff476` |
| Ossokine et al., arXiv:2004.09442v1 (TeX; explicit later statement of the same mass fit) | <https://export.arxiv.org/e-print/2004.09442v1> | `/tmp/pass20/ossokine-v1.tar`, `b6558246a5a854b240ac717199748ab5eba9ebd861b70e348c08527d770c8874` |

The native comparison files were read only after the source derivation:
`src/torchwave/approximants/eob/remnant.py` SHA-256
`e7690cd5df1bcde2f579954762116beb20db606dae809835b148dae73c4cb007` and
`tests/test_clean_room_eob_remnant.py` SHA-256
`cdaa0882d17edd4904134070b02e5fd72afe7b5ff511947a075314743976eab3`.

## Final mass: exact prescription and domain

The SEOBNRv4 paper (1611.03703v2, merger-ringdown discussion) states that it
uses the final-mass formula of Taracchini et al. and that this is based on
Barausse et al. and Hemberger et al.  Taracchini (1311.2544, ringdown
paragraph) makes the selection explicit: use the Barausse phenomenological
formula, replacing its equal-mass limit (Barausse Eq. 11) with Hemberger Eq. 9.
Hemberger Eq. 9/10 is the equal-mass aligned/anti-aligned radiated-energy fit

\[
 E_{\rm eq}(\chi_i)=b_0+\frac{b_1}{b_2+\chi_i},\qquad
 (b_0,b_1,b_2)=(0.00258,-0.07730,-1.6939),
\]

where the fit is for equal masses and equal signed spin projections.  Barausse's
Eq. (11) is the interpolation that is first order in the test-particle limit and
exactly reaches the equal-mass fit at \(\nu=1/4\):

\[
 E_{\rm rad}=\nu(1-4\nu)[1-E_{\rm ISCO}(a)] +16\nu^2 E_{\rm eq}(\chi_i).
\]

The later Ossokine source (2004.09442, final-remnant section) prints this same
previous-EOB mass fit explicitly.  Its displayed form is

\[
 \frac{M_f}{M}=1-\{[1-E_{\rm ISCO}(a)]\nu
 +16\nu^2[0.00258-\frac{0.0773}{a(1+1/q)^2/(1+1/q^2)-1.6939}
 -\tfrac14(1-E_{\rm ISCO}(a))]\},
\]

which expands exactly to the preceding interpolation because
\(\nu=q/(1+q)^2\) implies \(1-4\nu=(q-1)^2/(1+q)^2\).  The native
implementation's `1 - nu*(1-4*nu)*(1-energy) - 16*nu**2*equal_mass_radiation`
is therefore algebraically identical.

For aligned spins, the source variables reduce to

\[
 a=\frac{m_1^2\chi_1+m_2^2\chi_2}{(m_1+m_2)^2},\qquad
 \chi_i=\frac{m_1^2\chi_1+m_2^2\chi_2}{m_1^2+m_2^2}.
\]

The source conventions differ only in mass-ratio naming: Hofmann writes
\(q=M_2/M_1\le1\), whereas Ossokine writes \(q=m_1/m_2\ge1\).  The native
symmetric expressions are exchange invariant and cover the same aligned-spin
domain: positive component masses and signed projections \(\chi_{1,2}\in[-1,1]\).
The source fit is empirical; it does not supply a separate unequal-mass branch
switch.  The equal-mass replacement is embedded in the smooth \(16\nu^2\)
interpolation, not selected by a runtime equality test.

## Final spin: exact ansatz, blocks, and unresolved selection

Hofmann Eq. (7) gives the aligned/anti-aligned signed projection

\[
 a_f=a_{\rm tot}+\nu[L_{\rm ISCO}(a_{\rm eff})
 -2a_{\rm tot}(E_{\rm ISCO}(a_{\rm eff})-1)]
 +\sum_{i=0}^{n_M}\sum_{j=0}^{n_J} k_{ij}\nu^{2+i}a_{\rm eff}^{j},
\]

with

\[
 a_{\rm tot}=\frac{a_1+a_2q^2}{(1+q)^2},\qquad
 a_{\rm eff}=a_{\rm tot}+\xi\nu(a_1+a_2),
\]

where Hofmann's (q=M_2/M_1\le1), and the ISCO energy/angular momentum are
the Kerr expressions in their Eqs. (2)--(6).  Eq. (11) constrains (k_{00}) by

\[
 \frac{\sqrt3}{2}+\sum_{i=0}^{n_M}\frac{k_{i0}}{4^{2+i}}=0.68646.
\]

Table 1 supplies three separate calibrated blocks:

* \((n_M,n_J)=(1,2)\), \(\xi=0.41616\), with \(k_{01}=-1.2019\),
  \(k_{02}=-1.20764\), \(k_{10}=3.79245\), \(k_{11}=1.18385\),
  \(k_{12}=4.90494\);
* \((3,3)\), \(\xi=0.463926\), with the 16-coefficient block printed in
  the middle of Table 1;
* \((3,4)\), \(\xi=0.474046\), with the 20-coefficient block printed in
  the bottom of Table 1.

The first block has five free polynomial entries plus the constrained (k_{00});
the latter two likewise have their (k_{00}) fixed by Eq. (11).  Hofmann's
text says all three were fitted to the 246 parallel-spin simulations and that
\((1,2)\) is the “optimal choice to avoid overfitting.”  That is a fit-quality
recommendation, not an assertion that SEOBNRv4 selected it.

The SEOBNRv4 paper (1611.03703v2) says only “We use the final spin formula of
Ref. [Hofmann et al.]” and gives no \((n_M,n_J)\), coefficient block, or
“optimal-choice” adoption.  The later Ossokine source likewise says it adopts
the Hofmann fit to agree with SEOBNRv4/SEOBNRv4HM but does not identify one of
Hofmann's three Table-1 blocks.  Consequently, the permitted primary sources do
not establish a unique SEOBNRv4 default.  Choosing `(1, 2)` from Hofmann's
anti-overfitting recommendation, or choosing `(3, 3)`/`(3, 4)` from their lower
reported residuals, would each add an assumption absent from the target paper.

## Native comparison and action

`remnant.py` implements the derived mass variables and interpolation, the Hofmann
Eq. (7) ansatz, the Eq. (11) `k00` constraint, and all three printed Table-1
blocks.  Its explicit required `fit_order` is therefore the source-faithful
behavior available from the papers.  The direct tests cover the mass equation,
Eq. (11) equal-mass constraint, test-particle limits, exchange/batch behavior,
finite-domain validation, and gradients.

No exact source-complete correction was found.  **Do not change `remnant.py` or
add a default test.**  If a future target specification names a block, the
smallest implementation is to pass that explicit `(n_M,n_J)` through the
SEOBNRv4 caller and add equation, batch, and gradient tests against the chosen
Table-1 block.  Until then, making `fit_order` optional would silently select an
unpublished target convention.
