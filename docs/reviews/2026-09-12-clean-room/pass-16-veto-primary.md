# Pass 16 — primary-source audit of IMRPhenomXHM amplitude vetoes

**Result:** the paper supports a raw Fourier-domain interpretation for the `0.2`
test, but does not specify the normalization of the `0.01` test. It also does
not specify a four-point sequence or a monotonicity direction. Therefore the
pass-15 four-point bidirectional monotonicity rule is **unresolved** as a
primary-source claim.

## Scope and provenance

The source used here is García-Quirós et al., arXiv:2001.10914v2 (revised 3
November 2020): [abstract/version page](https://arxiv.org/abs/2001.10914v2),
[PDF](https://arxiv.org/pdf/2001.10914v2), and the corresponding [arXiv source
archive](https://arxiv.org/src/2001.10914v2). The downloaded source archive has
SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`.

The primary TeX file `hmc2.tex` has SHA-256
`5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.
The ancillary `anc/ParspaceFits/README` has SHA-256
`368184fb645eb93dc04963902c61bd37fd6be8be69be1842abed7994c20684f4`; it
establishes the provenance of the parameter-space fit assets but adds no veto
semantics. No author notebook, implementation port, or generated source is
used as evidence.

Line references below are to `hmc2.tex` in that v2 source archive.

## Amplitude conventions in the paper

The paper defines the modeled positive-frequency Fourier mode by
`\tilde h_{\ell-m}=A_{\ell m}e^{-i\Phi_{\ell m}}`, with non-negative Fourier
amplitude `A_{\ell m}` (lines 283–293). It separately introduces the
frequency-dependent normalized quantity
`\mathcal H_{\ell m}=|A^{SPA}_{\ell m}|/A^0_{22}` (lines 633–644), and the
inspiral fit is written in terms of `\mathcal H` (lines 647–650). Thus
`A`, `A/A^0_{22}`, and `\mathcal H` are not interchangeable by definition.

For the inspiral `(2,1)` veto, the paper explicitly says to remove points where
the Fourier-domain amplitude `|\tilde h_{21}|` is below `0.2`, “in geometric
units” (line 669). The intermediate `(2,1)` prose repeats the same raw
Fourier-amplitude notation for the `0.2` test (line 900), and Table VI applies
it at `f_Int1`/`f_Int2` for `q<8` (line 945). The strongest supported reading
is therefore raw `|\tilde h_{21}|`/`A_{21}` in the paper's geometric-unit
convention, not an explicitly stated `A/A^0_{22}` or `\mathcal H` threshold.

The `0.01` rule is only described as “the amplitude at the ringdown cutting
frequency” being below `0.01` (line 900); Table VI repeats “Amplitude at
`f_RD < 0.01`” without defining the quantity (lines 947–948). The same
section defines the intermediate fit as `A^{Inter}_{\ell m}/A_0`, with
frequency-independent `A_0=\pi\sqrt{2\eta/3}` (lines 830–836). The source
never states whether the `0.01` comparison is raw `|\tilde h|`, that
`A^{Inter}/A_0` fit quantity, `\mathcal H`, or another converted value.
Consequently the exact threshold convention for `0.01` is **unresolved**.

## Points and monotonicity

For inspiral, the three points are explicitly
`[0.5 f_Ins, 0.75 f_Ins, f_Ins]` (lines 655–659). For `(2,1)`, after the
`0.2` removals, the paper says to test whether the three amplitude values form
a monotonic sequence and otherwise remove the middle point (line 669).
The direction (increasing or decreasing) is not stated.

For the intermediate region, the two points are explicitly
`f_Int1=f_Ins+(f_RD-f_Ins)/3` and
`f_Int2=f_Ins+2(f_RD-f_Ins)/3` (lines 830–836). The `(2,1)` and `(3,2)` prose
then says that, after the preliminary checks, the intermediate collocation
points must form a monotonic sequence; on failure it removes `f_Int2` (lines
900–904). Table VI clarifies that this check runs only if both `f_Int1` and
`f_Int2` survived, and removes `f_Int2` on failure (lines 938–950).

The source does not say that `f_Ins` or `f_RD` participates in this
intermediate monotonicity check, and it gives no monotonicity direction. It
therefore does not support substituting the four-point sequence
`[f_Ins,f_Int1,f_Int2,f_RD]` or requiring either direction by an explicit
bidirectional comparison. The pass-15 four-point bidirectional rule is
**unresolved**: it is not contradicted by the paper's underspecification, but
it is not supported as a source-derived rule.

## Ordered removal behavior stated by the source

For inspiral `(2,1)`, the stated sequence is: remove every collocation point
whose `|\tilde h_{21}|` is below `0.2` for `q<8`; then test the three surviving
values for monotonicity and remove the middle point if the test fails (line
669). Each removed point zeros one pseudo-PN coefficient, highest order first;
removing all three removes the pseudo-PN terms (line 667). The paper does not
say to re-run the monotonicity test after each threshold removal.

For intermediate amplitudes, Table VI says the vetoes are “sorted in order of
application” (line 938), giving this order:

1. `(2,1)`: at `f_Int1`/`f_Int2`, remove each point below `0.2` in the `q<8`
   region (line 945).
2. Ringdown amplitude below `0.01`: for `(2,1)`, remove both intermediate
   points; for `(3,2)`, remove both points and the boundary derivatives (lines
   947–948).
3. If both intermediate points passed the preceding checks, test monotonicity
   for `(2,1)` and `(3,2)`; on failure remove `f_Int2` (line 950).
4. Apply the bad-`a_\lambda` veto for `(3,3)`/`(4,4)` at
   `q>7,\chi_1>0.95`, removing both intermediate points (line 952).
5. Apply the parameter-space vetoes in Table V (Table V is lines 912–933):
   unconditional `(2,1)` removal for `\eta<0.23,\chi_1>0.7,\chi_2<-0.5`;
   unconditional `(3,2)` removal for
   `q>2.5,\chi_1<-0.6,\chi_2>0` or
   `\chi_1<-0.9,\chi_2<-0.9`; and the `q>40,\chi_1>0.9` rows only when the
   two points are still alive (lines 920–929). These rows remove both points.
6. Finally check whether the resulting denominator crosses zero; if it does,
   remove boundary derivatives and lower the polynomial order by relaxing
   boundary conditions iteratively (lines 893–909 and 957).

Removing intermediate points switches to a lower-order inverse polynomial,
with minimum order one (line 893). The paper does not provide a missing
threshold conversion that would make the numeric tests executable in a
different amplitude convention.

## Read-only comparison with native TorchWave

After deriving the source facts above, the native helper documents that the
numeric thresholds are intentionally inactive because raw `A` versus `A/A0`
is unresolved (`src/torchwave/approximants/imrphenom_xhm.py:86–93`). It then
uses a full four-value bidirectional check (`:99–107`), explicitly describing
that as an implementation choice because the paper does not identify the
sequence. The production call site treats the intermediate fits as
`A^{Inter}/A0` (`:1006–1011`). These are useful statements of the current
implementation state, but the primary paper does not certify the four-point
rule or the missing threshold conversion. This audit makes no runtime or test
changes.
