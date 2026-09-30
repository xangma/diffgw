# NRTidalv2 D amplitude and taper provenance audit — 27 September 2026

**Disposition:** No source-backed change to the waveform or its frozen gates.
The default `binary_compatible` D variant remains pinned-binary verified, but
its tidal-amplitude normalization is not derived from the published model.
The explicit `paper_2019` D phase-only variant and the separate Eq. (24)
amplitude component retain their distinct source-scoped contracts. This is a
follow-up to the [26 September qualification receipt](nrtidalv2_amplitude_20260926.md),
not a new qualification claim. No LAL implementation source or constants were
used in this audit.

## Can the historical `1/2` be derived?

[Dietrich et al. (2019), Eq. (24)](https://arxiv.org/pdf/1905.06011v2)
specifies the *frequency-domain* additive tidal amplitude with the prefactor
`sqrt(5*pi*nu/24) * M^2/D_L`; Eq. (23) and
[Kawaguchi et al. (2018), Eq. (2.25)](https://arxiv.org/pdf/1802.06518)
have the same leading normalization. In particular,
`(27/16)*sqrt(160*pi*nu/27) = 9*sqrt(5*pi*nu/24)`, with no extra `1/2`.
Equation (25) adds that correction once to the BBH amplitude and then tapers
the sum. Table II marks tidal amplitude absent for
`IMRPhenomD_NRTidalv2`, present for `IMRPhenomPv2_NRTidalv2`.

The same paper's Appendix A.2, Eq. (40), does contain a `1/2`, in the
stationary-phase conversion of a **time-domain** amplitude. Equations
(40)–(45) develop a different amplitude prescription; the paragraph after
Eq. (45) explicitly says the main-text Eq. (24) prescription was chosen for
implementation. Therefore Eq. (40) does not authorize multiplying Eq. (24)
by a second `1/2` or adding a nonzero tidal amplitude to the paper's D
composition. The frozen default's `M_sec^2/(2 D_sec)` prefactor remains a
historical binary-compatible observation, not a sourced paper coefficient.
For the frozen equal- and unequal-mass cases, the historical/published
component scale ratios are respectively 1.236077446 and 1.248886881
([earlier receipt](nrtidalv2_amplitude_20260926.md)). A single polarization
projection cannot account for both at fixed inclination.

The 2019 paper describes public CoRe numerical-relativity waveforms but does
not identify an independently licensed companion implementation of its
NRTidalv2 D amplitude or full-precision taper fit. A bounded search of the
paper and public author-artifact records found no such artifact. This is an
absence of provenance found in this audit, not proof that none exists.

## Can a published merger-frequency convention explain the taper difference?

[Dietrich et al. (2018), Eqs. (10)–(11)](https://arxiv.org/pdf/1804.02235)
give a merger-frequency fit in `kappa_2^T` and explicitly state that their
LAL implementation substitutes `kappa_eff^T`. Using their Eq. (10),
`Lambda_i = (2/3) k_2^i/C_i^5`, and Dietrich 2019 Eq. (8), independent
algebra gives

```
kappa_2^T   = 3 [X_A^4 X_B Lambda_A + X_B^4 X_A Lambda_B]
kappa_eff^T = (3/13) [(X_A + 12 X_B) X_A^4 Lambda_A
                       + (X_B + 12 X_A) X_B^4 Lambda_B].
```

For the frozen equal-mass `(1.35,1.35;400,400)` case, both couplings are
exactly 75; substitution changes the Eq. (11) merger frequency by **zero**.
For `(1.6,1.2;200,800)` and its exchanged ordering, they are respectively
73.683584220860 and 73.496848840456; using `kappa_2^T` instead of
`kappa_eff^T` changes the printed Eq. (11) frequency by **-594.064 ppm**.
The earlier public-binary diagnostic yielded only **-25.54 ppm** equal-mass
and **-25.14 ppm** unequal-mass *equivalent* boundary shifts, conditional on
the untapered amplitude continuing to agree inside the taper. The published
coupling substitution cannot explain both cases and should not be changed
to fit them.

The papers specify a taper from `f_mrg` to `1.2 f_mrg`. Printed Eq. (11)
coefficients have only four significant digits. At `kappa_eff^T = 75`, moving
one coefficient at a time by half a unit of its last printed digit changes
the merger frequency by up to approximately 100 ppm (`n1`), 7.5 ppm (`n2`),
47 ppm (`d1`), or 36 ppm (`d2`). Thus unpublished precision *could* explain
a nonzero-tide ~25 ppm discrepancy while leaving the zero-tide boundary
unchanged; this is a sensitivity bound, **not** evidence that the pinned
binary uses such precision. No unprinted coefficient was inferred or added.

## Exact remaining gate and next discriminating evidence

The frozen `binary_compatible` default passed unchanged strict `len`
catalog **19/19** and holdout **14/14** with LALSuite distribution 7.26.15;
its focused scalar/backend tests passed **67**, with **18 strict xfails**.
The `paper_2019` D composition differs from the same pinned binary at
nonzero tide, and the default's nonzero-tide high-frequency taper quotient
retains the two strict expected failures: maximum absolute ratio errors
**0.000153757344** equal-mass and **0.000189164604** unequal-mass versus
the unchanged **0.00005** limit. Host, commands, logs, and remaining test
coverage are in the [earlier receipt](nrtidalv2_amplitude_20260926.md).
These failures prevent a claim that the taper is both paper-grounded and
binary-qualified; the D amplitude provenance prevents a paper-grounded
claim for the default despite its binary parity.

The next read-only discriminator is a public-binary, dense-frequency taper
comparison for equal masses at several nonzero `Lambda` values, plus zero
tide, repeated across documented LALSuite releases. Compare each output to
the *printed* Eq. (11) and independently checked Planck window, without
fitting a production coefficient. This can distinguish a version-dependent
or tide-dependent effective boundary from a fixed convention. Promoting a
different coefficient would still require an author-published full-precision
fit or another independently licensed source. Resolving the D amplitude
contract would require a published correction or author artifact that
explicitly assigns a nonzero amplitude and its normalization to D.
