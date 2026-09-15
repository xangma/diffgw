# Pass 15: ordered XHM amplitude vetoes

Primary source: García-Quirós et al., arXiv:2001.10914v2, Sec. V.A and
Tables V–VI.

The runtime now applies the source-defined ordering for intermediate-node
selection. The four-value amplitude ordering is checked after endpoint and
inner values are expressed in the common `A/A0` convention. A non-monotonic
sequence removes the higher-frequency inner node. The implementation uses all
four values (`fIns`, `fInt1`, `fInt2`, `fRD`) for this check; the paper does not
explicitly identify the sequence used in the monotonicity check, so the four-value choice is recorded as an
implementation inference rather than a uniquely prescribed rule.

The unconditional Table V parameter rows remain active through
`_published_parameter_amplitude_veto`. The ordered path also applies the
`q > 40`, `chi1 > 0.9` rows to each node still alive at that point. Thus a
prior node removal is never resurrected, and the conditional row can remove
any remaining node independently.

The numeric Table VI checks (`|h~| < 0.2` and `< 0.01`) are intentionally
deferred. Eqs. (4.5)–(4.6) relate `A/A0` and `H_lm`, while Eq. (5.3) fits
`A^Inter/A0`; the veto prose calls the constants raw `|h~|` and does not say
which convention they use. Applying either threshold to the current fit
values would therefore invent a normalization. The runtime leaves these
values available to a later explicitly normalized implementation and does not
activate the special `(3,2)` derivative-removal row.

When a node is removed, the reconstruction retains the surviving conditions:
quartic inverse polynomial for one inner value, endpoint Hermite cubic when
both inner values are removed, and linear when endpoint derivatives are also
removed. If the selected inverse polynomial has a pole, the corresponding
value-only cubic, quadratic, or linear fallback is used. The runtime uses
numerical stationary-point checks over the interval; the parent pass owns that
detector and its integration.

Focused tests cover strict non-activation of unresolved thresholds, ordered
batch masks, conditional high-q behavior, production node retention, and
finite heterogeneous-batch gradients.
