# Pass 16 — XHM amplitude-veto candidate disposition

The candidate implementation of the `(2,1)` numeric `0.2` veto was rejected
after independent clean-room review and parent validation. The final runtime
and pass-15 veto tests are byte-for-byte unchanged from pass 15.

## Why the candidate was rejected

The primary paper states that the veto compares the Fourier strain amplitude
in geometric units, while the native reconstruction carries fitted quantities
in normalized amplitude conventions. Multiplying the intermediate fit only by
`A0`, and reconstructing the inspiral value with the proposed common factor,
was not independently established as the complete conversion required by the
published threshold. The independent review records the unresolved mapping in
[pass-16-veto-independent-review.md](pass-16-veto-independent-review.md).

The candidate also left the source-described `0.01` ringdown and monotonicity
stages incomplete. Its direct reduced-collocation algebra tests passed, but
that did not validate the missing physical normalization.

Parent validation found that the candidate failed the existing XHM parity
smoke gate with a match of `0.976882` against the required `0.98`. Removing
the pass-15 four-value monotonicity convention produced the same failure;
restoring that documented convention gave `0.980638`. The convention is kept
with its existing explicit status because the paper requires a monotonicity
stage but does not identify the exact sequence.

## Final state

No candidate runtime or test code remains. The primary-source audits are kept
as evidence for the unresolved work. No acceptance threshold, parity test,
calibration coefficient, or frozen verifier was changed.
