# IMRPhenomNSBH joining frequencies

## Scope and status

Specification **nsbh-joining-frequencies/v1** transcribes the joining
frequencies in [Thompson et al. (2020), Table I](https://arxiv.org/pdf/2002.08383v1).
It fixes the grid and limits before numerical native comparison. Inputs are
the independently supplied $f_{\rm tide}>0$, $f_{\rm RD}>0$,
$\widetilde f_{\rm RD}>0$, $q\ge1$, torus ratio $T\ge0$, and the
already qualified Table I $\epsilon_{\rm ins}$. No remnant or QNM provider
is certified by supplying these values.

## Inputs, conventions, and options

The merger classification uses the **unadjusted** $f_{\rm RD}$ and the
closed high-frequency boundary $f_{\rm tide}\ge f_{\rm RD}$; $T>0$ means
torus.

## Mathematics and components

The paper prints:

| Class | $\widetilde f_0$ | $\widetilde f_1$ | $\widetilde f_2$ |
| --- | --- | --- | --- |
| high, no torus | $\widetilde f_{\rm RD}$ | $\widetilde f_{\rm RD}$ | $\widetilde f_{\rm RD}$ |
| high, torus | $\epsilon_{\rm ins}\widetilde f_{\rm RD}$ | $\epsilon_{\rm ins}\widetilde f_{\rm RD}$ | $\widetilde f_{\rm RD}$ |
| low, no torus | $[(q-1)\widetilde f_{\rm RD}+\epsilon_{\rm ins}f_{\rm tide}]/q$ | $[(q-1)\widetilde f_{\rm RD}+f_{\rm tide}]/q$ | — |
| low, torus | $\epsilon_{\rm ins}f_{\rm tide}$ | $f_{\rm tide}$ | — |

Here “high” means $f_{\rm tide}\ge f_{\rm RD}$; “low” means strict
$f_{\rm tide}<f_{\rm RD}$. A dash supplies no paper value and cannot be
qualified numerically. The native carrier's disabled ringdown still assigns
$\widetilde f_2=\widetilde f_{\rm RD}$ in low classes; that inert value is
outside this source gate.

## Implementation mapping

An independent mpmath scalar evaluator implements the four printed rows.
The 30 controlled inputs and 180 composition inputs are exactly the grid in
[the branch-factor specification](nsbh-branch-amplitude-fits.md), including
equality and all four merger classes. The scalar joining reference consumes
the independent scalar branch factors. Torch and JAX CPU FP64 consume their
native branch factors. For each paper-defined frequency, 70- and 90-digit
scalar evaluations must differ by at most $10^{-30}$ absolutely. Maximum
native relative error must be at most $2\times10^{-13}$ for a row to pass.
No limit may be enlarged after comparing native output.

## Evidence and limitations

The existing native low/no-torus $\widetilde f_0$ applies an additional
outer $\epsilon_{\rm ins}$ to the **whole** mixed expression. Table I does
not print that factor. Keep the binary-compatibility adapter unchanged in
this slice and report the discrepancy as a bounded diagnostic, with a
nonzero controlled-grid example. Thus the other three rows may qualify,
but the low/no-torus $\widetilde f_0$ and the four-row joining-frequency
component cannot yet be called paper-qualified. Its $\widetilde f_1$ can
still be checked against the printed row. This policy separates a measured
source mismatch from a silently adjusted specification.

The printed signs of Appendix A Eqs. (26) and (29) also differ from the
continuity-preserving native adapters. They are **excluded** until a
published-source variant is established; this gate cannot infer a correction
from binary output. Windows, amplitude assembly, and the whole waveform
remain provisional.

The [variant provenance gate](nsbh-amplitude-variant-provenance.md) compares
the final 2020 revision with its cited 2015 amplitude model and records why
these exceptions still cannot be promoted to paper-qualified amplitude gates.

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src pytest -q
tests/test_nsbh_joining_frequencies_math_reference.py` with reference, Torch and JAX extras
on CPU. The [validation receipt](../validation/nsbh_joining_frequencies_20260927.md)
records measured errors, diagnostic, and environment.
