# IMRPhenomT dominant merger: supplied-input specification v1

## Scope and status

This specification implements the updated [IMRPhenomTHM paper,
Eqs. (28)–(32)](https://arxiv.org/html/2012.11923v1). It applies to the
shared aligned `(2,2)` carrier in T, THM, TP, and TPHM. The [earlier T/TP
paper](https://arxiv.org/html/2004.08302v2) uses different collocation nodes
and merger ansätze and is not substituted here.

The executable scalar source is
[`phenomt_merger.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/phenomt_merger.py), version
`phenomt-merger/v1`. It imports no production carrier, fit, or amplitude
module. Its numerical solves use `mpmath`; an optional `SymPy` entry point
exposes the exact symbolic ansätze and their derivatives for audit.

## Inputs, conventions, and options

In total-mass time `u`, let `x=asinh(alpha u)`, where `alpha=2 pi f_damp`
and `omega_RD=2 pi f_ring` are in inverse total-mass units. The paper's
merger ansatz is

## Mathematics and components

### Frequency and phase

`omega(u)=omega_RD [1-sum_{k=0}^4 a_k x^k]`.

The five supplied conditions are `omega` at the inspiral cut, an interior
point, and the peak `u=0`, plus `d omega/du` at the cut and peak. The scalar
source solves all five coefficients at arbitrary precision. Its analytic
derivative is the derivative of that basis; its phase difference is the
high-precision quadrature of `omega`, with no native phase primitive.

For the current dominant carrier, `u_cut=-26.982976386771437/eta` and
`u_mid=-5/(eta*0.95^8)`. The former is an author-carrier join specialization;
the latter is the actual dominant interior node, distinct from the paper's
generic `-25 M` description. The peak slope is supplied by the joined
ringdown branch. Neither QNM rate nor fitted interior/peak target is inferred
from a waveform comparison.

### Amplitude

With `tau=u-u_peak`, the paper's amplitude basis is

`H(u)=b0+b1 tau^2+b2 sech(2 alpha tau)^(1/7)+b3 sech(alpha tau)`.

The four conditions are signed `H` at the inspiral cut `-150 M`, interior
`-25 M`, and peak `0`, plus `dH/du` at `-150 M`. The scalar source solves the
four coefficients at arbitrary precision and provides the analytic slope.
The native strain uses `|H|`; a signed zero can therefore make the observed
amplitude nondifferentiable and produce a mode-phase jump. The checked finite
cases compare magnitudes after solving the signed system.

## Implementation mapping

The PN cutoff value and slope, fitted midpoint and peak, and damping rate
are supplied inputs. The cutoff slope is computed from the existing PN
amplitude and frequency derivatives, so the gate does not independently
validate their PN coefficients. The current native amplitude stores merger
coefficients in `[b0,b3,b2,b1]` order; the reference follows the paper's
`[b0,b1,b2,b3]` order.

## Evidence and limitations

[`test_phenomt_merger_math_reference.py`](https://github.com/xangma/diffgw/blob/main/tests/test_phenomt_merger_math_reference.py)
checks the symbolic forms, scalar collocation identities, and derivatives, then compares
coefficients, frequency, slope, integrated phase, and amplitude at off-node
times in Torch and JAX FP64. It covers three mass/spin cases, including
mass ratio 10. The [len receipt](../validation/phenomt_merger_20260929.md)
records the execution. This is independent equation and join evidence with
supplied inputs; it is not a reconstruction of the calibrated target
surfaces, QNM/remnant fits, higher-mode merger, or whole-waveform accuracy.

The pinned MIT author `phenomxpy` asset remains the source for unprinted
calibrated targets. [THM Sec. III](https://arxiv.org/html/2012.11923v1)
identifies the fit form and training sources but does not print all target
values or surface coefficients. [Berti's Kerr QNM data](https://pages.jh.edu/eberti2/ringdown/)
and [Jiménez-Forteza's remnant fits](https://arxiv.org/html/1611.00332v2)
offer separate primary lineage checks; they do not replace the particular
one-dimensional fits used in this carrier. The existing [provenance ledger](../provenance_ledger.md)
and `THIRD_PARTY_NOTICES.md` retain the MIT adaptation attribution.

## Reproduce the checks

The [len receipt](../validation/phenomt_merger_20260929.md) records the
environment and commands for the scalar and native checks.
