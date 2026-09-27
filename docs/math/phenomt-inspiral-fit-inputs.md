# IMRPhenomT inspiral fit inputs

## Scope and status

`T-PN-FIT-INPUT-v1` covers the aligned `(2,2)` inspiral's calibrated
TaylorT3 time shift `t0` and five fitted frequency values at
`theta={0.45,0.55,0.65,0.75,0.82}`. With the earlier paper-derived TaylorT3
frequency at `theta=0.33`, these form the six targets of
`T-PN-FREQ-COLLOC-v0`. The source-backed evaluator is conditional on the
pinned MIT author `fits.py` file; committed finite-grid values allow an
unconditional backend regression. This is fit transcription, not a
recalibration or complete waveform qualification.

## Inputs, conventions, and options

The public inputs are ordered solar masses `m1>=m2>0` and aligned
dimensionless spins `chi1,chi2 in [-1,1]`. Set `M=m1+m2`,
`eta=m1*m2/M^2`, `delta=(m1-m2)/M`, `dchi=chi1-chi2`, and
`S=(m1^2 chi1+m2^2 chi2)/(m1^2+m2^2)`. `t0` is in total-mass time `t/M`;
the five fitted values are dimensionless positive `(2,2)` angular
frequencies. The native carrier's `S` is linear in spins. The source-backed
grid uses float64 on CPU and these five cases, in order:

| `m1` | `m2` | `chi1` | `chi2` | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 30 | 30 | 0 | 0 | equal mass, nonspinning |
| 30 | 30 | 0.6 | -0.2 | equal mass, unequal spins |
| 35 | 25 | 0.7 | -0.5 | unequal mass, mixed spins |
| 60 | 6 | 0.9 | 0.1 | high mass ratio, positive primary spin |
| 90 | 5 | -0.8 | 0.6 | `q=18`, anti-aligned primary spin |

The `fits.py` asset is
[`phenomxpy/phenomt/fits.py` at `4121fafa0484f3db2a7352f4eea697faaaf29f65`](https://gitlab.com/imrphenom-dev/phenomxpy/-/blob/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt/fits.py),
SHA-256 `2dfcca659cd50d927b9e164867913a586e9b1372f58ffde424f8bfbd74b56e21`.
It is author software under the repository's MIT license; the full notice is
in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). The fit expressions
are attributed adaptations already present in `diffgw_core/phenomt_fits.py`.
The numerical calibration inputs are not redistributed here.

## Mathematics and components

[Estellés et al., arXiv:2012.11923v1, Eqs. (8)–(11) and Sec. III](https://arxiv.org/pdf/2012.11923v1)
specifies six collocation nodes and the linear system, but does not print
the fit coefficients. The paper says each target was drawn from NR if the
simulation extends to its node, otherwise SEOBNRv4, with
Teukolsky-EOB hybrids at high mass ratio. The pinned author source supplies
the complete algebraic maps `t0(eta,S,dchi,delta)` and
`F_i(eta,S,dchi,delta)` for `i=1..5`. No coefficients are inferred from
native outputs or LAL binaries.

At `theta_0=0.33`, set `t_early=-5/(eta theta_0^8)` and
`theta_shift=[eta(t0-t_early)/5]^(-1/8)`. The first target is the
`T-PN-T3-v1` frequency at `theta_shift`; the other five are the author-fit
values. The six targets feed the independent `T-PN-FREQ-COLLOC-v0` solve.

| Component ID | Source | Status | Reference | Native | Check |
| --- | --- | --- | --- | --- | --- |
| `T-PN-FIT-INPUT-v1` | THM 8–11; pinned MIT author `fits.py` | conditional | hash-checked author AST evaluator and finite-grid fixture | shared `phenomt_fits.py`, Torch/JAX `_coeff` | `test_phenomt_inspiral_fit_inputs_math_reference.py` |

## Implementation mapping

The optional reference evaluator reads the pinned author file by path,
checks its SHA-256, extracts only the two named fit functions from its AST,
discards Numba decorators, and evaluates their arithmetic without importing
native code or the rest of phenomxpy. It checks the committed grid fixture
against the author file. The strict no-network test compares that fixture
with the shared fit functions evaluated as Python floats, Torch tensors,
and JAX arrays. It also passes source-derived `t0` and targets with the
independent TaylorT3 coefficients into the scalar collocation solve and
compares native pseudo-PN coefficients. Neither native frontend imports the
reference or fixture. The pinned author fit is an external source of
calibrated coefficients, not a paper-derived algebraic expression.

## Evidence and limitations

The fixed thresholds are `rtol=2e-11, atol=5e-10` for `t0`,
`rtol=2e-11, atol=2e-12` for fitted frequencies, and
`rtol=2e-8, atol=1e-8` for the conditioned six-node coefficient solve.
No skips are permitted in the strict test. The author-file verifier requires
the exact asset hash and must reproduce each committed value to `2e-13`
relative or `2e-13` absolute. The [validation receipt](../validation/phenomt_inspiral_fit_inputs_20260927.md)
records the actual result and package versions. These checks establish a
finite-grid transcription and assembly only. They do not reconstruct the
NR/SEOBNRv4/Teukolsky-EOB calibration, bound interpolation error between
grid points, or validate full inspiral/modes against physical waveforms.

## Reproduce the checks

On `len`, obtain the pinned author file separately and run
`python scripts/verify_phenomt_author_inspiral_fits.py --author-fits /path/to/fits.py`.
For the maintained no-network gate, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests python -m pytest -q tests/test_phenomt_inspiral_fit_inputs_math_reference.py tests/test_phenomt_inspiral_collocation_math_reference.py --reference-device=cpu -ra`.
The repository-wide `--require-math-reference` option requires collecting
the entire math suite. Any change to the author file/hash or fit mapping
requires a new component version and reviewed fixture regeneration.
