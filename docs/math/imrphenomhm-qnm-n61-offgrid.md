# IMRPhenomHM N61 six-mode QNM off-grid gate

Specification ID: `hm.qnm.n61.offgrid-six-mode/v1`.

## Scope and status

This gate checks the named `n61_main` runtime QNM polynomial artifact against
new Kerr eigenproblem solutions away from its **hypothesized** training grid.
The six fundamental gravitational modes are `(2,2,0)`, `(2,1,0)`, `(3,3,0)`,
`(3,2,0)`, `(4,4,0)`, and `(4,3,0)`. The checked spin set is exactly
`a = {-0.71, 0.37, 0.79}`. It is a selected interior interval, not a
full-domain or near-extremal claim. The associated
[validation receipt](../validation/imrphenomhm_qnm_n61_offgrid_20260927.md)
records the measured outcome.

## Inputs, conventions, and options

The input `a` is oriented, dimensionless remnant Kerr spin. The independent
solver uses remnant mass `M_f=1`, spin weight `s=-2`, overtone `n=0`, and
`exp(-i omega t)`: a physical root has `Re(omega)>0`, `Im(omega)<0`. The
runtime returns `(M_f omega_R, M_f/tau)`, with **positive** angular damping.
For a binary, the public class converts both components to dimensionless
total-mass cyclic frequency with `(M_total/M_f)/(2 pi)`; this gate tests that
conversion on supplied remnant states and does not qualify the remnant fit.

## Mathematics and components

The [London--Fauchon-Jones paper (2019), Eq. (19)](https://arxiv.org/pdf/1810.03550v2)
defines

$$
\kappa(a,\ell,m)=\left[\frac{\ln(2-a)}{\ln 3}\right]^{1/(2+\ell-|m|)}.
$$

Its Eqs. (20) and (21) motivate the zero-damped and damped polynomial
families. The runtime `n61_main` variant has five complex coefficients for
`(2,2)`, `(3,3)`, `(4,4)`, `(4,3)` and seven for `(2,1)`, `(3,2)`:

$$
\widehat\omega_{\ell m0}(a)=
\begin{cases}
m/2+\kappa\sum_{j=0}^{4}c_j\kappa^j,&\text{zero-damped family},\\
\sum_{j=0}^{6}c_j\kappa^j,&\text{damped family}.
\end{cases}
$$

The fit uses the paper's positive-imaginary display convention; the
independent root uses negative imaginary damping. Therefore the complex
comparison is between `omega_fit = omega_R - i/tau` and the solver's root.
No LAL implementation or constants enter the reference.

## Implementation mapping

The [Cook--Zalutskiy Kerr eigenproblem (2014), Eqs. (23)--(56)](https://arxiv.org/abs/1410.7698v2)
supplies the angular spectral problem and radial continued fraction.
[`scripts/generate_hm_qnm_n61_offgrid_reference.py`](https://github.com/xangma/diffgw/blob/main/scripts/generate_hm_qnm_n61_offgrid_reference.py)
loads only `src/torchwave/qnm/solver.py`, with no fit-derived seed, runtime
coefficient import, QNM table, or waveform oracle. It continues the
fundamental branch from a large-`ell` Schwarzschild WKB seed and requires
strict solver convergence plus estimated frequency error below `1e-9`.
The frozen [root fixture](https://github.com/xangma/diffgw/blob/main/tests/data/hm_qnm_n61_offgrid_roots.json)
records each root, angular eigenvalue, radial residual, continued-fraction
depth/error, solver estimate, branch origin, and solver SHA-256. Regeneration
is an explicit maintenance action; ordinary frontend tests read the frozen
roots and fail on missing data or a changed solver hash.

The frozen roots are independent of the fit. The test compares them with
`evaluate_n61_qnm` in both
[`torchwave`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_hm_qnm_n61.py)
and [`jaxwave`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_hm_qnm_n61.py),
then checks the `IMRPhenomHM.mode_qnm_frequencies` adapter with supplied
remnant states. No production coefficients are changed by this gate.

## Evidence and limitations

The three spins are at least `1e-3` from every point of the declared
`a=0.995 sin(theta)`, 61-point grid. That grid and its count are a **project
sampling hypothesis**, not identified historical author training data. For
each mode/spin, the project acceptance bound is
`abs(omega_fit - omega_root)/abs(omega_root) <= 0.01` in complex frequency.
The paper reports sub-1% fractional error for its own printed models (Fig. 1
and text following Eq. (30)); applying this numerical bound to the separate
N61 refit is a declared project criterion, not a claim inherited from that
paper. Torch and JAX FP64 evaluations must agree within `1e-12` relative
complex error and the public total-mass conversion within `2e-12` relative
error after supplied remnant mass/spin injection.

The paper does not specify enough historical fit training data and fitting
details to reproduce the authors' coefficients. The gate establishes bounded
agreement of **this** clean-room N61 approximation with independent Kerr
solutions and native frontend evaluation. It does not establish historical
London coefficient identity, uncertainty relative to numerical relativity,
near-extremal behavior, remnant-provider accuracy, amplitude/phase assembly,
polarizations, or a whole-waveform qualification.

## Reproduce the checks

On a CPU environment with NumPy, SciPy, Torch, JAX and pytest installed:

```bash
PYTHONPATH=src python scripts/generate_hm_qnm_n61_offgrid_reference.py --output /tmp/hm_qnm_n61_offgrid_roots.json
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests pytest -q tests/test_imrphenomhm_qnm_n61_offgrid_math_reference.py --reference-device=cpu -ra
```

Compare the regenerated JSON with the committed fixture before accepting a
solver or dependency update. The test command requires the committed fixture
and does not silently regenerate it.
