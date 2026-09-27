# MemoryWaveform direct-class dominant-pair memory

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `MemoryWaveform22Direct` |
| Reference composition | `memory-dominant-22-direct-class/v1` |
| Scope | Only Torch/JAX MemoryWaveform.evaluate_22 with supplied physical-strain h22, symmetric mirror, and relative memory offset on selected CPU float64 grids. |
| Domain and modes | `time`; `(2, 0)`, `(4, 0)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 1 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `call_route` | `evaluate_22` | `evaluate_22` | None |
| `mirror` | `equatorial_symmetry` | `equatorial_symmetry` | None |
| `grid_rule` | `gradient_trapezoid` | `gradient_trapezoid` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

`MemoryWaveform22Direct`, reference `memory-dominant-22-direct-class/v1`, covers **only**
`MemoryWaveform.evaluate_22` in Torch and JAX. It consumes a supplied
oscillatory physical-strain $h_{22}(t)$ and assumes its equatorial-symmetry
mirror $h_{2,-2}=\bar h_{22}$. It returns relative $h_{20}^{\rm mem}$ and the
full dominant-pair plus/cross pattern. The legacy class call (`forward` or
`__call__`) is a distinct polarization-only diagnostic proxy and has no
catalog claim. See the [component derivation](memory-dominant-22.md).

## Inputs, conventions, and options

`times` is a one-dimensional, strictly increasing grid in SI seconds with at
least two samples; `h22` is complex dimensionless strain on that grid, at
explicit positive `distance_m` in meters. Inclination is in radians from the
orbital angular momentum. The mode convention is
$h=h_+-ih_\times=\sum h_{\ell m}{}_{-2}Y_{\ell m}$; $h_\times^{\rm mem}=0$.
The first returned memory sample is zero, fixing an integration constant only.
The only choices are `call_route=evaluate_22`,
`mirror=equatorial_symmetry`, and `grid_rule=gradient_trapezoid`; all run
natively in both frontends. No external fit or asset is needed. Autodiff is
checked for selected supplied-mode coefficients and inclination at fixed grid.

## Mathematics and components

Favata [Sec. 6.2, Eq. (21)](https://arxiv.org/pdf/1003.3486v1) gives,
in SI seconds, for the symmetric dominant pair:

$$
h_+^{\rm mem}=\frac{R}{192\pi c}\sin^2\iota(17+\cos^2\iota)
\int|\dot h_{22}|^2dt,\qquad h_\times^{\rm mem}=0.
$$

Projection yields
$h_{20}^{\rm mem}=R\sqrt{5/(6\pi)}\int|\dot h_{22}|^2dt/(7c)$.
The full plus pattern reconstructed from $h_{20}$ retains the $\ell=4,m=0$
contribution. Talbot [Eqs. (1)–(9)](https://arxiv.org/pdf/1807.00990v2)
supplies the strain-mode convention. The numerical rule takes endpoint
secants and unequal-grid quadratic-interpolant interior derivatives, squares
their complex magnitudes, and accumulates adjacent trapezoids.

| Component ID/version | Source | Variant | Status | Scalar reference | Torch | JAX | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `memory.dominant_22.v1` | Favata Eq. (21); Talbot Eqs. (1)–(9) | published dominant pair | executable | `h20_coefficient`, `plus_from_flux_integral` | `compute_memory_h20_22`, `memory_polarizations_22` | same | `tests/test_memory_dominant_22_math_reference.py` |
| `memory.sampled_flux.v1` | Project unequal-grid rule | sampled adapter | executable | `sampled_h20` | `MemoryWaveform.evaluate_22` | same | same |

## Implementation mapping

The independent `diffgw_reference.memory_dominant_22` module uses mpmath and
does not call either frontend. Torch uses `torch.gradient` on actual time
coordinates. JAX explicitly evaluates the same unequal-grid three-point
derivative, including endpoint secants; both integrate trapezoids and project
the full plus pattern. `evaluate_22` returns `(h_plus, h_cross, h20)` and
does not use the constructor's legacy `scale_factor`.

## Evidence and limitations

The strict CPU float64 gate checks the 70-digit angular projection, exact
linear-mode values and derivatives on a four-sample nonuniform grid at four
inclinations, a two-sample minimum-grid case, and one nonlinear complex
polynomial on that grid against an independent scalar sampled evaluator.
It checks Torch and JAX values, zero
cross strain, selected derivatives, and JAX JIT at relative tolerance
`5e-13` and strain absolute tolerance `1e-35`. This is sampled
implementation agreement, not a continuous-domain numerical error bound.
The separate strict expected failure records the legacy proxy's coefficient
mismatch.

Physical accuracy of the supplied $h_{22}$, earlier memory offset, general
mode-pair coupling, precession, nonlinear iteration, other class routes,
same-model external comparison, and numerical-relativity validation remain
unqualified. In particular, no LAL comparison or physical validation is
claimed for this direct-class slice.

## Reproduce the checks

With pytest, mpmath, CPU Torch and JAX available, run
`PYTHONPATH=src python -m pytest -q tests/test_memory_dominant_22_math_reference.py tests/test_memory_diagnostic.py`.
The latter includes one strict expected failure. Catalog and generated-doc
checks run with `tests/test_waveform_definitions.py` and
`python scripts/generate_waveform_library_table.py --check`. A change to the
source law, mode convention, or numerical rule requires a new version and
requalification.
