# Dominant-mode nonlinear memory: provisional component specification

## Scope and status

`memory.dominant_22/v1` is a source-backed, time-domain memory **component** for
a supplied physical-strain oscillatory mode $h_{22}(t)$ and its equatorial
symmetry mirror $h_{2,-2}=\bar h_{22}$. It returns the accumulated $h_{20}$
and the full dominant-pair plus/cross polarizations. It does not synthesize
$h_{22}$, estimate the unobserved earlier memory, or implement general mode-pair
coupling. The direct `MemoryWaveform(times, hp, hc, ...)` carrier remains a
diagnostic proxy; only its explicit `evaluate_22` method has this scope.
This component stays outside the public `WaveformDefinition` catalog.

| Component | State | Qualification boundary |
| --- | --- | --- |
| `memory.dominant_22/v1` | executable | Supplied $h_{22}$, symmetric $-m$ mirror, first-order nonlinear memory, CPU float64 |
| `memory.mode_coupling.v0` | pending | General Talbot $\Gamma$ contraction and all output modes |
| `memory.carrier_h20.v0` | diagnostic | Legacy $1/\sqrt6$ flux proxy disagrees with the published 22 coefficient |

## Inputs, conventions, and options

`times` are strictly increasing SI seconds, with at least two samples and a
common one-dimensional grid. `h22` is complex, dimensionless **physical strain**
at an explicitly supplied positive `distance_m` in meters. It is the mode in
$h=h_+-i h_\times=\sum h_{\ell m}\,{}_{-2}Y_{\ell m}$. The inclination is in
radians, measured from the binary orbital angular momentum. The $-m$ input is
assumed to be the equatorial symmetry mirror; this does not cover precession or
an independent $h_{2,-2}$. The polarization basis is that of Favata's
nonprecessing dominant-pair calculation, with $h_\times^{\rm mem}=0$.

The integration starts at the first time sample, where the returned memory is
zero. That is a relative offset, not a claim that the physical memory was zero
before the supplied waveform. The `scale_factor`, guessed Mpc/meters unit
switch, and strain-magnitude-dependent 100-Mpc branch belong solely to the
legacy carrier. There are no external calibration assets or fits. The
finite-difference derivative uses each actual time coordinate and the integral
uses adjacent trapezoids. Autodiff is with respect to the supplied samples,
amplitude, distance, and inclination for a fixed grid; source-waveform
generation and varying sample count are outside this component.

## Mathematics and components

Talbot, Thrane, Lasky & Lin,
[arXiv:1807.00990v2, Eqs. (1)–(9)](https://arxiv.org/html/1807.00990), define
the flux from $|\dot h|^2$, the pair integral $H$, and angular coefficient
$\Gamma$. For the equatorially symmetric $h_{2,\pm2}$ pair, Favata,
[arXiv:1003.3486v1, Sec. 6.2, Eq. (21)](https://arxiv.org/html/1003.3486),
gives the resulting full pattern (restoring SI seconds with $R/c$):

$$
h_+^{\rm mem}(t,\iota)=\frac{R}{192\pi c}
\sin^2\!\iota\,(17+\cos^2\!\iota)
\int_{t_0}^{t}|\dot h_{22}(u)|^2\,du,
\qquad h_\times^{\rm mem}=0.
$$

With ${}_{-2}Y_{20}=\sqrt{15/(32\pi)}\sin^2\!\iota$, orthonormal projection
of this angular function yields

$$
h_{20}^{\rm mem}(t)=\frac{R}{7c}\sqrt{\frac{5}{6\pi}}
\int_{t_0}^{t}|\dot h_{22}(u)|^2\,du.
$$

The reference also evaluates the angular projection integral directly:
$[96]^{-1}\int_{-1}^{1}{}_{-2}Y_{20}(x)(1-x^2)(17+x^2)\,dx$
$=\sqrt{5/(6\pi)}/7$. Reconstructing the **full** dominant-pair pattern
from $h_{20}$ requires the additional $\ell=4,m=0$ projection:
$h_+^{\rm mem}=h_{20}\,{}_{-2}Y_{20}\,[7(17+\cos^2\!\iota)/120]$.
The bracket is absent from the legacy `memory_polarizations` projection.

The numerical component differentiates $h_{22}$ on the supplied grid and
integrates the resulting squared modulus with trapezoids. Its scalar gate uses
the exact $h_{22}(t)=s(t-t_0)$ case: $h_{20}=R\sqrt{5/(6\pi)}|s|^2
(t-t_0)/(7c)$, so the value and slope derivative have no time-discretization
error. At fixed complex slope $s=s_R+i s_I$, the reference derivative is
$\partial h_{20}/\partial s_R=2R\sqrt{5/(6\pi)}s_R(t-t_0)/(7c)$.

## Implementation mapping

| ID | Source | Scalar reference | Torch | JAX | Maintained gate |
| --- | --- | --- | --- | --- | --- |
| `memory.dominant_22/v1` | Favata Sec. 6.2 Eq. (21); Talbot Eqs. (1)–(9) | `diffgw_reference.memory_dominant_22` | `compute_memory_h20_22`, `memory_polarizations_22`, `MemoryWaveform.evaluate_22` | same names | `tests/test_memory_dominant_22_math_reference.py` |
| `memory.carrier_h20.v0` | Project diagnostic proxy | no published scalar equivalence | `compute_memory_h20` | same name | `tests/test_memory_diagnostic.py` strict expected-failure coefficient comparison |

The independent mpmath reference evaluates Favata's full formula and its
$\ell=2$ projection separately; neither frontend imports it. The new native
methods require a complex mode and an explicit distance. The historical
`forward`/`__call__` API takes only one line-of-sight polarization pair, which
does not determine the required sphere of angular radiation flux. It therefore
cannot be converted into Talbot's arbitrary-mode memory by adjusting a single
normalization constant.

## Evidence and limitations

The maintained gate checks the Favata projection integral at 70 digits, exact
full-pattern reconstruction at four inclinations, the 4-sample nonuniform-grid
linear-mode $h_{20}$ and $h_+$ values, Torch/JAX slope and inclination
autodiff against scalar analytic/mpmath derivatives, and JAX `jit` at CPU
float64. The thresholds are relative `5e-13`, absolute `1e-35` for strain,
and zero absolute tolerance for the nonzero slope derivative. A separate strict
expected failure records the legacy mode-sum coefficient mismatch. These are sampled
implementation-agreement checks, not a uniform numerical error bound.

On `len`, the initial combined focused run returned **5 passed, 1 strict expected failure**
in 6.47 s using Torch 2.14.0+cpu, JAX 0.11.2, SymPy 1.14.0, and mpmath 1.3.0.
The passing gates qualify this component's stated value/derivative cases for
both frontends. The expected failure preserves the unresolved carrier mismatch.

General $\Gamma_{\ell m}^{\ell_1\ell_2m_1m_2}$, non-axisymmetric memory,
precessing sources, nonlinear iteration (“memory of the memory”), physical
accuracy of the supplied $h_{22}$, and prior-history offset remain unqualified.
No LAL same-model comparison or independent physical-data validation is
claimed. The source's full pattern is restricted to the dominant symmetric
pair and cannot be extrapolated to arbitrary input modes or the class's
polarization-only carrier.

## Reproduce the checks

With `mpmath`, `pytest`, CPU Torch and JAX installed, enable JAX float64 and run
`PYTHONPATH=src python -m pytest -q tests/test_memory_dominant_22_math_reference.py`.
Run `PYTHONPATH=src python -m pytest -q tests/test_memory_diagnostic.py` for
the separately selected strict `xfail` on the legacy formula; an unexpected pass
fails that diagnostic suite. The runs use no external asset. Changing the
normalization, mode convention, or source equation requires
a new component version and review of this fixture and the research ledger.
