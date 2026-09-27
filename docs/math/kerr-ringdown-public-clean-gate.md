# Kerr 220 opt-in public-clean composition gate

Status: **blocked**. This is a qualification receipt, not a versioned
whole-waveform specification. The existing
[`kerr-220-energy-carrier-conditional/v1`](kerr-ringdown.md) remains valid only
when QNM frequency, damping, and angular factors are supplied. The opt-in
`kerr220_td/fd` public-clean `(2,2,0)` path in Torch and JAX does not yet
meet the independent angular source-of-truth gate. This receipt does not
assess the separate direct `KerrRingdown` class, whose spectral-angular
physical-energy slice is [versioned separately](kerr-ringdown-direct-class.md).

## Scope and status

The check composes the public-clean QNM and angular factors with the causal
TD and Fourier-minus FD polarizations for one physical-energy mode and its
equatorial mirror. It checks two off-axis spins in CPU FP64. It does not
qualify other modes, overtones, excitation, an entire spin domain, or
intrinsic/angle gradients.

## Inputs, conventions, and options

The cases are `(final_spin, final_mass/Msun, distance/Mpc, energy_fraction,
inclination/rad, phic/rad, tc/s)` equal to
`(0.35, 53, 230, 0.017, 1.1, 0.41, 0.002)` and
`(0.7, 92, 430, 0.005, 0.55, -0.27, -0.001)`.
The TD samples are `tc + (-0.001, 0, 0.002, 0.008)` seconds and the FD
samples are `(-130, 0, 35, 110, 270)` Hz. The reference uses the same
unit-sphere angular norm, `exp(+2i phic)` azimuthal dependence, equatorial
mirror, `H(0)=1`, and Fourier kernel `exp(-2 pi i f t)` as the
[conditional carrier](kerr-ringdown.md). A complex phase rotation makes the
reference spheroidal value at `(theta, phi)=(pi/2, 0)` positive real, matching
the public angular evaluator's gauge before any comparison.

## Mathematics and components

The independent 220 frequency comes from the radial continued fraction and
angular eigenvalue equations recorded in the
[scalar QNM qualification](kerr-qnm-scalar-qualification.md). For its complex
oblateness `c=final_spin * M omega`, the angular reference diagonalizes the
spin-weight `-2` operator in a unit-normalized spherical-harmonic basis. It
continues the 220 eigenvalue from `c=0`, then evaluates the positive and
reflected angular factors. This follows Cook and Zalutskiy's
[angular equation, spherical expansion, and reflection symmetry](https://arxiv.org/html/1410.7698v2),
Eqs. (4), (48), and (49)–(57), as detailed in the
[angular projection specification](kerr-angular-projection.md). The
[finite-quality-factor energy amplitude](https://arxiv.org/pdf/gr-qc/0512160v2)
is mapped under the stated convention in the carrier specification, using
the [mode energy flux](https://arxiv.org/pdf/0707.4654v3). The scalar carrier
then computes both polarizations and the exact two-pole Fourier transform.

## Implementation mapping

[`audit_kerr220_public_angular_gate.py`](https://github.com/xangma/diffgw/blob/main/scripts/audit_kerr220_public_angular_gate.py)
uses `diffgw_reference.kerr_qnm`, `diffgw_reference.kerr_angular`, and
`diffgw_reference.kerr_ringdown` without passing native QNM or angular values
into the scalar calculation. It compares the resulting complete TD and FD
pairs with `kerr220_td` and `kerr220_fd` in both
`torchwave.approximants.kerr_ringdown_public_clean` and
`jaxwave.approximants.kerr_ringdown_public_clean`. Its source SHA-256 output
allows a remote source snapshot to be checked against a Git checkout.

The public-clean angular path uses the separate
`kerr_angular_table_convention` implementation. Its absolute `0.01`
Frobenius-series stopping rule and adaptive normalization were inferred from
public binary outputs rather than specified by the angular paper. This is a
compatibility convention; the source-backed spectral projection is a
distinct calculation.

An [angular cutoff ablation](https://github.com/xangma/diffgw/blob/main/scripts/audit_kerr220_angular_cutoff.py)
evaluates the same complex oblateness with a converged Frobenius series and
the independent scalar Galerkin projection. Cook and Zalutskiy's
[power series, Eq. (15b), and spectral expansion, Eq. (49)](https://arxiv.org/html/1410.7698v2)
define the regular angular solution through convergent expansions; neither
defines a first-term-below-`0.01` stop. All ablation values use the same
unit-sphere norm and an equatorial-real phase gauge.

## Evidence and limitations

Run on `len` with the matched source snapshot and Torch/JAX CPU FP64, SymPy,
and mpmath. The QNM error is in dimensionless `M omega`; angular errors are
absolute complex-value errors. TD/FD errors are relative L2 norms over both
polarizations and the sample grids above.

| Spin | QNM error | Angular `ell_max=10` versus `12` | Public angular `S_+` error | Public angular mirror error | TD error | FD error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.35 | `3.9e-16` | `3.5e-18` | `6.1e-5` | `1.6e-5` | `1.68e-4` | `1.98e-4` |
| 0.7 | `4.2e-16` | `3.0e-14` | `5.35e-4` | `1.6e-6` | `9.44e-4` | `9.47e-4` |

Torch and JAX agree with each other to approximately `1e-16` on these
comparisons. Their agreement does not resolve the independent angular mismatch.
The mismatch remains after gauge alignment and exceeds the reference's
angular truncation change by orders of magnitude. The ablation isolates the
inferred angular-series stop and the norm computed from that stopped shape:

| Spin | Full series/full norm vs. scalar spectral, max relative | Stopped series/full norm, max relative | Full series/stopped norm, max relative | Stopped series/stopped norm, max relative |
| --- | ---: | ---: | ---: | ---: |
| 0.35 | `2e-15` | `2.51e-4` | `3.79e-4` | `3.70e-4` |
| 0.7 | `4.4e-15` | `5.14e-4` | `6.93e-4` | `9.13e-4` |

The three angles are `1.1`, `pi-1.1`, and `pi/2` radians. At spin `0.35`, the
first included term below `0.01` is index 2 at all three angles; at `0.7`
it is index 3, 2, and 3, respectively. The stopped/full norm ratios minus
one are `+3.7936e-4` and `-6.9287e-4`. The stopped-series/stopped-norm
values reproduce the separate inferred public-compatible evaluator to
`<4e-16` relative, while the full-series/full-norm values agree with the
independent equation solver near rounding precision. Thus the independent
spectral solution needs no angular correction to explain this gap. The stop
and its normalization are numerical compatibility choices, not a published
angular equation or physical normalization. Consequently no
versioned public-clean whole-waveform claim or catalog promotion is made.

To clear this gate, select and document a source-backed public angular
variant or provide published support for the current convention, then rerun
independent QNM–angular–carrier TD/FD checks, a fresh pinned binary holdout,
and the applicable gradient gates for its declared inputs. Changing the
public-clean convention may alter binary parity and must be assessed separately.

## Reproduce the checks

In a checkout with the required scalar and frontend dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  python scripts/audit_kerr220_public_angular_gate.py
```

The script freezes the exact cases in source and reports source hashes and
per-backend errors as JSON. The matched `len` snapshot used
`/home/xangma/.cache/diffgw-xp-pycbc-venv/bin/python`.
For the separate angular ablation, run
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python
scripts/audit_kerr220_angular_cutoff.py` with Torch, SymPy, and mpmath.
