# IMRPhenomXPNR published components, version 1.0.0

## Scope and status

This is a **research component specification**, not a qualified full
IMRPhenomXPNR waveform. `xpnr-published-components-sympy-1.0.0` independently
encodes the published algebra as 37 executable SymPy expressions with mpmath
evaluation. Components that require a transition spin, calibration value,
boundary track, or carrier input accept it explicitly. The current public
Torch/JAX waveform remains provisional; see the [family ledger](../waveforms/research/imrphenomxpnr.md).

## Inputs, conventions, and options

Frequencies `f`, `f0`, `f1`, and `fc` are dimensionless `Mf`; angles and spins
are dimensionless. `m1 >= m2`, the spin vectors `chi_i = S_i/m_i²`, and unit
`Lhat` must refer to the same caller-supplied transition frame. `eta`, `chi`,
`c = cos(theta_LS)`, `q`, boundary values and derivatives, and ringdown
parameters are explicit scalar inputs. `f0 < f1`, positive frequencies, and a
nonzero base beta at the final-rescaling join are preconditions. The zero-spin
map sets `c=0` as a numerical extension because the published angle is
undefined there. Piecewise derivatives mean derivatives *within* a branch;
join checks use the appropriate one-sided limits.

The optional reference needs SymPy and mpmath. Native attachment kernels take
Torch/JAX tensors, support float64 CPU values and autodiff, and do not import
the reference. The tests exercise JAX JIT and vmap and Torch broadcasting.
No inherited `A/B/Bf` tensor is an input to the independent test expectations.

## Mathematics and components

| Component ID | Pinned primary equations | Reference expressions | Native implementation | Status |
| --- | --- | --- | --- | --- |
| `XPNR-MAP/1` | [XPNR](https://arxiv.org/html/2507.02604v1) Eqs. 3–6 | `map.*` | Existing Torch/JAX precession PNR map | Conditional on transition state |
| `XPNR-FIT/1` | [XO4a](https://arxiv.org/html/2312.10025v1) Eq. 35; XPNR Eq. 22 | `calibration_polynomial`, `fit.eta_frozen`, `fit.chi_frozen` | Existing PNR fit evaluators | Conditional on independently sourced coefficients |
| `XPNR-MR/1` | XO4a Eqs. 25–34; XPNR Eqs. 15–16 | `fit.a*`, `fit.b*`, `fit.beta_rd`, `angles.*` | Existing PNR angle kernels | Algebra executable; fit assets conditional |
| `XPNR-ATTACH/1` | [Final Twist](https://arxiv.org/html/2107.08876v1) Eqs. 41–45, 56–58, 62; XO4a Eq. 37 and denominator correction; XPNR Eqs. 17–21 | `attach.*` | `pnr_attachment.py` in Torch/JAX | Algebra executable; full boundary construction pending |
| `XPNR-CARRIER/1` | XPNR Eqs. 7–8, 10–13 | `carrier.*` | None for XPNR carrier modification/asymmetry | Conditional on `u_k`, calibrated carrier and asymmetry inputs |
| `XPNR-HMMAP/1` | XO4a Eqs. 50–52 | `higher_mode.frequency_map` | Existing PNR higher-mode mapping | Executable component |
| `XO4A-WINDOW/1` | XO4a Eq. 38 | `window.endpoint_complete` | Existing PNR window | Explicit endpoint-completing interpretation |

The bound uses Final Twist Eq. 62 with `p=0.002`, algebraically evaluated as
`pi/2 + (beta-pi/2) [atan(t)/t]^p`, where
`t = ((beta-pi/2)/(pi/2)^(1-p))^(1/p)` and its center limit is `beta`.
This even integer power form avoids a distribution term when differentiated
symbolically. Native kernels use equivalent small/large forms to stay finite
in float64. The public waveform still clips beta rather than calling this
standalone bound.

## Implementation mapping

The versioned specification is
[`xpnr_symbolic.py`](../../src/diffgw_reference/xpnr_symbolic.py). The
independently written float64 scalar forms are
[`xpnr_angles.py`](../../src/diffgw_reference/xpnr_angles.py). Native
attachment forms live in the [Torch](../../src/torchwave/approximants/pnr_attachment.py)
and [JAX](../../src/jaxwave/approximants/pnr_attachment.py) modules. The
reference accepts caller-supplied carrier deviations and asymmetry amplitudes;
neither native frontend composes the XHM-CP deviations, adjusted ringdown, or
dominant-mode asymmetry into the public route. Endpoint-completing window
behavior is an explicit interpretation of XO4a Eq. 38, not a literal copy of
its printed trigonometric argument.

## Evidence and limitations

The maintained [symbolic test](../../tests/test_xpnr_symbolic_reference.py)
checks all 37 expressions against separately constructed scalar expectations,
exact rational polynomial coefficients, four join values/slopes, and high
precision derivative stability. Eleven isolated attachment kernels pass
value and gradient comparison in Torch and JAX on CPU float64; JAX JIT/vmap
and Torch batch behavior are also checked. On 2026-09-29, the focused XPNR
suite passed 164 tests on `len` with SymPy 1.14.0, mpmath 1.3.0, Torch
2.14.0+cu130, JAX 0.11.2, and LALSuite 7.26.15. Four default-option LAL
binary/PyCBC optimized plus-polarization comparisons passed their declared
`match >= 0.99` threshold over 20–400 Hz; they are diagnostics only.

The transition sampling event and complete SpinTaylor integration, beta
input tracks and landmark/root decisions, 80-waveform fit targets and
licensed tensor release, XHM-CP `u_k` mapping, and dominant-mode asymmetry
remain unavailable from the examined public primary assets. The unchanged
strict inherited XHM/XPHM binary catalog and holdout gates record failures.
No DiffGW-to-NR physical comparison is claimed. Equation and native component
passes do not qualify the public waveform.

## Reproduce the checks

On `len`, from `/tmp/diffgw-xpnr-qualification-332f`, run:

```sh
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 /tmp/diffgw-xpnr-portability-0172-pypi-clean/.venv/bin/python -m pytest -q tests/test_xpnr_symbolic_reference.py tests/test_xpnr_attachment_math_reference.py tests/test_xpnr_component_evidence.py tests/test_xpnr_attachment_gate.py tests/test_xpnr_coefficient_provenance.py tests/test_imrphenom_xpnr.py
```

The optional binary cases skip without LALSuite/PyCBC. The published
expression version must change if an equation, branch convention, or input
meaning changes; recheck the independent scalar expectations and both native
frontends before updating the [machine-readable qualification](../waveforms/research/imrphenomxpnr-qualification.json).
