# IMRPhenomXHM inspiral carrier rescaling

## Scope and status

This research component specifies only the analytic rescaling of the
IMRPhenomXAS (22) phase used by the IMRPhenomXHM higher-mode inspiral. The
separate [amplitude closure](imrphenomxhm-inspiral-amplitude.md) records the
earliest remaining carrier mismatch. Its
stable component IDs are
`xhm.phase.inspiral.carrier-rescaling.v1` and
`xhm.phase.inspiral.carrier-coefficients.v1`; both are executable in the
independent scalar reference, Torch, and JAX.

This is not a complete IMRPhenomXHM inspiral phase and is not a public catalog
entry. In particular, it excludes the complex-PN-amplitude argument, the
high-mass-ratio fitted replacement, the calibrated linear attachment term,
the mode phase constant, the intermediate phase, ringdown phase, and (32)
mode mixing.

## Inputs, conventions, and options

The mode index (m) is a positive integer. The direct helper receives the
dimensionless (22) carrier phase already evaluated at the scaled frequency
(2f/m). The coefficient helper receives the integer PN index (i) and the
real (22)-phase coefficients (c_{22}^{i}) and (d_{22}^{i}). Inputs may be
scalar or broadcast arrays. Native output preserves the input floating dtype;
Torch gradients and JAX transformations remain supported.

The component uses the Fourier and positive-frequency mode conventions of
García-Quirós et al. It does not choose the additive phase or linear time
gauge. Those choices are deliberately outside this component.

## Mathematics and components

The source is García-Quirós et al.,
[arXiv:2001.10914v2](https://arxiv.org/html/2001.10914v2), Eq. (29) and
Appendix D, Eqs. (61)--(63). The carrier contribution is

\[
  \phi^{\mathrm{carrier}}_{\ell m}(f)
  = \frac{m}{2}\,\phi_{22}\!\left(\frac{2f}{m}\right).
\]

For a (22)-phase term

\[
  x^{(i-5)/3}\left(c_{22}^{i}+d_{22}^{i}\log x\right),
\]

the algebraically equivalent mode coefficients are

\[
  c_m^i=\left(\frac m2\right)^{(8-i)/3}
  \left[c_{22}^i-d_{22}^i\log\!\left(\frac m2\right)\right],\qquad
  d_m^i=\left(\frac m2\right)^{(8-i)/3}d_{22}^i.
\]

| Component ID/version | Source | Status | Scalar reference | Torch | JAX |
| --- | --- | --- | --- | --- | --- |
| `xhm.phase.inspiral.carrier-rescaling.v1` | Eq. (29) | Executable | `rescaled_carrier_phase` | `rescale_carrier_phase` | `rescale_carrier_phase` |
| `xhm.phase.inspiral.carrier-coefficients.v1` | App. D Eq. (63) | Executable | `rescaled_coefficients` | `rescale_carrier_phase_coefficients` | `rescale_carrier_phase_coefficients` |

## Implementation mapping

The independent SymPy construction is in
`diffgw_reference.imrphenomxhm_inspiral_carrier`. It proves term by term that
direct frequency rescaling and the Appendix-D coefficient transformation are
identical. The native implementations are
`torchwave.approximants.xhm_inspiral_phase_clean` and
`jaxwave.approximants.xhm_inspiral_phase_clean`. The production XHM evaluators
route their existing carrier multiplication through these helpers; neither
native frontend imports or evaluates the scalar reference.

The current production attachment constants and all fitted phase surfaces
remain in `imrphenom_xhm.py`. Routing the multiplication through the clean
helper is therefore an equation-level refactor, not a new full-phase model.

## Evidence and limitations

Source transcription is checked by exact SymPy simplification for modes
(m=1,2,3,4) and five representative PN indices. Torch and JAX values are
checked independently against the scalar expressions in float64, including
broadcasting, invalid inputs, a Torch gradient, and JAX JIT.

The maintained black-box diagnostic
`scripts/diagnose_xhm_frozen_components.py` separately evaluates the exact
frozen catalog and holdout parameters mode by mode. With lal 7.7.1,
lalsimulation 6.2.1, Torch 2.14.0+cu130, and NumPy 2.3.5, the (22) mode agrees
to numerical precision after a constant phase convention. Higher-mode
inspiral phase residuals after a diagnostic affine gauge removal are between
(4.5\times10^{-5}) and (1.33\times10^{-4}) rad in those two cases. This is
external numerical-oracle evidence, not part of the symbolic proof and not a
license to infer or fit missing coefficients.

The same diagnostic finds much larger nonlinear residuals in higher-mode
intermediate phase regions (6.24--28.80 rad) and intermediate amplitude
relative L2 errors (0.77--7.04). Inspiral amplitude errors also remain as high
as 0.48 in the catalog (32) mode. Consequently this component carries no
claim for `XHM-INS-PHASE`, `XHM-INS-AMP`, intermediate, mixing, ringdown,
projection, or complete-waveform parity. The frozen full-polarization gates
remain failed.

The exact regional values, environment inventory, and earliest `(3,2)`
amplitude finding are frozen in the
[component receipt](../validation/xhm_carrier_components_20260925.json).

In the same latest-package environment, the broader XHM regression selection
passes 106 tests and skips 5, but its two existing Torch
`no_grad`/`inference_mode` connection checks fail on an absolute difference of
about (7.3\times10^{-25}). The failures reproduce unchanged from an untouched
`github/main` checkout, so they are recorded as a baseline/environment
reproducibility blocker rather than evidence for or against this component.

## Reproduce the checks

Run the independent and native component checks with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_xhm_inspiral_carrier_phase.py
```

With the optional pinned LAL Python bindings installed, reproduce the
mode/regime diagnostic without writing a repository artifact:

```bash
PYTHONPATH=src python scripts/diagnose_xhm_frozen_components.py \
  --output /tmp/diffgw-xhm-frozen-components.json
```

Review the JSON's version block, case parameters, native transition
frequencies, and separately labeled amplitude, complex-shape, and phase
metrics. A change to the equations or conventions requires a new component
version and corresponding scalar-reference tests; changing only diagnostic
package versions requires recording the new versions with the evidence.
