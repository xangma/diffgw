# IMRPhenomHM frequency-map component

## Scope and status

This research specification covers the piecewise map from a positive higher-mode
geometric Fourier frequency $f=M f_{\rm physical}$ to the $(2,2)$ carrier
frequency. `hm.fmap.eq5.v1` is the model-independent London map used for the
amplitude. `hm.fmap.phase.eq8.v1` uses the same first two branches and the
IMRPhenomD-specific phase ringdown branch. These are executable components in
the [provisional higher-mode ledger](../waveforms/research/phenom-higher-modes.md),
not a public whole-waveform definition or a verified IMRPhenomHM claim.

The source is [London et al., arXiv:1708.00404v4](https://arxiv.org/pdf/1708.00404v4),
Eq. (5) and the paragraph immediately below it (page 2), with the phase
replacement in Eq. (8) (page 3). No LAL source, QNM coefficients, or
calibration data enter this component reference.

## Inputs, conventions, and options

Inputs are $f>0$, $f_0>0$, $f^{\rm RD}_{\ell m}>f_0$,
$f^{\rm RD}_{22}>0$, and a positive integer $m$. All frequencies have the same
dimensionless $Mf$ units. The QNM frequencies are supplied by the caller; this
component does not specify their fit, remnant convention, or numerical value.
The paper sets $f_0^A=0.018 f^{\rm RD}_{\ell m}/f^{\rm RD}_{22}$ and
$f_0^\phi=0.014 f^{\rm RD}_{\ell m}/f^{\rm RD}_{22}$. These are London's
printed higher-mode mapping cutoffs. The public Torch and JAX HM assembly
currently uses $0.014 f^{\rm RD}_{\ell m}/f^{\rm RD}_{22}$ for amplitude and
$0.018 f^{\rm RD}_{\ell m}/f^{\rm RD}_{22}$ for phase, following the distinct
[PhenomD carrier joins in Khan et al., Eqs. (37)--(38)](https://arxiv.org/html/1508.07253v2).
The component reference evaluates either map for a supplied $f_0$; its tests
do not validate the public assembly's choice of cutoff. These expressions use
geometric-frequency normalization and do not apply directly to hertz.

The paper's Eq. (5) applies to a generic carrier. For the IMRPhenomD phase,
Eq. (8) replaces only the $f>f^{\rm RD}_{\ell m}$ branch. We therefore keep
the two maps as separate component IDs. At each join the lower-frequency
branch owns the value and derivative, following the text after Eq. (5).

## Mathematics and components

Write $R=f^{\rm RD}_{\ell m}$, $Q=f^{\rm RD}_{22}$, and
$s=(Q-2f_0/m)/(R-f_0)$. Then Eq. (5) is

$$
g_5(f)=\begin{cases}
2f/m, & f\le f_0,\\
2f_0/m+s(f-f_0), & f_0<f\le R,\\
f-(R-Q), & f>R.
\end{cases}
$$

Eq. (8) changes the final branch to $g_8(f)=Qf/R$ for $f>R$ and
otherwise uses $g_5$. Both joins are $C^0$: $g(f_0)=2f_0/m$ and
$g(R)=Q$. Their one-sided frequency derivatives are $(2/m,s)$ at $f_0$
and $(s,1)$ at $R$ for Eq. (5), or $(s,Q/R)$ at $R$ for Eq. (8).
Neither paper equation generally makes the map $C^1$. For $m=2$ and $R=Q$,
all branches reduce exactly to $g(f)=f$.

## Implementation mapping

The independent SymPy expressions and mpmath evaluator are in
[`diffgw_reference.imrphenomhm_frequency_map`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomhm_frequency_map.py).
The Eq. (5) amplitude map binds to
[`Torch frequency_mapping_amplitude`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_hm.py)
and [`JAX frequency_mapping_amplitude`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_hm.py).
The D-carrier phase variant binds to the corresponding
`frequency_mapping_phase` function in each frontend. The reference passes
QNM frequencies as independent scalar inputs and imports neither frontend.
The component-to-source-to-code map is `COMPONENT_BINDINGS` in the reference
module, checked by the maintained test.

## Evidence and limitations

The [reference tests](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomhm_frequency_map_math_reference.py)
check the exact symbolic joins and slopes, a hand-reduced rational example,
Torch/JAX values on both sides of every join, one-sided frequency derivatives,
and gradients with respect to $f$, $f_0$, $R$, and $Q$. Synthetic QNM inputs
exercise all six supported $(\ell,m)$ labels, including $(4,3)$, without
copying a QNM fit. This establishes the stated frequency-map algebra and
frontend implementation agreement at the sampled CPU FP64 points. It does
not audit QNM fits, phase-offset attachment, mode amplitude, polarizations,
LAL agreement, or physical accuracy. The higher-mode ledger retains those
gates.

## Reproduce the checks

With reference, Torch, JAX, and test dependencies installed, run the focused
check and then the repository's full strict reference gate:

```sh
python -m pytest tests/test_imrphenomhm_frequency_map_math_reference.py \
  --reference-device=cpu -ra
python -m pytest tests/test_waveform_definitions.py tests/test_*math_reference*.py \
  --require-math-reference --reference-device=cpu -ra
```

The CI math-reference command discovers this file by its name; the existing
strict skip/deselection and tolerance gates remain unchanged.
