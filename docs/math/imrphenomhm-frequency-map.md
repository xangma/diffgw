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
uses by default $0.014 f^{\rm RD}_{\ell m}/f^{\rm RD}_{22}$ for amplitude and
$0.018 f^{\rm RD}_{\ell m}/f^{\rm RD}_{22}$ for phase, following the distinct
[PhenomD carrier joins in Khan et al., Eqs. (37)--(38)](https://arxiv.org/html/1508.07253v2).
The component reference evaluates either map for a supplied $f_0$. Its
component tests accept that input; the public selection gate additionally
checks the assembly's actual cutoff choice. These expressions use
geometric-frequency normalization and do not apply directly to hertz.

The public D-carrier HM option `hm_mapping_transitions` selects only the two
outer-map cutoffs. `phenomd_runtime` (default) retains `(0.014, 0.018)` for
amplitude and phase, respectively; `london_2018` selects the paper-printed
`(0.018, 0.014)`. Both pairs multiply
$f^{\rm RD}_{\ell m}/f^{\rm RD}_{22}$. The opt-in variant does not change
the PhenomD carrier's own Eqs. (37)--(38) joins, QNM fits, PN amplitude,
phase offsets, or projection. It is available for the public Torch/JAX
`IMRPhenomHM` D-carrier path, including `get_fd_waveform`; a non-D carrier
with `london_2018` is rejected rather than assigned a D-specific Eq. (8)
interpretation. This is the bounded component `hm.fmap.selection.london2018.v1`,
not a whole-waveform paper-fidelity claim.

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

For the public selection gate, the scalar reference computes the two cutoffs
from supplied QNM frequencies without importing either frontend. CPU FP64
calls cover all six supported positive-mode labels, `(2,2)`, `(2,1)`, `(3,3)`,
`(3,2)`, `(4,4)`, and `(4,3)`, at two remnant-producing mass/spin inputs:
`(m1,m2,chi1,chi2)=(35,20,0.2,-0.1)` and `(50,10,0.7,-0.3)` in solar-mass
units. Each mode and input uses both `phenomd_runtime` and `london_2018`.
The public class call captures the two map inputs and output arrays at both
cutoffs and at ringdown. The native QNM result is supplied to the independent
scalar map; this gate does not derive or audit the QNM fit. The frozen checks
require relative map-value error at most $2\times10^{-11}$, one-sided slope
error at most $2\times10^{-6}$ using a $10^{-6}$ geometric-frequency step,
and unchanged default complex mode values to $10^{-13}$ relative error when
`phenomd_runtime` is passed explicitly. The slope test uses points on each
side of a join, not autodiff at the join itself. `(2,2)` additionally checks
the identity map for both selections. These thresholds test selection and map
algebra, not the full carrier or physical calibration.

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

The public D-carrier selection test checks both named choices for all six
modes and both mass/spin inputs through Torch and JAX class calls. It compares
captured map frequencies and one-sided slopes with the independent scalar
evaluator and checks that explicit `phenomd_runtime` reproduces omitted-option
complex modes. A `(3,3)` dispatcher check retains public `get_fd_waveform`
coverage; unknown choices and non-D London carriers remain rejected. It does
not establish independent correctness of the full complex HM mode. The
[len validation receipt](../validation/imrphenomhm_mapping_selection_20260927.md)
records the integrated CPU FP64 result.

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
