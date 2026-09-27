# IMRPhenomHM six-mode projection component

## Scope and status

`hm.projection.london-eq1.v1` is a bounded, research-only projection gate for
the six aligned-spin IMRPhenomHM modes. It checks the angular conversion of
public positive-frequency complex modes into complex plus/cross polarizations.
The intrinsic mode carrier, QNM inputs, LAL agreement, and physical accuracy
are outside this component.

## Inputs, conventions, and options

The modes are `(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)` at positive Fourier
frequency. Inclination is in `[0,π]`; native `phi0` is an observer azimuth,
`phic` is the orbital phase, and `tc` is in seconds. The frontends use the
positive-frequency time factor `exp(-2πiftc)`. This gate uses FP64 and the
default PhenomD carrier. It treats each public `return_unprojected_modes`
value as an input, so no intrinsic mode formula is assumed.

[London et al., arXiv:1708.00404v4, Eq. (1)](https://arxiv.org/pdf/1708.00404v4)
write the time-domain strain as `h = h₊ − i h× = Σ hₗₘ {}₋₂Yₗₘ`.
For aligned, nonprecessing binaries the equatorial-reflection relation is
`hₗ,₋ₘ(t)=(-1)^ℓ h*ₗₘ(t)`. London Eq. (1) alone does not specify a Fourier
kernel. [García-Quirós et al., arXiv:2001.10914v2, Sec. II,
Eqs. (2), (4)–(9)](https://arxiv.org/html/2001.10914v2) explicitly use
`exp(-2πift)`: at positive `f`, their carrier is
`Nₗₘ(f)=FT[hₗ,₋ₘ](f)`. The existing HM public carrier has the equivalent
positive-`m` adapter `Pₗₘ=(-1)^ℓ Nₗₘ`, with the HM azimuth argument opposite
the physical azimuth used in those equations. This identifies the convention
being checked; the gate does not infer it from London's Eq. (1) alone.

## Mathematics and components

Construct the harmonics independently from the Wigner small-`d` matrix:

`Yₗₘ(ι,φ) = {}₋₂Yₗₘ = (-1)^m sqrt((2ℓ+1)/(4π)) d^ℓ_{m,2}(ι) exp(imφ)`.

The normalization gives
`Y₂₂=sqrt(5/(64π))(1+cos ι)² exp(2iφ)`. Write
`p=(-1)^ℓ` and evaluate both harmonics at the native `phi0`. The coefficients
multiplying each public unprojected mode are

`D₊ = (Yₗₘ + p Y*ₗ,₋ₘ)/2`,
`D× = -i(Yₗₘ - p Y*ₗ,₋ₘ)/2`.

Thus `h₊,ₗₘ=D₊ Pₗₘ` and `h×,ₗₘ=D× Pₗₘ`; the full polarizations are the
sum over the requested modes. Equivalently, García-Quirós's negative-`m`
carrier formula at physical azimuth `−phi0` gives these expressions after
`Nₗₘ=p Pₗₘ`. This equality fixes both the odd-ℓ sign and the cross sign.
The mode's orbital phase is `exp(im phic)` and the common time shift is
`exp(-2πiftc)`. At face-on only the two `m=2` modes, `(2,2)` and `(3,2)`,
survive in this six-mode set; edge-on and generic angles exercise both members
of each pair.

## Implementation mapping

The independent SymPy Wigner-`d` evaluator lives in
`src/diffgw_reference/imrphenomhm_projection.py`. The public Torch and JAX
`IMRPhenomHM` implementations return unprojected modes and polarizations;
their private harmonic helpers are not inputs to the reference gate.
`tests/test_imrphenomhm_projection_math_reference.py` compares each mode and
the six-mode sum at source angles and nonzero phase/time shifts.

## Evidence and limitations

The focused CPU FP64 gate covers six per-mode complex plus/cross contributions
and their sum at face-on, edge-on, and two generic inclinations with nonzero
azimuth, orbital phase, and time shift. It separately checks the `m=2` pole
selection, odd-ℓ orientation, azimuth/phase/time covariance, body-exchange
`(-1)^m` parity, and the equal-mass/equal-spin odd-mode zero. The reference
builds all harmonics with SymPy Wigner matrices and evaluates them with
50-digit mpmath; it imports no Torch/JAX projection helper.
[The focused len validation](../validation/imrphenomhm_projection_20260927.md)
passed 13/13 strict math-reference cases on both frontends, with no skips.

This component is conditional on the public positive-frequency carrier
adapter above. It checks source harmonic normalization, the six angular
coefficients, and their assembly with live carriers. It cannot qualify the
carrier amplitude, phase, fit inputs, negative-frequency reconstruction,
detector response, or a complete waveform. No whole-model verification claim
is implied by passing this gate.

An independent [public-binary polarization holdout](../validation/imrphenomhm_lal_polarizations_20260927.md)
compares complete plus/cross from both frontends with pinned LALSuite at three
unseen generic orientations and phase/time settings. That sampled binary gate
does not promote this source-projection component to a whole-model claim.

## Reproduce the checks

With SymPy, mpmath, Torch, JAX, and pytest installed, run
`PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q tests/test_imrphenomhm_projection_math_reference.py`.
