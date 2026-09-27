# NRTidalv2 spinning paper-D composition through the merger taper

## Scope and status

Specification **`nrtidalv2-spinning-paper-d-composition/v1`** checks the
dominant-mode, aligned-spin `amplitude_variant="paper_2019"` option relative
to each frontend's existing IMRPhenomD carrier. It composes the independent
scalar [tidal phase](nrtidalv2-phase.md),
[self-spin](nrtidalv2-self-spin.md), and
[merger/taper](nrtidalv2-amplitude-taper.md) references. The frozen gate
crosses both printed taper boundaries with positive deformabilities and
nonzero spins, for Torch and JAX CPU FP64. It is a conditional
printed-equation qualification, not a full waveform or physical validation.

## Inputs, conventions, and options

[Dietrich et al. (2019), Eqs. (17)–(21), (25)–(29), Table II](https://arxiv.org/html/1905.06011)
give the tidal phase, self-spin phase, Planck window, and D/Pv2 model
distinction. [Dietrich et al. (2018), Eq. (11)](https://arxiv.org/pdf/1804.02235)
gives the printed merger-frequency fit, with the paper's stated
`kappa_eff^T` substitution. [Wade et al. (2014), Eq. (5)](https://arxiv.org/pdf/1402.5156v1)
gives the mass-weighted deformability.

## Mathematics and components

With `x=(pi M_sec f)^(2/3)`, the tested quotient is

\[
\frac{h^{\rm BNS}_{+,\times}(f)}{h^{\rm BBH}_{+,\times}(f)}
=W(f;f_{\rm mrg})\exp\left[-i\left(
\psi_T(f)+S_2(f)+S_3(f)\right)\right].
\]

The paper-D option has no Eq. (24) tidal amplitude and excludes the Pv2
3.5PN self-spin/cubic-spin component (Table II). The carrier quotient
removes the inherited BBH amplitude and phase while checking the actual
complex BNS composition in both polarizations. No alignment, amplitude
rescaling, or parameter fit is permitted. The gate uses
`phasing_convention="coalescence"`, so the separate native
reference-frequency subtraction is outside this slice.

## Implementation mapping

The independent mpmath/SymPy modules
`diffgw_reference.nrtidalv2_phase`,
`diffgw_reference.nrtidalv2_self_spin`, and
`diffgw_reference.nrtidalv2_amplitude` compute the right-hand side.
The [maintained test](../validation/nrtidalv2_paper_d_composition_20260927.md)
compares it against Torch/JAX `IMRPhenomD_NRTidalv2` divided by each
frontend's `IMRPhenomD`. Neither frontend imports these references.

## Evidence and limitations

### Frozen systems and limit

The three systems `(m1,m2;chi1,chi2;Lambda1,Lambda2)`, in solar masses,
are `(1.35,1.35;0.20,-0.12;350,350)`,
`(1.6,1.2;0.08,-0.12;250,800)`, and the body-exchanged copy of the
second. Distance is 100 Mpc, inclination 0.4 rad, coalescence phase and
time zero. For each case, the scalar printed-fit merger frequency defines
the Hz grid: 100, 800, then
`f/f_mrg = 0.75, 0.99, 1, 1.025, 1.05, 1.1, 1.15, 1.195, 1.2, 1.25`.
The absolute complex quotient limit is `1e-10` at every sample for both
polarizations; the two native polarization quotients must also agree within
that limit. The strictly post-endpoint quotient must be exactly zero.
An ulp-sized mismatch between independently rounded merger frequencies is
allowed at the nominal `1.2 f_mrg` endpoint under the same limit.

### Interpretation and open conflicts

The [execution receipt](../validation/nrtidalv2_paper_d_composition_20260927.md)
records the strict result. This extends the earlier component gates across
the taper as a single complex composition check. It does not establish
agreement with a public binary implementing **exactly** this paper-D
contract: the pinned LALSuite 7.26.15 D binary has a nonzero-tide amplitude,
contrary to Table II, and its nonzero-tide effective taper boundary differs
by about 25 ppm from the printed fit
([source audit](../validation/nrtidalv2_provenance_20260927.md)).
No binary gate is asserted for this option. Further, the paper's Fig. 7
example for `Lambda=350` conflicts with its printed Eqs. (28)–(29);
this gate follows the numbered equations and cannot resolve which
quadrupole relation was intended. The historical
`binary_compatible` default and its existing gates are unchanged.

## Reproduce the checks

With the reference, Torch, JAX, and pytest dependencies installed, run
`PYTHONPATH=src:tests JAX_ENABLE_X64=1 python -m pytest -q tests/test_nrtidalv2_paper_d_composition_math_reference.py`.
