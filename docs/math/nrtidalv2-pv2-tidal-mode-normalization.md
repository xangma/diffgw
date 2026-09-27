# NRTidalv2 Pv2 tidal-amplitude mode normalization

## Scope and status

Specification **`nrtidalv2-pv2-tidal-mode-normalization/v1`** covers the
conversion of the published signed Eq. (24) *face-on plus-strain* amplitude
into the positive-frequency `(2,-2)` input of the Pv2 rotation. It composes
the already-gated aligned finite-size term and positive-22 projector at a
supplied set of Euler angles. It does not qualify a complete native
`IMRPhenomPv2_NRTidalv2` waveform.

## Inputs, conventions, and options

The signed complex face-on finite-size term is
`T=A_T W exp[-i(psi_T+S_2+S_3+S_3.5)]`, with the masses, distance, spin,
tidal deformability, and frequency conventions of the
[aligned additive-term specification](nrtidalv2-pv2-aligned-additive-terms.md).
The target is the positive-frequency `h^L_{2,-2}` supplied to the
[Pv2 projector](nrtidalv2-pv2-positive-22-projection.md). The angles
`alpha`, `gamma`, `beta`, `theta_JN`, and `zeta` are supplied independently;
this gate does not predict their BNS dynamics. CPU FP64 and positive
frequencies are in scope.

## Mathematics and components

[Dietrich et al. (2019), Eqs. (22)–(25)](https://arxiv.org/html/1905.06011)
obtain Eq. (24) by preserving the low-frequency coefficient of Eq. (22).
Immediately below Eq. (22), the authors say they substitute luminosity
distance for the *effective distance* of their cited source. That source,
[Hotokezaka et al. (2016), Eqs. (15), (17), and (24)](https://arxiv.org/html/1603.01286),
defines the Fourier quantity as the plus polarization and puts detector and
inclination response in the effective distance. Thus, the Eq. (24) coefficient
has face-on plus-strain normalization. The introductory description of a
dominant 2,2 mode in Dietrich et al. does not itself fix its mode-to-strain
normalization; the cited derivation does.

For the [Pratten et al. (2020), Eqs. (11)–(13)](https://arxiv.org/html/2004.06503)
positive-22 projector, zero Euler and basis-rotation angles give

\[
h_+(\theta=0)=\tfrac12\sqrt{5/(4\pi)}\,h^L_{2,-2},\qquad
h_\times(\theta=0)=-i h_+(\theta=0).
\]

Consequently the source-normalized aligned tidal input is

\[
\Delta h^L_{2,-2}=\frac{2T}{\sqrt{5/(4\pi)}}.
\]

At zero Euler and basis angles, arbitrary inclination has the independent
closed form `Delta h_plus=T(1+cos(theta)^2)/2` and
`Delta h_cross=-i T cos(theta)`. The complex sign of `T` is retained.

| Component ID/version | Source | Variant | Status | Scalar reference | Torch/JAX boundary | Check |
| --- | --- | --- | --- | --- | --- | --- |
| `nrtidalv2-pv2-tidal-mode-normalization/v1` | Dietrich (2019), Eqs. (22)–(25); Hotokezaka (2016), Eqs. (15), (17), (24); Pratten (2020), Eqs. (11)–(13) | Published Pv2 finite-size term, supplied angles | Conditional executable | `nrtidalv2_pv2_projection.mode_from_face_on_strain` and `project_positive_22` | Pv2 `face_on_to_positive_22` and positive-22 projectors | `test_nrtidalv2_pv2_tidal_mode_normalization.py` |

## Implementation mapping

The scalar reference uses mpmath and the previously versioned aligned-term
reference. Torch and JAX each implement the conversion in their native Pv2
driver, which also uses it for its existing BBH face-on carrier. The
maintained test feeds the independently evaluated finite-size term and the
native Eq. (24)/phase term through that same conversion and each native
projector. No LAL source, binary residual, or fitted conversion enters.

## Evidence and limitations

The frozen systems are `(1.35,1.35;chi=0.20,-0.12;Lambda=350,350)`,
`(1.6,1.2;0.08,-0.12;250,800)`, and the exchanged version, all at 100 Mpc.
Frequencies are 100 Hz, 800 Hz, `1.05 f_mrg`, and `1.25 f_mrg`. Two angle
sets are used: zero rotation with `theta_JN=0,0.4,1.1` for the analytic
polarization limit, and nondegenerate precession/basis angles as frozen in
the test. Both complex polarizations must agree with the scalar reference
to `1e-10`, normalized pointwise by the nonzero `|A_T W|`; the post-taper
term must be exactly zero. No alignment or output rescaling is permitted.
The [receipt](../validation/nrtidalv2_pv2_tidal_mode_normalization_20260927.md)
records the result.

This gate fixes the Eq. (24) normalization *at the supplied-mode boundary*.
It does not implement the native Pv2 BNS carrier, prove the paper's tidal
amplitude fit against NR, determine spin-dependent precession angles or
reference-frequency anchoring, resolve the merger-frequency/source Fig. 7
conflict, or establish binary parity. The existing D paper and historical
binary-compatible variants remain distinct.

## Reproduce the checks

With mpmath, Torch, JAX, and pytest installed on CPU FP64:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_nrtidalv2_pv2_tidal_mode_normalization.py
```
```
