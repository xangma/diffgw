# Clean-room pass 23: TEOBResumS radial radiation reaction

## Source equation

The primary TEOBResumS publication, Nagar et al., arXiv:1806.01772v2, writes

\[
\frac{d p_{r_*}}{dt}=-\left(\frac{A}{B}\right)^{1/2}
\frac{\partial \hat H_{\rm EOB}}{\partial r},\qquad
\frac{d p_\varphi}{dt}=\hat{\mathcal F}_\varphi,
\]

and explicitly states that the radial radiation-reaction force
\(\hat{\mathcal F}_{r_*}\) is set to zero. The existing generic TorchWave
EOB reaction instead added \(F_{r_*}=(p_{r_*}/p_\varphi)F_\varphi\) to every
model. That is a model-level equation mismatch for TEOBResumS, independent of
any oracle waveform residual.

`TEOBResumSRadiationReaction` now retains the published azimuthal torque and
returns an identically zero radial force. The generic `EOBRadiationReaction`
prescription remains unchanged for the other EOB prototypes.

## Checks

`tests/test_clean_room_teob_potential.py::test_teob_quasicircular_radiation_has_only_azimuthal_force`
checks both the source equation \(F_{r_*}=0\) and the independent energy/torque
identity \(\nu\Omega F_\varphi=-\mathcal F\) over two positive frequencies.

## Frozen public binary measurement

The strict fixture uses `m1=m2=1.4`, zero spins/tides, `f_lower=30 Hz`,
`f_final=500 Hz`, a 0.5 Hz grid, and `torch.set_num_threads(1)`. The public
`lalsimulation.SimInspiralFD` output is used only as a binary comparison
oracle. Relative to the immediately preceding TEOB baseline (the independent
initial-frequency correction), the result was measured with the same harness:

| polarization | complex L2 before | complex L2 after |
|---|---:|---:|
| plus | 1.0009161561 | 1.0009161561 |
| cross | 0.9993295254 | 0.9993295254 |

The frozen frequency-domain metric is unchanged at the displayed precision:
the low-frequency radial force is numerically negligible for this fixture.
The model remains a prototype with uncalibrated potential, flux, merger, and
ringdown pieces, so this equation correction does not establish full
TEOBResumS parity.
