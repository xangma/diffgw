# Pass 3: common tidal EOB metric and tortoise dynamics

This bounded repair used native TorchWave code and freshly retrieved primary
papers. No LAL/PyCBC source, GPL implementation port, target binary, oracle
output, or residual fit informed it. TEOBResumS remains a prototype API name.

| Primary paper | Fresh PDF SHA256 |
|---|---|
| Damour–Nagar, [0704.3550v2](https://arxiv.org/pdf/0704.3550v2) | `9c20b4d0835c1ea4eccd65b1677f65a1054371570ed02de0b6ac5a6c473af352` |
| Nagar et al., [1806.01772v2](https://arxiv.org/pdf/1806.01772v2) | `d5efb16e4bfc3852aa443879dba5c49dd0460b4812fee7f58e26b50b82bc8ebc` |

Damour–Nagar Eqs. (2)–(3) define the 3PN series and `D=A*B`; the adjacent
Padé prescription gives `A=P^1_3[A_3PN]`, `D=P^0_3[D_3PN]`. Eq. (6) gives
`xi=dr/dr*=sqrt(A/B)=A/sqrt(D)` and `p_r*=xi*p_r`. Eq. (7) places `p_r*^2`
outside `A` and includes `A*z3*p_r*^4/r^2`, `z3=2*nu*(4-3*nu)`, with the
paper's stated 3PN truncation of the quartic tortoise transformation.

The tidal class now shares the native uncalibrated 3PN point-mass metric.
For its additive `A=A0+At`, it retains `D=A0*B0`, computes `B=D/A`, and uses
`xi=A/sqrt(D)`. This replaces the inverted `D`, incomplete Taylor metric,
and artificial potential floor. The same quadratic and quartic radial
kinetic corrections are propagated through its Hamiltonian.

Nagar et al. Eqs. (6), (10), and (23) confirm the tortoise kinetic structure,
Hamilton equations, and additive tidal potential. Their calibrated model
has further point-mass, spin, centrifugal-radius, and tidal structures absent
here. The inherited simple tidal amplification, quadrupole defaults, spin
terms, merger-frequency coefficients, and waveform taper remain explicitly
identified as heuristics/prototypes. No independent tidal correction to `D`
is added. Removing the floor does not extend the physical dynamical domain
through a zero of `A`; use is restricted to positive `A` outside the horizon.

Conservative derivatives reuse the native EOB coordinate-partial calculation,
retaining incoming state and tidal gradients and supporting inference tensors.
Intrinsic parameters broadcast together before deriving their combinations;
otherwise a batch introduced only by deformability acquired an extra batch
axis when conservative coordinates were expanded. The dissipative wrapper
uses these corrected conservative derivatives. No radiation-reaction or
waveform-calibration formula is changed.

The shared initial-condition wrapper now detects gradient requests through
all tensor attributes of the supplied Hamiltonian and torque, including
tidal deformability and spin-induced quadrupole inputs. Its prior fixed
mass/spin list detached the complete solve when only an extension parameter
required gradients. The underlying circular solve and radial-momentum
formula are unchanged by this adjacent propagation repair.

## Validation

The new focused test file independently checks nonzero-tide `A*B` against the
printed Padé `D`, the tortoise metric identity, canonical radial mass shell
including the 3PN quartic term, all three conservative Hamilton equations,
and finite-difference state/deformability Jacobians. It also covers gradients
requested only through tides, shared scalar coordinates with mass/tidal
parameter batches, and construction inside `no_grad`/`inference_mode`.
Zero-tide agreement supplements these independent identities.
Complete initial-condition Jacobians for deformability-only and
quadrupole-only inputs agree with finite differences in scalar and batched
cases; inference contexts still return detached results.

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_clean_room_teob_potential.py tests/test_teobresums.py tests/test_clean_room_eob_conservative.py
```

Result: **29 passed**, including the previously failing zero-tide metric
test and the existing tidal waveform smoke check. These results establish
the repaired metric/kinetic identities, not TEOBResumS waveform parity.
