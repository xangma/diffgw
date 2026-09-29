# SpinTaylor reference-frequency angle qualification (2026-09-29)

**Scope:** Native Torch/JAX `evaluate_spintaylor_precession`, convention 1, the
37.3 Hz reference case. This is a component qualification. It does not qualify
full IMRPhenomXPHM-SpinTaylor waveform fidelity.

## Public derivation and numerical choice

[Colleoni et al. (2025), §II.B, Eqs. (2.18)–(2.23)](https://arxiv.org/pdf/2412.16721)
define the SpinTaylor initial state in the source frame at `f_ref`, evolve the
spin and orbital-direction ODEs, then obtain Euler angles from the evolved
orbital direction in a J frame. [Pratten et al. (2021), Appendix C,
Eqs. (C4)–(C17)](https://arxiv.org/pdf/2004.06503) fixes the J frame and its
reference gauge for XPHM convention 1:

- `z_J = J_ref/|J_ref|`; `x_J` is the normalized projection of the line of
  sight onto the plane orthogonal to `z_J`; `y_J = z_J × x_J`.
- `alpha_ref = atan2(L_ref · y_J, L_ref · x_J) = pi - kappa` modulo `2 pi`,
  `beta_ref = acos(L_ref · z_J)`, and `gamma_ref = pi - phi_J`.
- Minimal rotation gives `d gamma = -cos(beta) d alpha`.

The ODE must start at the state defined at `f_ref`. Changing the requested
output grid while holding `f_ref` and the state fixed cannot change the
continuum solution at a shared frequency. The implementation now integrates
from `f_ref` toward both 15 Hz and the inspiral transition, then samples the
same trajectory for any output grid. It uses 100 backward and 200 forward RK4
nodes with linear interpolation in PN velocity. Those node counts, the 15 Hz
floor, branch unwrapping, and interpolation are numerical choices, not
prescriptions from the papers. Frequencies below 15 Hz and singular viewing
geometries (`N || J` or `L_ref || J`) are outside this qualification.

## Reproduction and component evidence

Host `len`; scratch checkout `/tmp/diffgw-spintaylor-cleanroom-20260929`;
PyCBC environment `/home/xangma/venv-pycbc/bin/python`. Inputs: `(m1,m2) =
(36,29) Msun`, spins `(0.3,0.2,-0.1)` and `(0.1,-0.2,0.2)`, `inclination=0.6`,
`phi_ref=0`, `f_ref=37.3 Hz`, float64. Baseline SHA
`85032f41153d3f9ec8ba79cc1e66042c94a228cb`:

| Result | Grid `[37.3, 80]` | Grid `[20, 37.3, 80]` |
| --- | ---: | ---: |
| Baseline `alpha(80)` | 2.334345669365079 | 2.149787802812899 |
| Baseline `alpha(37.3)` | 0 | 0.016311446380131 |
| Corrected `alpha(80)` | 3.908742114401342 | 3.908742114401342 |
| Corrected `alpha(37.3)` | 1.574396445033033 | 1.574396445033033 |

Torch and JAX give zero observed common-frequency difference for all three
angles on these two grids (at float64 output precision). Their results agree
within `3e-13` on this case. An independent 60-digit mpmath vector-projection
calculation, saved in
[`scripts/qualification/spintaylor_reference_geometry.py`](../../scripts/qualification/spintaylor_reference_geometry.py),
gives `L_ref=0.88950649216544358283`, `alpha_ref=1.57439644503303349055`,
`beta_ref=0.12616479751384345551`, and
`gamma_ref=2.95148672230356904348`. The native reference angles agree within
`2e-15`. This geometry check is conditional on the native model's declared
3PN `L_ref` prescription; it does not independently qualify those PN
coefficients. The Python tests check these values, grid invariance, backward
orbital tilt, and finite nonzero Torch/JAX angle gradients. On `len`, the new
focused test file passed 3/3, the legacy SpinTaylor file passed 7/7, and
the two selected Torch/JAX angle and waveform parity tests passed 2/2. A
two-binary JAX batch agreed exactly with separate evaluations for all three
angles. The len environment has Torch `2.13.0+cu130`, JAX `0.4.20`, SciPy
`1.13.0`, and NumPy `1.26.4`; the test runner temporarily supplied
`scipy.linalg.tril/triu` from NumPy before importing that JAX version.

## Whole-waveform comparison and boundary

Black-box LALSuite: `lal 7.4.1`, `lalsimulation 5.3.1`, with
`IMRPhenomXPHM`, `prec_version=320`, `final_spin_mod=2`, `df=0.25 Hz`,
`20–300 Hz`, `f_ref=37.3 Hz`, `100 Mpc`, phase 0, and the same binary above.
The native Torch and JAX waveforms were compared directly to LAL's complex plus/cross
polarizations on the same frequency bins, with no time, phase, or amplitude
optimization. Both backends give relative complex L2 errors of **0.253636**
(plus) and **0.255445** (cross); Torch amplitude-only L2 errors are 0.039654 and 0.041749.
Even a fitted constant complex scale leaves errors 0.238858 and 0.239574.
This is a failed whole-waveform fidelity gate; the angle reference fix does
not establish full LAL parity. The three older 20 Hz PyCBC tests use optimized
overlap and do not measure this direct complex error.

The published equations do not specify this native implementation's RK4 node
counts, interpolation, branch convention at singular angles, or an API
sentinel such as `f_ref=0`. No LAL implementation source or prior
qualification implementation was consulted to fill those gaps. Further
whole-waveform differences require a separate public-equation derivation and
component isolation.
