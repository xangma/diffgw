# Pass 2: EOB conservative prototype

This bounded repair used the inherited TorchWave files and fresh primary
papers. No LAL/PyCBC source, implementation port, target binary, oracle result,
or residual fit informed the implementation. The result is an uncalibrated
nonspinning 3PN conservative baseline, not the calibrated SEOBNRv4 Hamiltonian.
The inherited spin couplings remain a prototype and are labeled accordingly.

## Primary sources

| Paper | Fresh PDF | SHA256 |
|---|---|---|
| Damour–Nagar, PRD 76, 044003 (2007) | [0704.3550v2](https://arxiv.org/pdf/0704.3550v2) | `9c20b4d0835c1ea4eccd65b1677f65a1054371570ed02de0b6ac5a6c473af352` |
| Buonanno–Damour, PRD 62, 064015 (2000) | [gr-qc/0001013v2](https://arxiv.org/pdf/gr-qc/0001013v2) | `c7746eb232e14fc4479ec2aec84152fb646a069956537d4e8d014954ccb87f7a` |

Damour–Nagar Eqs. (2)-(3) specify the 3PN series for `A` and `D=A*B`.
The following paragraph and footnote prescribe `P^1_3[A]` and `P^0_3[D]`.
Solving the Padé coefficient equations gives the implemented rational forms;
they contain no calibration constants. Eq. (6) gives `xi=sqrt(A/B)`.
Eq. (7) puts `pr_star^2` outside `A` and includes
`A*2*nu*(4-3*nu)*pr_star^4/r^2`. Its stated truncation of the quartic
tortoise transformation is retained. Eqs. (8)-(11) give the conservative
equations with momenta divided by the reduced mass and time divided by total
mass. The code's `H_real/M` equals the paper's `nu*H_hat`.

The prior implementation inverted `D`, omitted its 3PN term and the quartic
momentum term, and multiplied the quadratic tortoise momentum by `A`.
Its potential floor also removed the effective horizon. These are corrected.
Local coordinate clones retain their incoming graph while creating separate
nodes for Hamiltonian partial derivatives; detaching them had discarded
state dependence in those derivatives.

Buonanno–Damour Eq. (4.1) defines circular motion by the radial Hamiltonian
derivative. Eq. (4.13) gives `rdot=F_phi/j'(r)`; Eqs. (4.14)-(4.16)
relate small radial velocity to momentum. Differentiating the circular
condition gives `j'=-H_eff,rr/H_eff,rj`. Transforming to tortoise momentum
then gives `pr_star=F_phi/(j' * xi * H_eff,prpr/H_real)` at zero radial
momentum. This replaces the inherited Newtonian approximation to `j'` and
the incomplete velocity-to-momentum conversion. The supplied torque must
already use the momentum normalization expected by the equations of motion.

The Newton solve retains parameter/frequency derivatives and performs a final
update even if its starting guess is exact, so the root's implicit derivative
is present. Failed solves and unstable circular starting points are rejected.
Graphs are discarded only for inference calls where outer differentiation
was not requested; ordinary ODE inference does not accumulate local-partial
graphs. Local derivative helpers disable `inference_mode` before enabling
autograd and clone inference tensors in coordinates and model attributes
without mutating the supplied model. This also supports Hamiltonians and
inputs constructed inside `torch.inference_mode()`. The adiabatic approximation is restricted to the stable inspiral
branch and breaks down near the last stable orbit.

## Validation and limits

Independent checks cover the Schwarzschild mass shell at nonzero radial
momentum, all conservative geodesic derivatives, circular radius and angular
momentum, leading adiabatic radial momentum, the PN series and horizon,
quartic momentum correction, batch/scalar agreement, finite-difference
agreement for state/parameter/frequency gradients, and inference graph use.

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_clean_room_eob_conservative.py tests/test_eob_hamiltonian.py tests/test_eob_dynamics.py
```

Result: **18 passed**, including both `no_grad` and `inference_mode` with
inputs created inside and outside each context. A direct check with the real
radiation-reaction class also returns finite, detached state and derivatives
when all objects are constructed in `inference_mode`. No target binary is imported by the new equation tests.
The spin Hamiltonian, calibrated v4 potentials and spin maps, factorized
radiation flux, NQC corrections, and merger attachment remain outside this
repair. Passing these checks does not establish SEOBNRv4 waveform parity.
