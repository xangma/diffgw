# Scalar qualification of the Kerr 220 QNM input

This bounded check concerns the dimensionless fundamental `(s,l,m,n)=(-2,2,2,0)`
frequency `M_f omega` and its angular separation constant `A(a omega)`. It
does not qualify other modes, overtones, amplitudes, remnant fits, or the
waveform as a whole.

The independent reference in `src/diffgw_reference/kerr_qnm.py` implements
the transformed radial equation and three-term recurrence from
[Cook and Zalutskiy (2014), Eqs. (23)--(43)](https://arxiv.org/abs/1410.7698v2).
It uses `M=1`, `exp(-i omega t)`, spin weight `s=-2`, and the ingoing-horizon,
outgoing-infinity choices `zeta=+i omega`, `xi=2-i sigma_+`,
`eta=2+i sigma_-`. Here `r_±=1±sqrt(1-a²)`,
`sigma_±=(2 omega r_±-2a)/(r_+-r_-)`, `p=i omega sqrt(1-a²)`, and
`alpha=1-4i omega`. The residual is `beta_0+alpha_0 R_0`, where
`R_n=-gamma_(n+1)/(beta_(n+1)+alpha_(n+1) R_(n+1))` and the tail is set to
zero at finite depth. mpmath solves this scalar complex residual. This differs
from the generated asset solver's inverted fraction and asymptotic tail.

The angular `A` is an mpmath eigenvalue of the Galerkin operator
`diag(l(l+1)-2)-4(a omega)X-(a omega)²X²`, whose matrix elements are integrated
from independent SymPy Wigner-d harmonics. Its spherical branch starts at
`A(0)=4`. No LAL implementation or numerical constants enter the reference.

The predeclared CPU FP64 acceptance gates are absolute complex-frequency
error `1e-8`, angular-eigenvalue error `1e-10`, and spin-derivative error
`2e-5` for each frequency component. Scalar truncation gates are `2e-12`
between radial depths 200 and 400 and `2e-11` between angular bases ending
at `ell=8` and `ell=10`. Checks use spins `0`, `0.35`, `0.417`, `0.7`, and
`0.733` for values, including two off-grid points; derivatives use `0.35`
and `0.7`, with the scalar centered difference at
`h=1e-4`. The derivative tolerance includes centered-difference truncation.
These thresholds and cases were chosen before native-output comparisons.

The experiment can be repeated with the existing len environment:

```sh
cd /home/xangma/repos/diffgw-worktrees/kerr-angular-qualification
PYTHONPATH=src /home/xangma/.cache/diffgw-xp-pycbc-venv/bin/python -m pytest -q tests/test_kerr_qnm_scalar_reference.py
```

The radial zero-tail convergence is tested only in this moderate-spin domain.
It should not be extrapolated to near-extremal spins or higher overtones, where
the minimal-solution tail converges more slowly and root identity needs wider
continuation checks.
