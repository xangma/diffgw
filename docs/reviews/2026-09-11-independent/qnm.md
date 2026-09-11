# Independent Kerr QNM representation

TorchWave evaluates the fundamental gravitational Kerr mode (l=2, m=2, n=0) using two Chebyshev polynomials and PyTorch arithmetic. It reads no runtime QNM files, imports no external solver, and uses no LAL calibration values. The polynomial domain is signed spin [-0.9999, 0.9999]; inputs outside it are clamped. The waveform remnant fit has its own narrower clamp.

## Derivation and provenance

The development generator [generate_kerr_qnm.py](../../../scripts/generate_kerr_qnm.py) solves the radial continued-fraction and angular eigenvalue problems using [Stein's qnm package (2019)](https://doi.org/10.21105/joss.01683). The development environment used qnm 0.4.4, NumPy 2.3.5, and SciPy 1.16.3. These are not new TorchWave runtime dependencies.

Each signed-spin branch is continued from the Schwarzschild fundamental root. The initial guess 0.37-0.09i and angular guess 4 are refined by the equations; no input frequency tables or waveform outputs supply training values. Negative signed spin is evaluated with nonnegative Kerr spin and m=-2. Dense continuation support prevents inadvertent branch switching.

The coordinate is t=sqrt(1-a), linearly mapped to [-1,1]. Degree-95 Chebyshev polynomials represent Re(omega) and -Im(omega)/t, using 128 Chebyshev-Gauss training nodes. Dividing out t regularizes the near-extremal damping behavior. Clenshaw evaluation uses two vectors of 96 embedded coefficients. It preserves tensor shape, dtype, device and gradients within the clamped domain. Float32 is limited by its arithmetic, particularly near extremality.

## Numerical verification

Training and validation solves use angular truncation l_max=28, continued-fraction tolerance 2e-15, minimum 600 and maximum 30000 iterations. Seven additional checks use l_max=40, tolerance 1e-16 and minimum 2000 iterations. MINPACK root tolerance starts at 1e-12; when it reports stagnation, the last candidate is retried at 1e-10 without relaxing the continued-fraction tolerance. Roots with radial residual above 1e-9 are rejected.

There are 2013 unused validation spins: 1600 deterministic random draws in the transformed coordinate, 401 uniformly spaced signed spins, and edge/zero/outlier checks, with duplicates removed. Maximum float64 relative errors are 1.0073e-11 in frequency and 2.1546e-11 in damping. These are measured finite-set errors, not a proof of a uniform bound. All stricter roots pass the existing 1.1e-10 relative acceptance check. Tightening the solver fixed the earlier endpoint failure; the acceptance check was not relaxed.

Tests also cover finite-difference gradients at seven spins, an independent-root derivative at zero, batch/scalar equivalence, float32/float64, inference followed by autograd without file access, domain clamping, and a waveform spin gradient. The combined QNM, phase-convention and existing PhenomD tests passed: 26 tests.

The validation fixture is [kerr_qnm_220_leaver.json](../../../tests/data/kerr_qnm_220_leaver.json). SHA256: `dbcb329c194a7e10955f3e703ed08e81c4eb37bffd4299d61d82213821fb6c93`.

Reproduce calibration and validation in a separate development environment:

```sh
python scripts/generate_kerr_qnm.py --output-dir /tmp/torchwave-qnm
```

## Author-data discrepancy

The previous Lo, Sabani and Cardoso (2025) data files are retained under [author-data](author-data/) only as historical research evidence. They are no longer packaged runtime assets or interpolation inputs. Their prograde a=0.9976 damping value is 0.0160559974464903. Independent higher-accuracy solves instead give approximately 0.01579614695503, a discrepancy of about 1.6%. Nearby points and solver refinement support the latter value. The new representation deliberately does not reproduce that isolated table value.

This verifies the independent QNM component. It does not establish complete IMRPhenomD parity with LAL; other calibration and convention errors remain under investigation.
