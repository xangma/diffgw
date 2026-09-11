# IMRPhenomD source repair — 11 September 2026

This change improves numerical Kerr ringdown frequencies. Full LAL parity remains unachieved.

## Primary sources

- [Khan et al., PRD 93, 044007 (2016), Paper II](https://arxiv.org/abs/1508.07253v2): amplitude, phase and Table V calibration. The [original TeX archive](https://arxiv.org/src/1508.07253v2) has SHA256 `e70cde9d5e9ee92e4592c4c93152f6eb2410264358486f6a68064fb03b39b1cf`.
- [Husa et al., PRD 93, 044006 (2016), Paper I: New numerical waveforms and anatomy of the signal](https://arxiv.org/abs/1508.07250v2): remnant mass and spin. The source archive has SHA256 `79dd5b277fd99d8ff3576800fb274b296bea8ae1553f134a015205f40fae5b4a`.
- [Lo, Sabani and Cardoso, PRD 111, 124002 (2025)](https://arxiv.org/abs/2504.00084v2): the numerical Kerr frequencies used in the final code. Its authors link the [Center of Gravity data index](https://the-center-of-gravity.com/data-and-routines/), which makes the data available with a request to cite the original work. No academic-only restriction is stated.

The selected [2025 archive](https://the-center-of-gravity.com/documents/169/QNMQNEF_s-2.tar.gz) has SHA256 `9d2cad8459d26abe3f0b9534398d6e397aad7f71e451182753b9093634f48b75`.

| Bundled file | Original archive member | SHA256 |
| --- | --- | --- |
| `kerr_qnm_220_prograde.dat` | `s-2/s-2l2m2n0.dat` | `6969fd6a9c6806aea564591fa807675a356c18444fe8973bd1c5ce377afc7e2d` |
| `kerr_qnm_220_retrograde.dat` | `s-2/s-2l2m-2n0.dat` | `7789924c12e9737d75328da1b475b0eeddb24dd2c9c1a2f7a26a07b69540ab20` |

Each unmodified CSV file contains 1,900 rows. Columns 0–2 are dimensionless spin, real angular frequency and negative imaginary angular frequency; the remaining author columns are retained. The fundamental overtone uses `n=0`.

## Numerical implementation and review

Agy initially implemented interpolation of the older Berti/Cardoso/Will data from the same index. Codex review found cold-cache inference and zero-spin derivative bugs; Agy corrected those. Further review found material disagreement between the older and newer primary data near extremal retrograde spin: at spin magnitude 0.9999, their real frequencies are 0.2076047727 and 0.2915589799182418, respectively. The older archive is retained in the external audit evidence and is not bundled.

After the bounded Agy attempts, Codex replaced the data with the 2025 release and implemented a natural cubic spline on its nonuniform signed grid. Negative spins use the retrograde branch; zero occurs once. The tridiagonal continuity equations are solved independently using ordinary Python arithmetic. Evaluation, batching and derivatives use Torch. Cached tensor constants are created outside inference mode. Frequencies are converted from remnant angular units to total-mass cycle units by multiplying by `(M/M_f)/(2*pi)`; the imaginary part is negated to obtain positive damping.

Tests check every source node, off-node values and derivatives against SciPy's independently implemented natural cubic spline, zero and extreme-spin gradients, float32/64 batches, cold inference followed by autograd, and normalized waveform gradients and polarization scaling. Interpolation is clamped outside the data domain. These tests validate the numerical implementation, not the physical accuracy of every source datum or full-waveform LAL parity.

## Clean-room boundary and unresolved calibration

Codex inspected LAL documentation/source for compatibility diagnosis. Agy received abstract behavioral requirements, public-oracle diagnostic measurements, and independent primary papers/data. No LAL code, calibration table, or source-specific implementation outline was passed to Agy or copied into TorchWave.

The original Paper II Table V includes 19 rows, including alpha1 and beta1. Simply substituting their fits for the final connection constants would not preserve the required phase continuity; the paper requires analytical matching. Its printed calibration numbers are rounded, and its arbitrary global time freedom does not establish LAL's exact time-origin convention. Full-precision independent calibration data and a reproducible compatibility specification remain unresolved. No oracle residuals were fitted, and no parity thresholds or canonical cases were changed.

See [the verification report](../report.md) for measured results and the remaining catalog failures.
