# NRTidal v1 public-binary taper convention diagnostic — 27 September 2026

**Result:** The pinned LALSuite 7.26.15 binary uses the published descending
Planck-window *shape* for all six declared systems, but its nonzero-tide
window begins at a merger frequency below the value obtained from the
coefficients printed in Dietrich et al. (2018), Eq. (11). The source does not
give a higher-precision coefficient set sufficient to identify the binary's
frequency. This does not qualify the nonzero-tide tapered carrier.

This is a follow-up to the [frozen boundary failure](nrtidalv1_taper_20260927.md).
The [diagnostic script](https://github.com/xangma/diffgw/blob/main/scripts/diagnose_nrtidalv1_taper_conventions.py)
calls only the public `SimInspiralChooseFDWaveformSequence` interface. It
samples the modulus of the binary BNS/BBH quotient at seven frequencies
$f/f_{\rm mrg}^{\rm paper}=1.025,1.05,1.075,1.1,1.125,1.15,1.175$.
It inverts [McKechan et al. (2010), Eq. (7)](https://arxiv.org/pdf/1003.2939)
at the central sample to infer one merger frequency, then checks the other
six without fitting the window shape. The printed merger fit and the
$\kappa_2^T\to\kappa_{\rm eff}^T$ substitution come from
[Dietrich et al. (2018), Eq. (11) and preceding paragraph](https://arxiv.org/pdf/1804.02235).

| System | Printed-fit $f_{\rm mrg}$ Hz | Inferred binary $f_{\rm mrg}$ Hz | Relative shift | Maximum modulus error, printed / inferred |
| --- | ---: | ---: | ---: | ---: |
| Equal, $1.35/1.35$, $\Lambda=400/400$ | 2037.930727302 | 2037.878643198 | $-2.55573474\times10^{-5}$ | $2.81138\times10^{-4}$ / $7.77\times10^{-16}$ |
| Unequal, $1.6/1.2$, $\Lambda=200/800$ | 1709.953767101 | 1709.910763599 | $-2.51489269\times10^{-5}$ | $2.76645\times10^{-4}$ / $8.33\times10^{-16}$ |
| Exchanged bodies | 1709.953767101 | 1709.910763599 | $-2.51489269\times10^{-5}$ | $2.76645\times10^{-4}$ / $8.33\times10^{-16}$ |
| Zero tide, $1.35/1.35$ | 4291.579157959 | 4291.579157959 | $0$ | $1.05\times10^{-15}$ / $1.05\times10^{-15}$ |
| Holdout unequal, $1.48/1.22$, $\Lambda=600/250$ | 1766.115917226 | 1766.062949709 | $-2.99909630\times10^{-5}$ | $3.29910\times10^{-4}$ / $1.44\times10^{-15}$ |
| Holdout equal, $1.25/1.25$, $\Lambda=900/900$ | 1819.558741158 | 1819.470113491 | $-4.87083296\times10^{-5}$ | $5.35818\times10^{-4}$ / $1.11\times10^{-15}$ |

Independent inversion of *each* of the seven interior samples gives a
merger-frequency spread no greater than $7.42\times10^{-10}$ Hz (including
zero tide). Thus a single frequency shift explains the observed modulus
throughout the taper; a different Planck-window shape is unnecessary for
these cases. The explicit alternative of using $\kappa_2^T$ instead of
$\kappa_{\rm eff}^T$ gives errors of $6.26\times10^{-3}$ and
$2.77\times10^{-2}$ in the two unequal-mass cases. It cannot explain the
equal-mass mismatch because those couplings coincide there.

The printed coefficient $d_2=2.236\times10^{-4}$ has a half-last-digit
rounding interval of $5\times10^{-8}$. Varying *only* that coefficient over
its printed rounding interval contains all inferred frequencies. This is a
precision illustration, **not** evidence that $d_2$ alone differs in the
binary: the paper does not publish the binary's full-precision coefficients,
and other printed coefficients also have rounding intervals. We did not
extract or fit a replacement coefficient from the binary. The paper-derived
scalar reference and Torch/JAX tests retain the printed coefficients, and
the strict nonzero-tide boundary failure remains open.

The run used host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv1-phase-20260927`, and the pinned
environment documented in the earlier receipt. The foreground command
completed normally in 3.3 s:

```sh
OMP_NUM_THREADS=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python \
  scripts/diagnose_nrtidalv1_taper_conventions.py
```

This is a diagnostic receipt outside the required strict mathematical
reference selection. No passing binary taper gate is inferred from the
frequency inversion, which uses the binary's own output.
