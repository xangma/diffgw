# Public PE API with IMRPhenomD

Run [the maintained example](https://github.com/xangma/diffgw/blob/main/examples/04_public_parameter_estimation.py)
from a checkout with the selected backend installed:

```bash
python examples/04_public_parameter_estimation.py --backend torch --device cuda \
  --draws 500 --warmup 300 --output /tmp/diffgw-pe-torch.npz
python examples/04_public_parameter_estimation.py --backend jax --device gpu \
  --draws 500 --warmup 300 --output /tmp/diffgw-pe-jax.npz
```

Use `--device cpu` for a CPU workflow. This example generates a zero-noise
IMRPhenomD injection with masses `(30, 25)` solar masses, fixed zero aligned spins,
distance 1,000 Mpc and inclination 0.4 radians. H1 and L1 use fixed sky coordinates,
polarization and epoch, with the analytic aLIGO design PSD. The frequency grid is
20–128 Hz at 2 Hz spacing with rectangular integration. The prior is independent
uniform in `mass1 ∈ (29.5, 30.5)` and `mass2 ∈ (24.5, 25.5)` solar masses.

The example uses `diffgw.inference.WhittleLikelihood`, `UniformBoxPrior`,
`Posterior` and `run_nuts`. Two chains start at `(29.9, 25.1)` and `(30.1, 24.9)`.
The sampler returns physical samples, reports posterior means and optionally
saves a compressed archive with named parameters, sampler diagnostics and JSON
metadata. All waveform, likelihood and chart evaluations use float64/complex128.

This is a short API example. Its coarse grid and short chains are not the
[frozen qualification protocol](../validation/inference.md#parameter-estimation-campaign).
Assess your posterior against a refined dense reference and verify mixing and
Monte Carlo uncertainty before using scientific results. See the
[PE guide](../guides/parameter-estimation.md) for API contracts, device setup and
diagnostic limitations.
