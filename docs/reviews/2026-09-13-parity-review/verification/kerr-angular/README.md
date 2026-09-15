# Kerr angular phase gauge diagnostic

`gauge.json` records the clean-room Leaver Frobenius comparison at spins
`0, .3, .7, .9` and inclinations `0, .4, .8, pi/2, 2.3, pi`, using the
validated fresh helper family (`m=2`, `ell=2,3`). The default
`spherical_projection` gauge is unchanged. `a0_real_positive` retains the
recurrence gauge `a[0]=1` after positive real Hermitian normalization.

The public LALSimulation output (version 6.2.0) was divided by `sqrt(2*pi)`.
Independent Gauss-Legendre integration records its Hermitian and bilinear
norms. Hermitian norms are near one but have finite, sometimes nonmonotonic
quadrature/truncation deviations (the JSON retains every order); bilinear norms
are complex for complex `c`, as expected for the complex-symmetric angular problem. The Leaver ODE
residual is below `4e-13` in the sampled interior. Gauge ratios are retained
per angle in the JSON; no binary-fitted rotation or residual correction was
applied.

The helper was supplied as a clean-room untracked file, so an original startup
hash was unavailable; the JSON records the actual before/after hashes of this
diagnostic rerun and says so explicitly.
The script intentionally stops at the angular diagnostic and does not modify
the ringdown adapter.
