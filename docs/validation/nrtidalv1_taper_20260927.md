# NRTidal v1 merger taper and premerger carrier — 27 September 2026

**Outcome: 71/71 strict tests passed**, with no skips or expected failures,
for the [v1 phase](../math/nrtidalv1-phase.md) and
[merger-taper composition](../math/nrtidalv1-merger-taper.md) specifications.
The pinned public binary fails the separately measured nonzero-tide taper
boundary gate. The full tapered carrier is therefore **not qualified**.

The run used `len`, Linux 6.8.0-138-generic x86_64, in
`/home/xangma/tmp/diffgw-nrtidalv1-phase-20260927`, populated from the
isolated v1 worktree at `ebe8d0cb074a3c4981aec89e98c08ea2bbb329e8`
plus this slice's files. The worker PID was `1895615`; it exited normally
after 24.95 s. Its log is `qualification_taper.log` in that directory. The
strict command was:

```sh
DIFFGW_REQUIRE_LAL_BINARY=1 OMP_NUM_THREADS=1 \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
  tests/test_nrtidalv1_phase_math_reference.py \
  tests/test_nrtidalv1_phase_lal_binary.py \
  tests/test_nrtidalv1_taper_math_reference.py \
  tests/test_nrtidalv1_taper_lal_binary.py
```

Python was 3.12.3; NumPy 2.5.3, SymPy 1.14.0, mpmath 1.3.0,
Torch 2.14.0+cpu, JAX 0.11.2, and LALSuite 7.26.15. The pinned
`_lalsimulation.cpython-312-x86_64-linux-gnu.so` SHA-256 was
`e816879d23e652e8b3928496d6fd261de0832b5c3781c50e33a458042495b42e`.
Only the public binary API was called; no LAL source was inspected.

The scalar merger-frequency fit comes from Dietrich et al. (2018) Eq. (11),
including its stated $\kappa_2^T\to\kappa_{\rm eff}^T$ implementation
substitution. The descending window begins at $f_{\rm mrg}$ and ends at
$1.2f_{\rm mrg}$, as stated there and reiterated by Dietrich et al. (2019)
around Eq. (25). A symbolic check proves the reused scalar fit's coupling
equals the v1 phase coupling. Six-case Torch/JAX merger-frequency and
complex-factor gates met relative $10^{-11}$ and absolute $10^{-10}$ limits,
respectively, including both window boundaries and the exchanged-body case.
The analytic descending branch is McKechan et al. (2010) Eq. (7), the
Planck-taper source cited by the 2018 paper.

For each of the six declared systems, the public-binary and Torch/JAX full
complex $h_+$ were compared *without* time or phase alignment at 30–800 Hz
in 5 Hz steps, plus $0.9 f_{\rm mrg}$ and $f_{\rm mrg}$. The predeclared
relative complex $L_2$ limit was $10^{-8}$; the largest observed value was
`5.06616992e-12` (Torch, equal-mass holdout). The binary BNS/BBH quotient
also met its $5\times10^{-5}$ absolute limit before merger, and the
zero-tide taper met the same limit at the declared boundary fractions.

The separate, executable [boundary diagnostic](https://github.com/xangma/diffgw/blob/main/scripts/diagnose_nrtidalv1_taper_boundary.py)
uses the same six cases and the frozen fractions
`[0.9, 1, 1.025, 1.05, 1.1, 1.15, 1.195, 1.2, 1.25]` times the
paper-derived merger frequency. It compares the **modulus** of the public
binary BNS/BBH quotient to the published taper, with an absolute
$5\times10^{-5}$ gate. It also compares the full, unaligned Torch/JAX
complex $h_+$ to the binary at 30–800 Hz in 5 Hz steps plus those taper
fractions, with a frozen relative $L_2\le10^{-6}$ gate. Run separately in
the same `len` environment:

```sh
OMP_NUM_THREADS=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python \
  scripts/diagnose_nrtidalv1_taper_boundary.py
```

| Case | $f_{\rm mrg}$ (Hz) | Maximum modulus error | Frozen gate |
| --- | ---: | ---: | --- |
| Equal, $1.35/1.35$, $\Lambda=400/400$ | 2037.930727302 | 0.000281137999140 | fail |
| Unequal, $1.6/1.2$, $\Lambda=200/800$ | 1709.953767101 | 0.000276645146647 | fail |
| Exchanged bodies | 1709.953767101 | 0.000276645146647 | fail |
| Zero tide, $1.35/1.35$ | 4291.579157959 | $2.22044604925\times10^{-16}$ | pass |
| Holdout unequal, $1.48/1.22$, $\Lambda=600/250$ | 1766.115917226 | 0.000329910475047 | fail |
| Holdout equal, $1.25/1.25$, $\Lambda=900/900$ | 1819.558741158 | 0.000535817673631 | fail |

| Case | Torch full complex $L_2$ | JAX full complex $L_2$ | Frozen $10^{-6}$ gate |
| --- | ---: | ---: | --- |
| Equal | $8.03401562076\times10^{-7}$ | $8.03401562064\times10^{-7}$ | pass |
| Unequal | $9.59380727294\times10^{-7}$ | $9.59380727303\times10^{-7}$ | pass |
| Exchanged | $9.59380727294\times10^{-7}$ | $9.59380727303\times10^{-7}$ | pass |
| Zero tide | $2.07469505500\times10^{-12}$ | $2.07828008433\times10^{-12}$ | pass |
| Holdout unequal | $1.10511323958\times10^{-6}$ | $1.10511323963\times10^{-6}$ | fail |
| Holdout equal | $1.73914933990\times10^{-6}$ | $1.73914933989\times10^{-6}$ | fail |

The diagnostic was kept outside the required strict tests so an expected
failure cannot hide a qualification gap. No coefficient was extracted from
the binary or adjusted to pass the gate. The successful results establish
the published scalar-to-native taper and an unaligned premerger full-carrier
comparison for these cases. They do not establish the nonzero-tide tapered
binary carrier, physical fidelity to independent neutron-star simulations,
spinning cases, other modes, or broad parameter-domain coverage.
