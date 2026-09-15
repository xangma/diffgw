# Paired clean TEOB physical diagnostic

`diagnose_teob_paired_driver_clean.py` is a bounded component composition
diagnostic.  It uses the public binary fixture's exact
`Kepler_radius_double = 19.999999999999996`, with `nu=0.24`, `delta=0.2`,
zero spins and tides, PA order 8, `r_min=14`, `dr=0.1`, the clean adaptive
defaults, and the fixed continuation cap `0.1`.  The waveform carrier is
`teob_waveform_state_mode(state, ham, 2, 2, Omega=omega,
residual_phase="zero-unpublished")`; its `TEOBRadiationClean.mode` applies
the TEOB factor `exp(-i*m*Hreal*Omega)` exactly once.  The radial convention
is `F_r*=0`; the reported conservative radial quantity is `r_ddot`.

The current clean horizon prescription is the baseline.  The separate
leading-order run replaces only the horizon term with the diagnostic
zero-spin Eq. (98)--(99) expression from
`scripts/diagnose_teob_pa_residual_clean.py`:

```
F_H=(32/5)*nu**2*r_omega**4*Omega**5*(X1**4+X2**4)*vphi**8
F_phi,H=-F_H/nu
```

Both branches use the existing 2017 NQC target solve, UIB2016 remnant, and
clean ringdown assembly.  Output is compared at every sample of the full
public uniform `dt=0.5M` grid.  The independent comparison evaluates natural
cubics separately on signed amplitude and lifted phase.  Raw complex,
amplitude, and unwrapped phase errors are same-index and unaligned; no
fitting, truncation, or full-inspiral validation is claimed.

The public inputs are fixture SHA256
`09eaa8b00288c94947b2bb4b40ada7780dc51ac98491a48cbb8f60ed50ab686c` and
metadata SHA256
`047faee14ab7d0747d5ee887ce75d83019a2d0e6ff74cf6408f4f8a7dfe2a070`.
The canonical patched runs loaded radiation SHA256
`dca2c9791403467d85572f25a062cbc995d8228743d33b6b2e5dc28c13336bd6`
before and after.  The diagnostic script SHA256 was
`52c5267c4bbb389e0801f27d99e2c95ea96d3ca6c67d48d16e3e8586f9fdef22` and
the leading-order source SHA256 was
`f93908b2dbc381bf6353f44b34680b35bf4e876d208aacdede554ef34f4fc8a4`.
Each JSON report records the complete clean-source hash map, host, command,
PID, and elapsed time.

The patched current baseline is in `diagnostic.json` and
`paired-physical-output.npz` (cache SHA256
`6b693eb0bab2e041bb98f07a02a085e41ed2f70f93264d82f622b3e7acd39c03`).  It
completed in 31.10 s.  It produced 60 PA samples and 668 continuation
samples, with the peak at `t=12316.733081416947M`,
`Omega=0.17533020087458825`, and NQC coefficients
`A=(-0.0662764943647725, 1.287602792056405)` and
`P=(0.18375403307983354, 1.7443450240164475)`.  The NQC-corrected branch
has post-boundary L2 errors (raw, amplitude, phase) of
`(4.1134141e-22, 7.1213404e-23, 0.12897928 rad)` on 307 public samples;
the raw branch has `(1.1000689e-20, 7.6806551e-23, 6.97600281 rad)`.  Full
grid corrected errors are `(4.1134141e-22, 2.4933568e-22, 0.13583598 rad)`.

The leading-order branch is in `diagnostic-lo-horizon.json` and
`paired-physical-output-lo-horizon.npz` (cache SHA256
`bdf872bb49873aa6818c6b28cbc17fd601cfcaa37a3ecb7a28e5ebf1e2140dae`).  It
completed in 29.35 s.  Its peak is later by `0.02964967M` at
`t=12316.76273108648M`, with `Omega=0.175369746043458`, and its NQC
coefficients are `A=(-0.0667673120434196, 1.2891348921645176)` and
`P=(0.1853703207593379, 1.796257451621234)`.  Relative to the current
baseline, the continuous peak, merger, and `t_match` each move by
`+0.05524371M`; the strict-earlier joined index and sampled attachment index
move from `(750, 749)` to `(751, 750)`.  The public output boundary remains
sample index 24631 (`t=12315.5M`).  Post-boundary corrected errors become
`(1.7305246e-23, 2.3204753e-24, 0.16239831 rad)`, while the raw branch is
`(1.1541030e-20, 9.7247634e-23, 7.39143279 rad)`.  These are attribution
measurements; no horizon policy was selected from residual size.

The earlier current process imported radiation before the concurrent patch.
Its report and cache are preserved as `diagnostic-current-prepatch.json` and
`paired-physical-output-current-prepatch.npz` (cache SHA256
`2cd9906941a78e23227456aba69da5d9502cd72e0491a068c49276d79bf83c9f`).  Its
report explicitly records before/after radiation hashes
`f903bfd44cc62b8d8e08ef08dfd8e55fe15e78f55914edbfa96bcc887b68d416` and
`dca2c9791403467d85572f25a062cbc995d8228743d33b6b2e5dc28c13336bd6`; it is
not treated as the frozen baseline.  The first leading-order artifact is
also retained as `diagnostic-lo-horizon-initial.json` and
`paired-physical-output-lo-horizon-initial.npz`; it loaded the patched hash
and matches the canonical leading-order cache.

The run command was executed with
`PYTHONPATH=src:tests:scripts:.`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and the requested pycbcorig313 Python.
Focused clean component validation had already passed with **66 passed, 1
skipped**.  No runtime production routing or source dependency was changed.
