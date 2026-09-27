# IMRPhenomT dominant ringdown amplitude supplied-input gate, 2026-09-27

The [component specification](../math/phenomt-ringdown-amplitude.md)
pins Estellés et al., arXiv:2012.11923v1, Eqs. (26), (27c)–(27e).
This is an equation-level test of the dominant `(2,2)` ringdown ansatz on
supplied QNM damping rates, peak amplitude, and shape input. It does not
validate those inputs or the full waveform.

The test used a private snapshot of the isolated worktree based on
integration commit `bbccb1d` at
`/home/xangma/tmp/diffgw-phenomt-ringdown-amplitude-20260927` on `len`.
The scalar reference and test file in the snapshot had SHA-256 values
`4baa86838c6792df0aadb5a75fb4b70b6dc70d4c40a6290c535e5603a5ec7530`
and `0a343219193014728eec83e3f3eebefcff6403e5c4efcd2fdd2f62e8fa413212`,
respectively, matching the committed files. The existing environment used
Python 3.12.3, NumPy 2.5.3, PyTorch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0,
and pytest 9.1.1. No packages or native source were changed.

```sh
cd /home/xangma/tmp/diffgw-phenomt-ringdown-amplitude-20260927
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_phenomt_ringdown_amplitude_math_reference.py --reference-device=cpu -ra
```

The `len` CPU run returned **8 passed in 60.48 s**, with no skips. The
equation identities `H(0)=H_peak` and `H'(0)=0` passed at 100-digit
precision on two independent scalar inputs. A high-precision central
difference checked the analytic derivative at `u=2,25`.

The public `IMRPhenomT.time_domain` Torch and JAX polarizations were
converted to the intrinsic face-on amplitude for `(30,30,0,0)`,
`(35,25,0.7,-0.5)`, and `(60,6,0.9,0.1)`, with masses in solar masses.
At seven ringdown times `u=0,0.5,2,8,25,80,160`, amplitudes agreed
with the scalar reference to `rtol=3e-11, atol=2e-13`. Automatic
derivatives of the public amplitude at `u=0,2,25,80` agreed to
`rtol=5e-10, atol=2e-12`; the zero-time peak derivative also met
`atol=2e-12`. Both frontends used FP64 CPU.

The first two binaries take the printed `d2=(alpha2-alpha1)/2` branch.
The high mass-ratio case instead activates the carriers' existing
bounded-`d2` convention: actual `d2≈0.0469122`, versus printed
`d2≈0.0622423`. Its match is a carrier-ansatz check with supplied actual
`d2`, **not** Eq. (27d) fidelity. The bound and all native defaults were
left unchanged. This evidence does not cover the QNM/remnant, peak and
shape fits, inspiral or merger, other modes, calibration against NR,
physical accuracy, or whole IMRPhenomT qualification.
