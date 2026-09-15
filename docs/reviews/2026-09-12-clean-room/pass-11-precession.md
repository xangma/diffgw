# Pass 11: MSA precession averaged couplings

## Assessment

The primary source is Pratten et al., arXiv:2004.06503v2, Sec. IV D. Equation
(4.20) defines

    <S_fin^2>_pr = S_av^2 + L_fin^2
                + 2 L_fin (<S1.Lhat>_pr + <S2.Lhat>_pr).

The native MSA initialization already evaluates the same averaged couplings:
`S_av^2=(S_+^2+S_-^2)/2` and the two averaged spin projections. This pass
exposes those values through an opt in return path and provides a standalone
helper for the published squared-spin identity when `L_fin` is supplied
explicitly. In total-mass-normalized units, the helper returns
`<S_fin^2>/M^4` from `S_av^2/M^4`, the two projections divided by `M^2`, and
`L_fin/M^2`.

The source does not resolve the final Kerr normalization: Eq. (4.17) and
the `L_fin` definition in Eq. (4.18) imply division by `M_fin^2`, while the
explicit FinalSpinMod3 formula Eq. (4.29) prints division by `M^2`; the notation section defines
`M` as the initial total mass. No source statement equates these masses or
specifies which convention overrides the inconsistency. The exact-zero sign
case is also unspecified: the paper says the signed result follows the sign
of the nonprecessing final spin, but does not define a tie-break when that
spin is zero and the precessional magnitude is nonzero. XP/XPHM therefore do
not select an MSA remnant convention in this pass.

## Changes

- Added `compute_msa_averaged_couplings` in `precession_msa.py`, preserving the
  existing four-value `evaluate_msa_dynamics` API and broadcasting scalar and
  batched inputs consistently.
- Added `compute_msa_final_spin_squared`, which evaluates Eq. (4.20) without
  normalization, clamping, or signed-Kerr conversion.
- Kept the existing pass-10 NNLO remnant handling unchanged; XP/XPHM do not
  automatically select an unresolved MSA remnant convention.
- Added focused tests for independent vector and aligned identities, batched
  versus scalar averages, and finite gradients in `tests/test_pass11_precession.py`.

## Sources

- Pratten et al., "Computationally efficient models for the dominant and
  sub-dominant harmonic modes of precessing binary black holes," [arXiv:2004.06503](https://arxiv.org/abs/2004.06503),
  v2; local TeX SHA-256:
  `d1e611c582b2861bc34d975e99b6c2d6ef3d4447442aab249541616a331443f5`.
- Chatziioannou et al., "Constructing gravitational waves from generic
  spin-precessing compact binary inspirals," [arXiv:1703.03967](https://arxiv.org/abs/1703.03967), cited by
  Pratten for the averaged couplings in Eqs. (A9)--(A11).
- Khan et al., "Phenomenological model for the gravitational-wave signal from
  precessing binary black holes with two-spin effects," [arXiv:1809.10113](https://arxiv.org/abs/1809.10113);
  local TeX SHA-256:
  `542ba40840e79e65df8ea65b590b9f557925d9ac861f08d1e4e3657d49377c16`.

## Validation

Run with the requested environment:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest tests/test_pass11_precession.py tests/test_clean_room_precession_conventions.py
```

The final agent run passed **95 tests** across the focused precession/MSA
suites. Parent integration passed **55 tests** across the new pass-11 cases,
NNLO remnant, precession equations, XP MSA and XPHM reconstruction. The parent
restored XAS/XP/XPHM byte-for-byte to the pass-10 snapshot, including their
existing NNLO equation references. Full verification is recorded separately.
