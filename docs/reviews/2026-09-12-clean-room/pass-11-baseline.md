# Pass 11 baseline: NSBH remnant spin variable

## Finding

The Zappa et al. NSBH remnant map is defined as

\[
 F(\nu,a_{\rm BH},\Lambda)\to(X_\bullet,a_\bullet)
\]

in Eq. (1), with the fit in Eqs. (4)--(6):

\[
 F=F_{\rm BBH}(\nu,a_{\rm BH})
 \frac{1+p_1(\nu,a_{\rm BH})\Lambda+p_2(\nu,a_{\rm BH})\Lambda^2}
 {(1+[p_3(\nu,a_{\rm BH})]^2\Lambda)^2},
\]

where `p_k` is quadratic in `nu` and each coefficient is linear in
`a_BH`. The paper states that the NS spin is neglected. The native
implementation had instead formed `chi` as a mass-weighted combination of
`chi1` and `chi2`, which is not the source-defined independent variable for
the tidal polynomials.

The implementation now uses `chi1` directly for those polynomials. The BBH
baseline is still evaluated with both caller-supplied spins, preserving the
existing API and its exact `Lambda=0` construction. The published NSBH
calibration assumes `chi2=0`; nonzero NS-spin behavior remains outside that
calibration domain and is documented rather than silently remapped.

The existing Thompson classification remains unchanged: its Appendix A
workflow computes the remnant before the ringdown quantities, and Table I
partitions the amplitude by `f_tide >= f_RD` and the torus ratio. This pass
only corrects the source-defined spin variable used by the remnant fit.

## Primary sources

- Thompson et al., *Modeling the gravitational wave signature of neutron
  star black hole coalescences: PhenomNSBH*, arXiv:2002.08383v2:
  <https://arxiv.org/abs/2002.08383> (Appendix A Eqs. (13)--(15), Table I,
  and the remnant/classification workflow).
  SHA-256 of downloaded `https://arxiv.org/pdf/2002.08383v2`:
  `7e8bf4ba34a7fb98d54fa1a405267fdb4d06a78e1b164f57361b0b2abebd34e2`.
- Zappa et al., *On the mass, spin, and luminosity of the remnant black
  hole from binary neutron star-black hole mergers*, arXiv:1903.11622v2:
  <https://arxiv.org/abs/1903.11622> (Eqs. (1), (4)--(6), Table I, and the
  statement that NS spin is neglected).
  SHA-256 of downloaded `https://arxiv.org/pdf/1903.11622`:
  `2dd12d6c75fd9a5551f1e6eab014af6355fceea6e5676aea8709f6530a3aefdb`.

## Changes and validation

- `src/torchwave/approximants/imrphenom_nsbh.py`: use BH spin `chi1` in the
  Zappa tidal mass and spin correction polynomials; retain the supplied
  two-spin BBH baseline.
- `tests/test_pass11_baseline.py`: equation-level checks reconstruct both
  Table I polynomial families and check the exact `Lambda=0` BBH limit with
  a nonzero second spin.

Focused command:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_pass11_baseline.py tests/test_phenomnsbh_reconstruction.py
```

Initial agent subset: 9 passed. Parent review added both the calibrated zero
NS-spin input and the retained nonzero-spin API path; the final focused run
including `tests/test_clean_room_nsbh.py` passed 23 tests. No calibration coefficients were regenerated or fit, and
no external implementation or oracle waveform was used.

## Limits

The correction is source-defined but does not extend the Zappa calibration to
spinning neutron stars. Parent-level LAL comparisons should treat nonzero
`chi2` as an API compatibility path whose tidal remnant correction is still
calibrated only in the `chi2=0` domain.
