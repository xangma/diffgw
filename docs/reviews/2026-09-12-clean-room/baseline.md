# PhenomD and Kerr QNM equation review, 2026-09-12

Scope: inherited `imrphenomd.py` and `kerr_qnm_interp.py` at repository baseline
`84ef9b3`. This review used the present TorchWave source, freshly retrieved
scientific papers, and independently published author QNM data. It did not use
LAL/PyCBC source or runtime output, translated implementations, historical source,
prior calibration/audit documents, or residual fitting. It does not certify the
origin of inherited full-precision calibration constants.

## Fresh inputs

| Input | Version and relevant material | SHA-256 of downloaded bytes |
| --- | --- | --- |
| [Khan et al., PRD 93, 044007](https://arxiv.org/pdf/1508.07253v2) | arXiv:1508.07253v2; Eqs. (19)–(21), amplitude matching in Sec. VIII; phase Eqs. (13)–(16), (28), (37); coefficient map Eq. (31), Table V | `c3ed056e896d0ef02b639966a60a197587baf0ee514c9c952576bf979fc7c907` |
| [Husa et al., PRD 93, 044006](https://arxiv.org/pdf/1508.07250v2) | arXiv:1508.07250v2; remnant fits Eqs. (16), (17) | `1fcb99449d1127ab3c2403ddbd81b8d76845528ff2a594b9aac640b9d29d459e` |
| [Cook and Zalutskiy, PRD 90, 124021](https://arxiv.org/pdf/1410.7698v2) | arXiv:1410.7698v2; angular/radial Teukolsky equations (4), (5a), numerical methods Sec. II and accuracy discussion Sec. III | `33973382059461835989057a47e93336e5233cccede48529b4b19cdc306d7330` |
| [Cook, KerrQNM_00.h5](https://zenodo.org/records/2650358/files/KerrQNM_00.h5?download=1) | [Zenodo 2650358, version 1, 2019-04-24](https://zenodo.org/records/2650358); gravitational `s=-2`, `n=0` author numerical data | `d5853acabc1a07b66b4fae437c02e1435a9d2b3561bd7e4752b9721c9cd48923` |

The HDF5 download also matches the author's published MD5,
`34c04f3b03bbb99935d1f91f7296bd02`. The author explicitly attributes the data
calculation to the Cook–Zalutskiy paper. No data values were obtained from a
waveform-library package. Source downloads are retained locally under
`/tmp/torchwave-clean-baseline/`; the 106 MB HDF5 file is not a runtime dependency
or repository fixture.

## Equation repairs

Let `g=gamma2`, `w=gamma3*f_damp` and `x=f-f_rd`. The scaled merger amplitude is
proportional to `exp(-g*x/w)/(x*x+w*w)`. Direct differentiation gives

```text
dA/df = A * (-2*x/(x*x+w*w) - g/w).
```

For `|g|<1`, the inherited peak numerator `sqrt(1-g*g)-1` is rationalized to
`-g*g/(1+sqrt(1-g*g))`, so its ratio to `g` is evaluated as
`-g/(1+sqrt(1-g*g))`. This removes cancellation near zero, the artificial `1e-12`
square-root shift, and the `1e-10` root cutoff. No fit coefficients change.

For `|g|>=1`, the code retains its inherited continuation `x=-w/g`; the cited
paper's real stationary-point equation does not define that regime. For
`g>1`, this continuation has a nonzero amplitude slope. The intermediate
polynomial previously imposed zero slope at that join. It now uses the analytic
derivative above, satisfying the paper's C1 matching requirement. A physical
example with masses `(60,15)` and aligned spins `(.99,.99)` gives
`gamma2≈1.0085`; its normalized scaled-amplitude slope at the join is approximately
`-0.8156`, so the old zero-slope condition was inconsistent with the merger
amplitude itself. This example extrapolates beyond the unequal-mass calibration
spins and establishes continuity, not waveform fidelity there.

The exact `|g|=1` branch has no finite two-sided derivative as approached from
the real-root side. Autograd uses the continuation derivative at equality;
inactive square-root branches are kept finite without changing function values.

## Independent QNM validation

The current approximation was evaluated directly at every author node with
`a<=0.9999` in datasets `n00/m+02/{2,2,0}` and `n00/m-02/{2,-2,0}`. Columns
`0,1,2` are `a, Re(M omega), Im(M omega)`. For the second dataset, spin is
mapped to `-a`, using the simultaneous spin/azimuthal-sign symmetry of the
Teukolsky equations; damping is always `-Im(M omega)`. No interpolation,
optimization, coefficient regeneration or rounding to a waveform-library table
was performed.

| Branch | Nodes | Maximum absolute frequency error | Maximum absolute damping error | Maximum relative frequency error | Maximum relative damping error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m=+2`, positive spin | 1,014 | `9.75664e-13` | `1.11519e-12` | `9.934e-13` | `1.35839e-10` |
| `m=-2`, reflected negative spin | 1,012 | `7.64611e-13` | `1.89598e-12` | `2.61807e-12` | `2.15331e-11` |

These small discrepancies do not justify changing the inherited QNM
coefficients. They are numerical agreement with independently published values,
not evidence establishing how the inherited coefficients were generated. The
author paper conservatively estimates absolute accuracy of at least `1e-10`
and generally expects about `1e-12`; the comparison should not be read as an
independent absolute-truth guarantee below that scale. Nine author rows are
included explicitly in the new test, with `3e-12` absolute regression tolerance.

The full comparison can be reproduced with `h5py`, NumPy and Torch by loading the
two datasets, selecting `rows[:,0] <= .9999`, calling `evaluate_kerr_qnm_220` at
the signed first column, and comparing to columns 1 and negative 2. Compare in
float64 and include both zero-spin rows to reproduce the 2,026-node count.

## Validation and remaining limits

Command (local host, repository root):

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_clean_room_phenomd_baseline.py
```

Result: **6 passed**. Checks cover a 70-digit decimal evaluation of the published
peak formula near zero and one; batch finite-difference/autograd agreement;
finite branch-point autograd; full-waveform scaled-amplitude C1 continuity in
both real-root and continuation cases; independent QNM author nodes; and QNM
batch gradients. No oracle library is imported by these tests.

The reviewed phase pieces and their C1 assembly have the published structure.
The paper leaves global time and phase shifts free in physical comparisons.
This review did not derive a new implementation-specific raw-complex time gauge
and did not change the existing phase anchoring. The candidate explanation that
QNM accuracy causes the reported additional-case parity failures is unsupported
by the author-data comparison. Strict raw-complex `1e-6` agreement remains the
independent verifier's responsibility and is not claimed here.

Finally, Table V and the remnant-fit papers print rounded coefficients. Those
citations alone cannot certify all digits in `phenomd_unrounded_fits.py`.
The source docstrings now distinguish equations and numerical validation from
unresolved inherited provenance. Fresh author full-precision supplements or a
separately auditable regeneration would be needed to close that issue.
