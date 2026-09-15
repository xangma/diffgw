# Independent London--Fauchon-Jones QNM fit reconstruction

This directory records a bounded clean-room reconstruction of the fundamental
Kerr QNM polynomial ansätze in London and Fauchon-Jones,
[arXiv:1810.03550v2](https://arxiv.org/abs/1810.03550).  The numerical data are
fresh roots from `scripts/kerr_leaver_solver.py`; no LAL, PyCBC, author code,
calibration table, waveform output, or public target binary was used.

The declared spin grid is `a=0.995*sin(theta)` with uniform theta in
`[-pi/2, pi/2]`.  The finest grid has 257 nodes.  The 65 and 129 node fits
select nested nodes from that same grid.  Each mode also has 33 fixed
off-grid check nodes, solved without a training-grid seed.  The solver is run
with `strict=True`; unconverged, nonphysical, or weakly estimated roots are
rejected.  Fixed midpoint continuation is used only to recover a failed
continuation step.

The main regression follows the paper's constrained form: for zero-damped
`(2,2)`, `(3,3)`, `(4,4)`, and `(4,3)`, it fits
`(omega-m/2)/kappa` using powers 0 through 4.  For `(2,1)` and `(3,2)`, it
fits `omega` using powers 0 through 6.  Here
`kappa=[log_3(2-a)]**(1/(2+ell-|m|))`, and the independent solver roots are
conjugated to the paper's positive-imaginary damping convention.  A separate
zero-damped sensitivity fit uses the direct constrained frequency with powers
1 through 5 and residual weight `1/kappa`; it is reported separately and does
not choose the reconstruction.

The machine-readable result is [report.json](report.json), with an incremental
[progress.json](progress.json) and the preflight [benchmark.json](benchmark.json).
Raw root rows were written under `/tmp/torchwave-london-qnm-fit`; their paths
and SHA-256 hashes are recorded in `report.json`.  The report also records the
source hashes, software versions, all grid and fit choices, convergence
comparisons, printed-coefficient comparisons, and independent check residuals.

All 1,542 training roots and 198 off-grid check roots in the completed six-mode
run converged, with no continuation refinements.  The strict root error
estimate was at most about `7.4e-15`.  The independent N=257 main-fit check
residuals range from about `8.9e-5` to `3.9e-4` in dimensionless frequency.
The independently generated coefficients do not all lie inside four-decimal rounding intervals around the
printed coefficients.  Since the paper does not specify its historical grid
count or all GMVP selections, these results are a bounded physics regression
and should not be treated as an exact historical recreation.

Re-run from the repository root with:

```text
PYTHONPATH=src:tests:scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/diagnose_london_qnm_fit_clean.py
```
