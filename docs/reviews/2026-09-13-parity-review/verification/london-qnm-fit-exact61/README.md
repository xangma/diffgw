# Exact independent N=61 London QNM reconstruction

This directory records a bounded clean-room reconstruction of the six
fundamental Kerr QNM polynomial fits in London and Fauchon-Jones,
[arXiv:1810.03550v2](https://arxiv.org/html/1810.03550).  Every training root
was solved independently on an exact 61-node grid with
`a=0.995*sin(theta)` and theta uniform on `[-pi/2, pi/2]`.  The 33 fixed
off-grid check roots were also solved independently with no training-grid
seed.  The original clean diagnostic supplies the fit and comparison helpers
read-only; this run does not interpolate the earlier N=257 roots and does
not change runtime coefficients.

N=61 is a metadata hypothesis inferred from the public four-decimal
coefficient rounding, not a published fact.  The paper does not specify its
historical training count, hidden GMVP selections, numerical precision, or
all fit preprocessing choices, so this result is a bounded reconstruction.

`report.json` contains the declared source and uncertainty, execution host and
job metadata, source hashes, software versions, all N=61 coefficients, every
per-coefficient radius/phase rounding check, exact training residuals,
independent check residuals, and SHA-256 hashes for each raw root artifact.
`progress.json` is the final incremental copy.  Raw roots are stored at
`/tmp/torchwave-london-qnm-fit-exact61` on the run host and are referenced by
path and hash in the report.

The completed run used host `RONI2`, PID `59222`, and finished in about 36.25
seconds.  It can be rerun from the repository root with:

```text
PYTHONPATH=src:tests:scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/diagnose_london_qnm_fit_exact61.py
```
