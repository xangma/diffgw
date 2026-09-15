# Production clean XP MSA verification

`diagnostic.json` is produced by
`scripts/diagnose_xp_msa_production_clean.py`. It first proves on the frozen
catalog/holdout rows and fresh generic, equal-mass, aligned, zero-spin, and
small in-plane-spin probes that omitted public IMRPhenomXP choices equal the
explicit production tuple `prec_version=223`, `convention=1`,
`final_spin_mod=3`. It then reports the unchanged frozen rows separately from
derived explicit-tuple rows and fresh production probes. Both convention 0
and convention 1 are included for the exact aligned and zero-spin boundaries;
every probe supplies all three spin components for both bodies.

The diagnostic reports direct complex relative L2 errors using the unchanged
repository gate (`1e-6`) and records the clean helper SHA256 before and after
the run. The small in-plane-spin probe remains visible as a known boundary
failure near the pole; it is not masked and does not support a full-domain
parity claim. Rows with `L+S_parallel<=0` use the existing legacy fallback
row by row; this extreme guard is reported separately from ordinary exact
aligned/zero-spin rows. The convention-1 exact aligned/zero-spin chart
differences remain visible in the JSON (about `2.5e-5` to `3.5e-5`) and are
not relabeled as passes.

Run it with:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
scripts/diagnose_xp_msa_production_clean.py
```
