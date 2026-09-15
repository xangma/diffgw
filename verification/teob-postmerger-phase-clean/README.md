# TEOBResumS postmerger phase clean-room verification

`postmerger_22` now evaluates the phase logarithm with
`exp(-c2phi*tau)`, where `c2phi = alpha21`.  The amplitude tanh retains its
separate rate `c2A = alpha21/2`.  This is the distinction stated in Appendix
F.1, Eqs. 74--80, of [Nagar et al., arXiv:1806.01772v2](https://arxiv.org/html/1806.01772v2).

The focused tests check the merger frequency anchor for the fitted and
explicitly overridden frequencies, the zero amplitude slope at merger, late
QNM frequency and amplitude-rate limits, FP64 batches, and finite Torch
gradients.  The report records a reconstructed pre-fix pointwise discrepancy
and SHA-256 hashes for the implementation and test source.

Run the focused checks with:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_teob_postmerger.py \
  tests/test_clean_room_teob_uib2016_postmerger.py
```

This is a bounded component verification.  No trajectory, production
routing, NQC amplitude-rate convention, binary residual, or parameter fit is
part of this record.
