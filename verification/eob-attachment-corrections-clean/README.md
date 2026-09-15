# Clean EOB attachment corrections

`run_diagnostic.py` performs a short paired check using the permitted public
`dt_0p5M` and `dt_1_4096s` state/mode fixtures. It recomputes the clean
Hamiltonian, flux, waveform, remnant, QNM, and numerical attachment path; it
does not evolve a trajectory or import reference code. Results are recorded in
`diagnostic.json`.

The optional amplitude curvature clamp remains deferred. From the independent
Eq. 2.12 differentiation, at zero amplitude derivative
``A''/A = -d^2 - 2*d*c1*tanh(c2)``; for ``c2 < 0`` the nonpositive-curvature
condition gives ``c1 <= -d/(2*tanh(c2))``. No clamp is applied until that
compatibility choice has its own independent test.

Run it with:

```text
PYTHONPATH=src:tests:scripts:. \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  verification/eob-attachment-corrections-clean/run_diagnostic.py
```

The focused contract suite is:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_eob_numerical_contract_clean.py
```
