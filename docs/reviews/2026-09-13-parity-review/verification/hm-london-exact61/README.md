# HM waveform impact of the independent exact N=61 fit

`report.json` records a bounded diagnostic that evaluates the full precision
MAIN coefficients from the exact-N=61 clean-room fit through the fresh HM
assembly.  It compares all six positive-m modes and the projected `hp` and
`hc` arrays with the public HM callable on the four cases and the fixed
frequency array from `diagnose_hm_qnm_leaver_clean.py`.

The report includes two declared controls: the public four-decimal London
fit and the original N=257 MAIN fit.  The N=61 count remains a metadata
hypothesis inferred from published coefficient intervals; it is not asserted
to be the historical grid or a production full-precision source.  The
assembly compatibility values are passed explicitly as amplitude join
`0.014` and phase join `0.018`, while the printed London defaults are not
selected from waveform residuals.  For each case, the outer frequency map
uses its HM QNM values; Eq. (14) uses the D carrier's absolute ringdown
frequency and damping, the HM frequency and damping ratios, and a C1 join at
`0.5*fRD_D`.  The report records the corrected final mass fields as
`final_mass_solar_mass` and `final_mass_fraction`, plus before/after source
hashes for the diagnostic inputs.

This is a diagnostic only.  No runtime coefficients are changed, and no
strict parity gate is claimed.  The public LAL HM API supplies the comparison
target through the existing diagnostic helper; no LAL, PyCBC, TEOB, or author
implementation is copied here.

The completed run used host `RONI2` and took about 1.15 seconds.  Re-run from
the repository root with:

```text
PYTHONPATH=src:tests:scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/diagnose_hm_london_exact61.py
```
