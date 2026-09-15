# Native MSA v223 contract verification

This directory records the bounded, opt-in compatibility work for the clean
MSA composition. The implementation keeps equation-level defaults and exposes
the native branch through explicit compatibility="v223" or waveform_v223(...)
selectors.

The focused contract controls are run with:

    PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
    tests/test_precession_msa_v223_contract_clean.py

The public binary comparison is a diagnostic only:

    PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
    scripts/diagnose_msa_v223_contract_clean.py

It evaluates the supplied three fresh systems on 81 frequencies from each
reference frequency through 200 Hz and reports raw complex hp/hc differences
against public IMRPhenomXP on the common integer-Hz band. It also includes the
frozen XP catalog and holdout records. Public calls use precession version 223,
expansion order 5, mprime=2, convention 0, and MSA final-spin mode 3. No
phase, time, amplitude, or residual fit is applied.

The explicit contracts covered by the controls are the distinct 259/72 MSA
opening and 259/75 source-frame orbital-L expressions, initial reference-spin
flux contractions, the named quadratic phase convention, stable oscillatory
branch primitives and pole signs, degenerate-root suppression, mass-swap
invariance, and initial-total-mass-squared final-spin normalization.
