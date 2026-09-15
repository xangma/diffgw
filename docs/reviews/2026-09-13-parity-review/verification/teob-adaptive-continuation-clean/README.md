# Clean TEOB adaptive continuation verification

This bounded probe uses the clean scalar vacuum field at ``nu=0.24``.  It
builds the order-8 post-adiabatic profile from ``r=20`` to ``r=14`` and then
continues through the first accepted frequency decrease for ``2M`` with a
fixed ``0.1M`` cap.  It does not construct a waveform or run a BNS case.

## Launch record

The final run was launched on host ``RONI2`` from
``/Users/xangma/repos/torchwave`` with the following command:

```text
env PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/diagnose_teob_adaptive_continuation_clean.py --output docs/reviews/2026-09-13-parity-review/verification/teob-adaptive-continuation-clean/diagnostic.json
```

The PTY job was session ``42740`` and the Python process PID was ``58407``.
The requested log path was
``docs/reviews/2026-09-13-parity-review/verification/teob-adaptive-continuation-clean/probe.log``;
the foreground PTY captured stdout while the JSON report was written to
``diagnostic.json``.  Stop an active PTY run with ``Ctrl-C`` in session
``42740``.

## Result

The run completed in 30.11 seconds on CPU.  The PA endpoint was at
``t=9608.553736478192`` with ``r=14``.  The accepted peak occurred at
``t=12316.726873406744`` and ``r=1.8410829047817987``.  The fixed continuation
ended at ``t=12318.826873406742`` after 21 accepted fixed steps.  The phase
seam error was exactly zero; the combined accepted trajectory contains 661
valid samples (579 adaptive and 22 fixed samples, with the event seam emitted
once).

The machine-readable report records both pre-run and post-run SHA-256 values
for the PA, adaptive driver, continuation, vacuum driver, and contract files;
``source_hashes_unchanged`` is ``true``.

Focused validation after the continuation changes:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_teob_adaptive_continuation_clean.py tests/test_eob_adaptive_driver_clean.py tests/test_teob_postadiabatic_clean.py
39 passed in 1.50s
```
