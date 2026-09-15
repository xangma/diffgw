# Clean TEOB merger NQC verification

This bounded probe builds the scalar zero spin, zero tide ``nu=0.24``
post-adiabatic profile from ``r=20`` to ``r=14``, continues through the first
accepted orbital frequency decrease and two additional mass units, evaluates
the clean factorized ``(2,2)`` mode, and runs the isolated auxiliary grid and
explicit target NQC assembly.  It does not route a production waveform or run
a BNS case.

The target values use the explicitly named 2017 published provider:
Nagar, Riemenschneider & Pratten, arXiv:1703.06814v1, Appendix A Table II for
equal mass and Table IV with Eq. (A3) for ``0.16 < nu <= 0.25``.  The
normalization recorded by the module is ``A_RW=A/(nu*sqrt(24))=|Psi_22|/nu``.

## Launch record

The completed run was launched on host ``RONI2`` from
``/Users/xangma/repos/torchwave`` with:

```text
env PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/diagnose_teob_merger_nqc_clean.py --output docs/reviews/2026-09-13-parity-review/verification/teob-merger-nqc-clean/diagnostic.json --cache docs/reviews/2026-09-13-parity-review/verification/teob-merger-nqc-clean/physical-input-output.npz
```

The PTY session was ``26330`` and the Python PID was ``90124``.  The requested
log path was
``docs/reviews/2026-09-13-parity-review/verification/teob-merger-nqc-clean/probe.log``;
the foreground PTY carried stdout while the JSON report and NPZ cache were
written to their specified paths.  Stop an active PTY run with ``Ctrl-C`` in
session ``26330``.

## Result

The run completed in 28.65 seconds on CPU.  It retained the 661 accepted
trajectory samples from the continuation (579 adaptive samples and 22 fixed
samples including the event seam).  The NQC auxiliary grid has 202 samples at
``0.1M`` spacing, with cut support indices 574 through 659.  Its sampled
frequency maximum is at ``t=12316.452448035621`` and the zero-spin matching
time is ``t_NQC=12315.452448035621`` (auxiliary matching index 168).

The solved amplitude coefficients are
``(-0.12638567597533554, 1.482289030442369)`` and the phase coefficients are
``(0.16430793959517773, 4.22116780033991)``.  Scaled condition estimates are
1.1504891688329797 and 1.2636734408062182; both matching residuals are at the
floating point level.  No auxiliary corrected amplitudes were negative.

The reusable cache is ``physical-input-output.npz``.  Its SHA-256 is
``06bb6d88d9b148cc481c8b45a694ea0d132ec195f0c109fc83ba0e024421d334``.
``diagnostic.json`` records the complete named-array inventory and SHA-256
values for every source before and after the run; all source hashes were
unchanged.

Focused validation:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_teob_merger_nqc_clean.py
11 passed in 0.75s
```
