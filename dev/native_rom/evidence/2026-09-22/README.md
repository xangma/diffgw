# Initial native-generator qualification: failed

Frozen source commit: `85009ec13a3b78aaadbbed7421781a7cce0b7b11`. Exact configuration, source/input hashes,
installed binary identities, costs and results are in [qualification.json](qualification.json).
All five bounded scalar CPU float64 solves completed, and sources remained unchanged.
The row is 30 + 20 solar masses, zero spins, starting orbital frequency `M Omega=0.04`.

| Refined comparison | Complex relative L2 | Amplitude relative L2 | Gate |
| --- | ---: | ---: | --- |
| ODE tolerance | 0.000204642401 | 8.98683049e-06 | **Failed** |
| Output grid | 3.61020504e-08 | 9.75402891e-10 | Passed |

The numerical error budget is `1e-6`. ODE refinement decreases the error but
remains above budget. Peak, match and boundary time checks pass their `0.05 M`
limits. This row does not qualify the generator for ROM training or release.

The pinned LAL public-binary comparison is an unaligned diagnostic, with no
parity claim: its phase semantics and time origins have not been established as
equivalent. Neither these discrepancies nor numerical convergence alone certify
physical fidelity. Derivatives, batches, JAX and a physical parameter-domain
campaign are still unqualified; the known QNM knot-provenance gap also remains.
No training array or replacement ROM table was generated.

Reproduce from the repository root (with the recorded runtime versions):

```sh
PYTHONPATH=src python -m dev.native_rom.qualify_generator \
  --compare-lal --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out /tmp/native-generator-qualification.json
```

The expected exit status for this evidence is 1. Do not loosen thresholds or
fit coefficients to this comparison to manufacture a passing result. Diagnose
numerical sensitivity and convention/provenance gaps on the development branch,
then rerun the declared gates before any independent ROM training.
