# TorchWave parity status — superseded September 11, 2026

The current report is [the September 11 Agy review and verification](../2026-09-11-agy/report.md). Full catalog parity is **not complete**.

- Final regression suite: **238 passed**.
- Strict analytic acceptance: **6/6 passed**, comprising four TaylorF2 cases and two TaylorF2_Eccentric cases.
- Strict catalog acceptance: **6/16 cases passed**, ten failed, and five of the 17 manifest models have no independently validated oracle mapping in the harness.
- Direct complex/norm gates remain `1e-6`; direct flat and named-PSD mismatch gates remain `1e-8`, on both polarizations and six detector/sky projections per case.

Optimized matches are diagnostics and do not establish unnormalized amplitude parity. Earlier claims of verification across parameter space, blanket high overlaps for higher modes/precession, and external Kerr ringdown parity exceeded the recorded evidence. The current report and manifest restrict claims to measured configurations.

Historical [round 6 evidence](round-6-evidence/harness.json) and earlier round evidence bundles remain unchanged. The old six-case result was an analytic subset, not a catalog acceptance result. The pre-dispatch editable report was retained with the dirty-workspace baseline in the local Agy run directory.
