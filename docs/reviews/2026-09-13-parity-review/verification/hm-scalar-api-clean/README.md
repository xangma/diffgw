# HM scalar API cleanup

The native `IMRPhenomHM` path now treats a scalar frequency as one value per
parameter batch row, while an explicit frequency vector retains its trailing
frequency axis. Carrier callback `TypeError` exceptions propagate directly.

Validation with the pinned Python environment:

- Scalar batch, explicit-grid parity, and carrier-error tests: **3 passed**.
- Gradient smoke check for scalar batched parameters and frequency grids:
  finite gradients for frequency, masses, and spins.
- Full `tests/test_imrphenom_hm.py`: **9 passed, 1 known failure**. The failure
  is the pre-existing XHM/LAL overlap check (`0.974452 < 0.98`).
- Focused reconstruction and clean-HM suites: **41 passed, 2 existing Eq. 9
  higher-mode failures**.
