# TEOB NQC attribution

The bounded same-grid counterfactual reconstructs the saved `nqc_original`
amplitude and lifted phase through the recorded attachment, then compares it
with the saved processed native polarizations and the exact-start public full
route.  It uses natural-cubic interpolation, no alignment, and no fitted
quantities.

Applying NQC improves agreement with the public full route. On the 2,637-sample PA
prefix, the uncorrected native mode has relative L2 `1.0713e-3` against the
public full route; the processed native mode is `4.0408e-6`.  The independent
PA-only control gives `3.4257e-7` for uncorrected native versus public PA-only.
The PA-only waveform comparison is close, while the native full route's NQC
correction is already active at the first sample and improves agreement with
the public full route. Through ODE inspiral, processed residual is
`1.1269e-4`; the eight cached auxiliary-to-attachment samples are
`1.2359e-3`.

No documentary code fix is justified. This establishes the importance of the
correction, not the cause of its remaining error. Targets, timing, trajectory,
interpolation and projection conventions remain unresolved. The experiment
has no public full-route NQC-off control to isolate them. See `report.json`
for hashes, windows, and exact limits.
