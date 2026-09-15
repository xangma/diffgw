# NSBH remnant phase-offset acceptance diagnostic

`nsbh_remnant_offset_diagnostic.py` evaluates full `df=0.25` grids at
`f_final=512,1024`, for the frozen catalog/holdout cases and references
`f_ref={30,27,60}` or `{40,27,60}`.  It returns both native `hp/hc` and public
LAL binary `hp/hc`, with raw relative L2, magnitude L2, unit-phase L2, and
affine phase slope.  Length and support are checked for every record.

The native `_legacy_phase_offsets` callable is used only through its public
signature and temporarily replaced in a `try/finally` block.  For each case,
endpoint, and reference, freeze hypotheses copy `remnant_mass`, then both
`remnant_mass` and `remnant_spin`, exclusively from that call's lambda-zero
offset inputs.  Captured calls and exact substitutions are in the JSON.

The second offset output changes by `+0.00320211661` rad for the catalog
lambda-restored call at `512 Hz`; multiplying by the physical total mass in
seconds gives the explicit Fourier-sign predictions `-1.00940777e-7` and
`+1.00940777e-7 rad/Hz` (both signs are recorded).  No fitted slope is used.
Against the public binary, mass-only freezing improves catalog raw L2 from
`2.9515e-5` to `2.9512e-6` and holdout from `1.7580e-3` to `5.8443e-6`.
Freezing spin as well worsens holdout to `2.2541e-3`; therefore the direct
binary acceptance result supports mass-only substitution as the bounded scope.
Residual amplitude differences are unchanged, and both polarizations agree.

The PhenomD class exposes `IMRPhenomD.remnant_properties` with documented
outputs `(m_final [solar masses], a_final, f_rd, f_damp)`, and this callable is
used black-box to supply the phase-offset mass while retaining NSBH spin.  At
`f_final=1024,f_ref=30`, this reduces catalog native-vs-binary hp raw L2 to
`7.77e-8` and unit-phase L2 to `6.85e-9`; for holdout `f_ref=40`, raw L2 is
`6.57e-7` and unit-phase L2 `1.04e-5`.  This PhenomD-mass composition is the
strongest tested model-identity hypothesis.  The acceptance JSON records
full-bin raw metrics, support mismatches, restoration check, callable
inventory/docstring, and native/diagnostic hashes.
