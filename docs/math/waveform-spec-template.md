# Waveform specification template

Copy this outline for each waveform model, component, and extension
specification. Replace guidance with specific facts and retain the six
level-two headings in order. Add level-three headings as needed. The canonical
model page links to component pages instead of repeating their derivations.

For a public cataloged model, set `WaveformDefinition.spec_page` to the new
`math/<model>.md` path and leave the two markers below. Run
`python scripts/generate_waveform_library_table.py`; the catalog fills in the
identity, scope, supported backends, and option availability. A research
specification outside the catalog may use this outline with a manually stated
status and blocker table. Its presence does not make it a public waveform.
Remove the generated markers for standalone component or extension pages and
give their stable ID and version in **Scope and status**.

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

The generator replaces this sentence for a registered model.

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

State the exact model and variant: time or frequency domain, modes, physical
range, what is implemented, what remains conditional, and what is pending.
Distinguish the mathematical generator, any trained or interpolated surrogate,
and wrappers around external data. Give each a separate status. Link to the
public generation guide only if the specified variant is actually dispatched.

## Inputs, conventions, and options

Record public and reference inputs, units, domains, defaults, frequency or time
grid, Fourier sign, polarization and mode convention, phase and time origin,
frame, cutoffs, and derivative behavior at branch boundaries. Explain which
option choices run in Torch/JAX and which exist only in the scalar reference.
Name every required numerical table, interpolation rule, calibration file, or
trained asset with its version, hash, availability, and terms.

## Mathematics and components

Write the composition equation and link separately versioned component pages.
For each component give a stable ID, equation or algorithm, pinned source and
location, variant, reference module, and status. Use `pending` when an equation
or algorithm is unresolved, `conditional` when evaluation needs declared
external inputs, `executable` when the stated reference can run, and
`diagnostic` for checks that are not production mathematics. These statuses
describe implementation state, not verification.

| Component ID/version | Source equation or algorithm | Variant | Status | Scalar reference | Torch | JAX | Maintained check or blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Replace with a stable ID and version | Pin a paper/table/asset location | Paper, carrier, or project adapter | Pending, conditional, executable, or diagnostic | Function or pending | Function or pending | Function or pending | Test or unresolved gate |

For a time-domain generator, include initial conditions, ODE right-hand side,
integration and event rules, numerical convergence, and any time-to-frequency
transform. For a reduced-order model, specify the training set, basis,
interpolation, error measure, asset provenance, and independent holdouts.

## Implementation mapping

Map each component ID to the independent scalar expression and native Torch
and JAX locations. State where a paper-literal form differs from the carrier,
including signs, normalizations, branches, interpolation, and additional fit
digits. Say explicitly when a frontend or option is absent. The native
evaluator must not call the scalar reference, and a reference-only expression
must not appear as a native option in the catalog.

## Evidence and limitations

Keep these claims separate: source transcription, scalar-reference behavior,
Torch/JAX implementation agreement, same-model external comparison, and
physical validation. For each claimed result give component IDs, variants,
inputs and parameter range, precision/device, metrics, thresholds, failures,
and reproducible evidence. List unresolved gates and qualification blockers.
An executable component or passing sample is not a whole-model verification.

## Reproduce the checks

Give maintained test commands, required dependencies and optional assets,
expected skips, and the relevant CI gate. Explain how a changed equation,
convention, source, or asset is versioned and how generated docs and fixtures
are refreshed and reviewed.
