# `diffgw` Top-Level Dispatcher API Reference

This section provides the auto-generated reference documentation for the unified backend dispatch interface in **`diffgw`**.

For physical coverage and execution restrictions, see [model support](../models.md).
For numerical qualification, see [accuracy and validation](../validation.md).
For source-backed model and component relationships, see the
[waveform library](../waveforms/index.md).

---

## Unified Waveform Dispatch

::: diffgw.get_fd_waveform
    options:
      show_root_heading: true
      heading_level: 3

::: diffgw.get_td_waveform
    options:
      show_root_heading: true
      heading_level: 3

## Waveform library metadata

`diffgw.waveforms` describes the versioned mathematical-reference scope for
`TaylorF2`, `IMRPhenomD`, and `IMRPhenomXAS`. These functions inspect model
metadata; generation still uses `get_fd_waveform` or `get_td_waveform` above.
The optional `diffgw_reference` SymPy package is not needed to browse the
catalog. A model entry does not claim that every model variant or physical
parameter is independently validated. The TaylorF2 entry currently describes
its nonspinning reference; its separately versioned spin, tidal, and quadrupole
references are linked from the [TaylorF2 specification](../math/taylorf2.md).

```python
from diffgw.waveforms import describe_waveform, get_waveform, list_waveforms

models = list_waveforms()
definition = get_waveform("TaylorF2")
summary = describe_waveform("TaylorF2")  # JSON-safe dictionary
```

`definition.spec_page` identifies the canonical page under `docs/`; that
page's identity and option tables are generated from this same record. An
option's `choices` include mathematical reference variants, while
`native_choices` (when present) names the subset implemented by both Torch
and JAX. For example, XAS's paper fifth-order amplitude remains a
reference-only choice.

::: diffgw.waveforms.list_waveforms
    options:
      show_root_heading: true
      heading_level: 3

::: diffgw.waveforms.get_waveform
    options:
      show_root_heading: true
      heading_level: 3

::: diffgw.waveforms.describe_waveform
    options:
      show_root_heading: true
      heading_level: 3

The independent SymPy reference is optional. Install `diffgw[reference]` to
load the module named by a model or component definition:

```python
from diffgw.reference import get_reference_module
from diffgw.waveforms import get_waveform

definition = get_waveform("IMRPhenomXAS")
reference = get_reference_module(definition.id)
component = get_reference_module(definition.id, component_id=definition.components[0].id)
```

The returned modules expose their own exact expressions and scalar evaluators.
Their catalog scope and variants determine what each expression represents.

::: diffgw.reference.get_reference_module
    options:
      show_root_heading: true
      heading_level: 3
