# Retired waveform prototypes

This is a historical record, not a list of available release waveforms or a
validation claim. The [current coverage table](../coverage.md) is generated from
the Torch and JAX dispatch registries. The pre-retirement model-manifest entries
are preserved [as a snapshot](retired-model-manifest.json) from
[commit `6aa67a3`](https://github.com/xangma/diffgw/commit/6aa67a3dfed03f64a5428eb472275239cc6b9e9d);
they must not be read as current API documentation.

| Former route | Former public names | Current status |
| --- | --- | --- |
| Native SEOBNR ROM pilot | `seobnrv5rom` and the former `SEOBNR_ROM` pilot class | Pilot asset and alias removed from releases. The retained `SEOBNR_ROM` class now aliases the official external-data evaluator. |
| `SEOBNR_ODE` | `seobnr`, `seobnrode`, `seobnrv5` | Complete prototype waveform removed from release dispatch. Independently scoped EOB equation components remain. |
| `SEOBNRv4Published` and `SEOBNRv4PublishedNonspinning` | `seobnrv4published`, `seobnrv4publishednonspinning` | Experimental complete-waveform assemblies removed from release dispatch. |
| `TEOBResumS` | `teobresums`, `teobresum`, `teob` | Complete prototype waveform removed from release dispatch. Scoped nonspinning vacuum TEOB components remain. Its [source and component audit](../../math/standalone-family-research-ledger.md) is retained as a snapshot. |

The former bundled Torch pilot asset `src/torchwave/data/seobnr_surrogate.pt`
had SHA-256 `8316be966b297f474be810024a6add2d5e64559ee3c506302e8ea83372085414`.
It sampled geometric `Mf` from 0.0002 to 0.30 at 256 nodes, with 12 amplitude
and 16 phase SVD basis vectors and degree-three parameter maps over mass ratio
and aligned spins. These describe the retired prototype, **not** the retained
official ROM's HDF5 data. The [provenance ledger](../../provenance_ledger.md)
keeps the historical failed/limited waveform comparisons; neither a retired
route nor a surviving component inherits evidence from the official ROM.

Development code and qualification records remain on the
[`codex/native-seobnr-rom` branch](https://github.com/xangma/diffgw/tree/codex/native-seobnr-rom/dev).
Reintroducing any route requires a newly declared model scope, source and asset
provenance, independent reference, and validation of the complete composition.
