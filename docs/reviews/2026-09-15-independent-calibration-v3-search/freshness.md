# SXS catalog freshness check

**Retrieved:** 2026-09-15 11:44:58 UTC (read-only; no waveform/H5 files)

## Primary sources and hashes

- Release listing: [SXS Collaboration catalog releases](https://api.github.com/repos/sxs-collaboration/sxscatalogdata/releases), HTTP 200, 13 release records, response SHA-256 `2174110cc1b8a26d93cc3bc62e17eb1dbdbfdc717fead6b95782724d2c322aae`.
- Latest release identified by `published_at`: `v3.0.0`, published 2025-05-14 18:17:30 UTC.
- Full catalog: [v3.0.0/simulations.json](https://raw.githubusercontent.com/sxs-collaboration/sxscatalogdata/v3.0.0/simulations.json), HTTP 200, 32,618,880 bytes, 4,170 entries, SHA-256 `132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6`.
- Freshness cache: `/tmp/torchwave-nr33-v3-freshness/catalog/simulations.json`; refresh metadata and response headers are in `refresh-metadata.json` in that directory.

The cached 2025-05-14 catalog at `/tmp/torchwave-nr33-v2/catalog/simulations.json` has the same byte count and SHA-256 (`132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6`). The current tagged catalog therefore remains current at the metadata snapshot level. This checks the public GitHub release source; it is not a claim that every CaltechDATA record or all SXS data sources are unchanged.

## Re-run of frozen q6/q8 metadata filters

Using the v3 frozen windows q6=[5.97, 6.03] and q8=[7.96, 8.04], no deprecated marker, at least two levels, catalog reference eccentricity `<=1e-3`, maximum reference spin norm `<=1e-4`, and excluding the original seven plus failed frozen SXS:BBH:3630 and SXS:BBH:3622:

- q6: **SXS:BBH:2489** (levels Lev4/Lev5), **SXS:BBH:2164** (Lev2/Lev3).
- q8: **SXS:BBH:2707** (Lev3/Lev4).

No additional eligible q6 or q8 IDs appear in the current full catalog relative to the cached scan. SXS:BBH:0181 remains excluded because its catalog `keywords` value is `deprecated`. Ranking remains unchanged. No H5, waveform, or numerical fit inspection was performed.
