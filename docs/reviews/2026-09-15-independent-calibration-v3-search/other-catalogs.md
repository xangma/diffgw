# Independent non-SXS catalog search (2026-09-15)

Scope: metadata and documentation only. No waveform H5 or numerical waveform was downloaded or inspected. The fixed policy requires `|chi| <= 1e-4`, eccentricity upper bound `<= 1e-3`, at least two numerical resolutions providing the highest two orders (N2/N3), and at least 150M after the `(2,2)` amplitude peak. Target q matching is within 0.5%; q=6 and q=3 are holdouts and q=8 is a training target.

## RIT Waveform Catalog (moving-puncture finite-difference NR)

Primary sources: [catalog and metadata table](https://ccrgpages.rit.edu/~RITCatalog/), [third-catalog paper](https://arxiv.org/abs/2007.07910), and [fourth-catalog paper](https://arxiv.org/abs/2202.00018). The catalog documentation states that modes and strain extrapolated to infinite observer location are available, and that each row carries a global resolution tag. The table also exposes type, q, spins, eccentricity, merger time and orbit count, but does not identify N2/N3 extrapolation orders or a post-peak duration field.

* **Exact q=6 candidate:** RIT:BBH:0090 (`q=0.1667`, nonspinning, `e=0.0008`, `n100`) and RIT:BBH:0121 (`q=0.1667`, nonspinning, `e=0.0018`, `n140`) are the same nominal physical configuration at two listed resolution tags. The second record violates the fixed `e <= 0.001` gate. The public metadata does not establish that these tags are N2/N3, does not provide two extrapolation orders, and does not state `>=150M` post-peak duration. Thus this pair cannot enter the fixed calibration screen; the exact q match and spin values alone are insufficient.
* **q=8 coverage:** the closest nonspinning rows visible in the catalog are `q=0.117` (roughly 8.55) and `q=0.1429` (7), both outside the 0.5% target tolerance. No exact or within-tolerance q=0.125 nonspinning pair with two usable resolution/order records was found in the public metadata. This is a metadata-search conclusion, not proof that unpublished or undocumented runs do not exist.
* **Usage/attribution:** the catalog exposes BibTeX keys per row and links the RIT catalog publications; the public catalog pages state no explicit dataset license or separate reuse terms. Any use would therefore require the catalog papers and row-specific BibTeX citations, with licensing treated as unresolved from the public documentation.

**Viability:** fail under the unchanged policy. RIT is a useful independent source of infinity-extrapolated strain and an exact q=6 physical family, but the available metadata fails the eccentricity gate for the two-resolution pair and does not document the required N2/N3 products or post-peak duration.

## Georgia Tech waveform catalog

Primary sources: [official catalog documentation](https://einstein.gatech.edu/catalog/), [official metadata table](https://raw.githubusercontent.com/cevans216/gt-waveform-catalog/master/catalog-table.txt), and [catalog paper](https://arxiv.org/abs/1605.03204). The official documentation says the 452 catalog waveforms contain modes `l=2:6`, are computed at **finite extraction radius and finite resolution**, and that multiple extraction radii are available in raw simulation data. It does not claim null-infinity strain, N2/N3 extrapolation orders, or a post-peak duration field.

* **q=6:** GT0604 has `q=6.00`, `a1=a2=0`, and the metadata tag `D10_q6.00_a0.00_0.00_m280`. It is a single catalog row; no second numerical resolution or N2/N3 pair is identified. The catalog table has no eccentricity column, so eccentricity is absent/unparseable under the fixed policy. Post-peak duration is also absent. It therefore cannot pass the strict screen.
* **q=8:** GT0886 has `q=8.00` but `a1z=a2z=0.4253` (spinning), so it fails the `|chi| <= 1e-4` gate. No nonspinning q=8 row with the required resolution/order metadata was found. The paper's broad statement that nonspinning systems reach q<=15 does not substitute for a qualifying, documented q=8 pair.
* **Usage/attribution:** the official page directs users to the Georgia Tech catalog paper and provides the table, scripts, and downloads, but states no explicit dataset license or dedicated reuse/acknowledgement text. Cite Jani et al., *Georgia Tech Catalog of Gravitational Waveforms* (arXiv:1605.03204 / CQG 33, 204001); licensing remains unresolved from the public catalog documentation.

**Viability:** fail under the unchanged policy. Georgia Tech offers independent metadata and exact q=6/q=8 rows, but only finite-radius/finite-resolution products are documented; strict eccentricity, N2/N3 order pairing, and post-peak-duration requirements are not met or not verifiable.

## BAM catalogue (Cardiff)

Primary source: [BAM catalogue documentation](https://data.cardiffgravity.org/bam-catalogue/). It describes 80 publicly available single-spin, precessing configurations at four mass ratios and provides `Psi_4` at several extraction radii. The stated scope is not a nonspinning q=6/q=8 catalog, and the page does not document N2/N3 extrapolated strain or the required duration. It is therefore not a candidate under this search.

## Bottom line

No public non-SXS catalog located here supplies an admissible q=6/q=8 nonspinning record set with the fixed policy's two numerical resolutions, explicit N2/N3 extrapolated strain, and `>=150M` post-peak support. Do not fit or alter the calibration policy from these catalogs. The strongest lead is the RIT q=6 pair (0090/0121), but it fails the eccentricity gate and has unresolved order/duration metadata.
