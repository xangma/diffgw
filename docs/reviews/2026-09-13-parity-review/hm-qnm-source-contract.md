# IMRPhenomHM QNM source contract (clean-room note)

Date: 2026-09-14. Scope: publication and author-data documentation only. No waveform implementation source, LAL/PyCBC source, fitting to an oracle, or NR data was used here.

## Primary published statements

London et al., *First higher-multipole model* ([arXiv:1708.00404](https://arxiv.org/html/1708.00404)), defines the higher-mode map in Eq. (4). Its frequency map is piecewise linear (Eqs. (5)--(7), rendered as the domain-map equations): inspiral uses `2 f/m`, the intermediate branch linearly joins the mode QNM frequency, and the high-frequency branch uses the QNM-frequency difference. The paper explicitly defines

`f_lm^RD = omega_lm0 / (2 pi)`,

where `omega_lm0` is the real frequency of the fundamental QNM. The same section sets `f0^A = 0.018 f_lm^RD/f_22^RD` and `f0^phi = 0.014 f_lm^RD/f_22^RD`. In its PhenomD application, Eq. (14) replaces the high-frequency map by `f_22(f)=(f_22^RD/f_lm^RD)f` and inserts the same ratio plus the mode damping frequency in the MR arctangent. Thus the relevant inputs are mode-dependent fundamental oscillation and damping frequencies, together with the carrier 22 values.

The same paper states that PhenomHM includes `(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)` and omits spherical-spheroidal mixing. That omission is part of the model definition; it is not evidence that a mixing correction belongs in this target.

Husa et al., *Frequency-domain gravitational waves from non-precessing black-hole binaries I* ([arXiv:1508.07250](https://arxiv.org/html/1508.07250)), gives the remnant/QNM contract. Its Eq. (17) converts a dimensionless Kerr QNM quantity to a dimensionful angular frequency as

`omega_RD = (M omega_RD) / M_f = (M omega_RD)/(M-E_rad)`.

The text immediately before Eq. (17) says that `M omega_RD(a_f)` was obtained by interpolating the data in its Refs. [82,83]. Reference [82] is Berti, Cardoso & Starinets, CQG 26, 163001 (2009), [arXiv:0905.2975](https://arxiv.org/abs/0905.2975); Ref. [83] is the author-hosted gravitational Kerr-QNM tables ([URL cited by the paper](http://www.phy.olemiss.edu/~berti/qnms.html)). The companion PhenomD paper ([arXiv:1508.07253](https://arxiv.org/abs/1508.07253)) is the model-II source cited by London for the PhenomD application.

London & Fauchon-Jones ([arXiv:1810.03550](https://arxiv.org/html/1810.03550)) is later QNM-data methodology, not a statement of the historical PhenomHM data path. It defines the complex QNM frequency with final mass and spin, describes Leaver-computed numerical points, and explicitly says that PhenomHM and derivative models use QNM-frequency fits in the mode mapping (Sec. I). Its recommended fits are named in the paper as `positive.physics.cw181003550`; this is a later fit resource and must not be silently substituted for the historical PhenomHM prescription.

## Data, units, and provenance

The author-hosted [Berti ringdown resource](https://pages.jh.edu/eberti2/ringdown/) is documented as freely available data/routines with citation requested. Its published gravitational-QNM data format is `a/M, M omega_R, M omega_I, Re(A_lmn), Im(A_lmn)`; its public fit file is [fitcoeffsWEB.dat](https://pages.jh.edu/eberti2/ringdown/fitcoeffsWEB.dat), with rows `l,m,n,f1,f2,f3,q1,q2,q3` and the formulas shown on the page. The resource links separate SharePoint archives for the gravitational `s=-2` tables at `l=2,3,4` and a Leaver-method calculator. These are author data products, not MIT-licensed TorchWave source; any clean use requires preserving the cited attribution and the downloaded-file checksum. In this run, the direct shell request returned HTTP 403, while the author-hosted text was independently retrieved through the public page interface and preserved as an external `/tmp` copy for the diagnostic. Its SHA256 is `665bbb7edafadda81ab26d2c33899b6afd7689f0d549e991dcb2ed7505661c92`; this is not redistributed under the repository license. Direct SharePoint archive downloads returned authorization/time-out failures, so no archive bytes were used as numerical evidence.

| quantity | published representation | conversion required in a comparison |
|---|---|---|
| Kerr table frequency | dimensionless `M omega_R`, with spin `a/M` | `f_Hz = (M_BH omega_R)/(2 pi M_BH_seconds)` when `M` is the Kerr remnant mass in the table; equivalently `M_total f=(M_BH omega_R)/(2 pi)/(M_BH/M_total)` when reporting total-mass units. Use PhenomD Eq. (17) with `M_f` for the binary remnant conversion |
| Kerr damping | dimensionless imaginary/damping entry or the corresponding quality factor | preserve the table's sign convention; convert to the same positive/decay-rate convention before using London Eq. (14) |
| London input | `f_lm^RD = omega_lm0/(2 pi)` | use the fundamental (`n=0` in a Leaver convention only after verifying the table's overtone indexing); keep Hz and dimensionless `M f` separate |
| PhenomD remnant | final spin `a_f`, final mass `M_f` | interpolate dimensionless Kerr data at `a_f`, then divide by `M_f` for the dimensionful angular frequency as Eq. (17) states |

The author page's fit file contains n=0 entries for the six HM modes (22, 21, 33, 32, 44, 43), in addition to other modes. Its table files and Leaver calculators may expose different overtone numbering conventions; the selected file header must be checked before any numerical comparison. No interpolation order, table revision, damping sign normalization, or extrapolation rule is specified by the PhenomHM paper. PhenomD says only “interpolated the data set”; it does not identify the interpolation order or exact historical file revision. The later 1810.03550 fits are therefore useful as a separately labeled data product, not proof of what PhenomHM used. `diagnose_hm_qnm_leaver_clean.py` now evaluates the author n=0 fit rows as a separate, non-selected branch using `M_total/M_f` conversion.

## What is established and what remains unknown

Established from primary text: the HM high-frequency phase/amplitude map depends on `f_lm^RD`, `f_lm^damp`, and carrier 22 QNM quantities; the mode frequency is the fundamental QNM; remnant-mass conversion matters; PhenomD's historical source is Berti data/interpolation; no spheroidal mixing is part of this model.

Unresolved from the allowed sources: the exact Berti file revision used by PhenomD/HM, interpolation method/order and endpoint handling, whether higher-mode values were directly interpolated from Berti data or supplied by another published fit, and the precise damping sign/quality-factor convention at the HM call boundary. Public EOB QNM functions and a Leaver calculation are independent controls, not evidence of the historical HM choice.

Two existing diagnostic labels must not be treated as evidence. `qnm_ratio_samples` in an earlier revision divided dimensionless `M f`-style values by samples in Hz; those ratios were dimensionally invalid. The current revision converts samples to the same total-mass `M f` convention. Earlier diagnostic versions divided dimensionless `M f` values by Hz and used `ordinary_assembly_path`, a same-function tautology. The current diagnostic converts samples to total-mass `M f` and uses an independent direct 22 carrier-times-amplitude identity; that identity is only a mode-22 control.

## Next executable, publication-grounded experiment

1. The diagnostic preserves the author fit URL, external-copy SHA256, column format, n=0 selection, and retrieval limitation; a future exact archive download must additionally preserve response headers and stated spin domain.
2. For each frozen HM case, use the same independently documented remnant `(M_f,a_f)` and evaluate the six fundamental mode frequencies and damping rates. Convert all values either to physical Hz or to dimensionless `M f`; never mix them.
3. Compare three explicitly labeled controls: historical-style Berti interpolation, the later 1810.03550 fit, and the independent Leaver solver. Keep Eq. (4), the London piecewise map, and Eq. (14) fixed. Report joins and per-mode amplitude/phase/raw residuals; do not select or fit a prescription from waveform residuals.
4. Correct the diagnostic's unit bookkeeping and replace the same-function assembly guard with an independently computed identity (for example, a direct 22 carrier control) before using any QNM conclusion.

This experiment can determine whether the remaining higher-mode discrepancy is compatible with a documented QNM-data choice. It cannot, from the currently published descriptions alone, recover an undocumented historical interpolation rule.
