# Calibration reconstruction feasibility

Date: 2026-09-15. Bounded primary-source review for a clean-room, independently
recalibrated XHM/TEOB implementation. No LAL source, source-only calibration
tables, waveform outputs, fitting, or bulk data were inspected.

## XHM: what is public and reconstructible

The primary paper is García-Quirós et al., [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2).
It states that XHM used 504 hybrids: 466 comparable-mass cases
(`q≤18`) and 38 EMRI/Teukolsky cases at q=200 and 1000. The comparable-mass
hybrids join SEOBNRv4/PN inspirals to NR from SXS, BAM, and Einstein Toolkit;
the large-q boundary joins EOB to perturbative Teukolsky waveforms. The paper
gives the mode ansätze, alignment convention, transition frequencies,
collocation locations, direct-fit then parameter-space-fit architecture, and
the hierarchical-fit idea. It links the author ancillary archive, including
the amplitude/phase coefficient files and the QNM-fit files. The cited NR
catalogue entry is the public [SXS/SpEC waveform page](https://www.black-holes.org/waveforms).

An independent *new* fit is therefore feasible if the public hybrid/NR inputs
are obtained and the following are reimplemented and recorded: waveform IDs
and revisions, hybridization and 22-mode alignment, mode sign/tetrad handling,
per-quantity quality exclusions, direct-fit windows and weights, collocation
target extraction, hierarchical basis/order/model-selection, and the exact
QNM/remnant inputs. The paper explicitly says higher-mode data are noisy and
that exclusions are quantity-specific; these choices materially affect a fit.

## Exact-release gap

The paper-era ancillary coefficients are useful controls, but are not evidence
of the installed `122022` release. The paper says the later SXS catalogue
expanded the set to 355 SpEC simulations and updated phase fits, while amplitude
fits were deliberately left unchanged. The local audit records `122022` as a
later amplitude release with changed data/cuts/collocation/normalization, but
does not identify a complete public `122022` training manifest, annotations,
or release-specific fit recipe. Thus a clean refit can be scientifically
credible and independently licensed according to the underlying data terms,
but must be named/versioned as a new calibration. Exact parity with `122022`
could still be obtained by reproducing the original fitting pipeline from its
exact inputs, versions, preprocessing, selections, and objective; numerical
surfaces are a sufficient shortcut, not a logical requirement. Current public
evidence does not establish that reproducibility for `122022`.

## TEOB Eq. 91: recipe and gaps

The primary [TEOB paper, arXiv:1806.01772v2](https://arxiv.org/abs/1806.01772v2),
Appendix F.1.2 states that the QNM targets are public Berti-table data, with
final spin computed using Jiménez-Forteza et al., and prints the full rational
form and coefficients of Eq. 91. The [Berti landing page](https://pages.jh.edu/eberti2/ringdown/)
exposes Kerr tables, fit files, and a Leaver-method notebook; it says the data
and routines are freely available, with a near-extremal reliability caveat.
The paper does not specify the original spin grid/domain, point weights,
normalization beyond factoring the zero-spin limit, optimizer/objective,
rounding/precision, or outlier policy. Therefore Eq. 91 is independently
reimplementable and physically refittable, but exact reproduction of the
published fit is not demonstrated from the paper alone.

## Hybrid archive availability

The XHM paper's reference [34](https://arxiv.org/html/2001.10914v2#S7.R34)
is explicitly “in preparation” (Husa et al., 2020), with no linked hybrid
archive, file manifest, version tag, or license. Its NR references point to
the [SXS/SpEC waveform page](https://www.black-holes.org/waveforms), a source
catalogue rather than an identified XHM hybrid release; no exact hybrid files
or training manifest were exposed by the inspected primary pages. No data were
downloaded. Thus public NR availability supports a new calibration project,
but does not establish access to the exact XHM training set or its release
terms.

The arXiv page identifies its terms as an arXiv perpetual non-exclusive
license; that is not a blanket redistribution grant for SXS, BAM, or
Teukolsky data. Each source catalogue's terms and attribution must be retained
in provenance before shipping derived coefficients.
