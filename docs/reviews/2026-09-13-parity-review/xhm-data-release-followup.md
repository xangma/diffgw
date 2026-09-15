# XHM 122022 data-release follow-up

Date: 2026-09-14. Scope: clean-room search for a standalone, redistributable
numerical calibration dataset and schema for the default IMRPhenomXHM release
`122022`, with particular attention to the `(3,2)` phase targets. No LAL
implementation source, source-only fit table, private audit cache, or waveform
output was inspected or used to derive coefficients.

## Result

No usable release-matched standalone artifact was located in the checked
primary sources. This is a bounded “not found in the checked sources” result,
not a claim that the artifact does not exist. The current repository therefore
has no admissible input for a `122022` numerical provider and no integration is
made here.

## Evidence

* The author paper, García-Quirós et al., [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2), describes the calibration construction and
  says its ancillary material contains paper-era fit assets. It describes a
  calibration set of hybrid NR/Teukolsky waveforms and fitting procedures, but
  does not identify a `122022` release, a standalone machine-readable schema,
  or redistribution terms for current-release arrays.
* The paper's public [arXiv v2 e-print archive](https://arxiv.org/e-print/2001.10914v2)
  contains `ParspaceFits` and spheroidal `(3,2)` assets. The audit documents
  that these are paper-era intermediate and spheroidal/alignment expressions;
  they do not resolve the current-release spheroidal target condition or the
  rule selecting a reduced versus unreduced system, nor do they identify
  release `122022`.
* A public [LALSimulation waveform-parameter documentation URL](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_waveform_params_8c_source.html)
  appeared in search results, but its direct open returned an internal error;
  no LAL implementation source page was inspected. The version tuple quoted
  here comes from prior repository audit metadata in
  `verification/xhm-inspiral/xhm-inspiral-public.json`: `ReleaseVersion=122022`,
  phase versions `122019`, amplitude fit/frequency versions `122018`, and
  inspiral amplitude variant `3`. Those controls do not publish numerical
  surfaces or their schema.
* The [LIGO review wiki](https://git.ligo.org/waveforms/reviews/imrphenomx/-/wikis/home)
  was identified as a candidate upstream location, but its contents were not
  relied upon or inspected here. The [GWOSC/LIGO data portal](https://www.ligo.caltech.edu/page/ligo-data)
  is a public observation-data portal; neither URL yielded a release-matched
  XHM calibration bundle in the checked material. The unrelated [Zenodo
  generation-X posterior release](https://zenodo.org/records/4079188) contains
  posterior samples, not calibration arrays or a provider schema.

The missing artifact remains the complete set of release-specific numerical
surfaces and metadata listed in `xhm-calibration-critical-path.md`: inspiral
node targets, intermediate amplitude and phase targets, ringdown amplitude and
phase surfaces, `(3,2)` mixing/alignment targets, region masks/cuts, units,
term ordering, precision, and license/redistribution statement.

## Concrete upstream request draft (do not send)

> Subject: Request for redistributable IMRPhenomXHM release 122022 calibration data and schema
>
> Dear IMRPhenomXHM/LALSuite maintainers, we are implementing an independent
> MIT-licensed clean-room implementation and need exact parity with the public
> default IMRPhenomXHM release `122022`. Could you point us to (or publish) the
> release-matched numerical calibration artifact, with a versioned URL or DOI,
> SHA-256, license/redistribution terms, and a machine-readable schema?
>
> Please include the release/version tuple and units, parameter ordering and
> polynomial term ordering for every `(2,1)`, `(3,3)`, `(3,2)`, and `(4,4)`
> surface; inspiral node fractions and geometric/Fourier-strain normalization;
> intermediate amplitude and phase targets with their collocation frequencies;
> ringdown amplitude/phase targets and width factors; `(3,2)` spheroidal
> mixing/alignment coefficients, including the independent-condition contract
> or a citable rule selecting a reduced system; region masks, cuts, vetoes, and
> precision. Please also state which values are inherited from `122019` or
> `122018` and which were recalibrated for `122022`.
>
> If the arrays cannot be redistributed, a primary technical note specifying
> the canonical external location, retrieval procedure, immutable revision,
> schema, and permitted use would still let downstream clean-room projects
> implement an auditable external-data provider.
