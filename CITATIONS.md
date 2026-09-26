# Scientific Citations and Provenance

The models follow the primary literature cited below and identified author calibration releases. Software adaptations, data provenance and unresolved terms are recorded in the [provenance ledger](docs/provenance_ledger.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

If you use `torchwave` in scientific work, please cite the underlying models and papers:

---

## 1. Post-Newtonian Theory & TaylorF2

The [aligned-spin mathematical reference](docs/math/taylorf2-aligned-spin.md)
uses [Wade et al., arXiv:1306.3901v2](https://arxiv.org/abs/1306.3901v2),
Eq. (4) and spin definitions (Phys. Rev. D 88, 083002), and
[Mishra, Kela, Arun and Faye, arXiv:1601.05588v1](https://arxiv.org/abs/1601.05588v1),
Eqs. (5)–(7) (Phys. Rev. D 93, 084054). The 3.5PN quadratic-spin phase uses
energy balance with [Bohé et al., arXiv:1501.01529v1](https://arxiv.org/abs/1501.01529v1),
Eqs. (3.33), (4.14), and the corrected spin-spin tail flux in
[Messina–Nagar, arXiv:1703.08107v4](https://arxiv.org/abs/1703.08107v4), Eq. (26).
The resulting correction is also printed directly in
[Pratten et al., arXiv:2001.11412v1](https://arxiv.org/abs/2001.11412v1),
Eq. (A24), supplementing that paper's baseline Eq. (A19).

The `TaylorF2_HM` author-source component gates also use the Mathematica-readable
`supl-mkaf16.m` file in the author-published arXiv v1 source bundle for
Mishra et al. Its bundle SHA256 is
`f0b6c6810da95f45141dbb9f0c5cccec0ba0742f60b836dba7698c90a25d5154`
and the member SHA256 is
`501ea53c29b77e061585540b9ccfd69a4cd0283e805ea1e9c24c38800e627a70`.
The repository retains generated amplitude-ratio evaluations, a transformed
normalized nonspinning `TF28Phasing` expression, and provenance. The
complex component gate combines those records with the paper's Eq. (11)
absolute normalization; the repository does not retain the author source file.
The transformed expression remains third-party-derived verification material
with unresolved redistribution terms; the repository's MIT grant does not
assert ownership of it.

The [quadrupolar tidal reference](docs/math/taylorf2-tidal.md) uses
[Wade et al., arXiv:1402.5156v1](https://arxiv.org/abs/1402.5156v1),
Eqs. (4)–(6) and (A26), for the 5PN/6PN phase and tidal parameters, and
[Damour, Nagar and Villain, arXiv:1203.4352v1](https://arxiv.org/abs/1203.4352v1),
Eqs. (B8)–(B12), (B14), for the 6.5PN tail and an independent component-mass
expression. The specification derives the conversion between their conventions.

The [spin-induced quadrupole reference](docs/math/taylorf2-spin-quadrupole.md)
uses [Nagar et al., arXiv:1812.07923v1](https://arxiv.org/pdf/1812.07923v1),
Eqs. (44), (46), (47), (50)–(52), for quadratic-spin terms and their tail;
[Dietrich et al., arXiv:1905.06011v2](https://arxiv.org/pdf/1905.06011v2),
Eqs. (26)–(27), for the quadrupole-dependent cubic-spin term; and
[Marsat, arXiv:1411.4118v2](https://arxiv.org/pdf/1411.4118v2),
Eqs. (6.20)–(6.21), as an independent cubic-spin cross-check.

```bibtex
@article{Blanchet:2013haa,
    author = "Blanchet, Luc",
    title = "{Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries}",
    eprint = "1310.1528",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.12942/lrr-2014-2",
    journal = "Living Rev. Rel.",
    volume = "17",
    pages = "2",
    year = "2014"
}

@article{Buonanno:2009zt,
    author = "Buonanno, Alessandra and Iyer, Bala R. and Ochsner, Evan and Pan, Yi and Sathyaprakash, B. S.",
    title = "{Comparison of post-Newtonian templates for compact binary inspiral signals in gravitational-wave detectors}",
    eprint = "0907.0700",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.80.084043",
    journal = "Phys. Rev. D",
    volume = "80",
    pages = "084043",
    year = "2009"
}

@article{Arun:2008kb,
    author = "Arun, K. G. and Buonanno, Alessandra and Faye, Guillaume and Ochsner, Evan",
    title = "{Higher-order spin effects in the amplitude and phase of gravitational waveforms emitted by inspiraling compact binaries: Ready-to-use gravitational waveforms}",
    eprint = "0810.5336",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.79.104023",
    journal = "Phys. Rev. D",
    volume = "79",
    pages = "104023",
    year = "2009",
    note = "[Erratum: Phys.Rev.D 84, 049901 (2011)]"
}

@article{Bohe:2013cla,
    author = "Boh{\'e}, Alejandro and Marsat, Sylvain and Blanchet, Luc",
    title = "{Next-to-next-to-leading order spin-orbit effects in the gravitational wave flux and orbital phasing of compact binaries}",
    eprint = "1303.7412",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/0264-9381/30/13/135009",
    journal = "Class. Quant. Grav.",
    volume = "30",
    pages = "135009",
    year = "2013"
}

@article{Flanagan:2007ix,
    author = "Flanagan, {\'E}anna {\'E}. and Hinderer, Tanja",
    title = "{Constraining neutron star tidal Love numbers with gravitational wave detectors}",
    eprint = "0709.1915",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph",
    doi = "10.1103/PhysRevD.77.021502",
    journal = "Phys. Rev. D",
    volume = "77",
    pages = "021502",
    year = "2008"
}

@article{Vines:2011ud,
    author = "Vines, Justin and Hinderer, Tanja and Flanagan, {\'E}anna {\'E}.",
    title = "{Post-1-Newtonian tidal effects in the gravitational waveform from binary inspirals}",
    eprint = "1101.1673",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.83.084051",
    journal = "Phys. Rev. D",
    volume = "83",
    pages = "084051",
    year = "2011"
}

@article{Wade:2014vqa,
    author = "Wade, Leslie and Creighton, Jolien D. E. and Ochsner, Evan and Lackey, Benjamin D. and Farr, Benjamin F. and Littenberg, Tyson B. and Raymond, Vivien",
    title = "{Systematic and statistical errors in a bayesian approach to the estimation of the neutron-star equation of state using advanced gravitational wave detectors}",
    eprint = "1402.5156",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.89.103012",
    journal = "Phys. Rev. D",
    volume = "89",
    number = "10",
    pages = "103012",
    year = "2014"
}
```

---

## 2. IMRPhenomD Phenomenological Model

The current PhenomD fundamental Kerr-mode representation uses eigenvalues from the native Kerr solver, with rounding and natural cubic interpolation described in the [provenance ledger](docs/provenance_ledger.md). The earlier spectral representation used development eigenproblem solves from [Stein (2019), qnm](https://doi.org/10.21105/joss.01683); its [historical derivation and validation record](https://github.com/xangma/diffgw/blob/archive/clean-room-reviews/docs/reviews/2026-09-11-independent/qnm.md) describes that earlier implementation.

```bibtex
@article{Husa:2015iqa,
    author = "Husa, Sascha and Khan, Sebastian and Hannam, Mark and P{\"u}rrer, Michael and Ohme, Frank and Jim{\'e}nez Forteza, Xisco and Boh{\'e}, Alejandro",
    title = "{Frequency-domain gravitational waves from nonprecessing black-hole binaries. I. New numerical waveforms and anatomy of the signal}",
    eprint = "1508.07250",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.93.044006",
    journal = "Phys. Rev. D",
    volume = "93",
    number = "4",
    pages = "044006",
    year = "2016"
}

@article{Khan:2015jqa,
    author = "Khan, Sebastian and Husa, Sascha and Hannam, Mark and Ohme, Frank and P{\"u}rrer, Michael and Jim{\'e}nez Forteza, Xisco and Boh{\'e}, Alejandro",
    title = "{Frequency-domain gravitational waves from nonprecessing black-hole binaries. II. A phenomenological model for the advanced detector era}",
    eprint = "1508.07253",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.93.044007",
    journal = "Phys. Rev. D",
    volume = "93",
    number = "4",
    pages = "044007",
    year = "2016"
}

@article{Berti:2005ys,
    author = "Berti, Emanuele and Cardoso, Vitor and Will, Clifford M.",
    title = "{On gravitational-wave spectroscopy of massive black holes with the space interferometer LISA}",
    eprint = "gr-qc/0512160",
    archivePrefix = "arXiv",
    doi = "10.1103/PhysRevD.73.064030",
    journal = "Phys. Rev. D",
    volume = "73",
    pages = "064030",
    year = "2006"
}
```

---

## 3. Effective-One-Body (SEOBNR) Theory & Models

```bibtex
@article{Bohe:2016gbl,
    author = "Boh{\'e}, Alejandro and others",
    title = "{An improved effective-one-body model of spinning, nonprecessing binary black holes for the era of gravitational-wave astrophysics with advanced detectors}",
    eprint = "1611.03703",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.95.044028",
    journal = "Phys. Rev. D",
    volume = "95",
    number = "4",
    pages = "044028",
    year = "2017"
}

@article{Pompili:2023tfa,
    author = "Pompili, Lorenzo and others",
    title = "{Laying the foundation of the effective-one-body waveform models SEOBNRv5: Improved accuracy and efficiency for spinning nonprecessing binary black holes}",
    eprint = "2303.18039",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.108.124035",
    journal = "Phys. Rev. D",
    volume = "108",
    number = "12",
    pages = "124035",
    year = "2023"
}

@article{Ramos-Buades:2023ehm,
    author = "Ramos-Buades, Antoni and others",
    title = "{SEOBNRv5PHM: Next generation of accurate and efficient multipolar precessing-spin effective-one-body waveforms for binary black holes}",
    eprint = "2303.18046",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.108.124037",
    journal = "Phys. Rev. D",
    volume = "108",
    number = "12",
    pages = "124037",
    year = "2023"
}

@article{Purrer:2014fza,
    author = "P{\"u}rrer, Michael",
    title = "{Frequency domain reduced order models for gravitational waves from aligned-spin compact binaries}",
    eprint = "1402.4146",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/0264-9381/31/19/195010",
    journal = "Class. Quant. Grav.",
    volume = "31",
    number = "19",
    pages = "195010",
    year = "2014"
}

@article{Purrer:2015tud,
    author = "P{\"u}rrer, Michael",
    title = "{Frequency domain reduced order model of aligned-spin effective-one-body waveforms with generic mass-ratios and spins}",
    eprint = "1512.02248",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.93.064041",
    journal = "Phys. Rev. D",
    volume = "93",
    number = "6",
    pages = "064041",
    year = "2016"
}
```

---

## 4. Tidal IMR Waveforms (NRTidal & NRTidalv2)

```bibtex
@article{Dietrich:2017aum,
    author = "Dietrich, Tim and Bernuzzi, Sebastiano and Tichy, Wolfgang",
    title = "{Closed-form tidal approximants for binary neutron star gravitational waveforms constructed from high-resolution numerical relativity simulations}",
    eprint = "1706.02969",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.96.121501",
    journal = "Phys. Rev. D",
    volume = "96",
    number = "12",
    pages = "121501",
    year = "2017"
}

@article{Dietrich:2018uni,
    author = "Dietrich, Tim and others",
    title = "{Matter imprints in waveform models for neutron star binaries: tidal and self-spin effects}",
    eprint = "1804.02235",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.99.024029",
    journal = "Phys. Rev. D",
    volume = "99",
    number = "2",
    pages = "024029",
    year = "2019"
}

@article{Dietrich:2019kaq,
    author = "Dietrich, Tim and others",
    title = "{Improving the NRTidal model for binary neutron star systems}",
    eprint = "1905.06011",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.100.044003",
    journal = "Phys. Rev. D",
    volume = "100",
    number = "4",
    pages = "044003",
    year = "2019"
}
```

---

## 5. Multipolar Higher Harmonic Modes & Quasi-Normal Modes

```bibtex
@article{Berti:2005ys,
    author = "Berti, Emanuele and Cardoso, Vitor and Will, Clifford M.",
    title = "{On gravitational-wave spectroscopy of massive black holes with the space interferometer LISA}",
    eprint = "gr-qc/0512160",
    doi = "10.1103/PhysRevD.73.064030",
    journal = "Phys. Rev. D",
    volume = "73",
    pages = "064030",
    year = "2006"
}

@article{Kidder:2007rt,
    author = "Kidder, Lawrence E.",
    title = "{Using Full Information When Computing Modes of Post-Newtonian Waveforms From Inspiralling Compact Binaries in Circular Orbit}",
    eprint = "0710.0614",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.77.044016",
    journal = "Phys. Rev. D",
    volume = "77",
    pages = "044016",
    year = "2008"
}

@article{London:2017bcn,
    author = "London, Lionel and Khan, Sebastian and Fauchon-Jones, Edward and Garc{\'i}a, Cecilio and Hannam, Mark and Husa, Sascha and Jim{\'e}nez-Forteza, Xisco and Kalaghatgi, Chinmay and Ohme, Frank and Pannarale, Francesco",
    title = "{First higher multipole model of gravitational waves from spinning and coalescing black-hole binaries}",
    eprint = "1708.00404",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevLett.120.161102",
    journal = "Phys. Rev. Lett.",
    volume = "120",
    number = "16",
    pages = "161102",
    year = "2018"
}

@article{Garcia-Quiros:2020qpx,
    author = "Garc{\'i}a-Quir{\'o}s, Cecilio and Colleoni, Marta and Husa, Sascha and Estell{\'e}s, H{\'e}ctor and Pratten, Geraint and Ramos-Buades, Antoni and Mateu-Lucena, Maite and Jaume, Rafel",
    title = "{Multimode frequency-domain model for the gravitational wave signal from nonprecessing black-hole binaries}",
    eprint = "2001.10914",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.102.064002",
    journal = "Phys. Rev. D",
    volume = "102",
    number = "6",
    pages = "064002",
    year = "2020"
}

@article{Mezzasoma:2022qvh,
    author = "Mezzasoma, Stefano and Yunes, Nicol{\'a}s",
    title = "{Theory-agnostic framework for inspiral tests of general relativity with higher-harmonic gravitational waves}",
    eprint = "2203.15934",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.106.024026",
    journal = "Phys. Rev. D",
    volume = "106",
    number = "2",
    pages = "024026",
    year = "2022"
}

@article{Mishra:2016whh,
    author = "Mishra, Chandra Kant and Kela, Aditya and Arun, K. G. and Faye, Guillaume",
    title = "{Ready-to-use post-Newtonian gravitational waveforms for binary black holes with nonprecessing spins: An update}",
    eprint = "1601.05588",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.93.084054",
    journal = "Phys. Rev. D",
    volume = "93",
    number = "8",
    pages = "084054",
    year = "2016"
}
```

---

## 6. Detector Networks & Coordinate Projections

```bibtex
@article{Anderson:2000yy,
    author = "Anderson, Warren G. and Brady, Patrick R. and Creighton, Jolien D. E. and Flanagan, {\'E}anna {\'E}.",
    title = "{An excess power statistic for detection of burst sources of gravitational radiation}",
    eprint = "gr-qc/0008066",
    doi = "10.1103/PhysRevD.63.042003",
    journal = "Phys. Rev. D",
    volume = "63",
    pages = "042003",
    year = "2001"
}

@article{Schutz:2011tw,
    author = "Schutz, Bernard F.",
    title = "{Networks of gravitational wave detectors and three figures of merit}",
    eprint = "1102.5421",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1088/0264-9381/28/12/125023",
    journal = "Class. Quant. Grav.",
    volume = "28",
    pages = "125023",
    year = "2011"
}

@article{Allen:2005fk,
    author = "Allen, Bruce and Anderson, Warren G. and Brady, Patrick R. and Brown, Duncan A. and Creighton, Jolien D. E.",
    title = "{FINDCHIRP: An Algorithm for detection of gravitational waves from inspiraling compact binaries}",
    eprint = "gr-qc/0509116",
    doi = "10.1103/PhysRevD.85.122006",
    journal = "Phys. Rev. D",
    volume = "85",
    pages = "122006",
    year = "2012"
}
```

---

## 7. Precessing Multipolar Higher Harmonics (`IMRPhenomXPHM`)

```bibtex
@article{Pratten:2020ceb,
    author = "Pratten, Geraint and others",
    title = "{Computationally efficient models for the dominant and sub-dominant harmonic modes of precessing binary black holes}",
    eprint = "2004.06503",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.103.104056",
    journal = "Phys. Rev. D",
    volume = "103",
    pages = "104056",
    year = "2021"
}

@article{Khan:2019kot,
    author = "Khan, Sebastian and others",
    title = "{Including higher order multipoles in gravitational-wave models for precessing binary black holes}",
    eprint = "1911.06050",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.101.024056",
    journal = "Phys. Rev. D",
    volume = "101",
    pages = "024056",
    year = "2020"
}
```

---

## 8. NSBH Tidal Disruption & Remnant Model (`IMRPhenomNSBH`)

```bibtex
@article{Pannarale:2015jka,
    author = "Pannarale, Francesco and others",
    title = "{Gravitational-wave cutoff frequencies of tidally disruptive neutron star-black hole binary mergers}",
    eprint = "1509.06209",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.92.081504",
    journal = "Phys. Rev. D",
    volume = "92",
    pages = "081504",
    year = "2015"
}

@article{Matas:2020wab,
    author = "Matas, Andrew and others",
    title = "{An aligned-spin neutron-star--black-hole waveform model based on the effective-one-body approach and numerical-relativity simulations}",
    eprint = "2004.10001",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.102.043023",
    journal = "Phys. Rev. D",
    volume = "102",
    pages = "043023",
    year = "2020"
}

@article{Yagi:2013bca,
    author = "Yagi, Kent and Yunes, Nicol{\'a}s",
    title = "{I-Love-Q: Unexpected Universal Relations for Neutron Stars and Quark Stars}",
    eprint = "1302.4499",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1126/science.1236462",
    journal = "Science",
    volume = "341",
    pages = "365--368",
    year = "2013"
}
```

---

## 9. Eccentric Post-Newtonian Waveforms (`EccentricTaylorF2`)

```bibtex
@article{Moore:2016qxz,
    author = "Moore, Blake and Favata, Marc and Arun, K. G. and Mishra, Chandra Kant",
    title = "{Gravitational-wave phasing for low-eccentricity inspiralling compact binaries to 3PN order}",
    eprint = "1605.00304",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.93.124061",
    journal = "Phys. Rev. D",
    volume = "93",
    pages = "124061",
    year = "2016"
}

@article{Yunes:2009yz,
    author = "Yunes, Nicol{\'a}s and Arun, K. G. and Berti, Emanuele and Will, Clifford M.",
    title = "{Post-Circular Expansion of Eccentric Binary Inspirals: Fourier-Domain Waveforms in the Stationary Phase Approximation}",
    eprint = "0906.0313",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.80.084001",
    journal = "Phys. Rev. D",
    volume = "80",
    pages = "084001",
    year = "2009"
}
```

---

## 10. Kerr Black Hole Ringdown Overtones & Spectroscopy (`KerrRingdown`)

```bibtex
@article{Berti:2005ys,
    author = "Berti, Emanuele and Cardoso, Vitor and Will, Clifford M.",
    title = "{On gravitational-wave spectroscopy of massive black holes with the space interferometer LISA}",
    eprint = "gr-qc/0512160",
    doi = "10.1103/PhysRevD.73.064030",
    journal = "Phys. Rev. D",
    volume = "73",
    pages = "064030",
    year = "2006"
}

@article{Giesler:2019uxc,
    author = "Giesler, Matthew and Isi, Maximiliano and Scheel, Mark A. and Teukolsky, Saul A.",
    title = "{Black Hole Ringdown: The Importance of Overtones}",
    eprint = "1903.08284",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevX.9.041060",
    journal = "Phys. Rev. X",
    volume = "9",
    pages = "041060",
    year = "2019"
}
```

---

## 11. Differentiable Likelihood & Analytical Detector PSD Suite

```bibtex
@article{Cutler:1994ys,
    author = "Cutler, Curt and Flanagan, {\'E}anna {\'E}.",
    title = "{Gravitational waves from merging compact binaries: How accurately can one extract the binary's parameters from the inspiral waveform?}",
    eprint = "gr-qc/9402014",
    doi = "10.1103/PhysRevD.49.2658",
    journal = "Phys. Rev. D",
    volume = "49",
    pages = "2658--2697",
    year = "1994"
}

@article{Veitch:2014wba,
    author = "Veitch, John and others",
    title = "{Robust parameter estimation for compact binaries with ground-based gravitational-wave observations using the LALInference software library}",
    eprint = "1409.7215",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.91.042003",
    journal = "Phys. Rev. D",
    volume = "91",
    pages = "042003",
    year = "2015"
}

@article{Thrane:2018qnx,
    author = "Thrane, Eric and Talbot, Colm",
    title = "{An introduction to Bayesian inference in gravitational-wave astronomy: parameter estimation, model selection, and hierarchical models}",
    eprint = "1809.02293",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1017/pasa.2019.2",
    journal = "Publ. Astron. Soc. Austral.",
    volume = "36",
    pages = "e010",
    year = "2019"
}

@article{Hild:2010id,
    author = "Hild, Stefan and others",
    title = "{Sensitivity Studies for Third-Generation Gravitational Wave Observatories}",
    eprint = "1012.0908",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/0264-9381/28/9/094013",
    journal = "Class. Quant. Grav.",
    volume = "28",
    pages = "094013",
    year = "2011"
}

@article{Evans:2021gyd,
    author = "Evans, Matthew and others",
    title = "{A Horizon Study for Cosmic Explorer: Science, Observatories, and Community}",
    eprint = "2109.09882",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    year = "2021"
}

@article{Vallisneri:2007ev,
    author = "Vallisneri, Michele",
    title = "{Use and abuse of the Fisher information matrix in the assessment of gravitational-wave parameter-estimation prospects}",
    eprint = "gr-qc/0703086",
    doi = "10.1103/PhysRevD.77.042001",
    journal = "Phys. Rev. D",
    volume = "77",
    pages = "042001",
    year = "2008"
}

@article{Pratten:2020fqn,
    author = "Pratten, Geraint and others",
    title = "{Setting the cornerstone for the IMRPhenomX family of models for gravitational waves from compact binaries: The dominant harmonic for non-precessing quasi-circular black holes}",
    eprint = "2001.11412",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.102.064001",
    journal = "Phys. Rev. D",
    volume = "102",
    pages = "064001",
    year = "2020"
}

@article{Chatziioannou:2017tdb,
    author = "Chatziioannou, Katerina and Klein, Antoine and Yunes, Nicol{\'a}s and Cornish, Neil",
    title = "{Constructing Gravitational Waves from Generic Spin-Precessing Compact Binary Inspirals}",
    eprint = "1703.03967",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.95.104004",
    journal = "Phys. Rev. D",
    volume = "95",
    pages = "104004",
    year = "2017"
}

@article{Kidder:1995zr,
    author = "Kidder, Lawrence E.",
    title = "{Coalescing binary systems of compact objects to (post)$^{5/2}$-Newtonian order. V. Spin effects}",
    eprint = "gr-qc/9506022",
    doi = "10.1103/PhysRevD.52.821",
    journal = "Phys. Rev. D",
    volume = "52",
    pages = "821--847",
    year = "1995"
}

@article{Racine:2008qv,
    author = "Racine, Etienne",
    title = "{Analysis of spin precession in binary black hole systems including quadrupole-monopole interaction}",
    eprint = "0803.1820",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.78.044021",
    journal = "Phys. Rev. D",
    volume = "78",
    pages = "044021",
    year = "2008"
}

@article{Robson:2018ifk,
    author = "Robson, Travis and Cornish, Neil J. and Liu, Chang",
    title = "{The construction and use of LISA sensitivity curves}",
    eprint = "1803.01944",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1088/1361-6382/ab1101",
    journal = "Class. Quant. Grav.",
    volume = "36",
    pages = "105011",
    year = "2019"
}

@article{Tinto:2004wu,
    author = "Tinto, Massimo and Dhurandhar, Sanjeev V.",
    title = "{Time-delay interferometry}",
    eprint = "gr-qc/0409034",
    doi = "10.12942/lrr-2005-4",
    journal = "Living Rev. Rel.",
    volume = "8",
    pages = "4",
    year = "2005"
}
```

---

## 15. Relative Binning (Heterodyned Likelihood)

```bibtex
@article{Zackay:2018qdy,
    author = "Zackay, Barak and Dai, Liang and Venumhav, Tejaswi",
    title = "{Relative Binning and Fast Likelihood Evaluation for Gravitational Wave Parameter Estimation}",
    eprint = "1806.08792",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    year = "2018"
}

@article{Cornish:2010qw,
    author = "Cornish, Neil J.",
    title = "{Fast Fisher Matrices and Lazy Likelihoods}",
    eprint = "1007.4820",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    year = "2010"
}
```

---

## 16. Relativistic EMRI Kludge Models

```bibtex
@article{Barack:2003fp,
    author = "Barack, Leor and Cutler, Curt",
    title = "{LISA capture sources: Approximate waveforms, signal-to-noise ratios, and parameter estimation accuracy}",
    eprint = "gr-qc/0310125",
    doi = "10.1103/PhysRevD.69.082005",
    journal = "Phys. Rev. D",
    volume = "69",
    pages = "082005",
    year = "2004"
}

@article{Babak:2006uv,
    author = "Babak, Stanislav and Fang, Hua and Gair, Jonathan R. and Glampedakis, Kostas and Hughes, Scott A.",
    title = "{'Kludge' gravitational waveforms for a test-body orbiting a Kerr black hole}",
    eprint = "gr-qc/0607007",
    doi = "10.1103/PhysRevD.75.024005",
    journal = "Phys. Rev. D",
    volume = "75",
    pages = "024005",
    year = "2007"
}

@article{Chua:2020stf,
    author = "Chua, Alvin J. K. and Katz, Michael L. and Warburton, Niels and Hughes, Scott A.",
    title = "{Rapid generation of fully relativistic extreme-mass-ratio-inspiral waveform templates for LISA data analysis}",
    eprint = "2008.06071",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevLett.126.051102",
    journal = "Phys. Rev. Lett.",
    volume = "126",
    pages = "051102",
    year = "2021"
}
```

---

## 17. Pulsar Timing Arrays & Hellings-Downs Correlation

```bibtex
@article{Hellings:1983fr,
    author = "Hellings, R. W. and Downs, G. S.",
    title = "{Upper limits on the isotropic gravitational radiation background from pulsar timing analysis}",
    doi = "10.1086/183954",
    journal = "Astrophys. J. Lett.",
    volume = "265",
    pages = "L39--L42",
    year = "1983"
}

@article{Arzoumanian:2020vkk,
    author = "Arzoumanian, Zaven and others",
    collaboration = "NANOGrav",
    title = "{The NANOGrav 12.5 yr Data Set: Search for an Isotropic Stochastic Gravitational-wave Background}",
    eprint = "2009.04496",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.3847/2041-8213/abd401",
    journal = "Astrophys. J. Lett.",
    volume = "905",
    pages = "L34",
    year = "2020"
}

@article{Allen:2022dzg,
    author = "Allen, Bruce",
    title = "{Variance of the Hellings-Downs correlation}",
    eprint = "2205.05637",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.107.043018",
    journal = "Phys. Rev. D",
    volume = "107",
    pages = "043018",
    year = "2023"
}
```

---

## 18. Parameterized Post-Einsteinian (ppE) and Modified Dispersion

```bibtex
@article{Yunes:2009bv,
    author = "Yunes, Nicolas and Pretorius, Frans",
    title = "{Fundamental Theoretical Bias in Gravitational Wave Astrophysics and the Parameterized Post-Einsteinian Framework}",
    eprint = "0909.3328",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.80.122003",
    journal = "Phys. Rev. D",
    volume = "80",
    pages = "122003",
    year = "2009"
}

@article{Mirshekari:2011yq,
    author = "Mirshekari, Saeed and Yunes, Nicolas and Will, Clifford M.",
    title = "{Constraining Lorentz-violating, Modified Dispersion Relations with Gravitational Waves}",
    eprint = "1110.2720",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.85.024041",
    journal = "Phys. Rev. D",
    volume = "85",
    pages = "024041",
    year = "2012"
}
```

---

## 19. No-U-Turn Sampler (NUTS) & Hamiltonian Monte Carlo

```bibtex
@article{Hoffman:2011nc,
    author = "Hoffman, Matthew D. and Gelman, Andrew",
    title = "{The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo}",
    eprint = "1111.4246",
    archivePrefix = "arXiv",
    primaryClass = "stat.CO",
    journal = "J. Mach. Learn. Res.",
    volume = "15",
    pages = "1593--1623",
    year = "2014"
}

@article{Betancourt:2017cba,
    author = "Betancourt, Michael",
    title = "{A Conceptual Introduction to Hamiltonian Monte Carlo}",
    eprint = "1701.02434",
    archivePrefix = "arXiv",
    primaryClass = "stat.ME",
    year = "2017"
}
```

---

## 20. Resummed Tidal Effective-One-Body (TEOBResumS)

```bibtex
@article{Damour:2009wj,
    author = "Damour, Thibault and Nagar, Alessandro",
    title = "{Effective One Body description of tidal effects in inspiralling compact binaries}",
    eprint = "0911.5041",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.81.084016",
    journal = "Phys. Rev. D",
    volume = "81",
    pages = "084016",
    year = "2010"
}

@article{Bernuzzi:2014kca,
    author = "Bernuzzi, Sebastiano and Nagar, Alessandro and Dietrich, Tim and Damour, Thibault",
    title = "{Modeling the dynamics of tidally-interacting binary neutron stars up to merger}",
    eprint = "1412.4553",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevLett.114.161103",
    journal = "Phys. Rev. Lett.",
    volume = "114",
    pages = "161103",
    year = "2015"
}

@article{Nagar:2018zoe,
    author = "Nagar, Alessandro and others",
    title = "{Time-domain effective-one-body gravitational waveforms for coalescing compact binaries with nonprecessing spins, tides and self-spin effects}",
    eprint = "1806.01772",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.98.104052",
    journal = "Phys. Rev. D",
    volume = "98",
    pages = "104052",
    year = "2018"
}
```

---

## 21. Continuous Waves & Differentiable F-Statistic

```bibtex
@article{Jaranowski:1998qm,
    author = "Jaranowski, Piotr and Kr{\'o}lak, Andrzej and Schutz, Bernard F.",
    title = "{Data analysis of gravitational-wave signals from spinning neutron stars. 1. The Signal and its detection}",
    eprint = "gr-qc/9804014",
    doi = "10.1103/PhysRevD.58.063001",
    journal = "Phys. Rev. D",
    volume = "58",
    pages = "063001",
    year = "1998"
}
```

---

## 22. Non-Linear Gravitational Wave Memory

```bibtex
@article{Christodoulou:1991cr,
    author = "Christodoulou, Demetrios",
    title = "{Nonlinear nature of gravitation and gravitational-wave experiments}",
    doi = "10.1103/PhysRevLett.67.1486",
    journal = "Phys. Rev. Lett.",
    volume = "67",
    pages = "1486--1489",
    year = "1991"
}

@article{Favata:2010ic,
    author = "Favata, Marc",
    title = "{The gravitational-wave memory effect}",
    eprint = "1003.3486",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/0264-9381/27/8/084036",
    journal = "Class. Quant. Grav.",
    volume = "27",
    pages = "084036",
    year = "2010"
}

@article{Talbot:2018sll,
    author = "Talbot, Colm and Thrane, Eric and Lasky, Paul D. and Lin, Fuhui",
    title = "{Gravitational-wave memory: waveforms and phenomenology}",
    eprint = "1807.00990",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1103/PhysRevD.98.064031",
    journal = "Phys. Rev. D",
    volume = "98",
    pages = "064031",
    year = "2018"
}
```

---

## 23. Non-Gaussian Glitch Model & Wavelet Likelihood

```bibtex
@article{Cornish:2014kda,
    author = "Cornish, Neil J. and Littenberg, Tyson B.",
    title = "{BayesWave: Bayesian Inference for Gravitational Wave Bursts and Instrument Glitches}",
    eprint = "1410.3835",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/0264-9381/32/13/135012",
    journal = "Class. Quant. Grav.",
    volume = "32",
    pages = "135012",
    year = "2015"
}

@article{Chatziioannou:2021ezd,
    author = "Chatziioannou, Katerina and Isi, Maximiliano and Haster, Carl-Johan and Littenberg, Tyson B.",
    title = "{A morphology-independent test of the mixed polarization content of gravitational wave signals from compact binary coalescences}",
    eprint = "2105.01521",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.104.044005",
    journal = "Phys. Rev. D",
    volume = "104",
    pages = "044005",
    year = "2021"
}
```


## Native IMRPhenomT family

```bibtex
@article{Estelles:2021PhenomT,
    author = "Estellés, Héctor and others",
    title = "{IMRPhenomTP: A phenomenological time domain model for dominant quadrupole gravitational wave signal of coalescing binary black holes}",
    eprint = "2004.08302",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.103.124060",
    journal = "Phys. Rev. D",
    volume = "103",
    pages = "124060",
    year = "2021"
}
```

The aligned-spin carrier and higher harmonics use the MIT-licensed author release
[phenomxpy](https://gitlab.com/imrphenom-dev/phenomxpy/-/tree/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt)
for calibration expressions and reconstruction conventions. See
[third-party notices](THIRD_PARTY_NOTICES.md) for the retained license.

The higher-mode reconstruction is described in:

```bibtex
@article{Estelles:2020PhenomTHM,
    author = "Estellés, Héctor and others",
    title = "{Time domain phenomenological model of gravitational wave subdominant harmonics for quasi-circular non-precessing binary black hole coalescences}",
    eprint = "2012.11923",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.105.084039",
    journal = "Phys. Rev. D",
    volume = "105",
    pages = "084039",
    year = "2022"
}
```

For TP/TPHM, cite the precessing family paper, whose Appendix A describes
the precession options exposed here:

```bibtex
@article{Estelles:2022PhenomTPHM,
    author = "Estellés, Héctor and others",
    title = "{New twists in compact binary waveform modelling: a fast time domain model for precession}",
    eprint = "2105.05872",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.105.084040",
    journal = "Phys. Rev. D",
    volume = "105",
    pages = "084040",
    year = "2022"
}
```

The numerical precession equations are documented in Riccardo Sturani,
[Note on the derivation of the angular momentum and spin precessing equations in SpinTaylor codes, LIGO-T1500554](https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf),
Eqs. (3), (21), (22), (24), and (25). The PhenomTP numerical path excludes
the Eq. (24) N3LO quadratic-spin terms, as specified by the `phenomtp`
parameter in the
[public LAL API](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_inspiral_8h.html);
the internal derivative helper retains the full-equation option. MSA uses
[Chatziioannou et al., Phys. Rev. D 95, 104004 (2017)](https://doi.org/10.1103/PhysRevD.95.104004)
and the precession conventions of
[Pratten et al., Phys. Rev. D 103, 104056 (2021)](https://doi.org/10.1103/PhysRevD.103.104056).
