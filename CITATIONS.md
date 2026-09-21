# Scientific Citations and Provenance

`torchwave` is implemented under a strict clean-room protocol. Physical models follow primary literature; additional permissively licensed calibration releases are identified below and in the provenance ledger.

If you use `torchwave` in scientific work, please cite the underlying models and papers:

---

## 1. Post-Newtonian Theory & TaylorF2

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
    title = "{Higher-order spin effects in the amplitude and phase of gravitational waveforms emitted by inspiraling compact binaries: Aligned systems}",
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
    title = "{Next-to-next-to-leading order spin-orbit effects in the gravitational wave flux and phasing of compact binaries}",
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
    title = "{Constraining neutron star tidal deformability with gravitational wave observations of binary inspirals}",
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
    author = "Vines, Justin and Flanagan, {\'E}anna {\'E}. and Hinderer, Tanja",
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
    author = "Wade, Leslie and Creighton, Jolien D. E. and Ochsner, Evan and Lackey, Benjamin D. and Farr, Will M. and Littenberg, Tyson B. and Raymond, Vivien",
    title = "{Systematic and statistical errors in a methodology for estimating equation-of-state parameters from coalescing binary neutron stars}",
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

The fundamental Kerr mode now uses an independently generated spectral approximation, with development eigenproblem solves from [Stein (2019), qnm](https://doi.org/10.21105/joss.01683). It has no QNM data-file or solver runtime dependency. See the [derivation and validation record](https://github.com/xangma/diffgw/blob/archive/clean-room-reviews/docs/reviews/2026-09-11-independent/qnm.md).

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
    doi = "10.1103/PhysRevD.108.124040",
    journal = "Phys. Rev. D",
    volume = "108",
    number = "12",
    pages = "124040",
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
    title = "{Closed-form tidal approximants for binary neutron star waveforms constructed from high-resolution numerical-relativity simulations}",
    eprint = "1706.01111",
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
    title = "{An improved analytical tidal waveform model for binary neutron star inspirals}",
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
    title = "{Matter transcriptions: Tidal effects in binary neutron star mergers within the SEOBNR and IMRPhenom frameworks and an improved NRTidal model}",
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
    title = "{Using full test-particle metric perturbations to compute gravitational waveforms from inspiralling compact binaries: Amplitude corrections to 3PN order}",
    eprint = "0710.0614",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.77.044012",
    journal = "Phys. Rev. D",
    volume = "77",
    pages = "044012",
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
```

---

## 6. Detector Networks & Coordinate Projections

```bibtex
@article{Anderson:2000yy,
    author = "Anderson, Warren G. and Brady, Patrick R. and Creighton, Jolien D. E. and Flanagan, {\'E}anna {\'E}.",
    title = "{An excess power statistic for detection of gravitational wave bursts}",
    eprint = "gr-qc/0008066",
    doi = "10.1103/PhysRevD.63.042003",
    journal = "Phys. Rev. D",
    volume = "63",
    pages = "042003",
    year = "2001"
}

@article{Schutz:2011tw,
    author = "Schutz, Bernard F.",
    title = "{Networks of gravitational wave detectors and their run planning}",
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
    title = "{Computationally efficient models for the spectrum of full inspiral, merger and ringdown of precessing black-hole binaries: IMRPhenomXPHM}",
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
    title = "{Phenomenological model for the gravitational-wave signal from precessing of black hole binaries with higher-order modes}",
    eprint = "1911.06050",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevLett.124.081101",
    journal = "Phys. Rev. Lett.",
    volume = "124",
    pages = "081101",
    year = "2020"
}
```

---

## 8. NSBH Tidal Disruption & Remnant Model (`IMRPhenomNSBH`)

```bibtex
@article{Pannarale:2015jka,
    author = "Pannarale, Francesco and others",
    title = "{Aligning spins in neutron-star black-hole binaries}",
    eprint = "1509.06209",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.92.084032",
    journal = "Phys. Rev. D",
    volume = "92",
    pages = "084032",
    year = "2015"
}

@article{Matas:2020wab,
    author = "Matas, Andrew and others",
    title = "{Aligned-spin neutron-star\textendash{}black-hole waveforms with tidal disruption effects using the SEOBNRv4_ROM_NRTidalv2_NSBH model}",
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
    title = "{Gravitational-wave Phasing for Eccentric Compact Binaries in the Stationary Phase Approximation: Including Post-Newtonian Corrections to Small Eccentricity}",
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
    title = "{Post-Newtonian, quasi-circular binary inspirals in parameterized post-Einsteinian gravity}",
    eprint = "0909.4263",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.80.084018",
    journal = "Phys. Rev. D",
    volume = "80",
    pages = "084018",
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
    title = "{Gravitational waves from coalescing compact binaries: How accurately can one measure their properties?}",
    eprint = "gr-qc/9402014",
    doi = "10.1103/PhysRevD.49.2658",
    journal = "Phys. Rev. D",
    volume = "49",
    pages = "2658--2697",
    year = "1994"
}

@article{Veitch:2014wba,
    author = "Veitch, John and others",
    title = "{Parameter estimation for compact binaries with ground-based gravitational-wave observations using the LALInference software library}",
    eprint = "1409.7788",
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
    title = "{An introduction to parameter estimation for gravitational-wave astronomy}",
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
    title = "{Let’s twist again: canonically transformed aligned-spin waveform model for binary black holes}",
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
    title = "{Two-body problem in general relativity with spins: Precession on radiation-reaction timescales}",
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
    title = "{Coalescing binary systems of compact objects to (post)**(5/2)-Newtonian order. 5. Spin effects}",
    eprint = "gr-qc/9506014",
    doi = "10.1103/PhysRevD.52.821",
    journal = "Phys. Rev. D",
    volume = "52",
    pages = "821--847",
    year = "1995"
}

@article{Racine:2008qv,
    author = "Racine, Etienne",
    title = "{Analysis of spin precession in binary black hole systems including two-spin effects}",
    eprint = "0803.1848",
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

@article{Babak:2021mhe,
    author = "Babak, Stanislav and others",
    title = "{LISA Data Challenge: First sprint}",
    eprint = "2101.05872",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.103.084014",
    journal = "Phys. Rev. D",
    volume = "103",
    pages = "084014",
    year = "2021"
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
    title = "{Heterodyned Likelihood for Fast Gravitational Wave Parameter Estimation}",
    eprint = "1007.4820",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.82.022002",
    journal = "Phys. Rev. D",
    volume = "82",
    pages = "022002",
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
    author = "Chua, Alvin J. K. and Galley, Chad R. and Vallisneri, Michele",
    title = "{Fast and accurate gravitational-wave waveforms for extreme mass-ratio inspirals with deep neural networks}",
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
    title = "{Dynamical Systems Approach to Runaway Instabilities in Black Hole-Torus Systems}",
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
    title = "{Constraining Generic Post-Einsteinian Extensions of Gravity with Gravitational Waves}",
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
    title = "{Effective one body description of tidal effects in inspiralling compact binaries}",
    eprint = "0906.0096",
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
    title = "{Modeling the dynamics of tidally interacting binary neutron stars}",
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
    title = "{Time-domain effective-one-body gravitational waveforms for coalescing compact binaries with spin: I. Aligned spins}",
    eprint = "1805.03891",
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

@article{Prix:2011qv,
    author = "Prix, Reinhard",
    title = "{The F-statistic and its generalizations in continuous gravitational-wave data analysis}",
    eprint = "1108.3546",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    year = "2011"
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
    author = "Talbot, Colm and Thrane, Eric and Biscoveanu, Sylvia and Smith, Rory",
    title = "{Measuring the gravitational-wave memory effect of binary black hole mergers with second-generation gravitational-wave detectors}",
    eprint = "1807.00990",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1103/PhysRevD.98.084031",
    journal = "Phys. Rev. D",
    volume = "98",
    pages = "084031",
    year = "2018"
}
```

---

## 23. Non-Gaussian Glitch Model & Wavelet Likelihood

```bibtex
@article{Cornish:2014kda,
    author = "Cornish, Neil J. and Littenberg, Tyson B.",
    title = "{BayesWave: Bayesian Inference for Gravitational Wave Bursts and Instrument Glitches}",
    eprint = "1410.3838",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/0264-9381/32/13/135012",
    journal = "Class. Quant. Grav.",
    volume = "32",
    pages = "135012",
    year = "2015"
}

@article{Chatziioannou:2021ezd,
    author = "Chatziioannou, Katerina and others",
    title = "{Modeling transient noise in gravitational-wave detectors}",
    eprint = "2105.01521",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.104.044015",
    journal = "Phys. Rev. D",
    volume = "104",
    pages = "044015",
    year = "2021"
}
```


## Native IMRPhenomT

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

The aligned-spin `(2,±2)` implementation uses the MIT-licensed author release
[phenomxpy](https://gitlab.com/imrphenom-dev/phenomxpy/-/tree/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt)
for calibration expressions and reconstruction conventions. See
[third-party notices](THIRD_PARTY_NOTICES.md) for the retained license.

The updated dominant-mode reconstruction is also described in:

```bibtex
@article{Estelles:2020PhenomTHM,
    author = "Estellés, Héctor and others",
    title = "{Time domain phenomenological model of gravitational wave subdominant harmonics for quasi-circular non-precessing binary black hole coalescences}",
    eprint = "2012.11923",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    year = "2020"
}
```
Only its dominant-mode improvements are implemented here.
