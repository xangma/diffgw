# Scientific Citations and Provenance

`torchwave` is implemented under a strict clean-room protocol. All mathematical formulas, post-Newtonian expansions, collocation points, and fit coefficients are derived exclusively from published peer-reviewed journal articles and arXiv preprints.

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
