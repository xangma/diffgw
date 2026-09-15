# Pass 18 detector geometry and time audit (primary sources)

Date: 2026-09-13.  This is a source audit only.  The audit excludes LALSuite,
PyCBC, translated or generated implementations, notebooks, git history, public
binaries, parity outputs, residuals, and verification directories.  The native
TorchWave file was inspected only after the primary-source derivation below.

## Source ledger

* **LIGO-T980044-10**, *Determination of Global and Local Coordinate Axes for
  the LIGO Sites*, version 10 (2001-02-07),
  [DCC record](https://dcc.ligo.org/LIGO-T980044/public),
  [PDF](https://dcc.ligo.org/public/0002/T980044/001/T980044-10.pdf), SHA-256
  `3db2c004cd3709f8f7084ffe72463519b4bd5e5cd09a7dbebdd61d12776d424f`.
  Tables 4 and 18 give WGS-84 geodetic vertices and ECEF arm direction cosines
  (the PDF pages headed “Hanford, WA” and “Livingston, LA”).  The printed
  direction precision is six decimal places; the ECEF vertices are printed to
  roughly centimetres to decimetres.  The document states that the axes are
  determined from as-built site surveys and reports millimetre-scale fit RMS.
* **EGO, h_October2007_v14.FH11**, *The Gravitational Voice*, 2007-10-10,
  [official EGO PDF](https://www.ego-gw.it/wp-content/grand-media/application/h_6_OCTOBER2007.pdf),
  SHA-256 `30dede3e281725a38c9e6b156c057a69f979290915425a3e42ac4fa4ca283cd9`.
  Page 7, “GPS coordinates of the Virgo vertexes”, prints WGS-84 BS, NTT and
  WTT latitude, longitude and ellipsoidal height.  The table precision is
  0.0001 arcsec and millimetres in height; the surrounding text says the
  coordinates came from the January 1997, November 2004 and June 2005 GPS
  surveys.
* **USNO Circular 179**, *The IERS Conventions (2010)*, official USNO-hosted
  copy, [PDF](https://aa.usno.navy.mil/downloads/Circular_179_bw.pdf), SHA-256
  `b6eee4a8e8d270a5930a0e0385c658bafed0b45c9b5bafa1ab417d80224032e8`.
  Section 2.6.2, equation (2.10)--(2.12), defines the Earth Rotation Angle
  from UT1 and GMST from that angle plus the TT/TDB precession polynomial.
  Sections 2.3--2.4 explain that UTC differs from UT1 by the measured
  `UT1-UTC` and that leap seconds only keep UTC near UT1.  The official USNO
  [GMST FAQ](https://aa.usno.navy.mil/faq/GAST), retrieved asset SHA-256
  `c0a6256702ca71af496015b73f7a61be4c07cb2816dbc438d775a08e08d02371`, gives
  the corresponding approximate formula and explicitly requires UT1 for
  sub-second accuracy.

No authoritative, publicly retrievable KAGRA asset with exact ECEF vertex and
arm vectors was available in the permitted source set.  The official KAGRA
DocDB design record is access-controlled from this environment
([MIF Design record](https://gwdoc.icrr.u-tokyo.ac.jp/DocDB/0009/T1200913/002/MIF_Design.pdf),
HTTP 401), so it is not used as a numerical source and no unverifiable hash is
reported.  The remaining official overview material gives only coarse site
coordinates/orientation and cannot certify the native K1 vectors at their
stored precision.

## Geometry comparison

### H1 and L1

Table 4 of T980044-10 gives H1's ECEF vertex
`(-2.1614149e6, -3.8346952e6, 4.6003502e6) m`, x-arm direction
`(-0.223892, 0.799831, 0.556905)`, and y-arm direction
`(-0.913978, 0.0260945, -0.404923)`.  The native H1 vertex is
`(-2161414.926331955, -3834695.17880384, 4600350.226900666)` and its
direction vectors differ from the six-decimal printed vectors by at most
`8.6e-7` in Euclidean norm.  The vertex difference from the rounded table is
`0.0432 m`, within the table's displayed precision.

Table 18 gives L1's ECEF vertex
`(-74276.044, -5.496283721e6, 3.224257018e6) m`, x-arm direction
`(-0.954574, -0.14158, -0.262189)`, and y-arm direction
`(0.297741, -0.48791, -0.820545)`.  The native L1 vertex differs from those
rounded coordinates by `0.00157 m`; the direction-vector differences are at
most `7.9e-7` in norm.  These are consistent with the source's printed
precision.  No source-complete H1/L1 geometry correction is identified.

### Virgo

Converting the EGO page-7 WGS-84 BS/NTT/WTT coordinates to ECEF with the
standard WGS-84 ellipsoid and taking BS-to-NTT and BS-to-WTT chords gives,
respectively,

```
BS       (4546375.145583,  842989.907643, 4378577.805105) m
BS->NTT  (-0.700665994090, 0.208453878578, 0.682359249393)
BS->WTT  (-0.054318427729,-0.969176229210, 0.240305940716)
```

The native V1 vertex is about `(1.047, 0.210, 0.842) m` from this rounded
BS conversion, and its x/y vectors differ from these endpoint chords by
`2.9e-4` and `7.3e-4`, respectively.  The official table identifies surveyed
vertex endpoints, not whether the detector response convention requires the
BS-to-endpoint chord, a local tangent at BS, or a surveyed as-built axis.  The
native V1 vectors therefore cannot be replaced by these derived chords without
guessing a geometric convention.  This is a source-completeness blocker, not
an exact mismatch.

### K1

The permitted official material found for KAGRA states that the two arms are
orthogonal 3-km tunnels on a sloped plane, but does not provide an exact,
machine-readable ECEF vertex plus arm-vector record that can be hashed and
compared at native precision.  Coarse latitude/longitude and an orientation
rounded to hundredths of a degree do not determine the stored K1 vectors.
No K1 patch is justified.

## Time and sidereal-angle semantics

USNO Circular 179 equation (2.10) is
`theta = 0.7790572732640 + 1.00273781191135448 DU`, where
`DU = JD(UT1)-2451545.0`; equation (2.12) then forms GMST from `theta` and a
precession polynomial in TT/TDB.  The native
`greenwich_mean_sidereal_time` converts GPS to UTC with a leap-second table
and uses UTC as the rotation time, while its own docstring records that this
is an approximation made because no Earth-orientation data are supplied.
This is a documented API limitation: an exact correction needs a UT1-UTC/EOP
input or an authorized, versioned EOP data source.  Substituting a guessed
offset or extending the leap table would not be source-complete, so no runtime
change is made.  The constant Earth rotation rate is consistent with the
coefficient in USNO equation (2.10), but that does not remove the UT1 input
requirement.

## Decision

No exact source-complete detector mismatch was established.  H1/L1 agree with
the authoritative printed vectors; V1 has an endpoint-versus-axis convention
blocker; K1 lacks an exact permitted geometry record; and GMST requires UT1
data absent from the native API.  Therefore this pass changes only this audit
document and adds no runtime or test changes.
