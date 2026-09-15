# Pass 3: detector geometry and time precision

This bounded pass corrects time arithmetic, leap dates, unit-arm normalization, and frequency-domain batching in `src/torchwave/detectors/network.py`. The antenna convention and geometric-delay sign agree with the published equations. No detector site coordinate or arm direction was fitted to an external implementation. This report does not certify binary parity.

## Equations and changes

**Sidereal time.** Aoki et al. (1982), Eq. (13), gives the midnight GMST in sidereal seconds,

\[
\theta_0=24110.54841+8640184.812866T_0+0.093104T_0^2-6.2\,10^{-6}T_0^3.
\]

Its Eq. (19) gives the sidereal/solar rate

\[
r=1.002737909350795+5.9006\,10^{-11}T_0-5.9\,10^{-15}T_0^2.
\]

The existing polynomial and daily-rate approximation are retained. For nominal UTC calendar seconds `u` since the GPS epoch, compute `d=floor(u/86400)`, `s=u-86400*d`, and `T0=(d-7300.5)/36525`. Evaluate the wrapped angle from `theta0+r*s`. This algebra avoids adding fractional days to a Julian date near 2.4 million, which previously lost tens of microseconds even in float64. It also reduces sidereal seconds before conversion to radians.

GPS inputs now enter the calculation as float64, including integer tensors. Detector modules preserve the epoch as float64 even after `.float()`; only the final wrapped angle is cast to the detector geometry dtype. Integer sky tensors also receive the requested geometry dtype explicitly.

**Leap seconds.** IERS lists UTC effective midnights and TAI−UTC. Taking nominal GPS=TAI−19 s gives `GPS_boundary=calendar_seconds+(TAI−UTC−19)`. The old table applied each new offset at the start of the inserted second, one second before the effective midnight. The 2012 entry also contained a separate date error: `1024012815` instead of `1025136016` GPS. That is 13 days and one second early. All 18 entries are now independently checked against the IERS calendar dates.

The scalar UTC coordinate uses ordinary 86400-second calendar days. The inserted `23:59:60` has the same nominal coordinate as the following `00:00:00`; it is not a continuous elapsed-time coordinate. For example, GPS `1167264016, 1167264017, 1167264018` maps to nominal UTC calendar seconds `1167263999, 1167264000, 1167264000`. This folding is now explicit in the docstring.

**Geometry.** Anderson et al., Appendix B, Eqs. (B2)–(B8), gives

\[
D=\tfrac12(\hat u\hat u^T-\hat v\hat v^T),\quad
F_+=X^TDX-Y^TDY,\quad F_\times=X^TDY+Y^TDX.
\]

The existing `X,Y` agree with Eqs. (B4),(B5) after `phi=ra−GMST` and `theta=pi/2−dec`. The rounded stored arm directions are now individually normalized before constructing `D`. Their mutual angle is retained; they are not forced orthogonal. Consequently the unit-arm identities `tr(D)=0` and `D:D=(1−(u·v)^2)/2` hold to rounding precision. The stored site coordinates and direction tuples are unchanged.

For a source-facing unit vector `n`, the detector arrival delay is `−r·n/c`. This agrees with Anderson Eq. (5.10), which writes the detector signal with the opposite-sign time argument offset. With the module's negative-exponent Fourier transform, its phase factor remains `exp(−2*pi*i*f*delay)`.

**Batches.** The last waveform axis is frequency; sky/time parameters describe leading batch axes. Those parameters now receive a final singleton axis even when the frequency grid itself is one-dimensional. This fixes both batched waveforms and a common waveform projected over multiple sky positions.

**Existing gradient assertion.** The prior detector test required a nonzero derivative of face-on circular-polarization power with respect to polarization angle. However, `hc=±i*hp` implies power `|hp|^2*(F+^2+Fx^2)`, which is polarization invariant. That assertion now checks zero to arithmetic precision. A new elliptical-polarization, phase-sensitive test supplies meaningful nonzero-gradient coverage.

## Validation

Command:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest tests/test_clean_room_detectors.py tests/test_detectors.py -k 'not lal' -q
```

Result: **30 passed, 1 oracle test deselected**. `git diff --check` passes for the detector changes. Tests cover:

- All IERS leap boundaries, including the inserted second and following midnight; ordinary and boundary epochs from 1980 through 2026.
- A 50-digit `Decimal` evaluation of the published Aoki equations, with angle error below `5e-14` rad; resolved microsecond increments and the analytic daily sidereal rate.
- Python/integer/float64 epoch behavior and float32 detector output precision.
- Published survey precision, trace-free unit-arm tensors, Anderson's ideal-XY closed forms (B9),(B10), polarization rotation, and geometric-delay extrema.
- Batched FD projection versus scalar evaluation, both API argument orders, network forwarding, finite noncircular gradients, and the rotation-generator identity `d/dt = −omega*d/dra`.

The small analytic-identity tolerances are tied to float64 rounding; the survey tolerances follow the precision of the printed source data. No frozen oracle threshold or harness gate was changed.

## Precision and scope limits

- **UTC approximates UT1.** No Earth-orientation series, DUT1, polar motion, nutation, or modern celestial-frame transformation is supplied. Accurate evaluation of the chosen polynomial is not equivalent to equally accurate physical Earth orientation. The leap-folded approximation is not continuous at the effective midnight.
- **Input precision is irreversible.** Float32 GPS tensors near modern epochs have 128-second spacing; promotion cannot recover information already lost. Python floats or float64 tensors are required for precise epochs. A float32 detector intentionally casts the final angle and computes geometry at float32 accuracy.
- **Ideal time scales.** The GPS/TAI relation ignores the small realized clock offset described by BIPM. The fixed leap table is verified through IERS Bulletin C 72, July 2026; it needs updating if a later bulletin announces an insertion.
- **Survey digits are limited.** LIGO T980044-08 supports the inherited H1/L1 inputs within its printed position/direction accuracy; Anderson's table supports only lower precision for V1. The KAGRA published survey tables support millimeter-scale position agreement and roughly `1e-7` direction agreement. These checks do not independently establish every inherited decimal digit. No extra digits were invented or inferred from binary comparisons.
- **TD projection is unchanged.** It rounds to an integer sample shift, wraps circularly, and requires a scalar geometric delay. It lacks fractional-delay gradients and batched-sky shifts. Its docstring now states this explicitly.

## Fresh primary-source ledger

Retrieved or viewed on 2026-09-12. Downloaded PDFs are outside the repository under `/tmp/torchwave-clean-detectors/`.

| Source | Use | SHA-256 of downloaded source |
|---|---|---|
| [Aoki et al., A&A 105, 359–361 (1982)](https://articles.adsabs.harvard.edu/pdf/1982A%26A...105..359A) | Original scanned pages, Eqs. (13),(19); inspected visually | `10b9a32e7279b24cd9e66a7818dbb9d20d9ca618c4d0a4070a450a27778ff77f` |
| [Anderson et al., gr-qc/0008066v1](https://arxiv.org/pdf/gr-qc/0008066v1) | Eq. (5.10), Appendix B Eqs. (B1)–(B17), Table I | `cb3e40491b9dca920a5f60d01c1132d95b8ce270b979c006ad5b9cbe123d7256` |
| [IERS Leap_Second.dat, updated through Bulletin C 72](https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat) | Effective UTC dates, TAI−UTC; source expires 2027-06-28 | `6cb6f5d4b819f2e568e25db4b0b26d89dedf031fdffb18bc94d40f4e94e268d7` |
| [LIGO T980044-08](https://dcc.ligo.org/public/0002/T980044/000/T980044-08.pdf) | Site survey, Tables 4 and 18 | `d804ac42c23d0bbc48982d8e8a28ba2854e15dc502d294958efa88768bfceee2` |
| [BIPM Annual Report 2019, p. 45](https://webtai.bipm.org/ftp/pub/tai/annual-reports/bipm-annual-report/annual_report_2019.pdf) | GPS/TAI 19-second relation and realized clock correction | `b462c12540bd9052ff3923e335d387ebd48a9bc6007119e3fe08e93e18094b6f` |
| [KAGRA JGW-T1605295-v1](https://gwdoc.icrr.u-tokyo.ac.jp/DocDB/0052/T1605295/001/kagra_location.pdf) | Survey Tables 1, 4, 5 | Web-rendered document only; direct download returned HTTP 401, so no file hash |

No LAL/PyCBC implementation, port, source code, binary call, or oracle output was used. The KAGRA survey note includes a quotation of an API's azimuth terminology on its first page; only its physical survey tables were used here. Existing repository detector code and tests were reviewed, and the oracle test was excluded from execution.
