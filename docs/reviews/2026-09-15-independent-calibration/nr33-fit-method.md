# Independent `(3,3)` amplitude pilot

## Scope

This pilot refits only the XHM ringdown amplitude subcomponent. It cannot
establish full IMRPhenomXHM or LAL numerical parity. The input is raw complex
NR `h33(t)` plus explicit mass ratio and remnant metadata; no model output or
published coefficient body is used.

## Published recipe and data

The primary paper reports 504 hybrid waveforms: 466 comparable-mass cases
(`1 <= q <= 18`) and 38 extreme-mass-ratio cases (`q=200,1000`). The
comparable hybrids glue EOB/PN inspiral to NR IMR data. For amplitude calibration
the NR sources were 186 SXS/SpEC (2018), 95 BAM, and 16 Einstein Toolkit
waveforms; a later SXS expansion updated phase fits but explicitly did not
recalibrate amplitudes. See [XHM v2, Sec. III](https://arxiv.org/html/2001.10914v2#S3).

The implementable subcomponent is Eq. (42):

`A_RD^lm/A_0^22 = f^(-1/12) |a_lambda| f_damp sigma / ((f-f_ring)^2 + (f_damp sigma)^2) exp(-(f-f_ring) lambda/(f_damp sigma))`, with `A_0^22 = pi sqrt(2 eta/3) (pi f)^(-7/6)` (Eq. 26).

For `(3,3)`, the paper fixes `sigma=1.3` (Table 8), and the amplitude
ringdown band is `[0.95 f_ring, 0.3]` in `Mf` (Eq. 41 and Table 2). The pilot
uses independently generated `(3,3,0)` Kerr QNMs from remnant spin and mass
fraction when supplied; explicit QNM metadata is accepted and recorded.

## Frozen pilot controls

`fit_nr33_amplitude_pilot.py` discards samples before the supplied relaxation
time, sorts/deduplicates time, cubic-spline resamples to fixed `dt=0.1M`, and
applies raised-cosine endpoint tapers (first 500M after
relaxation and last 20M), zero-pads by 4, conjugates SXS-convention `h33`, and
uses `dt*FFT`, retaining positive cycles/M. The merger must lie in the flat
window. The output records native/resampled `dt`, retained time interval and
`nfft`. Post-freeze checks use start tapers 250M and 1000M, end tapers 10M and
40M, and resampling at 0.05M. They report spectral sensitivity without refitting.
Direct fits minimize RMS-normalized amplitude residuals with SciPy
`least_squares`, `soft_l1`, `f_scale=1`, `x_scale='jac'`, 2000 evaluations,
`log|a_lambda|` bounds `[-30,30]`, and `lambda` bounds `[-10,10]`.

The q split is frozen before fitting: the seven pilot IDs ending `0008,0169,
0182,0056,0063` train; `0030,0166` hold out. The nonspinning parameter fit
is fixed degree two in `x=4 eta-1`, fitting `log(|a_lambda|/delta)` with
`delta=(q-1)/(q+1)` and `lambda`.
The two same-case N2 files in the manifest are marked `validation_only` and
excluded before any amplitude array is opened. The runner checks each declared
training NPZ SHA-256 and records those hashes, the manifest hash, and the
independent Kerr-helper hash in the calibration JSON. It writes split and
run-identity freeze files before fitting, then writes a calibration hash
sidecar.

## Reproducibility boundary

The paper specifies the ansatz, QNM-scaled band, two-level direct/parameter
space strategy, and hierarchical nonspinning-first idea ([Sec. II.3](https://arxiv.org/html/2001.10914v2#S2)). It does **not** fully pin the optimizer,
frequency-bin weights, interpolation/FFT endpoint treatment, numerical
conditioning, outlier annotations, or hybrid splice windows; it says the
quality procedures are not documented in detail ([Sec. III](https://arxiv.org/html/2001.10914v2#S3)). Therefore this is a plausible independent
refit recipe, with its controls auditable in the JSON artifact, rather than a
guarantee of release122022 numerical identity.
