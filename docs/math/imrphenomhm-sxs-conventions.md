# IMRPhenomHM–SXS mode conventions

## Scope and status

`hm.conventions.sxs-bridge.v1` is a standalone, source-derived convention
bridge for the six positive-frequency IMRPhenomHM carriers and the SXS strain
modes. Its SymPy/mpmath reference fixes Fourier signs, the negative-$m$ pair,
orbital/time transformations, and a phase statistic that separates gauges
from mode-relative discrepancies. The finite-$f_\mathrm{ref}$ public HM
adapter is checked conditionally against both frontends. This is a
convention and implementation gate, **not** a fit to NR or a physical
qualification of the six-mode waveform.

This v1 bridge retains the historical `hm_reference_phase="legacy"` adapter.
The separately versioned [orbital reference-phase extension](imrphenomhm-waveform.md#orbital-reference-phase-extension)
omits its additional finite-reference factor. Existing bridge evidence is not
silently reassigned to that new option.

## Inputs, conventions, and options

Use positive $f$ in Hz and $m>0$. SXS defines
$h=h_+-ih_\times=\sum_{\ell m}h_{\ell m}\,{}_{-2}Y_{\ell m}$.
For a Fourier sign $s=\pm1$, define
$\mathcal F_s[x](f)=\int x(t)e^{s2\pi ift}\,dt$.
The SXS holdout uses $s=-1$ and the local positive carrier
$P_{\ell m}(f)=\mathcal F_-[\overline{h_{\ell m}}](f)$.
The sampled transform checks also cover $s=+1$ to expose an accidental
sign assumption. The public HM comparison covers
$(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)$, the asymmetric aligned-spin
binary $(35,20)M_\odot$, $(\chi_1,\chi_2)=(0.2,-0.1)$,
$\phi_c=0.37$, $t_c=0.008\,$s, and
$f_\mathrm{ref}\in\{\mathrm{None},40\,\mathrm{Hz}\}$.
Only the CPU-FP64 public check evaluates frontend values.

## Mathematics and components

[García-Quirós et al., Sec. II Eqs. (2), (4)–(9)](https://arxiv.org/html/2001.10914v2)
give $h_{\ell,-m}(t)=(-1)^\ell\overline{h_{\ell m}(t)}$ and the
negative-$m$ positive-frequency carrier $N_{\ell m}=\mathcal F_-[h_{\ell,-m}]$.
Therefore

$$
N_{\ell m}(f)=(-1)^\ell\overline{\mathcal F_-[h_{\ell m}](-f)},
\qquad P_{\ell m}(f)=(-1)^\ell N_{\ell m}(f).
$$

The frequency reversal in the first equality is essential. The second
identity maps the paper's mode pair to the SXS holdout target, independent
of an empirical phase alignment. [London et al., Eq. (1)](https://arxiv.org/html/1708.00404v4)
and the [2019 SXS catalog, Eq. (52) and its sign discussion](https://arxiv.org/html/1904.04831v2)
fix the strain expansion and the catalog's current $h_+-ih_\times$ sign;
the [third catalog](https://arxiv.org/html/2505.13378v2) describes the
retroactive convention update. A common strain sign $c=-1$ multiplies
every $P_{\ell m}$ by $c$; it does not select an arbitrary per-mode phase.

With $\mathcal F_-$, a delay $t\mapsto t-t_c$ multiplies $P_{\ell m}$ by
$e^{-2\pi ift_c}$. [García-Quirós et al., Eq. (14)](https://arxiv.org/html/2001.10914v2)
give the physical rotation $h_{\ell m}\mapsto e^{im\delta}h_{\ell m}$,
so $P_{\ell m}\mapsto e^{-im\delta}P_{\ell m}$. The native public phase
$\phi_c$ acts as $e^{im\phi_c}$ on $P$, hence its rotation angle in that
equation is $\delta=-\phi_c$.

The public HM adapter also supplies
$B=(d\Psi_D/d(Mf))_\mathrm{peak}M_\mathrm{seconds}f_\mathrm{ref}$ and
$e^{i(m/2-1)B}$, with $B=0$ when $f_\mathrm{ref}=\mathrm{None}$.
This $B$ term documents the **existing native adapter**, not a factor
deduced from SXS. The scalar reference takes $B$ as an input and never
reimplements the calibrated D slope.

For nonzero carriers, let $u_{\ell m}=P_{\ell m}/|P_{\ell m}|$ and
$T_{\ell m}(f)=u_{\ell m}(f)^2\overline{u_{22}(2f/m)}^{\,m}$.
The orbital rotation and delay cancel exactly; the finite-reference
adapter rotates $T$ by $e^{i(m-2)B}$. A common strain factor $c$ rotates
it by $c^{2-m}$, giving $(-1)^m$ for a catalog sign flip. This statistic
does not erase frequency-dependent mode-shape errors.

| Component ID/version | Source and rule | Reference | Public targets | Maintained check |
| --- | --- | --- | --- | --- |
| `hm.conventions.sxs-bridge.v1` | London Eq. (1); García-Quirós Sec. II Eqs. (2), (4)–(9), (14); SXS 2019 Eq. (52): pair, Fourier, rotation and strain sign | `imrphenomhm_sxs_conventions` | Torch/JAX HM positive carriers through the finite-reference adapter | `test_imrphenomhm_sxs_conventions_math_reference.py`; `test_imrphenomhm_intrinsic_modes_math_reference.py` |

## Implementation mapping

`src/diffgw_reference/imrphenomhm_sxs_conventions.py` contains the symbolic
gauge expression, scalar phase factor, mode-pair conversion, and doubled
relative invariant. It imports neither Torch nor JAX. The sampled chirp test
computes both Fourier integrals directly, including $f\leftrightarrow-f$.
The public six-mode test composes the independent HM scalar reference with
frontend-supplied D amplitude/phase, remnant/QNM inputs, and the D peak
slope, then compares full complex values with Torch and JAX. The native
implementations do not call the reference. The paper's mode-pair convention,
native positive carrier, and SXS time-domain modes remain separately named.

## Evidence and limitations

The [run receipt](../validation/imrphenomhm_sxs_conventions_20260927.md)
records the exact checks. The algebra and synthetic chirp test establish
the stated identities for either Fourier sign; the frontend test establishes
conditional agreement for two transition selections and the sampled input.
Neither evaluates an independent PhenomD slope or HM calibration fit.

The diagnostic phase $-e^{i\pi/4}$ seen in the earlier
[3631 receipt](../validation/imrphenomhm_sxs_3631_receipt_20260927.md)
is not introduced as a production correction. A quarter-cycle stationary
phase term appears in [García-Quirós et al., Eqs. (22)–(24)](https://arxiv.org/html/2001.10914v2),
but that later model does not prove a constant HM-to-SXS conversion for
the current implementation. The frozen [2265](../validation/imrphenomhm_sxs_2265_receipt_20260927.md),
[3631](../validation/imrphenomhm_sxs_3631_receipt_20260927.md), and
[3630 projection](../validation/imrphenomhm_sxs_3630_projection_receipt_20260927.md)
failures remain red. A future physical qualification needs a separately
predeclared NR gate after any source-justified model change.

The [full-carrier reference-frequency audit](../validation/imrphenomhm_published_options_20260927.md)
includes the D carrier's own phase subtraction. After an orbital compensation,
it leaves the common analytic phase $e^{-iB}$, generally not a physical
polarization rotation. Removing the explicit HM-relative factor restores
orbital covariance in the diagnostic while preserving $(2,2)$; this is not yet
a production change or a physical qualification result.

## Reproduce the checks

With SymPy, mpmath, NumPy, Torch, JAX and CPU FP64 available, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
tests/test_imrphenomhm_sxs_conventions_math_reference.py
tests/test_imrphenomhm_intrinsic_modes_math_reference.py
--reference-device=cpu -ra`.
Both files are discovered by strict math-reference CI. A changed Fourier
sign, phase gauge, or mode-pair convention requires a new component version,
new source review, and refreshed tests and receipt; the NR protocols are
versioned separately.
