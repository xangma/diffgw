# NSBH joining frequencies gate — 27 September 2026

The [frozen specification](../math/nsbh-joining-frequencies.md) was
committed as `f18bc77` **before** numerical native comparison. It transcribes
[Thompson et al. (2020), Table I](https://arxiv.org/pdf/2002.08383v1)
literally. The scalar reference uses high-precision arithmetic and the
independent scalar branch factors; no LAL source or binary output is used as
an equation oracle. The native helper is called by the production amplitude
path in both backends.

The frozen grid has 30 controlled and 180 composed inputs, covers every
class and the exact $f_{\rm tide}=f_{\rm RD}$ boundary, and supplies rather
than derives $f_{\rm RD}$. All paper-defined frequencies in all 210 cases
passed the 70-versus-90-digit scalar stability limit of $10^{-30}$ absolute.
Torch and JAX CPU FP64 each had a maximum **qualified** joining-frequency
relative error of $4.371\times10^{-16}$, below the frozen
$2\times10^{-13}$ limit. This qualifies the high/no-torus, high/torus, and
low/torus rows, plus low/no-torus $\widetilde f_1$. Table I gives no
$\widetilde f_2$ for either low row, so the native value there has no
paper-literal comparator.

The low/no-torus $\widetilde f_0$ is a measured exception. Table I prints
$[(q-1)\widetilde f_{\rm RD}+\epsilon_{\rm ins}f_{\rm tide}]/q$;
the native carrier applies one more outer $\epsilon_{\rm ins}$ to this
entire center. Its maximum relative mismatch against the printed formula
was **$0.1505$** in both backends on the frozen grid. The existing
compatibility adapter is unchanged. This row and the four-row joining
component remain **conditional**, rather than paper-qualified.

The next amplitude corrections also have source-variant blockers. The
[same paper's Eq. (26)](https://arxiv.org/pdf/2002.08383v1) prints
$\gamma'_1=1-0.5\Lambda-0.25\Lambda^2$ for $\Lambda\le1$ and $1.25$
for $\Lambda>1$. The native small-$\Lambda$ factor has **plus**
$0.5\Lambda$ and multiplies the PhenomC $\gamma_1$. At $\Lambda=1$,
the printed small-branch value is $0.25$ while the native factor is $1.25$;
the paper does not provide an unambiguous source for this adapter.
[Eq. (29)](https://arxiv.org/pdf/2002.08383v1) prints
$(A/2)\omega^-_{x_3,d_3}$ for $\Lambda>1$, with $A/2=b_0=0.81248$;
the native helper uses $b_0\omega^+_{x_3,d_3}$. The opposite window sign
is material away from its center: for
$x=(f_{\rm tide}-\widetilde f_{\rm RD})/\widetilde f_{\rm RD}=x_3+d_3/4$,
the printed value is $0.09684999$ and the native
value is $0.71563001$ (absolute difference $0.61878002$). Neither
correction is included in this gate or retuned to binary output.

On `len` (Linux x86-64 CPU), from
`/home/xangma/tmp/diffgw-nsbh-joins-20260927`, the command
`JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q -s
tests/test_nsbh_joining_frequencies.py tests/test_nsbh_branch_amplitude_fits.py
tests/test_nsbh_branch_reference.py tests/test_nsbh_torus_reference.py
tests/test_nsbh_disruption_reference.py tests/test_nsbh_clean_amplitude.py
tests/test_clean_room_nsbh.py tests/parity/test_nsbh_parity.py`
passed **104/104** in 79.39 s with no skips or expected failures. The log
is `pytest.log` in that scratch directory. Environment: Python 3.12.3,
Torch 2.14.0+cpu, JAX 0.11.2, NumPy 2.5.3, mpmath 1.3.0,
pytest 9.1.1. Remnant/QNM, Eq. (26)/(29) variants, amplitude assembly,
phase, and the whole waveform remain provisional.
