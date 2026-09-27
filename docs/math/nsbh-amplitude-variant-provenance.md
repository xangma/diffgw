# IMRPhenomNSBH amplitude variant provenance

## Frozen source decision

Specification **nsbh-amplitude-variants/v1** compares the final
[Thompson et al. (2020), arXiv v2](https://arxiv.org/pdf/2002.08383v2)
with the cited [Pannarale et al. (2015), arXiv v1](https://arxiv.org/pdf/1509.00512v1).
The source of truth for a paper-literal gate is the printed equation, not
binary output. A compatible native variant may be retained without being
called paper-qualified. A source-supported correction would require a
published equation or explicit erratum that uniquely specifies the variant;
continuity arguments and implementation behavior alone do not meet that bar.

The following source claims are frozen before the new numerical diagnostic:

| Component | Printed source | Native compatibility variant | Provenance decision |
| --- | --- | --- | --- |
| Low/no-torus PN center | Thompson Table I: $[(q-1)\widetilde f_{\rm RD}+\epsilon_{\rm ins}f_{\rm tide}]/q$. Pannarale Eq. (26) and its following sentence give the same $f_1$. | An additional outer $\epsilon_{\rm ins}$ multiplies the entire mixed center. | Neither paper prints that outer factor for this branch. The general Pannarale Eq. (13) has an outer inspiral center $\epsilon_{\rm ins}\widetilde f_0$, but its branch-specific Eq. (26) directly sets $f_1$ and so does not uniquely license applying it again. |
| Small-$\Lambda$ premerger correction | Thompson Eq. (26): $\gamma'_1=1-0.5\Lambda-0.25\Lambda^2$ for $\Lambda\le1$, and $1.25$ for $\Lambda>1$. | The polynomial is $1+0.5\Lambda-0.25\Lambda^2$, then multiplied by baseline $\gamma_1$. | The printed polynomial jumps from $0.25$ to $1.25$ at $\Lambda=1$, despite the text's smooth-match claim. Eq. (26) also does not explicitly print the baseline $\gamma_1$ factor. No cited source uniquely resolves both choices. |
| Large-$\Lambda$ ringdown-width correction | Thompson Eq. (29): $(A/2)\omega^-_{x_3,d_3}(x)$, where $x=(f_{\rm tide}-\widetilde f_{\rm RD})/\widetilde f_{\rm RD}$ and $A/2=0.81248$. Pannarale Eq. (20) also prints a minus window, without the later factor-of-two correction. | $0.81248\omega^+_{x_3,d_3}(x)$. | Thompson explicitly identifies the **factor of two** as a correction to Pannarale, but prints no correction of the window sign. Large-$x$ BBH continuity favors the native sign, yet the cited published equations do not establish it. |

Thompson v1 and v2 print the same three relevant variants. Its Eq. (30)
defines $\omega^\pm_{x_0,d}(x)=[1\pm\tanh(4(x-x_0)/d)]/2$.
The 2015 and 2020 formulas differ elsewhere by explicit model revisions;
their shared sign and mixed center are therefore corroborating source
evidence, not proof that the native behavior is scientifically wrong.

## Frozen diagnostic protocol

Use independent 70- and 90-digit mpmath scalar evaluations with absolute
agreement at most $10^{-30}$. Evaluate three controlled, source-discriminating
inputs: $(q,\widetilde f_{\rm RD},f_{\rm tide},\epsilon_{\rm ins})=(3,0.09,0.06,0.8)$;
$\Lambda=1$ approached from the small branch; and
$x=x_3+d_3/4$ for the ringdown correction. Source versus native differences
must be nonzero by more than $10^{-3}$ in their dimensionless representative
values (relative for the frequency, absolute for the correction factors).
Torch and JAX CPU FP64 compatibility helpers must match the independent
**native-variant** scalar values within relative $2\times10^{-13}$.
These are discrepancy and regression checks, **not** a qualification of the
paper-literal formulas. No limit or input may be changed after comparison.

Do not modify the native defaults or run binary fitting. With no uniquely
published correction for all three variants, the next amplitude assembly
component remains blocked. The independent disruption root, torus/ISCO,
branch selection, and scoped branch factors keep their existing status;
remnant/QNM supply and whole-waveform qualification remain separate gates.
