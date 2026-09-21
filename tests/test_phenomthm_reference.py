"""Checked-in MIT IMRPhenomTHM mode oracle values.

Values were generated independently with phenomxpy/phenomt at commit
4121fafa0484f3db2a7352f4eea697faaaf29f65 (Cecilio Garcia Quiros, 2023--25),
using ``IMRPhenomTHM(..., total_mass=0, numba_ansatze=False)``.  The pinned
MIT source is an update oracle only; it is never imported while testing.
The fixture checks the mode amplitude and the unwrapped complex phase of h_lm
relative to the sample at t=0.  The source's higher-mode ``imr_omega`` helper is not the derivative of the
physical waveform phase, so it is intentionally excluded from this oracle.
"""

import json

import numpy as np
import pytest


_REFERENCE = json.loads(r'''{"cases":{"A":{"modes":{"21":{"amp":[0.0006877943608495448,0.0007314103923758456,0.0008723404492224498,0.00027697832581500696,0.0001018465173689029],"phrel":[-3.5789911532011924,-6.127035225987836,-3.0306104782691183,0.0,-0.9512947988689324]},"22":{"amp":[0.19161000098427658,0.20516227847737123,0.3301027896108619,0.3899818448992198,0.1666987375834275],"phrel":[-2.762866115503244,-1.4535517595022691,-0.6548344109206143,0.0,-3.1049490797942667]},"33":{"amp":[0.004804519310089073,0.005238037995485759,0.011507153916978615,0.0173436299317998,0.009527184536452297],"phrel":[5.745314590851107,4.555630901318002,2.334362012759108,0.0,-2.2923751649012445]},"44":{"amp":[0.004583293271780857,0.005272637488658829,0.013831924936815631,0.020930680540382725,0.009117046554236013],"phrel":[-5.666335389725954,-3.038100537557243,-1.517429540052106,0.0,0.05231723898186491]},"55":{"amp":[0.0003168759990496148,0.00036556659860427694,0.0013564362569142322,0.0026828518622032172,0.001714947894029689],"phrel":[2.6710082032438844,2.8040597210948635,1.4408298607489618,0.0,1.1017461421380976]}},"params":[30.0,25.0,0.2,-0.1]},"B":{"modes":{"21":{"amp":[0.005123628461267095,0.005622313462712778,0.006605447077160312,0.005356050398963173,0.004045751677487139],"phrel":[-6.554824473700778,-4.724436653641676,-2.0634271593082407,0.0,-0.6648568229724177]},"22":{"amp":[0.08326798197970137,0.08848209053404878,0.11722578051923072,0.12372734727224127,0.1033193898184487],"phrel":[3.1235415906865764,0.596991289556327,1.3959488211844258,0.0,1.8266486240867583]},"33":{"amp":[0.021050817366742726,0.02306649461036107,0.03694726002513012,0.044378019036366824,0.04144491019218042],"phrel":[-2.968240761221807,-3.641096905310166,-0.534333439100394,0.0,2.4913834811004643]},"44":{"amp":[0.007701088586766448,0.008717611148938062,0.015614005772795403,0.019527250249332998,0.020512373356491916],"phrel":[-2.9798190130438362,-1.7859902269350942,-3.029609430375623,0.0,2.7104213453133927]},"55":{"amp":[0.0028725428495657928,0.003392824292356744,0.006961323781179447,0.009129119506831686,0.009980481188642469],"phrel":[-0.7574701079901671,2.29982229120721,-0.30799960798885495,0.0,-2.8391908727036523]}},"params":[60.0,6.0,0.9,0.1]},"C":{"modes":{"21":{"amp":[0.007831206786910747,0.008568577470686303,0.0157921578742076,0.02210917379759687,0.011621376272832827],"phrel":[-1.525586045360983,-4.584811592447807,-2.904123477600386,0.0,-0.3592049556964906]},"22":{"amp":[0.1182125459884702,0.12572774451563234,0.18541092916481106,0.20839129060058487,0.10507864834513542],"phrel":[1.86827906651711,2.0702938060412626,-0.35882222481510695,0.0,-2.7526529666470116]},"33":{"amp":[0.02237169337765619,0.02458735533913432,0.04522290959694727,0.059497429450377906,0.03688283390493135],"phrel":[6.288885554273752,3.437914470980372,2.800126272912997,0.0,-1.7383738401049271]},"44":{"amp":[0.006902083016905081,0.007807542531477462,0.017128827977366496,0.0239286650711656,0.012673407983157395],"phrel":[-1.9164018446211835,-1.5258893830638505,-0.5054224942620211,0.0,-0.49740579486060366]},"55":{"amp":[0.0022490814579544105,0.002635675396126709,0.007139862258527454,0.011244290144725947,0.006022397941005075],"phrel":[2.206990569345406,-0.44696703532714377,2.553374609625919,0.0,1.1042439202281864]}},"params":[50.0,10.0,0.5,-0.4]}},"times":[-200.0,-150.0,-20.0,0.0,20.0]}''')


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("case", sorted(_REFERENCE["cases"]))
@pytest.mark.parametrize("mode_key", ["21", "22", "33", "44", "55"])
def test_thm_positive_modes_against_pinned_mit_oracle(backend, case, mode_key):
    """Check amplitude and relative phase for one HM in one backend/case."""
    m1, m2, chi1, chi2 = _REFERENCE["cases"][case]["params"]
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenomthm import IMRPhenomTHM

        def array(v):
            return torch.tensor(v, dtype=torch.float64)

        def abs_(v):
            return torch.abs(v)

        def exp_(v):
            return torch.exp(v)
    else:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomthm import IMRPhenomTHM

        def array(v):
            return jnp.asarray(v, dtype=jnp.float64)

        def abs_(v):
            return jnp.abs(v)

        def exp_(v):
            return jnp.exp(v)

    times = array(_REFERENCE["times"])
    model = IMRPhenomTHM()
    vals = [array(v) for v in (m1, m2, chi1, chi2)]
    state = model._tmodel._coeff(*vals)
    _, eta, delta, spin, dchi = state[:5]
    mode = (int(mode_key[0]), int(mode_key[1]))
    amp, phase, _ = model._positive(
        times, state, mode, eta, delta, spin, dchi, vals[2], vals[3]
    )
    expected = _REFERENCE["cases"][case]["modes"][mode_key]
    got_amp = np.asarray(abs_(amp))
    got_unit = amp * exp_(-1j * phase)
    got_unit = got_unit / abs_(got_unit)
    got_rel = np.asarray(got_unit / got_unit[3])
    np.testing.assert_allclose(
        got_amp,
        expected["amp"],
        rtol=3e-10,
        atol=3e-12,
        err_msg=f"{case} {mode_key} amplitude",
    )
    np.testing.assert_allclose(
        got_rel,
        np.exp(1j * np.asarray(expected["phrel"])),
        rtol=3e-10,
        atol=3e-12,
        err_msg=f"{case} {mode_key} phase",
    )
