"""Black-box NSBH amplitude-limit probe for independent strict failures.

The public NSBH construction (Thompson et al., arXiv:2002.08383, Sec. II.A
and Appendix A) uses a disruption/tidal-radius branch for the NSBH amplitude,
while Lambda=0 is the black-hole limit C=1/2.  This records the public
callable outputs at that limit and compares the resulting waveform directly
with LAL, without fitting or changing the model.
"""
import hashlib, json, pathlib, sys
import lal, lalsimulation as ls
import numpy as np
import torch

from torchwave import get_fd_waveform
from torchwave.approximants import imrphenom_nsbh

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = {
    "independent_c": dict(m1=3.2, m2=1.2, chi1=-.25, lam=800., fref=32.),
    "independent_f": dict(m1=9., m2=1.1, chi1=-.4, lam=2000., fref=42.),
}


def binary(c, lam):
    d = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertTidalLambda1(d, 0.)
    ls.SimInspiralWaveformParamsInsertTidalLambda2(d, lam)
    args = [c["m1"] * lal.MSUN_SI, c["m2"] * lal.MSUN_SI, 0., 0., c["chi1"], 0.,
            0., 0., 100e6 * lal.PC_SI, .5, .3, 0., 0., 0., .25, 30., 1024.,
            c["fref"], d, ls.IMRPhenomNSBH]
    hp, hc = ls.SimInspiralChooseFDWaveform(*args)
    return np.asarray(hp.data.data, dtype=np.complex128), np.asarray(hc.data.data, dtype=np.complex128)


def native(c, lam):
    f = torch.arange(0., 1024.0001, .25, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        "IMRPhenomNSBH", mass1=c["m1"], mass2=c["m2"], chi1=c["chi1"],
        chi2=0., lambda1=0., lambda2=lam, distance=100., inclination=.5,
        phic=.3, f_ref=c["fref"], f_lower=30., f_final=1024.,
        sample_frequencies=f, dtype=torch.float64)
    return f.numpy(), hp.detach().numpy(), hc.detach().numpy()


def main(path="docs/nsbh_amplitude_limit_diagnostic.json"):
    rows = []
    for name, c in CASES.items():
        m1, m2, s = (torch.tensor(c[k], dtype=torch.float64) for k in ("m1", "m2", "chi1"))
        for lam in (0., 1e-6, .1, 1., 100., c["lam"]):
            f, hp, hc = native(c, lam)
            bp, bc = binary(c, lam)
            public_properties = ls.SimIMRPhenomNSBHProperties(
                c["m1"] * lal.MSUN_SI, c["m2"] * lal.MSUN_SI, c["chi1"], lam)
            mf, sf = imrphenom_nsbh.zappa_remnant_properties(
                m1, m2, s, torch.zeros_like(s), torch.tensor(lam, dtype=torch.float64))
            ft, fi, fr, compactness = imrphenom_nsbh.nsbh_frequencies(
                m1, m2, s, torch.tensor(lam, dtype=torch.float64))
            finite = bool(np.isfinite(hp).all() and np.isfinite(hc).all())
            assert finite
            rows.append({"case": name, "lambda2": lam,
                "properties": {"remnant_mass_msun": float(mf), "remnant_spin": float(sf)},
                "frequency_properties_hz": {"f_tide": float(ft), "f_isco": float(fi),
                    "f_rd": float(fr), "compactness": float(compactness)},
                "public_property_outputs": [float(x) for x in public_properties],
                "native_support": int(np.count_nonzero(np.abs(hp) > 0)),
                "binary_support": int(np.count_nonzero(np.abs(bp) > 0)),
                "native_amp_at_30_100_500": [float(np.abs(hp[i])) for i in (120, 400, 2000)],
                "binary_amp_at_30_100_500": [float(np.abs(bp[i])) for i in (120, 400, 2000)],
                "native_binary_relative_amp_l2": float(np.linalg.norm(np.abs(hp)-np.abs(bp)) /
                    np.linalg.norm(np.abs(bp))),
                "native_binary_relative_complex_l2": float(np.linalg.norm(hp-bp) /
                    np.linalg.norm(bp)),
                "finite": finite})
    hashes = {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
        "scripts/nsbh_amplitude_limit_diagnostic.py",
        "src/torchwave/approximants/imrphenom_nsbh.py")}
    pathlib.Path(path).write_text(json.dumps({"metadata": {"python": sys.executable,
        "source_sha256": hashes, "equation_basis": "Thompson et al. arXiv:2002.08383 Sec. II.A and Appendix A; Lambda=0 black-hole limit C=1/2"}, "records": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/nsbh_amplitude_limit_diagnostic.json")
