import math
import pytest
import torch

ls = pytest.importorskip("lalsimulation")
lal = pytest.importorskip("lal")
from torchwave.approximants.phenompv2_waveform_clean import _geometry, effective_chip


CASES = [
    (35., 25., .2, .1, .2, -.1, .15, -.1, .7, .3, 20.),
    (42., 17., .3, -.15, -.3, .12, .08, .2, 1.1, .71, 35.),
    (40., 20., .3, .2, .4, -.2, .1, -.2, .7, 0., 20.),
    (12., 3., .7, -.2, .4, -.3, .1, -.2, 1.0, .4, 30.),
    (35., 25., 0., 0., .2, 0., 0., -.1, .7, .3, 20.),
]

def _periodic(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def _public(c):
    m1,m2,x,y,z,X,Y,Z,inc,phi,fr = c
    return ls.SimIMRPhenomPCalculateModelParametersFromSourceFrame(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, fr, phi, inc, x,y,z,X,Y,Z,
        ls.IMRPhenomPv2_V)

@pytest.mark.parametrize("case", CASES)
def test_clean_source_frame_matches_public(case):
    vals = [torch.tensor(float(x), dtype=torch.float64) for x in case[:8]]
    got = _geometry(*vals, torch.tensor(case[8],dtype=torch.float64),
                    torch.tensor(case[9],dtype=torch.float64),
                    torch.tensor(case[10],dtype=torch.float64))
    pub = _public(case)
    chip = float(effective_chip(*vals[:2], *vals[2:4], *vals[5:7]))
    assert abs(_periodic(float(got[0]), pub[4])) < 1e-12
    assert abs(float(got[1]) - pub[3]) < 1e-12
    if abs(case[2]) + abs(case[3]) + abs(case[5]) + abs(case[6]) > 0:
        assert abs(_periodic(float(got[3]), pub[5])) < 1e-12
    assert abs(_periodic(float(got[2]), pub[6])) < 1e-12
    assert abs(chip - pub[2]) < 1e-12

def test_aligned_limit_is_finite_without_alpha_claim():
    for inc in (0., math.pi, .7):
        c=(35.,25.,0.,0.,.2,0.,0.,-.1,inc,.3,20.)
        vals=[torch.tensor(float(x),dtype=torch.float64) for x in c[:8]]
        out=_geometry(*vals,torch.tensor(inc,dtype=torch.float64),torch.tensor(.3,dtype=torch.float64),torch.tensor(20.,dtype=torch.float64))
        assert all(torch.isfinite(x) for x in out)
