"""Unit tests for PyCBC plugin integration with TorchWave."""

import numpy as np
import pytest
import torch

from torchwave.interfaces.pycbc import (
    imrphenomd_fd,
    imrphenomxas_fd,
    imrphenomxhm_fd,
    imrphenomxp_fd,
    imrphenomxphm_fd,
    taylorf2_fd,
    torchwave_fd_plugin,
)

try:
    import pycbc.waveform
    _HAVE_PYCBC = True
except ImportError:
    _HAVE_PYCBC = False


def test_torchwave_fd_plugin_standalone():
    """Verify standalone execution of torchwave_fd_plugin."""
    hp, hc = torchwave_fd_plugin(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=20.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.5,
        f_lower=20.0,
        f_final=256.0,
    )
    if isinstance(hp, torch.Tensor):
        assert hp.shape[0] == int(256.0 / 0.5) + 1
        assert not torch.isnan(hp).any()
        assert hp.abs().max() > 0.0
    else:
        # PyCBC FrequencySeries
        assert len(hp) == int(256.0 / 0.5) + 1
        assert not np.isnan(hp.numpy()).any()
        assert np.abs(hp.numpy()).max() > 0.0


@pytest.mark.skipif(not _HAVE_PYCBC, reason="PyCBC not installed")
def test_pycbc_entrypoints_end_to_end():
    """Verify that PyCBC resolves torch approximants and returns FrequencySeries."""
    for approx in (
        "torch_TaylorF2",
        "torch_IMRPhenomD",
        "torch_IMRPhenomXAS",
        "torch_IMRPhenomXP",
        "torch_IMRPhenomXHM",
        "torch_IMRPhenomXPHM",
    ):
        is_prec = ("XP" in approx) or ("Pv2" in approx) or approx.endswith("PhenomP")
        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant=approx,
            mass1=30.0,
            mass2=20.0,
            spin1z=0.2,
            spin2z=-0.1,
            spin1x=0.1 if is_prec else 0.0,
            spin2x=-0.1 if is_prec else 0.0,
            delta_f=0.5,
            f_lower=20.0,
            f_final=256.0,
            prec_version=223,
        )
        assert hp is not None and hc is not None
        assert hasattr(hp, "delta_f")
        assert float(hp.delta_f) == 0.5
        assert len(hp) == int(256.0 / 0.5) + 1
        assert not np.isnan(hp.numpy()).any()
        assert np.abs(hp.numpy()).max() > 0.0
