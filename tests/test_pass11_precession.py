import torch
import pytest

from torchwave.approximants.precession_msa import (
    compute_msa_averaged_couplings,
    compute_msa_final_spin,
    compute_msa_final_spin_squared,
)


def _sample(dtype=torch.float64):
    return dict(
        v0=torch.tensor(0.2, dtype=dtype),
        m1=torch.tensor(35.0, dtype=dtype),
        m2=torch.tensor(25.0, dtype=dtype),
        s1x=torch.tensor(0.2, dtype=dtype),
        s1y=torch.tensor(0.1, dtype=dtype),
        s1z=torch.tensor(0.3, dtype=dtype),
        s2x=torch.tensor(-0.1, dtype=dtype),
        s2y=torch.tensor(0.05, dtype=dtype),
        s2z=torch.tensor(-0.2, dtype=dtype),
    )


def test_msa_averages_match_single_rows_of_batched_evaluation():
    first = _sample()
    second = _sample()
    second.update(
        v0=torch.tensor(0.17, dtype=torch.float64),
        m1=torch.tensor(30.0, dtype=torch.float64),
        m2=torch.tensor(20.0, dtype=torch.float64),
        s1x=torch.tensor(0.15, dtype=torch.float64),
        s1z=torch.tensor(-0.25, dtype=torch.float64),
        s2y=torch.tensor(-0.08, dtype=torch.float64),
        s2z=torch.tensor(0.18, dtype=torch.float64),
    )
    batched = {name: torch.stack((first[name], second[name])) for name in first}
    batch_result = compute_msa_averaged_couplings(**batched)
    assert bool(batch_result[-1].all())
    for row in range(2):
        scalar_result = compute_msa_averaged_couplings(
            **{name: values[row] for name, values in batched.items()}
        )
        for batch_value, scalar_value in zip(batch_result[:3], scalar_result[:3]):
            torch.testing.assert_close(batch_value[row], scalar_value, rtol=0.0, atol=0.0)


def test_msa_averages_mark_degenerate_rows_without_poisoning_batch():
    valid_row = _sample()
    degenerate_row = _sample()
    degenerate_row.update(
        m1=torch.tensor(30.0, dtype=torch.float64),
        m2=torch.tensor(30.0, dtype=torch.float64),
    )
    batched = {
        name: torch.stack((valid_row[name], degenerate_row[name]))
        for name in valid_row
    }
    s_av_sq, s1_l_avg, s2_l_avg, valid = compute_msa_averaged_couplings(**batched)
    assert bool(valid[0])
    assert not bool(valid[1])
    assert torch.isfinite(torch.stack((s_av_sq, s1_l_avg, s2_l_avg))).all()


def test_msa_squared_helper_is_the_orthogonal_vector_norm():
    s_perp = torch.tensor([0.25, -0.125], dtype=torch.float64)
    s_parallel = torch.tensor(-0.25, dtype=torch.float64)
    l_fin = torch.tensor(0.5, dtype=torch.float64)
    s_av_sq = s_perp.square().sum() + s_parallel.square()
    result = compute_msa_final_spin_squared(
        s_av_sq, s_parallel, torch.tensor(0.0), l_fin
    )
    final_vector = torch.cat((s_perp, (s_parallel + l_fin).reshape(1)))
    expected = torch.dot(final_vector, final_vector)
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


def test_msa_squared_helper_preserves_aligned_vector_identity():
    s_parallel = torch.tensor(0.5, dtype=torch.float64)
    l_fin = torch.tensor(-0.25, dtype=torch.float64)
    result = compute_msa_final_spin_squared(
        s_parallel.square(), s_parallel, torch.tensor(0.0), l_fin
    )
    torch.testing.assert_close(result, (s_parallel + l_fin).square(), rtol=0.0, atol=0.0)


def test_msa_averages_and_squared_helper_have_finite_gradients():
    p = _sample()
    for name in ("v0", "s1x", "s1y", "s2x"):
        p[name].requires_grad_()
    l_fin = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    s_av_sq, s1_l_avg, s2_l_avg, valid = compute_msa_averaged_couplings(**p)
    assert bool(valid)
    s_fin_sq = compute_msa_final_spin_squared(s_av_sq, s1_l_avg, s2_l_avg, l_fin)
    (s_fin_sq + s_av_sq + s1_l_avg.square() + s2_l_avg.square()).backward()
    for name in ("v0", "s1x", "s1y", "s2x"):
        assert torch.isfinite(p[name].grad)
    assert torch.isfinite(l_fin.grad)


def test_msa_final_spin_has_aligned_limit_and_mass_scale_invariance():
    p = _sample()
    p.update(s1x=torch.tensor(0.0, dtype=torch.float64), s1y=torch.tensor(0.0, dtype=torch.float64),
             s2x=torch.tensor(0.0, dtype=torch.float64), s2y=torch.tensor(0.0, dtype=torch.float64))
    got = compute_msa_final_spin(**p)
    from torchwave.approximants.imrphenom_xas import XASParameterFits
    eta = p['m1'] * p['m2'] / (p['m1'] + p['m2']).square()
    expected = XASParameterFits.fit_final_spin(eta, p['s1z'], p['s2z'])
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)
    scaled = {k: (v * 3 if k in ('m1', 'm2') else v) for k, v in p.items()}
    torch.testing.assert_close(compute_msa_final_spin(**scaled), got, rtol=1e-10, atol=1e-10)


def test_msa_final_spin_batches_and_differentiates():
    first, second = _sample(), _sample()
    second.update(m1=torch.tensor(30., dtype=torch.float64), m2=torch.tensor(20., dtype=torch.float64),
                  s1x=torch.tensor(.15, dtype=torch.float64))
    batch = {k: torch.stack((first[k], second[k])) for k in first}
    batch['s1x'].requires_grad_()
    result = compute_msa_final_spin(**batch)
    assert result.shape == (2,)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(batch['s1x'].grad).all()


def test_msa_final_spin_near_aligned_limit_and_unsupported_domain_are_explicit():
    p = _sample()
    p.update(s1x=torch.tensor(0., dtype=torch.float64), s1y=torch.tensor(0., dtype=torch.float64),
             s2x=torch.tensor(0., dtype=torch.float64), s2y=torch.tensor(0., dtype=torch.float64))
    aligned = compute_msa_final_spin(**p)
    # Machine-level perturbations use the analytic degenerate limit and do
    # not create a jump from the exactly aligned carrier.
    for x in (1e-12, 1e-10, 1e-8):
        q = dict(p, s1x=torch.tensor(x, dtype=torch.float64))
        torch.testing.assert_close(compute_msa_final_spin(**q), aligned, rtol=1e-10, atol=1e-10)
    q = dict(p, m1=torch.tensor(30., dtype=torch.float64), m2=torch.tensor(30., dtype=torch.float64),
             s1x=torch.tensor(.1, dtype=torch.float64))
    with pytest.raises(ValueError, match="equal-mass"):
        compute_msa_final_spin(**q)


@pytest.mark.parametrize("field,value", [("m1", 0.), ("v0", float("nan"))])
def test_msa_final_spin_rejects_unphysical_inputs(field, value):
    p = _sample()
    p[field] = torch.tensor(value, dtype=torch.float64)
    with pytest.raises(ValueError):
        compute_msa_final_spin(**p)
