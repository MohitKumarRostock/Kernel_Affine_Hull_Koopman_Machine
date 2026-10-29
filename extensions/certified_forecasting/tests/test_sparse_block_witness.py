"""Independent mathematical checks for the signed-edge block dual."""
import numpy as np
from certified_forecasting.block_witness import lipschitz_block_bracket as dense_bracket
from certified_forecasting.sparse_block_witness import lipschitz_block_brackets


def test_signed_dual_recovers_closed_form_pairs():
    rng = np.random.default_rng(491701)
    x = rng.normal(size=(40, 2, 3))
    y = rng.normal(size=(40, 2))
    r = lipschitz_block_brackets(x, y, .4, tolerance=1e-9)
    exact = .5*np.maximum(np.abs(y[:, 0]-y[:, 1])-.4*np.linalg.norm(x[:, 0]-x[:, 1], axis=1), 0.)**2
    np.testing.assert_allclose(r['lower'], exact, atol=2e-9)
    assert np.max(r['gap']) < 3e-9


def test_collapsed_and_nearly_collapsed_features():
    rng = np.random.default_rng(491702)
    y = rng.uniform(-1., 1., (12, 8))
    x = np.zeros((12, 8, 2))
    r = lipschitz_block_brackets(x, y, 2., tolerance=1e-9)
    exact = np.sum((y-y.mean(axis=1)[:, None])**2, axis=1)/7
    np.testing.assert_allclose(r['lower'], exact, atol=1e-9)
    x[:, :, 0] = rng.uniform(0., 1e-11, (12, 8))
    x[:, :, 1] = 1.-x[:, :, 0]
    near = lipschitz_block_brackets(x, y, 2., tolerance=1e-9)
    np.testing.assert_allclose(near['lower'], exact, atol=1e-8)
    assert np.max(near['gap']) < 2e-9


def test_sparse_matches_independent_dense_programs_and_max_block_metric():
    rng = np.random.default_rng(491703)
    x = rng.normal(size=(12, 7, 4))
    y = rng.uniform(-1., 1., (12, 7))
    slices = [slice(0, 2), slice(2, 4)]
    r = lipschitz_block_brackets(x, y, .12, slices, tolerance=1e-9)
    for q in range(len(y)):
        d = dense_bracket(x[q], y[q], .12, slices)
        assert r['lower'][q] <= d['upper']+1e-10
        assert d['lower'] <= r['upper'][q]+1e-10
        assert r['gap'][q] < 2e-9
        i, j = np.triu_indices(7, 1)
        alpha = r['signed_dual'][q]
        div = np.zeros(7)
        np.add.at(div, i, alpha)
        np.add.at(div, j, -alpha)
        distances = np.maximum.reduce([np.linalg.norm(x[q, i, s]-x[q, j, s], axis=1) for s in slices])
        dual = alpha@(y[q, i]-y[q, j])-.12*np.abs(alpha)@distances-.25*(div@div)
        np.testing.assert_allclose(dual, r['dual_raw'][q], atol=1e-12)


def test_stopping_limit_retains_valid_lower_values():
    rng = np.random.default_rng(491704)
    x = rng.normal(size=(5, 8, 3))
    y = rng.uniform(-1., 1., (5, 8))
    r = lipschitz_block_brackets(x, y, .2, max_iter=1, check_every=1, tolerance=1e-12)
    for q in range(len(y)):
        exact = dense_bracket(x[q], y[q], .2)
        assert r['lower'][q] <= exact['upper']+1e-10
        assert np.all(r['primal'][q] <= y[q].max()+1e-12)
        assert np.all(r['primal'][q] >= y[q].min()-1e-12)
    assert not np.all(r['converged'])
