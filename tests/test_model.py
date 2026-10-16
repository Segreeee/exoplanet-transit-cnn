import numpy as np
import pytest

torch = pytest.importorskip("torch")

from exotransit.model import TwoTowerCNN, count_parameters, fit  # noqa: E402
from exotransit.preprocess import GLOBAL_BINS, LOCAL_BINS, N_EXTRA  # noqa: E402


def _batch(n=8):
    g = torch.randn(n, GLOBAL_BINS)
    l_ = torch.randn(n, LOCAL_BINS)
    e = torch.randn(n, N_EXTRA)
    return g, l_, e


def test_forward_shape_and_parameter_budget():
    model = TwoTowerCNN()
    assert model(*_batch(8)).shape == (8,)
    assert count_parameters(model) < 1_500_000  # adatto a una GPU piccola


def test_model_works_without_extra_features():
    model = TwoTowerCNN(n_extra=0)
    g, l_, _ = _batch(4)
    assert model(g, l_, None).shape == (4,)


def test_predict_proba_in_unit_interval_and_eval_mode():
    model = TwoTowerCNN()
    g, l_, e = (t.numpy() for t in _batch(10))
    p = model.predict_proba(g, l_, e, batch_size=4)
    assert p.shape == (10,) and (p >= 0).all() and (p <= 1).all()
    assert not model.training


def _separable_data(n, seed):
    """Classe 1 = dip al centro della vista; classe 0 = solo rumore."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.float32)
    g = rng.normal(0, 0.05, (n, GLOBAL_BINS)).astype(np.float32)
    l_ = rng.normal(0, 0.05, (n, LOCAL_BINS)).astype(np.float32)
    mid = LOCAL_BINS // 2
    l_[y == 1, mid - 10 : mid + 10] -= 1.0
    g[y == 1, GLOBAL_BINS // 2 - 8 : GLOBAL_BINS // 2 + 8] -= 1.0
    e = rng.normal(0, 1, (n, N_EXTRA)).astype(np.float32)
    return g, l_, e, y


def test_fit_learns_a_separable_toy_problem_and_restores_best_weights():
    torch.manual_seed(0)
    train, val = _separable_data(160, 0), _separable_data(60, 1)
    model = TwoTowerCNN(base=8)
    res = fit(model, train, val, epochs=15, batch_size=32, device=torch.device("cpu"))
    assert res.best_val_pr_auc > 0.95
    assert 0 <= res.best_epoch < 15 and len(res.history) <= 15
    assert res.history[-1]["train_loss"] < 0.5 * res.history[0]["train_loss"]
    scores = model.predict_proba(val[0], val[1], val[2])
    assert scores[val[3] == 1].mean() > scores[val[3] == 0].mean() + 0.5


def test_fit_is_reproducible_with_same_seed():
    train, val = _separable_data(64, 2), _separable_data(32, 3)
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        model = TwoTowerCNN(base=8)
        r = fit(model, train, val, epochs=2, batch_size=32, device=torch.device("cpu"), seed=0)
        runs.append(r.history[-1]["train_loss"])
    assert runs[0] == pytest.approx(runs[1], rel=1e-4)


def test_single_tower_variants_ignore_the_disabled_view_and_have_fewer_parameters():
    full = TwoTowerCNN()
    g_only, l_only = TwoTowerCNN(use_local=False), TwoTowerCNN(use_global=False)
    assert count_parameters(g_only) < count_parameters(full)
    assert count_parameters(l_only) < count_parameters(g_only)  # torre locale: 3 stadi, globale 4
    g, l_, e = _batch(6)
    g_only.eval()
    out = g_only(g, l_, e)
    assert out.shape == (6,)
    assert torch.equal(out, g_only(g, torch.randn_like(l_), e))  # la vista locale non conta
    l_only.eval()
    assert torch.equal(l_only(g, l_, e), l_only(torch.randn_like(g), l_, e))


def test_at_least_one_tower_is_required():
    with pytest.raises(ValueError, match="torre"):
        TwoTowerCNN(use_global=False, use_local=False)
