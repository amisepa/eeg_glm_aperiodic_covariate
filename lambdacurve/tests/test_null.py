import numpy as np
import pytest

from lambdacurve import matched_null, spectral_noise

F = np.arange(2, 40.01, 0.5)
ALPHA = (F >= 8) & (F <= 12)


def make_templates(n, seed=0):
    """Each participant's mean spectrum: own background and alpha."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = 10 ** rng.normal(1.5, 0.3) / F ** rng.normal(1.5, 0.2)
        out.append(L * (1 + rng.uniform(0.5, 3) * np.exp(-0.5 * ((F - rng.normal(10, 1)) / 1.5) ** 2)))
    return out


def simulate(template, rng):
    """Eyes open and eyes closed with no change at all: estimation noise only."""
    return spectral_noise(template, 40, rng), spectral_noise(template, 80, rng)


def analyse(pair):
    eo, ec = pair
    return np.log(ec[ALPHA].mean()) - np.log(eo[ALPHA].mean())


def mean(outputs):
    return float(np.mean(outputs))


def test_draws_seed_and_matched_n():
    T = make_templates(25)
    r = matched_null(simulate, analyse, mean, T, n_draws=40, seed=3)
    assert r.draws.shape == (40,) and r.n_draws == 40
    assert r.n == r.n_real == 25
    assert np.array_equal(r.draws, matched_null(simulate, analyse, mean, T, n_draws=40, seed=3).draws)
    assert not np.array_equal(r.draws, matched_null(simulate, analyse, mean, T, n_draws=40, seed=4).draws)
    assert abs(np.mean(r.draws)) < 0.02
    lo, hi = r.interval()
    assert lo < 0 < hi


def test_result_does_not_depend_on_the_map():
    T = make_templates(10)
    backwards = lambda fn, jobs: list(map(fn, list(jobs)[::-1]))[::-1]
    r1 = matched_null(simulate, analyse, mean, T, n_draws=10, seed=1)
    r2 = matched_null(simulate, analyse, mean, T, n_draws=10, seed=1, map_fn=backwards)
    assert np.array_equal(r1.draws, r2.draws)


def test_other_n_is_warned_and_resampled():
    T = make_templates(10)
    with pytest.warns(UserWarning, match="depends on n"):
        r = matched_null(simulate, analyse, mean, T, n_draws=5, n=30)
    assert r.n == 30 and r.n_real == 10
    with pytest.raises(ValueError), pytest.warns(UserWarning):
        matched_null(simulate, analyse, mean, T, n_draws=5, n=30, resample=False)


def test_array_statistics_and_p_values():
    r = matched_null(lambda t, rng: rng.normal(size=3), lambda d: d,
                     lambda outs: np.mean(outs, 0), 12, n_draws=99, seed=0)
    assert r.draws.shape == (99, 3)
    p = r.p_value(np.array([10.0, 0.0, -10.0]), tail="greater")
    assert p[0] == pytest.approx(0.01) and p[2] == 1.0
    assert r.p_value(10.0)[0] == pytest.approx(0.02)
