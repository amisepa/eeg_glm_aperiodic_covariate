import numpy as np
import pytest

from lambdacurve import coupling_levels, coupling_two_conditions


def two_conditions(lam, n=150, c0=0.8, t_shape=400, seed=1):
    """Split-half band powers, a = c b^lambda with c independent of b, and a
    common intrinsic change of ln 2.5 in condition 2. The background level
    varies between subjects and its change varies too (the identifying
    variation). t_shape sets the noise in the total (Gamma, mean one)."""
    rng = np.random.default_rng(seed)
    lb1 = rng.normal(0, 0.7, n)
    b1, b2 = np.exp(lb1), np.exp(lb1 + rng.normal(0.3, 0.3, n))
    c = c0 * np.exp(rng.normal(0, 0.3, n))
    a1, a2 = c * b1 ** lam, 2.5 * c * b2 ** lam
    total = lambda v: v[:, None] * rng.gamma(t_shape, 1 / t_shape, (n, 2))
    fitted = lambda v: v[:, None] * rng.gamma(2000, 1 / 2000, (n, 2))
    return total(a1 + b1), total(a2 + b2), fitted(b1), fitted(b2)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_two_conditions_recovers_lambda(lam):
    r = coupling_two_conditions(*two_conditions(lam), nboot=60)
    assert r.lam == pytest.approx(lam, abs=0.1)
    assert r.delta == pytest.approx(np.log(2.5), abs=0.1)
    assert r.ci[0] < lam < r.ci[1]
    assert r.instrument_r > 0.9 and r.boot_fail < 0.1


def test_two_conditions_keeps_negative_periodic_estimates():
    # weak rhythm and noisy totals: many a = t - b come out <= 0 and stay in
    r = coupling_two_conditions(*two_conditions(0.0, n=300, c0=0.12, t_shape=150, seed=4), nboot=0)
    assert r.frac_a_nonpositive > 0.05
    assert r.lam == pytest.approx(0.0, abs=0.3)


def levels(lam, units=30, epochs=40, seed=2):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(units), epochs)
    b = np.exp(rng.normal(0, 0.7, units)[g] + rng.normal(0, 0.3, g.size))
    a = np.exp(rng.normal(0, 0.3, units))[g] * b ** lam * rng.gamma(25, 1 / 25, g.size)
    T = (a + b) * rng.gamma(400, 1 / 400, g.size)
    B, Bz = (b * rng.gamma(2000, 1 / 2000, g.size) for _ in range(2))
    return T, B, Bz, g


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_levels_recovers_lambda(lam):
    r = coupling_levels(*levels(lam), nboot=20)
    assert r.lam == pytest.approx(lam, abs=0.15)
    assert r.n == 30 and r.instrument_r > 0.9


def test_input_checks():
    t1, t2, b1, b2 = two_conditions(0.0, n=20)
    with pytest.raises(ValueError):
        coupling_two_conditions(t1[:, 0], t2, b1, b2)
    T, B, Bz, g = levels(0.0, units=3, epochs=5)
    B[0] = 0.0
    with pytest.raises(ValueError):
        coupling_levels(T, B, Bz, g)
