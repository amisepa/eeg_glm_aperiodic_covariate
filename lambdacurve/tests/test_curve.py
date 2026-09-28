import numpy as np
import pytest

from lambdacurve import VERDICTS, lambda_curve, verdict


def exact_slopes(x, s_a, s_b, sd=0.3, seed=0, covariates=None):
    """ln a and ln b whose least-squares slopes on x are exactly s_a and s_b."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones_like(x), x] + ([] if covariates is None else [covariates]))
    E = rng.normal(0, sd, (x.size, 2))
    E -= X @ np.linalg.lstsq(X, E, rcond=None)[0]
    return 1.0 + s_a * x + E[:, 0], 2.0 + s_b * x + E[:, 1]


def exact_means(m_a, m_b, n=40, sd=0.2, seed=0):
    """Paired differences with sample means exactly m_a and m_b."""
    rng = np.random.default_rng(seed)
    E = rng.normal(0, sd, (n, 2))
    E -= E.mean(0)
    return m_a + E[:, 0], m_b + E[:, 1]


def test_lambda_star_of_a_constructed_slope():
    x = np.random.default_rng(1).uniform(5, 20, 400)
    sex = np.random.default_rng(2).integers(0, 2, 400).astype(float)
    la, lb = exact_slopes(x, -0.06, -0.10, covariates=sex)
    r = lambda_curve(la, lb, x=x, covariates=sex, nboot=500)
    assert r.s_a == pytest.approx(-0.06, abs=1e-12)
    assert r.s_b == pytest.approx(-0.10, abs=1e-12)
    assert r.lam_star == pytest.approx(0.6, abs=1e-9)
    assert r.ci[0] < 0.6 < r.ci[1] and r.hdi[0] < 0.6 < r.hdi[1]
    assert r.p_cross_in_01 > 0.9 and r.lam_star_bounded
    # the curve is s_a - lambda s_b and crosses zero at lambda*
    assert np.allclose(r.curve[:, 0], -0.06 + 0.10 * r.grid)
    est, lo, hi = r.effect(r.lam_star)
    assert abs(est) < 1e-12 and lo < 0 < hi
    assert verdict(r) == "reverses"


def test_lambda_star_of_a_constructed_contrast():
    da, db = exact_means(0.9, 0.3)
    r = lambda_curve(da, db, nboot=500)
    assert r.design == "within" and r.n == 40
    assert r.lam_star == pytest.approx(3.0, abs=1e-9)
    assert r.p_cross_in_01 == 0.0
    assert r.lam0[0] == pytest.approx(0.9) and r.lam1[0] == pytest.approx(0.6)


@pytest.mark.parametrize("m_a, m_b, expected", [
    (1.0, 0.2, "holds under both"),     # lambda = 0: +1.0, lambda = 1: +0.8
    (0.3, 0.6, "reverses"),             # +0.3 and -0.3
    (0.5, 0.5, "depends on lambda"),    # +0.5 and 0
    (0.0, 0.0, "null under both"),
])
def test_verdicts(m_a, m_b, expected):
    r = lambda_curve(*exact_means(m_a, m_b), nboot=400)
    assert verdict(r) == expected
    assert r.to_dict()["verdict"] == expected


def test_verdict_from_table_rows():
    row = dict(lam0=-0.05, lam0_lo=-0.07, lam0_hi=-0.03, lam1=0.05, lam1_lo=0.03, lam1_hi=0.06)
    assert verdict(row) == "reverses"
    assert verdict(dict(row, lam1_lo=-0.01)) == "depends on lambda"
    assert verdict(dict(row, lam0_lo=-0.07, lam0_hi=0.01, lam1_lo=-0.01)) == "null under both"
    assert verdict(dict(row, lam1_lo=-0.08, lam1_hi=-0.02)) == "holds under both"
    assert set(VERDICTS) == {"holds under both", "reverses", "depends on lambda", "null under both"}


def test_non_finite_rows_are_not_dropped_silently():
    da, db = exact_means(0.5, 0.2)
    da[:3] = [-np.inf, -np.inf, np.nan]         # what np.log gives for a = 0 and a < 0
    with pytest.raises(ValueError, match="dropna"):
        lambda_curve(da, db)
    r = lambda_curve(da, db, dropna=True, nboot=200)
    assert r.n == 37 and r.n_dropped == 3


def test_unbounded_crossover_is_flagged():
    da, db = exact_means(0.5, 0.0)          # no background change
    r = lambda_curve(da, db, nboot=400)
    assert not r.lam_star_bounded
    assert "unbounded" in str(r)


def test_reproducible_and_argument_checks():
    da, db = exact_means(0.4, 0.3)
    r1, r2 = lambda_curve(da, db, nboot=200, seed=7), lambda_curve(da, db, nboot=200, seed=7)
    assert r1.hdi == r2.hdi and np.array_equal(r1.curve, r2.curve)
    assert lambda_curve(da, db, nboot=200, seed=8).hdi != r1.hdi
    with pytest.raises(ValueError):
        lambda_curve(da, db, covariates=da)
    with pytest.raises(ValueError):
        lambda_curve(da, db[:-1])
