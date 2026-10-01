import numpy as np
import pytest

from lambdacurve import from_specparam, lambda_curve


def test_recovers_power_at_the_peak():
    # a spectrum built as specparam models it: log10 P = L + pw at cf
    offset, exponent, cf, pw = 1.2, 1.5, 10.0, 0.6
    b = 10.0 ** offset / cf ** exponent
    total = 10.0 ** (np.log10(b) + pw)
    la, lb = from_specparam(offset, exponent, cf, pw)
    assert np.exp(lb) == pytest.approx(b)
    assert np.exp(la) == pytest.approx(total - b)


def test_knee_mode():
    offset, knee, exponent, cf, pw = 2.0, 30.0, 2.2, 9.0, 0.4
    b = 10.0 ** offset / (knee + cf ** exponent)
    la, lb = from_specparam(offset, exponent, cf, pw, knee=knee)
    assert np.exp(lb) == pytest.approx(b)
    assert np.exp(la) == pytest.approx((10.0 ** pw - 1.0) * b)


def test_no_peak_is_nan_and_lambda_curve_refuses_it():
    la, lb = from_specparam([1.0, 1.0, 1.0], [1.5, 1.5, 1.5],
                            [10.0, np.nan, 10.0], [0.5, np.nan, 0.0])
    assert np.isfinite(la[0]) and np.isnan(la[1]) and np.isnan(la[2])
    with pytest.raises(ValueError):
        lambda_curve(np.r_[la, la], np.r_[lb, lb], x=np.arange(6.0))


def test_constant_peak_height_is_lambda_one():
    # the same pw at every age: no effect at lambda = 1, the background's at lambda = 0
    rng = np.random.default_rng(0)
    age = rng.uniform(16, 75, 300)
    offset = 1.0 - 0.004 * age + rng.normal(0, 0.1, 300)
    la, lb = from_specparam(offset, np.full(300, 1.5), np.full(300, 10.0), np.full(300, 0.5))
    r = lambda_curve(la, lb, x=age, nboot=200)
    assert r.lam1[0] == pytest.approx(0.0, abs=1e-12)
    assert r.s_a == pytest.approx(r.s_b)
