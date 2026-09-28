import numpy as np
import pytest
from _sim import gauss, welch_spectrum

from lambdacurve import ap_eval, band_power, fit_aperiodic, fit_mask, peak_frequency

OFF, CHI = 1.5, 1.5                     # log10 offset, exponent
L = lambda f: 10 ** OFF / f ** CHI
PEAK = lambda f: 2.0 * L(10.0) * gauss(f, 10.0, 1.5)     # twice the background at 10 Hz


@pytest.fixture(scope="module")
def alpha_spectrum():
    return welch_spectrum(lambda f: L(f) + PEAK(f), np.random.default_rng(11))


def test_fixed_fit_recovers_parameters_through_a_peak(alpha_spectrum):
    f, P = alpha_spectrum
    s = (f >= 2) & (f <= 40)
    fit = fit_aperiodic(P[s], f[s], fit_mask(f[s], [(6, 16)]), "fixed")
    assert fit["exponent"] == pytest.approx(CHI, abs=0.08)
    assert fit["offset"] == pytest.approx(OFF, abs=0.1)


def test_band_power_splits_the_band(alpha_spectrum):
    f, P = alpha_spectrum
    bp = band_power(P, f, (8, 12), fit_range=(2, 40), censor=[(6, 16)])
    band = f[(f >= 8) & (f <= 12)]
    assert bp["b"] == pytest.approx(L(band).mean(), rel=0.08)
    assert bp["a"] == pytest.approx(PEAK(band).mean(), rel=0.1)
    assert bp["plateau"] == 0.0
    assert peak_frequency(P, f, fit_range=(2, 40)) == pytest.approx(10.0, abs=0.5)


def test_knee_plateau_fit_recovers_parameters():
    f = np.arange(2, 55.01, 0.25)
    theta = [2.5, 2.8, 2.8 * np.log10(5.0), np.log10(0.02)]    # knee at 5 Hz, plateau 0.02
    P = ap_eval(theta, f, "knee_plateau")
    fit = fit_aperiodic(P, f, model="knee_plateau")
    assert fit["exponent"] == pytest.approx(2.8, rel=0.02)
    assert fit["knee_freq"] == pytest.approx(5.0, rel=0.03)
    assert fit["plateau"] == pytest.approx(0.02, rel=0.03)
    assert fit["offset"] == pytest.approx(2.5, abs=0.05)
    # the plateau is instrumental: band power leaves it out of b but not of a
    bp = band_power(P, f, (30, 38), fit_range=(2, 55), censor=None, model="knee_plateau")
    band = f[(f >= 30) & (f <= 38)]
    assert bp["b"] == pytest.approx(ap_eval(theta, band, "knee_plateau", neural=True).mean(), rel=0.03)
    assert bp["plateau"] == pytest.approx(0.02, rel=0.03)
    assert abs(bp["a"]) < 0.01 * bp["tot"]


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        fit_aperiodic(np.ones(10), np.arange(1, 11.0), model="lorentzian")
