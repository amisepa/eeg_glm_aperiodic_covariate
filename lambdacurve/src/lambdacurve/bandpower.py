"""Periodic and aperiodic power in a band, from one spectrum."""
import numpy as np

from .aperiodic import ap_band_power, fit_aperiodic, fit_mask


def peak_frequency(P, f, search=(6.0, 14.0), censor=((6.0, 16.0),), fit_range=None):
    """Frequency of the largest positive residual above a censored log-log line.

    The line is fitted by least squares to log10 power outside the censor
    windows, over fit_range if given (default: all of f). Returns nan when
    no frequency in the search range lies above the line.
    """
    P, f = np.asarray(P, float), np.asarray(f, float)
    keep = fit_mask(f, censor) & (f > 0)
    if fit_range is not None:
        keep &= (f >= fit_range[0]) & (f <= fit_range[1])
    X = np.column_stack([np.ones(keep.sum()), np.log10(f[keep])])
    beta = np.linalg.lstsq(X, np.log10(P[keep]), rcond=None)[0]
    s = (f >= search[0]) & (f <= search[1]) & (f > 0)
    resid = P[s] - 10.0 ** (beta[0] + beta[1] * np.log10(f[s]))
    if not np.any(resid > 0):
        return np.nan
    return float(f[s][np.argmax(resid)])


def band_power(P, f, band, fit_range=(2.0, 55.0), censor=((6.0, 16.0),), model="fixed"):
    """Total, aperiodic and periodic power in a band.

    P, f: one spectrum (linear power) and its frequencies. The aperiodic
    model is fitted over fit_range with the censor windows left out, by
    Whittle deviance. Censor a wide window around the peak (6-16 Hz for
    alpha) and, if the band's harmonic falls inside the fit range, that too:
    peak and harmonic power leaking into the fit makes b follow the peak.

    Returns a dict, or None if the fit fails:
        tot       mean observed power in the band
        b         mean fitted neural background in the band (plateau excluded)
        plateau   fitted plateau (0 for models without one)
        a         tot - b - plateau, the periodic power. It can be <= 0;
                  keep such values rather than dropping them (see lambda_curve)
        exponent, offset, knee_freq, dev   from the aperiodic fit
    """
    f = np.asarray(f, float)
    sel = (f >= fit_range[0]) & (f <= fit_range[1]) & (f > 0)
    ff = f[sel]
    PP = np.clip(np.nan_to_num(np.asarray(P, float)[sel], nan=1e-12), 1e-12, None)
    fit = fit_aperiodic(PP, ff, fit_mask(ff, censor), model)
    if fit is None:
        return None
    lo, hi = band
    inband = (ff >= lo) & (ff <= hi)
    if not inband.any():
        raise ValueError("no frequencies of the fit range fall inside the band")
    b = ap_band_power(fit, ff, lo, hi, neural=True)
    b_all = ap_band_power(fit, ff, lo, hi, neural=False)     # b + plateau
    tot = float(np.mean(PP[inband]))
    return dict(tot=tot, b=b, plateau=float(fit["plateau"]), a=tot - b_all,
                exponent=fit["exponent"], offset=fit["offset"],
                knee_freq=fit["knee_freq"], dev=fit["dev"])
