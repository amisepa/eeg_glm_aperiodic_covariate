"""Joint fit of the aperiodic background and the periodic peaks.

The periodic amplitude is estimated as a model parameter (alpha peak plus its
harmonic on a knee+plateau background, Whittle deviance, staged optimisation)
instead of being taken as the residual P_band - b_hat. Used by
sim_joint_calibration.py.
"""
import numpy as np
from scipy.optimize import minimize

from ap_models import fit_aperiodic


def model_psd(theta, f, npk):
    b, chi, lk, lp = theta[0], theta[1], theta[2], theta[3]
    mu = 10.0 ** b / (10.0 ** lk + f ** chi) + 10.0 ** lp
    for n in range(npk):
        A, cf, bw = theta[4 + 3 * n], theta[5 + 3 * n], theta[6 + 3 * n]
        mu = mu + (10.0 ** A) * np.exp(-0.5 * ((f - cf) / bw) ** 2)
    return mu


def neural_bg(theta, f):
    return 10.0 ** theta[0] / (10.0 ** theta[2] + f ** theta[1])


def _dev(theta, f, P, npk):
    mu = np.maximum(model_psd(theta, f, npk), 1e-300)
    return float(np.sum(np.log(mu) + P / mu))


def fit_joint(P, f, peak_guesses, censor=(6.0, 16.0)):
    """Fit aperiodic + peaks over the FULL range.

    peak_guesses: list of (cf, bw_lo, bw_hi, cf_lo, cf_hi) for each peak.
    The censored aperiodic fit is used only to initialise.
    """
    P = np.asarray(P, float)
    f = np.asarray(f, float)
    npk = len(peak_guesses)

    keep = ~((f >= censor[0]) & (f <= censor[1]))
    ap0 = fit_aperiodic(P, f, keep, "knee_plateau")
    if ap0 is None:
        return None
    th0 = [ap0["theta"][0], ap0["theta"][1], ap0["theta"][2], ap0["theta"][3]]
    lo = [th0[0] - 4, 0.05, -5.0, th0[3] - 3]
    hi = [th0[0] + 4, 6.0, 8.0, th0[3] + 2]

    bg0 = neural_bg(np.array(th0), f) + 10.0 ** th0[3]
    for (cf, bwlo, bwhi, cflo, cfhi) in peak_guesses:
        m = np.argmin(np.abs(f - cf))
        amp0 = max(P[m] - bg0[m], bg0[m] * 1e-3)
        th0 += [np.log10(amp0), cf, 0.5 * (bwlo + bwhi)]
        lo += [np.log10(amp0) - 5, cflo, bwlo]
        hi += [np.log10(amp0) + 3, cfhi, bwhi]

    th0 = np.clip(th0, lo, hi)
    bounds = list(zip(lo, hi))

    # Staged optimisation. A cold 10-parameter start does not converge: on
    # spectra simulated from identical parameters it returned chi anywhere
    # between 2.79 and 3.56, with peak and background errors strongly
    # anticorrelated (the two components trade off against each other).
    # Stage 1 holds the aperiodic fixed and fits only the peaks, which is
    # well conditioned; stage 2 releases everything from that point.
    def _dev_peaks(thp, thap):
        return _dev(np.concatenate([thap, thp]), f, P, npk)

    best = None
    for amp_scale in (0.5, 1.0, 2.0):
        th_try = np.array(th0, float)
        for n in range(npk):
            th_try[4 + 3 * n] = np.clip(th0[4 + 3 * n] + np.log10(amp_scale),
                                        lo[4 + 3 * n], hi[4 + 3 * n])
        try:
            r1 = minimize(_dev_peaks, th_try[4:], args=(th_try[:4],),
                          method="L-BFGS-B", bounds=bounds[4:],
                          options=dict(maxiter=600, ftol=1e-12))
            start = np.concatenate([th_try[:4], r1.x])
            r2 = minimize(_dev, start, args=(f, P, npk), method="L-BFGS-B",
                          bounds=bounds, options=dict(maxiter=2000, ftol=1e-12))
        except Exception:
            continue
        if best is None or r2.fun < best.fun:
            best = r2
    if best is None:
        return None
    r = best
    th = r.x
    return dict(theta=th, npk=npk, dev=float(r.fun),
                offset=float(th[0]), exponent=float(th[1]),
                knee=10.0 ** th[2], plateau=10.0 ** th[3],
                knee_freq=(10.0 ** th[2]) ** (1.0 / max(th[1], 1e-6)),
                peaks=[dict(amp=10.0 ** th[4 + 3 * n], cf=th[5 + 3 * n],
                            bw=th[6 + 3 * n]) for n in range(npk)])


def band_ab(fit, f, lo, hi, peak_idx=0):
    """Periodic and neural-aperiodic band power from a joint fit.

    a is the fitted peak integrated over the band -- a PARAMETER, not a
    truncated residual, so it is always positive and no subject is dropped.
    b excludes the instrumental plateau.
    """
    m = (f >= lo) & (f <= hi)
    ff = f[m]
    pk = fit["peaks"][peak_idx]
    a = float(np.mean(pk["amp"] * np.exp(-0.5 * ((ff - pk["cf"]) / pk["bw"]) ** 2)))
    b = float(np.mean(neural_bg(fit["theta"], ff)))
    return a, b


ALPHA_GUESS = (10.0, 0.8, 4.0, 6.5, 14.0)     # cf, bw_lo, bw_hi, cf_lo, cf_hi
HARMONIC_GUESS = (20.0, 1.0, 6.0, 16.0, 26.0)
THETA_GUESS = (5.0, 0.8, 3.5, 3.0, 6.5)
