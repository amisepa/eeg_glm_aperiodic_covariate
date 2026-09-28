"""Aperiodic spectral models fitted by Whittle deviance.

Models (L is linear power):
    fixed         L(f) = 10^b / f^chi
    plateau       L(f) = 10^b / f^chi + p
    knee          L(f) = 10^b / (k + f^chi)
    knee_plateau  L(f) = 10^b / (k + f^chi) + p

The plateau p is a flat high-frequency floor (amplifier noise, residual
muscle), not neural background, so ap_band_power leaves it out unless asked.
In the knee models chi is the asymptotic high-frequency slope and is not
comparable with a fixed-model exponent.

The deviance sum(log mu + P / mu) is the Gamma log-likelihood of a Welch
estimate up to a constant. Least squares on log power instead sits below the
mean (by e^-0.577 for a single-taper periodogram), and the residual a = P - L
then grows with the background.
"""
import numpy as np
from scipy.optimize import minimize

MODELS = ("fixed", "plateau", "knee", "knee_plateau")


def n_params(model):
    return {"fixed": 2, "plateau": 3, "knee": 3, "knee_plateau": 4}[model]


def fit_mask(f, censor):
    """True for frequencies outside every (lo, hi) censor window."""
    f = np.asarray(f, float)
    keep = np.ones(f.size, bool)
    for lo, hi in censor or ():
        keep &= ~((f >= lo) & (f <= hi))
    return keep


def ap_eval(theta, f, model="fixed", neural=False):
    """Aperiodic power at f for theta = [b, chi, (log10 k), (log10 p)].

    neural=True leaves out the plateau.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    b, chi = theta[0], theta[1]
    i = 2
    has_k = model in ("knee", "knee_plateau")
    has_p = model in ("plateau", "knee_plateau")
    k = 10.0 ** theta[i] if has_k else 0.0
    if has_k:
        i += 1
    p = 10.0 ** theta[i] if (has_p and not neural) else 0.0
    return 10.0 ** b / (k + f ** chi) + p


def whittle_dev(theta, f, P, model):
    mu = np.maximum(ap_eval(theta, f, model), 1e-300)
    return float(np.sum(np.log(mu) + P / mu))


def _ols_start(P, f):
    X = np.column_stack([np.ones(f.size), np.log10(f)])
    beta, *_ = np.linalg.lstsq(X, np.log10(P), rcond=None)
    return float(beta[0]), float(-beta[1])


def fit_aperiodic(P, f, keep=None, model="fixed", restarts=2):
    """Fit an aperiodic model to the kept frequencies of one spectrum.

    P, f: linear power and frequencies (f > 0). keep: boolean mask over f of
    the frequencies to fit, e.g. fit_mask(f, [(6, 16)]) for a censored fit;
    default all. Returns a dict with theta, model, dev (Whittle deviance),
    offset, exponent, knee, knee_freq, plateau and n_fit, or None if every
    start failed. Use knee_plateau to report aperiodic parameters of real
    scalp spectra; a flexible model also absorbs peaks, so keep the simpler
    one for band power used in a coupling analysis.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    P = np.asarray(P, float)
    f = np.asarray(f, float)
    if keep is None:
        keep = np.ones_like(f, dtype=bool)
    keep = np.asarray(keep, bool)
    ff, PP = f[keep], P[keep]
    b0, chi0 = _ols_start(PP, ff)

    lo_p = np.log10(max(PP.min() * 1e-4, 1e-18))
    hi_p = np.log10(max(PP.min() * 2.0, 1e-17))
    bounds = {"fixed": [(b0 - 5, b0 + 5), (0.05, 6.0)],
              "plateau": [(b0 - 5, b0 + 5), (0.05, 6.0), (lo_p, hi_p)],
              "knee": [(b0 - 5, b0 + 8), (0.05, 6.0), (-5.0, 8.0)],
              "knee_plateau": [(b0 - 5, b0 + 8), (0.05, 6.0), (-5.0, 8.0),
                               (lo_p, hi_p)]}[model]

    # starting points: no knee, and a knee near the low edge of the fit range
    base = [b0, chi0]
    if model == "fixed":
        starts = [base]
    elif model == "plateau":
        starts = [base + [np.log10(max(PP.min() * 0.3, 1e-18))]]
    else:
        starts = []
        for lk in (-4.0, chi0 * np.log10(max(ff[0] * 1.5, 1.5))):
            th = base + [lk]
            if model == "knee_plateau":
                th = th + [np.log10(max(PP.min() * 0.3, 1e-18))]
            th[0] = b0 + (lk if lk > 0 else 0.0)   # a knee raises the level
            starts.append(th)
        starts = starts[:max(restarts, 1)]

    best = None
    for th0 in starts:
        th0 = np.clip(th0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            r = minimize(whittle_dev, th0, args=(ff, PP, model),
                         method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=800, ftol=1e-12))
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        return None
    th = best.x
    out = dict(theta=th, model=model, dev=float(best.fun),
               offset=float(th[0]), exponent=float(th[1]),
               knee=(10.0 ** th[2] if model in ("knee", "knee_plateau") else 0.0),
               plateau=(10.0 ** th[-1] if model in ("plateau", "knee_plateau") else 0.0),
               n_fit=int(keep.sum()))
    out["knee_freq"] = out["knee"] ** (1.0 / out["exponent"]) if out["knee"] > 0 else 0.0
    return out


def ap_band_power(fit, f, lo, hi, neural=True):
    """Mean fitted aperiodic power over the frequencies of f in [lo, hi].

    neural=True (default) excludes the plateau: an oscillation should not be
    treated as coupled to amplifier noise. The plateau is negligible in the
    alpha band but can be a quarter of the background above 30 Hz.
    """
    f = np.asarray(f, float)
    m = (f >= lo) & (f <= hi)
    return float(np.mean(ap_eval(fit["theta"], f[m], fit["model"], neural=neural)))
