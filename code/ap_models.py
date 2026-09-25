"""Aperiodic spectral models fitted by Whittle deviance.

Models (L is linear power):
    fixed         L(f) = 10^b / f^chi
    plateau       L(f) = 10^b / f^chi + p
    knee          L(f) = 10^b / (k + f^chi)
    knee_plateau  L(f) = 10^b / (k + f^chi) + p

The plateau is a flat high-frequency floor (amplifier and line noise), so the
neural part of the background is 10^b / (k + f^chi). In knee mode chi is the
asymptotic high-frequency slope and is not comparable with fixed-mode values.
Matches code/lib/oof_est_kneeplateau.m.
"""
import numpy as np
from scipy.optimize import minimize

MODELS = ("fixed", "plateau", "knee", "knee_plateau")


def n_params(model):
    return {"fixed": 2, "plateau": 3, "knee": 3, "knee_plateau": 4}[model]


def ap_eval(theta, f, model="knee_plateau"):
    """theta = [b, chi, (log10 knee), (log10 plateau)]; returns linear power."""
    b, chi = theta[0], theta[1]
    i = 2
    has_k = model in ("knee", "knee_plateau")
    has_p = model in ("plateau", "knee_plateau")
    k = 10.0 ** theta[i] if has_k else 0.0
    if has_k:
        i += 1
    p = 10.0 ** theta[i] if has_p else 0.0
    return 10.0 ** b / (k + f ** chi) + p


def whittle_dev(theta, f, P, model):
    mu = np.maximum(ap_eval(theta, f, model), 1e-300)
    return float(np.sum(np.log(mu) + P / mu))


def ols_start(P, f):
    lf, lp = np.log10(f), np.log10(P)
    X = np.column_stack([np.ones(f.size), lf])
    beta, *_ = np.linalg.lstsq(X, lp, rcond=None)
    return float(beta[0]), float(-beta[1])


def fit_aperiodic(P, f, keep=None, model="knee_plateau", restarts=2):
    """Fit an aperiodic model to the kept frequencies of one spectrum.

    Returns dict with theta, the model name, the Whittle deviance, and
    convenience fields offset/exponent/knee/plateau. `keep` is a boolean mask
    over f selecting the frequencies used (i.e. the censored fit).
    """
    P = np.asarray(P, float)
    f = np.asarray(f, float)
    if keep is None:
        keep = np.ones_like(f, dtype=bool)
    ff, PP = f[keep], P[keep]
    b0, chi0 = ols_start(PP, ff)

    lo_p = np.log10(max(PP.min() * 1e-4, 1e-18))
    hi_p = np.log10(max(PP.min() * 2.0, 1e-17))
    bounds = {"fixed": [(b0 - 5, b0 + 5), (0.05, 6.0)],
              "plateau": [(b0 - 5, b0 + 5), (0.05, 6.0), (lo_p, hi_p)],
              "knee": [(b0 - 5, b0 + 8), (0.05, 6.0), (-5.0, 8.0)],
              "knee_plateau": [(b0 - 5, b0 + 8), (0.05, 6.0), (-5.0, 8.0),
                               (lo_p, hi_p)]}[model]

    # starting points: no knee, and a knee near the low edge of the fit range
    starts = []
    base = [b0, chi0]
    if model == "fixed":
        starts = [base]
    elif model == "plateau":
        starts = [base + [np.log10(max(PP.min() * 0.3, 1e-18))]]
    else:
        for lk in (-4.0, chi0 * np.log10(max(ff[0] * 1.5, 1.5))):
            th = base + [lk]
            if model == "knee_plateau":
                th = th + [np.log10(max(PP.min() * 0.3, 1e-18))]
            # a knee raises the fitted level, so lift the offset to match
            th[0] = b0 + (lk if lk > 0 else 0.0)
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
               plateau=(10.0 ** th[-1] if model in ("plateau", "knee_plateau")
                        else 0.0),
               n_fit=int(keep.sum()))
    out["knee_freq"] = (out["knee"] ** (1.0 / out["exponent"])
                        if out["knee"] > 0 else 0.0)
    return out


def ap_band_power(fit, f, lo, hi):
    """Mean fitted aperiodic power inside a band."""
    m = (f >= lo) & (f <= hi)
    return float(np.mean(ap_eval(fit["theta"], f[m], fit["model"])))


def compare_models(P, f, keep=None, models=MODELS):
    """Fit each model and return deviances with a BIC-style penalty.

    2*(D_simple - D_complex) is the likelihood-ratio statistic scaled by the
    Gamma shape K; K is unknown here, so report raw deviance differences and
    let the caller scale them.
    """
    out = {}
    for m in models:
        fit = fit_aperiodic(P, f, keep, m)
        if fit is not None:
            out[m] = fit
    return out
