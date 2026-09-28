"""How a periodic-power effect depends on the assumed coupling exponent.

With a the periodic and b the aperiodic power in a band, periodic power
under an assumed lambda is y = ln a - lambda ln b: lambda = 0 subtracts the
background in linear power (as IRASA does), lambda = 1 divides it out (as
specparam and dB baselines do). Any linear effect on y is linear in lambda,

    s(lambda) = s_a - lambda s_b,

with s_a and s_b the same effect computed on ln a and on ln b. It changes
sign at the crossover lambda* = s_a / s_b. A crossover outside [0, 1] means
the conclusion does not depend on the assumption; one inside [0, 1] means
it does.
"""
from dataclasses import dataclass, field

import numpy as np

GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
VERDICTS = ("holds under both", "reverses", "depends on lambda", "null under both")


def hdi(x, mass=0.95):
    """Shortest interval containing `mass` of the finite values of x."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    if x.size < 2:
        return (np.nan, np.nan)
    n = int(np.floor(mass * x.size))
    i = int(np.argmin(x[n:] - x[:x.size - n]))
    return float(x[i]), float(x[i + n])


def _coef(Y, X, w=None):
    """(Weighted) least-squares coefficient of column 1 of X, for each column of Y."""
    if w is None:
        return np.linalg.lstsq(X, Y, rcond=None)[0][1]
    WX = X * w[:, None]
    return np.linalg.solve(X.T @ WX, WX.T @ Y)[1]


def _pct(x):
    return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))


@dataclass
class LambdaCurve:
    """Result of lambda_curve. Intervals are 95%.

    s_a, s_b        effect on ln a and on ln b (slope, or mean difference)
    lam_star        crossover s_a / s_b
    ci              pairs-bootstrap percentile interval of lam_star
    hdi             Bayesian-bootstrap highest-density interval of lam_star
    p_cross_in_01   share of Bayesian-bootstrap draws with lam_star in [0, 1]
    s_a_ci, s_b_ci  pairs-bootstrap intervals of s_a and s_b
    grid, curve     lambda values; estimate, lower and upper limit at each
    n, n_dropped    rows used, and rows dropped as non-finite
    design          "between" (regression on x) or "within" (mean difference)
    boot, bayes     bootstrap draws of (s_a, s_b), nboot x 2
    """
    n: int
    n_dropped: int
    design: str
    s_a: float
    s_b: float
    lam_star: float
    ci: tuple
    hdi: tuple
    p_cross_in_01: float
    s_a_ci: tuple
    s_b_ci: tuple
    grid: np.ndarray
    curve: np.ndarray
    boot: np.ndarray = field(repr=False)
    bayes: np.ndarray = field(repr=False)

    def effect(self, lam):
        """Estimate and 95% pairs-bootstrap interval of s(lam), for any lam."""
        est = self.s_a - lam * self.s_b
        lo, hi = _pct(self.boot[:, 0] - lam * self.boot[:, 1])
        return float(est), lo, hi

    @property
    def lam0(self):
        """Effect with the background subtracted in linear power (lambda = 0)."""
        return self.effect(0.0)

    @property
    def lam1(self):
        """Effect with the background divided out (lambda = 1)."""
        return self.effect(1.0)

    @property
    def lam_star_bounded(self):
        """False when the interval of s_b includes zero. lam_star is then
        unbounded (the background does not change reliably with the
        predictor): ci and hdi are not meaningful, while the effects at
        lambda = 0 and 1 and p_cross_in_01 still are."""
        lo, hi = self.s_b_ci
        return bool(lo > 0 or hi < 0)

    def to_dict(self):
        """Flat dict with the column names of the research tables."""
        (e0, lo0, hi0), (e1, lo1, hi1) = self.lam0, self.lam1
        return dict(n=self.n, n_dropped=self.n_dropped, s_a=self.s_a, s_b=self.s_b,
                    lam_star=self.lam_star, ci_lo=self.ci[0], ci_hi=self.ci[1],
                    hdi_lo=self.hdi[0], hdi_hi=self.hdi[1],
                    p_cross_in_01=self.p_cross_in_01,
                    s_b_lo=self.s_b_ci[0], s_b_hi=self.s_b_ci[1],
                    lam0=e0, lam0_lo=lo0, lam0_hi=hi0, lam1=e1, lam1_lo=lo1, lam1_hi=hi1,
                    verdict=verdict(self))

    def __str__(self):
        (e0, lo0, hi0), (e1, lo1, hi1) = self.lam0, self.lam1
        what = "slope" if self.design == "between" else "mean difference"
        lines = [
            f"lambda-curve, {what}, n = {self.n}"
            + (f" ({self.n_dropped} non-finite rows dropped)" if self.n_dropped else ""),
            f"  s_a (ln a) {self.s_a:+.4g}   s_b (ln b) {self.s_b:+.4g} "
            f"[{self.s_b_ci[0]:+.4g}, {self.s_b_ci[1]:+.4g}]",
            f"  lambda = 0: {e0:+.4g} [{lo0:+.4g}, {hi0:+.4g}]",
            f"  lambda = 1: {e1:+.4g} [{lo1:+.4g}, {hi1:+.4g}]",
            f"  lambda* {self.lam_star:.3g}, CI [{self.ci[0]:.3g}, {self.ci[1]:.3g}], "
            f"HDI [{self.hdi[0]:.3g}, {self.hdi[1]:.3g}], "
            f"P(lambda* in [0, 1]) {self.p_cross_in_01:.3f}",
            f"  verdict: {verdict(self)}",
        ]
        if not self.lam_star_bounded:
            lines.append("  s_b's interval includes 0: lambda* is unbounded, "
                         "read its CI and HDI as uninformative")
        return "\n".join(lines)


def lambda_curve(la, lb, x=None, covariates=None, grid=None, nboot=2000, seed=0,
                 dropna=False):
    """Effect on ln a - lambda ln b as a function of lambda.

    Between subjects: la, lb are ln a and ln b per subject, x the predictor
    of interest and covariates (n, or n x k) nuisance terms; the effect is
    the least-squares coefficient of x. Within subjects: x is None and la,
    lb are paired differences (condition 2 - condition 1) of ln a and ln b;
    the effect is the mean difference.

    ln a does not exist for a <= 0. By default any non-finite row raises an
    error, because dropping those rows selects on the periodic estimate,
    which is correlated with the background. dropna=True drops them; report
    result.n_dropped with the result.

    nboot pairs-bootstrap and Bayesian-bootstrap draws are taken from
    numpy's default_rng(seed) (alternating, as in code/lambda_curve.py, so
    the same seed gives the same numbers). Returns a LambdaCurve.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    grid = GRID if grid is None else np.asarray(grid, float)
    la = np.asarray(la, float).ravel()
    lb = np.asarray(lb, float).ravel()
    if la.shape != lb.shape:
        raise ValueError("la and lb must have the same length")
    if x is None and covariates is not None:
        raise ValueError("covariates need a predictor x")
    cols = [np.ones_like(la)]
    if x is not None:
        cols.append(np.asarray(x, float).ravel())
        if covariates is not None:
            C = np.asarray(covariates, float)
            cols += list(C.T if C.ndim == 2 else [C])
    X = np.column_stack(cols)
    if X.shape[0] != la.size:
        raise ValueError("x and covariates must have one row per subject")
    ok = np.isfinite(la) & np.isfinite(lb) & np.all(np.isfinite(X), 1)
    n_dropped = int((~ok).sum())
    if n_dropped and not dropna:
        raise ValueError(
            f"{n_dropped} of {ok.size} rows are not finite (ln a of a <= 0?). "
            "Dropping them selects on the periodic estimate; pass dropna=True "
            "to drop them anyway and report result.n_dropped.")
    la, lb, X = la[ok], lb[ok], X[ok]
    Y = np.column_stack([la, lb])
    n = la.size
    between = x is not None
    if n < X.shape[1] + 2:
        raise ValueError("too few complete rows")

    def coefs(w=None):
        if not between:
            ww = np.ones(n) / n if w is None else w
            return ww @ Y
        return _coef(Y, X, w)

    s_a, s_b = coefs()
    boot = np.empty((nboot, 2))
    bayes = np.empty((nboot, 2))
    for k in range(nboot):
        i = rng.integers(0, n, n)
        boot[k] = Y[i].mean(0) if not between else _coef(Y[i], X[i])
        bayes[k] = coefs(rng.dirichlet(np.ones(n)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ls_boot = boot[:, 0] / boot[:, 1]
        ls_bayes = bayes[:, 0] / bayes[:, 1]
    curve = np.array([[s_a - g * s_b, *np.percentile(boot[:, 0] - g * boot[:, 1], [2.5, 97.5])]
                      for g in grid])
    return LambdaCurve(
        n=n, n_dropped=n_dropped, design="between" if between else "within",
        s_a=float(s_a), s_b=float(s_b),
        lam_star=float(s_a / s_b) if s_b != 0 else np.nan,
        ci=tuple(float(v) for v in np.nanpercentile(ls_boot, [2.5, 97.5])),
        hdi=hdi(ls_bayes),
        p_cross_in_01=float(np.mean((ls_bayes >= 0) & (ls_bayes <= 1))),
        s_a_ci=_pct(boot[:, 0]), s_b_ci=_pct(boot[:, 1]),
        grid=grid, curve=curve, boot=boot, bayes=bayes)


def _significant(lo, hi):
    return bool(lo > 0 or hi < 0)


def verdict(result):
    """Classify a result by its 95% intervals at lambda = 0 and lambda = 1.

    "holds under both"   both exclude zero, same sign
    "reverses"           both exclude zero, opposite signs
    "depends on lambda"  only one excludes zero
    "null under both"    neither does

    result: a LambdaCurve, or a mapping with lam0_lo, lam0_hi, lam1_lo,
    lam1_hi (e.g. a row of a results table). As in code/breadth_summary.py,
    except that the sign is read from the interval rather than the point
    estimate (the same unless an estimate lies outside its own interval).
    """
    if isinstance(result, LambdaCurve):
        (_, lo0, hi0), (_, lo1, hi1) = result.lam0, result.lam1
    else:
        lo0, hi0 = result["lam0_lo"], result["lam0_hi"]
        lo1, hi1 = result["lam1_lo"], result["lam1_hi"]
    sig0, sig1 = _significant(lo0, hi0), _significant(lo1, hi1)
    if sig0 and sig1:
        return VERDICTS[0] if (lo0 > 0) == (lo1 > 0) else VERDICTS[1]
    if sig0 or sig1:
        return VERDICTS[2]
    return VERDICTS[3]
