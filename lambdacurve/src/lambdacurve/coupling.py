"""Coupling exponent without logging the periodic power.

Band-power model: a = c b^lambda, with c the intrinsic strength of the
oscillation. For two conditions with an intrinsic change delta common to
all subjects (c2 = c1 e^delta), every subject satisfies

    r_2 - e^delta r_1 = 0,   r_k = a_k / b_k^lambda,

so lambda and delta solve the moment conditions

    sum_i w_i (r_i2 - e^delta r_i1)       = 0
    sum_i w_i z_i (r_i2 - e^delta r_i1)   = 0

with z_i the centred change in ln b. Unlike a regression of the change in
ln a on the change in ln b, nothing here needs a > 0, so subjects with a
negative periodic estimate stay in. Selecting a > 0 keeps the positive
residuals, which scale with the background and look like lambda = 1.

Noise. With the data split in two halves (A, B), the total t comes from half
A, the background inside a = t - b and b^lambda from half B, and the
instrument and weights from half A's background fit, which uses only
frequencies outside the band. The three are then estimated from disjoint
data; b and the instrument from the same half attenuate lambda to about
0.8 of its value. Both assignments of the halves are pooled.

What this does not fix: any intrinsic change that co-varies with the
background change biases the estimate by cov(delta_i, d ln b_i) / var(d ln b),
and peak or harmonic power leaking into the background fit biases it
upwards. Eyes open/closed and between-subject developmental contrasts move
state or skull with the background and do not identify lambda. Simulate
the estimator under your own design before reading lambda-hat against 0 and 1.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.optimize import brentq

GRID = np.linspace(-0.5, 1.5, 81)


@dataclass
class Coupling:
    """lam: estimate (nan if the moment has no root on the grid); roots: all
    roots found (more than one: the estimate is not unique, report them);
    ci, boot_sd, boot_fail: subject (or unit) bootstrap
    percentile interval, s.d., and share of draws without a root;
    instrument_r: correlation between the two independent estimates of the
    background predictor (check it before trusting the estimate: 0.74-0.90
    was healthy in our data; a weak instrument, e.g. 0.17, returns
    plausible-looking values with huge variance); frac_a_nonpositive: share of
    periodic estimates <= 0 that were kept; delta or gamma: intrinsic change
    (two conditions) or confounder coefficients (levels)."""
    lam: float
    roots: list
    n: int
    instrument_r: float
    frac_a_nonpositive: float
    ci: tuple = (np.nan, np.nan)
    boot_sd: float = np.nan
    boot_fail: float = np.nan
    delta: float = np.nan
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self):
        return dict(lam=self.lam, ci_lo=self.ci[0], ci_hi=self.ci[1], boot_sd=self.boot_sd,
                    boot_fail=self.boot_fail, n=self.n, instrument_r=self.instrument_r,
                    frac_a_nonpositive=self.frac_a_nonpositive, delta=self.delta,
                    n_roots=len(self.roots))


# ---- two conditions ------------------------------------------------------

def _stack(t1, t2, b1, b2):
    """Pool the two half assignments into flat arrays."""
    T1, T2, B1, B2, Z, W = [], [], [], [], [], []
    for sa, sb in ((0, 1), (1, 0)):
        ok = (np.all(np.isfinite([t1[:, sa], t2[:, sa], b1[:, sb], b2[:, sb],
                                  b1[:, sa], b2[:, sa]]), 0)
              & (b1[:, sb] > 0) & (b2[:, sb] > 0) & (b1[:, sa] > 0) & (b2[:, sa] > 0))
        T1.append(t1[ok, sa]); T2.append(t2[ok, sa])
        B1.append(b1[ok, sb]); B2.append(b2[ok, sb])
        z = np.log(b2[ok, sa]) - np.log(b1[ok, sa])
        Z.append(z - z.mean())
        W.append(np.sqrt(b1[ok, sa] * b2[ok, sa]))       # scale from half A
    return [np.concatenate(v) for v in (T1, T2, B1, B2, Z, W)]


def _m2(lam, T1, T2, B1, B2, Z, S):
    """Second moment, profiled over delta; weights 1 / S^(1 - lambda)."""
    W = S ** (lam - 1.0)
    R1 = (T1 - B1) / B1 ** lam
    R2 = (T2 - B2) / B2 ** lam
    s1 = np.sum(W * R1)
    if s1 == 0:
        return np.nan, np.nan
    ed = np.sum(W * R2) / s1
    return np.sum(W * Z * (R2 - ed * R1)) / np.sum(W), ed


def _estimate(t1, t2, b1, b2, grid=None, near=None):
    grid = GRID if grid is None else np.asarray(grid, float)
    arrs = _stack(t1, t2, b1, b2)
    f = lambda g: _m2(g, *arrs)[0]
    vals = np.array([f(g) for g in grid])
    roots, down = [], []
    for i in range(grid.size - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            roots.append(brentq(f, grid[i], grid[i + 1]))
            down.append(vals[i] > 0)
    if not roots:
        return dict(lam=np.nan, delta=np.nan, roots=[])
    if near is not None:
        lam = min(roots, key=lambda r: abs(r - near))
    else:
        # the moment crosses downwards at the true value; take the downward
        # root nearest the grid point where the moment is smallest
        cand = [r for r, d in zip(roots, down) if d] or roots
        j = int(np.nanargmin(np.abs(vals)))
        lam = min(cand, key=lambda r: abs(r - grid[j]))
    ed = _m2(lam, *arrs)[1]
    return dict(lam=float(lam), delta=float(np.log(ed)) if ed > 0 else np.nan,
                roots=[float(r) for r in roots])


def _corr(u, v):
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(u[ok], v[ok])[0, 1])


def coupling_two_conditions(t1, t2, b1, b2, nboot=300, seed=0, grid=None):
    """lambda and the intrinsic change delta from split-half band powers.

    t1, t2: total band power in conditions 1 and 2; b1, b2: fitted
    aperiodic band power (from a fit that excludes the band). Each has shape
    (n_subjects, 2), column s holding the estimate from half s of the data
    (e.g. odd and even segments). The profiled moment is scanned over grid
    (default -0.5 to 1.5; outside it the weights become extreme and spurious
    roots appear). The subject bootstrap takes, in each draw, the root
    nearest the full-sample estimate. Returns a Coupling.
    """
    t1, t2, b1, b2 = (np.asarray(v, float) for v in (t1, t2, b1, b2))
    for v in (t1, t2, b1, b2):
        if v.ndim != 2 or v.shape[1] != 2 or v.shape != t1.shape:
            raise ValueError("t1, t2, b1, b2 must all have shape (n_subjects, 2)")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    est = _estimate(t1, t2, b1, b2, grid)
    n = t1.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dA = np.log(b2[:, 0]) - np.log(b1[:, 0])
        dB = np.log(b2[:, 1]) - np.log(b1[:, 1])
    a_used = np.concatenate([t1[:, 0] - b1[:, 1], t2[:, 0] - b2[:, 1],
                             t1[:, 1] - b1[:, 0], t2[:, 1] - b2[:, 0]])
    out = Coupling(lam=est["lam"], roots=est["roots"], n=n, delta=est["delta"],
                   instrument_r=_corr(dA, dB),
                   frac_a_nonpositive=float(np.mean(a_used[np.isfinite(a_used)] <= 0)))
    if nboot:
        near = est["lam"] if np.isfinite(est["lam"]) else None
        bs = np.array([_estimate(t1[i], t2[i], b1[i], b2[i], grid, near=near)["lam"]
                       for i in (rng.integers(0, n, n) for _ in range(nboot))])
        ok = np.isfinite(bs)
        if ok.sum() > 10:
            out.ci = tuple(float(v) for v in np.percentile(bs[ok], [2.5, 97.5]))
            out.boot_sd = float(np.std(bs[ok]))
        out.boot_fail = float(1 - ok.mean())
    return out


# ---- levels within units -------------------------------------------------

def _levels_moments(lam, gamma, T, B, Z, X, W, groups, starts):
    Wl = W ** (lam - 1.0)
    r = (T - B) / B ** lam
    g = np.exp(X @ gamma) if X.shape[1] else np.ones_like(r)
    num = np.add.reduceat(Wl * r, starts)
    den = np.add.reduceat(Wl * g, starts)
    c = (num / np.where(den == 0, np.nan, den))[groups]
    e = Wl * (r - c * g)
    m = [np.sum(e * Z)] + [np.sum(e * X[:, k]) for k in range(X.shape[1])]
    return np.array(m) / np.sum(Wl)


def _levels(T, B, Bz, groups, X=None, grid=None):
    grid = GRID if grid is None else np.asarray(grid, float)
    order = np.argsort(groups, kind="stable")
    T, B, Bz, groups = T[order], B[order], Bz[order], np.asarray(groups)[order]
    X = np.zeros((T.size, 0)) if X is None else np.asarray(X, float)[order]
    _, starts, inv = np.unique(groups, return_index=True, return_inverse=True)

    def centre(v):
        m = np.add.reduceat(v, starts) / np.diff(np.r_[starts, v.size])
        return v - m[inv]

    # x centred within units; instrument = ln Bz centred within units,
    # residualised on x
    Xc = np.column_stack([centre(X[:, k]) for k in range(X.shape[1])]) if X.shape[1] else X
    z = centre(np.log(Bz))
    if X.shape[1]:
        z = z - Xc @ np.linalg.lstsq(Xc, z, rcond=None)[0]
    k = Xc.shape[1]

    def profile(lam):
        if k == 0:
            return _levels_moments(lam, np.zeros(0), T, B, z, Xc, Bz, inv, starts)[0], np.zeros(0)
        sol = optimize.least_squares(
            lambda gm: _levels_moments(lam, gm, T, B, z, Xc, Bz, inv, starts)[1:], np.zeros(k))
        return _levels_moments(lam, sol.x, T, B, z, Xc, Bz, inv, starts)[0], sol.x

    vals = np.array([profile(g)[0] for g in grid])
    roots = [brentq(lambda l: profile(l)[0], grid[i], grid[i + 1])
             for i in range(grid.size - 1)
             if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0]
    if not roots:
        return dict(lam=np.nan, gamma=np.full(k, np.nan), roots=[])
    j = int(np.nanargmin(np.abs(vals)))
    lam = min(roots, key=lambda r: abs(r - grid[j]))
    return dict(lam=float(lam), gamma=profile(lam)[1], roots=[float(r) for r in roots])


def coupling_levels(T, B, Bz, groups, X=None, grid=None, nboot=0, seed=0):
    """Coupling from fluctuations of band power within units (e.g. epochs
    within a participant), without logging the periodic power.

    Model: a_ij = c_i exp(gamma' x_ij) b_ij^lambda u_ij, E[u - 1] = 0, for
    unit i and epoch j, with x measured confounders (arousal, eye or muscle
    proxies), centred within unit. The residual a / b^lambda - c_i exp(gamma' x)
    is uncorrelated with the instrument (ln Bz centred within unit and
    residualised on x) and with each x; c_i is profiled per unit.

    T: total band power from one estimate (e.g. taper or bin set A); B:
    aperiodic band power from an independent estimate, used in a = T - B and
    b^lambda; Bz: aperiodic band power from a third estimate, used for the
    instrument and the weights. groups: unit label per epoch. With nboot > 0
    a unit (cluster) bootstrap gives the interval. Within-session
    fluctuations share state and artefact with the background (in the
    Dortmund data the estimate was 1.3-1.9 eyes closed and -0.1-0.7 eyes
    open), so a value above 1 signals a common drive or leakage.
    """
    T, B, Bz = (np.asarray(v, float).ravel() for v in (T, B, Bz))
    groups = np.asarray(groups)
    if not (T.shape == B.shape == Bz.shape == groups.shape):
        raise ValueError("T, B, Bz and groups must have the same length")
    if not (np.all(np.isfinite(T)) and np.all(B > 0) and np.all(Bz > 0)
            and np.all(np.isfinite(B)) and np.all(np.isfinite(Bz))):
        raise ValueError("T must be finite and B, Bz finite and positive")
    if X is not None:
        X = np.asarray(X, float)
        X = X[:, None] if X.ndim == 1 else X
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    est = _levels(T, B, Bz, groups, X, grid)
    units = np.unique(groups)
    # instrument strength: ln B against ln Bz, both centred within units
    _, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    lB, lZ = np.log(B), np.log(Bz)
    cB = lB - (np.bincount(inv, lB) / cnt)[inv]
    cZ = lZ - (np.bincount(inv, lZ) / cnt)[inv]
    a = T - B
    out = Coupling(lam=est["lam"], roots=est["roots"], n=int(units.size), gamma=est["gamma"],
                   instrument_r=_corr(cB, cZ),
                   frac_a_nonpositive=float(np.mean(a[np.isfinite(a)] <= 0)))
    if nboot:
        idx = {u: np.where(groups == u)[0] for u in units}
        bs = []
        for _ in range(nboot):
            pick = rng.choice(units, units.size, replace=True)
            rows = np.concatenate([idx[u] for u in pick])
            g2 = np.concatenate([np.full(idx[u].size, j) for j, u in enumerate(pick)])
            e = _levels(T[rows], B[rows], Bz[rows], g2, None if X is None else X[rows], grid)
            r = e["roots"]
            bs.append(min(r, key=lambda v: abs(v - est["lam"]))
                      if r and np.isfinite(est["lam"]) else e["lam"])
        bs = np.array(bs, float)
        ok = np.isfinite(bs)
        if ok.sum() > 10:
            out.ci = tuple(float(v) for v in np.percentile(bs[ok], [2.5, 97.5]))
            out.boot_sd = float(np.std(bs[ok]))
        out.boot_fail = float(1 - ok.mean())
    return out
