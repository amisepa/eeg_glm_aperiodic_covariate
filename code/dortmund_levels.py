"""Coupling exponent from segment-to-segment fluctuations within a recording
(Dortmund Vital Study), before and after a 2-hour task battery.

Input: dortmund_extract_psd.py output (per-segment posterior ROI spectra,
4-s Hann, 0.25 Hz; kept segments only). Within each segment three nearly
independent estimates come from disjoint frequency bins: the band total T
from the bins of the alpha band (individual alpha frequency +/- 2 Hz, from
the recording's mean spectrum); the background B inside a = T - B from a
censored log-log fit (2-45 Hz, 6-16 Hz left out, + Euler's constant for the
log of an exponential periodogram) on every fourth bin; the instrument and
weights from the same fit on the interleaved bins half-way between. Both
assignments of the two bin sets are pooled. lambda is estimated with
lambda_gmm.estimate_levels (participant-specific intrinsic strength), with a
participant-cluster bootstrap, for the eyes-closed recordings before and
after the task battery (session 1): a stable property should give the same
lambda-hat in both, although arousal differs.

--simulate L builds synthetic segments from each participant's own mean
background, alpha peak and segment-to-segment background fluctuation, with
coupling L, exponential noise per bin and the same analysis.

Usage: python dortmund_levels.py [--data DIR] [--nboot 200] [--simulate L]
       [--cond ec_pre] [--max-subjects N]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_gmm as G

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIT = (2.0, 45.0)
CENSOR = (6.0, 16.0)
EULER = 0.5772156649


def fit_sets(f):
    """Two interleaved sets of fit bins (every 4th bin, offset by 2)."""
    idx = np.where((f >= FIT[0]) & (f <= FIT[1]) & ~((f >= CENSOR[0]) & (f <= CENSOR[1])))[0]
    return idx[idx % 4 == 0], idx[idx % 4 == 2]


def lnL(P, f, bins):
    X = np.column_stack([np.ones(bins.size), np.log(f[bins])])
    beta = np.linalg.lstsq(X, np.log(np.maximum(P[:, bins], 1e-30)).T, rcond=None)[0]
    beta[0] += EULER
    return beta[0][:, None] + beta[1][:, None] * np.log(np.maximum(f, 1e-9))[None, :]


def iaf(Pm, f):
    b = np.where((f >= FIT[0]) & (f <= FIT[1]) & ~((f >= CENSOR[0]) & (f <= CENSOR[1])))[0]
    X = np.column_stack([np.ones(b.size), np.log(f[b])])
    beta = np.linalg.lstsq(X, np.log(Pm[b]), rcond=None)[0]
    s = (f >= 7) & (f <= 13)
    resid = np.log(Pm[s]) - (beta[0] + beta[1] * np.log(f[s]))
    return float(f[s][np.argmax(resid)]) if np.max(resid) > 0.1 else np.nan


def load(data, cond, max_subjects):
    units = []
    for p in sorted(glob.glob(os.path.join(data, "*_ses-1.npz"))):
        d = np.load(p, allow_pickle=True)
        if f"{cond}_seg_post" not in d:
            continue
        keep = d[f"{cond}_keep"].astype(bool)
        if keep.sum() < 20:
            continue
        units.append(dict(subject=str(d["subject"]), age=float(d["age"]),
                          f=d["freqs"].astype(float), P=d[f"{cond}_seg_post"][keep].astype(float),
                          fp=d[f"{cond}_seg_fp"][keep].astype(float),
                          temp=d[f"{cond}_seg_temp"][keep].astype(float)))
        if max_subjects and len(units) >= max_subjects:
            break
    return units


def simulate(units, lam, rng):
    out = []
    for u in units:
        f, P = u["f"], u["P"]
        cf = iaf(P.mean(0), f)
        if not np.isfinite(cf):
            continue
        s0, s2 = fit_sets(f)
        L0 = lnL(P, f, s0)
        i = np.argmin(np.abs(f - cf))
        lev = L0[:, i]
        L2 = lnL(P, f, s2)[:, i]
        noise_var = np.var(lev - L2) / 2
        sd = np.sqrt(max(np.var(lev) - noise_var, 1e-4))
        mean_lnL = L0.mean(0)
        Lcf = np.exp(mean_lnL[i])
        a_rel = max(P.mean(0)[i] / Lcf - 1, 0.1)
        c = a_rel * Lcf ** (1 - lam)
        g = np.exp(-0.5 * ((f - cf) / 1.5) ** 2)
        n = P.shape[0]
        eta = rng.normal(0, sd, n)
        eps = rng.normal(0, 0.3, n)
        L = np.exp(mean_lnL[None, :] + eta[:, None])
        S = L + c * np.exp(eps)[:, None] * g[None, :] * L ** lam
        v = dict(u)
        v["P"] = S * rng.exponential(1.0, S.shape)
        out.append(v)
    return out


def arrays(units, use_cov):
    T, B, Bz, grp, X = [], [], [], [], []
    for u in units:
        f, P = u["f"], u["P"]
        cf = iaf(P.mean(0), f)
        if not np.isfinite(cf):
            continue
        band = (f >= cf - 2) & (f <= cf + 2)
        s0, s2 = fit_sets(f)
        Lb = [np.exp(lnL(P, f, s))[:, band].mean(1) for s in (s0, s2)]
        t = P[:, band].mean(1)
        C = None
        if use_cov:
            C = np.column_stack([np.log(u["fp"][:, (f >= 1) & (f <= 4)].mean(1)),
                                 np.log(u["temp"][:, (f >= 60) & (f <= 95)].mean(1))])
            C = (C - C.mean(0)) / np.where(C.std(0) > 0, C.std(0), 1)
        for bi, zi in ((0, 1), (1, 0)):
            T.append(t); B.append(Lb[bi]); Bz.append(Lb[zi])
            grp.append(np.full(t.size, u["subject"]))
            if use_cov:
                X.append(C)
    return (np.concatenate(T), np.concatenate(B), np.concatenate(Bz), np.concatenate(grp),
            np.concatenate(X) if use_cov else None)


GRID = np.linspace(-1.0, 3.0, 161)


def run(units, use_cov, nboot, rng):
    T, B, Bz, grp, X = arrays(units, use_cov)
    est = G.estimate_levels(T, B, Bz, grp, X, grid=GRID)
    subs = np.unique(grp)
    idx = {s: np.where(grp == s)[0] for s in subs}
    bs = []
    for _ in range(nboot):
        pick = rng.choice(subs, subs.size, replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        g2 = np.concatenate([np.full(idx[s].size, f"{s}_{j}") for j, s in enumerate(pick)])
        e = G.estimate_levels(T[rows], B[rows], Bz[rows], g2, None if X is None else X[rows],
                              grid=GRID)
        r = e["roots"]
        bs.append(min(r, key=lambda x: abs(x - est["lam"])) if r and np.isfinite(est["lam"])
                  else e["lam"])
    bs = np.array(bs, float)
    ok = np.isfinite(bs)
    lo, hi = np.percentile(bs[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan)
    rel = np.mean([np.corrcoef(np.log(B[grp == s][: idx[s].size // 2]),
                               np.log(Bz[grp == s][: idx[s].size // 2]))[0, 1] for s in subs[:200]])
    return dict(lam=est["lam"], lo=lo, hi=hi, fail=float(1 - ok.mean()), n_units=int(subs.size),
                n_segments=int(T.size // 2), reliability=float(rel),
                gamma=np.round(est["gamma"], 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DORTMUND_OUT", "dortmund_psd"))
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--simulate", type=float, default=None)
    ap.add_argument("--cond", default="ec_pre,ec_post")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--censor-hi", type=float, default=16.0,
                    help="upper edge of the censored band in the background fit")
    ap.add_argument("--no-cov", action="store_true")
    a = ap.parse_args()
    global CENSOR
    CENSOR = (6.0, a.censor_hi)
    rng = np.random.default_rng(a.seed)
    out = []
    for cond in a.cond.split(","):
        units = load(a.data, cond, a.max_subjects)
        if a.simulate is not None:
            units = simulate(units, a.simulate, rng)
        for use_cov in ((False,) if (a.simulate is not None or a.no_cov) else (False, True)):
            r = run(units, use_cov, a.nboot, rng)
            r.update(cond=cond, covariates=use_cov, censor_hi=a.censor_hi,
                     sample="data" if a.simulate is None else f"simulated {a.simulate}")
            out.append(r)
            print(f"{r['sample']} {cond} covariates={use_cov}: lambda {r['lam']:+.3f} "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}] fail {r['fail']:.2f} "
                  f"n {r['n_units']}/{r['n_segments']} bin-set reliability {r['reliability']:.2f} "
                  f"gamma {r['gamma']}", flush=True)
    tag = f"_c{a.censor_hi:.0f}"
    name = (f"dortmund_levels{tag}.csv" if a.simulate is None
            else f"dortmund_levels_sim{a.simulate}{tag}.csv")
    pd.DataFrame(out).to_csv(os.path.join(RES, name), index=False)


if __name__ == "__main__":
    main()
