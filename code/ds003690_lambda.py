"""Coupling exponent from epoch-to-epoch fluctuations within a session.

Data: ds003690_epochs.py output (2-s eyes-open epochs, posterior ROI, three
DPSS tapers per epoch, pupil, EOG and temporal high-frequency power).

Per epoch and taper the aperiodic background is fitted by least squares on
ln power over 2-40 Hz with 6-16 Hz left out, corrected for the log of an
exponentially distributed periodogram (+ Euler's constant); alpha band =
individual alpha frequency +/- 2 Hz (from the participant's mean spectrum).
Band total T comes from one taper, the background B inside a = T - B from a
second and the instrument/weights from a third; all six assignments are
pooled. lambda is estimated with lambda_gmm.estimate_levels (participant-
specific intrinsic strength, no a > 0 needed), without and with the measured
confounders (pupil diameter, VEOG and HEOG power, temporal 60-95 Hz power),
with a participant-cluster bootstrap. For comparison: the within-participant
slope of ln a on ln b over epochs with a > 0, instrumented by the third
taper's ln b.

--simulate L replaces the data by synthetic epochs built from each
participant's own mean background, alpha peak and epoch-to-epoch background
fluctuation, with coupling L and the same three-taper noise, to check that
the analysis recovers L on data like these.

Usage: python ds003690_lambda.py [--data DIR] [--nboot 200] [--simulate L]
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_gmm as G

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIT = (2.0, 40.0)
CENSOR = (6.0, 16.0)
EULER = 0.5772156649


def fit_background(P, f):
    """Vectorised censored log-log fit; P (..., n_f). Returns ln L (..., n_f)."""
    keep = (f >= FIT[0]) & (f <= FIT[1]) & ~((f >= CENSOR[0]) & (f <= CENSOR[1]))
    X = np.column_stack([np.ones(keep.sum()), np.log(f[keep])])
    Y = np.log(np.maximum(P[..., keep], 1e-30)).reshape(-1, keep.sum()).T
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]              # 2 x N
    beta[0] += EULER
    lf = np.log(np.maximum(f, 1e-9))
    lnL = beta[0][:, None] + beta[1][:, None] * lf[None, :]
    return lnL.reshape(P.shape)


def iaf(Pm, f):
    lnL = fit_background(Pm[None, :], f)[0]
    s = (f >= 7) & (f <= 13)
    resid = np.log(Pm[s]) - lnL[s]
    return float(f[s][np.argmax(resid)]) if np.max(resid) > 0.1 else np.nan


def load(data):
    units = []
    for p in sorted(glob.glob(os.path.join(data, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        units.append(dict(subject=str(d["subject"]), group=str(d["group"]), age=float(d["age"]),
                          f=d["freqs"].astype(float), P=d["post"].astype(float),
                          cov=np.column_stack([d["pupil"], d["veog"], d["heog"], d["emg"]])))
    return units


def simulate(units, lam, rng):
    """Synthetic three-taper spectra matched to each participant."""
    out = []
    for u in units:
        f, P = u["f"], u["P"]
        Pm = P.mean(axis=(0, 1))
        cf = iaf(Pm, f)
        if not np.isfinite(cf):
            continue
        lnL = fit_background(P, f)                      # epochs x tapers x f
        mean_lnL = lnL.mean(axis=(0, 1))
        # epoch fluctuation of the background: mean over tapers of the level at cf
        i = np.argmin(np.abs(f - cf))
        lev = lnL[:, :, i].mean(1)
        noise_var = np.var(lnL[:, 0, i] - lnL[:, 1, i]) / 2
        sd = np.sqrt(max(np.var(lev) - noise_var / 3, 1e-4))
        Lcf = np.exp(mean_lnL[i])
        a_rel = max(Pm[i] / Lcf - 1, 0.1)                # peak height relative to background
        c = a_rel * Lcf ** (1 - lam)
        g = np.exp(-0.5 * ((f - cf) / 1.5) ** 2)
        n_ep = P.shape[0]
        eta = rng.normal(0, sd, n_ep)
        eps = rng.normal(0, 0.3, n_ep)                  # intrinsic fluctuation, independent
        L = np.exp(mean_lnL[None, :] + eta[:, None])
        S = L + c * np.exp(eps)[:, None] * g[None, :] * L ** lam
        Pk = S[:, None, :] * rng.exponential(1.0, (n_ep, 3, f.size))
        v = dict(u)
        v["P"] = Pk
        v["cov"] = np.full_like(u["cov"], np.nan)
        out.append(v)
    return out


def band_arrays(units):
    """Per epoch and taper: T (band total), B (fitted background, band mean)."""
    rows = []
    for u in units:
        f, P = u["f"], u["P"]
        cf = iaf(P.mean(axis=(0, 1)), f)
        if not np.isfinite(cf):
            continue
        band = (f >= cf - 2) & (f <= cf + 2)
        L = np.exp(fit_background(P, f))
        T = P[..., band].mean(-1)                        # epochs x tapers
        B = L[..., band].mean(-1)
        rows.append(dict(subject=u["subject"], group=u["group"], T=T, B=B, cov=u["cov"], iaf=cf))
    return rows


COVSETS = {"none": [], "pupil": [0], "pupil + EOG": [0, 1, 2],
           "pupil + EOG + EMG": [0, 1, 2, 3]}


def stack(rows, use_cov):
    """use_cov: list of covariate columns (pupil, VEOG, HEOG, EMG), or empty."""
    T, B, Bz, grp, X = [], [], [], [], []
    for r in rows:
        C = r["cov"][:, use_cov].copy() if use_cov else r["cov"][:, :0]
        if use_cov:
            for k in range(C.shape[1]):
                col = C[:, k]
                m = np.nanmean(col) if np.any(np.isfinite(col)) else 0.0
                col[~np.isfinite(col)] = m
                sd = np.std(col)
                C[:, k] = (col - m) / sd if sd > 0 else 0.0
        for a, b, c in itertools.permutations(range(3)):
            T.append(r["T"][:, a]); B.append(r["B"][:, b]); Bz.append(r["B"][:, c])
            grp.append(np.full(r["T"].shape[0], r["subject"]))
            X.append(C)
    return (np.concatenate(T), np.concatenate(B), np.concatenate(Bz),
            np.concatenate(grp), np.concatenate(X) if use_cov else None)


def instrument_strength(rows, use_cov):
    """Within-participant correlation of the instrument (third taper's ln b,
    residualised on the covariates) with the second taper's ln b."""
    rs = []
    for r in rows:
        zb = np.log(r["B"][:, 2]); xb = np.log(r["B"][:, 1])
        zb, xb = zb - zb.mean(), xb - xb.mean()
        if use_cov:
            C = r["cov"][:, use_cov].copy()
            for k in range(C.shape[1]):
                col = C[:, k]
                m = np.nanmean(col) if np.any(np.isfinite(col)) else 0.0
                col[~np.isfinite(col)] = m
                C[:, k] = col - col.mean()
            zb = zb - C @ np.linalg.lstsq(C, zb, rcond=None)[0]
        if np.std(zb) > 0:
            rs.append(np.corrcoef(zb, xb)[0, 1])
    return float(np.mean(rs))


def iv_log(rows):
    """Within-participant IV slope of ln a (taper A) on ln b (B), instrument ln b (C)."""
    num = den = 0.0
    n = 0
    for r in rows:
        for a, b, c in itertools.permutations(range(3)):
            A = r["T"][:, a] - r["B"][:, b]
            ok = A > 0
            if ok.sum() < 10:
                continue
            y, x, z = np.log(A[ok]), np.log(r["B"][ok, b]), np.log(r["B"][ok, c])
            z = z - z.mean()
            num += np.sum(z * (y - y.mean()))
            den += np.sum(z * (x - x.mean()))
            n += ok.sum()
    return num / den if den else np.nan, n


def run(rows, use_cov, nboot, rng):
    T, B, Bz, grp, X = stack(rows, use_cov)
    est = G.estimate_levels(T, B, Bz, grp, X)
    subs = [r["subject"] for r in rows]
    bs = []
    for _ in range(nboot):
        pick = rng.choice(len(rows), len(rows), replace=True)
        rr = []
        for j, i in enumerate(pick):
            r = dict(rows[i])
            r["subject"] = f"{r['subject']}_{j}"
            rr.append(r)
        T2, B2, Bz2, g2, X2 = stack(rr, use_cov)
        e = G.estimate_levels(T2, B2, Bz2, g2, X2)
        if e["roots"]:
            bs.append(min(e["roots"], key=lambda x: abs(x - est["lam"])) if np.isfinite(est["lam"])
                      else e["lam"])
        else:
            bs.append(np.nan)
    bs = np.array(bs)
    ok = np.isfinite(bs)
    lo, hi = (np.percentile(bs[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan))
    return dict(lam=est["lam"], lo=lo, hi=hi, fail=float(1 - ok.mean()), n_units=len(subs),
                n_epochs=int(T.size / 6), gamma=np.round(est["gamma"], 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DS003690_OUT", "ds003690_epochs"))
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--simulate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    units = load(a.data)
    if a.simulate is not None:
        units = simulate(units, a.simulate, rng)
    rows = band_arrays(units)
    # instrument strength: within-participant correlation of ln B across two tapers
    rel = np.nanmean([np.corrcoef(np.log(r["B"][:, 0]), np.log(r["B"][:, 1]))[0, 1] for r in rows])
    lam_iv, n_iv = iv_log(rows)
    tag = "data" if a.simulate is None else f"simulated lambda = {a.simulate}"
    print(f"{tag}: {len(rows)} participants with an alpha peak; within-participant "
          f"reliability of ln b across tapers r = {rel:.2f}; IV-log (a > 0) {lam_iv:.3f} (n = {n_iv})")
    out = []
    for grp in ("all", "Young", "older"):
        rr = rows if grp == "all" else [r for r in rows if r["group"].lower() == grp.lower()]
        for cname, use_cov in (COVSETS.items() if a.simulate is None else [("none", [])]):
            r = run(rr, use_cov, a.nboot, rng)
            r.update(sample=tag, group=grp, covariates=cname, reliability=rel,
                     instrument_r=instrument_strength(rr, use_cov))
            out.append(r)
            print(f"  {grp:6s} covariates={cname:18s} instrument r {r['instrument_r']:.2f} lambda {r['lam']:+.3f} "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}] fail {r['fail']:.2f} "
                  f"n {r['n_units']}/{r['n_epochs']} gamma {r['gamma']}", flush=True)
    name = "ds003690_lambda.csv" if a.simulate is None else f"ds003690_lambda_sim{a.simulate}.csv"
    pd.DataFrame(out).to_csv(os.path.join(RES, name), index=False)


if __name__ == "__main__":
    main()
