"""Calibration curve for the coupling estimate under the HBN design.

Simulates two-condition subjects over a grid of true lambda (with a 2 x cf
harmonic and a high-frequency plateau), runs the same estimators and routes
as the real-data analysis, and reports lambda_hat against lambda_true for
each aperiodic window and route, with a monotonicity check. The observed
lambda_hat (results/hbn_controls.csv, quality-controlled sample) is read
through the monotone curves only: the point value through the mean curve,
and a 95% highest-density interval from Monte Carlo draws that combine the
observation's standard error with the Monte Carlo error of the mean curve
(s.d. over repetitions / sqrt(repetitions) at each grid point; each drawn
curve is made monotone by pooling adjacent violators, and values beyond the
grid are extrapolated linearly from the end segments). The mapping is written
to results/hbn_ctrl_calibration.csv.

Usage: python sim_calibration.py [--n 400] [--reps 3]
       python sim_calibration.py --from-csv results/sim_calibration_fixedharm.csv
           reads an existing simulation instead of running one
       [--obs 0.574 --obs-se 0.039] also maps a given observation
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hbn_lambda as HL
from hbn_controls import mask_for, ols, ap_band, log_flanks, WINDOWS
from sim_control_band import synth_segments, FIT

# the two windows carried through to the real analysis
WIN_KEEP = {"censor 6-16": ("censor", [(6, 16)]),
            "flanks 3-6, 26-36": ("flanks", [(3, 6), (26, 36)])}


def build(lam_true, N, rng, harmonic=0.35, plateau=0.25):
    rows = []
    fsel = None
    for i in range(N):
        gain = 1.4 * rng.standard_normal()
        expo_eo = 1.3 + 0.30 * rng.standard_normal()
        expo_ec = expo_eo + (0.25 + 0.20 * rng.random())
        off_eo = 1.0 + gain
        off_ec = off_eo + (0.10 + 0.10 * rng.random())
        cf = 10.0 + 0.8 * rng.standard_normal()
        bw = 1.5 + 0.15 * rng.standard_normal()
        Lref = 10.0 ** 1.0 / 10.0 ** 1.3
        c_eo = 0.8 * Lref ** (1 - lam_true) * 10 ** (0.25 * rng.standard_normal())
        c_ec = c_eo * 2.5
        rec = {}
        for cond, (off, ex, c) in (("eo", (off_eo, expo_eo, c_eo)),
                                   ("ec", (off_ec, expo_ec, c_ec))):
            segs, ff = synth_segments(off, ex, cf, bw, c, lam_true, rng,
                                      harmonic, plateau=plateau)
            if fsel is None:
                fsel = (ff >= FIT[0]) & (ff <= FIT[1])
            rec[cond] = dict(full=segs.mean(0)[fsel],
                             odd=segs[0::2].mean(0)[fsel],
                             even=segs[1::2].mean(0)[fsel])
        f = ff[fsel]
        lf = np.log10(f)
        lo, hi = cf - 2, cf + 2
        for cond in ("eo", "ec"):
            for split in ("full", "odd", "even"):
                P = np.clip(rec[cond][split], 1e-15, None)
                lP = np.log10(P)
                tt = float(np.mean(P[(f >= lo) & (f <= hi)]))
                for name, (kind, spec) in WIN_KEEP.items():
                    keep = mask_for(f, kind, spec)
                    off, ex = ols(lP, lf, keep)
                    bb = ap_band(off, ex, f, lo, hi)
                    rows.append(dict(subject=f"s{i:04d}", cond=cond, split=split,
                                     estimator=name, offset=off, exponent=ex,
                                     b_alpha=bb, tot_alpha=tt, a_alpha=tt - bb,
                                     iaf=cf))
    return pd.DataFrame(rows)


def simulate(grid, n, reps, harmonic, plateau):
    out = []
    for lam_true in grid:
        for rep in range(reps):
            rng = np.random.default_rng(int(5000 + 1000 * lam_true + 13 * rep))
            T = build(lam_true, n, rng, harmonic=harmonic, plateau=plateau)
            for est in WIN_KEEP:
                for sa, sb in (("full", "full"), ("odd", "even")):
                    r = HL.pq_within(T, est, None, sa, sb)
                    if r is None:
                        continue
                    out.append(dict(lambda_true=lam_true, rep=rep, estimator=est,
                                    route=("within" if sa == "full" else "within IV"),
                                    lam_hat=r["p"], se=r["p_se"],
                                    lam_hat_q=r["lam_from_q"], n=r["n"]))
    return pd.DataFrame(out)


def hdi(x, mass=0.95):
    x = np.sort(x[np.isfinite(x)])
    n = int(np.floor(mass * x.size))
    i = int(np.argmin(x[n:] - x[:x.size - n]))
    return x[i], x[i + n]


def pava(y):
    """Closest non-decreasing sequence (pool adjacent violators, equal weights)."""
    vals, sizes = [], []
    for v in y:
        vals.append(float(v))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = sizes[-2] + sizes[-1]
            vals[-2] = (vals[-2] * sizes[-2] + vals[-1] * sizes[-1]) / w
            sizes[-2] = w
            vals.pop()
            sizes.pop()
    return np.repeat(vals, sizes)


def invert(y, curve, gs):
    """Inverse of a non-decreasing piecewise-linear curve, linear beyond the grid."""
    if y < curve[0] or y > curve[-1]:
        i, j = (0, 1) if y < curve[0] else (-2, -1)
        slope = (curve[j] - curve[i]) / (gs[j] - gs[i])
        k = i if y < curve[0] else j
        return gs[k] + (y - curve[k]) / slope if slope > 0 else np.nan
    k = int(np.clip(np.searchsorted(curve, y, side="right") - 1, 0, gs.size - 2))
    d = curve[k + 1] - curve[k]
    return gs[k] + (y - curve[k]) * (gs[k + 1] - gs[k]) / d if d > 0 else gs[k]


def read_back(S, obs, obs_se, ndraw, seed=0):
    """Calibrated lambda for one observation through one simulated curve."""
    gs = np.array(sorted(S.lambda_true.unique()))
    Y = [S[S.lambda_true == g].lam_hat.to_numpy() for g in gs]
    mu = np.array([y.mean() for y in Y])
    sd = np.array([max(y.std(ddof=1), 1e-6) for y in Y])
    mc = sd / np.sqrt([y.size for y in Y])
    # how many SDs the observation sits from each endpoint
    out = dict(obs=obs, obs_se=obs_se, monotone=bool(np.all(np.diff(mu) > 0)),
               z0=(obs - mu[0]) / np.hypot(sd[0], obs_se),
               z1=(obs - mu[-1]) / np.hypot(sd[-1], obs_se))
    if not out["monotone"]:
        return out
    rng = np.random.default_rng(seed)
    draws = np.array([invert(obs + obs_se * rng.standard_normal(),
                             pava(mu + mc * rng.standard_normal(mu.size)), gs)
                      for _ in range(ndraw)])
    lo, hi = hdi(draws)
    out.update(lam_cal=invert(obs, mu, gs), hdi_lo=lo, hdi_hi=hi,
               beyond_grid=float(np.mean((draws < gs[0]) | (draws > gs[-1]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--harmonic", type=float, default=0.35)
    ap.add_argument("--plateau", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    ap.add_argument("--from-csv", default=None,
                    help="read this simulation output instead of simulating")
    ap.add_argument("--obs", type=float, default=None)
    ap.add_argument("--obs-se", type=float, default=None)
    ap.add_argument("--ndraw", type=int, default=20000)
    a = ap.parse_args()
    grid = [float(x) for x in a.grid.split(",")]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if a.from_csv:
        D = pd.read_csv(a.from_csv)
        grid = sorted(D.lambda_true.unique())
        source = os.path.basename(a.from_csv)
        print(f"read {a.from_csv}: {D.rep.nunique()} reps, lambda_true grid {grid}\n")
    else:
        print(f"N = {a.n} per rep, {a.reps} reps, lambda_true grid {grid}\n")
        D = simulate(grid, a.n, a.reps, a.harmonic, a.plateau)
        source = f"sim_calibration{a.tag}.csv"
        p = os.path.join(here, "results", source)
        D.to_csv(p, index=False)
        print(f"wrote {p}\n")

    print("=== calibration curve: lambda_hat (mean +/- SD over reps) ===")
    print(f"{'estimator':22s} {'route':10s} " +
          " ".join(f"{g:>13.2f}" for g in grid))
    for est in WIN_KEEP:
        for route in ("within", "within IV"):
            S = D[(D.estimator == est) & (D.route == route)]
            cells = []
            for g in grid:
                v = S[S.lambda_true == g].lam_hat
                cells.append(f"{v.mean():6.3f}+-{v.std():5.3f}" if len(v) else "     -     ")
            print(f"{est:22s} {route:10s} " + " ".join(cells))

    # ---- read the observed HBN values back through the curve ----
    obs_path = os.path.join(here, "results", "hbn_controls.csv")
    O = pd.read_csv(obs_path) if os.path.exists(obs_path) else None
    rows = []
    print("\n=== calibrated estimate for HBN (monotone curves only) ===")
    for est in WIN_KEEP:
        for route in ("within", "within IV"):
            S = D[(D.estimator == est) & (D.route == route)]
            todo = []
            if O is not None:
                row = O[O.sweep.str.startswith("WINDOW") & (O.route == route)
                        & (O.estimator == est)]
                if not row.empty:
                    todo.append(("hbn_controls.csv", float(row.p.iloc[0]),
                                 float(row.p_se.iloc[0]), int(row.n.iloc[0])))
            if a.obs is not None:
                todo.append(("given", a.obs, a.obs_se, np.nan))
            for src, obs, obs_se, n in todo:
                r = read_back(S, obs, obs_se, a.ndraw)
                rows.append(dict(estimator=est, route=route, observed=src, n_obs=n,
                                 **r, calibration=source))
                head = f"  {est:18s} {route:9s} {src:16s} {obs:6.3f} ({obs_se:.3f})"
                if not r["monotone"]:
                    print(f"{head} -> curve not monotone, not inverted")
                    continue
                print(f"{head} -> lambda {r['lam_cal']:5.2f}, 95% HDI "
                      f"[{r['hdi_lo']:4.2f}, {r['hdi_hi']:4.2f}] (beyond grid "
                      f"{100 * r['beyond_grid']:.0f}%)   z vs lambda=0: {r['z0']:6.2f}   "
                      f"z vs lambda=1: {r['z1']:6.2f}")
    if rows:
        q = os.path.join(here, "results", "hbn_ctrl_calibration.csv")
        pd.DataFrame(rows).to_csv(q, index=False)
        print(f"\nwrote {q}")


if __name__ == "__main__":
    main()
