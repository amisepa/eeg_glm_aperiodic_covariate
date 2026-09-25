"""Calibration curve for the coupling estimate under the HBN design.

Simulates two-condition subjects over a grid of true lambda (with a 2 x cf
harmonic and a high-frequency plateau), runs the same estimators and routes
as the real-data analysis, and reports lambda_hat against lambda_true for
each aperiodic window and route, with a monotonicity check. The observed
lambda_hat is read through the monotone curves only.

Usage: python sim_calibration.py [--n 400] [--reps 3]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--harmonic", type=float, default=0.35)
    ap.add_argument("--plateau", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    grid = [float(x) for x in a.grid.split(",")]

    print(f"N = {a.n} per rep, {a.reps} reps, lambda_true grid {grid}\n")
    out = []
    for lam_true in grid:
        for rep in range(a.reps):
            rng = np.random.default_rng(int(5000 + 1000 * lam_true + 13 * rep))
            T = build(lam_true, a.n, rng, harmonic=a.harmonic,
                      plateau=a.plateau)
            for est in WIN_KEEP:
                for sa, sb in (("full", "full"), ("odd", "even")):
                    r = HL.pq_within(T, est, None, sa, sb)
                    if r is None:
                        continue
                    out.append(dict(lambda_true=lam_true, rep=rep, estimator=est,
                                    route=("within" if sa == "full" else "within IV"),
                                    lam_hat=r["p"], se=r["p_se"],
                                    lam_hat_q=r["lam_from_q"], n=r["n"]))
    D = pd.DataFrame(out)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "results", f"sim_calibration{a.tag}.csv")
    D.to_csv(p, index=False)

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
    if os.path.exists(obs_path):
        O = pd.read_csv(obs_path)
        print("\n=== calibrated estimate for HBN ===")
        for est in WIN_KEEP:
            for route in ("within", "within IV"):
                row = O[(O.sweep.str.startswith("WINDOW")) &
                        (O.route == route)]
                row = row[row.get("estimator", pd.Series(dtype=str)).eq(est)] \
                    if "estimator" in O.columns else row
                if row.empty:
                    continue
                obs, obs_se = float(row.p.iloc[0]), float(row.p_se.iloc[0])
                S = D[(D.estimator == est) & (D.route == route)]
                gs = np.array(sorted(S.lambda_true.unique()))
                mu = np.array([S[S.lambda_true == g].lam_hat.mean() for g in gs])
                sd = np.array([max(S[S.lambda_true == g].lam_hat.std(), 1e-6)
                               for g in gs])
                # invert the monotone curve
                lam_cal = float(np.interp(obs, mu, gs))
                lo = float(np.interp(obs - 1.96 * obs_se, mu, gs))
                hi = float(np.interp(obs + 1.96 * obs_se, mu, gs))
                # how many SDs the observation sits from each endpoint
                z0 = (obs - mu[0]) / np.hypot(sd[0], obs_se)
                z1 = (obs - mu[-1]) / np.hypot(sd[-1], obs_se)
                print(f"  {est:22s} {route:10s} observed {obs:6.3f} -> "
                      f"lambda_calibrated {lam_cal:5.2f} [{lo:4.2f}, {hi:4.2f}]   "
                      f"z vs lambda=0: {z0:6.2f}   z vs lambda=1: {z1:6.2f}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
