"""Does the log-free within-subject estimator recover lambda, and does it stay
silent in a band without a rhythm?

Simulates two-condition subjects (eyes-open/eyes-closed-like background
change that varies across subjects, a common intrinsic alpha change, an
optional harmonic and high-frequency plateau) over a grid of true lambda.
For the alpha band (IAF +/- 2 Hz) and a peak-free control band (30-38 Hz) it
compares the split-half instrumental-variable slope of the change in ln a on
the change in ln b (subjects with a > 0 only; the estimator used so far) with
lambda_gmm.estimate (no a > 0 needed).

Usage: python sim_lambda_gmm.py [--n 400] [--reps 3] [--harmonic 0.35]
       [--plateau 0.25] [--out results/sim_lambda_gmm.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_gmm
from hbn_controls import mask_for, ols, ap_band, log_flanks
from sim_control_band import synth_segments, FIT, CONTROL_BAND


def simulate(lam_true, n, rng, harmonic, plateau, delta_sd=0.0, delta_corr=0.0):
    """Band powers per subject: dict band -> arrays (n, cond, split) of a and b.

    The intrinsic condition-2 change is ln 2.5 plus a subject-specific part
    with s.d. delta_sd, correlated delta_corr with the subject's change in
    exponent (a state confound when nonzero).
    """
    out = {k: dict(a=np.full((n, 2, 2), np.nan), b=np.full((n, 2, 2), np.nan),
                   bc=np.full((n, 2, 2), np.nan))
           for k in ("alpha", "control")}
    for i in range(n):
        gain = 1.4 * rng.standard_normal()
        expo_eo = 1.3 + 0.30 * rng.standard_normal()
        u = rng.random()
        expo_ec = expo_eo + (0.25 + 0.20 * u)
        off_eo = 1.0 + gain
        off_ec = off_eo + (0.10 + 0.10 * rng.random())
        cf = 10.0 + 0.8 * rng.standard_normal()
        bw = 1.5 + 0.15 * rng.standard_normal()
        Lref = 10.0 ** 1.0 / 10.0 ** 1.3
        c_eo = 0.8 * Lref ** (1 - lam_true) * 10 ** (0.25 * rng.standard_normal())
        zu = (u - 0.5) / np.sqrt(1 / 12)                   # standardised exponent change
        d_i = delta_sd * (delta_corr * zu + np.sqrt(1 - delta_corr ** 2) * rng.standard_normal())
        c_ec = c_eo * 2.5 * np.exp(d_i)
        bands = {"alpha": (cf - 2, cf + 2), "control": CONTROL_BAND}
        for k, (off, ex, c) in enumerate(((off_eo, expo_eo, c_eo),
                                          (off_ec, expo_ec, c_ec))):
            segs, ff = synth_segments(off, ex, cf, bw, c, lam_true, rng,
                                      harmonic, plateau)
            sel = (ff >= FIT[0]) & (ff <= FIT[1])
            f = ff[sel]
            lf = np.log10(f)
            for s, part in enumerate((segs[0::2], segs[1::2])):
                P = np.clip(part.mean(0)[sel], 1e-15, None)
                lP = np.log10(P)
                for name, (lo, hi) in bands.items():
                    if name == "alpha":
                        keep = mask_for(f, "censor", [(6, 16)])
                    else:
                        keep = mask_for(f, "flanks", log_flanks(lo, hi))
                    o, e = ols(lP, lf, keep)
                    b = ap_band(o, e, f, lo, hi)
                    t = float(np.mean(P[(f >= lo) & (f <= hi)]))
                    out[name]["a"][i, k, s] = t - b
                    out[name]["b"][i, k, s] = b
                    # background at the band's geometric centre (the peak)
                    out[name]["bc"][i, k, s] = 10.0 ** (o - e * np.log10(np.sqrt(lo * hi)))
    return out


def iv_log(a, b):
    """Split-half IV slope of d ln a on d ln b, subjects with all a > 0."""
    ok = np.all(a > 0, axis=(1, 2)) & np.all(b > 0, axis=(1, 2))
    if ok.sum() < 20:
        return np.nan, int(ok.sum())
    da = np.log(a[ok, 1, :]) - np.log(a[ok, 0, :])      # (n, split)
    db = np.log(b[ok, 1, :]) - np.log(b[ok, 0, :])
    cov = lambda x, y: np.mean((x - x.mean()) * (y - y.mean()))
    num = 0.5 * (cov(db[:, 0], da[:, 1]) + cov(db[:, 1], da[:, 0]))
    return num / cov(db[:, 0], db[:, 1]), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--grid", default="0,0.5,1")
    ap.add_argument("--harmonic", type=float, default=0.35)
    ap.add_argument("--plateau", type=float, default=0.25)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--bdef", default="band", choices=["band", "center"],
                    help="background regressor: band mean, or value at the band centre")
    ap.add_argument("--delta-sd", type=float, default=0.0,
                    help="s.d. of the subject-specific intrinsic change (ln units)")
    ap.add_argument("--delta-corr", type=float, default=0.0,
                    help="its correlation with the change in exponent")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for lam_true in [float(g) for g in a.grid.split(",")]:
        for rep in range(a.reps):
            rng = np.random.default_rng(int(9000 + 1000 * lam_true + 17 * rep))
            S = simulate(lam_true, a.n, rng, a.harmonic, a.plateau,
                         a.delta_sd, a.delta_corr)
            for band, d in S.items():
                A, B = d["a"], (d["b"] if a.bdef == "band" else d["bc"])
                lam_iv, n_iv = iv_log(A, B)
                T = d["a"] + d["b"]                  # total band power per half
                g = lambda_gmm.bootstrap(T[:, 0, :], T[:, 1, :],
                                         d["b"][:, 0, :], d["b"][:, 1, :],
                                         nboot=a.nboot, rng=np.random.default_rng(rep))
                rows.append(dict(lambda_true=lam_true, rep=rep, band=band,
                                 harmonic=a.harmonic, plateau=a.plateau,
                                 delta_sd=a.delta_sd, delta_corr=a.delta_corr,
                                 iv_log=lam_iv, n_iv=n_iv, gmm=g["lam"],
                                 gmm_lo=g["ci"][0], gmm_hi=g["ci"][1],
                                 gmm_fail=g["boot_fail"], n=a.n,
                                 frac_neg=float(np.mean(A <= 0))))
                print(f"lambda_true {lam_true:.2f} rep {rep} {band:8s} "
                      f"IV(log, a>0) {lam_iv:+.3f} (n {n_iv:3d})   "
                      f"GMM {g['lam']:+.3f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}] "
                      f"fail {g['boot_fail']:.2f}", flush=True)
    D = pd.DataFrame(rows)
    print("\nmean over reps:")
    print(D.groupby(["band", "lambda_true"])[["iv_log", "gmm", "gmm_lo", "gmm_hi",
                                              "frac_neg"]].mean().round(3).to_string())
    if a.out:
        D.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
