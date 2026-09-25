"""Calibration curve for the joint aperiodic + peak fit (joint_fit.py).

Same generative model as sim_calibration_kp.py. Reports lambda_hat against
lambda_true and the split-half reliability of the background change, which
determines whether the instrumented estimate is usable.

Usage: python sim_joint_calibration.py [--n 250] [--reps 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_fit import fit_joint, band_ab, ALPHA_GUESS, HARMONIC_GUESS
from sim_calibration_kp import synth, FIT, CONTROL_BAND


def cov(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def build(lam_true, N, rng, harmonic=0.35, plat_rel=0.3):
    rows = []
    fsel = None
    for i in range(N):
        gain = 1.4 * rng.standard_normal()
        chi_eo = 3.4 + 0.5 * rng.standard_normal()
        chi_ec = chi_eo + (0.25 + 0.20 * rng.random())
        fk = float(np.clip(7.0 + 1.5 * rng.standard_normal(), 3.0, 14.0))
        b_eo = 3.0 + gain
        b_ec = b_eo + (0.10 + 0.10 * rng.random())
        cf = 10.0 + 0.8 * rng.standard_normal()
        bw = 1.5 + 0.15 * rng.standard_normal()
        Lref = 10.0 ** 3.0 / (7.0 ** 3.4 + 10.0 ** 3.4)
        c_eo = 0.8 * Lref ** (1 - lam_true) * 10 ** (0.25 * rng.standard_normal())
        c_ec = c_eo * 2.5
        rec = {}
        for cond, (bb, cc, aa) in (("eo", (b_eo, chi_eo, c_eo)),
                                   ("ec", (b_ec, chi_ec, c_ec))):
            segs, ff = synth(bb, cc, fk, plat_rel, cf, bw, aa, lam_true, rng,
                             harmonic)
            if fsel is None:
                fsel = (ff >= FIT[0]) & (ff <= FIT[1])
            rec[cond] = dict(full=segs.mean(0)[fsel],
                             odd=segs[0::2].mean(0)[fsel],
                             even=segs[1::2].mean(0)[fsel])
        f = ff[fsel]
        for cond in ("eo", "ec"):
            for split in ("full", "odd", "even"):
                P = np.clip(rec[cond][split], 1e-15, None)
                fit = fit_joint(P, f, [ALPHA_GUESS, HARMONIC_GUESS])
                if fit is None:
                    continue
                a, b = band_ab(fit, f, cf - 2, cf + 2, peak_idx=0)
                ac, bc = band_ab(fit, f, *CONTROL_BAND, peak_idx=0)
                rows.append(dict(subject=f"s{i:04d}", cond=cond, split=split,
                                 a=a, b=b, a_ctrl=ac, b_ctrl=bc,
                                 exponent=fit["exponent"],
                                 knee_freq=fit["knee_freq"]))
    return pd.DataFrame(rows)


def lam_from(T, acol="a", bcol="b"):
    piv = {s: T[T.split == s].pivot_table(index="subject", columns="cond",
                                          values=[acol, bcol])
           for s in ("full", "odd", "even")}
    idx = piv["full"].index
    for s in ("odd", "even"):
        idx = idx.intersection(piv[s].index)
    if len(idx) < 20:
        return np.nan, np.nan, 0
    ok = np.ones(len(idx), bool)
    for s in ("full", "odd", "even"):
        g = piv[s].loc[idx]
        for key in (acol, bcol):
            for c in ("ec", "eo"):
                v = g[(key, c)].to_numpy()
                ok &= np.isfinite(v) & (v > 0)
    if ok.sum() < 20:
        return np.nan, np.nan, int(ok.sum())

    def delta(split, key):
        g = piv[split].loc[idx]
        return (np.log(g[(key, "ec")].to_numpy()) -
                np.log(g[(key, "eo")].to_numpy()))[ok]

    da_f, db_f = delta("full", acol), delta("full", bcol)
    lam_ols = cov(db_f, da_f) / cov(db_f, db_f)
    da_o, db_o = delta("odd", acol), delta("odd", bcol)
    da_e, db_e = delta("even", acol), delta("even", bcol)
    den = cov(db_o, db_e)
    lam_iv = (0.5 * (cov(db_o, da_e) + cov(db_e, da_o)) / den
              if den != 0 else np.nan)
    return lam_ols, lam_iv, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1.0")
    a = ap.parse_args()
    grid = [float(x) for x in a.grid.split(",")]
    print(f"joint fit calibration: N = {a.n} per rep, {a.reps} reps, "
          f"grid {grid}\n", flush=True)
    out = []
    for lam_true in grid:
        for rep in range(a.reps):
            rng = np.random.default_rng(int(4400 + 1000 * lam_true + 19 * rep))
            T = build(lam_true, a.n, rng)
            lo, li, n = lam_from(T, "a", "b")
            lo_c, li_c, n_c = lam_from(T, "a_ctrl", "b_ctrl")
            out.append(dict(lambda_true=lam_true, rep=rep, lam_ols=lo,
                            lam_iv=li, n=n, lam_iv_ctrl=li_c, n_ctrl=n_c))
            print(f"  lambda_true {lam_true:.2f} rep {rep}: "
                  f"lam_OLS {lo:6.3f}  lam_IV {li:6.3f}  n {n}  "
                  f"(control band lam_IV {li_c:6.3f})", flush=True)
    D = pd.DataFrame(out)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "results", "sim_joint_calibration.csv")
    D.to_csv(p, index=False)
    print("\n=== joint-fit calibration curve ===")
    print(f"{'stat':10s} " + " ".join(f"{g:>13.2f}" for g in grid))
    for col in ("lam_ols", "lam_iv", "lam_iv_ctrl"):
        cells = []
        for g in grid:
            v = D[D.lambda_true == g][col].dropna()
            cells.append(f"{v.mean():6.3f}+-{v.std():5.3f}" if len(v) else "     -     ")
        print(f"{col:10s} " + " ".join(cells))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
