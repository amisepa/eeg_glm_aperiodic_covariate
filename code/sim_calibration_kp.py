"""Calibration curve with a knee+plateau estimator.

As sim_calibration.py, but the aperiodic background is fitted with the same
knee+plateau model used to generate it, and coupling is defined on the neural
part of the background (plateau excluded).

Usage: python sim_calibration_kp.py [--n 400] [--reps 4]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import fit_aperiodic, ap_eval
from hbn_kp import WINDOWS, mask_for, lambda_delta

SRATE = 250.0
DUR = 100.0
WIN_SEC = 4.0
FIT = (2.0, 55.0)
CONTROL_BAND = (30.0, 38.0)


def welch_of(S, ff_full, n, rng):
    sigma = np.sqrt(np.maximum(S, 0) * SRATE * n / 2.0)
    X = np.zeros(n, dtype=complex)
    nh = n // 2
    X[1:nh + 1] = sigma[1:nh + 1] * (rng.standard_normal(nh) +
                                     1j * rng.standard_normal(nh)) / np.sqrt(2)
    if n % 2 == 0:
        X[nh] = sigma[nh] * rng.standard_normal()
        X[nh + 1:] = np.conj(X[nh - 1:0:-1])
    else:
        X[nh + 1:] = np.conj(X[nh:0:-1])
    x = np.real(np.fft.ifft(X))
    nper = int(WIN_SEC * SRATE)
    step = nper // 2
    nseg = 1 + (x.size - nper) // step
    win = np.hanning(nper + 1)[:nper]
    scale = 1.0 / (SRATE * np.sum(win ** 2))
    ff = np.fft.rfftfreq(nper, 1 / SRATE)
    acc = np.zeros((nseg, ff.size))
    for s in range(nseg):
        seg = x[s * step: s * step + nper]
        seg = seg - seg.mean()
        Xs = np.fft.rfft(seg * win)
        pw = np.abs(Xs) ** 2 * scale * 2.0
        pw[0] /= 2
        if nper % 2 == 0:
            pw[-1] /= 2
        acc[s] = pw
    return acc, ff


def synth(b, chi, fknee, plat_rel, cf, bw, amp, lam, rng, harmonic=0.35):
    n = int(SRATE * DUR)
    fg = np.fft.rfftfreq(n, 1 / SRATE)
    k = fknee ** chi
    Ln = 10.0 ** b / (k + np.maximum(fg, 1e-9) ** chi)
    plateau = plat_rel * (10.0 ** b / (k + 40.0 ** chi))
    G = np.exp(-0.5 * ((fg - cf) / bw) ** 2)
    G2 = np.exp(-0.5 * ((fg - 2 * cf) / (1.6 * bw)) ** 2)
    # The harmonic must be specified by its RELATIVE height at its own
    # frequency, not by sharing the fundamental's absolute amplitude. Writing
    # it as `harmonic * amp * G2 * Ln**lam` makes its relative height depend
    # on lambda: with these parameters it is 0.28x the local background at
    # lambda = 1 but 2.34x at lambda = 0, because the background at 20 Hz is
    # 8.4x lower than at 10 Hz. A harmonic twice the size of its background is
    # not a feature of real spectra, and it wrecks the aperiodic fit in
    # exactly the arm that anchors the calibration curve.
    L_cf = 10.0 ** b / (k + cf ** chi)
    fund_rel = amp * L_cf ** lam / L_cf          # relative height at cf
    S = (Ln
         + amp * G * Ln ** lam                    # fundamental, coupled
         + harmonic * fund_rel * G2 * Ln          # harmonic, relative bump
         + plateau)
    S[0] = 0.0
    return welch_of(S, fg, n, rng)


def build(lam_true, N, rng, harmonic=0.35, plat_rel=0.3):
    rows = []
    fsel = None
    for i in range(N):
        gain = 1.4 * rng.standard_normal()
        chi_eo = 3.4 + 0.5 * rng.standard_normal()
        chi_ec = chi_eo + (0.25 + 0.20 * rng.random())
        fk = 7.0 + 1.5 * rng.standard_normal()
        fk = float(np.clip(fk, 3.0, 14.0))
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
        bands = {"alpha": (cf - 2, cf + 2), "control": CONTROL_BAND}
        for cond in ("eo", "ec"):
            for split in ("full", "odd", "even"):
                P = np.clip(rec[cond][split], 1e-15, None)
                for wname, (kind, spec) in WINDOWS.items():
                    keep = mask_for(f, kind, spec)
                    fit = fit_aperiodic(P, f, keep, "knee_plateau")
                    if fit is None:
                        continue
                    th = fit["theta"]
                    for bname, (lo, hi) in bands.items():
                        m = (f >= lo) & (f <= hi)
                        tot_fit = float(np.mean(ap_eval(th, f[m], "knee_plateau")))
                        # neural aperiodic only: drop the additive plateau
                        neural = float(np.mean(
                            10.0 ** th[0] / (10.0 ** th[2] + f[m] ** th[1])))
                        rows.append(dict(
                            subject=f"s{i:04d}", cond=cond, split=split,
                            window=wname, model="knee_plateau", band=bname,
                            a=float(np.mean(P[m])) - tot_fit, b=neural,
                            tot=float(np.mean(P[m])), iaf=cf,
                            exponent=fit["exponent"], knee_freq=fit["knee_freq"],
                            plateau=fit["plateau"], dev=fit["dev"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--harmonic", type=float, default=0.35)
    ap.add_argument("--plateau", type=float, default=0.3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    grid = [float(x) for x in a.grid.split(",")]
    print(f"N = {a.n} per rep, {a.reps} reps, grid {grid}, "
          f"harmonic {a.harmonic}, plateau_rel {a.plateau}\n", flush=True)

    out = []
    for lam_true in grid:
        for rep in range(a.reps):
            rng = np.random.default_rng(int(9000 + 1000 * lam_true + 17 * rep))
            T = build(lam_true, a.n, rng, a.harmonic, a.plateau)
            for band in ("alpha", "control"):
                for wname in WINDOWS:
                    r = lambda_delta(T, band, "knee_plateau", wname, nboot=120,
                                     rng=rng)
                    if r is None:
                        continue
                    out.append(dict(lambda_true=lam_true, rep=rep, band=band,
                                    window=wname, lam_ols=r.get("lam_ols"),
                                    lam_iv=r.get("lam_iv"), n=r.get("n"),
                                    frac_kept=r.get("frac_kept")))
            print(f"  lambda_true {lam_true}, rep {rep} done", flush=True)
    D = pd.DataFrame(out)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "results", f"sim_calibration_kp{a.tag}.csv")
    D.to_csv(p, index=False)

    print("\n=== calibration curve, knee+plateau (lam_IV, mean +/- SD) ===")
    print(f"{'band':9s} {'window':20s} " + " ".join(f"{g:>13.2f}" for g in grid))
    for band in ("alpha", "control"):
        for wname in WINDOWS:
            S = D[(D.band == band) & (D.window == wname)]
            cells = []
            for g in grid:
                v = S[S.lambda_true == g].lam_iv.dropna()
                cells.append(f"{v.mean():6.3f}+-{v.std():5.3f}" if len(v)
                             else "     -     ")
            print(f"{band:9s} {wname:20s} " + " ".join(cells))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
