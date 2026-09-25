"""Coupling estimate in a peak-free control band, in simulation.

Simulates spectra with known lambda and estimates coupling both in the alpha
band and in a band with no oscillation (30-38 Hz). Anything returned in the
control band is produced by the procedure: the truncation a > 0 of a residual
periodic estimate, or a misspecified aperiodic model.

Usage: python sim_control_band.py [--n 300]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hbn_lambda as HL
from hbn_controls import log_flanks, mask_for, ols, ap_band

SRATE = 250.0
DUR = 100.0
WIN_SEC = 4.0
FIT = (2.0, 40.0)
CONTROL_BAND = (30.0, 38.0)


def synth_segments(offset, exponent, cf, bw, amp, lam, rng, harmonic=0.0,
                   plateau=0.0):
    n = int(SRATE * DUR)
    fg = np.fft.rfftfreq(n, 1 / SRATE)
    L = 10.0 ** offset / np.maximum(fg, 1e-9) ** exponent
    if plateau > 0:
        L = L + plateau * 10.0 ** offset / 40.0 ** exponent
    G = np.exp(-0.5 * ((fg - cf) / bw) ** 2)
    S = L + amp * G * L ** lam
    if harmonic > 0:
        # Specify the harmonic by its RELATIVE height at its own frequency.
        # Sharing the fundamental's absolute amplitude makes the harmonic's
        # relative height depend on lambda -- 0.28x the local background at
        # lambda = 1 but 2.34x at lambda = 0 -- which is not a feature of real
        # spectra and corrupts the aperiodic fit in the arm that anchors the
        # calibration curve.
        G2 = np.exp(-0.5 * ((fg - 2 * cf) / (1.6 * bw)) ** 2)
        L_cf = 10.0 ** offset / cf ** exponent
        if plateau > 0:
            L_cf = L_cf + plateau * 10.0 ** offset / 40.0 ** exponent
        fund_rel = amp * L_cf ** lam / L_cf
        S = S + harmonic * fund_rel * G2 * L
    S[0] = 0.0
    sigma = np.sqrt(S * SRATE * n / 2.0)
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
        p = np.abs(Xs) ** 2 * scale * 2.0
        p[0] /= 2
        if nper % 2 == 0:
            p[-1] /= 2
        acc[s] = p
    return acc, ff


def build(lam_true, N, rng, harmonic=0.0, plateau=0.0):
    rows = {"alpha": [], "control": []}
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
                                      harmonic, plateau)
            if fsel is None:
                fsel = (ff >= FIT[0]) & (ff <= FIT[1])
            rec[cond] = dict(full=segs.mean(0)[fsel],
                             odd=segs[0::2].mean(0)[fsel],
                             even=segs[1::2].mean(0)[fsel])
        f = ff[fsel]
        lf = np.log10(f)
        bands = {"alpha": (cf - 2, cf + 2), "control": CONTROL_BAND}
        for cond in ("eo", "ec"):
            for split in ("full", "odd", "even"):
                P = np.clip(rec[cond][split], 1e-15, None)
                lP = np.log10(P)
                for bname, (lo, hi) in bands.items():
                    keep = mask_for(f, "flanks", log_flanks(lo, hi))
                    if keep.sum() < 8:
                        continue
                    off, ex = ols(lP, lf, keep)
                    bb = ap_band(off, ex, f, lo, hi)
                    tt = float(np.mean(P[(f >= lo) & (f <= hi)]))
                    rows[bname].append(dict(
                        subject=f"s{i:04d}", cond=cond, split=split,
                        estimator=bname, offset=off, exponent=ex,
                        b_alpha=bb, tot_alpha=tt, a_alpha=tt - bb,
                        iaf=(cf if bname == "alpha" else np.sqrt(lo * hi))))
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    print(f"N = {a.n} simulated subjects per arm\n")
    print(f"{'lam_true':>8s} {'harm':>5s} {'plat':>5s} {'band':8s} {'route':10s} "
          f"{'p':>15s} {'q':>15s} {'lam|p':>7s} {'p_add':>10s} {'p_mult':>10s} {'n':>5s}")
    out = []
    arms = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 0.35, 0.0), (1.0, 0.35, 0.0),
            (0.0, 0.0, 0.25), (0.0, 0.35, 0.25)]
    for lam_true, harm, plat in arms:
        rng = np.random.default_rng(int(97 + 1000 * lam_true + 37 * harm + 11 * plat))
        T = build(lam_true, a.n, rng, harm, plat)
        for bname in ("alpha", "control"):
            for sa, sb in (("full", "full"), ("odd", "even")):
                r = HL.pq_within(T[bname], bname, None, sa, sb)
                if r is None:
                    continue
                route = "within" if sa == "full" else "within IV"
                print(f"{lam_true:8.2f} {harm:5.2f} {plat:5.2f} {bname:8s} "
                      f"{route:10s} {r['p']:8.3f} ({r['p_se']:.3f}) "
                      f"{r['q']:8.3f} ({r['q_se']:.3f}) {r['lam_from_p']:7.3f} "
                      f"{r['p_additive']:10.1e} {r['p_multiplicative']:10.1e} "
                      f"{r['n']:5d}")
                rr = {k: v for k, v in r.items()
                      if k not in ("estimator", "split_a", "split_b")}
                out.append(dict(lambda_true=lam_true, harmonic=harm, plateau=plat,
                                band=bname, route=route, **rr))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "results", "sim_control_band.csv")
    pd.DataFrame(out).to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
