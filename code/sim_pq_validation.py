"""Positive and negative controls for the real-data coupling analysis.

Synthesises two-condition spectra with a known coupling exponent, with and
without a common instrumental gain, formats them like results/hbn_roi.csv and
runs the same functions that produce the HBN numbers (hbn_fit estimators,
hbn_lambda tests).

Usage: python sim_pq_validation.py [--n 400] [--reps 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hbn_fit as HF
import hbn_lambda as HL

SRATE = 250.0
DUR = 100.0          # s per condition, matching ~5 x 20 s eyes-open blocks
WIN_SEC = 4.0


def synth_psd(offset, exponent, cf, bw, amp, lam, f, rng):
    """Welch PSD of a Gaussian process with the model PSD as its expectation."""
    n = int(SRATE * DUR)
    fg = np.fft.rfftfreq(n, 1 / SRATE)
    L = 10.0 ** offset / np.maximum(fg, 1e-9) ** exponent
    G = np.exp(-0.5 * ((fg - cf) / bw) ** 2)
    S = L + amp * G * L ** lam
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
    P, ff = HF_welch(x, nper)
    return P, ff


def HF_welch(x, nper):
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


def build_table(lam_true, N, rng, instrumental_gain=False):
    """instrumental_gain=True multiplies each subject's WHOLE spectrum, periodic
    part included, by a common factor in both conditions -- what electrode gain,
    skull thickness or head size actually do. It cannot change the within-subject
    contrast, but it drives the between-subject slope to 1 whatever lambda is."""
    fsel = None
    rows = []
    for i in range(N):
        gain = 1.4 * rng.standard_normal()                 # log10 gain, +-1.4 dex
        expo_eo = 1.3 + 0.30 * rng.standard_normal()
        expo_ec = expo_eo + (0.25 + 0.20 * rng.random())   # EC steeper
        off_eo = 1.0 + (0.0 if instrumental_gain else gain)
        off_ec = off_eo + (0.10 + 0.10 * rng.random())     # EC broadband up
        cf = 10.0 + 0.8 * rng.standard_normal()
        bw = 1.5 + 0.15 * rng.standard_normal()
        # intrinsic strength INDEPENDENT of this subject's background, scaled by
        # a fixed reference level so that SNR is comparable across lambda
        Lref = 10.0 ** 1.0 / 10.0 ** 1.3
        c_eo = 0.8 * Lref ** (1 - lam_true) * 10 ** (0.25 * rng.standard_normal())
        c_ec = c_eo * 2.5                                   # same +150% on every subject
        rec = {}
        for cond, (off, ex, c) in (("eo", (off_eo, expo_eo, c_eo)),
                                   ("ec", (off_ec, expo_ec, c_ec))):
            segs, ff = synth_psd(off, ex, cf, bw, c, lam_true, None, rng)
            if instrumental_gain:
                segs = segs * (10.0 ** gain)       # scales L AND the peak
            if fsel is None:
                fsel = (ff >= HF.FIT_RANGE[0]) & (ff <= HF.FIT_RANGE[1])
            f = ff[fsel]
            rec[cond] = dict(full=segs.mean(0)[fsel],
                             odd=segs[0::2].mean(0)[fsel],
                             even=segs[1::2].mean(0)[fsel])
        f = ff[fsel]
        lf = np.log10(f)
        iaf = HF.find_iaf(rec["ec"]["full"], f)
        alo, ahi = ((iaf - 2, iaf + 2) if np.isfinite(iaf) else (8.0, 12.0))
        keep_cens = ~((f >= HF.CENSOR[0]) & (f <= HF.CENSOR[1]))
        keep_loc = (((f >= HF.FLANKS[0][0]) & (f <= HF.FLANKS[0][1])) |
                    ((f >= HF.FLANKS[1][0]) & (f <= HF.FLANKS[1][1])))
        for cond in ("eo", "ec"):
            for split in ("full", "odd", "even"):
                P = rec[cond][split][None, :]
                logP = np.log10(np.clip(P, 1e-15, None))
                tot = float(HF.band_mean(P, f, alo, ahi)[0])
                for est, keep in (("censored", keep_cens), ("local_flank", keep_loc),
                                  ("full_reg", np.ones_like(f, bool))):
                    off, ex = HF.ols_mask(logP, lf, keep)
                    b = float(HF.ap_band_power(off, ex, f, alo, ahi)[0])
                    rows.append(dict(subject=f"s{i:04d}", cond=cond, split=split,
                                     estimator=est, offset=float(off[0]),
                                     exponent=float(ex[0]), b_alpha=b,
                                     tot_alpha=tot, a_alpha=tot - b, iaf=iaf,
                                     age=np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    print(f"validation: N = {a.n} simulated subjects per lambda, "
          f"{a.reps} repetition(s)\n")
    print(f"{'lam_true':>8s} {'gain':>5s} {'rep':>3s} {'estimator':12s} {'route':14s} "
          f"{'p':>14s} {'q':>14s} {'lam|p':>7s} {'lam|q':>7s} "
          f"{'p_add':>9s} {'p_mult':>9s}")
    out = []
    arms = [(0.0, False), (0.5, False), (1.0, False), (0.0, True), (1.0, True)]
    for lam_true, gain_on in arms:
        for rep in range(a.reps):
            rng = np.random.default_rng(1000 + 17 * rep + int(100 * lam_true)
                                        + (7 if gain_on else 0))
            T = build_table(lam_true, a.n, rng, instrumental_gain=gain_on)
            for est in ("censored", "local_flank", "full_reg"):
                F = T[(T.estimator == est) & (T.cond == "ec") & (T.split == "full")]
                d = HL.pq_test(F.a_alpha.to_numpy(), F.offset.to_numpy(),
                               F.exponent.to_numpy(), F.iaf.to_numpy())
                w = HL.pq_within(T, est, None, "full", "full")
                wi = HL.pq_within(T, est, None, "odd", "even")
                for nm, r in (("between", d), ("within", w), ("within IV", wi)):
                    if r is None:
                        continue
                    print(f"{lam_true:8.2f} {str(gain_on):>5s} {rep:3d} {est:12s} "
                          f"{nm:14s} {r['p']:7.3f} ({r['p_se']:.3f}) {r['q']:7.3f} "
                          f"({r['q_se']:.3f}) {r['lam_from_p']:7.3f} "
                          f"{r['lam_from_q']:7.3f} {r['p_additive']:9.1e} "
                          f"{r['p_multiplicative']:9.1e}")
                    rr = {k: v for k, v in r.items()
                          if k not in ("estimator", "split_a", "split_b")}
                    out.append(dict(lambda_true=lam_true, gain=gain_on, rep=rep,
                                    estimator=est, route=nm, **rr))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "results", "sim_pq_validation.csv")
    pd.DataFrame(out).to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
