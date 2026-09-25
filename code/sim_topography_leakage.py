"""Null simulation for the topographic test.

Simulates 129-channel eyes-open and eyes-closed spectra in which the
aperiodic background does not change at all (true change in exponent and in
background level is zero everywhere) and alpha is additive (lambda = 0), with
eyes-open and eyes-closed alpha heights taken from the median HBN
topography. The data are synthesised in the time domain with the HBN block
durations and Welch settings, then passed through the same per-subject code
as the real data (hbn_topography.subject_changes and flank_changes). Any map
correlation it reports is produced by the analysis itself.

Usage: python sim_topography_leakage.py [--n 200] [--harmonic 0.15]
       [--iaf-shift -0.3] [--model fixed|knee_plateau] [--workers 12]
Needs results/hbn_alpha_rel_heights.npz (created from the HBN PSDs on first
run if missing).
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_topography import (subject_changes, flank_changes, ols_channels,
                            peak_freq, ROI)

SRATE = 250.0
DUR = {"eo": 90.0, "ec": 180.0}
WIN_SEC = 4.0
SPECS = [((2.0, 40.0), (6.0, 16.0)), ((4.0, 40.0), (6.0, 16.0)),
         ((5.0, 40.0), (7.0, 16.0)), ((2.0, 30.0), (6.0, 16.0))]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def real_heights(psd_dir, n=150):
    """Median over subjects of per-channel alpha band a/b, EO and EC."""
    files = sorted(glob.glob(os.path.join(psd_dir, "*.npz")))[:2 * n:2]
    R = {"eo": [], "ec": []}
    ch = None
    for p in files:
        d = np.load(p, allow_pickle=True)
        f0 = d["freqs"].astype(float)
        sel = (f0 >= 2) & (f0 <= 40)
        f = f0[sel]
        keep = ~((f >= 6) & (f <= 16))
        c = [str(x) for x in d["ch_names"]]
        if len(c) != 129:
            continue
        ch = c
        ix = [c.index(r) for r in ROI]
        iaf = peak_freq(d["ec"][ix][:, sel].mean(0), f, keep)
        if iaf is None:
            continue
        am = (f >= iaf - 2) & (f <= iaf + 2)
        for cond in R:
            P = np.clip(d[cond][:, sel].astype(float), 1e-12, None)
            o, e = ols_channels(np.log10(P), np.log10(f), keep)
            b = ((10.0 ** o)[:, None] / f[am][None, :] ** e[:, None]).mean(1)
            R[cond].append(P[:, am].mean(1) / b - 1)
    return (np.nanmedian(np.array(R["eo"]), 0),
            np.nanmedian(np.array(R["ec"]), 0), ch)


def aperiodic(fg, b, chi, knee_freq, plateau):
    return 10.0 ** b / (knee_freq ** chi + fg ** chi) + plateau


def synth_welch(S, dur, rng):
    """Channels x freqs target PSD -> Welch estimate of a Gaussian process."""
    n = int(SRATE * dur)
    nch = S.shape[0]
    nh = n // 2
    sigma = np.sqrt(S * SRATE * n / 2.0)
    X = np.zeros((nch, nh + 1), dtype=complex)
    X[:, 1:nh] = sigma[:, 1:nh] * (rng.standard_normal((nch, nh - 1)) +
                                   1j * rng.standard_normal((nch, nh - 1))) / np.sqrt(2)
    X[:, nh] = sigma[:, nh] * rng.standard_normal(nch)
    x = np.fft.irfft(X, n=n, axis=1)
    nper = int(WIN_SEC * SRATE)
    step = nper // 2
    nseg = 1 + (n - nper) // step
    win = np.hanning(nper + 1)[:nper]
    scale = 1.0 / (SRATE * np.sum(win ** 2))
    acc = 0.0
    for s in range(nseg):
        seg = x[:, s * step: s * step + nper]
        seg = seg - seg.mean(1, keepdims=True)
        p = np.abs(np.fft.rfft(seg * win, axis=1)) ** 2 * scale * 2.0
        p[:, 0] /= 2
        p[:, -1] /= 2
        acc = acc + p
    return acc / nseg, np.fft.rfftfreq(nper, 1 / SRATE)


def simulate_subject(args):
    """One simulated subject under the null; returns {spec: changes}."""
    seed, harmonic, iaf_shift, model, rel_eo, rel_ec, ch_off, ch_chi, ch = args
    rng = np.random.default_rng(seed)
    b = rng.normal(2.2, 0.35)
    chi = np.clip(rng.normal(3.0, 0.3), 2.0, 4.0)
    kf = np.clip(rng.normal(5.0, 1.0), 2.0, 9.0)
    plat = 10.0 ** rng.normal(np.log10(0.016), 0.2)
    cf = rng.normal(9.5, 0.8)
    bw = 1.8
    gain = np.exp(rng.normal(0, 0.5))        # subject alpha strength
    gain_ec = np.exp(rng.normal(0, 0.3))     # subject EC reactivity

    spectra = {}
    for cond in ("eo", "ec"):
        n = int(SRATE * DUR[cond])
        fg = np.fft.rfftfreq(n, 1 / SRATE)
        L = aperiodic(fg[None, :], b + ch_off[:, None], chi + ch_chi[:, None],
                      kf, plat)
        # lambda = 0: absolute peak amplitude set from a REFERENCE background
        # (median parameters, no subject offset), never from this subject's
        # own background.
        Lref_cf = aperiodic(cf, 2.2 + ch_off, 3.0 + ch_chi, 5.0, 0.016)
        rel = rel_eo if cond == "eo" else rel_ec * gain_ec
        c_f = cf + (iaf_shift if cond == "eo" else 0.0)
        amp = gain * rel * Lref_cf
        G = np.exp(-0.5 * ((fg[None, :] - c_f) / bw) ** 2)
        S = L + amp[:, None] * G
        if harmonic > 0:
            # harmonic specified by its RELATIVE height at its own frequency
            L2 = aperiodic(2 * c_f, b + ch_off, chi + ch_chi, kf, plat)
            Lcf = aperiodic(c_f, b + ch_off, chi + ch_chi, kf, plat)
            rel_here = amp / Lcf
            G2 = np.exp(-0.5 * ((fg[None, :] - 2 * c_f) / (1.6 * bw)) ** 2)
            S = S + (harmonic * rel_here * L2)[:, None] * G2
        S[:, 0] = 0.0
        spectra[cond], f0 = synth_welch(S, DUR[cond], rng)
    out = {}
    for spec in SPECS:
        r = subject_changes(f0, spectra["eo"], spectra["ec"], ch, *spec,
                            model=model)
        out[spec] = None if r is None else r[:3]
    out["flanks"] = flank_changes(f0, spectra["eo"], spectra["ec"], ch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--harmonic", type=float, default=0.15,
                    help="harmonic height relative to the local background, "
                         "as a fraction of the fundamental's relative height")
    ap.add_argument("--iaf-shift", type=float, default=0.0,
                    help="eyes-open IAF minus eyes-closed IAF, Hz")
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="fixed", choices=["fixed", "knee_plateau"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    hfile = os.path.join(RES, "hbn_alpha_rel_heights.npz")
    if not os.path.exists(hfile):
        heo, hec, ch = real_heights(a.psd_dir)
        np.savez(hfile, eo=heo, ec=hec, ch=np.array(ch))
    h = np.load(hfile)
    ch = [str(c) for c in h["ch"]]
    # band-mean a/b over IAF +/- 2 Hz is ~0.75 of the peak's relative height
    rel_eo = np.maximum(h["eo"], 0.05) / 0.75
    rel_ec = np.maximum(h["ec"], 0.05) / 0.75
    nch = len(ch)
    print(f"relative peak height, median over channels: "
          f"EO {np.median(rel_eo):.2f}, EC {np.median(rel_ec):.2f}")

    # fixed per-channel background map (identical in both conditions)
    ch_off = rng.normal(0, 0.15, nch)
    ch_chi = rng.normal(0, 0.15, nch)

    seeds = np.random.SeedSequence(a.seed).spawn(a.n)
    jobs = [(sd, a.harmonic, a.iaf_shift, a.model, rel_eo, rel_ec, ch_off,
             ch_chi, ch) for sd in seeds]
    out = {spec: [] for spec in SPECS}
    out["flanks"] = []
    if a.workers > 1:
        with Pool(a.workers) as pool:
            res = list(pool.imap(simulate_subject, jobs, chunksize=2))
    else:
        res = list(map(simulate_subject, jobs))
    for r in res:
        for spec, v in r.items():
            if v is not None:
                out[spec].append(v)

    post = [ch.index(c) for c in ROI]
    print(f"\nNULL: true d_exponent = 0, true d_log b = 0, lambda = 0, "
          f"harmonic {a.harmonic}, EO-EC IAF shift {a.iaf_shift:+.2f} Hz, "
          f"aperiodic model {a.model}")
    print(f"{'fit range':>10} {'censor':>8} {'n':>4} {'r(dExp,dLogA)':>14} "
          f"{'r(dLogB,dLogA)':>15} {'dExp post':>10} {'dExp all':>9} "
          f"{'dLogB post':>11}")
    fl = out.pop("flanks")
    for spec, rows in out.items():
        DE = np.array([x[0] for x in rows])
        DA = np.array([x[1] for x in rows])
        DB = np.array([x[2] for x in rows])
        mE, mA, mB = (np.nanmean(X, 0) for X in (DE, DA, DB))
        g = np.isfinite(mE) & np.isfinite(mA) & np.isfinite(mB)
        rEA = np.corrcoef(mE[g], mA[g])[0, 1]
        rBA = np.corrcoef(mB[g], mA[g])[0, 1]
        print(f"{spec[0][0]:>4.0f}-{spec[0][1]:<4.0f} {spec[1][0]:>3.0f}-{spec[1][1]:<4.0f}"
              f" {len(rows):>4} {rEA:>+14.3f} {rBA:>+15.3f} "
              f"{np.nanmean(DE[:, post]):>+10.3f} {np.nanmean(DE):>+9.3f} "
              f"{np.nanmean(DB[:, post]):>+11.3f}")

    print("\nfit-free flank test: spatial r(map d_log P(band), "
          "map d_log P(IAF+/-2))")
    DA = np.array([x[1] for x in fl])
    for k in fl[0][0]:
        DF = np.array([x[0][k] for x in fl])
        mF, mA = np.nanmean(DF, 0), np.nanmean(DA, 0)
        print(f"  {k:>5}: r = {np.corrcoef(mF, mA)[0, 1]:+.3f}, "
              f"mean d_log P = {np.nanmean(DF):+.4f} (n = {len(fl)})")


if __name__ == "__main__":
    main()
