"""Specificity controls for the HBN coupling estimate.

Re-runs the within-subject coupling analysis under different aperiodic
windows (censor windows and log-symmetric flanking windows, including ones
that avoid the alpha harmonic near 2 x IAF) and in several bands, including a
peak-free control band (30-38 Hz) where no coupling can exist.

Writes results/hbn_controls.csv, hbn_ctrl_window.csv and hbn_ctrl_band.csv.

Usage: python hbn_controls.py [--psd-dir DIR] [--limit N]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hbn_lambda as HL

FIT_RANGE = (2.0, 40.0)
IAF_SEARCH = (6.0, 14.0)
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]

# ---- window variants: how the aperiodic component is estimated ------------
# ("name", kind, spec) with kind "censor" (exclude these ranges) or
# "flanks" (fit ONLY these ranges)
WINDOWS = [
    ("censor 6-16",          "censor", [(6, 16)]),
    ("censor 4-20",          "censor", [(4, 20)]),
    ("censor 5-14",          "censor", [(5, 14)]),
    ("censor 6-16 + 18-24",  "censor", [(6, 16), (18, 24)]),
    ("flanks 3-6, 17-30",    "flanks", [(3, 6), (17, 30)]),
    ("flanks 3-6, 26-36",    "flanks", [(3, 6), (26, 36)]),
]


def mask_for(f, kind, spec):
    m = np.zeros_like(f, dtype=bool)
    for lo, hi in spec:
        m |= (f >= lo) & (f <= hi)
    return ~m if kind == "censor" else m


def ols(logP, lf, keep):
    X = np.column_stack([np.ones(keep.sum()), lf[keep]])
    beta, *_ = np.linalg.lstsq(X, logP[keep], rcond=None)
    return float(beta[0]), float(-beta[1])


def ap_band(offset, exponent, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.mean(10.0 ** offset / f[m] ** exponent))


def find_iaf(P, f):
    keep = ~((f >= 6) & (f <= 16))
    off, ex = ols(np.log10(P), np.log10(f), keep)
    resid = P - 10.0 ** off / f ** ex
    s = (f >= IAF_SEARCH[0]) & (f <= IAF_SEARCH[1])
    if not np.any(resid[s] > 0):
        return np.nan
    return float(f[s][np.argmax(resid[s])])


def log_flanks(lo, hi, fmin=2.5, fmax=40.0):
    """Flanking windows placed symmetrically in log frequency around a band."""
    lo_f = (max(fmin, lo / 2.2), max(fmin + 0.5, lo / 1.35))
    hi_f = (min(fmax - 0.5, hi * 1.35), min(fmax, hi * 2.2))
    return [lo_f, hi_f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(here, "results")

    files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} PSD files", flush=True)

    win_rows, band_rows = [], []
    for k, path in enumerate(files):
        try:
            d = np.load(path, allow_pickle=True)
            f0 = d["freqs"].astype(float)
            sel = (f0 >= FIT_RANGE[0]) & (f0 <= FIT_RANGE[1])
            f = f0[sel]
            lf = np.log10(f)
            ch = [str(c) for c in d["ch_names"]]
            ix = [ch.index(c) for c in ROI if c in ch]
            if len(ix) < 3:
                continue
            sub = str(d["subject"])
            iaf = find_iaf(d["ec"][ix][:, sel].astype(float).mean(0), f)
            if not np.isfinite(iaf):
                continue
            bands = {"theta": (4.0, 7.0), "alpha": (iaf - 2, iaf + 2),
                     "beta": (18.0, 25.0), "control": (30.0, 38.0)}

            for cond in ("eo", "ec"):
                for split, key in (("full", cond), ("odd", cond + "_odd"),
                                   ("even", cond + "_even")):
                    P = d[key][ix][:, sel].astype(float).mean(0)
                    P = np.clip(np.nan_to_num(P, nan=1e-12), 1e-12, None)
                    lP = np.log10(P)

                    # --- window sweep, alpha band only ---
                    alo, ahi = bands["alpha"]
                    tot = float(np.mean(P[(f >= alo) & (f <= ahi)]))
                    for name, kind, spec in WINDOWS:
                        keep = mask_for(f, kind, spec)
                        if keep.sum() < 8:
                            continue
                        off, ex = ols(lP, lf, keep)
                        b = ap_band(off, ex, f, alo, ahi)
                        win_rows.append(dict(subject=sub, cond=cond, split=split,
                                             estimator=name, offset=off,
                                             exponent=ex, b_alpha=b,
                                             tot_alpha=tot, a_alpha=tot - b,
                                             iaf=iaf))

                    # --- band sweep, uniform log-symmetric flanks ---
                    for bname, (lo, hi) in bands.items():
                        keep = mask_for(f, "flanks", log_flanks(lo, hi))
                        if keep.sum() < 8:
                            continue
                        off, ex = ols(lP, lf, keep)
                        bb = ap_band(off, ex, f, lo, hi)
                        tt = float(np.mean(P[(f >= lo) & (f <= hi)]))
                        band_rows.append(dict(subject=sub, cond=cond, split=split,
                                              estimator=bname, offset=off,
                                              exponent=ex, b_alpha=bb,
                                              tot_alpha=tt, a_alpha=tt - bb,
                                              iaf=(iaf if bname == "alpha"
                                                   else np.sqrt(lo * hi))))
        except Exception as e:
            print(f"  {os.path.basename(path)}: {type(e).__name__}: {e}", flush=True)
            continue
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(files)}", flush=True)

    W = pd.DataFrame(win_rows)
    B = pd.DataFrame(band_rows)
    W.to_csv(os.path.join(outdir, "hbn_ctrl_window.csv"), index=False)
    B.to_csv(os.path.join(outdir, "hbn_ctrl_band.csv"), index=False)
    print(f"\n{W.subject.nunique()} subjects\n")

    def report(T, title, keys):
        print(f"=== {title} ===")
        print(f"{'variant':22s} {'route':10s} {'p':>15s} {'q':>15s} "
              f"{'lam|p':>7s} {'lam|q':>7s} {'p_add':>10s} {'p_mult':>10s} {'n':>5s}")
        rows = []
        for est in keys:
            for sa, sb in (("full", "full"), ("odd", "even")):
                r = HL.pq_within(T, est, None, sa, sb)
                if r is None:
                    continue
                route = "within" if sa == "full" else "within IV"
                print(f"{est:22s} {route:10s} {r['p']:8.3f} ({r['p_se']:.3f}) "
                      f"{r['q']:8.3f} ({r['q_se']:.3f}) {r['lam_from_p']:7.3f} "
                      f"{r['lam_from_q']:7.3f} {r['p_additive']:10.1e} "
                      f"{r['p_multiplicative']:10.1e} {r['n']:5d}")
                rows.append(dict(sweep=title, route=route, **r))
        print()
        return rows

    out = []
    out += report(W, "WINDOW sweep (alpha band)", [w[0] for w in WINDOWS])
    out += report(B, "BAND sweep (log-symmetric flanks)",
                  ["theta", "alpha", "beta", "control"])
    pd.DataFrame(out).to_csv(os.path.join(outdir, "hbn_controls.csv"), index=False)
    print(f"wrote {os.path.join(outdir, 'hbn_controls.csv')}")


if __name__ == "__main__":
    main()
