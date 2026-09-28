"""Specificity controls for the HBN coupling estimate.

Re-runs the eyes-closed versus eyes-open coupling analysis under different
aperiodic windows (censor windows and flanking windows, including ones that
avoid the alpha harmonic near 2 x IAF) and in several bands, including a
peak-free control band (30-38 Hz) where no coupling can exist. Each variant
is estimated three ways:

  (p, q) regression  within-subject regression of d log10 a on the changes in
                     aperiodic offset and exponent (hbn_lambda.pq_within,
                     lambda = p), on the full spectra ("within") and with the
                     outcome from odd and the predictors from even segments
                     ("within IV", one direction)
  slope IV           symmetric split-half slope of d ln a on d ln b
                     (hbn_kp.lambda_delta)
  log-free           lambda_gmm.bootstrap on the odd/even halves, condition
                     1 = eyes open, 2 = eyes closed
The first two need a > 0 (in both conditions; in every split for the slope),
a selection that biases them towards 1, so the number and share of
participants kept are reported. The log-free estimator keeps everyone; it is
reported with its 95% participant-bootstrap interval, the share of bootstrap
samples without a root (boot_fail) or with several, the roots of the
full-sample moment and the split-half reliability of the change in ln b
(instrument strength).

Backgrounds are log-log least-squares fits over 2-40 Hz; the band sweep uses
flanking windows placed log-symmetrically around each band. The control band
is the exception: its log-symmetric flanks (13.6-22.2 and 39.5-40 Hz) lie
almost entirely below it, on beta and the alpha harmonic. It is fitted with a
local power law on flanks on both sides, 24-29 and 39-42 Hz, 1 Hz clear of
the band so that the instrument and the weights share no bins with it, and
ending at 42 Hz because HBN spectra flatten into the amplifier floor above
that (a power law through the floor would tilt the fit). As a check on the
background model, the control band is also fitted with the knee+plateau model
(ap_models, Whittle) over 2-55 Hz with 6-16 and 28-40 Hz left out ("control
kp"), which models the floor instead of avoiding it; there the log-free
estimate is given with the whole fitted background and with its neural part
10^b / (k + f^chi) as the regressor (a = total - whole background in both).
The (p, q) regression is not run on it (knee-mode offset and exponent do not
determine the band background alone).

Only participants passing quality control (results/hbn_qc_flags.csv, qc_ok)
enter the estimates unless --all is given.

Writes results/hbn_controls.csv ((p, q) regression), results/hbn_ctrl_gmm.csv
(log-free estimator and slope IV, one row per variant), and the
per-participant fits results/hbn_ctrl_window.csv and hbn_ctrl_band.csv
(subject IDs; kept out of the repository).

Usage: python hbn_controls.py [--psd-dir DIR] [--limit N] [--nboot 300] [--all]
                              [--workers N] [--out-dir DIR]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hbn_lambda as HL
import lambda_gmm as G
from ap_models import fit_aperiodic, ap_band_power
from hbn_kp import lambda_delta, neural_band_power

FIT_RANGE = (2.0, 40.0)
IAF_SEARCH = (6.0, 14.0)
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]
CONTROL_BAND = (30.0, 38.0)
CONTROL_FLANKS = [(24.0, 29.0), (39.0, 42.0)]
KP_RANGE = (2.0, 55.0)
KP_CENSOR = [(6.0, 16.0), (28.0, 40.0)]

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
BANDS = ["theta", "alpha", "beta", "control", "control kp"]


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


def band_flanks(name, lo, hi):
    return CONTROL_FLANKS if name == "control" else log_flanks(lo, hi)


def subject_rows(path):
    """Window-sweep and band-sweep rows for one PSD file."""
    d = np.load(path, allow_pickle=True)
    f0 = d["freqs"].astype(float)
    sel = (f0 >= FIT_RANGE[0]) & (f0 <= KP_RANGE[1])
    f = f0[sel]
    lf = np.log10(f)
    in_fit = f <= FIT_RANGE[1]            # the 2-40 Hz fits
    top = f <= CONTROL_FLANKS[-1][1]      # the control-band power law
    kp_keep = mask_for(f, "censor", KP_CENSOR) & (f >= KP_RANGE[0])
    ch = [str(c) for c in d["ch_names"]]
    ix = [ch.index(c) for c in ROI if c in ch]
    if len(ix) < 3:
        return [], []
    sub = str(d["subject"])
    iaf = find_iaf(d["ec"][ix][:, sel].astype(float).mean(0)[in_fit], f[in_fit])
    if not np.isfinite(iaf):
        return [], []
    bands = {"theta": (4.0, 7.0), "alpha": (iaf - 2, iaf + 2),
             "beta": (18.0, 25.0), "control": CONTROL_BAND}

    win_rows, band_rows = [], []
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
                keep = mask_for(f, kind, spec) & in_fit
                if keep.sum() < 8:
                    continue
                off, ex = ols(lP, lf, keep)
                b = ap_band(off, ex, f, alo, ahi)
                win_rows.append(dict(subject=sub, cond=cond, split=split,
                                     estimator=name, offset=off,
                                     exponent=ex, b_alpha=b,
                                     tot_alpha=tot, a_alpha=tot - b,
                                     iaf=iaf))

            # --- band sweep, flanking windows ---
            for bname, (lo, hi) in bands.items():
                keep = mask_for(f, "flanks", band_flanks(bname, lo, hi))
                keep &= top if bname == "control" else in_fit
                if keep.sum() < 8:
                    continue
                off, ex = ols(lP, lf, keep)
                bb = ap_band(off, ex, f, lo, hi)
                tt = float(np.mean(P[(f >= lo) & (f <= hi)]))
                band_rows.append(dict(subject=sub, cond=cond, split=split,
                                      estimator=bname, offset=off,
                                      exponent=ex, b_alpha=bb, b_neural=bb,
                                      tot_alpha=tt, a_alpha=tt - bb,
                                      iaf=(iaf if bname == "alpha"
                                           else np.sqrt(lo * hi))))

            # --- control band, knee+plateau background with the band left out ---
            fit = fit_aperiodic(P, f, kp_keep, "knee_plateau")
            if fit is not None:
                lo, hi = CONTROL_BAND
                bb = ap_band_power(fit, f, lo, hi)
                tt = float(np.mean(P[(f >= lo) & (f <= hi)]))
                band_rows.append(dict(subject=sub, cond=cond, split=split,
                                      estimator="control kp", offset=fit["offset"],
                                      exponent=fit["exponent"], b_alpha=bb,
                                      b_neural=neural_band_power(fit, f, lo, hi),
                                      tot_alpha=tt, a_alpha=tt - bb,
                                      iaf=np.sqrt(lo * hi)))
    return win_rows, band_rows


def subject_job(path):
    try:
        return subject_rows(path) + (None,)
    except Exception as e:
        return [], [], f"{os.path.basename(path)}: {type(e).__name__}: {e}"


def gmm_row(T, est, nboot, background="total", seed=0):
    """Log-free estimate for one variant, with its diagnostics.

    background "neural": a = total - whole background as before, but the
    regressor, instrument and weights use b_neural (lambda_gmm is given
    t - p, with p = b - b_neural from the half that supplies b).
    """
    F = T[(T.estimator == est) & (T.split != "full")]
    cols = ["tot_alpha", "b_alpha"] + (["b_neural"] if background == "neural" else [])
    w = F.pivot_table(index="subject", columns=["cond", "split"], values=cols).dropna()
    get = lambda v, c: w[v][c][["odd", "even"]].to_numpy()
    t1, t2, b1, b2 = get("tot_alpha", "eo"), get("tot_alpha", "ec"), \
        get("b_alpha", "eo"), get("b_alpha", "ec")
    if background == "neural":
        # column s of t pairs with column 1 - s of b inside lambda_gmm
        t1 = t1 - (b1 - get("b_neural", "eo"))[:, ::-1]
        t2 = t2 - (b2 - get("b_neural", "ec"))[:, ::-1]
        b1, b2 = get("b_neural", "eo"), get("b_neural", "ec")
    n = len(w)
    r = G.bootstrap(t1, t2, b1, b2, nboot=nboot, rng=np.random.default_rng(seed))
    # number of roots in each bootstrap sample (same draws as the bootstrap)
    near = r["lam"] if np.isfinite(r["lam"]) else None
    rng = np.random.default_rng(seed)
    nroots = np.array([len(G.estimate(t1[i], t2[i], b1[i], b2[i], near=near)["roots"])
                       for i in (rng.integers(0, n, n) for _ in range(nboot))])
    z = np.log(b2) - np.log(b1)
    full = T[(T.estimator == est) & (T.split == "full")]
    res = lambda c: 100 * np.median(full[full.cond == c].a_alpha / full[full.cond == c].tot_alpha)
    return dict(background=background, n=n, lam=r["lam"], ci_lo=r["ci"][0], ci_hi=r["ci"][1],
                boot_sd=r["boot_sd"], boot_fail=r["boot_fail"],
                boot_multi_root=float(np.mean(nroots > 1)), delta=r["delta"],
                n_roots=len(r["roots"]), roots=" ".join(f"{x:.3f}" for x in r["roots"]),
                r_instrument=float(np.corrcoef(z[:, 0], z[:, 1])[0, 1]),
                resid_eo_pct=float(res("eo")), resid_ec_pct=float(res("ec")))


def slope_row(T, est):
    """Previous estimator: symmetric split-half slope of d ln a on d ln b."""
    K = T[T.estimator == est].rename(columns={"a_alpha": "a", "b_alpha": "b"})
    K = K.assign(band=est, model="ols", window=est)
    r = lambda_delta(K, est, "ols", est)
    if r is None:
        return {}
    return dict(slope_iv=r.get("lam_iv", np.nan), slope_ci_lo=r.get("ci_lo", np.nan),
                slope_ci_hi=r.get("ci_hi", np.nan), slope_n=r["n"],
                slope_frac_kept=r.get("frac_kept", np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--all", action="store_true",
                    help="use every participant, not only those passing QC")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = a.out_dir or os.path.join(here, "results")

    files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} PSD files", flush=True)

    pool = Pool(a.workers) if a.workers > 1 else None
    results = pool.imap(subject_job, files, chunksize=4) if pool else map(subject_job, files)
    win_rows, band_rows = [], []
    for k, (w, b, err) in enumerate(results):
        if err:
            print(f"  {err}", flush=True)
        win_rows += w
        band_rows += b
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(files)}", flush=True)
    if pool:
        pool.close()

    W = pd.DataFrame(win_rows)
    B = pd.DataFrame(band_rows)
    W.to_csv(os.path.join(outdir, "hbn_ctrl_window.csv"), index=False)
    B.to_csv(os.path.join(outdir, "hbn_ctrl_band.csv"), index=False)
    print(f"\n{W.subject.nunique()} subjects with an alpha peak")
    if not a.all:
        Q = pd.read_csv(os.path.join(here, "results", "hbn_qc_flags.csv"), index_col=0)
        ok = set(Q.index[Q.qc_ok])
        W, B = W[W.subject.isin(ok)], B[B.subject.isin(ok)]
    print(f"{W.subject.nunique()} subjects analysed "
          f"({'all' if a.all else 'passing quality control'})\n")

    reg, gmm = [], []
    for T, title, keys in ((W, "WINDOW sweep (alpha band)", [w[0] for w in WINDOWS]),
                           (B, "BAND sweep (flanking windows)", BANDS)):
        print(f"=== {title} ===")
        print(f"{'variant':20s} {'within':>20s} {'within IV':>20s} {'slope IV':>26s} "
              f"{'log-free [95%]':>24s} {'fail':>5s} {'roots':>5s} {'r_iv':>5s} {'n':>5s}")
        for est in keys:
            cells = []
            for sa, sb in (("full", "full"), ("odd", "even")):
                r = HL.pq_within(T, est, None, sa, sb) if est != "control kp" else None
                if r is None:
                    cells.append("-")
                    continue
                r["frac_kept"] = r["n"] / r["n_total"]
                reg.append(dict(sweep=title, route="within" if sa == "full" else "within IV", **r))
                cells.append(f"{r['p']:6.3f} ({r['p_se']:.3f}) {100 * r['frac_kept']:3.0f}%")
            sl = slope_row(T, est)
            slope = (f"{sl['slope_iv']:6.3f} [{sl['slope_ci_lo']:5.2f},{sl['slope_ci_hi']:5.2f}] "
                     f"{100 * sl['slope_frac_kept']:3.0f}%" if "slope_iv" in sl else "-")
            for bg in (("total", "neural") if est == "control kp" else ("total",)):
                g = dict(sweep=title, variant=est, **gmm_row(T, est, a.nboot, bg), **sl)
                gmm.append(g)
                label = est + (" (neural)" if bg == "neural" else "")
                print(f"{label:20s} {cells[0]:>20s} {cells[1]:>20s} {slope:>26s} "
                      f"{g['lam']:7.3f} [{g['ci_lo']:6.2f}, {g['ci_hi']:6.2f}] "
                      f"{g['boot_fail']:5.2f} {g['n_roots']:5d} {g['r_instrument']:5.2f} "
                      f"{g['n']:5d}")
        print()

    pd.DataFrame(reg).to_csv(os.path.join(outdir, "hbn_controls.csv"), index=False)
    D = pd.DataFrame(gmm)
    D.to_csv(os.path.join(outdir, "hbn_ctrl_gmm.csv"), index=False)
    print("=== log-free diagnostics (residual = median a / total, %) ===")
    print(D[["variant", "background", "roots", "boot_multi_root", "delta",
             "resid_eo_pct", "resid_ec_pct"]].round(3).to_string(index=False))
    print(f"\nwrote {os.path.join(outdir, 'hbn_controls.csv')} and hbn_ctrl_gmm.csv")


if __name__ == "__main__":
    main()
