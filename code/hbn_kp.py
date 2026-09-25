"""HBN coupling analysis with a knee-plus-plateau aperiodic model.

Fits fixed and knee+plateau models to the posterior ROI spectra, reports
residuals by frequency band for each model, and re-estimates the
within-subject coupling exponent under both.

Writes results/hbn_kp_fits.csv and results/hbn_kp_lambda.csv.

Usage: python hbn_kp.py [--limit N] [--models fixed,knee_plateau]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import fit_aperiodic, ap_band_power

FIT_RANGE = (2.0, 55.0)
CENSOR = (6.0, 16.0)
IAF_SEARCH = (6.0, 14.0)
CONTROL_BAND = (30.0, 38.0)
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]
WINDOWS = {"censor 6-16": ("censor", [(6, 16)]),
           "flanks 3-6, 26-36": ("flanks", [(3, 6), (26, 36)])}


def mask_for(f, kind, spec):
    m = np.zeros_like(f, dtype=bool)
    for lo, hi in spec:
        m |= (f >= lo) & (f <= hi)
    return ~m if kind == "censor" else m


def find_iaf(P, f):
    lf, lp = np.log10(f), np.log10(P)
    keep = ~((f >= CENSOR[0]) & (f <= CENSOR[1]))
    X = np.column_stack([np.ones(keep.sum()), lf[keep]])
    beta, *_ = np.linalg.lstsq(X, lp[keep], rcond=None)
    resid = P - 10.0 ** beta[0] / f ** (-beta[1])
    s = (f >= IAF_SEARCH[0]) & (f <= IAF_SEARCH[1])
    if not np.any(resid[s] > 0):
        return np.nan
    return float(f[s][np.argmax(resid[s])])


def cov(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def lambda_delta(T, band, model, window, iv=True, nboot=600, rng=None):
    """Within-subject slope of dlog a on dlog b, optionally split-sample IV."""
    F = T[(T.band == band) & (T.model == model) & (T.window == window)]
    piv = {s: F[F.split == s].pivot_table(index="subject", columns="cond",
                                          values=["a", "b"])
           for s in ("full", "odd", "even")}
    idx = piv["full"].index
    for s in ("odd", "even"):
        idx = idx.intersection(piv[s].index)
    if len(idx) < 20:
        return None

    def d(split, key):
        g = piv[split].loc[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.log(np.where(g[(key, "ec")].to_numpy() > 0,
                                    g[(key, "ec")].to_numpy(), np.nan)),
                    np.log(np.where(g[(key, "eo")].to_numpy() > 0,
                                    g[(key, "eo")].to_numpy(), np.nan)))

    ok = np.ones(len(idx), bool)
    for s in ("full", "odd", "even"):
        g = piv[s].loc[idx]
        for key in ("a", "b"):
            for c in ("ec", "eo"):
                v = g[(key, c)].to_numpy()
                ok &= np.isfinite(v) & (v > 0)
    n_total = len(idx)
    if ok.sum() < 20:
        return dict(band=band, model=model, window=window, n=int(ok.sum()),
                    n_total=n_total, lam_ols=np.nan, lam_iv=np.nan,
                    note="too few subjects with positive periodic power")

    def delta(split, key):
        ec, eo = d(split, key)
        return (ec - eo)[ok]

    da_f, db_f = delta("full", "a"), delta("full", "b")
    lam_ols = cov(db_f, da_f) / cov(db_f, db_f)
    res = dict(band=band, model=model, window=window, n=int(ok.sum()),
               n_total=n_total, frac_kept=float(ok.mean()), lam_ols=lam_ols)
    if iv:
        da_o, db_o = delta("odd", "a"), delta("odd", "b")
        da_e, db_e = delta("even", "a"), delta("even", "b")
        num = 0.5 * (cov(db_o, da_e) + cov(db_e, da_o))
        den = cov(db_o, db_e)
        res["lam_iv"] = num / den if den != 0 else np.nan
        rng = rng or np.random.default_rng(0)
        n = ok.sum()
        bs = []
        for _ in range(nboot):
            j = rng.integers(0, n, n)
            dnm = cov(db_o[j], db_e[j])
            if dnm == 0:
                continue
            bs.append(0.5 * (cov(db_o[j], da_e[j]) + cov(db_e[j], da_o[j])) / dnm)
        if bs:
            res["ci_lo"], res["ci_hi"] = np.percentile(bs, [2.5, 97.5])
            res["se_iv"] = float(np.std(bs))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="fixed,knee_plateau")
    a = ap.parse_args()
    models = a.models.split(",")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(here, "results")

    files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} PSD files, models {models}", flush=True)

    rows = []
    for k, path in enumerate(files):
        try:
            d = np.load(path, allow_pickle=True)
            f0 = d["freqs"].astype(float)
            sel = (f0 >= FIT_RANGE[0]) & (f0 <= FIT_RANGE[1])
            f = f0[sel]
            ch = [str(c) for c in d["ch_names"]]
            ix = [ch.index(c) for c in ROI if c in ch]
            if len(ix) < 3:
                continue
            sub = str(d["subject"])
            iaf = find_iaf(d["ec"][ix][:, sel].astype(float).mean(0), f)
            if not np.isfinite(iaf):
                continue
            bands = {"alpha": (iaf - 2, iaf + 2), "control": CONTROL_BAND}
            for cond in ("eo", "ec"):
                for split, key in (("full", cond), ("odd", cond + "_odd"),
                                   ("even", cond + "_even")):
                    P = d[key][ix][:, sel].astype(float).mean(0)
                    P = np.clip(np.nan_to_num(P, nan=1e-12), 1e-12, None)
                    for wname, (kind, spec) in WINDOWS.items():
                        keep = mask_for(f, kind, spec)
                        for model in models:
                            fit = fit_aperiodic(P, f, keep, model)
                            if fit is None:
                                continue
                            for bname, (lo, hi) in bands.items():
                                b = ap_band_power(fit, f, lo, hi)
                                t = float(np.mean(P[(f >= lo) & (f <= hi)]))
                                rows.append(dict(
                                    subject=sub, cond=cond, split=split,
                                    window=wname, model=model, band=bname,
                                    a=t - b, b=b, tot=t, iaf=iaf,
                                    exponent=fit["exponent"],
                                    knee_freq=fit["knee_freq"],
                                    plateau=fit["plateau"], dev=fit["dev"]))
        except Exception as e:
            print(f"  {os.path.basename(path)}: {type(e).__name__}: {e}", flush=True)
            continue
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(files)}", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(outdir, "hbn_kp_fits.csv"), index=False)
    print(f"\n{T.subject.nunique()} subjects\n")

    print("=== aperiodic fit quality (eyes closed, full split) ===")
    for wname in WINDOWS:
        for model in models:
            S = T[(T.window == wname) & (T.model == model) &
                  (T.cond == "ec") & (T.split == "full") & (T.band == "control")]
            if S.empty:
                continue
            frac = 100 * S.a / S.tot
            print(f"  {wname:20s} {model:14s} chi {S.exponent.median():5.2f}  "
                  f"f_knee {S.knee_freq.median():6.2f}  "
                  f"control-band residual median {frac.median():6.1f}% "
                  f"(IQR {np.percentile(frac,25):6.1f} to {np.percentile(frac,75):6.1f})")

    print("\n=== within-subject coupling, dlog a on dlog b ===")
    print("(a peak-free control band SHOULD lose most subjects once the "
          "aperiodic model fits:\n periodic power there is an estimation "
          "residual centred on zero, not a signal)\n")
    print(f"{'band':9s} {'model':14s} {'window':20s} {'lam OLS':>9s} "
          f"{'lam IV':>9s} {'95% CI':>18s} {'n':>5s}")
    out = []
    for band in ("alpha", "control"):
        for model in models:
            for wname in WINDOWS:
                r = lambda_delta(T, band, model, wname)
                if r is None:
                    continue
                ci = (f"[{r.get('ci_lo', np.nan):5.2f}, {r.get('ci_hi', np.nan):5.2f}]")
                print(f"{band:9s} {model:14s} {wname:20s} {r['lam_ols']:9.3f} "
                      f"{r.get('lam_iv', np.nan):9.3f} {ci:>18s} {r['n']:5d}"
                      f"  ({100*r.get('frac_kept', np.nan):.0f}% of {r.get('n_total','?')}"
                      f" had positive periodic power)")
                out.append(r)
    pd.DataFrame(out).to_csv(os.path.join(outdir, "hbn_kp_lambda.csv"), index=False)
    print(f"\nwrote {os.path.join(outdir, 'hbn_kp_lambda.csv')}")


if __name__ == "__main__":
    main()
