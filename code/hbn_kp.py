"""HBN coupling analysis with a knee-plus-plateau aperiodic model.

Fits fixed and knee+plateau models to the posterior ROI spectra, reports
residuals by frequency band for each model, and re-estimates the eyes-closed
versus eyes-open coupling exponent under both, in the alpha band and in a
peak-free control band (30-38 Hz).

Two backgrounds are kept per band: b, the band mean of the whole fitted
background, and b_neural, the band mean of 10^b / (k + f^chi), i.e. without
the plateau p (amplifier and residual muscle noise). Periodic power is always
a = total - b; b_neural can replace b as the coupling regressor, since nothing
couples to amplifier noise. For the power law b_neural = b.

Estimators (participants passing quality control, results/hbn_qc_flags.csv,
unless --all):
  lambda_delta  the previous estimator: symmetric split-half slope of d ln a
                on d ln b over participants with a > 0 in every split; the
                selection biases it towards 1
  log-free      lambda_gmm.bootstrap on the odd/even halves (condition 1 =
                eyes open, 2 = eyes closed), once with b and once with
                b_neural as the regressor; for knee+plateau also on the
                participants whose plateau stays below half of the band
                background in every half and condition (where the knee
                collapses, the plateau takes the level and b_neural is tiny)
Both fitting windows contain bins of the control band (censor 6-16 fits
2-55 Hz outside 6-16 Hz; the flanks include 30-36 Hz), so in that band the
instrument shares data with the band total. hbn_controls.py fits the control
band on flanks that exclude it.

Writes results/hbn_kp_fits.csv (per participant, kept out of the
repository), results/hbn_kp_lambda.csv (previous estimator) and
results/hbn_kp_gmm.csv (log-free).

Usage: python hbn_kp.py [--limit N] [--models fixed,knee_plateau] [--workers N]
                        [--from-fits] [--all] [--nboot 300] [--out-dir DIR]
  --from-fits  skip the fits and read hbn_kp_fits.csv from the output
               directory; a missing b_neural is then b - plateau, which is
               exact because the plateau is flat
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import fit_aperiodic, ap_band_power
import lambda_gmm as G

FIT_RANGE = (2.0, 55.0)
CENSOR = (6.0, 16.0)
IAF_SEARCH = (6.0, 14.0)
CONTROL_BAND = (30.0, 38.0)
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]
WINDOWS = {"censor 6-16": ("censor", [(6, 16)]),
           "flanks 3-6, 26-36": ("flanks", [(3, 6), (26, 36)])}
DF = 0.25          # frequency resolution of the extracted PSDs (Hz)
PLATEAU_MAX = 0.5  # subset check: plateau share of the band background


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


def neural_band_power(fit, f, lo, hi):
    """Band mean of the fitted background without the plateau."""
    m = (f >= lo) & (f <= hi)
    return float(np.mean(10.0 ** fit["offset"] / (fit["knee"] + f[m] ** fit["exponent"])))


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


def halves(T, band, model, window):
    """Odd/even band powers per condition, arrays (n, 2): column 0 = odd."""
    F = T[(T.band == band) & (T.model == model) & (T.window == window)
          & (T.split != "full")]
    w = F.pivot_table(index="subject", columns=["cond", "split"],
                      values=["tot", "b", "b_neural", "iaf"]).dropna()
    H = {(v, c): w[v][c][["odd", "even"]].to_numpy()
         for v in ("tot", "b", "b_neural", "iaf") for c in ("eo", "ec")}
    return H, len(w)


def gmm_lambda(H, background, nboot, seed=0):
    """Log-free lambda with the total or the neural background as regressor.

    a = t - b (total background) in both cases. With the neural background,
    lambda_gmm is given t - p, where p = b - b_neural comes from the half that
    supplies b, so that it forms (t - b) / b_neural^lambda, and the
    instrument and weights come from b_neural.
    """
    t1, t2 = H["tot", "eo"], H["tot", "ec"]
    b1, b2 = H["b", "eo"], H["b", "ec"]
    if background == "neural":
        # column s of t pairs with column 1 - s of b inside lambda_gmm
        t1 = t1 - (b1 - H["b_neural", "eo"])[:, ::-1]
        t2 = t2 - (b2 - H["b_neural", "ec"])[:, ::-1]
        b1, b2 = H["b_neural", "eo"], H["b_neural", "ec"]
    r = G.bootstrap(t1, t2, b1, b2, nboot=nboot, rng=np.random.default_rng(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.log(b2) - np.log(b1)
    ok = np.all(np.isfinite(z), 1)
    return dict(lam=r["lam"], lo=r["ci"][0], hi=r["ci"][1], boot_sd=r["boot_sd"],
                boot_fail=r["boot_fail"], delta=r["delta"], n_roots=len(r["roots"]),
                roots=" ".join(f"{x:.3f}" for x in r["roots"]),
                r_instrument=float(np.corrcoef(z[ok, 0], z[ok, 1])[0, 1]))


def band_in_fit(iaf, band, window):
    """Share of participants whose band shares frequency bins with the fit."""
    f = np.arange(FIT_RANGE[0], FIT_RANGE[1] + DF / 2, DF)
    kind, spec = WINDOWS[window]
    keep = mask_for(f, kind, spec)
    out = []
    for x in iaf:
        lo, hi = (x - 2, x + 2) if band == "alpha" else CONTROL_BAND
        out.append(np.any(keep & (f >= lo) & (f <= hi)))
    return float(np.mean(out))


def subject_rows(args):
    """Fit rows for one PSD file; returns (rows, error message or None)."""
    path, models = args
    rows = []
    try:
        d = np.load(path, allow_pickle=True)
        f0 = d["freqs"].astype(float)
        sel = (f0 >= FIT_RANGE[0]) & (f0 <= FIT_RANGE[1])
        f = f0[sel]
        ch = [str(c) for c in d["ch_names"]]
        ix = [ch.index(c) for c in ROI if c in ch]
        if len(ix) < 3:
            return rows, None
        sub = str(d["subject"])
        iaf = find_iaf(d["ec"][ix][:, sel].astype(float).mean(0), f)
        if not np.isfinite(iaf):
            return rows, None
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
                                plateau=fit["plateau"], dev=fit["dev"],
                                b_neural=neural_band_power(fit, f, lo, hi)))
    except Exception as e:
        return [], f"{os.path.basename(path)}: {type(e).__name__}: {e}"
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="fixed,knee_plateau")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--from-fits", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="use every participant, not only those passing QC")
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    models = a.models.split(",")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = a.out_dir or os.path.join(here, "results")
    fits_path = os.path.join(outdir, "hbn_kp_fits.csv")

    if a.from_fits:
        T = pd.read_csv(fits_path)
        T = T[T.model.isin(models)]
        if "b_neural" not in T:
            T["b_neural"] = T.b - T.plateau
        print(f"read {fits_path}", flush=True)
    else:
        files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
        if a.limit:
            files = files[:a.limit]
        print(f"{len(files)} PSD files, models {models}", flush=True)

        jobs = [(p, models) for p in files]
        if a.workers > 1:
            pool = Pool(a.workers)
            results = pool.imap(subject_rows, jobs, chunksize=4)
        else:
            results = map(subject_rows, jobs)
        rows = []
        for k, (r, err) in enumerate(results):
            if err:
                print(f"  {err}", flush=True)
            rows += r
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(files)}", flush=True)
        if a.workers > 1:
            pool.close()

        T = pd.DataFrame(rows)
        T.to_csv(fits_path, index=False)
    print(f"\n{T.subject.nunique()} subjects fitted")
    if not a.all:
        Q = pd.read_csv(os.path.join(here, "results", "hbn_qc_flags.csv"), index_col=0)
        T = T[T.subject.isin(Q.index[Q.qc_ok])]
    print(f"{T.subject.nunique()} subjects analysed "
          f"({'all' if a.all else 'passing quality control'})\n")

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
                  f"(IQR {np.percentile(frac,25):6.1f} to {np.percentile(frac,75):6.1f})  "
                  f"plateau share of b {100 * np.median(1 - S.b_neural / S.b):5.1f}%")

    print("\n=== previous estimator: within-subject dlog a on dlog b ===")
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

    print("\n=== log-free estimator (lambda_gmm), eyes open -> eyes closed ===")
    print("(regressor: total fitted background, or the neural part without the "
          "plateau;\n a = total - full background in both; 'in fit' = share of "
          "participants whose band\n shares bins with the background fit; subset "
          f"'p<{PLATEAU_MAX}': participants whose plateau is\n below {PLATEAU_MAX} of "
          "the band background in every half and condition)\n")
    print(f"{'band':8s} {'model':13s} {'window':18s} {'subset':6s} {'regressor':9s} "
          f"{'lambda':>7s} {'95% interval':>16s} {'fail':>5s} {'roots':>5s} {'r_iv':>5s} "
          f"{'in fit':>6s} {'n':>5s}")
    gout = []
    for band in ("alpha", "control"):
        for model in models:
            for wname in WINDOWS:
                H, n = halves(T, band, model, wname)
                if n < 20:
                    continue
                share = band_in_fit(H["iaf", "ec"][:, 0], band, wname)
                subsets = [("all", H)]
                if model != "fixed":
                    # a knee that collapses lets the plateau take the level;
                    # b_neural is then tiny and dominates the moments
                    p_share = np.max([1 - H["b_neural", c] / H["b", c] for c in ("eo", "ec")],
                                     axis=(0, 2))
                    keep = p_share < PLATEAU_MAX
                    subsets.append((f"p<{PLATEAU_MAX}", {k: v[keep] for k, v in H.items()}))
                for sname, Hs in subsets:
                    ns = len(Hs["tot", "eo"])
                    for bg in ("total", "neural"):
                        r = gmm_lambda(Hs, bg, a.nboot)
                        print(f"{band:8s} {model:13s} {wname:18s} {sname:6s} {bg:9s} "
                              f"{r['lam']:7.3f} [{r['lo']:6.2f}, {r['hi']:5.2f}] "
                              f"{r['boot_fail']:5.2f} {r['n_roots']:5d} "
                              f"{r['r_instrument']:5.2f} {share:6.2f} {ns:5d}")
                        gout.append(dict(band=band, model=model, window=wname, subset=sname,
                                         background=bg, n=ns, **r, frac_band_in_fit=share))
    pd.DataFrame(gout).to_csv(os.path.join(outdir, "hbn_kp_gmm.csv"), index=False)
    print(f"\nwrote {os.path.join(outdir, 'hbn_kp_lambda.csv')} and hbn_kp_gmm.csv")


if __name__ == "__main__":
    main()
