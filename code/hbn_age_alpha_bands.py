"""Band definition, specparam itself, and scalp gain in the alpha-vs-age result.

On the quality-controlled sample (hbn_qc_flags.csv, from
hbn_age_alpha_robust.py), from the posterior-ROI spectra of each participant:

  fixed band   background and periodic power in a fixed 8-12 Hz band instead
               of the individual alpha frequency +/- 2 Hz. The individual band
               moves up the 1/f slope as the alpha frequency rises with age,
               which by itself lowers the background under it.
  specparam    specparam 2.0 (fixed aperiodic mode, 2-40 Hz, peak width 1-8
               Hz, up to 6 peaks): "aperiodic-adjusted" alpha as the mean of
               the flattened spectrum log10 P - log10 L over IAF +/- 2 Hz, the
               measure of Troendle et al. (2022); and the height of the fitted
               alpha peak.
  gain         the age slope of log power at 30-45 Hz, where alpha does not
               contribute, as a crude upper reference for a frequency-flat
               (scalp transfer) component of the background decline.
  tipping      the sign of the age trend of the intrinsic rhythm over true
               coupling lambda and the share phi of the background's age
               slope that is gain: it declines iff (1 - lambda)(1 - phi) >
               1 - lambda*, where lambda* is the crossover of the observed
               slopes.

Slopes are ordinary least squares on age with HC3 CIs, in natural-log units
per year unless stated. Writes results/hbn_age_alpha_bands.csv.

Usage: python hbn_age_alpha_bands.py [--psd-dir DIR] [--workers 12]
"""
import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import fit_aperiodic

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROI = ["E62", "E65", "E70", "E75", "E83", "E90"]
warnings.filterwarnings("ignore")


def one(args):
    path, iaf = args
    try:
        from specparam import SpectralModel
        d = np.load(path, allow_pickle=True)
        f0 = d["freqs"].astype(float)
        ch = [str(c) for c in d["ch_names"]]
        ix = [ch.index(c) for c in ROI if c in ch]
        out = dict(subject=str(d["subject"]), age=float(d["age"]))
        for cond in ("eo", "ec"):
            P = d[cond][ix].astype(float).mean(0)
            sel = (f0 >= 2) & (f0 <= 55)
            f, p = f0[sel], P[sel]
            keep = ~((f >= 6) & (f <= 16))
            fit = fit_aperiodic(p, f, keep, "fixed")
            L = 10 ** fit["theta"][0] / f ** fit["theta"][1]
            fb = (f >= 8) & (f <= 12)
            out[f"tot812_{cond}"] = p[fb].mean()
            out[f"b812_{cond}"] = L[fb].mean()
            hf = (f >= 30) & (f <= 45)
            out[f"lnhf_{cond}"] = np.log(p[hf].mean())
            sm_ = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                                aperiodic_mode="fixed", verbose=False)
            sm_.fit(f0, P, [2, 40])
            off, expo = sm_.get_params("aperiodic")[:2]
            fr = (f0 >= 2) & (f0 <= 40)
            flat = np.log10(P[fr]) - (off - expo * np.log10(f0[fr]))
            band = (f0[fr] >= iaf - 2) & (f0[fr] <= iaf + 2)
            out[f"sp_flat_{cond}"] = flat[band].mean()
            pk = np.atleast_2d(sm_.get_params("peak"))
            h = np.nan
            if pk.size and np.isfinite(pk).all():
                j = np.where((pk[:, 0] >= 7) & (pk[:, 0] <= 14))[0]
                if j.size:
                    h = pk[j[np.argmax(pk[j, 1])], 1]
            out[f"sp_peak_{cond}"] = h
        return out
    except Exception:
        return None


def slope(y, x):
    m = np.isfinite(y) & np.isfinite(x)
    r = sm.OLS(y[m], sm.add_constant(x[m])).fit(cov_type="HC3")
    return r.params[1], r.conf_int()[1][0], r.conf_int()[1][1], r.pvalues[1], int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    k = k[(k.model == "fixed") & (k.window == "censor 6-16") & (k.band == "alpha")
          & (k.split == "full") & (k.cond == "ec")].set_index("subject")
    files = glob.glob(os.path.join(a.psd_dir, "*.npz"))
    jobs = []
    for p in files:
        sub = os.path.basename(p).split("_")[1].replace(".npz", "")
        if sub in Q.index and Q.loc[sub, "qc_ok"] and sub in k.index:
            jobs.append((p, float(k.loc[sub, "iaf"])))
    with Pool(a.workers) as pool:
        R = pd.DataFrame([r for r in pool.imap(one, jobs, chunksize=8) if r is not None])
    R = R.drop_duplicates("subject").set_index("subject")
    R = R.join(k[["iaf", "b"]].rename(columns={"b": "b_iaf_ec"}), how="left")
    x = R.age.to_numpy()
    rows = []
    print(f"quality-controlled sample: n = {len(R)}")
    s, lo, hi, p, n = slope(R.iaf.to_numpy(), x)
    print(f"IAF (eyes closed): {s:+.3f} Hz/year [{lo:+.3f}, {hi:+.3f}]")
    rows.append(dict(measure="IAF (Hz/year)", cond="ec", est=s, lo=lo, hi=hi, p=p, n=n))
    for cond in ("ec", "eo"):
        a812 = R[f"tot812_{cond}"] - R[f"b812_{cond}"]
        ok = a812 > 0
        lb = np.log(R[f"b812_{cond}"])
        for lam in (0.0, 1.0):
            y = (np.log(a812.where(ok)) - lam * lb).to_numpy()
            s, lo, hi, p, n = slope(y, x)
            rows.append(dict(measure=f"fixed 8-12 Hz band, lambda={lam:.0f}", cond=cond,
                             est=s, lo=lo, hi=hi, p=p, n=n))
        s_b, *_ = slope(lb.to_numpy(), x)
        s_a, *_ = slope(np.log(a812.where(ok)).to_numpy(), x)
        rows.append(dict(measure="fixed 8-12 Hz band, background ln b", cond=cond,
                         est=s_b, lo=np.nan, hi=np.nan, p=np.nan, n=len(R)))
        rows.append(dict(measure="fixed 8-12 Hz band, crossover lambda*", cond=cond,
                         est=s_a / s_b, lo=np.nan, hi=np.nan, p=np.nan, n=int(ok.sum())))
        for meas, lab, conv in ((f"sp_flat_{cond}", "specparam flattened alpha (log10)", 1),
                                (f"sp_peak_{cond}", "specparam alpha peak height (log10)", 1)):
            s, lo, hi, p, n = slope(R[meas].to_numpy(), x)
            rows.append(dict(measure=lab, cond=cond, est=s, lo=lo, hi=hi, p=p, n=n))
        s, lo, hi, p, n = slope(R[f"lnhf_{cond}"].to_numpy(), x)
        rows.append(dict(measure="ln power 30-45 Hz", cond=cond, est=s, lo=lo, hi=hi,
                         p=p, n=n))
    O = pd.DataFrame(rows)
    for _, r in O.iterrows():
        if "log10" in r.measure:
            f = lambda v: 100 * (10 ** v - 1)
            unit = "%/y"
        elif "Hz/year" in r.measure or "lambda*" in r.measure:
            f = lambda v: v
            unit = ""
        else:
            f = lambda v: 100 * (np.exp(v) - 1)
            unit = "%/y"
        print(f"  {r.cond.upper()} {r.measure:45s} {f(r.est):+8.3f}{unit} "
              f"[{f(r.lo):+.2f}, {f(r.hi):+.2f}] p={r.p:.2g} n={r.n}")
    # gain reference: flat share of the background slope, using IAF-band background
    cr = pd.read_csv(os.path.join(RES, "hbn_age_alpha_crossover.csv"))
    for cond in ("ec", "eo"):
        c = cr[(cr["sample"] == "qc") & (cr.model == "fixed") & (cr.cond == cond)].iloc[0]
        s_hf = O[(O.measure == "ln power 30-45 Hz") & (O.cond == cond)].est.iloc[0]
        phi_ref = s_hf / c.slope_ln_b
        lam_true_crit = 1 - (1 - c.lam_star) / max(1 - phi_ref, 1e-9)
        print(f"  {cond.upper()}: lambda* = {c.lam_star:.2f}; 30-45 Hz slope / background "
              f"slope = phi_ref {phi_ref:.2f}; intrinsic alpha declines iff "
              f"(1-lambda)(1-phi) > {1 - c.lam_star:.2f}; at phi_ref the critical true "
              f"lambda is {lam_true_crit:.2f}; with no gain it is {c.lam_star:.2f}")
        O = pd.concat([O, pd.DataFrame([dict(measure="gain reference phi (30-45 Hz / "
                                                     "background slope)", cond=cond,
                                             est=phi_ref, lo=np.nan, hi=np.nan, p=np.nan,
                                             n=len(R))])])
    O.to_csv(os.path.join(RES, "hbn_age_alpha_bands.csv"), index=False)


if __name__ == "__main__":
    main()
