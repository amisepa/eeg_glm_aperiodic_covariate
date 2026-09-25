"""Topographic test: do the alpha change and the aperiodic change overlap?

For each subject and channel, computes the eyes-closed minus eyes-open change
in fitted exponent, in aperiodic power under the alpha band, and in periodic
alpha power, then correlates the group maps across channels (the test
proposed by M. Miyakoshi in sccn/OneOverF discussion #6).

The statistic depends on the fit range and is affected by alpha leaking into
the aperiodic fit; compare it with sim_topography_leakage.py before
interpreting it.

subject_changes() and flank_changes() are shared with the simulation so that
real and simulated data go through the same code.

Usage: python hbn_topography.py [--fit-range 2,40] [--censor 6,16]
       [--model fixed|knee_plateau] [--workers N] [--tag SUFFIX] [--limit N]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import fit_aperiodic

IAF_SEARCH = (6.0, 14.0)
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]


def ols_channels(logP, lf, keep):
    """Vectorised log-log OLS over channels. logP: (nch, nf)."""
    X = np.column_stack([np.ones(keep.sum()), lf[keep]])
    beta, *_ = np.linalg.lstsq(X, logP[:, keep].T, rcond=None)
    return beta[0], -beta[1]


def peak_freq(P, f, keep, search=IAF_SEARCH):
    """Frequency of the largest positive residual above a log-log OLS fit."""
    off, ex = ols_channels(np.log10(P)[None, :], np.log10(f), keep)
    resid = P - 10.0 ** off[0] / f ** ex[0]
    s = (f >= search[0]) & (f <= search[1])
    if not np.any(resid[s] > 0):
        return None
    return float(f[s][np.argmax(resid[s])])


def subject_changes(f0, P_eo, P_ec, ch, fit_range=(2.0, 40.0),
                    censor=(6.0, 16.0), roi=ROI, model="fixed"):
    """Per-channel EC-minus-EO changes for one subject.

    Returns (d_exponent, d_log_alpha, d_log_b, iaf) with one value per
    channel, or None if the subject fails the checks. The IAF is taken from
    the eyes-closed ROI spectrum and the same IAF +/- 2 Hz band is used in
    both conditions. Shared with sim_topography_leakage.py so that the
    simulation runs exactly this code. model is "fixed" (log-log OLS) or
    "knee_plateau" (Whittle fit per channel, ~12 s per subject).
    """
    sel = (f0 >= fit_range[0]) & (f0 <= fit_range[1])
    f = f0[sel]
    lf = np.log10(f)
    keep = ~((f >= censor[0]) & (f <= censor[1]))
    ix = [ch.index(c) for c in roi if c in ch]
    if len(ix) < 3:
        return None
    iaf = peak_freq(np.asarray(P_ec)[ix][:, sel].astype(float).mean(0), f, keep)
    if iaf is None:
        return None
    am = (f >= iaf - 2) & (f <= iaf + 2)

    vals = {}
    for cond, P in (("eo", P_eo), ("ec", P_ec)):
        P = np.asarray(P)[:, sel].astype(float)
        P = np.clip(np.nan_to_num(P, nan=1e-12), 1e-12, None)
        tot = P[:, am].mean(1)
        if model == "fixed":
            o, e = ols_channels(np.log10(P), lf, keep)
            L = (10.0 ** o)[:, None] / f[am][None, :] ** e[:, None]
            b = L.mean(1)
            vals[cond] = (e, b, tot - b)
        else:
            # knee+plateau per channel: e is the asymptotic slope chi (not
            # comparable with fixed-mode values), b is the NEURAL aperiodic
            # power in the band (plateau excluded), a is what is left after
            # removing neural + plateau.
            e = np.full(P.shape[0], np.nan)
            b = np.full(P.shape[0], np.nan)
            a_ = np.full(P.shape[0], np.nan)
            for c in range(P.shape[0]):
                fit = fit_aperiodic(P[c], f, keep, "knee_plateau")
                if fit is None:
                    continue
                th = fit["theta"]
                neural = 10.0 ** th[0] / (10.0 ** th[2] + f[am] ** th[1])
                e[c] = th[1]
                b[c] = neural.mean()
                a_[c] = tot[c] - b[c] - 10.0 ** th[3]
            vals[cond] = (e, b, a_)
    ee, be, ae = vals["ec"]
    eo_, bo, ao = vals["eo"]
    ok = (ae > 0) & (ao > 0) & (be > 0) & (bo > 0)
    if ok.sum() < 100:
        return None
    de = ee - eo_
    dla = np.where(ok, np.log(np.where(ae > 0, ae, np.nan)) -
                   np.log(np.where(ao > 0, ao, np.nan)), np.nan)
    dlb = np.where(ok, np.log(be) - np.log(bo), np.nan)
    return de, dla, dlb, iaf


FLANKS = {"low": (2.0, 4.0), "high": (30.0, 40.0)}


def flank_changes(f0, P_eo, P_ec, ch, roi=ROI, flanks=FLANKS):
    """Fit-free version of the topographic test.

    Per channel: EC-minus-EO change in log mean power in bands well away from
    alpha and its harmonic, and in the IAF +/- 2 Hz band. No aperiodic model
    is fitted, so nothing can leak from the alpha peak into the "background"
    estimate except through the Gaussian tails. Returns ({band: d_log P},
    d_log_alpha_total, iaf) or None.
    """
    sel = (f0 >= 2.0) & (f0 <= 40.0)
    f = f0[sel]
    keep = ~((f >= 6.0) & (f <= 16.0))
    ix = [ch.index(c) for c in roi if c in ch]
    if len(ix) < 3:
        return None
    iaf = peak_freq(np.asarray(P_ec)[ix][:, sel].astype(float).mean(0), f, keep)
    if iaf is None:
        return None

    def band(P, lo, hi):
        m = (f0 >= lo) & (f0 <= hi)
        return np.log(np.clip(np.asarray(P)[:, m].astype(float), 1e-30, None).mean(1))
    out = {k: band(P_ec, *v) - band(P_eo, *v) for k, v in flanks.items()}
    dla = band(P_ec, iaf - 2, iaf + 2) - band(P_eo, iaf - 2, iaf + 2)
    return out, dla, iaf


def _one(args):
    path, fit_range, censor, model = args
    try:
        d = np.load(path, allow_pickle=True)
        ch = [str(c) for c in d["ch_names"]]
        r = subject_changes(d["freqs"].astype(float), d["eo"], d["ec"],
                            ch, fit_range, censor, model=model)
        if r is None:
            return None
        return ch, r, str(d["subject"]), float(d["age"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--censor", default="6,16",
                    help="aperiodic censor window in Hz, e.g. 4,20")
    ap.add_argument("--fit-range", default="2,40",
                    help="aperiodic fit range in Hz, e.g. 5,40")
    ap.add_argument("--model", default="fixed",
                    choices=["fixed", "knee_plateau"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default="",
                    help="where to write the .npz (default: results/)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, e.g. _c4-20")
    a = ap.parse_args()
    censor = tuple(float(x) for x in a.censor.split(","))
    fit_range = tuple(float(x) for x in a.fit_range.split(","))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = a.outdir or os.path.join(here, "results")

    files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} PSD files", flush=True)

    d_exp, d_loga, d_logb, chan_ref, subs, ages, iafs = [], [], [], None, [], [], []
    jobs = [(p, fit_range, censor, a.model) for p in files]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            results = pool.imap(_one, jobs, chunksize=4)
            results = list(results)
    else:
        results = map(_one, jobs)
    for k, res in enumerate(results):
        if (k + 1) % 250 == 0:
            print(f"  {k+1}/{len(files)}", flush=True)
        if res is None:
            continue
        ch, (de, dla, dlb, iaf), sub, age = res
        if chan_ref is None:
            chan_ref = ch
        elif ch != chan_ref:
            continue
        d_exp.append(de); d_loga.append(dla); d_logb.append(dlb)
        subs.append(sub); ages.append(age); iafs.append(iaf)

    DE = np.array(d_exp); DA = np.array(d_loga); DB = np.array(d_logb)
    print(f"\n{DE.shape[0]} subjects x {DE.shape[1]} channels\n")
    np.savez_compressed(os.path.join(outdir, f"hbn_topography{a.tag}.npz"),
                        d_exponent=DE, d_log_a=DA, d_log_b=DB,
                        ch_names=np.array(chan_ref), subject=np.array(subs),
                        age=np.array(ages), iaf=np.array(iafs),
                        censor=np.array(censor),
                        fit_range=np.array(fit_range), model=a.model)

    # ---- group topographies ----
    mE = np.nanmean(DE, 0)
    mA = np.nanmean(DA, 0)
    mB = np.nanmean(DB, 0)
    good = np.isfinite(mE) & np.isfinite(mA) & np.isfinite(mB)
    r_EA = np.corrcoef(mE[good], mA[good])[0, 1]
    r_BA = np.corrcoef(mB[good], mA[good])[0, 1]
    print("=== Miyakoshi's test: do the group topographies overlap? ===")
    print(f"  spatial r( mean d_exponent , mean d_log alpha ) = {r_EA:+.3f}")
    print(f"  spatial r( mean d_log b    , mean d_log alpha ) = {r_BA:+.3f}")
    print(f"  channels used: {good.sum()}")
    k = 8
    order_a = np.argsort(-mA[good]); order_e = np.argsort(-mE[good])
    names = np.array(chan_ref)[good]
    print(f"  top {k} channels for alpha change:    {list(names[order_a[:k]])}")
    print(f"  top {k} channels for exponent change: {list(names[order_e[:k]])}")

    # ---- within-subject spatial correlation ----
    rs = []
    for i in range(DE.shape[0]):
        m = np.isfinite(DE[i]) & np.isfinite(DA[i])
        if m.sum() > 60:
            rs.append(np.corrcoef(DE[i][m], DA[i][m])[0, 1])
    rs = np.array(rs)
    print("\n=== within-subject, across-channel correlation ===")
    print(f"  r(d_exponent, d_log alpha) across channels, per subject:")
    print(f"    median {np.nanmedian(rs):+.3f}, IQR [{np.nanpercentile(rs,25):+.3f}, "
          f"{np.nanpercentile(rs,75):+.3f}], n = {rs.size}")

    # ---- across-subject correlation at the posterior ROI ----
    ix = [chan_ref.index(c) for c in ROI if c in chan_ref]
    aROI = np.nanmean(DA[:, ix], 1)
    bROI = np.nanmean(DB[:, ix], 1)
    m = np.isfinite(aROI) & np.isfinite(bROI)
    print("\n=== across-subject correlation at the posterior ROI ===")
    print(f"  r(d_log b, d_log alpha) across subjects = "
          f"{np.corrcoef(bROI[m], aROI[m])[0,1]:+.3f}  (n = {m.sum()})")
    print("\n  Strong across-subject with weak across-channel is the")
    print("  signature of a shared state factor, not of coupling.")
    print(f"\nwrote {os.path.join(outdir, f'hbn_topography{a.tag}.npz')}")


if __name__ == "__main__":
    main()
