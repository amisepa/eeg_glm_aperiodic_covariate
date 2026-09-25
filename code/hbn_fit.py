"""Fit aperiodic parameters and band power to the extracted HBN PSDs.

Writes to results/:
    hbn_roi.csv   one row per subject x condition x split x estimator, for the
                  posterior ROI-average spectrum
    hbn_chan.csv  one row per subject x condition x channel (censored
                  regression only), for topographies

Usage: python hbn_fit.py [--psd-dir DIR] [--limit N]
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

FIT_RANGE = (2.0, 40.0)   # 40 Hz upper bound: HBN spectra plateau above ~42 Hz
CENSOR = (6.0, 16.0)
IAF_SEARCH = (6.0, 14.0)
ALPHA_HALFWIDTH = 2.0
# flanking windows for a LOCAL aperiodic estimate under the alpha band; robust
# to global curvature (real child spectra steepen with frequency, so a single
# power law over 2-40 Hz is misspecified)
FLANKS = ((3.0, 6.0), (17.0, 30.0))

# GSN-HydroCel-129 posterior cluster (O1, Oz, O2, Pz, PO7-ish, PO8-ish)
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]


def ols_mask(logP, lf, keep):
    """Vectorised log-log OLS. logP: (n, nf). Returns offset, exponent."""
    x = lf[keep]
    X = np.column_stack([np.ones(x.size), x])
    beta, *_ = np.linalg.lstsq(X, logP[:, keep].T, rcond=None)
    return beta[0], -beta[1]


def theilsen(logP, lf):
    nf = lf.size
    i, j = np.triu_indices(nf, 1)
    d = lf[j] - lf[i]
    good = np.abs(d) > 1e-12
    i, j, d = i[good], j[good], d[good]
    s = np.median((logP[:, j] - logP[:, i]) / d, axis=1)
    off = np.median(logP - s[:, None] * lf[None, :], axis=1)
    return off, -s


def band_mean(P, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    return P[..., m].mean(axis=-1)


def ap_band_power(offset, exponent, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    ff = f[m]
    L = (10.0 ** offset)[:, None] / ff[None, :] ** exponent[:, None]
    return L.mean(axis=1)


def find_iaf(P, f):
    """Peak of the residual spectrum after a censored aperiodic fit."""
    lf, logP = np.log10(f), np.log10(P)
    keep = ~((f >= CENSOR[0]) & (f <= CENSOR[1]))
    off, ex = ols_mask(logP[None, :], lf, keep)
    L = 10.0 ** off[0] / f ** ex[0]
    resid = P - L
    s = (f >= IAF_SEARCH[0]) & (f <= IAF_SEARCH[1])
    if not np.any(resid[s] > 0):
        return np.nan
    return float(f[s][np.argmax(resid[s])])


def run_specparam(P, f, n_peaks):
    try:
        from specparam import SpectralModel
    except Exception:
        return dict(offset=np.nan, exponent=np.nan, pk_cf=np.nan,
                    pk_h=np.nan, pk_bw=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SpectralModel(peak_width_limits=(2.0, 12.0), max_n_peaks=n_peaks,
                           aperiodic_mode="fixed", verbose=False)
        try:
            sm.fit(f, P)
            ap = np.atleast_1d(sm.get_params("aperiodic"))
            pk = sm.get_params("peak")
        except Exception:
            return dict(offset=np.nan, exponent=np.nan, pk_cf=np.nan,
                        pk_h=np.nan, pk_bw=np.nan)
    pk = np.atleast_2d(pk)
    cf = hh = bw = np.nan
    if pk.size and np.isfinite(pk[:, 0]).any():
        inband = (pk[:, 0] >= IAF_SEARCH[0]) & (pk[:, 0] <= IAF_SEARCH[1])
        if inband.any():
            k = np.argmax(pk[inband, 1])
            cf, hh, bw = pk[inband][k][:3]
    return dict(offset=float(ap[0]), exponent=float(ap[-1]),
                pk_cf=float(cf), pk_h=float(hh), pk_bw=float(bw))


def process_file(path, want_specparam=True):
    d = np.load(path, allow_pickle=True)
    f0 = d["freqs"].astype(float)
    sel = (f0 >= FIT_RANGE[0]) & (f0 <= FIT_RANGE[1])
    f = f0[sel]
    lf = np.log10(f)
    ch = [str(c) for c in d["ch_names"]]
    roi_ix = [ch.index(c) for c in ROI if c in ch]
    if len(roi_ix) < 3:
        return None, None
    meta = dict(subject=str(d["subject"]), dataset=str(d["dataset"]),
                age=float(d["age"]), sex=str(d["sex"]),
                release=str(d["release"]), srate=float(d["srate"]),
                n_seg_eo=int(d["n_seg_eo"]), n_seg_ec=int(d["n_seg_ec"]))

    # individual alpha frequency from the eyes-closed ROI spectrum
    roi_ec = d["ec"][roi_ix][:, sel].astype(float).mean(0)
    iaf = find_iaf(roi_ec, f)
    if np.isfinite(iaf):
        alo, ahi = iaf - ALPHA_HALFWIDTH, iaf + ALPHA_HALFWIDTH
    else:
        alo, ahi = 8.0, 12.0

    roi_rows, chan_rows = [], []
    keep_cens = ~((f >= CENSOR[0]) & (f <= CENSOR[1]))
    keep_full = np.ones_like(f, dtype=bool)

    for cond in ("eo", "ec"):
        for split, key in (("full", cond), ("odd", cond + "_odd"), ("even", cond + "_even")):
            Pall = d[key][:, sel].astype(float)
            if not np.all(np.isfinite(Pall)) or np.any(Pall <= 0):
                Pall = np.clip(np.nan_to_num(Pall, nan=1e-12), 1e-12, None)
            logPall = np.log10(Pall)

            # ---- ROI average spectrum ----
            Proi = Pall[roi_ix].mean(0)[None, :]
            logProi = np.log10(Proi)
            tot = float(band_mean(Proi, f, alo, ahi)[0])
            for est, keep in (("censored", keep_cens), ("full_reg", keep_full)):
                off, ex = ols_mask(logProi, lf, keep)
                b = float(ap_band_power(off, ex, f, alo, ahi)[0])
                roi_rows.append(dict(**meta, cond=cond, split=split, estimator=est,
                                     offset=float(off[0]), exponent=float(ex[0]),
                                     b_alpha=b, tot_alpha=tot, a_alpha=tot - b,
                                     iaf=iaf, alo=alo, ahi=ahi))
            keep_loc = (((f >= FLANKS[0][0]) & (f <= FLANKS[0][1])) |
                        ((f >= FLANKS[1][0]) & (f <= FLANKS[1][1])))
            off, ex = ols_mask(logProi, lf, keep_loc)
            b = float(ap_band_power(off, ex, f, alo, ahi)[0])
            roi_rows.append(dict(**meta, cond=cond, split=split, estimator="local_flank",
                                 offset=float(off[0]), exponent=float(ex[0]),
                                 b_alpha=b, tot_alpha=tot, a_alpha=tot - b,
                                 iaf=iaf, alo=alo, ahi=ahi))
            off, ex = theilsen(logProi, lf)
            b = float(ap_band_power(off, ex, f, alo, ahi)[0])
            roi_rows.append(dict(**meta, cond=cond, split=split, estimator="theilsen",
                                 offset=float(off[0]), exponent=float(ex[0]),
                                 b_alpha=b, tot_alpha=tot, a_alpha=tot - b,
                                 iaf=iaf, alo=alo, ahi=ahi))
            if want_specparam:
                for npk, nm in ((1, "specparam1"), (3, "specparam3")):
                    sp = run_specparam(Proi[0], f, npk)
                    if np.isfinite(sp["offset"]):
                        b = float(ap_band_power(np.array([sp["offset"]]),
                                                np.array([sp["exponent"]]), f, alo, ahi)[0])
                        # absolute periodic power implied by the log-space peak
                        if np.isfinite(sp["pk_h"]) and np.isfinite(sp["pk_cf"]):
                            Lcf = 10.0 ** sp["offset"] / sp["pk_cf"] ** sp["exponent"]
                            a_from_peak = (10.0 ** sp["pk_h"] - 1.0) * Lcf
                        else:
                            a_from_peak = np.nan
                    else:
                        b, a_from_peak = np.nan, np.nan
                    roi_rows.append(dict(**meta, cond=cond, split=split, estimator=nm,
                                         offset=sp["offset"], exponent=sp["exponent"],
                                         b_alpha=b, tot_alpha=tot, a_alpha=tot - b,
                                         pk_cf=sp["pk_cf"], pk_h=sp["pk_h"],
                                         pk_bw=sp["pk_bw"], a_from_peak=a_from_peak,
                                         iaf=iaf, alo=alo, ahi=ahi))

            # ---- per channel, censored only, full split only ----
            if split == "full":
                off, ex = ols_mask(logPall, lf, keep_cens)
                b = ap_band_power(off, ex, f, alo, ahi)
                tt = band_mean(Pall, f, alo, ahi)
                for k, cname in enumerate(ch):
                    chan_rows.append(dict(subject=meta["subject"], age=meta["age"],
                                          cond=cond, channel=cname,
                                          offset=off[k], exponent=ex[k],
                                          b_alpha=b[k], tot_alpha=tt[k],
                                          a_alpha=tt[k] - b[k], iaf=iaf))
    return roi_rows, chan_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-specparam", action="store_true")
    a = ap.parse_args()
    outdir = a.out_dir or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    os.makedirs(outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} PSD files", flush=True)

    roi, chan = [], []
    for k, p in enumerate(files):
        try:
            r, c = process_file(p, want_specparam=not a.no_specparam)
        except Exception as e:
            print(f"  {os.path.basename(p)}: {type(e).__name__}: {e}", flush=True)
            continue
        if r:
            roi += r
            chan += c
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(files)}", flush=True)

    pd.DataFrame(roi).to_csv(os.path.join(outdir, "hbn_roi.csv"), index=False)
    pd.DataFrame(chan).to_csv(os.path.join(outdir, "hbn_chan.csv"), index=False)
    print("wrote hbn_roi.csv", len(roi), "rows; hbn_chan.csv", len(chan), "rows")


if __name__ == "__main__":
    main()
