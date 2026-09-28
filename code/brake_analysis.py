"""Propofol loss of consciousness (Brake et al. 2024 Nat Commun): do the
band-power changes before and after LOC depend on lambda?

Data: Cz spectrograms of 14 patients, aligned to loss of consciousness
(object drop), -300 to +60 s, 0.5-Hz resolution (figshare 24777990, CC BY
4.0). Brake et al. detrended each spectrum by dividing out a fitted
aperiodic model (their "eq6": a synaptic filter with a noise floor, plus up
to three Gaussian peaks for delta, alpha and beta, fitted in log10 power by
least absolute deviation; code at github.com/niklasbrake/EEG_modelling),
i.e. lambda = 1, and reported that detrended delta rises only at LOC while
alpha and beta rise earlier.

Here each spectrogram is averaged into 5-s bins, the same model is fitted to
each bin over 0.5-100 Hz (55-65 Hz excluded), and band power is split into
the fitted aperiodic part b and the rest a = total - b for delta (1-4 Hz),
alpha (8-15 Hz) and beta (15-30 Hz). Changes from baseline to a pre-LOC
window (-60 to -10 s) and a post-LOC window (+10 to +60 s) are reported as a
function of lambda, with the crossover lambda*, and, for comparison, Brake's
own detrended measure, the mean of 10 log10(P / L) over the band.

Two baselines: "pre-infusion", the minute before each patient's infusion
onset (to 5 s before it; from the authors' timing table in the manuscript
source data, figshare file 43599408, exported from MATLAB as
data_time_information.csv; the spectrograms start 300 s before LOC, so one
patient whose infusion began 285 s before LOC has 15 s), and "first 60 s"
of the spectrogram, the approximation used when the timing was not
available.

Usage: python brake_analysis.py [--data ZIP] [--timing CSV] [--bin 5]
Writes results/brake_bins.csv (per patient and bin) and results/brake_lambda.csv.
"""
import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambda_curve import effect_curve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
BANDS = {"delta": (1.0, 4.0), "alpha": (8.0, 15.0), "beta": (15.0, 30.0)}
WINDOWS = {"baseline": (-300.0, -240.0), "pre-LOC": (-60.0, -10.0), "post-LOC": (10.0, 60.0)}

# ---- Brake et al. eq6 model, as in their detrending.py -------------------
LB_AP, UB_AP = [7e-3, 3.9e-3, -21, 1], [75e-3, 4.1e-3, -7, 5]
SP_AP = [17e-3, 4e-3, -10.5, 4.2]
LB_P = [0, 0, 0.2, 6, 0, 0.6, 15, 0, 1]
UB_P = [4, 4, 3, 15, 4, 4, 40, 3, 10]
SP_P = [0.5, 2, 1.5, 8, 0.3, 1, 22, 0.1, 4]


def eq6(f, tau1, tau2, offset, mag):
    w2 = (2 * np.pi * f) ** 2
    tau2 = 4e-3
    x2 = (tau1 - tau2) ** 2 / ((1 + tau1 ** 2 * w2) * (1 + tau2 ** 2 * w2))
    return mag + np.log10(np.exp(offset) + x2)


def peaks(f, p):
    y = np.zeros_like(f)
    for i in range(0, len(p), 3):
        c, h, w = p[i:i + 3]
        y = y + h * np.exp(-(f - c) ** 2 / (2 * w ** 2))
    return y


def fit_bin(f, logP, start=None):
    """LAD fit of eq6 + 3 peaks in log10 power; returns the parameter vector."""
    lb, ub = LB_AP + LB_P, UB_AP + UB_P
    x0 = np.clip(np.array(start if start is not None else SP_AP + SP_P, float),
                 np.array(lb) + 1e-9, np.array(ub) - 1e-9)
    obj = lambda x: np.mean(np.abs(logP - eq6(f, *x[:4]) - peaks(f, x[4:])))
    r = optimize.least_squares(lambda x: [obj(x)], x0, bounds=(lb, ub))
    return r.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"),
                                                   "brake2024", "spectrogram_Cz_all_subjects.zip"))
    ap.add_argument("--timing", default=os.path.join(
        os.environ.get("EEG_DATA", "eeg_data"), "brake2024", "source", "_data", "EEG_data",
        "data_time_information.csv"))
    ap.add_argument("--bin", type=float, default=5.0)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.data)
    f = pd.read_csv(io.BytesIO(z.read("csv/frequency.csv"))).iloc[:, 0].to_numpy(float)
    t = pd.read_csv(io.BytesIO(z.read("csv/time.csv"))).iloc[:, 0].to_numpy(float)
    fsel = (f >= 0.5) & (f <= 100) & ~((f > 55) & (f < 65))
    edges = np.arange(-300, 60.0001, a.bin)
    rows = []
    for pt in range(1, 15):
        S = pd.read_csv(io.BytesIO(z.read(f"csv/pt_{pt:02d}.csv")), header=None).to_numpy(float)
        start = None
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t >= lo) & (t < hi)
            P = np.nanmean(S[:, m], axis=1)
            good = np.isfinite(P) & (P > 0)
            if m.sum() == 0 or good[fsel].mean() < 0.9:
                continue
            ff, lP = f[fsel & good], np.log10(P[fsel & good])
            x = fit_bin(ff, lP, start)
            start = x
            L = 10 ** eq6(f, *x[:4])                      # aperiodic, linear
            row = dict(patient=pt, t=(lo + hi) / 2, n_frames=int(m.sum()))
            for name, (b0, b1) in BANDS.items():
                bm = (f >= b0) & (f < b1) & good
                tot, b = float(np.mean(P[bm])), float(np.mean(L[bm]))
                row.update({f"{name}_tot": tot, f"{name}_b": b, f"{name}_a": tot - b,
                            f"{name}_brake_db": float(np.mean(10 * np.log10(P[bm] / L[bm])))})
            rows.append(row)
        print(f"patient {pt}: {sum(r['patient'] == pt for r in rows)} bins", flush=True)
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(RES, "brake_bins.csv"), index=False)

    out = []
    W = {w: B[(B.t >= lo) & (B.t < hi)].groupby("patient").mean(numeric_only=True)
         for w, (lo, hi) in WINDOWS.items()}
    bases = {"first 60 s": W["baseline"]}
    if os.path.exists(a.timing):
        T = pd.read_csv(a.timing)
        T.index = np.arange(1, len(T) + 1)               # row k = patient k
        onset = (T.infusion_onset - T.object_drop).reindex(B.patient).to_numpy()
        pre = B[(B.t >= onset - 60) & (B.t < onset - 5)]
        bases["pre-infusion"] = pre.groupby("patient").mean(numeric_only=True)
        dur = pre.groupby("patient").size() * a.bin
        print("pre-infusion baseline, s per patient:", dur.to_dict())
    for base, W0 in bases.items():
        for band in BANDS:
            for win in ("pre-LOC", "post-LOC"):
                d = W[win].join(W0, rsuffix="_0", how="inner")
                with np.errstate(invalid="ignore", divide="ignore"):
                    dA = np.log(d[f"{band}_a"].where(d[f"{band}_a"] > 0)) - \
                        np.log(d[f"{band}_a_0"].where(d[f"{band}_a_0"] > 0))
                dB = np.log(d[f"{band}_b"]) - np.log(d[f"{band}_b_0"])
                dT = np.log(d[f"{band}_tot"]) - np.log(d[f"{band}_tot_0"])
                dBrake = d[f"{band}_brake_db"] - d[f"{band}_brake_db_0"]
                r = effect_curve(dA.to_numpy(), dB.to_numpy(), rng=np.random.default_rng(0))
                tt, tb = stats.ttest_1samp(dT, 0), stats.ttest_1samp(dBrake, 0)
                c = r["curve"]
                out.append(dict(baseline=base, band=band, window=win, n=r["n"], n_total=len(d),
                                total_dB=float(10 / np.log(10) * dT.mean()),
                                total_p=float(tt.pvalue), brake_dB=float(dBrake.mean()),
                                brake_p=float(tb.pvalue), s_a=r["s_a"], s_b=r["s_b"],
                                s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
                                lam_star_bounded=r["lam_star_bounded"],
                                lam_star=r["lam_star"], hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1],
                                lam0=c[0][0], lam0_lo=c[0][1], lam0_hi=c[0][2],
                                lam1=c[-1][0], lam1_lo=c[-1][1], lam1_hi=c[-1][2],
                                p_cross_in_01=r["p_cross_in_01"]))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(RES, "brake_lambda.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
