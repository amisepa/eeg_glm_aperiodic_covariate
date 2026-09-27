"""Graded propofol sedation (Chennu et al. 2016 PLoS Comput Biol): does the
frontal shift of alpha, and the alpha change with dose, depend on lambda?

Data: 20 healthy adults, 91-channel EEG (preprocessed by the authors: 0.5-45
Hz, 10-s epochs, average reference), about 7 min eyes closed at baseline,
mild (target 0.6 ug/ml) and moderate (1.2 ug/ml) sedation and recovery,
with measured plasma propofol and hit rates in a two-choice task after each
period (Apollo doi:10.17863/CAM.68959, CC BY 2.0 UK). As in the paper, 7
participants whose hit rate at moderate sedation fell significantly below
baseline (binomial test) form the "drowsy" group, the others "responsive".

Per file: Welch spectra (4-s Hann, 50% overlap, within 10-s epochs) of the
posterior (O1, Oz, O2, Pz, P3, P4) and frontal (Fz, F3, F4) regions; the
aperiodic background fitted over 1-40 Hz with 6-16 Hz censored (6-25 Hz as a
sensitivity check); alpha band 8-15 Hz (as in the paper): total, background
b and periodic a = total - b, also from odd and even epochs.

Reported (baseline -> moderate, paired): the paper's measure, relative alpha
(8-15 Hz power over 0.5-40 Hz power), frontal minus posterior; the same
anteriorization index for periodic alpha as a function of lambda, with the
crossover lambda*; posterior and frontal alpha separately; all overall and
for the drowsy and responsive groups.

Usage: python chennu_analysis.py [--data DIR]
Writes results/chennu_spectra.csv (per participant and level) and
results/chennu_lambda.csv (group results).
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambda_curve import band_power, effect_curve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROIS = {"posterior": ["O1", "Oz", "O2", "Pz", "P3", "P4"], "frontal": ["Fz", "F3", "F4"]}
ALPHA = (8.0, 15.0)
LEVELS = {1: "baseline", 2: "mild", 3: "moderate", 4: "recovery"}


def spectra(path):
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ep = mne.io.read_epochs_eeglab(path, verbose="ERROR")
    X = ep.get_data() * 1e6                              # epochs x ch x samples, uV
    sr = ep.info["sfreq"]
    nper = int(4 * sr)
    win = np.hanning(nper + 1)[:nper]
    scale = 1.0 / (sr * np.sum(win ** 2))
    f = np.fft.rfftfreq(nper, 1 / sr)
    per_epoch = []
    for e in X:
        ps = []
        for s in range(0, e.shape[1] - nper + 1, nper // 2):
            seg = e[:, s:s + nper]
            seg = seg - seg.mean(1, keepdims=True)
            p = np.abs(np.fft.rfft(seg * win, axis=1)) ** 2 * scale * 2
            ps.append(p)
        per_epoch.append(np.mean(ps, 0))
    per_epoch = np.array(per_epoch)                      # epochs x ch x f
    out = {}
    for roi, names in ROIS.items():
        ix = [ep.ch_names.index(c) for c in names if c in ep.ch_names]
        R = per_epoch[:, ix].mean(1)
        out[roi] = dict(full=R.mean(0), odd=R[0::2].mean(0), even=R[1::2].mean(0))
    return f, out, X.shape[0]


def drowsy_group(info, n=40):
    """The paper's rule: drowsy if the 95% (Clopper-Pearson) interval of the
    hit probability at moderate sedation lies wholly below the baseline one."""
    g = {}
    for sub, d in info.groupby("subject"):
        h0 = int(d.loc[d.level == 1, "hits"].iloc[0])
        h3 = int(d.loc[d.level == 3, "hits"].iloc[0])
        ci0 = stats.binomtest(h0, n).proportion_ci(method="exact")
        ci3 = stats.binomtest(h3, n).proportion_ci(method="exact")
        g[sub] = ci3.high < ci0.low
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"),
                                                   "chennu2016", "Sedation-RestingState"))
    a = ap.parse_args()
    m = sio.loadmat(os.path.join(a.data, "datainfo.mat"), squeeze_me=True)["datainfo"]
    info = pd.DataFrame(m, columns=["file", "level", "conc", "rt", "hits"])
    info["subject"] = info.file.str[:2]
    for c in ("level", "conc", "rt", "hits"):
        info[c] = pd.to_numeric(info[c])
    drowsy = drowsy_group(info)
    print(f"drowsy: {sum(drowsy.values())} of {len(drowsy)}")
    rows = []
    for r in info.itertuples():
        path = os.path.join(a.data, r.file + ".set")
        f, S, n_ep = spectra(path)
        for roi, parts in S.items():
            for split, P in parts.items():
                rel = P[(f >= ALPHA[0]) & (f <= ALPHA[1])].mean() / P[(f >= 0.5) & (f <= 40)].mean()
                for cname, censor in (("censor 6-16", ((6.0, 16.0),)), ("censor 6-25", ((6.0, 25.0),))):
                    bp = band_power(P, f, ALPHA, fit_range=(1.0, 40.0), censor=censor, model="fixed")
                    rows.append(dict(subject=r.subject, level=r.level, conc=r.conc, hits=r.hits,
                                     rt=r.rt, drowsy=drowsy[r.subject], roi=roi, split=split,
                                     window=cname, n_epochs=n_ep, rel_alpha=rel, **bp))
        print(f"  {r.file}: {n_ep} epochs", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "chennu_spectra.csv"), index=False)

    out = []
    for win in ("censor 6-16", "censor 6-25"):
        F = T[(T.split == "full") & (T.window == win)]
        w = F.pivot_table(index=["subject", "drowsy"], columns=["roi", "level"],
                          values=["tot", "b", "a", "rel_alpha"])
        w = w.reset_index()
        for grp, sel in (("all", slice(None)), ("drowsy", w.drowsy.to_numpy()),
                         ("responsive", ~w.drowsy.to_numpy())):
            W = w.loc[sel] if not isinstance(sel, slice) else w
            def col(v, roi, lev):
                return W[(v, roi, lev)].to_numpy(float)
            # paper's measure: relative alpha, frontal minus posterior, moderate - baseline
            ant_rel = ((col("rel_alpha", "frontal", 3) - col("rel_alpha", "posterior", 3))
                       - (col("rel_alpha", "frontal", 1) - col("rel_alpha", "posterior", 1)))
            t = stats.ttest_1samp(ant_rel, 0)
            out.append(dict(window=win, group=grp, measure="relative alpha anteriorization",
                            n=len(ant_rel), est=float(np.mean(ant_rel)), p=float(t.pvalue)))
            with np.errstate(invalid="ignore", divide="ignore"):
                la = {(roi, lev): np.log(np.where(col("a", roi, lev) > 0, col("a", roi, lev), np.nan))
                      for roi in ROIS for lev in (1, 3)}
                lb = {(roi, lev): np.log(col("b", roi, lev)) for roi in ROIS for lev in (1, 3)}
            contrasts = {
                "anteriorization (frontal - posterior)":
                    ((la[("frontal", 3)] - la[("posterior", 3)]) - (la[("frontal", 1)] - la[("posterior", 1)]),
                     (lb[("frontal", 3)] - lb[("posterior", 3)]) - (lb[("frontal", 1)] - lb[("posterior", 1)])),
                "posterior alpha": (la[("posterior", 3)] - la[("posterior", 1)],
                                    lb[("posterior", 3)] - lb[("posterior", 1)]),
                "frontal alpha": (la[("frontal", 3)] - la[("frontal", 1)],
                                  lb[("frontal", 3)] - lb[("frontal", 1)]),
            }
            for name, (dA, dB) in contrasts.items():
                r = effect_curve(dA, dB, rng=np.random.default_rng(0))
                c0 = r["curve"][0]
                c1 = r["curve"][-1]
                out.append(dict(window=win, group=grp, measure=name, n=r["n"],
                                s_a=r["s_a"], s_b=r["s_b"], lam_star=r["lam_star"],
                                hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1],
                                lam0=c0[0], lam0_lo=c0[1], lam0_hi=c0[2],
                                lam1=c1[0], lam1_lo=c1[1], lam1_hi=c1[2],
                                p_cross_in_01=r["p_cross_in_01"]))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(RES, "chennu_lambda.csv"), index=False)
    pd.set_option("display.width", 200)
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
