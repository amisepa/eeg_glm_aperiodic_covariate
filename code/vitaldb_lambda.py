"""Age and frontal alpha under general anaesthesia (VitalDB): does the
published age decline depend on lambda?

Input: vitaldb_extract.py output (two frontal BIS channels, spectra over
kept 4-s segments during surgery). Cases are classified by the medians over
kept segments: propofol (effect-site > 1 ug/ml, end-tidal sevoflurane < 0.3
kPa and desflurane < 0.5 kPa) or sevoflurane (end-tidal > 1 kPa, propofol
< 0.5 ug/ml); mixed and desflurane cases are left out. Kept: >= 150 clean
segments and median BIS 20-65.

Mean spectrum of the two channels; aperiodic background over 2-40 Hz with
6-16 Hz censored (power law; knee+plateau as a check); alpha = individual
peak frequency +/- 2 Hz (search 7-14 Hz) and, as in Purdon et al. 2015,
8-12 Hz; a = total - b.

Reported per agent: the age slope (per decade) of ln a - lambda ln b with
lambda*, adjusted for sex and remifentanil, with and without the anaesthetic
dose (propofol effect-site concentration, or end-tidal sevoflurane); and the
published measures: total alpha in dB (Purdon 2015), relative alpha
(alpha / 1-40 Hz power; Kreuzer 2020) and specparam-flattened alpha
(lambda = 1; Boncompte 2024).

Usage: python vitaldb_lambda.py [--data DIR] [--workers 12]
Writes results/vitaldb_cases.csv (per case; local) and results/vitaldb_lambda.csv.
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
from ap_models import ap_eval, fit_aperiodic
from lambda_curve import band_power, effect_curve, peak_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIT = (2.0, 40.0)
CENSOR = ((6.0, 16.0),)
warnings.filterwarnings("ignore")


def one(path):
    d = np.load(path, allow_pickle=True)
    f = d["freqs"].astype(float)
    P = d["full"].astype(float).mean(0)
    row = {k: (float(d[k]) if d[k].dtype.kind in "fi" else str(d[k]))
           for k in ("caseid", "age", "sex", "n_seg", "window_s", "bis", "sqi", "emg", "ppf_ce",
                     "rftn20_ce", "rftn50_ce", "sevo_et", "des_et", "mac")}
    iaf = peak_frequency(P, f, search=(7.0, 14.0))
    row["iaf"] = iaf
    cf = iaf if np.isfinite(iaf) else 10.0
    for model in ("fixed", "knee_plateau"):
        for bname, band in (("iaf", (cf - 2, cf + 2)), ("8-12", (8.0, 12.0))):
            bp = band_power(P, f, band, FIT, CENSOR, model)
            if bp is None:
                continue
            for k in ("tot", "b", "a"):
                row[f"{k}_{bname}_{model}"] = bp[k]
    sel = (f >= FIT[0]) & (f <= FIT[1])
    keep = ~((f[sel] >= 6) & (f[sel] <= 16))
    fit = fit_aperiodic(np.clip(P[sel], 1e-12, None), f[sel], keep, "fixed")
    if fit is not None:
        L = ap_eval(fit["theta"], f[sel], "fixed")
        ff = f[sel]
        flat = np.log10(P[sel]) - np.log10(L)
        row["flat_alpha"] = float(np.mean(flat[(ff >= cf - 2) & (ff <= cf + 2)]))
        row["exponent"] = fit["exponent"]
    row["rel_alpha"] = float(P[(f >= 8) & (f <= 12)].sum() / P[(f >= 1) & (f <= 40)].sum())
    return row


def ols(y, X):
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    r = sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="HC3")
    return float(r.params[1]), float(r.conf_int()[1][0]), float(r.conf_int()[1][1]), \
        float(r.pvalues[1]), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("VITALDB_OUT", "vitaldb_psd"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data, "case*.npz")))
    with Pool(a.workers) as pool:
        C = pd.DataFrame(pool.map(one, files, chunksize=16))
    C["rftn_ce"] = C[["rftn20_ce", "rftn50_ce"]].max(axis=1, skipna=True).fillna(0.0)
    prop = (C.ppf_ce > 1.0) & (C.sevo_et.fillna(0) < 0.3) & (C.des_et.fillna(0) < 0.5)
    sevo = (C.sevo_et > 1.0) & (C.ppf_ce.fillna(0) < 0.5) & (C.des_et.fillna(0) < 0.5)
    C["agent"] = np.where(prop, "propofol", np.where(sevo, "sevoflurane", "other"))
    C["keep"] = (C.n_seg >= 150) & C.bis.between(20, 65) & (C.agent != "other")
    C.to_csv(os.path.join(RES, "vitaldb_cases.csv"), index=False)
    print(C.groupby("agent").agg(n=("caseid", "size"), kept=("keep", "sum"),
                                 age_med=("age", "median")))
    out, pub = [], []
    for agent, dose in (("propofol", "ppf_ce"), ("sevoflurane", "sevo_et")):
        D = C[C.keep & (C.agent == agent)]
        male = (D.sex == "M").astype(float).to_numpy()
        base_cov = np.column_stack([male, D.rftn_ce.to_numpy()])
        dose_cov = np.column_stack([base_cov, D[dose].to_numpy()])
        for model in ("fixed", "knee_plateau"):
            for bname in ("iaf", "8-12"):
                A, B = D[f"a_{bname}_{model}"], D[f"b_{bname}_{model}"]
                ok = (A > 0).to_numpy()
                for adj, cov in (("sex + remifentanil", base_cov),
                                 ("+ dose", dose_cov)):
                    r = effect_curve(np.log(A[ok]), np.log(B[ok]), x=D.age.to_numpy()[ok] / 10,
                                     covariates=cov[ok], rng=np.random.default_rng(0))
                    out.append(dict(agent=agent, model=model, band=bname, adjust=adj, n=r["n"],
                                    excluded_a_le_0=int((~ok).sum()), s_a=r["s_a"], s_b=r["s_b"],
                                    s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
                                    lam_star_bounded=r["lam_star_bounded"],
                                    lam_star=r["lam_star"], hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1],
                                    p_cross_in_01=r["p_cross_in_01"],
                                    lam0=r["curve"][0][0], lam0_lo=r["curve"][0][1],
                                    lam0_hi=r["curve"][0][2], lam1=r["curve"][-1][0],
                                    lam1_lo=r["curve"][-1][1], lam1_hi=r["curve"][-1][2]))
        X = np.column_stack([D.age.to_numpy() / 10, base_cov])
        for meas, y in (("total alpha 8-12 Hz (dB)", 10 * np.log10(D["tot_8-12_fixed"])),
                        ("relative alpha 8-12 / 1-40 Hz", D.rel_alpha),
                        ("specparam flattened alpha (log10)", D.flat_alpha),
                        ("background at alpha (dB)", 10 * np.log10(D["b_iaf_fixed"])),
                        ("aperiodic exponent", D.exponent)):
            s, lo, hi, p, n = ols(np.asarray(y, float), X)
            pub.append(dict(agent=agent, measure=meas, per_decade=s, lo=lo, hi=hi, p=p, n=n))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(RES, "vitaldb_lambda.csv"), index=False)
    pd.DataFrame(pub).to_csv(os.path.join(RES, "vitaldb_published.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))
    print(pd.DataFrame(pub).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
