"""Collect every attempt to estimate lambda into one table.

- HBN eyes closed vs eyes open, log-free two-condition estimator
  (lambda_gmm.bootstrap, root-tracking) for the four specifications of
  hbn_kp.py; computed here from results/hbn_kp_fits.csv (local) and saved to
  results/hbn_lambda_gmm.csv.
- ds003690 within-session eyes-open fluctuations (ds003690_lambda.py), with
  its data-matched calibration.
- Dortmund within-session fluctuations, eyes closed and open, before and
  after the task battery (dortmund_levels.py), two background-fit windows,
  with calibration.
- Chennu graded propofol, baseline vs moderate (chennu_lambda_gmm.csv).
- Test-retest: SRM (later session) and Dortmund (5 years), log-free
  estimator with sessions as conditions (srm_lambda.py).
- Intracranial within-session fluctuations (ieeg_lambda.py), with its
  data-matched calibration.

Writes results/identification_summary.csv.

Usage: python identification_summary.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
import lambda_gmm as G


def hbn_gmm():
    path = os.path.join(RES, "hbn_lambda_gmm.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    k = k[k.subject.isin(Q.index[Q.qc_ok]) & (k.band == "alpha") & (k.split != "full")]
    rows = []
    for model in ("fixed", "knee_plateau"):
        for win in ("censor 6-16", "flanks 3-6, 26-36"):
            g = k[(k.model == model) & (k.window == win)]
            w = g.pivot_table(index="subject", columns=["cond", "split"], values=["tot", "b"]).dropna()
            arr = [w[v][c][["odd", "even"]].to_numpy() for v, c in
                   (("tot", "eo"), ("tot", "ec"), ("b", "eo"), ("b", "ec"))]
            r = G.bootstrap(*arr, nboot=300, rng=np.random.default_rng(0))
            rows.append(dict(model=model, window=win, n=len(w), lam=r["lam"], lo=r["ci"][0],
                             hi=r["ci"][1], fail=r["boot_fail"]))
    D = pd.DataFrame(rows)
    D.to_csv(path, index=False)
    return D


def main():
    out = []
    H = hbn_gmm()
    for r in H.itertuples():
        out.append(dict(design="between conditions: eyes closed vs open", dataset="HBN, 5-22 y",
                        spec=f"{'power law' if r.model == 'fixed' else 'knee + plateau'}, {r.window}",
                        n=r.n, lam=r.lam, lo=r.lo, hi=r.hi, calibration=""))
    d = pd.read_csv(os.path.join(RES, "ds003690_lambda.csv"))
    for r in d[(d.covariates.isin(["none", "pupil"])) & (d.group == "all")].itertuples():
        out.append(dict(design="within session: epoch fluctuations, eyes open",
                        dataset="ds003690, 20-75 y", spec=f"censor 6-16, covariates: {r.covariates}",
                        n=r.n_units, lam=r.lam, lo=r.lo, hi=r.hi,
                        calibration="0 -> -0.06, 0.5 -> 0.43, 1 -> 0.90"))
    for c in (16, 26):
        p = os.path.join(RES, f"dortmund_levels_c{c}.csv")
        if not os.path.exists(p):
            continue
        for r in pd.read_csv(p).itertuples():
            state = {"ec": "eyes closed", "eo": "eyes open"}[r.cond[:2]]
            when = "before" if r.cond.endswith("pre") else "after"
            out.append(dict(design=f"within session: segment fluctuations, {state}",
                            dataset="Dortmund, 20-70 y", spec=f"censor 6-{c}, {when} 2-h tasks",
                            n=r.n_units, lam=r.lam, lo=r.lo, hi=r.hi,
                            calibration="0 -> 0.13, 0.5 -> 0.70, 1 -> 1.23 (censor 6-16)"))
    x = pd.read_csv(os.path.join(RES, "chennu_lambda_gmm.csv"))
    for r in x[x.contrast == "baseline -> moderate"].itertuples():
        out.append(dict(design="between doses: propofol baseline vs moderate",
                        dataset="Chennu 2016", spec=f"{r.roi}, {r.window}", n=r.n, lam=r.lam,
                        lo=r.lo, hi=r.hi, calibration=""))
    p = os.path.join(RES, "srm_lambda.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        s = s[s.analysis.str.contains("log-free") & (s.model == "fixed")]
        for r in s.itertuples():
            srm = r.analysis.startswith("srm")
            out.append(dict(design="between sessions: test-retest",
                            dataset="SRM, 17-71 y" if srm else "Dortmund, 20-70 y",
                            spec=("SRM, eyes closed, later session" if srm else
                                  f"Dortmund, {'eyes closed' if 'ec_pre' in r.analysis else 'eyes open'}"
                                  ", 5 years"),
                            n=r.n, lam=r.lam_gmm, lo=r.gmm_lo, hi=r.gmm_hi, calibration=""))
    p = os.path.join(RES, "ieeg_lambda.csv")
    if os.path.exists(p):
        cal = []
        for L in ("0.0", "0.5", "1.0"):
            q = os.path.join(RES, f"ieeg_lambda_sim{L}.csv")
            if os.path.exists(q):
                c = pd.read_csv(q)
                c = c[c.subset == "all"]
                cal.append(f"{float(L):g} -> {c.lam.iloc[0]:.2f}")
        for r in pd.read_csv(p).query("covariates in ['none', 'EOG + EMG']").itertuples():
            out.append(dict(design="within session: epoch fluctuations, intracranial",
                            dataset="ds003688 iEEG", n=r.n_channels, lam=r.lam, lo=r.lo, hi=r.hi,
                            spec=f"{r.subset} channels, covariates: {r.covariates}",
                            calibration=", ".join(cal)))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(RES, "identification_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
