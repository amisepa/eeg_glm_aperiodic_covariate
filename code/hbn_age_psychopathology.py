"""Does psychopathology explain the lambda-dependence of the HBN alpha-age slope?

HBN is a clinical community sample. Its release participants.tsv files carry
bifactor scores from the Child Behavior Checklist (p_factor, attention,
internalizing, externalizing) and handedness (ehq_total); diagnoses and
medication are only in the controlled-access phenotype and are not used.

For each background model and condition, the age slope of
ln a - lambda ln b and the crossover lambda* (lambda_curve.effect_curve) are
estimated in the quality-controlled sample with phenotype scores:
  base          age only
  covar         + sex, release, ln(number of segments)   (as in the paper)
  +p            covar + p_factor
  +cbcl         covar + p_factor, attention, internalizing, externalizing
  +cbcl+ehq     covar + the four scores + handedness
  p<=median, p>median   covar, in each half of the p_factor distribution

Writes results/hbn_age_psychopathology.csv.

Usage: python hbn_age_psychopathology.py [--psd-dir DIR] [--pheno-dir DIR] [--nboot 2000]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hbn_age_alpha_robust import RES, meta, wide
from lambda_curve import effect_curve

SCORES = ["p_factor", "attention", "internalizing", "externalizing"]


def phenotype(pheno_dir):
    t = pd.concat([pd.read_csv(p, sep="\t") for p in
                   sorted(glob.glob(os.path.join(pheno_dir, "*_participants.tsv")))])
    t = t.drop_duplicates("participant_id").set_index("participant_id")
    for c in SCORES + ["ehq_total"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t[SCORES + ["ehq_total"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--pheno-dir", default=os.path.join(
        os.environ.get("EEG_DATA", "eeg_data"), "hbn_participants"))
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    M = meta(a.psd_dir)
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    P = phenotype(a.pheno_dir)
    rows = []
    for model in ("fixed", "knee_plateau"):
        W = wide(model).join(M, how="inner").join(P, how="inner")
        W = W.loc[W.index.intersection(Q.index[Q.qc_ok])]
        for cond in ("ec", "eo"):
            d = W[(W[f"a_{cond}_full"] > 0) & np.isfinite(W[f"b_{cond}_full"])
                  & W[SCORES + ["ehq_total"]].notna().all(1)].copy()
            la, lb = np.log(d[f"a_{cond}_full"]), np.log(d[f"b_{cond}_full"])
            base = pd.get_dummies(d[["sex", "release"]], drop_first=True, dtype=float)
            base["lnseg"] = np.log(d[f"n_seg_{cond}"])
            med = d.p_factor.median()
            specs = {"base": (None, slice(None)),
                     "covar": (base, slice(None)),
                     "+p": (base.join(d[["p_factor"]]), slice(None)),
                     "+cbcl": (base.join(d[SCORES]), slice(None)),
                     "+cbcl+ehq": (base.join(d[SCORES + ["ehq_total"]]), slice(None)),
                     "p<=median": (base, (d.p_factor <= med).to_numpy()),
                     "p>median": (base, (d.p_factor > med).to_numpy())}
            for spec, (C, m) in specs.items():
                Cm = None if C is None else C.to_numpy()[m]
                if Cm is not None:
                    Cm = Cm[:, Cm.std(0) > 0]            # drop dummies empty in a subsample
                r = effect_curve(la.to_numpy()[m], lb.to_numpy()[m], x=d.age.to_numpy()[m],
                                 covariates=Cm, nboot=a.nboot,
                                 rng=np.random.default_rng(0))
                c = r["curve"]
                rows.append(dict(model=model, cond=cond, spec=spec, n=r["n"],
                                 s0=c[0, 0], s0_lo=c[0, 1], s0_hi=c[0, 2],
                                 s1=c[-1, 0], s1_lo=c[-1, 1], s1_hi=c[-1, 2],
                                 s_b=r["s_b"], lam_star=r["lam_star"],
                                 hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1],
                                 p_cross_in_01=r["p_cross_in_01"]))
                print(f"{model:12s} {cond} {spec:10s} n {r['n']:4d}  lambda 0 "
                      f"{c[0, 0]*100:+.1f} [{c[0, 1]*100:+.1f}, {c[0, 2]*100:+.1f}] %/y  "
                      f"lambda 1 {c[-1, 0]*100:+.1f} [{c[-1, 1]*100:+.1f}, {c[-1, 2]*100:+.1f}]  "
                      f"lambda* {r['lam_star']:.2f} [{r['hdi'][0]:.2f}, {r['hdi'][1]:.2f}]",
                      flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "hbn_age_psychopathology.csv"), index=False)


if __name__ == "__main__":
    main()
