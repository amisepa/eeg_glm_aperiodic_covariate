"""Collect the re-tested published claims into one table.

Reads the group-level outputs of hbn_age_alpha_robust.py, aging_lambda.py,
vitaldb_lambda.py, chennu_analysis.py and brake_analysis.py and writes
results/breadth_summary.csv: for each claim, the effect under lambda = 0 and
lambda = 1 with 95% intervals, the crossover lambda* with its
Bayesian-bootstrap HDI, the posterior share of crossovers inside [0, 1],
and a verdict: "holds under both" (both intervals exclude zero, same sign),
"reverses" (both exclude zero, opposite signs), "depends on lambda" (only
one excludes zero) or "null under both".

Usage: python breadth_summary.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def row(domain, claim, source, dataset, n, r, unit):
    """One claim. lam_star_bounded is False when the interval of s_b, the effect
    on ln b, includes 0: the crossover is then unbounded and its HDI is not
    reported."""
    bounded = bool(r.get("lam_star_bounded", True))
    return dict(domain=domain, claim=claim, source=source, dataset=dataset, n=int(n),
                lam0=r["lam0"], lam0_lo=r["lam0_lo"], lam0_hi=r["lam0_hi"],
                lam1=r["lam1"], lam1_lo=r["lam1_lo"], lam1_hi=r["lam1_hi"],
                s_b_lo=r.get("s_b_lo", np.nan), s_b_hi=r.get("s_b_hi", np.nan),
                lam_star_bounded=bounded, lam_star=r["lam_star"],
                hdi_lo=r["hdi_lo"] if bounded else np.nan,
                hdi_hi=r["hdi_hi"] if bounded else np.nan,
                p_cross_in_01=r["p_cross_in_01"], unit=unit)


def hbn_p01():
    """Share of Bayesian-bootstrap crossovers inside [0, 1] for the HBN age
    slopes, recomputed from the local per-participant files (not in the repo;
    needs results/hbn_kp_fits.csv and the PSD files in $HBN_OUT)."""
    import glob
    import sys
    sys.path.insert(0, HERE)
    from lambda_curve import effect_curve
    try:
        k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
        Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    except FileNotFoundError:
        return {}
    ages = {}
    for p in glob.glob(os.path.join(os.environ.get("HBN_OUT", "hbn_psd"), "*.npz")):
        d = np.load(p, allow_pickle=True)
        ages[str(d["subject"])] = float(d["age"])
    if not ages:
        return {}
    k = k[(k.model == "fixed") & (k.window == "censor 6-16") & (k.band == "alpha") & (k.split == "full")]
    w = k.pivot_table(index="subject", columns="cond", values=["a", "b"])
    w.columns = [f"{v}_{c}" for v, c in w.columns]
    w["age"] = pd.Series(ages)
    w = w.loc[w.index.intersection(Q.index[Q.qc_ok])]
    out = {}
    for cond in ("ec", "eo"):
        d = w[w[f"a_{cond}"] > 0].dropna(subset=["age"])
        r = effect_curve(np.log(d[f"a_{cond}"]), np.log(d[f"b_{cond}"]), x=d.age,
                         rng=np.random.default_rng(0))
        out[cond] = dict(p_cross_in_01=r["p_cross_in_01"], s_b_lo=r["s_b_ci"][0],
                         s_b_hi=r["s_b_ci"][1], lam_star_bounded=r["lam_star_bounded"])
    return out


def main():
    out = []
    # HBN children (hbn_age_alpha_robust.py): curve endpoints from the robust table
    R = pd.read_csv(os.path.join(RES, "hbn_age_alpha_robust.csv"))
    C = pd.read_csv(os.path.join(RES, "hbn_age_alpha_crossover.csv"))
    p01s = hbn_p01()
    for cond, label in (("ec", "eyes closed"), ("eo", "eyes open")):
        g = R[(R["sample"] == "qc") & (R.model == "fixed") & (R.cond == cond) & (R.estimator == "ols")]
        c = C[(C["sample"] == "qc") & (C.model == "fixed") & (C.cond == cond)].iloc[0]
        r0, r1 = g[g.lam == 0].iloc[0], g[g.lam == 1].iloc[0]
        extra = p01s.get(cond, dict(p_cross_in_01=np.nan))
        out.append(row("development", f"alpha changes with age, {label}", "Tröndle et al. 2022",
                       "HBN, 5-22 y", r0.n,
                       dict(lam0=r0.est, lam0_lo=r0.lo, lam0_hi=r0.hi, lam1=r1.est, lam1_lo=r1.lo,
                            lam1_hi=r1.hi, lam_star=c.lam_star, hdi_lo=c.hdi_lo, hdi_hi=c.hdi_hi,
                            **extra), "ln per year"))
    A = pd.read_csv(os.path.join(RES, "aging_lambda.csv"))
    F = A[A.model == "fixed"]
    for contrast, claim, source, dataset, unit in (
            ("dortmund-cross ec_pre", "alpha falls with adult age, eyes closed",
             "Politanskaia et al. 2026; Yang et al. 2025", "Dortmund, 20-70 y", "ln per year"),
            ("dortmund-cross eo_pre", "alpha falls with adult age, eyes open", "",
             "Dortmund, 20-70 y", "ln per year"),
            ("lemon-group ec (older - younger)", "alpha lower in older adults, eyes closed",
             "Tröndle et al. 2023; Wilson et al. 2022", "LEMON, 20-35 vs 59-77 y", "ln")):
        r = F[F.contrast == contrast]
        if len(r):
            out.append(row("adult aging", claim, source, dataset, r.n.iloc[0], r.iloc[0], unit))
    V = pd.read_csv(os.path.join(RES, "vitaldb_lambda.csv"))
    for agent in ("propofol", "sevoflurane"):
        r = V[(V.agent == agent) & (V.model == "fixed") & (V.band == "iaf") & (V.adjust == "+ dose")]
        out.append(row("anaesthesia", f"frontal alpha falls with age under {agent}",
                       "Purdon et al. 2015; Boncompte et al. 2024", "VitalDB, 18-94 y",
                       r.n.iloc[0], r.iloc[0], "ln per decade"))
    X = pd.read_csv(os.path.join(RES, "chennu_lambda.csv"))
    X = X[X.window == "censor 6-16"]
    for grp, meas, claim in (("drowsy", "anteriorization (frontal - posterior)",
                              "alpha shifts frontally in drowsy participants"),
                             ("all", "frontal alpha", "frontal alpha rises with propofol sedation"),
                             ("all", "posterior alpha", "posterior alpha falls with propofol sedation")):
        r = X[(X.group == grp) & (X.measure == meas)]
        out.append(row("sedation", claim, "Chennu et al. 2016", "propofol, baseline vs moderate",
                       r.n.iloc[0], r.iloc[0], "ln"))
    B = pd.read_csv(os.path.join(RES, "brake_lambda.csv"))
    if "baseline" in B:
        B = B[B.baseline == ("pre-infusion" if (B.baseline == "pre-infusion").any()
                             else "first 60 s")]
    for band, win, claim in (("delta", "pre-LOC", "delta does not rise before loss of consciousness"),
                             ("alpha", "pre-LOC", "alpha rises before loss of consciousness"),
                             ("beta", "pre-LOC", "beta rises before loss of consciousness")):
        r = B[(B.band == band) & (B.window == win)]
        out.append(row("anaesthesia", claim, "Brake et al. 2024", "propofol induction, Cz",
                       r.n.iloc[0], r.iloc[0], "ln"))
    O = pd.DataFrame(out)
    sig0 = np.sign(O.lam0_lo) == np.sign(O.lam0_hi)
    sig1 = np.sign(O.lam1_lo) == np.sign(O.lam1_hi)
    same = np.sign(O.lam0) == np.sign(O.lam1)
    O["verdict"] = np.where(sig0 & sig1 & same, "holds under both",
                            np.where(sig0 & sig1 & ~same, "reverses",
                                     np.where(sig0 | sig1, "depends on lambda", "null under both")))
    O.to_csv(os.path.join(RES, "breadth_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(O[["domain", "claim", "dataset", "n", "lam0", "lam1", "lam_star", "hdi_lo", "hdi_hi",
             "p_cross_in_01", "verdict"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
