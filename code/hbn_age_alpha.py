"""Does the developmental trajectory of alpha depend on the separation rule?

For each HBN participant and condition, takes posterior-ROI alpha-band power
(results/hbn_kp_fits.csv: total power, fitted aperiodic power b and periodic
power a = total - b) and regresses on age:

    total      log P_alpha                   (no separation)
    lambda=0   log a                         (additive: subtract in linear power)
    lambda=1   log(a / b)                    (multiplicative: subtract in log power)
    lambda=.5  log a - 0.5 log b
    background log b

and the same for the eyes-closed minus eyes-open change (alpha reactivity).
Reports Spearman rho with age and the OLS slope per year with a subject
bootstrap 95% CI.

Ages are read from the per-subject PSD files ($HBN_OUT, default ./hbn_psd).

Usage: python hbn_age_alpha.py [--model fixed|knee_plateau]
       [--window "censor 6-16"] [--psd-dir DIR]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))


def ages(psd_dir):
    rows = []
    for p in glob.glob(os.path.join(psd_dir, "*.npz")):
        d = np.load(p, allow_pickle=True)
        rows.append((str(d["subject"]), float(d["age"])))
    return pd.DataFrame(rows, columns=["subject", "age"]).drop_duplicates("subject")


def slope_ci(x, y, rng, nboot=2000):
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    n = x.size
    bs = []
    for _ in range(nboot):
        s = rng.integers(0, n, n)
        bs.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0][1])
    return b, *np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fixed")
    ap.add_argument("--window", default="censor 6-16")
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    a = ap.parse_args()

    k = pd.read_csv(os.path.join(HERE, "..", "results", "hbn_kp_fits.csv"))
    k = k[(k.model == a.model) & (k.window == a.window) & (k.band == "alpha")
          & (k.split == "full")]
    w = k.pivot_table(index="subject", columns="cond", values=["a", "b", "tot"])
    w.columns = [f"{v}_{c}" for v, c in w.columns]
    w = w.join(ages(a.psd_dir).set_index("subject"), how="inner").dropna()

    rng = np.random.default_rng(0)
    print(f"model {a.model}, window {a.window}; n = {len(w)} with age "
          f"{w.age.min():.1f}-{w.age.max():.1f}")
    print("\nlog-scale measure vs age: Spearman rho, OLS slope per year "
          "[95% CI], n (subjects with a > 0)")
    out = []
    for cond in ("eo", "ec"):
        tot, aa, bb = w[f"tot_{cond}"], w[f"a_{cond}"], w[f"b_{cond}"]
        ok = aa > 0
        meas = {
            "total": np.log(tot),
            "lambda=0": np.log(aa.where(ok)),
            "lambda=0.5": np.log(aa.where(ok)) - 0.5 * np.log(bb),
            "lambda=1": np.log((aa / bb).where(ok)),
            "background": np.log(bb),
        }
        print(f"\n  {cond.upper()}")
        for name, y in meas.items():
            m = np.isfinite(y)
            rho = spearmanr(w.age[m], y[m])[0]
            s, lo, hi = slope_ci(w.age[m].to_numpy(), y[m].to_numpy(), rng)
            print(f"    {name:>11}: rho = {rho:+.3f}, slope {s:+.4f}/y "
                  f"[{lo:+.4f}, {hi:+.4f}], n = {m.sum()}")
            out.append(dict(cond=cond, measure=name, rho=rho, slope=s,
                            lo=lo, hi=hi, n=int(m.sum())))

    # alpha reactivity (EC - EO) vs age
    ok = (w.a_ec > 0) & (w.a_eo > 0)
    r0 = np.log(w.a_ec / w.a_eo)
    r1 = np.log((w.a_ec / w.b_ec) / (w.a_eo / w.b_eo))
    rt = np.log(w.tot_ec / w.tot_eo)
    print("\n  EC - EO reactivity")
    for name, y in (("total", rt), ("lambda=0", r0.where(ok)),
                    ("lambda=1", r1.where(ok))):
        m = np.isfinite(y)
        rho = spearmanr(w.age[m], y[m])[0]
        s, lo, hi = slope_ci(w.age[m].to_numpy(), y[m].to_numpy(), rng)
        print(f"    {name:>11}: rho = {rho:+.3f}, slope {s:+.4f}/y "
              f"[{lo:+.4f}, {hi:+.4f}], n = {m.sum()}")
        out.append(dict(cond="ec-eo", measure=name, rho=rho, slope=s,
                        lo=lo, hi=hi, n=int(m.sum())))
    tag = f"{a.model}_{a.window.replace(' ', '').replace(',', '_')}"
    pd.DataFrame(out).to_csv(os.path.join(HERE, "..", "results",
                                          f"hbn_age_alpha_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
