"""Fit-free version of the topographic test.

Correlates the group map of the eyes-closed minus eyes-open change in log
power in a flank band (2-4 Hz or 30-40 Hz) with the map of the change in log
power at IAF +/- 2 Hz, by age group. No aperiodic model is fitted, so alpha
cannot leak into a fitted background; the flank bands remain sensitive to
ocular (low) and muscle (high) activity.

Usage: python hbn_topography_flanks.py [--psd-dir DIR] [--workers 12]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_topography import flank_changes, ROI

TEMPORAL = ["E116", "E117", "E111", "E123", "E34", "E35", "E40", "E109"]


def _one(path):
    try:
        d = np.load(path, allow_pickle=True)
        ch = [str(c) for c in d["ch_names"]]
        r = flank_changes(d["freqs"].astype(float), d["eo"], d["ec"], ch)
        if r is None:
            return None
        return ch, r, float(d["age"])
    except Exception:
        return None


def group_r(DF, DA, rng, nboot=1000):
    mF, mA = np.nanmean(DF, 0), np.nanmean(DA, 0)
    r = np.corrcoef(mF, mA)[0, 1]
    n = DF.shape[0]
    bs = []
    for _ in range(nboot):
        s = rng.integers(0, n, n)
        bs.append(np.corrcoef(np.nanmean(DF[s], 0), np.nanmean(DA[s], 0))[0, 1])
    return r, *np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap(_one, files, chunksize=8) if r is not None]
    chan = res[0][0]
    res = [r for r in res if r[0] == chan]
    bands = list(res[0][1][0])
    DF = {k: np.array([r[1][0][k] for r in res]) for k in bands}
    DA = np.array([r[1][1] for r in res])
    age = np.array([r[2] for r in res])
    post = [chan.index(c) for c in ROI]
    temp = [chan.index(c) for c in TEMPORAL]
    print(f"{len(res)} subjects x {len(chan)} channels")

    rng = np.random.default_rng(0)
    ok = np.isfinite(age)
    edges = np.nanpercentile(age[ok], [0, 100 / 3, 200 / 3, 100])
    groups = [("all", np.ones(age.size, bool))]
    for k in range(3):
        hi = age <= edges[k + 1] if k == 2 else age < edges[k + 1]
        groups.append((f"{edges[k]:.1f}-{edges[k+1]:.1f}", ok & (age >= edges[k]) & hi))
    groups.append(("16-21.9", ok & (age >= 16)))

    print("\nspatial r(map d_log P(flank), map d_log P(IAF+/-2)), bootstrap 95% CI;"
          "\nmean d_log P at posterior ROI / temporal sites")
    rows = []
    for k in bands:
        print(f"\n  flank {k}:")
        for label, sel in groups:
            r, lo, hi = group_r(DF[k][sel], DA[sel], rng)
            rows.append(dict(flank=k, group=label, n=int(sel.sum()), r=r, lo=lo,
                             hi=hi,
                             dlogp_post=float(np.nanmean(DF[k][sel][:, post])),
                             dlogp_temp=float(np.nanmean(DF[k][sel][:, temp]))))
            print(f"    {label:>10} n={sel.sum():>4}  r = {r:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"   post {np.nanmean(DF[k][sel][:, post]):+.3f}"
                  f"  temp {np.nanmean(DF[k][sel][:, temp]):+.3f}")
    print(f"\n  alpha band d_log P: post {np.nanmean(DA[:, post]):+.3f}, "
          f"temp {np.nanmean(DA[:, temp]):+.3f}")
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "results", "hbn_topography_flanks.csv"),
                              index=False)


if __name__ == "__main__":
    main()
