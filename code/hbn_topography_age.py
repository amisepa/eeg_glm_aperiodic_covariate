"""Topographic test split by age tertile.

Reads a results/hbn_topography<tag>.npz file and reports the group-map
spatial correlation (with a subject bootstrap CI), the within-subject spatial
correlation and the across-subject ROI correlation for each age group, plus
a regression of the alpha change on the background change, age and their
interaction.

Usage: python hbn_topography_age.py [tag], e.g. _f4-40
"""
import os
import sys
import numpy as np

ROI = ["E62", "E65", "E70", "E75", "E83", "E90"]


def spatial_tests(DE, DA, DB, ch, rng):
    mE, mA = np.nanmean(DE, 0), np.nanmean(DA, 0)
    good = np.isfinite(mE) & np.isfinite(mA)
    r_group = np.corrcoef(mE[good], mA[good])[0, 1]

    # bootstrap over subjects for the group-map correlation
    boot = []
    n = DE.shape[0]
    for _ in range(1000):
        s = rng.integers(0, n, n)
        e, a = np.nanmean(DE[s], 0), np.nanmean(DA[s], 0)
        g = np.isfinite(e) & np.isfinite(a)
        boot.append(np.corrcoef(e[g], a[g])[0, 1])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    rs = []
    for i in range(n):
        m = np.isfinite(DE[i]) & np.isfinite(DA[i])
        if m.sum() > 60:
            rs.append(np.corrcoef(DE[i][m], DA[i][m])[0, 1])
    rs = np.array(rs)

    ix = [list(ch).index(c) for c in ROI if c in ch]
    a, b = np.nanmean(DA[:, ix], 1), np.nanmean(DB[:, ix], 1)
    m = np.isfinite(a) & np.isfinite(b)
    r_subj = np.corrcoef(b[m], a[m])[0, 1]

    # ratio of posterior to temporal-ish change, as in the main script:
    # where each map peaks
    names = np.array(ch)[good]
    top_a = list(names[np.argsort(-mA[good])[:5]])
    top_e = list(names[np.argsort(-mE[good])[:5]])
    return r_group, lo, hi, np.nanmedian(rs), r_subj, m.sum(), top_a, top_e


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    d = np.load(os.path.join(here, "..", "results", f"hbn_topography{tag}.npz"))
    if "censor" in d.files:
        print(f"censor window: {d['censor']} Hz")
    DE, DA, DB = d["d_exponent"], d["d_log_a"], d["d_log_b"]
    ch, age = list(d["ch_names"]), d["age"]
    ok = np.isfinite(age)
    print(f"subjects with age: {ok.sum()} of {age.size}; "
          f"age {np.nanmin(age):.1f}-{np.nanmax(age):.1f}, "
          f"median {np.nanmedian(age):.1f}")

    rng = np.random.default_rng(0)
    edges = np.nanpercentile(age[ok], [0, 100 / 3, 200 / 3, 100])
    print(f"tertile edges: {np.round(edges, 1)}\n")
    print(f"{'age group':>14} {'n':>5} {'group r':>8} {'95% CI':>17} "
          f"{'within r':>9} {'subj r (ROI)':>13}")
    groups = [(f"{edges[k]:.1f}-{edges[k+1]:.1f}",
               ok & (age >= edges[k]) & (age <= edges[k + 1]) if k == 2
               else ok & (age >= edges[k]) & (age < edges[k + 1]))
              for k in range(3)]
    groups.append(("16-21 (adult-like)", ok & (age >= 16)))
    for label, sel in groups:
        rg, lo, hi, rw, rsub, nsub, ta, te = spatial_tests(
            DE[sel], DA[sel], DB[sel], ch, rng)
        print(f"{label:>14} {sel.sum():>5} {rg:>+8.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] {rw:>+9.3f} {rsub:>+13.3f}")
        print(f"{'':>14}   alpha peaks {ta}; exponent peaks {te}")

    # does the across-subject coupling-like correlation depend on age?
    ix = [ch.index(c) for c in ROI if c in ch]
    a, b = np.nanmean(DA[:, ix], 1), np.nanmean(DB[:, ix], 1)
    m = ok & np.isfinite(a) & np.isfinite(b)
    X = np.column_stack([np.ones(m.sum()), b[m], age[m], b[m] * age[m]])
    beta = np.linalg.lstsq(X, a[m], rcond=None)[0]
    res = a[m] - X @ beta
    s2 = res @ res / (m.sum() - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print("\nd_log alpha ~ d_log b * age, posterior ROI:")
    for nm, bb, ss in zip(["const", "d_log b", "age", "d_log b x age"],
                          beta, se):
        print(f"  {nm:>14} {bb:+.4f} ({ss:.4f})  z = {bb/ss:+.2f}")


if __name__ == "__main__":
    main()
