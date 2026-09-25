"""Estimate the periodic/aperiodic coupling exponent in HBN resting-state EEG.

Routes:
  between-subject   lambda from the spread of aperiodic level across subjects,
                    with an optional correction for a common gain
  within-subject    eyes-closed minus eyes-open changes, so that anything that
                    scales a subject's whole spectrum (electrodes, skull, head
                    size) cancels
  instrumented      predictors from odd Welch segments and outcome from even
                    ones (and vice versa), so that estimation error shared
                    between the periodic and aperiodic estimates cancels

Also runs the 2-df (p, q) test of log a = c + p*offset + q*exponent against
additive (0, 0) and multiplicative (1, -log10 f_alpha) coupling.

Usage: python hbn_lambda.py [--roi results/hbn_roi.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

MIN_N = 12
F_ANCHOR = 25.0    # Hz, aperiodic power here is the per-spectrum gain proxy


def gamma_slope(a, b):
    """Gamma GLM with log link: log E[a] = c + lambda*log b. Returns lambda, SE, n."""
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < MIN_N:
        return np.nan, np.nan, int(ok.sum())
    X = sm.add_constant(np.log(b[ok]))
    try:
        r = sm.GLM(a[ok], X, family=sm.families.Gamma(sm.families.links.Log())).fit()
    except Exception:
        return np.nan, np.nan, int(ok.sum())
    return float(r.params[1]), float(r.bse[1]), int(ok.sum())


def gain_at(offset, exponent, f0=F_ANCHOR):
    """Aperiodic power at the anchor frequency: the per-spectrum gain proxy."""
    return 10.0 ** np.asarray(offset) / f0 ** np.asarray(exponent)


def ols_slope(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < MIN_N:
        return np.nan, np.nan, int(ok.sum())
    X = sm.add_constant(np.log(b[ok]))
    r = sm.OLS(np.log(a[ok]), X).fit()
    return float(r.params[1]), float(r.bse[1]), int(ok.sum())


def decompose(a, b, offset, exponent, f0=F_ANCHOR):
    """Separate the instrumental gain path from the spectral-shape path.

        log a = c + gamma * log g + lambda_shape * log(b/g)

    g = L(f0) is the per-spectrum gain, and b/g depends only on the exponent.
    gamma near 1 means the whole spectrum, periodic part included, is simply
    scaled by a common factor (electrode gain, skull, head size, reference).
    lambda_shape is the coupling that a neural account is about: does a
    steeper background carry proportionally more alpha, over and above gain.
    """
    g = gain_at(offset, exponent, f0)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(g) & (a > 0) & (b > 0) & (g > 0)
    if ok.sum() < MIN_N:
        return dict(gamma=np.nan, gamma_se=np.nan, lam=np.nan, lam_se=np.nan,
                    n=int(ok.sum()), f0=f0)
    X = np.column_stack([np.log(g[ok]), np.log(b[ok] / g[ok])])
    r = sm.GLM(a[ok], sm.add_constant(X),
               family=sm.families.Gamma(sm.families.links.Log())).fit()
    return dict(gamma=float(r.params[1]), gamma_se=float(r.bse[1]),
                lam=float(r.params[2]), lam_se=float(r.bse[2]),
                n=int(ok.sum()), f0=f0)


def pq_test(a, offset, exponent, f_alpha):
    """Anchor-free test of the coupling model.

    Regress log periodic band power on the two aperiodic parameters:

        log10 a_i = c + p * offset_i + q * exponent_i

    Because (offset, exponent) determines the aperiodic level at any
    frequency, this is the general linear model of how periodic power tracks
    the background, with no reference frequency to choose. Under a coupling
    exponent lambda, a = c * b^lambda with b = 10^offset / f_alpha^exponent,
    so the model predicts

        p = lambda,    q = -lambda * log10(f_alpha)

    Purely multiplicative coupling (lambda = 1) predicts (p, q) =
    (1, -log10 f_alpha); purely additive coupling (lambda = 0) predicts
    (0, 0). Both are 2-degree-of-freedom Wald tests. If p and q are mutually
    inconsistent, no single lambda describes the data -- which is itself the
    prediction of a mixed multiplicative/additive generative model.

    offset and exponent are strongly collinear (the offset is pivoted at
    1 Hz, far outside the fit range), so the individual coefficients are
    imprecise while the joint test is not. Report both.
    """
    ok = np.isfinite(a) & (a > 0) & np.isfinite(offset) & np.isfinite(exponent)
    if ok.sum() < MIN_N:
        return None
    y = np.log10(a[ok])
    X = sm.add_constant(np.column_stack([offset[ok], exponent[ok]]))
    r = sm.OLS(y, X).fit()
    lfa = float(np.log10(np.nanmedian(f_alpha)))
    Rm = np.array([[0., 1., 0.], [0., 0., 1.]])
    add_t = r.f_test(Rm)                                 # H0: p = 0, q = 0
    mul_t = r.f_test((Rm, np.array([1.0, -lfa])))        # H0: p = 1, q = -log10 f_a
    return dict(p=float(r.params[1]), p_se=float(r.bse[1]),
                q=float(r.params[2]), q_se=float(r.bse[2]),
                lam_from_p=float(r.params[1]),
                lam_from_q=float(-r.params[2] / lfa) if lfa else np.nan,
                F_additive=float(add_t.fvalue), p_additive=float(add_t.pvalue),
                F_multiplicative=float(mul_t.fvalue),
                p_multiplicative=float(mul_t.pvalue),
                log10_falpha=lfa, n=int(ok.sum()), r2=float(r.rsquared))


def pq_within(R, est, covar=None, split_a="full", split_b="full"):
    """Within-subject version of the (p, q) test, using the eyes-closed minus
    eyes-open contrast.

        dlog10 a_i = c + p * d_offset_i + q * d_exponent_i

    Everything that merely scales a subject's whole spectrum -- electrode
    gain, skull, head size, reference, age-related amplitude -- is constant
    across the two conditions and cancels exactly, so this route cannot be
    driven by the instrumental path that makes the between-subject test read
    p = 1 whatever the coupling is. The same predictions apply:
    lambda = 1 gives (p, q) = (1, -log10 f_alpha), lambda = 0 gives (0, 0).

    split_a / split_b let the outcome and the predictors come from
    independent halves of the recording (odd vs even Welch segments), which
    breaks the shared estimation error between a and the aperiodic fit.
    """
    F = R[R.estimator == est]

    def piv(split, cols):
        g = F[F.split == split].pivot_table(index="subject", columns="cond", values=cols)
        return g.dropna()

    A = piv(split_a, ["a_alpha"])
    B = piv(split_b, ["offset", "exponent", "iaf"])
    idx = A.index.intersection(B.index)
    if len(idx) < MIN_N:
        return None
    A, B = A.loc[idx], B.loc[idx]
    a_ec, a_eo = A[("a_alpha", "ec")].to_numpy(), A[("a_alpha", "eo")].to_numpy()
    ok = (a_ec > 0) & (a_eo > 0)
    y = np.log10(a_ec[ok] / a_eo[ok])
    doff = (B[("offset", "ec")] - B[("offset", "eo")]).to_numpy()[ok]
    dexp = (B[("exponent", "ec")] - B[("exponent", "eo")]).to_numpy()[ok]
    lfa = float(np.log10(np.nanmedian(B[("iaf", "ec")].to_numpy())))
    X = np.column_stack([doff, dexp])
    names = ["const", "d_offset", "d_exponent"]
    if covar is not None:
        X = np.column_stack([X, covar[idx][ok]])
        names.append("covar")
    r = sm.OLS(y, sm.add_constant(X)).fit()
    k = X.shape[1] + 1
    Rm = np.zeros((2, k)); Rm[0, 1] = 1.0; Rm[1, 2] = 1.0
    add_t = r.f_test(Rm)
    mul_t = r.f_test((Rm, np.array([1.0, -lfa])))
    return dict(estimator=est, split_a=split_a, split_b=split_b,
                p=float(r.params[1]), p_se=float(r.bse[1]),
                q=float(r.params[2]), q_se=float(r.bse[2]),
                lam_from_p=float(r.params[1]),
                lam_from_q=float(-r.params[2] / lfa) if lfa else np.nan,
                F_additive=float(add_t.fvalue), p_additive=float(add_t.pvalue),
                F_multiplicative=float(mul_t.fvalue),
                p_multiplicative=float(mul_t.pvalue),
                log10_falpha=lfa, n=int(ok.sum()), r2=float(r.rsquared))


def iv_slope(a1, b1, a2, b2):
    """cov(log a, log b_other) / cov(log b, log b_other), symmetrised."""
    ok = np.all([np.isfinite(v) & (v > 0) for v in (a1, b1, a2, b2)], axis=0)
    if ok.sum() < MIN_N:
        return np.nan, int(ok.sum())
    la1, lb1, la2, lb2 = (np.log(v[ok]) for v in (a1, b1, a2, b2))
    c = lambda x, y: np.mean((x - x.mean()) * (y - y.mean()))
    num = 0.5 * (c(lb2, la1) + c(lb1, la2))
    den = c(lb1, lb2)
    return float(num / den) if den != 0 else np.nan, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--roi", default=os.path.join(here, "results", "hbn_roi.csv"))
    ap.add_argument("--chan", default=os.path.join(here, "results", "hbn_chan.csv"))
    a = ap.parse_args()

    R = pd.read_csv(a.roi)
    ests = [e for e in ["censored", "local_flank", "full_reg", "theilsen",
                        "specparam1", "specparam3"] if e in set(R.estimator)]
    nsub = R.subject.nunique()
    print(f"\n{nsub} subjects, {len(R)} ROI rows, estimators: {ests}")
    if "age" in R:
        ag = R.drop_duplicates("subject").age
        print(f"age: median {ag.median():.1f}, range {ag.min():.1f}-{ag.max():.1f}")

    # ---------- 1. between-subject lambda ----------
    print("\n=== between-subject coupling exponent (posterior ROI) ===")
    print(f"{'estimator':14s} {'cond':4s} {'lambda (Gamma)':>18s} {'lambda (OLS)':>16s} "
          f"{'lambda (IV)':>11s} {'lambda anchored':>19s} {'n':>5s}")
    rows = []
    for est in ests:
        for cond in ("ec", "eo"):
            F = R[(R.estimator == est) & (R.cond == cond)]
            f_full = F[F.split == "full"].set_index("subject")
            f_odd = F[F.split == "odd"].set_index("subject")
            f_even = F[F.split == "even"].set_index("subject")
            idx = f_full.index.intersection(f_odd.index).intersection(f_even.index)
            aa = f_full.loc[idx, "a_alpha"].to_numpy()
            bb = f_full.loc[idx, "b_alpha"].to_numpy()
            gg = gain_at(f_full.loc[idx, "offset"].to_numpy(),
                         f_full.loc[idx, "exponent"].to_numpy())
            lg, sg, n = gamma_slope(aa, bb)
            lo, so, _ = ols_slope(aa, bb)
            lan, san, _ = ols_slope(aa / gg, bb / gg)
            liv, _ = iv_slope(f_odd.loc[idx, "a_alpha"].to_numpy(),
                              f_odd.loc[idx, "b_alpha"].to_numpy(),
                              f_even.loc[idx, "a_alpha"].to_numpy(),
                              f_even.loc[idx, "b_alpha"].to_numpy())
            print(f"{est:14s} {cond:4s} {lg:10.3f} ({sg:.3f}) {lo:10.3f} ({so:.3f}) "
                  f"{liv:11.3f} {lan:12.3f} ({san:.3f}) {n:5d}")
            rows.append(dict(estimator=est, cond=cond, lambda_gamma=lg, se=sg,
                             lambda_ols=lo, lambda_iv=liv, lambda_anchored=lan,
                             se_anchored=san, n=n))
    out = pd.DataFrame(rows)

    # ---------- 1b. gain / shape decomposition and anchor sensitivity ----------
    print("\n=== gain vs shape decomposition (eyes closed, posterior ROI) ===")
    print(f"{'estimator':14s} {'anchor':>7s} {'gamma (gain)':>17s} "
          f"{'lambda (shape)':>19s} {'n':>5s}")
    drows = []
    for est in ests:
        F = R[(R.estimator == est) & (R.cond == "ec") & (R.split == "full")]
        for f0 in (20.0, 25.0, 30.0, 35.0):
            d = decompose(F.a_alpha.to_numpy(), F.b_alpha.to_numpy(),
                          F.offset.to_numpy(), F.exponent.to_numpy(), f0)
            print(f"{est:14s} {f0:7.0f} {d['gamma']:10.3f} ({d['gamma_se']:.3f}) "
                  f"{d['lam']:12.3f} ({d['lam_se']:.3f}) {d['n']:5d}")
            drows.append(dict(estimator=est, **d))
    pd.DataFrame(drows).to_csv(os.path.join(os.path.dirname(a.roi),
                                            "hbn_decompose.csv"), index=False)

    # ---------- 1c. anchor-free (p, q) test ----------
    print("\n=== anchor-free coupling test: log10 a ~ offset + exponent ===")
    print(f"{'estimator':14s} {'cond':4s} {'p':>14s} {'q':>14s} "
          f"{'lam|p':>7s} {'lam|q':>7s} {'F_add (p)':>18s} {'F_mult (p)':>18s} {'n':>5s}")
    prows = []
    for est in ests:
        for cond in ("ec", "eo"):
            F = R[(R.estimator == est) & (R.cond == cond) & (R.split == "full")]
            d = pq_test(F.a_alpha.to_numpy(), F.offset.to_numpy(),
                        F.exponent.to_numpy(), F.iaf.to_numpy())
            if d is None:
                continue
            print(f"{est:14s} {cond:4s} {d['p']:7.3f} ({d['p_se']:.3f}) "
                  f"{d['q']:7.3f} ({d['q_se']:.3f}) {d['lam_from_p']:7.3f} "
                  f"{d['lam_from_q']:7.3f} {d['F_additive']:10.2f} "
                  f"({d['p_additive']:.1e}) {d['F_multiplicative']:8.2f} "
                  f"({d['p_multiplicative']:.1e}) {d['n']:5d}")
            prows.append(dict(estimator=est, cond=cond, **d))
    pd.DataFrame(prows).to_csv(os.path.join(os.path.dirname(a.roi),
                                            "hbn_pq.csv"), index=False)

    # ---------- 1d. within-subject (eyes closed - eyes open) test ----------
    print("\n=== within-subject coupling test (EC - EO; gain cancels) ===")
    print(f"{'estimator':14s} {'a<-':>5s} {'ap<-':>5s} {'p':>14s} {'q':>14s} "
          f"{'lam|p':>7s} {'lam|q':>7s} {'F_add (p)':>18s} {'F_mult (p)':>18s} {'n':>5s}")
    wrows = []
    for est in ests:
        for sa, sb in (("full", "full"), ("odd", "even"), ("even", "odd")):
            d = pq_within(R[R.split.isin([sa, sb, "full"])], est, None, sa, sb)
            if d is None:
                continue
            print(f"{est:14s} {sa:>5s} {sb:>5s} {d['p']:7.3f} ({d['p_se']:.3f}) "
                  f"{d['q']:7.3f} ({d['q_se']:.3f}) {d['lam_from_p']:7.3f} "
                  f"{d['lam_from_q']:7.3f} {d['F_additive']:10.2f} "
                  f"({d['p_additive']:.1e}) {d['F_multiplicative']:8.2f} "
                  f"({d['p_multiplicative']:.1e}) {d['n']:5d}")
            wrows.append(d)
    pd.DataFrame(wrows).to_csv(os.path.join(os.path.dirname(a.roi),
                                            "hbn_pq_within.csv"), index=False)

    # ---------- 2. odd-even reliability of the aperiodic estimate ----------
    print("\n=== odd-even reliability (Kalamala metric), eyes closed ===")
    for est in ests:
        F = R[(R.estimator == est) & (R.cond == "ec")]
        o = F[F.split == "odd"].set_index("subject")
        e = F[F.split == "even"].set_index("subject")
        idx = o.index.intersection(e.index)
        if len(idx) < 20:
            continue
        rex = np.corrcoef(o.loc[idx, "exponent"], e.loc[idx, "exponent"])[0, 1]
        rof = np.corrcoef(o.loc[idx, "offset"], e.loc[idx, "offset"])[0, 1]
        print(f"  {est:14s} exponent r = {rex:.3f}   offset r = {rof:.3f}   n = {len(idx)}")

    # ---------- 3. within-subject lambda across channels ----------
    if os.path.exists(a.chan):
        C = pd.read_csv(a.chan)
        print("\n=== within-subject coupling exponent, across 129 channels ===")
        for cond in ("ec", "eo"):
            lam = []
            lan = []
            for s_, g in C[C.cond == cond].groupby("subject"):
                l, _, n = gamma_slope(g.a_alpha.to_numpy(), g.b_alpha.to_numpy())
                if np.isfinite(l):
                    lam.append(l)
                gg = gain_at(g.offset.to_numpy(), g.exponent.to_numpy())
                l2, _, _ = ols_slope(g.a_alpha.to_numpy() / gg, g.b_alpha.to_numpy() / gg)
                if np.isfinite(l2):
                    lan.append(l2)
            lam, lan = np.array(lam), np.array(lan)
            if lam.size:
                print(f"  {cond} naive   : n = {lam.size}, median {np.median(lam):.3f}, "
                      f"IQR [{np.percentile(lam,25):.3f}, {np.percentile(lam,75):.3f}]")
            if lan.size:
                print(f"  {cond} anchored: n = {lan.size}, median {np.median(lan):.3f}, "
                      f"IQR [{np.percentile(lan,25):.3f}, {np.percentile(lan,75):.3f}]")

    # ---------- 4. the eyes-closed effect under each convention ----------
    print("\n=== eyes-closed minus eyes-open alpha effect (posterior ROI) ===")
    est = "censored" if "censored" in ests else ests[0]
    F = R[(R.estimator == est) & (R.split == "full")]
    piv = F.pivot_table(index="subject", columns="cond",
                        values=["a_alpha", "b_alpha", "tot_alpha"])
    piv = piv.dropna()
    a_ec, a_eo = piv[("a_alpha", "ec")].to_numpy(), piv[("a_alpha", "eo")].to_numpy()
    b_ec, b_eo = piv[("b_alpha", "ec")].to_numpy(), piv[("b_alpha", "eo")].to_numpy()
    t_ec, t_eo = piv[("tot_alpha", "ec")].to_numpy(), piv[("tot_alpha", "eo")].to_numpy()
    ok = (a_ec > 0) & (a_eo > 0) & (b_ec > 0) & (b_eo > 0)
    a_ec, a_eo, b_ec, b_eo, t_ec, t_eo = (v[ok] for v in (a_ec, a_eo, b_ec, b_eo, t_ec, t_eo))
    lrow = out[(out.estimator == est) & (out.cond == "ec")]
    lam = lrow.lambda_anchored.iloc[0]
    if not np.isfinite(lam):
        lam = lrow.lambda_gamma.iloc[0]

    lb0 = np.log(b_eo)                      # background level, eyes open
    def dep(v):
        return float(np.corrcoef(lb0, v)[0, 1])
    d_raw = t_ec - t_eo
    d_db = 10 * np.log10(t_ec / t_eo)
    d_abs = a_ec - a_eo
    d_log = np.log(a_ec) - np.log(a_eo)
    d_lam = (np.log(a_ec) - lam * np.log(b_ec)) - (np.log(a_eo) - lam * np.log(b_eo))
    print(f"  estimator = {est}, lambda = {lam:.3f}, n = {len(d_raw)}")
    print(f"  {'convention':26s} {'median':>10s} {'IQR width':>11s} "
          f"{'r with background':>20s} {'Spearman rho':>14s}")
    for nm, v in (("raw power difference", d_raw), ("dB difference", d_db),
                  ("periodic, absolute", d_abs), ("periodic, log ratio", d_log),
                  ("lambda-corrected", d_lam)):
        from scipy.stats import spearmanr
        iqr = np.percentile(v, 75) - np.percentile(v, 25)
        rho = spearmanr(lb0, v).statistic
        print(f"  {nm:26s} {np.median(v):10.3f} {iqr:11.3f} {dep(v):20.3f} {rho:14.3f}")

    outp = os.path.join(os.path.dirname(a.roi), "hbn_lambda.csv")
    out.to_csv(outp, index=False)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
