"""Figures 1-5 of the paper.

Figure 1  the coupling family and what specparam and IRASA assume
Figure 2  identifiability: single spectrum vs across spectra
Figure 3  consequences for condition contrasts (sim02)
Figure 4  alpha power against age under each separation rule (HBN)
Figure 5  estimating lambda from eyes open vs eyes closed (HBN)

Inputs: results/sim01.mat, results/sim02.mat, results/sim_calibration_fixedharm.csv,
results/hbn_kp_lambda.csv, results/sim_topography_null.csv, and the local
per-subject files results/hbn_kp_fits.csv and results/hbn_topography*.npz
(regenerable with the scripts in code/). Ages come from the PSD files in
$HBN_OUT.

Usage: python make_figures.py OUTDIR [--only 1,4]
"""
import argparse
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(HERE, ".."))

# ---- style -------------------------------------------------------------
C0 = "#2a78d6"     # lambda = 0 (additive, IRASA-like)
C1 = "#eb6834"     # lambda = 1 (multiplicative, specparam-like)
CH = "#1baf7a"     # lambda = 0.5
GREY = "#8a8984"   # total power, background, reference quantities
INK = "#0b0b0b"
INK2 = "#52514e"
MM = 1 / 25.4
W2 = 183 * MM      # Nature double column

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "legend.frameon": False,
    "lines.linewidth": 1.4, "savefig.dpi": 300,
})


def panel(ax, letter):
    ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def gauss(f, cf, bw):
    return np.exp(-0.5 * ((f - cf) / bw) ** 2)


# ---- Figure 1 -----------------------------------------------------------
def fig1(outdir):
    f = np.linspace(1, 40, 800)
    chi, cf, bw = 1.4, 10.0, 1.5
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.72, 2.3))

    # (a) same intrinsic peak on a low and a high background, lambda 0 vs 1
    ax = axs[0]
    ref = 10 ** 1.0 / cf ** chi
    for off, ls in ((1.0, "-"), (1.6, "--")):
        L = 10 ** off / f ** chi
        for lam, col in ((0, C0), (1, C1)):
            amp = 1.2 * ref / ref ** lam          # same size at the low background
            ax.loglog(f, L + amp * gauss(f, cf, bw) * L ** lam, color=col, ls=ls,
                      lw=1.1)
        ax.loglog(f, L, color=GREY, ls=ls, lw=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("Same rhythm on a low and a high background",
                 loc="left", color=INK2)
    ax.text(0.03, 0.05, "solid: low background\ndashed: high background",
            color=INK2, transform=ax.transAxes, fontsize=6)
    ax.text(0.97, 0.95, "λ = 0: peak added", color=C0, transform=ax.transAxes,
            ha="right", va="top")
    ax.text(0.97, 0.86, "λ = 1: peak scales", color=C1, transform=ax.transAxes,
            ha="right", va="top")
    ax.set_xlim(1, 40)
    panel(ax, "a")

    # (b) what each tool reports as the background level changes
    ax = axs[1]
    offs = np.linspace(0.6, 2.0, 50)
    for truth, ls in ((0, "-"), (1, "--")):
        rel, absd = [], []
        for off in offs:
            Lcf = 10 ** off / cf ** chi
            amp = 1.2 * ref / ref ** truth
            peak = amp * Lcf ** truth
            rel.append(np.log10((Lcf + peak) / Lcf))     # specparam: log ratio
            absd.append(peak)                            # IRASA: difference
        rel, absd = np.array(rel), np.array(absd)
        ax.plot(offs, rel / rel[0], color=C1, ls=ls)
        ax.plot(offs, absd / absd[0], color=C0, ls=ls)
    ax.set_yscale("log")
    ax.set_xlabel("Background offset (log$_{10}$ power)")
    ax.set_ylabel("Reported peak (relative to first point)")
    ax.set_title("What each tool reports as the background rises",
                 loc="left", color=INK2)
    ax.text(0.03, 0.95, "IRASA-style (difference)", color=C0,
            transform=ax.transAxes, va="top")
    ax.text(0.03, 0.86, "specparam-style (log ratio)", color=C1,
            transform=ax.transAxes, va="top")
    ax.text(0.03, 0.05, "solid: additive truth\ndashed: multiplicative truth",
            color=INK2, transform=ax.transAxes, va="bottom", fontsize=6)
    panel(ax, "b")

    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig1_coupling_family")


# ---- Figure 2 -----------------------------------------------------------
def whittle_fit(P, f, K, lam):
    """Fit aperiodic + one peak with lambda fixed; return the deviance."""
    def mu(th):
        off, chi, la, cf, lbw = th
        L = 10 ** off / f ** chi
        return L + 10 ** la * gauss(f, cf, 10 ** lbw) * L ** lam
    def dev(th):
        m = np.maximum(mu(th), 1e-300)
        r = P / m
        return 2 * K * np.sum(r - np.log(r) - 1)
    lp = np.log10(P)
    keep = (f < 6) | (f > 16)
    b = np.polyfit(np.log10(f[keep]), lp[keep], 1)
    off0, chi0 = b[1], -b[0]
    L0 = 10 ** off0 / f ** chi0
    i = np.argmax(P / L0 * ((f > 6) & (f < 14)))
    pk = max(P[i] - L0[i], 1e-6)
    best = None
    for la0 in (np.log10(pk / L0[i] ** lam),):
        r = minimize(dev, [off0, chi0, la0, f[i], np.log10(1.5)],
                     method="Nelder-Mead",
                     options=dict(maxiter=6000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.fun


def fig2(outdir):
    from sim_control_band import synth_segments
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))

    # (a) lambda = 1 peak vs its best Gaussian (lambda = 0) approximation
    ax = axs[0]
    f = np.linspace(2, 30, 2000)
    L = f ** -1.4
    pk1 = gauss(f, 10, 1.5) * L
    def res(th):
        return np.sum((th[0] * gauss(f, th[1], th[2]) - pk1) ** 2)
    th = minimize(res, [pk1.max(), 9.7, 1.5], method="Nelder-Mead",
                  options=dict(xatol=1e-10, fatol=1e-16, maxiter=20000)).x
    g0 = th[0] * gauss(f, th[1], th[2])
    s = 1 / pk1.max()
    ax.plot(f, pk1 * s, color=C1, label="λ = 1 peak")
    ax.plot(f, g0 * s, color=C0, ls="--", label="best λ = 0 peak")
    ax.plot(f, (pk1 - g0) * s * 100, color=GREY, lw=0.9,
            label="difference × 100")
    ax.set_xlim(4, 16)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Periodic power (peak = 1)")
    ax.legend(loc="upper right")
    ax.set_title(f"Max difference {np.abs(pk1 - g0).max() * s * 100:.2f}% of peak",
                 loc="left", color=INK2)
    panel(ax, "a")

    # (b) single-spectrum Whittle profiles over lambda: flat
    ax = axs[1]
    rng = np.random.default_rng(7)
    grid = np.linspace(0, 1, 11)
    for lam_true, col in ((0.0, C0), (1.0, C1)):
        for rep in range(3):
            off = rng.normal(1.0, 0.3)
            ref = 10 ** 1.0 / 10 ** 1.2
            amp = 1.0 * ref / ref ** lam_true
            acc, ff = synth_segments(off, 1.2, 10.0, 1.5, amp, lam_true, rng)
            m = (ff >= 2) & (ff <= 30)
            P, fm, K = acc.mean(0)[m], ff[m], acc.shape[0]
            d = np.array([whittle_fit(P, fm, K, lam) for lam in grid])
            ax.plot(grid, d - d.min(), color=col, lw=1.3, alpha=0.9)
    ax.axhline(3.84, color=INK2, lw=0.6, ls=":")
    ax.text(0.02, 3.84, " 95% threshold", color=INK2, va="bottom", fontsize=6)
    ax.set_xlabel("λ assumed in the fit")
    ax.set_ylabel("Δ deviance (one spectrum)")
    ax.text(0.97, 0.55, "true λ = 0", color=C0, transform=ax.transAxes,
            ha="right", va="top")
    ax.text(0.97, 0.46, "true λ = 1", color=C1, transform=ax.transAxes,
            ha="right", va="top")
    ax.set_ylim(-0.2, 4.6)
    ax.set_title("A single spectrum does not identify λ", loc="left", color=INK2)
    panel(ax, "b")

    # (c) across spectra: the GLM coefficient recovers lambda
    ax = axs[2]
    fh = h5py.File(os.path.join(RES, "sim01.mat"), "r")
    R = fh["R"]
    names = ["".join(chr(c) for c in fh[n[0]][:].flatten()) for n in fh["names"]]
    lt, est = [], {}
    oracle = []
    for i in range(3):
        lt.append(np.array(fh[R["lambda_true"][i][0]]).item())
        res_ = fh[R["res"][i][0]]
        lam = [np.array(fh[r]).item() for r in np.array(res_["lambda"]).ravel()]
        se = [np.array(fh[r]).item() for r in np.array(res_["lambda_se"]).ravel()]
        for n_, l_, s_ in zip(names, lam, se):
            est.setdefault(n_, []).append((l_, s_))
        tr = fh[R["truth"][i][0]]
        a_, b_ = np.array(tr["a_true"]).ravel(), np.array(tr["b_true"]).ravel()
        X = np.c_[np.ones(a_.size), np.log(b_)]
        oracle.append(np.linalg.lstsq(X, np.log(a_), rcond=None)[0][1])
    lt = np.array(lt)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
    ax.plot(lt, oracle, "o", color=GREY, ms=5, label="true background")
    for n_, col, dx, lab in (("Censored 6-16 Hz", C0, -0.025, "censored regression"),
                             ("specparam 3 peaks", C1, 0.025, "specparam, 3 peaks")):
        v = np.array(est[n_])
        ax.errorbar(lt + dx, v[:, 0], yerr=1.96 * v[:, 1], fmt="o", color=col,
                    ms=4, lw=1, capsize=0, label=lab)
    ax.set_xlabel("True λ")
    ax.set_ylabel("Estimated λ (GLM across 120 spectra)")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="upper left")
    ax.set_title("Across spectra, λ is identified", loc="left", color=INK2)
    panel(ax, "c")
    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig2_identifiability")


# ---- Figure 3 -----------------------------------------------------------
def fig3(outdir):
    fh = h5py.File(os.path.join(RES, "sim02.mat"), "r")
    S = fh["S"]

    def get(i, k):
        a = np.array(fh[S[k][i][0]]).squeeze()
        return a["real"] if a.dtype.names else a

    def getg(i, k):
        g = fh[S["g2"][i][0]]
        a = np.array(g[k]).squeeze()
        return a["real"] if a.dtype.names else a

    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.78, 2.3),
                            gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    labels = ["remove as λ = 0", "remove as λ = 1", "λ-GLM"]
    cols = [C0, C1, INK]
    truth_names = {0: "additive truth", 1: "multiplicative truth"}
    for gi, i in enumerate(range(2)):
        lam_t = float(get(i, "lambda_true"))
        t = float(get(i, "dlogc_true"))
        ok = get(i, "ok").astype(bool)
        dla, dlb = get(i, "dloga")[ok], get(i, "dlogb")[ok]
        vals = [dla, dla - dlb]
        x0 = gi * 4
        for j, v in enumerate(vals):
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
            ax.bar(x0 + j, m, color=cols[j], width=0.72)
            ax.errorbar(x0 + j, m, yerr=1.96 * se, color=INK, lw=0.8, capsize=0)
        d = float(getg(i, "delta"))
        ci = getg(i, "delta_ci").ravel()
        ax.bar(x0 + 2, d, color=INK2, width=0.72)
        ax.errorbar(x0 + 2, d, yerr=[[d - ci[0]], [ci[1] - d]], color=INK,
                    lw=0.8, capsize=0)
        ax.hlines(t, x0 - 0.5, x0 + 2.5, color=INK, lw=0.8, ls="--")
        ax.text(x0 + 1, -0.45, truth_names[int(round(lam_t))], ha="center",
                color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(["λ = 0", "λ = 1", "GLM"] * 2)
    ax.set_ylim(-0.55, 1.35)
    ax.set_ylabel("Recovered change in alpha (Δ log)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK, lw=0.8, ls="--",
                              label="true change")], loc="upper left")
    ax.set_title("One true change, three removal rules", loc="left", color=INK2)
    panel(ax, "a")

    # (b) per-trial dependence on background for raw power vs the GLM
    # (additive truth)
    ax = axs[1]
    i = 0
    ok = get(i, "ok").astype(bool)
    lb0 = get(i, "lb0")[ok]
    raw = get(i, "dPow")[ok]
    glm = get(i, "dlogc_hat")          # already restricted to ok trials
    z = lambda v: (v - v.mean()) / v.std()
    ax.plot(lb0 / np.log(10), z(raw), "o", ms=3, color=GREY, alpha=0.8,
            mec="none", label=f"raw band power  r = {np.corrcoef(lb0, raw)[0, 1]:.2f}")
    ax.plot(lb0 / np.log(10), z(glm), "o", ms=3, color=INK, alpha=0.8,
            mec="none", label=f"λ-GLM  r = {np.corrcoef(lb0, glm)[0, 1]:.2f}")
    ax.set_xlabel("Baseline background level (log$_{10}$ power)")
    ax.set_ylabel("Estimated change per trial (z)")
    ax.legend(loc="upper left")
    ax.set_title("Dependence on background (additive truth)", loc="left",
                 color=INK2)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig3_condition_contrasts")


# ---- Figure 4 -----------------------------------------------------------
def load_age_table(model):
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    k = k[(k.model == model) & (k.window == "censor 6-16") & (k.band == "alpha")
          & (k.split == "full")]
    w = k.pivot_table(index="subject", columns="cond", values=["a", "b", "tot"])
    w.columns = [f"{v}_{c}" for v, c in w.columns]
    rows = []
    for p in glob.glob(os.path.join(os.environ.get("HBN_OUT", "hbn_psd"), "*.npz")):
        d = np.load(p, allow_pickle=True)
        rows.append((str(d["subject"]), float(d["age"])))
    ages = pd.DataFrame(rows, columns=["subject", "age"]).drop_duplicates("subject")
    return w.join(ages.set_index("subject"), how="inner").dropna()


def slope_ci(x, y, rng, nboot=1000):
    X = np.c_[np.ones_like(x), x]
    s = np.linalg.lstsq(X, y, rcond=None)[0][1]
    n = x.size
    bs = [np.linalg.lstsq(X[i], y[i], rcond=None)[0][1]
          for i in (rng.integers(0, n, n) for _ in range(nboot))]
    return s, *np.percentile(bs, [2.5, 97.5])


def fig4(outdir):
    rng = np.random.default_rng(0)
    wf = load_age_table("fixed")
    wk = load_age_table("knee_plateau")
    fig = plt.figure(figsize=(W2, 2.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.25, 1.6])

    # (a) eyes-closed alpha vs age under three rules
    w = wf[wf.a_ec > 0]
    series = [("total power", wf.age, np.log10(wf.tot_ec), GREY),
              ("λ = 0 (subtract)", w.age, np.log10(w.a_ec), C0),
              ("λ = 1 (divide)", w.age, np.log10(w.a_ec / w.b_ec), C1)]
    for j, (lab, age, y, col) in enumerate(series):
        ax = fig.add_subplot(gs[0, j])
        age, y = age.to_numpy(), y.to_numpy()
        ax.plot(age, y, "o", ms=1.2, color=col, alpha=0.25, mec="none")
        b = np.polyfit(age, y, 1)
        xx = np.array([age.min(), age.max()])
        ax.plot(xx, np.polyval(b, xx), color=INK, lw=1.2)
        lo, hi = np.percentile(y, [1, 99])
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Age (years)")
        if j == 0:
            ax.set_ylabel("Eyes-closed alpha (log$_{10}$)")
            panel(ax, "a")
        ax.set_title(lab, loc="left", color=col if j else INK2)
        ax.text(0.97, 0.04, f"{b[0] * np.log(10):+.3f}/y", transform=ax.transAxes,
                ha="right", color=INK)

    # (b) slope per year as a function of the assumed lambda
    ax = fig.add_subplot(gs[0, 4])
    grid = np.round(np.arange(0, 1.0001, 0.1), 2)
    for w_, cond, col, ls, lab in ((wf, "ec", INK, "-", "eyes closed"),
                                   (wf, "eo", INK2, "-", "eyes open"),
                                   (wk, "ec", INK, "--", "eyes closed, knee+plateau")):
        w2 = w_[w_[f"a_{cond}"] > 0]
        x = w2.age.to_numpy()
        la, lb = np.log(w2[f"a_{cond}"].to_numpy()), np.log(w2[f"b_{cond}"].to_numpy())
        out = np.array([slope_ci(x, la - g * lb, rng, 400) for g in grid])
        ax.plot(grid, out[:, 0], color=col, ls=ls, lw=1.2, label=lab)
        if ls == "-":
            ax.fill_between(grid, out[:, 1], out[:, 2], color=col, alpha=0.12,
                            lw=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0\nIRASA-like", "0.25", "0.5", "0.75",
                        "1\nspecparam-like"])
    for t, col in ((ax.get_xticklabels()[0], C0), (ax.get_xticklabels()[-1], C1)):
        t.set_color(col)
    ax.set_xlabel("Assumed coupling λ")
    ax.set_ylabel("Change in alpha with age (log units per year)")
    ax.legend(loc="lower right")
    panel(ax, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, outdir, "fig4_age_reversal")


# ---- Figure 5 -----------------------------------------------------------
def fig5(outdir):
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3))

    # (a) specification curve
    ax = axs[0]
    k = pd.read_csv(os.path.join(RES, "hbn_kp_lambda.csv"))
    k = k[k.band == "alpha"].reset_index(drop=True)
    k["label"] = [f"{'knee+plateau' if m == 'knee_plateau' else 'power law'}\n"
                  f"{'censor 6-16 Hz' if w.startswith('censor') else 'flanks'}"
                  for m, w in zip(k.model, k.window)]
    k = k.sort_values("lam_iv").reset_index(drop=True)
    y = np.arange(len(k))
    ax.errorbar(k.lam_iv, y, xerr=[k.lam_iv - k.ci_lo, k.ci_hi - k.lam_iv],
                fmt="o", color=INK, ms=4, lw=1, capsize=0)
    ax.set_yticks(y)
    ax.set_yticklabels(k.label)
    ax.set_xlabel("Uncalibrated λ̂ (within subject, IV)")
    ax.set_xlim(0.3, 1.05)
    ax.set_title("Specification moves the estimate", loc="left", color=INK2)
    panel(ax, "a")

    # (b) calibration curves
    ax = axs[1]
    c = pd.read_csv(os.path.join(RES, "sim_calibration_fixedharm.csv"))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
    for (est, route), g in c.groupby(["estimator", "route"]):
        m = g.groupby("lambda_true").lam_hat.agg(["mean", "std"])
        good = est.startswith("flanks") and route == "within IV"
        col = INK if good else GREY
        ax.plot(m.index, m["mean"], "-o", color=col, ms=3 if good else 2,
                lw=1.3 if good else 0.8, alpha=1 if good else 0.7)
        if good:
            ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"],
                            color=INK, alpha=0.1, lw=0)
    ax.axhspan(0.574 - 0.039, 0.574 + 0.039, color=C1, alpha=0.18, lw=0)
    ax.axhline(0.574, color=C1, lw=1)
    ax.axvline(0.85, color=C1, lw=0.8, ls="--")
    ax.text(0.87, -0.24, "λ = 0.85", color=C1, ha="left", fontsize=6)
    ax.text(0.02, 0.60, "observed", color=C1, fontsize=6, va="bottom")
    ax.text(0.02, 0.97, "invertible route (black);\nother routes fold (grey)",
            color=INK2, fontsize=6, va="top", transform=ax.transAxes)
    ax.set_xlabel("True λ (simulation)")
    ax.set_ylabel("Estimated λ̂")
    ax.set_title("Calibration", loc="left", color=INK2)
    panel(ax, "b")

    # (c) spatial test: real vs null
    ax = axs[2]
    specs = [("2-40", ""), ("4-40", "_f4-40"), ("5-40", "_f5-40"), ("2-30", "_f2-30")]
    null = pd.read_csv(os.path.join(RES, "sim_topography_null.csv"))
    for j, (lab, tag) in enumerate(specs):
        lo_, hi_ = (float(v) for v in lab.split("-"))
        nn = null[(null.fit_lo == lo_) & (null.fit_hi == hi_)].r_exp_alpha
        ax.plot([j, j], [nn.min(), nn.max()], color=GREY, lw=6,
                solid_capstyle="butt", alpha=0.6)
        d = np.load(os.path.join(RES, f"hbn_topography{tag}.npz"))
        mE, mA = np.nanmean(d["d_exponent"], 0), np.nanmean(d["d_log_a"], 0)
        g = np.isfinite(mE) & np.isfinite(mA)
        ax.plot(j, np.corrcoef(mE[g], mA[g])[0, 1], "o", color=C0, ms=5)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels([s[0] + " Hz" for s in specs])
    ax.set_xlabel("Aperiodic fit range")
    ax.set_ylabel("Spatial r, Δexponent vs Δalpha maps")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=C0, ms=5,
                              label="HBN (n ≈ 1,300)"),
                       Patch(color=GREY, alpha=0.6,
                             label="simulations with no\nbackground change")],
              loc="lower left")
    ax.set_title("The spatial test against its null", loc="left", color=INK2)
    panel(ax, "c")
    fig.tight_layout(w_pad=3.0)
    save(fig, outdir, "fig5_estimating_lambda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--only", default="1,2,3,4,5")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    todo = {int(x) for x in a.only.split(",")}
    for k, fn in ((1, fig1), (2, fig2), (3, fig3), (4, fig4), (5, fig5)):
        if k in todo:
            fn(a.outdir)


if __name__ == "__main__":
    main()
