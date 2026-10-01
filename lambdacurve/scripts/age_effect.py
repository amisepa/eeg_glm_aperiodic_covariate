"""Effect of a predictor (e.g. age) on alpha power under both background rules.

Reads a CSV with one row per participant and prints the effect with the
background subtracted (lambda = 0) and divided out (lambda = 1), and the
crossover lambda* between them. Only summary numbers are printed and saved;
no participant-level data are written.

From a table of specparam parameters (a first look; see the README):

    python age_effect.py table.csv --x age --covariates sex medication \
        --offset offset --exponent exponent --cf CF --pw PW

    Add --knee <column> for knee-mode fits. If the table has one row per
    peak, add --id <column>: the largest peak with CF inside --alpha
    (default 7 14) is taken for each participant.

From the power spectra (preferred; no specparam needed):

    python age_effect.py spectra.csv --spectra --x age --covariates sex medication

    Every column whose name is a number is read as power (linear, not dB) at
    that frequency in Hz. The background is a power law fitted over --fit
    (default 2 45) with --censor (default 6 16) left out; the band is the
    alpha peak frequency +/- 2 Hz.

Covariates that are text (sex, diagnosis) are dummy-coded. Use one channel
or one ROI average per run. Needs numpy, scipy, pandas and lambdacurve
(pip install ./lambdacurve from the repository root).
"""
import argparse

import numpy as np
import pandas as pd

from lambdacurve import band_power, from_specparam, lambda_curve, peak_frequency


def from_table(D, a):
    cf, pw = D[a.cf].astype(float), D[a.pw].astype(float)
    alpha = cf.between(*a.alpha) & (pw > 0)
    if a.id:
        # one row per peak: keep the largest alpha peak, or any row if there is none
        D = (D.assign(_alpha=alpha, _pw=pw.where(alpha, -np.inf))
              .sort_values("_pw").groupby(a.id, as_index=False).last())
        alpha = D["_alpha"].to_numpy()
    elif D.duplicated([a.x, a.offset, a.exponent]).any():
        raise SystemExit("several rows share one aperiodic fit: give --id if the table "
                         "has one row per peak")
    la, lb = from_specparam(D[a.offset], D[a.exponent], D[a.cf], D[a.pw],
                            knee=D[a.knee] if a.knee else None)
    la = np.where(alpha, la, np.nan)
    return D, la, np.where(alpha, lb, np.nan)


def from_spectra(D, a):
    cols = []
    for c in D.columns:
        try:
            cols.append((float(c), c))
        except ValueError:
            pass
    if len(cols) < 20:
        raise SystemExit("--spectra needs one column per frequency, named by its value in Hz")
    cols.sort()
    f = np.array([v for v, _ in cols])
    P = D[[c for _, c in cols]].to_numpy(float)
    censor = (tuple(a.censor),)
    la, lb = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for i in range(len(D)):
        if not np.all(np.isfinite(P[i])) or np.any(P[i] <= 0):
            continue
        iaf = peak_frequency(P[i], f, search=tuple(a.alpha), censor=censor, fit_range=tuple(a.fit))
        if not np.isfinite(iaf):
            continue
        bp = band_power(P[i], f, (iaf - 2, iaf + 2), tuple(a.fit), censor)
        if bp is not None and bp["a"] > 0:
            la[i], lb[i] = np.log(bp["a"]), np.log(bp["b"])
    return D, la, lb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--x", required=True, help="predictor of interest, e.g. age")
    p.add_argument("--covariates", nargs="*", default=[])
    p.add_argument("--spectra", action="store_true", help="the CSV holds power spectra")
    p.add_argument("--id", help="participant column, if the table has one row per peak")
    p.add_argument("--offset", default="offset")
    p.add_argument("--exponent", default="exponent")
    p.add_argument("--knee", help="knee column (knee-mode fits)")
    p.add_argument("--cf", default="CF")
    p.add_argument("--pw", default="PW")
    p.add_argument("--alpha", nargs=2, type=float, default=[7.0, 14.0])
    p.add_argument("--fit", nargs=2, type=float, default=[2.0, 45.0])
    p.add_argument("--censor", nargs=2, type=float, default=[6.0, 16.0])
    p.add_argument("--nboot", type=int, default=2000)
    p.add_argument("--out", default="age_effect_summary.csv")
    a = p.parse_args()

    D = pd.read_csv(a.csv)
    D, la, lb = (from_spectra if a.spectra else from_table)(D, a)
    n_total = len(D)
    no_peak = ~np.isfinite(la)
    C = pd.get_dummies(D[a.covariates], drop_first=True).astype(float) if a.covariates else None
    x = D[a.x].astype(float).to_numpy()
    missing = ~np.isfinite(x) | (C.isna().any(axis=1).to_numpy() if C is not None else False)
    r = lambda_curve(la, lb, x=x, covariates=None if C is None else C.to_numpy(),
                     nboot=a.nboot, dropna=True)
    print(f"{n_total} participants, {int(no_peak.sum())} without an alpha peak "
          f"({100 * no_peak.mean():.1f}%), {int((missing & ~no_peak).sum())} more with missing "
          f"{a.x} or covariates")
    # does having a peak depend on the predictor?
    ok = ~missing
    if no_peak[ok].any() and (~no_peak[ok]).any():
        print(f"mean {a.x}: {x[ok & ~no_peak].mean():.2f} with a peak, "
              f"{x[ok & no_peak].mean():.2f} without")
    print(r)
    row = dict(source="spectra" if a.spectra else "specparam table", x=a.x,
               covariates=" ".join(a.covariates), n_total=n_total,
               n_no_peak=int(no_peak.sum()), **r.to_dict())
    pd.DataFrame([row]).to_csv(a.out, index=False)
    print(f"summary written to {a.out}")


if __name__ == "__main__":
    main()
