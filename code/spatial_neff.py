"""Effective number of independent channels for the spatial map correlations.

Neighbouring EEG channels are correlated, so a correlation between two scalp
maps has far fewer degrees of freedom than channels. Following Clifford,
Richardson and Hemon (1989) and Dutilleul (1993), the effective sample size
for the correlation between maps x and y is

    n_eff = 1 + n^2 / trace(Rx Ry),

where Rx and Ry are the spatial autocorrelation matrices of the two maps,
estimated from Moran-type correlograms in 10 equal-count distance classes on
the GSN HydroCel 129 electrode positions. The map correlation is then tested
with n_eff - 2 degrees of freedom.

Maps: the HBN group maps of the eyes-closed minus eyes-open change in fitted
exponent and in fitted background, each against the change in periodic alpha
power, for each aperiodic fit range (results/hbn_topography*.npz).

Writes results/spatial_neff.csv.

Usage: python spatial_neff.py
"""
import os

import mne
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def positions(names):
    mont = mne.channels.make_standard_montage("GSN-HydroCel-129")
    pos = mont.get_positions()["ch_pos"]
    out = []
    for n in names:
        key = n if n in pos else ("Cz" if n in ("E129", "Cz") else None)
        out.append(pos[key] if key else [np.nan] * 3)
    return np.array(out)


def autocorr_matrix(z, D, nclass=10):
    z = (z - z.mean()) / z.std()
    n = z.size
    iu = np.triu_indices(n, 1)
    d = D[iu]
    edges = np.quantile(d, np.linspace(0, 1, nclass + 1))
    cls = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, nclass - 1)
    prod = z[iu[0]] * z[iu[1]]
    rho = np.array([prod[cls == k].mean() for k in range(nclass)])
    R = np.eye(n)
    R[iu] = rho[cls]
    R[(iu[1], iu[0])] = rho[cls]
    return R


def neff_test(x, y, D):
    Rx, Ry = autocorr_matrix(x, D), autocorr_matrix(y, D)
    n = x.size
    ne = 1 + n ** 2 / np.trace(Rx @ Ry)
    r = np.corrcoef(x, y)[0, 1]
    df = max(ne - 2, 1)
    t = r * np.sqrt(df / (1 - r ** 2))
    return r, ne, 2 * stats.t.sf(abs(t), df)


def main():
    rows = []
    for tag, lab in (("", "2-40 Hz"), ("_f4-40", "4-40 Hz"), ("_f5-40", "5-40 Hz"),
                     ("_f2-30", "2-30 Hz")):
        d = np.load(os.path.join(RES, f"hbn_topography{tag}.npz"))
        names = [str(c) for c in d["ch_names"]]
        P = positions(names)
        mE, mA, mB = (np.nanmean(d[k], 0) for k in ("d_exponent", "d_log_a", "d_log_b"))
        g = np.isfinite(mE) & np.isfinite(mA) & np.isfinite(mB) & np.isfinite(P).all(1)
        D = np.linalg.norm(P[g][:, None, :] - P[g][None, :, :], axis=2)
        for other, olab in ((mE, "d exponent"), (mB, "d log background")):
            r, ne, p = neff_test(other[g], mA[g], D)
            rows.append(dict(fit_range=lab, map=olab, r=r, n_channels=int(g.sum()),
                             n_eff=ne, p=p))
    O = pd.DataFrame(rows)
    print(O.round(3).to_string(index=False))
    O.to_csv(os.path.join(RES, "spatial_neff.csv"), index=False)


if __name__ == "__main__":
    main()
