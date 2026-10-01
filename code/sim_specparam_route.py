"""Does a specparam parameter table recover s_a, s_b and lambda* (from_specparam)?
Known-truth spectra, age moving the background; specparam modes and fit ranges
against a censored fit. Usage: python code/sim_specparam_route.py [n] [real]
(real = knee + plateau background)."""
import os
import sys
import warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lambdacurve", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "lambdacurve", "tests"))
from _sim import welch_spectrum, gauss
from lambdacurve import from_specparam, band_power
from specparam import SpectralModel
warnings.filterwarnings("ignore")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
REAL = len(sys.argv) > 2          # knee + plateau background
CONFIGS = [("fixed", (1, 40)), ("fixed", (2, 40)), ("fixed", (3, 40)), ("fixed", (2, 30)),
           ("fixed", (1, 45)), ("knee", (1, 40)), ("knee", (2, 40))]


def slope(y, x):
    ok = np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0], ok.mean()


def run(lam, seed):
    rng = np.random.default_rng(seed)
    age = rng.uniform(16, 75, N)
    chi = 1.7 - 0.002 * (age - 16) + rng.normal(0, 0.15, N)       # flattening with age
    off = 1.5 - 0.012 * (age - 16) + rng.normal(0, 0.25, N)
    cf = 10.6 - 0.025 * (age - 16) + rng.normal(0, 0.6, N)        # IAF slows with age
    c = np.exp(rng.normal(0, 0.3, N))                             # intrinsic strength, age-free
    knee = 8.0 if REAL else 0.0
    L = lambda f, i: 10 ** off[i] / (knee + f ** chi[i])
    bref = np.exp(np.mean([np.log(L(cf[i], i)) for i in range(N)]))
    rows = {k: [] for k in CONFIGS + ["censored"]}
    truth = []
    for i in range(N):
        b_cf = L(cf[i], i)
        a_cf = c[i] * 1.0 * bref * (b_cf / bref) ** lam            # a = c b^lam at the peak
        h2 = 0.25                                                  # harmonic, relative height
        plateau = 0.02 * bref if REAL else 0.0
        S = lambda f: (L(f, i) + a_cf * gauss(f, cf[i], 1.2)
                       + h2 * L(2 * cf[i], i) * gauss(f, 2 * cf[i], 1.5) + plateau)
        f, P = welch_spectrum(S, rng)
        truth.append((np.log(a_cf), np.log(b_cf), np.log(L(10.0, i))))
        for mode, fr in CONFIGS:
            sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6, aperiodic_mode=mode, verbose=False)
            try:
                sm.fit(f, P, list(fr))
                ap = sm.get_params("aperiodic")
                pk = np.atleast_2d(sm.get_params("peak"))
                j = np.where((pk[:, 0] >= 7) & (pk[:, 0] <= 14))[0] if np.isfinite(pk).all() else []
                if len(j):
                    k = j[np.argmax(pk[j, 1])]
                    la, lb = from_specparam(ap[0], ap[-1], pk[k, 0], pk[k, 1],
                                            knee=ap[1] if mode == "knee" else None)
                else:
                    la = lb = np.nan
            except Exception:
                la = lb = np.nan
            rows[(mode, fr)].append((la, lb))
        bp = band_power(P, f, band=(cf[i] - 0.5, cf[i] + 0.5), fit_range=(2, 40),
                        censor=((6, 16), (17, 25)))
        rows["censored"].append((np.log(bp["a"]) if bp["a"] > 0 else np.nan, np.log(bp["b"])))
    T = np.array(truth)
    sa_t, sb_t, sb10_t = slope(T[:, 0], age)[0], slope(T[:, 1], age)[0], slope(T[:, 2], age)[0]
    print(f"\nlambda_true = {lam}   truth: s_a {sa_t:+.4f}  s_b(at CF) {sb_t:+.4f}  "
          f"s_b(at 10 Hz) {sb10_t:+.4f}  lambda* {sa_t / sb_t:+.2f}")
    print(f"{'estimator':22s} {'kept':>5s} {'s_a':>8s} {'s_b':>8s} {'lam*':>6s} {'med ln a err':>13s} {'med ln b err':>13s}")
    for k, v in rows.items():
        v = np.array(v, float)
        sa, kept = slope(v[:, 0], age)
        ok = np.isfinite(v[:, 0])
        sb = np.polyfit(age[ok], v[ok, 1], 1)[0]
        name = k if isinstance(k, str) else f"specparam {k[0]} {k[1][0]}-{k[1][1]}"
        print(f"{name:22s} {kept:5.2f} {sa:+8.4f} {sb:+8.4f} {sa / sb:+6.2f} "
              f"{np.nanmedian(v[:, 0] - T[:, 0]):+13.3f} {np.nanmedian(v[:, 1] - T[:, 1]):+13.3f}")


for lam in (0.0, 1.0):
    run(lam, seed=int(10 * lam) + 1)
