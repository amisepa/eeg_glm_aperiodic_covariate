"""How much data would a single spectrum need to distinguish lambda = 1 from 0?

For a noise-free spectrum generated with lambda = 1 (a Gaussian peak scaled by
the power-law background), fit the lambda = 0 model (power law plus an
additive Gaussian) by minimising the Whittle deviance. The minimum deviance
per independent Welch segment, d1, is the noncentrality that one segment
contributes to a likelihood-ratio test of lambda = 0 against lambda = 1, so a
test with 80% power at alpha = 0.05 (1 df, noncentrality 7.85) needs
K = 7.85 / d1 independent segments. With 4 s Hann windows at 50% overlap,
Welch's effective number of independent segments is about 1.8 per 4 s of
data (Welch 1967), so the recording length is K * 4 / 1.8 seconds.

Frequencies 2-30 Hz at 0.25 Hz resolution. Peak height is the peak's
height relative to the background at its centre frequency. Also reported:
the largest difference between the lambda = 1 periodic component and the best
plain Gaussian fitted to it, as a percentage of peak height (the shape
information that could distinguish the two).

Usage: python identifiability_power.py
"""
import itertools

import numpy as np
from scipy.optimize import minimize

F = np.arange(2.0, 30.0001, 0.25)
NC80 = 7.85           # noncentrality for 80% power, 1 df, alpha = 0.05


def spectrum(off, chi, amp_rel, cf, bw, lam):
    L = 10 ** off / F ** chi
    Lcf = 10 ** off / cf ** chi
    amp = amp_rel * Lcf / Lcf ** lam      # peak height = amp_rel x background at cf
    return L + amp * np.exp(-0.5 * ((F - cf) / bw) ** 2) * L ** lam


def dev_per_segment(P):
    def mu(th):
        off, chi, la, cf, lbw = th
        return 10 ** off / F ** chi + 10 ** la * np.exp(-0.5 * ((F - cf) / 10 ** lbw) ** 2)

    def dev(th):
        r = P / np.maximum(mu(th), 1e-300)
        return 2 * np.sum(r - np.log(r) - 1)

    best = np.inf
    for cf0 in (9.5, 10.0, 10.5):
        for lbw0 in (np.log10(1.0), np.log10(1.5), np.log10(2.5)):
            b = np.polyfit(np.log10(F), np.log10(P), 1)
            i = np.argmin(np.abs(F - cf0))
            th0 = [b[1], -b[0], np.log10(max(P[i] - 10 ** b[1] / F[i] ** -b[0], 1e-6)),
                   cf0, lbw0]
            r = minimize(dev, th0, method="Nelder-Mead",
                         options=dict(maxiter=20000, xatol=1e-9, fatol=1e-12))
            best = min(best, r.fun)
    return best


def shape_residual(chi, bw):
    """Max |lambda=1 peak - best Gaussian| as % of peak height."""
    f = np.linspace(2, 30, 4000)
    pk = np.exp(-0.5 * ((f - 10.0) / bw) ** 2) * f ** -chi
    def res(th):
        return np.sum((th[0] * np.exp(-0.5 * ((f - th[1]) / th[2]) ** 2) - pk) ** 2)
    th = minimize(res, [pk.max(), 9.7, bw], method="Nelder-Mead",
                  options=dict(xatol=1e-10, fatol=1e-18, maxiter=40000)).x
    g = th[0] * np.exp(-0.5 * ((f - th[1]) / th[2]) ** 2)
    return 100 * np.abs(pk - g).max() / pk.max()


def main():
    print(f"{'exponent':>8} {'bw (Hz)':>7} {'peak/bg':>7} {'shape %':>8} "
          f"{'dev/segment':>12} {'segments K':>11} {'minutes':>8}")
    for chi, bw, h in itertools.product((1.0, 1.4, 1.5, 2.0), (1.0, 1.5, 2.5),
                                        (0.5, 1.0, 3.0)):
        P = spectrum(1.0, chi, h, 10.0, bw, lam=1.0)
        d1 = dev_per_segment(P)
        K = NC80 / d1
        print(f"{chi:>8.1f} {bw:>7.1f} {h:>7.1f} {shape_residual(chi, bw):>8.2f} "
              f"{d1:>12.2e} {K:>11.0f} {K * 4 / 1.8 / 60:>8.1f}")


if __name__ == "__main__":
    main()
