"""What the coupling exponent means in a synaptic model of the EEG.

The signal is modelled as population spike trains filtered by postsynaptic
current kernels (Gao, Peterson and Voytek 2017; Brake et al. 2024), seen
through a fixed tissue filter D(f) and a gain G:

    V(t) = G * sum_s w_s sum_k g_k (x_k * h_k)(t),   k = E (AMPA), I (GABA_A)

x_k is the spike train of N_k Poisson neurons firing at rho * r_k *
(1 + m h_k x_s(t)), with x_s a unit-variance narrowband (alpha) modulation
of source s, h_k in {0, 1} saying which population carries the rhythm. The
expected one-sided spectrum of one source is

    aperiodic  B(f) = 2 G^2 w^2 D(f) sum_k g_k^2 |H_k(f)|^2 rho N_k r_k
    periodic   A(f) = G^2 w^2 D(f) |sum_k g_k H_k(f) rho N_k r_k m h_k|^2 S_x(f)

and sources add. The coupling exponent of a change in a parameter theta is
the ratio of elasticities d ln A / d ln B over the alpha band, so it is set
by which parameter differs between the conditions compared:

    gain G, or a synaptic gain scaling g_E and g_I together      1
    drive rho, rhythm a fixed fraction of it (m fixed)           2
    drive rho, rhythm of fixed absolute size (rho * m fixed)     0
    inhibitory weight g_I, rhythm carried by I                   1 / s_I
    inhibitory weight g_I, rhythm carried by E                   0
    rhythm from a separate generator, its gain unchanged         0

(s_I, the inhibitory share of the background in the band). With several
parameters or sources changing, an estimate of lambda is the average of these
values weighted by each one's share of the variance of the change in ln B,
and a changing source that carries more of the rhythm than of the background
pushes it above 1.

Each scenario simulates n units in two conditions: condition 2 changes the
scenario's parameter by a unit-specific amount, and the rhythm's intrinsic
strength doubles in every unit. Spectra are Welch estimates (4-s Hann, 50%
overlap) of the simulated signal, split into odd and even segments.
Reported per scenario:
    analytic     elasticity ratio at the baseline parameters
    ols_true     slope of the change in ln A on the change in ln B across
                 units, from the expected spectra (what an ideal estimator
                 of a single lambda returns)
    lf_true      log-free estimator (lambda_gmm) with the true aperiodic power
    lf_fixed, lf_knee
                 log-free estimator with the aperiodic power from a censored
                 fit (2-40 Hz, 6-16 Hz left out; power law or knee model)

Writes results/sim_mechanisms.csv.

Usage: python sim_mechanisms.py [--n 300] [--dur 60] [--workers 4]
       [--scenarios gain,tau_i,...] [--nboot 200]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lambda_gmm
from ap_models import ap_band_power, fit_aperiodic
from lambda_curve import fit_mask

FS = 500.0                       # spike counts in 2-ms bins
WIN_SEC = 4.0
FIT = (2.0, 40.0)
CENSOR = ((6.0, 16.0),)
BAND = (8.0, 12.0)
CF, BW = 10.0, 1.5
D_EXP = 1.0                      # fixed tissue low-pass, power ~ 1/f
GRID = np.linspace(-1.0, 3.0, 161)

# neurons, rate (Hz), PSC rise and decay (s); Gao et al. 2017
POP = {"E": dict(N=8000, r=2.0, tr=0.1e-3, td=2e-3),
       "I": dict(N=2000, r=5.0, tr=0.5e-3, td=10e-3)}
BASE = dict(w=1.0, gE=1.0, gI=2.6, tdI=10e-3, rho=1.0, m=0.03, hE=0.0, hI=1.0)


def kernels(f, src):
    w = 2j * np.pi * f
    HE = 1.0 / ((1 + w * POP["E"]["td"]) * (1 + w * POP["E"]["tr"]))
    HI = 1.0 / ((1 + w * src["tdI"]) * (1 + w * POP["I"]["tr"]))
    return HE, HI


def rhythm_density(f):
    """One-sided spectral density of a unit-variance alpha modulation."""
    return np.exp(-0.5 * ((f - CF) / BW) ** 2) / (BW * np.sqrt(2 * np.pi))


def expected(f, sources, G):
    """Expected aperiodic and periodic one-sided spectra (arbitrary units)."""
    D = np.maximum(f, 1e-9) ** -D_EXP
    B = np.zeros(f.size)
    A = np.zeros(f.size)
    for s in sources:
        HE, HI = kernels(f, s)
        lE = s["rho"] * POP["E"]["N"] * POP["E"]["r"]
        lI = s["rho"] * POP["I"]["N"] * POP["I"]["r"]
        B += 2 * s["w"] ** 2 * (s["gE"] ** 2 * np.abs(HE) ** 2 * lE
                                + s["gI"] ** 2 * np.abs(HI) ** 2 * lI)
        amp = s["m"] * (s["gE"] * HE * lE * s["hE"] + s["gI"] * HI * lI * s["hI"])
        A += s["w"] ** 2 * np.abs(amp) ** 2 * rhythm_density(f)
    return G ** 2 * D * B, G ** 2 * D * A


def band_mean(f, S, band=BAND):
    m = (f >= band[0]) & (f <= band[1])
    return float(np.mean(S[m]))


def simulate(sources, G, dur, rng):
    """Welch spectra (odd and even segments) of one simulated recording."""
    n = int(dur * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    D = np.zeros(f.size)
    D[1:] = f[1:] ** (-D_EXP / 2)
    V = np.zeros(f.size, complex)
    for s in sources:
        # narrowband modulation, unit variance
        X = np.fft.rfft(rng.standard_normal(n)) * np.sqrt(np.exp(-0.5 * ((f - CF) / BW) ** 2))
        x = np.fft.irfft(X, n)
        x /= x.std()
        HE, HI = kernels(f, s)
        for k, H, g, h in (("E", HE, s["gE"], s["hE"]), ("I", HI, s["gI"], s["hI"])):
            rate = s["rho"] * POP[k]["N"] * POP[k]["r"] * (1 + s["m"] * h * x)
            counts = rng.poisson(np.clip(rate, 0, None) / FS)
            spk = (counts - counts.mean()) * FS
            V += s["w"] * g * H * np.fft.rfft(spk)
    v = np.fft.irfft(G * D * V, n)
    nper = int(WIN_SEC * FS)
    step = nper // 2
    nseg = 1 + (n - nper) // step
    win = np.hanning(nper + 1)[:nper]
    scale = 2.0 / (FS * np.sum(win ** 2))
    fw = np.fft.rfftfreq(nper, 1 / FS)
    P = np.empty((nseg, fw.size))
    for i in range(nseg):
        seg = v[i * step: i * step + nper]
        P[i] = np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2 * scale
    return fw, P[0::2].mean(0), P[1::2].mean(0)


def fitted_b(f, P, model):
    sel = (f >= FIT[0]) & (f <= FIT[1])
    ff, PP = f[sel], P[sel]
    fit = fit_aperiodic(PP, ff, fit_mask(ff, CENSOR), model)
    return np.nan if fit is None else ap_band_power(fit, ff, *BAND)


# ---- scenarios -------------------------------------------------------------
# Each returns (sources_1, G_1, sources_2, G_2) for one unit, given its
# baseline rhythm depth m and the unit's change d (on ln power where that is
# natural, otherwise on the parameter's log).

def src(**kw):
    s = dict(BASE)
    s.update(kw)
    return s


def sc_gain(m, d):
    return [src(m=m)], 1.0, [src(m=m * np.sqrt(2))], np.exp(d / 2)


def sc_tau_i(m, d):
    return [src(m=m)], 1.0, [src(m=m * np.sqrt(2), tdI=BASE["tdI"] * np.exp(d))], 1.0


def sc_gi_rhythm_i(m, d):
    return [src(m=m)], 1.0, [src(m=m * np.sqrt(2), gI=BASE["gI"] * np.exp(d))], 1.0


def sc_gi_rhythm_e(m, d):
    # rhythm carried by E needs a larger depth for a comparable peak
    mE = 3 * m
    return ([src(m=mE, hE=1.0, hI=0.0)], 1.0,
            [src(m=mE * np.sqrt(2), hE=1.0, hI=0.0, gI=BASE["gI"] * np.exp(d))], 1.0)


def sc_drive_relative(m, d):
    return [src(m=m)], 1.0, [src(m=m * np.sqrt(2), rho=np.exp(d))], 1.0


def sc_drive_absolute(m, d):
    return [src(m=m)], 1.0, [src(m=m * np.sqrt(2) * np.exp(-d), rho=np.exp(d))], 1.0


def sc_separate_common_gain(m, d):
    bg = src(hE=0.0, hI=0.0)
    gen = dict(src(m=4 * m), w=0.5)
    gen2 = dict(gen, m=gen["m"] * np.sqrt(2))
    return [bg, gen], 1.0, [bg, gen2], np.exp(d / 2)


def sc_separate_background_drive(m, d):
    bg = src(hE=0.0, hI=0.0)
    gen = dict(src(m=4 * m), w=0.5)
    gen2 = dict(gen, m=gen["m"] * np.sqrt(2))
    return [bg, gen], 1.0, [dict(bg, rho=np.exp(d)), gen2], 1.0


def sc_synaptic_gain(m, d):
    k = np.exp(d / 2)
    return [src(m=m)], 1.0, [src(m=m * np.sqrt(2), gE=BASE["gE"] * k, gI=BASE["gI"] * k)], 1.0


def make_sources(phi_a):
    """Two sources with equal backgrounds; source A carries a share phi_a of
    the rhythm, and only A's synaptic gain changes (lambda_A = 1)."""
    def sc(m, d):
        # equal backgrounds, rhythm split phi_a : 1 - phi_a between A and B
        tot = 2 * m ** 2
        mA = np.sqrt(tot * phi_a)
        mB = np.sqrt(tot * (1 - phi_a))
        w = 1 / np.sqrt(2)
        k = np.exp(d / 2)
        A1, B1 = dict(src(m=mA), w=w), dict(src(m=mB), w=w)
        A2 = dict(A1, m=mA * np.sqrt(2), gE=BASE["gE"] * k, gI=BASE["gI"] * k)
        B2 = dict(B1, m=mB * np.sqrt(2))
        return [A1, B1], 1.0, [A2, B2], 1.0
    return sc


SCENARIOS = {
    "gain": (sc_gain, (0.4, 0.3)),
    "synaptic_gain": (sc_synaptic_gain, (0.4, 0.3)),
    "tau_i": (sc_tau_i, (0.3, 0.2)),
    "gi_rhythm_i": (sc_gi_rhythm_i, (0.2, 0.15)),
    "gi_rhythm_e": (sc_gi_rhythm_e, (0.2, 0.15)),
    "drive_relative": (sc_drive_relative, (0.3, 0.2)),
    "drive_absolute": (sc_drive_absolute, (0.3, 0.2)),
    "separate_common_gain": (sc_separate_common_gain, (0.4, 0.3)),
    "separate_background_drive": (sc_separate_background_drive, (0.3, 0.2)),
    "sources_phi0.2": (make_sources(0.2), (0.4, 0.3)),
    "sources_phi0.5": (make_sources(0.5), (0.4, 0.3)),
    "sources_phi0.9": (make_sources(0.9), (0.4, 0.3)),
}


def mix_unit(m, d_gain, d_drive):
    """Gain (lambda 1) and drive with a rhythm of fixed absolute size
    (lambda 0) changing together in one unit."""
    return ([src(m=m)], 1.0,
            [src(m=m * np.sqrt(2) * np.exp(-d_drive), rho=np.exp(d_drive))], np.exp(d_gain / 2))


def analytic(sc, m, eps=1e-4):
    """Elasticity ratio d ln A / d ln B at the baseline (band means): condition
    2 at d = eps against condition 2 at d = 0, so the intrinsic change cancels."""
    f = np.linspace(0.5, 60, 2400)
    B0, A0 = expected(f, *sc(m, 0.0)[2:])
    B1, A1 = expected(f, *sc(m, eps)[2:])
    return (np.log(band_mean(f, A1) / band_mean(f, A0))
            / np.log(band_mean(f, B1) / band_mean(f, B0)))


def run_unit(unit, dur, seed):
    rng = np.random.default_rng(seed)
    out = dict(unit)
    for c, (sources, G) in enumerate(((unit["s1"], unit["G1"]), (unit["s2"], unit["G2"]))):
        fw, Po, Pe = simulate(sources, G, dur, rng)
        B, A = expected(fw, sources, G)
        out[f"b_true{c}"] = band_mean(fw, B)
        out[f"a_true{c}"] = band_mean(fw, A)
        for h, P in (("o", Po), ("e", Pe)):
            out[f"t{c}{h}"] = band_mean(fw, P)
            out[f"bf{c}{h}"] = fitted_b(fw, P, "fixed")
            out[f"bk{c}{h}"] = fitted_b(fw, P, "knee")
    del out["s1"], out["s2"]
    return out


def estimates(D, nboot, seed):
    """All lambda estimates for one scenario's unit table."""
    arr = lambda p: np.column_stack([D[f"{p}o"], D[f"{p}e"]])
    t1, t2 = arr("t0"), arr("t1")
    res = {}
    dA = np.log(D.a_true1) - np.log(D.a_true0)
    dB = np.log(D.b_true1) - np.log(D.b_true0)
    res["ols_true"] = float(np.polyfit(dB, dA, 1)[0])
    res["sd_dlnb"] = float(np.std(dB))
    bt = lambda c: np.column_stack([D[f"b_true{c}"]] * 2)
    for name, b1, b2 in (("lf_true", bt(0), bt(1)),
                         ("lf_fixed", arr("bf0"), arr("bf1")),
                         ("lf_knee", arr("bk0"), arr("bk1"))):
        r = lambda_gmm.bootstrap(t1, t2, b1, b2, nboot=nboot,
                                 rng=np.random.default_rng(seed), grid=GRID)
        res[name] = r["lam"]
        res[name + "_lo"], res[name + "_hi"] = r["ci"]
        res[name + "_fail"] = r["boot_fail"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--mix", default="0.25,0.5,0.75",
                    help="shares of the ln B change variance from gain in the gain + drive mixture")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "sim_mechanisms.csv"))
    a = ap.parse_args()
    names = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
    mixes = [float(x) for x in a.mix.split(",")] if a.mix else []
    rows = []
    for si, name in enumerate(names + [f"mix_gain{w:g}" for w in mixes]):
        t0 = time.time()
        rng = np.random.default_rng(1000 + si)
        units = []
        if name.startswith("mix_gain"):
            w = float(name[len("mix_gain"):])
            # d ln B from gain is d_gain; from drive it is ~ d_drive (B is linear in rho)
            sd = 0.3
            for i in range(a.n):
                m = 0.03 * 10 ** (0.12 * rng.standard_normal())
                dg = 0.2 + np.sqrt(w) * sd * rng.standard_normal()
                dd = 0.2 + np.sqrt(1 - w) * sd * rng.standard_normal()
                s1, g1, s2, g2 = mix_unit(m, dg, dd)
                units.append(dict(unit=i, m=m, d=dg, d2=dd, s1=s1, G1=g1, s2=s2, G2=g2))
            pred = w * 1.0 + (1 - w) * 0.0
        else:
            sc, (mu, sd) = SCENARIOS[name]
            for i in range(a.n):
                m = 0.03 * 10 ** (0.12 * rng.standard_normal())
                d = mu + sd * rng.standard_normal()
                s1, g1, s2, g2 = sc(m, d)
                units.append(dict(unit=i, m=m, d=d, s1=s1, G1=g1, s2=s2, G2=g2))
            pred = analytic(sc, 0.03)
        seeds = rng.integers(0, 2 ** 31, a.n)
        out = Parallel(n_jobs=a.workers)(delayed(run_unit)(u, a.dur, int(s))
                                         for u, s in zip(units, seeds))
        D = pd.DataFrame(out)
        res = estimates(D, a.nboot, si)
        f = np.linspace(0.5, 60, 2400)
        s1, g1 = units[0]["s1"], units[0]["G1"]
        B, A = expected(f, s1, g1)
        rel = band_mean(f, A) / band_mean(f, B)
        row = dict(scenario=name, n=a.n, analytic=pred, peak_rel_band=rel, **res)
        rows.append(row)
        print(f"{name:28s} analytic {pred:+.2f}  ols_true {res['ols_true']:+.2f}  "
              f"lf_true {res['lf_true']:+.2f} [{res['lf_true_lo']:+.2f}, {res['lf_true_hi']:+.2f}]  "
              f"lf_fixed {res['lf_fixed']:+.2f} [{res['lf_fixed_lo']:+.2f}, {res['lf_fixed_hi']:+.2f}]  "
              f"lf_knee {res['lf_knee']:+.2f} [{res['lf_knee_lo']:+.2f}, {res['lf_knee_hi']:+.2f}]  "
              f"sd dlnB {res['sd_dlnb']:.2f}  a/b {rel:.2f}  ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
