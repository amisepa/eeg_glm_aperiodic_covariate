"""Spectra with known ground truth, synthesised in the time domain."""
import numpy as np


def welch_spectrum(S_of_f, rng, srate=250.0, dur=120.0, win_sec=4.0):
    """Welch estimate (Hann, 50% overlap) of a Gaussian process whose
    one-sided PSD is S_of_f(f). Returns (f, P)."""
    n = int(srate * dur)
    fg = np.fft.rfftfreq(n, 1 / srate)
    S = np.zeros_like(fg)
    S[1:] = S_of_f(fg[1:])
    sigma = np.sqrt(S * srate * n / 2.0)
    X = sigma * (rng.standard_normal(fg.size) + 1j * rng.standard_normal(fg.size)) / np.sqrt(2)
    if n % 2 == 0:
        X[-1] = sigma[-1] * rng.standard_normal()
    x = np.fft.irfft(X, n=n)
    nper = int(win_sec * srate)
    win = np.hanning(nper + 1)[:nper]
    scale = 2.0 / (srate * np.sum(win ** 2))
    segs = np.array([x[s:s + nper] for s in range(0, n - nper + 1, nper // 2)])
    segs -= segs.mean(1, keepdims=True)
    P = np.mean(np.abs(np.fft.rfft(segs * win, axis=1)) ** 2, 0) * scale
    return np.fft.rfftfreq(nper, 1 / srate), P


def gauss(f, cf, bw):
    return np.exp(-0.5 * ((f - cf) / bw) ** 2)
