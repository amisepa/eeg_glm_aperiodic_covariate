"""Shared preprocessing and Welch spectra for the resting-state extractors.

Same settings as hbn_extract_psd.py: 0.4-100 Hz band-pass, decimation to
250 Hz, average reference, 4-s Hann windows with 50% overlap, and rejection
of any segment whose power at any channel and frequency exceeds 50 times the
recording's median. Unlike the HBN extractor, recordings in which fewer than
MIN_KEEP segments pass are not kept whole: the rejection is applied and the
recording is flagged, so that uncleaned data cannot enter an analysis
unnoticed.
"""
import warnings

import numpy as np

SRATE_TARGET = 250.0
WIN_SEC = 4.0
OVERLAP = 0.5
REJECT_RATIO = 50.0
MIN_KEEP = 10
FRANGE = (1.0, 100.0)


def preprocess(raw, l_freq=0.4, h_freq=100.0):
    """MNE Raw (preloaded) -> (data in uV, srate, channel names)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        srate0 = raw.info["sfreq"]
        dec = max(1, int(round(srate0 / SRATE_TARGET)))
        raw.filter(l_freq, min(srate0 / dec / 2 - 5, h_freq), fir_design="firwin",
                   verbose="ERROR")
        if dec > 1:
            raw.resample(srate0 / dec, verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")
        return raw.get_data() * 1e6, raw.info["sfreq"], list(raw.ch_names)


def welch_segments(x, srate, spans=None):
    """Hann-windowed periodograms of every 4-s segment inside the spans.

    x: channels x samples (uV); spans: list of (start, stop) samples, default
    the whole recording. Returns P (segments x channels x freqs, uV^2/Hz),
    the frequencies, and the index of the span each segment came from.
    """
    nper = int(round(WIN_SEC * srate))
    step = nper - int(round(OVERLAP * nper))
    win = np.hanning(nper + 1)[:nper]
    scale = 1.0 / (srate * np.sum(win ** 2))
    f = np.fft.rfftfreq(nper, 1.0 / srate)
    spans = spans or [(0, x.shape[1])]
    out, src = [], []
    for j, (a, b) in enumerate(spans):
        b = min(b, x.shape[1])                   # a block may run past the recording
        for s in range(a, b - nper + 1, step):
            seg = x[:, s:s + nper]
            seg = seg - seg.mean(axis=1, keepdims=True)
            p = np.abs(np.fft.rfft(seg * win, axis=1)) ** 2 * scale * 2.0
            p[:, 0] /= 2.0
            p[:, -1] /= 2.0
            out.append(p.astype(np.float32))
            src.append(j)
    if not out:
        return None, f, np.array([], int)
    return np.stack(out), f, np.array(src)


def reject(P, ch_mask=None):
    """Keep mask: segments without a gross outlier at any channel/frequency.

    ch_mask selects the channels that count (e.g. EEG but not EOG).
    """
    Q = P if ch_mask is None else P[:, ch_mask]
    med = np.median(Q, axis=0, keepdims=True)
    ratio = np.max(Q / np.maximum(med, 1e-12), axis=(1, 2))
    return ratio < REJECT_RATIO


def summarise(P, f, keep, groups, ch_names):
    """Mean spectra (all channels) for all kept segments and their odd/even
    halves, plus per-segment spectra averaged within channel groups.

    groups: {name: [channel names]}; channels missing from the recording are
    skipped. Frequencies are restricted to FRANGE.
    """
    fm = (f >= FRANGE[0]) & (f <= FRANGE[1])
    K = P[keep][:, :, fm]
    out = dict(full=K.mean(0), odd=K[0::2].mean(0), even=K[1::2].mean(0),
               n_seg=int(keep.sum()), n_seg_total=int(keep.size),
               flag_few=bool(keep.sum() < MIN_KEEP))
    for g, names in groups.items():
        ix = [ch_names.index(c) for c in names if c in ch_names]
        out[f"seg_{g}"] = (P[:, ix][:, :, fm].mean(1) if ix
                           else np.zeros((P.shape[0], fm.sum()), np.float32))
    out["keep"] = keep
    return out, f[fm]
