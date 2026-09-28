"""Epoch-level spectra of intracranial resting recordings (OpenNeuro ds003688).

Berezutskaya et al. 2022 (CC0): patients with epilepsy implanted with
subdural grids and strips or depth electrodes, 3 min of rest in the
monitoring unit (Micromed, 2048 Hz), with EOG and EMG channels. Contacts
marked good in channels.tsv are re-referenced as bipolar pairs of
consecutive contacts of the same electrode group that lie within 15 mm of
each other (electrodes.tsv, ACPC space), which removes the recording
reference and keeps signals local. Then 0.4-100 Hz band-pass, decimation to
256 Hz, non-overlapping 2-s epochs within the rest period, and for every
epoch: the spectrum of each bipolar channel from each of three DPSS tapers
(NW = 2) up to 60 Hz; log power of the EOG channel at 0.5-4 Hz and of the
EMG channels at 60-95 Hz (eye movements, muscle); and the pair's midpoint.
There is no skull, so scalp gain and scalp muscle do not enter.

Output: one .npz per recording in $IEEG_OUT (default ./ieeg_epochs).

Usage: python ieeg_epochs.py [--data DIR] [--workers 4]
"""
import argparse
import glob
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

OUT = os.environ.get("IEEG_OUT", "ieeg_epochs")
EPOCH = 2.0
FMAX = 60.0
SRATE = 256.0
MAX_MM = 15.0


def pairs(chan, elec):
    """Bipolar pairs (a, b, midpoint) of neighbouring good contacts."""
    good = chan[(chan.status == "good") & chan.type.isin(["SEEG", "ECOG"])]
    pos = elec.set_index("name")[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
    out = []
    for grp, g in good.groupby("group"):
        num = g.name.map(lambda s: int(re.findall(r"(\d+)$", s)[0]) if re.findall(r"(\d+)$", s)
                         else -1)
        g = g.assign(num=num).sort_values("num")
        names = g.name.tolist()
        nums = g.num.tolist()
        for (a, na), (b, nb) in zip(zip(names, nums), zip(names[1:], nums[1:])):
            if nb != na + 1 or a not in pos.index or b not in pos.index:
                continue
            pa, pb = pos.loc[a].to_numpy(float), pos.loc[b].to_numpy(float)
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)) and \
                    np.linalg.norm(pa - pb) <= MAX_MM:
                out.append((a, b, (pa + pb) / 2))
    return out


def one(vhdr):
    base = os.path.basename(vhdr)
    sub = base.split("_")[0]
    outfile = os.path.join(OUT, base.replace("_ieeg.vhdr", ".npz"))
    if os.path.exists(outfile):
        return sub, "cached"
    import mne
    try:
        d = os.path.dirname(vhdr)
        chan = pd.read_csv(vhdr.replace("_ieeg.vhdr", "_channels.tsv"), sep="\t")
        elec = pd.read_csv(glob.glob(os.path.join(d, "*_electrodes.tsv"))[0], sep="\t")
        ev = pd.read_csv(glob.glob(os.path.join(d, base.split("_acq")[0] + "*_events.tsv"))[0],
                         sep="\t")
        prs = pairs(chan, elec)
        if not prs:
            return sub, "no-pairs"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        eog = chan.name[chan.type == "EOG"].tolist()
        emg = chan.name[chan.type == "EMG"].tolist()
        need = sorted({c for a, b, _ in prs for c in (a, b)} | set(eog) | set(emg))
        need = [c for c in need if c in raw.ch_names]
        raw.pick(need).load_data(verbose="ERROR")
        raw.set_channel_types({c: "eeg" for c in raw.ch_names}, verbose="ERROR")
        sr0 = raw.info["sfreq"]
        raw.filter(0.4, 100.0, fir_design="firwin", verbose="ERROR")
        raw.resample(SRATE, verbose="ERROR")
        x = raw.get_data() * 1e6
        ix = {c: i for i, c in enumerate(raw.ch_names)}
        prs = [p for p in prs if p[0] in ix and p[1] in ix]
        bip = np.stack([x[ix[a]] - x[ix[b]] for a, b, _ in prs])
        start = ev.onset[ev.trial_type.str.contains("start", case=False)].min()
        stop = ev.onset[ev.trial_type.str.contains("end", case=False)].max()
        start = 0.0 if not np.isfinite(start) else start
        stop = x.shape[1] / SRATE if not np.isfinite(stop) else stop
        n = int(EPOCH * SRATE)
        tapers = dpss(n, 2.0, 3)
        f = np.fft.rfftfreq(n, 1 / SRATE)
        fk = f <= FMAX
        spec, cov = [], []
        for s in np.arange(start, stop - EPOCH + 1e-9, EPOCH):
            i0 = int(round(s * SRATE))
            seg = bip[:, i0:i0 + n]
            if seg.shape[1] < n:
                continue
            seg = seg - seg.mean(1, keepdims=True)
            P = np.abs(np.fft.rfft(seg[None] * tapers[:, None, :], axis=2)) ** 2 / SRATE
            P[:, :, 1:-1] *= 2
            spec.append(P[:, :, fk].astype(np.float32))            # tapers x pairs x f
            row = []
            for chans, (lo, hi) in ((eog, (0.5, 4.0)), (emg, (60.0, 95.0))):
                cs = [ix[c] for c in chans if c in ix]
                if cs:
                    q = x[cs, i0:i0 + n]
                    q = q - q.mean(1, keepdims=True)
                    Q = np.abs(np.fft.rfft(q * tapers[0], axis=1)) ** 2 / SRATE
                    row.append(np.log(Q[:, (f >= lo) & (f <= hi)].mean()))
                else:
                    row.append(np.nan)
            cov.append(row)
        if not spec:
            return sub, "no-epochs"
        P = np.stack(spec)                                             # epochs x tapers x pairs x f
        # gross artefacts: any pair and frequency 50 times its median over epochs
        med = np.median(P.mean(1), axis=0, keepdims=True)
        bad = np.max(P.mean(1) / np.maximum(med, 1e-12), axis=(1, 2)) > 50
        np.savez_compressed(outfile, freqs=f[fk].astype(np.float32), P=P, bad=bad,
                            cov=np.asarray(cov, float),
                            pairs=np.array([f"{a}-{b}" for a, b, _ in prs]),
                            pos=np.stack([p for _, _, p in prs]), subject=sub,
                            srate_raw=sr0)
        return sub, f"ok {P.shape[0]} epochs, {len(prs)} pairs, {int(bad.sum())} flagged"
    except Exception as e:
        return sub, f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"),
                                                   "ds003688"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, "sub-*", "ses-*", "ieeg",
                                          "*task-rest*_ieeg.vhdr")))
    print(f"{len(files)} rest recordings", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for sub, st in ex.map(one, files):
            print(sub, st, flush=True)


if __name__ == "__main__":
    main()
