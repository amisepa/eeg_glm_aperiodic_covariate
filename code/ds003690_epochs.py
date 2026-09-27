"""Epoch-level spectra for the within-session coupling estimate (ds003690).

OpenNeuro ds003690 (Ribeiro & Castelo-Branco; CC0): 36 young and 39 older
adults, passive task, about 4.5 min of eyes-open fixation with a 250-ms tone
every 7-13 s; 61 EEG channels at 500 Hz, vertical and horizontal EOG, ECG and
pupil diameter in the same EEGLAB file.

Each tone-free interval (from 1.5 s after a tone to the next tone) is cut
into non-overlapping 2-s epochs. For each epoch: the posterior ROI spectrum
(O1, Oz, O2, PO3, POz, PO4) from each of three DPSS tapers (NW = 2), which
give nearly independent estimates from the same data; the mean pupil
diameter; log power of VEOG and HEOG at 0.5-4 Hz (eye movements); and log
power at 60-95 Hz over temporal sites (T7, T8, TP7, TP8, FT7, FT8; muscle).
Preprocessing as psd_utils (0.4-100 Hz, 250 Hz, average reference over EEG).

Output: one .npz per participant in $DS003690_OUT (default ./ds003690_epochs).

Usage: python ds003690_epochs.py [--data DIR] [--workers 6]
"""
import argparse
import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

import psd_utils as U

OUT = os.environ.get("DS003690_OUT", "ds003690_epochs")
POST = ["O1", "Oz", "O2", "PO3", "POz", "PO4"]
TEMP = ["T7", "T8", "TP7", "TP8", "FT7", "FT8"]
EPOCH = 2.0
GAP = 1.5


def one(args):
    path, meta = args
    sub = os.path.basename(path).split("_")[0]
    outfile = os.path.join(OUT, f"{sub}.npz")
    if os.path.exists(outfile):
        return sub, "cached"
    import mne
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        types = {}
        for c in raw.ch_names:
            u = c.upper()
            if u.startswith("VEO") or u.startswith("HEO"):
                types[c] = "eog"
            elif u.startswith("EKG") or u.startswith("ECG"):
                types[c] = "ecg"
            elif "DIA" in u:
                types[c] = "misc"
        raw.set_channel_types(types)
        pupil = raw.copy().pick("misc").get_data() if "misc" in raw else None
        sr0 = raw.info["sfreq"]
        ev = pd.read_csv(path.replace("_eeg.set", "_events.tsv"), sep="\t")
        cues = ev.loc[ev.trial_type == "cue", "onset"].to_numpy(float)
        raw.pick(["eeg", "eog"])
        x, sr, names = U.preprocess(raw)
        n = int(EPOCH * sr)
        tapers = dpss(n, 2.0, 3)                                  # 3 x n
        f = np.fft.rfftfreq(n, 1 / sr)
        ipost = [names.index(c) for c in POST if c in names]
        itemp = [names.index(c) for c in TEMP if c in names]
        ieog = [names.index(c) for c in names if c.upper().startswith(("VEO", "HEO"))]
        bounds = list(zip(cues + GAP, np.r_[cues[1:], cues[-1] + 8.0]))
        rows, spec = [], []
        for k, (a, b) in enumerate(bounds):
            for s in np.arange(a, b - EPOCH + 1e-9, EPOCH):
                i0 = int(s * sr)
                seg = x[:, i0:i0 + n]
                if seg.shape[1] < n:
                    continue
                seg = seg - seg.mean(1, keepdims=True)
                P = np.abs(np.fft.rfft(seg[None, :, :] * tapers[:, None, :], axis=2)) ** 2 / sr
                P[:, :, 1:-1] *= 2                                  # one-sided, uV^2/Hz
                post = P[:, ipost].mean(1)                           # taper x f
                eogp = np.log(P.mean(0)[ieog][:, (f >= 0.5) & (f <= 4)].mean(1))
                emg = np.log(P.mean(0)[itemp][:, (f >= 60) & (f <= 95)].mean())
                pu = np.nan
                if pupil is not None:
                    j0, j1 = int(s * sr0), int((s + EPOCH) * sr0)
                    v = pupil[:, j0:j1]
                    v = v[np.isfinite(v) & (v > 0)]
                    pu = float(np.mean(v)) if v.size > 0.5 * (j1 - j0) else np.nan
                rows.append(dict(t=s, interval=k, pupil=pu, emg=emg,
                                 veog=eogp[0] if len(eogp) > 0 else np.nan,
                                 heog=eogp[1] if len(eogp) > 1 else np.nan))
                spec.append(post.astype(np.float32))
        if not rows:
            return sub, "no-epochs"
        R = pd.DataFrame(rows)
        np.savez_compressed(outfile, freqs=f.astype(np.float32), post=np.stack(spec),
                            subject=sub, age=float(meta["age"]), sex=meta["sex"],
                            group=meta["group"], **{c: R[c].to_numpy() for c in R.columns})
        return sub, f"ok {len(R)}"
    except Exception as e:
        return sub, f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"), "ds003690"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(os.path.join(a.data, "participants.tsv"), sep="\t").set_index("participant_id")
    files = sorted(glob.glob(os.path.join(a.data, "sub-*", "eeg", "*task-passive*_eeg.set")))
    jobs = [(p, meta.loc[os.path.basename(p).split("_")[0]].to_dict()) for p in files]
    print(f"{len(jobs)} recordings", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(one, j) for j in jobs]):
            sub, st = fu.result()
            if not st.startswith(("ok", "cached")):
                print(sub, st, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
