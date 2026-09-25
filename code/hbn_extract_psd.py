"""Download HBN-EEG RestingState recordings and reduce each to Welch PSDs.

For every subject: fetch the .set file from the OpenNeuro S3 mirror, split
the recording into eyes-open and eyes-closed blocks from the instruction
markers (dropping the first 2 s after each cue), and compute per-channel
Welch PSDs per condition (4 s Hann, 50% overlap), plus odd/even segment
splits and per-block PSDs. One .npz is written per subject; subjects already
done are skipped, so the script can be restarted.

Output directory: $HBN_OUT (default ./hbn_psd).

Usage:
    python hbn_extract_psd.py ds005505 [ds005506 ...] --workers 6 [--limit N]
"""
import argparse
import csv
import io
import os
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

S3 = "https://s3.amazonaws.com/openneuro.org"
OUT = os.environ.get("HBN_OUT", "hbn_psd")
TMP = os.environ.get("HBN_TMP", "hbn_tmp")

# analysis settings
SRATE_TARGET = 250.0      # decimate; we only fit 2-45 Hz
WIN_SEC = 4.0
OVERLAP = 0.5
FRANGE = (1.0, 60.0)      # stored range; fits use a subrange
EO_SKIP, EO_LEN = 2.0, 17.0   # s after the open-eyes cue
EC_SKIP, EC_LEN = 2.0, 37.0   # s after the close-eyes cue
AMP_REJECT = 250.0        # uV, peak-to-peak per 4 s segment per channel
# posterior cluster, GSN-HydroCel-129; per-block spectra are stored for these
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]


def fetch(url, dest, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except Exception:
            if k == tries - 1:
                return False
            time.sleep(2 ** k)
    return False


def read_tsv(url, tries=4):
    # Retry like fetch() does; otherwise a transient network error looks like
    # a missing events file and the subject is silently dropped.
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                txt = r.read().decode("utf-8", "replace")
            return list(csv.DictReader(io.StringIO(txt), delimiter="\t"))
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(2 ** k)
    return None


def participants(ds):
    rows = read_tsv(f"{S3}/{ds}/participants.tsv")
    if rows is None:
        return {}
    out = {}
    for r in rows:
        pid = r.get("participant_id")
        if not pid:
            continue
        try:
            age = float(r.get("age", "nan"))
        except ValueError:
            age = float("nan")
        out[pid] = dict(age=age, sex=r.get("sex", ""),
                        resting=r.get("RestingState", ""),
                        release=r.get("release_number", ""))
    return out


def welch(x, srate, nper, noverlap):
    """One-sided PSD in uV^2/Hz per segment; returns (nseg, nch, nfreq)."""
    step = nper - noverlap
    nseg = 1 + (x.shape[1] - nper) // step if x.shape[1] >= nper else 0
    if nseg <= 0:
        return None, None
    win = np.hanning(nper + 1)[:nper]
    scale = 1.0 / (srate * np.sum(win ** 2))
    f = np.fft.rfftfreq(nper, 1.0 / srate)
    out = np.empty((nseg, x.shape[0], f.size), dtype=np.float32)
    for s in range(nseg):
        seg = x[:, s * step: s * step + nper]
        seg = seg - seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg * win, axis=1)
        p = (np.abs(X) ** 2) * scale * 2.0
        p[:, 0] /= 2.0
        if nper % 2 == 0:
            p[:, -1] /= 2.0
        out[s] = p
    return out, f


def blocks_psd(raw_data, srate, spans, nper, noverlap, roi_ix=None):
    """Welch segments over a list of (start, stop) sample spans.

    Returns the mean PSD, the odd/even segment split means, and -- for the ROI
    channels -- one PSD per BLOCK. The per-block spectra support the
    within-condition identification route, in which lambda is estimated from
    spontaneous block-to-block fluctuation in the aperiodic background rather
    than from a manipulation that moves the background and the oscillation at
    the same time.
    """
    segs, per_block = [], []
    for a, b in spans:
        if b - a < nper:
            continue
        p, f = welch(raw_data[:, a:b], srate, nper, noverlap)
        if p is not None:
            segs.append(p)
            if roi_ix is not None:
                per_block.append(p[:, roi_ix, :].mean(axis=(0, 1)))
    if not segs:
        return None, None, None, None, None
    P = np.concatenate(segs, axis=0)
    # amplitude-based segment rejection: drop segments whose broadband power
    # is a gross outlier in any channel
    med = np.median(P, axis=0, keepdims=True)
    ratio = np.max(P / np.maximum(med, 1e-12), axis=(1, 2))
    keep = ratio < 50.0
    if keep.sum() >= 4:
        P = P[keep]
    _, f = welch(raw_data[:, spans[0][0]:spans[0][0] + nper], srate, nper, noverlap)
    blocks = np.asarray(per_block, dtype=np.float32) if per_block else None
    return P.mean(0), P[0::2].mean(0), P[1::2].mean(0), (f, P.shape[0]), blocks


def process(args):
    ds, sub, meta = args
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    outfile = os.path.join(OUT, f"{ds}_{sub}.npz")
    if os.path.exists(outfile):
        return sub, "cached"

    base = f"{S3}/{ds}/{sub}/eeg/{sub}_task-RestingState"
    ev = read_tsv(base + "_events.tsv")
    if not ev:
        return sub, "no-events"
    setf = os.path.join(TMP, f"{sub}_RestingState.set")
    if not fetch(base + "_eeg.set", setf):
        return sub, "download-failed"

    try:
        import mne
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(setf, preload=True, verbose="ERROR")
            raw.pick("eeg")
            srate0 = raw.info["sfreq"]
            dec = max(1, int(round(srate0 / SRATE_TARGET)))
            raw.filter(0.4, min(SRATE_TARGET / 2 - 5, 100.0), fir_design="firwin",
                       verbose="ERROR")
            if dec > 1:
                raw.resample(srate0 / dec, verbose="ERROR")
            srate = raw.info["sfreq"]
            raw.set_eeg_reference("average", verbose="ERROR")
            X = raw.get_data() * 1e6          # uV
            ch = list(raw.ch_names)

        nper = int(round(WIN_SEC * srate))
        nov = int(round(OVERLAP * nper))

        def spans(cue, skip, length):
            out = []
            for r in ev:
                if r.get("value") != cue:
                    continue
                t0 = float(r["onset"]) + skip
                a, b = int(t0 * srate), int((t0 + length) * srate)
                if b <= X.shape[1]:
                    out.append((a, b))
            return out

        eo = spans("instructed_toOpenEyes", EO_SKIP, EO_LEN)
        ec = spans("instructed_toCloseEyes", EC_SKIP, EC_LEN)
        if len(eo) < 2 or len(ec) < 2:
            return sub, f"too-few-blocks eo={len(eo)} ec={len(ec)}"

        roi_ix = [ch.index(c) for c in ROI if c in ch]
        res = {}
        for name, sp in (("eo", eo), ("ec", ec)):
            full, odd, even, info, blocks = blocks_psd(
                X, srate, sp, nper, nov, roi_ix if len(roi_ix) >= 3 else None)
            if full is None:
                return sub, f"no-segments-{name}"
            res[name] = (full, odd, even, info, blocks)
        f = res["eo"][3][0]
        keep = (f >= FRANGE[0]) & (f <= FRANGE[1])

        np.savez_compressed(
            outfile,
            freqs=f[keep].astype(np.float32),
            ch_names=np.array(ch),
            eo=res["eo"][0][:, keep], eo_odd=res["eo"][1][:, keep],
            eo_even=res["eo"][2][:, keep],
            ec=res["ec"][0][:, keep], ec_odd=res["ec"][1][:, keep],
            ec_even=res["ec"][2][:, keep],
            n_seg_eo=res["eo"][3][1], n_seg_ec=res["ec"][3][1],
            eo_blocks=(res["eo"][4][:, keep] if res["eo"][4] is not None
                       else np.zeros((0, keep.sum()), np.float32)),
            ec_blocks=(res["ec"][4][:, keep] if res["ec"][4] is not None
                       else np.zeros((0, keep.sum()), np.float32)),
            roi_names=np.array([c for c in ROI if c in ch]),
            srate=srate, age=meta.get("age", np.nan), sex=meta.get("sex", ""),
            release=meta.get("release", ""), subject=sub, dataset=ds,
        )
        return sub, "ok"
    except Exception as e:
        return sub, f"error: {type(e).__name__}: {e}"
    finally:
        try:
            os.remove(setf)
        except OSError:
            pass
        for ext in (".fdt",):
            try:
                os.remove(setf.replace(".set", ext))
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    jobs = []
    for ds in a.datasets:
        pp = participants(ds)
        subs = [s for s, m in pp.items() if m["resting"] in ("available", "caution", "")]
        if a.limit:
            subs = subs[:a.limit]
        jobs += [(ds, s, pp[s]) for s in subs]
    print(f"{len(jobs)} subjects queued across {len(a.datasets)} datasets", flush=True)

    t0 = time.time()
    counts = {}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, j) for j in jobs]
        for fu in as_completed(futs):
            sub, status = fu.result()
            key = status.split(":")[0].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if done % 10 == 0 or status not in ("ok", "cached"):
                el = time.time() - t0
                print(f"[{done}/{len(jobs)}] {el/60:6.1f} min  {sub} -> {status}  {counts}",
                      flush=True)
    print(f"FINISHED {done} subjects in {(time.time()-t0)/60:.1f} min  {counts}", flush=True)


if __name__ == "__main__":
    main()
