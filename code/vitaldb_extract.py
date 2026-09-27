"""Reduce VitalDB intraoperative BIS EEG to spectra during maintenance of
general anaesthesia.

VitalDB (Lee et al. 2022 Sci Data; PhysioNet doi:10.13026/czw8-9p62, CC BY
4.0): 6,388 surgical cases with two frontal BIS EEG channels at 128 Hz and
anaesthetic concentrations. For each adult general-anaesthesia case the
.vital file is fetched from the PhysioNet S3 mirror, parsed with the vitaldb
package and deleted. Window: surgery start + 5 min to surgery end - 5 min
(times from clinical_data.csv, relative to the start of the recording). 4-s
Hann segments with 50% overlap are kept when both channels are present, the
BIS signal quality index is >= 50, the suppression ratio is 0 (no burst
suppression, which is commoner in older patients), the amplitude stays
within 150 uV of the segment mean, and no frequency exceeds 50 times the
case median. Saved per case: mean spectrum per channel over kept segments,
over odd/even segments and the median spectrum; the medians over kept
segments of BIS, SQI, EMG, propofol and remifentanil effect-site
concentrations, end-tidal sevoflurane and desflurane, and MAC.

Output: $VITALDB_OUT (default ./vitaldb_psd); needs clinical_data.csv in
$VITALDB_META (default ./vitaldb). Finished cases are skipped.

Usage: python vitaldb_extract.py [--workers 6] [--limit N] [--min-age 18]
"""
import argparse
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

URL = "https://physionet-open.s3.amazonaws.com/vitaldb/1.0.0/vital_files/{:04d}.vital"
OUT = os.environ.get("VITALDB_OUT", "vitaldb_psd")
TMP = os.environ.get("VITALDB_TMP", "vitaldb_tmp")
META = os.environ.get("VITALDB_META", "vitaldb")
FS = 128.0
NPER = 512                      # 4 s
STEP = 256
MARGIN = 300.0                  # s after surgery start / before its end
FKEEP = (0.5, 47.0)
NUMERIC = {"bis": "BIS/BIS", "sqi": "BIS/SQI", "sr": "BIS/SR", "emg": "BIS/EMG",
           "ppf_ce": "Orchestra/PPF20_CE", "rftn20_ce": "Orchestra/RFTN20_CE",
           "rftn50_ce": "Orchestra/RFTN50_CE", "sevo_et": "Primus/EXP_SEVO",
           "des_et": "Primus/EXP_DES", "mac": "Primus/MAC"}


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
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == tries - 1:
                return False
            time.sleep(2 ** k)
        except Exception:
            if k == tries - 1:
                return False
            time.sleep(2 ** k)
    return False


def one(args):
    caseid, row = args
    outfile = os.path.join(OUT, f"case{caseid:04d}.npz")
    if os.path.exists(outfile):
        return caseid, "cached"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, f"{caseid:04d}.vital")
    try:
        if not fetch(URL.format(caseid), path):
            return caseid, "download-failed"
        import vitaldb
        vf = vitaldb.VitalFile(path, ["BIS/EEG1_WAV", "BIS/EEG2_WAV"] + list(NUMERIC.values()))
        if "BIS/EEG1_WAV" not in vf.get_track_names():
            return caseid, "no-eeg"
        x = vf.to_numpy(["BIS/EEG1_WAV", "BIS/EEG2_WAV"], 1 / FS).T      # 2 x n
        num = vf.to_numpy(list(NUMERIC.values()), 1)                       # n_s x k
        t0 = max(float(row["opstart"]) + MARGIN, 0.0)
        t1 = min(float(row["opend"]) - MARGIN, x.shape[1] / FS)
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 - t0 < 600:
            return caseid, "short-window"
        sqi = pd.Series(num[:, 1]).ffill().to_numpy()
        sr = pd.Series(num[:, 2]).ffill().to_numpy()
        win = np.hanning(NPER + 1)[:NPER]
        scale = 1.0 / (FS * np.sum(win ** 2))
        f = np.fft.rfftfreq(NPER, 1 / FS)
        fm = (f >= FKEEP[0]) & (f <= FKEEP[1])
        segs, times = [], []
        for s in range(int(t0 * FS), int(t1 * FS) - NPER, STEP):
            seg = x[:, s:s + NPER]
            if not np.all(np.isfinite(seg)):
                continue
            a, b = int(s / FS), int((s + NPER) / FS) + 1
            q, r = sqi[a:b], sr[a:b]
            if not (np.all(np.isfinite(q)) and np.min(q) >= 50):
                continue
            if not (np.all(np.isfinite(r)) and np.max(r) <= 0):
                continue
            seg = seg - seg.mean(1, keepdims=True)
            if np.max(np.abs(seg)) > 150:
                continue
            p = np.abs(np.fft.rfft(seg * win, axis=1)) ** 2 * scale * 2.0
            segs.append(p[:, fm].astype(np.float32))
            times.append(s / FS)
        if len(segs) < 30:
            return caseid, f"few-segments {len(segs)}"
        P = np.stack(segs)
        med = np.median(P, 0, keepdims=True)
        keep = np.max(P / np.maximum(med, 1e-12), axis=(1, 2)) < 50
        P, times = P[keep], np.array(times)[keep]
        idx = np.clip(times.astype(int), 0, num.shape[0] - 1)
        meds = {k: float(np.nanmedian(num[idx, j])) if np.any(np.isfinite(num[idx, j]))
                else np.nan for j, k in enumerate(NUMERIC)}
        np.savez_compressed(outfile, freqs=f[fm].astype(np.float32), full=P.mean(0),
                            odd=P[0::2].mean(0), even=P[1::2].mean(0),
                            median=np.median(P, 0), n_seg=P.shape[0],
                            window_s=t1 - t0, caseid=caseid, age=float(row["age"]),
                            sex=str(row["sex"]), **meds)
        return caseid, f"ok {P.shape[0]}"
    except Exception as e:
        return caseid, f"error {type(e).__name__}: {e}"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-age", type=float, default=18)
    a = ap.parse_args()
    C = pd.read_csv(os.path.join(META, "clinical_data.csv"))
    C["age"] = pd.to_numeric(C.age, errors="coerce")
    C = C[(C.ane_type == "General") & (C.age >= a.min_age)]
    if a.limit:
        C = C.head(a.limit)
    jobs = [(int(r["caseid"]), {k: r[k] for k in ("opstart", "opend", "age", "sex")})
            for r in C.to_dict("records")]
    print(f"{len(jobs)} cases queued", flush=True)
    t0 = time.time()
    counts = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            cid, st = fu.result()
            key = st.split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            if (k + 1) % 50 == 0 or key not in ("ok", "cached", "no-eeg"):
                print(f"[{k + 1}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min case {cid} "
                      f"{st} {counts}", flush=True)
    print(f"FINISHED {len(jobs)} in {(time.time() - t0) / 60:.1f} min {counts}", flush=True)


if __name__ == "__main__":
    main()
