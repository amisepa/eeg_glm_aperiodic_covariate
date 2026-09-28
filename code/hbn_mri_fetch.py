"""Fetch T1-weighted MRI and ANTs brain segmentations for an age-stratified
subset of the HBN participants whose resting EEG passed quality control.

Only the two Siemens Prisma sites (CBIC, RU) are used, so that scanner and
protocol do not vary with age. For each selected participant the HCP T1w
image, the ANTs brain-extraction mask, the ANTs six-class segmentation and
the brain volumes table are downloaded from the FCP-INDI bucket (about
15 MB per participant). The ANTs outputs are in the T1w's voxel space.

Output: $EEG_DATA/hbn_mri/<site>/<subject>/..., and hbn_mri_selected.csv
(subject, site, age) in the same directory.

Usage: python hbn_mri_fetch.py [--n 300] [--workers 6] [--root DIR]
"""
import argparse
import glob
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from fetch_open_data import fetch

BUCKET = "https://fcp-indi.s3.amazonaws.com"
MRI = "data/Projects/HBN/MRI"
SITES = ("Site-CBIC", "Site-RU")
AGE_EDGES = (5, 7, 9, 11, 13, 15, 18, 22.5)
ANTS_FILES = ("antsBrainExtractionMask.nii.gz", "antsBrainSegmentation.nii.gz",
              "antsbrainvols.csv")
HERE = os.path.dirname(os.path.abspath(__file__))


def _get(url, tries=6):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def s3_ls(prefix, delimiter="/"):
    """(sub-prefixes, [(key, size)]) under a prefix, all pages."""
    dirs, keys, tok = [], [], None
    while True:
        url = (f"{BUCKET}?list-type=2&prefix={urllib.parse.quote(prefix)}"
               f"&delimiter={delimiter}&max-keys=1000")
        if tok:
            url += "&continuation-token=" + urllib.parse.quote(tok)
        x = ET.fromstring(_get(url))
        ns = {"s": x.tag.split("}")[0].strip("{")}
        dirs += [c.find("s:Prefix", ns).text for c in x.findall("s:CommonPrefixes", ns)]
        keys += [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text))
                 for c in x.findall("s:Contents", ns)]
        if x.find("s:IsTruncated", ns).text != "true":
            return dirs, keys
        tok = x.find("s:NextContinuationToken", ns).text


def eeg_participants(root):
    """QC-passing HBN EEG participants with age, from the release participants.tsv files."""
    q = pd.read_csv(os.path.join(HERE, "..", "results", "hbn_qc_flags.csv"), index_col=0)
    ok = set(q.index[q.qc_ok])
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "hbn_participants", "*_participants.tsv"))):
        t = pd.read_csv(p, sep="\t")
        rows.append(t[["participant_id", "age"]])
    t = pd.concat(rows).drop_duplicates("participant_id").set_index("participant_id")
    return t.loc[t.index.intersection(sorted(ok))]


def mri_files(site, sub):
    """Matching raw T1w and ANTs derivative keys for one participant, or None.

    Prefers the single HCP T1w (derivative folder T1w_HCP, or ants/ directly
    under the participant at RU); with several runs, run-01.
    """
    dirs, _ = s3_ls(f"{MRI}/{site}/derivatives/{sub}/")
    names = [d.rstrip("/").rsplit("/", 1)[1] for d in dirs]
    for deriv, run in (("T1w_HCP", ""), ("T1w_HCP_run-01", "_run-01"), ("ants", "")):
        if deriv not in names:
            continue
        ants = (f"{MRI}/{site}/derivatives/{sub}/{deriv}/ants/" if deriv != "ants"
                else f"{MRI}/{site}/derivatives/{sub}/ants/")
        _, have = s3_ls(ants)
        have = dict(have)
        if not all(ants + f in have for f in ANTS_FILES):
            continue
        _, raw = s3_ls(f"{MRI}/{site}/{sub}/anat/")
        raw = dict(raw)
        t1 = f"{MRI}/{site}/{sub}/anat/{sub}_acq-HCP{run}_T1w.nii.gz"
        if t1 not in raw:
            continue
        return [(t1, raw[t1])] + [(ants + f, have[ants + f]) for f in ANTS_FILES]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--root", default=os.environ.get("EEG_DATA", "eeg_data"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = os.path.join(a.root, "hbn_mri")
    os.makedirs(out, exist_ok=True)

    eeg = eeg_participants(a.root)
    cand = []
    for site in SITES:
        dirs, _ = s3_ls(f"{MRI}/{site}/derivatives/")
        subs = {d.rstrip("/").rsplit("/", 1)[1] for d in dirs}
        cand += [(s, site) for s in sorted(subs & set(eeg.index))]
    C = pd.DataFrame(cand, columns=["subject", "site"]).drop_duplicates("subject")
    C["age"] = eeg.loc[C.subject, "age"].to_numpy()
    C["bin"] = pd.cut(C.age, AGE_EDGES, right=False)
    print(f"{len(eeg)} QC-passing EEG participants; {len(C)} with derivatives at "
          f"{', '.join(SITES)}", flush=True)
    print(C.groupby(["bin", "site"], observed=False).size().unstack().to_string(), flush=True)

    # age-stratified draw: equal numbers per age bin where possible, the
    # shortfall of small bins filled from the others in random order
    rng = np.random.default_rng(a.seed)
    queue = C.iloc[rng.permutation(len(C))]

    def check(r):
        # a listing that still fails after the retries is kept apart from a
        # participant without the files, so that it is not silently dropped
        try:
            return r, mri_files(r.site, r.subject), None
        except Exception as e:
            return r, None, f"{type(e).__name__}: {e}"
    with ThreadPoolExecutor(a.workers) as ex:
        found = list(ex.map(check, list(queue.itertuples())))
    err = [(r.subject, e) for r, _, e in found if e]
    valid = [(r, f) for r, f, e in found if f is not None]
    print(f"{len(valid)} of {len(found)} candidates have a T1w with ANTs outputs; "
          f"{len(err)} listings failed", flush=True)
    for s, e in err[:10]:
        print("   ", s, e, flush=True)
    bins = {}
    for r, f in valid:
        bins.setdefault(r.bin, []).append((r, f))
    per_bin = int(np.ceil(a.n / (len(AGE_EDGES) - 1)))
    chosen = [x for v in bins.values() for x in v[:per_bin]]
    rest = [x for v in bins.values() for x in v[per_bin:]]
    chosen = (chosen + [rest[i] for i in rng.permutation(len(rest))])[:a.n]
    sel = pd.DataFrame([dict(subject=r.subject, site=r.site, age=r.age) for r, _ in chosen])
    sel.to_csv(os.path.join(out, "hbn_mri_selected.csv"), index=False)
    print(f"selected {len(sel)}; "
          f"{sum(s for _, f in chosen for _, s in f) / 1e9:.2f} GB to fetch", flush=True)

    def get(item):
        r, files = item
        res = []
        for key, size in files:
            dest = os.path.join(out, r.site, r.subject, key.rsplit("/", 1)[1])
            res.append(fetch(f"{BUCKET}/{urllib.parse.quote(key)}", dest, size))
        return r.subject, res

    bad = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (sub, res) in enumerate(ex.map(get, chosen)):
            if any(x not in ("ok", "cached") for x in res):
                bad += 1
                print(sub, res, flush=True)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(chosen)} fetched", flush=True)
    print(f"done: {len(chosen)} participants, {bad} with failures", flush=True)


if __name__ == "__main__":
    main()
