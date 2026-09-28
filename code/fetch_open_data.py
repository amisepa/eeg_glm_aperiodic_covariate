"""Download the small open datasets used for the breadth and identification
analyses, keeping the raw files.

  brake2024    Brake et al. 2024 Nat Commun, Cz spectrograms around propofol
               loss of consciousness (figshare 24777990, CC BY 4.0, 0.11 GB)
  chennu2016   Chennu et al. 2016 PLoS Comput Biol, graded propofol sedation
               (Apollo doi:10.17863/CAM.68959, CC BY 2.0 UK, 3.7 GB)
  ds003690     OpenNeuro ds003690, passive auditory task of young and older
               adults with pupil and EOG (CC0, 2.7 GB)
  brake2024_source
               Brake et al. 2024 manuscript source data, which include the
               infusion-onset times (figshare 24777990, file 43599408, 5.1 GB)
  srm          OpenNeuro ds003775, SRM resting-state EEG, 111 adults, 42 of
               them retested (CC0, 4.8 GB)
  ds003688     OpenNeuro ds003688, intracranial EEG; only the resting runs
               and electrode files are fetched (CC0, 4.3 GB)
  hbn_participants
               participants.tsv of the eleven HBN-EEG releases (< 1 MB)

Files already present with the right size are skipped, so the script can be
re-run after an interruption.

Usage: python fetch_open_data.py [brake2024 chennu2016 ds003690 ...] [--root DIR]
"""
import argparse
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HEAD = {"User-Agent": "Mozilla/5.0 (research download)"}
S3 = "https://s3.amazonaws.com/openneuro.org"


def fetch(url, dest, size=None, tries=4):
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return "cached"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEAD)
            with urllib.request.urlopen(req, timeout=300) as r, open(dest + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return "ok"
        except Exception as e:
            if k == tries - 1:
                return f"failed: {type(e).__name__}: {e}"
            time.sleep(2 ** k)


def s3_list(prefix):
    out, tok = [], None
    while True:
        url = f"{S3}?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if tok:
            url += "&continuation-token=" + urllib.parse.quote(tok)
        x = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        ns = {"s": x.tag.split("}")[0].strip("{")}
        out += [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text))
                for c in x.findall("s:Contents", ns)]
        if x.find("s:IsTruncated", ns).text != "true":
            return out
        tok = x.find("s:NextContinuationToken", ns).text


def brake2024(root):
    d = os.path.join(root, "brake2024")
    return [fetch("https://ndownloader.figshare.com/files/43564131",
                  os.path.join(d, "spectrogram_Cz_all_subjects.zip"), 112838295)]


def chennu2016(root):
    d = os.path.join(root, "chennu2016")
    base = "https://www.repository.cam.ac.uk/bitstreams/{}/download"
    out = []
    for i, bid in enumerate(("db691a24-0250-42bb-b03b-553c54f121b9",
                             "e94a6722-da5b-4e53-8673-5e8ec106e0f7")):
        req = urllib.request.Request(base.format(bid), headers=HEAD, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            cd = r.headers.get("Content-Disposition", "")
            size = int(r.headers.get("Content-Length", 0)) or None
        name = cd.split("filename=")[-1].strip('"; ') if "filename=" in cd else f"file{i}"
        out.append(fetch(base.format(bid), os.path.join(d, name), size))
    return out


def ds003690(root):
    d = os.path.join(root, "ds003690")
    out = []
    for key, size in s3_list("ds003690/"):
        rel = key.split("/", 1)[1]
        if "/" in rel and "task-passive" not in rel:
            continue                       # other tasks are not needed
        out.append(fetch(f"{S3}/{urllib.parse.quote(key)}", os.path.join(d, rel), size))
    return out


def brake2024_source(root):
    import json
    d = os.path.join(root, "brake2024")
    files = json.load(urllib.request.urlopen(
        "https://api.figshare.com/v2/articles/24777990/files?page_size=100", timeout=60))
    size = next(f["size"] for f in files if f["id"] == 43599408)
    return [fetch("https://ndownloader.figshare.com/files/43599408",
                  os.path.join(d, "manuscript_source_data.zip"), size)]


def openneuro(ds, root, keep=lambda rel: True):
    d = os.path.join(root, ds)
    return [fetch(f"{S3}/{urllib.parse.quote(key)}", os.path.join(d, key.split("/", 1)[1]), size)
            for key, size in s3_list(ds + "/") if keep(key.split("/", 1)[1])]


def srm(root):
    return openneuro("ds003775", root)


def ds003688(root):
    def keep(rel):
        if "/" not in rel:
            return True                    # top-level metadata
        if "/ieeg/" not in rel:
            return False                   # fMRI and anatomy are not needed
        name = rel.rsplit("/", 1)[1]
        return "task-rest" in name or "task-" not in name
    return openneuro("ds003688", root, keep)


HBN_RELEASES = ("ds005505", "ds005506", "ds005507", "ds005508", "ds005509", "ds005510",
                "ds005511", "ds005512", "ds005514", "ds005515", "ds005516")


def hbn_participants(root):
    """participants.tsv of every HBN-EEG release; ds005516 serves it by version ID."""
    import json
    d = os.path.join(root, "hbn_participants")
    out = []
    for ds in HBN_RELEASES:
        dest = os.path.join(d, f"{ds}_participants.tsv")
        res = fetch(f"{S3}/{ds}/participants.tsv", dest)
        if res not in ("ok", "cached"):
            q = json.dumps({"query": f'{{ dataset(id: "{ds}") {{ latestSnapshot {{ files '
                                     f'{{ filename urls }} }} }} }}'}).encode()
            req = urllib.request.Request("https://openneuro.org/crn/graphql", data=q,
                                         headers={**HEAD, "Content-Type": "application/json"})
            files = json.load(urllib.request.urlopen(req, timeout=120))["data"]["dataset"][
                "latestSnapshot"]["files"]
            url = next((f["urls"][0] for f in files if f["filename"] == "participants.tsv"), None)
            res = fetch(url, dest) if url else "failed: no participants.tsv in snapshot"
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["brake2024", "chennu2016", "ds003690"])
    ap.add_argument("--root", default=os.environ.get("EEG_DATA", "eeg_data"))
    a = ap.parse_args()
    for w in a.which:
        t0 = time.time()
        res = globals()[w](a.root)
        bad = [r for r in res if r not in ("ok", "cached")]
        print(f"{w}: {len(res)} files, {len(bad)} failed, {(time.time() - t0) / 60:.1f} min",
              flush=True)
        for b in bad[:10]:
            print("   ", b, flush=True)


if __name__ == "__main__":
    main()
