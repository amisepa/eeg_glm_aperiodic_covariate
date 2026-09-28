"""Scalp-to-cortex distance from HBN T1-weighted MRI.

For each participant fetched by hbn_mri_fetch.py: the head is segmented from
the T1w image (Otsu threshold on log intensity, closing, hole filling,
largest component) and the cortex taken from the ANTs six-class segmentation
(label 2, cortical grey matter), which is in the same voxel space. The
distance from every scalp-surface voxel to the nearest cortical voxel is the
scalp-to-cortex distance (scalp, skull and cerebrospinal fluid together); the
distance to the ANTs brain mask is reported as well. Values are summarised
over three scalp regions defined by direction from the brain's centroid in
RAS space: posterior (within 35 deg of posterior, 20 deg up, the region of the
EEG alpha ROI over O1/Oz/O2/Pz), vertex (within 30 deg of superior) and the
whole upper head.

A scan passes (mri_ok) when the segmentation fits the T1w intensities
(correlation >= 0.6 with the order CSF < GM < WM) and less than 10% of the
posterior and vertex regions lies more than 30 mm from the brain (the head
mask leaking into motion ghosts or padding).

Writes results/hbn_scalp_distance.csv (per participant; local) and, with
--qc DIR, a sagittal and an axial slice per participant for visual checks.

Usage: python hbn_scalp_distance.py [--mri DIR] [--workers 4] [--qc DIR] [--limit N]
"""
import argparse
import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROIS = {"post": (np.array([0.0, -1.0, np.tan(np.radians(20))]), 35.0),
        "vertex": (np.array([0.0, 0.0, 1.0]), 30.0)}
warnings.filterwarnings("ignore")


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def otsu(x, bins=256):
    """Threshold maximising the between-class variance of a histogram."""
    h, e = np.histogram(x, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def head_mask(t1, zooms):
    from scipy import ndimage as ndi
    x = np.log1p(np.clip(t1, 0, None))
    th = otsu(x[x > 0])
    m = x > th
    r = max(1, int(round(3.0 / min(zooms))))
    ball = ndi.generate_binary_structure(3, 1)
    m = ndi.binary_closing(m, ball, iterations=r)
    for ax in range(3):                     # fill holes slice by slice, then in 3-D
        m = np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(m, ax, 0)], ax)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return m


def label_fit(t1, S):
    """Correlation of T1 intensity with the tissue order CSF < GM < WM."""
    m = np.isin(S, (1, 2, 3))
    return float(np.corrcoef(t1[m], np.select([S == 1, S == 2, S == 3], [0.0, 1.0, 2.0])[m])[0, 1])


def one(args):
    folder, qc_dir = args
    import nibabel as nib
    from scipy import ndimage as ndi
    sub = os.path.basename(folder)
    site = os.path.basename(os.path.dirname(folder))
    try:
        t1p = glob.glob(os.path.join(folder, "*_T1w.nii.gz"))[0]
        raw = nib.load(t1p)
        t1 = np.asarray(raw.dataobj, dtype=np.float32)
        S = np.asarray(nib.load(os.path.join(folder, "antsBrainSegmentation.nii.gz")).dataobj
                       ).astype(np.int16)
        B = np.asarray(nib.load(os.path.join(folder, "antsBrainExtractionMask.nii.gz")).dataobj) > 0
        # The ANTs outputs are stored with voxel axes reversed (and at RU
        # sometimes in another axis order) relative to the BIDS T1w, and some
        # carry an oblique rotation the T1w does not, so resampling through
        # the headers misaligns them. Map in voxel space and keep whichever
        # axis order and flips fit the T1w intensities best.
        import itertools
        fits = {}
        sub3 = np.s_[::3, ::3, ::3]
        for perm in itertools.permutations(range(3)):
            if tuple(S.shape[i] for i in perm) != t1.shape:
                continue
            Sp = S.transpose(perm)
            for flips in itertools.product((False, True), repeat=3):
                sl = tuple(slice(None, None, -1) if fl else slice(None) for fl in flips)
                fits[(perm, flips)] = label_fit(t1[sub3], Sp[sl][sub3])
        if not fits:
            raise ValueError(f"shape {S.shape} vs {t1.shape}")
        (perm, flips) = max(fits, key=fits.get)
        sl = tuple(slice(None, None, -1) if fl else slice(None) for fl in flips)
        S = S.transpose(perm)[sl].copy()
        B = B.transpose(perm)[sl].copy()
        ranked = sorted(fits.values())
        mapping = f"perm{''.join(map(str, perm))}_flip{''.join(str(int(x)) for x in flips)}"
        fits = {mapping: ranked[-1], "next": ranked[-2] if len(ranked) > 1 else np.nan}
        img = nib.as_closest_canonical(nib.Nifti1Image(t1, raw.affine))
        ornt = nib.orientations.io_orientation(raw.affine)
        S = nib.orientations.apply_orientation(S, ornt)
        B = nib.orientations.apply_orientation(B, ornt)
        t1 = np.asarray(img.dataobj, dtype=np.float32)
        zooms = img.header.get_zooms()[:3]
        cortex = S == 2
        head = head_mask(t1, zooms) | B
        surf = head & ~ndi.binary_erosion(head)
        d_ctx = ndi.distance_transform_edt(~cortex, sampling=zooms)
        d_brain = ndi.distance_transform_edt(~B, sampling=zooms)
        ijk = np.argwhere(surf)
        xyz = nib.affines.apply_affine(img.affine, ijk)
        cen = nib.affines.apply_affine(img.affine, np.argwhere(B).mean(0))
        u = unit(xyz - cen)
        out = dict(subject=sub, site=site, t1=os.path.basename(t1p), mapping=mapping,
                   label_fit=fits[mapping], label_fit_other=fits["next"],
                   voxel_mm=float(np.mean(zooms)),
                   brain_ml=float(B.sum() * np.prod(zooms) / 1000),
                   cortex_ml=float(cortex.sum() * np.prod(zooms) / 1000),
                   head_ml=float(head.sum() * np.prod(zooms) / 1000))
        sel = {"upper": u[:, 2] > 0}
        for name, (d, ang) in ROIS.items():
            sel[name] = u @ unit(d) > np.cos(np.radians(ang))
        for name, s in sel.items():
            vals = d_ctx[tuple(ijk[s].T)]
            out[f"scd_{name}"] = float(np.median(vals)) if s.any() else np.nan
            out[f"sbd_{name}"] = float(np.median(d_brain[tuple(ijk[s].T)])) if s.any() else np.nan
            out[f"n_{name}"] = int(s.sum())
            # share of the region's surface more than 30 mm from the brain: a
            # head mask leaking into motion ghosts or padding
            out[f"leak_{name}"] = float(np.mean(d_brain[tuple(ijk[s].T)] > 30)) if s.any() else np.nan
        if qc_dir:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ci = np.round(np.linalg.inv(img.affine) @ np.r_[cen, 1])[:3].astype(int)
            fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
            roi = np.zeros(head.shape, bool)
            roi[tuple(ijk[sel["post"]].T)] = True
            for ax, sl in zip(axs, (np.s_[ci[0], :, :], np.s_[:, :, ci[2]])):
                ax.imshow(t1[sl].T, origin="lower", cmap="gray")
                ax.contour(head[sl].T, [0.5], colors="y", linewidths=0.5)
                ax.contour(cortex[sl].T, [0.5], colors="c", linewidths=0.3)
                ax.contour(ndi.binary_dilation(roi, iterations=2)[sl].T, [0.5], colors="r",
                           linewidths=0.8)
                ax.axis("off")
            fig.suptitle(f"{sub} {site}: posterior SCD {out['scd_post']:.1f} mm", fontsize=8)
            fig.savefig(os.path.join(qc_dir, f"{sub}.png"), dpi=80)
            plt.close(fig)
        return out
    except Exception as e:
        return dict(subject=sub, site=site, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"), "hbn_mri"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qc", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    folders = sorted(p for p in glob.glob(os.path.join(a.mri, "Site-*", "sub-*"))
                     if os.path.exists(os.path.join(p, "antsbrainvols.csv")))
    if a.limit:
        folders = folders[:a.limit]
    if a.qc:
        os.makedirs(a.qc, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(f, a.qc) for f in folders]))
    D = pd.DataFrame(rows)
    sel = pd.read_csv(os.path.join(a.mri, "hbn_mri_selected.csv")).set_index("subject")
    D = D.join(sel[["age"]], on="subject")
    D["mri_ok"] = (D.get("label_fit", np.nan) >= 0.6) & (D.get("leak_post", np.nan) < 0.1) & \
                  (D.get("leak_vertex", np.nan) < 0.1)
    D.to_csv(os.path.join(RES, "hbn_scalp_distance.csv"), index=False)
    print(f"passing checks (label fit >= 0.6, < 10% of the region > 30 mm from the brain): "
          f"{int(D.mri_ok.sum())} of {len(D)}")
    ok = D[D.mri_ok]
    print(f"{len(D)} participants, {len(ok)} measured")
    if "error" in D:
        print(D.loc[D["error"].notna(), ["subject", "error"]].head(10).to_string(index=False))
    if ok.empty:
        return
    print(ok[["age", "scd_post", "scd_vertex", "scd_upper", "sbd_post", "brain_ml"]]
          .describe().round(2).to_string())
    print("r with age:", ok[["scd_post", "scd_vertex", "scd_upper", "sbd_post", "brain_ml"]]
          .corrwith(ok.age).round(3).to_dict())


if __name__ == "__main__":
    main()
