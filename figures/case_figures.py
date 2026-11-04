#!/usr/bin/env python3
"""Appendix case figures (ported from the draft's make_case_figures.py): one held-out case per classification source,
and classification and segmentation success and failure cases across datasets and backbones.

    python case_figures.py [--only cohorts,cls,seg]
        -> out/fig_examples_cls, out/fig_cases_cls, out/fig_cases_seg_<dataset> (PDF + PNG; PDFs copied to ../paper/figures),
           out/selection/case_checks.txt

Cases come from out/selection/case_selection.json (select_cases.py, fixed rules). Segmentation slices are the saved
predictions under midl_paper/results/examples/seg_cases/ (read-only); the Dice values printed in the panel titles are
those of the per-case files the metrics use (runs.csv), and the saved predictions' own Dice are checked against them.
Classification images are read from the data and shown in one fixed view per source through the body's centre of mass:
coronal for AMOS, LUNA25 and RSNA 2023, sagittal for the cervical spine, axial for COVID-CT (thick slices), FDG-PET/CT
(whole body) and head CT. The view gives context and need not show the finding.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style
from select_cases import OUT_SEL, RES, V2, case_scores, metrics, names_of, run_row, seg_percase, cls_percase

# NO_VOCO (default; WITH_VOCO=1 turns it off): VoCo rows / markers are left out (same selected cases, read from OUT_SEL); outputs go to style.FIG_OUT and
# style.PAPER_DIR (defaults: out/ and ../paper/)
OUT = style.FIG_OUT
PAPER_FIG = style.PAPER_DIR / "figures"
CHECK_DIR = OUT / "selection"
INK, INK2, GRID = style.INK, style.INK2, style.GRID
FS = 6.0
BB_COL, BB_MARK, BB_TEX = style.BACKBONE_COL, style.BACKBONE_MARK, style.BACKBONE_TEX
BB_ORDER = ["tapct", "spectre", "pillar0", "voco"]
SEG_BACKBONES = ["tapct", "spectre", "voco"]
if style.NO_VOCO:
    BB_ORDER = [b for b in BB_ORDER if b != "voco"]
    SEG_BACKBONES = [b for b in SEG_BACKBONES if b != "voco"]
CVPR = "/projects/cvpr2026_challenge_fomo4ct/CVPR26-3DCTFMCompetition"
DP = "/mnt/kronos/data_public"
SOURCES = {  # key: (title, cohort, view, window)
    "amos": ("CVPR26: AMOS", "amos_covid_luna", "coronal", (-160, 240)),
    "covid": ("CVPR26: COVID-CT", "amos_covid_luna", "axial", (-1350, 150)),
    "luna": ("CVPR26: LUNA25", "amos_covid_luna", "coronal", (-1350, 150)),
    "fdg": ("FDG-PET/CT (CT)", "fdg", "axial", (-160, 240)),
    "rsna2022": ("RSNA 2022 C-spine", "rsna2022", "sagittal", (-300, 1200)),
    "rsna2023": ("RSNA 2023 trauma", "rsna2023", "coronal", (-160, 240)),
    "ich": ("RSNA 2019 hemorrhage", "rsna2019_ich", "axial", (0, 80)),
}
METHODS = [("lin", "Lin"), ("lora", "LoRA"), ("attnft", "Attn-only FT"), ("fullft", "Full FT")]
CHECKS: list[str] = []


def image_path(src: str, case: str) -> Path:
    if src == "amos":
        return Path(f"{CVPR}/AMOS-clf-tr-val/images/{case}")
    if src == "covid":
        return Path(f"{CVPR}/COVID-CT/images/{case}")
    if src == "luna":
        return Path(f"{CVPR}/LUNA25/images/{case}")
    if src == "fdg":
        for r in csv.DictReader(open(f"{DP}/FDG-PET-CT-Lesions/cls_labels.csv")):
            if r["case_id"] == case:
                return Path(r["ct_path"])
    if src == "rsna2022":
        return Path(f"{DP}/RSNA-2022-Cervical-Spine-Fracture-Detection/images/{case}.nii.gz")
    if src == "rsna2023":
        return Path(f"{DP}/RSNA-2023-Abdominal-Trauma-Detection/niftis/{case}/ct_scan.nii.gz")
    if src == "ich":
        for r in csv.DictReader(open(f"{DP}/RSNA-2019-Intracranial-Hemorrhage-Detection/niftis/patient_table_stratified_5k.csv")):
            if r["case_id"] == case:
                return Path(r["file_path"])
    raise FileNotFoundError(f"{src} {case}")


def view(path: Path, kind: str) -> tuple[np.ndarray, float]:
    """Central slice of the canonical (RAS) volume in radiological display; returns (HU slice, pixel aspect)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    a = np.asanyarray(img.dataobj).astype(np.float32)
    zx, zy, zz = img.header.get_zooms()[:3]
    body = a > -300  # soft tissue and bone: the centre of mass falls in the patient, not on the table
    c = [int(round(v)) for v in np.argwhere(body).mean(axis=0)] if body.any() else [n // 2 for n in a.shape]
    if kind == "axial":
        sl, asp = a[:, :, c[2]].T[::-1, ::-1], zy / zx
    elif kind == "coronal":
        sl, asp = a[:, c[1], :].T[::-1, ::-1], zz / zx
    else:
        sl, asp = a[c[0], :, :].T[::-1, ::-1], zz / zy
    if kind == "coronal" and sl.shape[0] * asp > 2.2 * sl.shape[1]:  # whole-body scan: central 40% of its length
        n = sl.shape[0]
        sl = sl[int(0.3 * n): int(0.7 * n)]
    m = sl > -500
    if m.sum() > 100:
        ys, xs = np.where(m)
        sl = sl[max(0, ys.min() - 4): ys.max() + 5, max(0, xs.min() - 4): xs.max() + 5]
    return sl, float(asp)


AGG = {"any", "patient_overall", "any_injury"}


def case_labels(cohort: str, src: str) -> dict[str, list[str]]:
    """Positive held-out labels of every case of a source, from the per-case file of the TAP-CT joint linear-probe run
    (it scores every label of a cohort); labels not scored in the metrics drop out; aggregate labels are dropped when
    a specific one is positive."""
    row = run_row("cls", cohort, "tapct", "joint", "lin")
    d = cls_percase(row)
    m = metrics("cls")
    scored = set(m[(m.cohort == cohort) & (m.backbone == "tapct") & (m["mode"] == "joint") & (m.method == "lin")
                   & (m.seed == 42) & (m.split == "test")].label)
    prefix = {"amos": "amos:", "covid": "covid_ct:", "luna": "luna25:"}.get(src, "")
    d = d[d.name.isin(scored) & d.name.str.startswith(prefix) & (d.label.astype(float) == 1)]
    pos: dict[str, set] = {}
    for c, name in zip(d.case_id, d.name):
        name = name.split(":")[-1]
        if name.upper() != "NEGATIVE":
            pos.setdefault(c, set()).add(name)
    return {c: sorted(v - AGG) if v - AGG else sorted(v) for c, v in pos.items()}


def show(ax, path: Path, src: str):
    sl, asp = view(path, SOURCES[src][2])
    lo, hi = SOURCES[src][3]
    ax.imshow(np.clip((sl - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1, aspect=asp, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def pretty(lab: str) -> str:
    names = {"patient_overall": "any fracture", "any_injury": "any injury", "any": "any hemorrhage", "covid": "COVID-19",
             "lung_nodule_malignancy": "malignant nodule"}
    if lab in names:
        return names[lab]
    return lab if lab[:1] == "C" and lab[1:].isdigit() else lab.replace("_injury", " injury").replace("_", " ").lower()


# ------------------------------------------------------------------ one held-out case per classification source
def fig_cohorts(out: Path) -> None:
    fig, axes = plt.subplots(1, 7, figsize=(style.TEXTWIDTH_IN, 1.95))
    for ax, src in zip(axes, SOURCES):
        title, ds, _, _ = SOURCES[src]
        pos = case_labels(ds, src)
        case = sorted(pos, key=lambda c: (-len(pos[c]), c))[0]  # held-out case with the most positive labels
        CHECKS.append(f"fig_examples_cls {src}: case {case}, positive labels {pos[case]}")
        show(ax, image_path(src, case), src)
        t = title.replace("CVPR26: ", "CVPR26\n").replace(" (CT)", "\n(CT)").replace("RSNA 20", "RSNA 20").replace(" C-spine", "\nC-spine") \
                 .replace(" trauma", "\ntrauma").replace(" hemorrhage", "\nhemorrhage")
        ax.set_title(t, fontsize=FS, color=INK, pad=2, linespacing=1.0)
        labs = [pretty(x) for x in pos[case]]
        lines, cur = [], ""
        for l in labs:
            nxt = (cur + ", " + l) if cur else l
            if len(nxt) > 16 and cur:
                lines.append(cur + ","); cur = l
            else:
                cur = nxt
        lines.append(cur)
        if len(lines) > 4:
            rest = len(labs) - sum(len(x.rstrip(",").split(", ")) for x in lines[:3])
            lines = lines[:3] + [f"+ {rest} more"]
        ax.set_xlabel("\n".join(lines), fontsize=FS, color=INK2, labelpad=2, linespacing=1.0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.80, bottom=0.30, wspace=0.10)
    style.save(fig, out, "fig_examples_cls")


# ------------------------------------------------------------------ classification success / failure
def fig_cls_cases(sel: dict, out: Path) -> None:
    cs = sel["cls"]
    srcs = list(SOURCES)
    fig = plt.figure(figsize=(style.TEXTWIDTH_IN, 1.05 * len(srcs) + 0.45))
    # columns: image, gap for the method names, dots, gap, image, gap, dots
    gs = fig.add_gridspec(len(srcs), 7, width_ratios=[0.85, 0.56, 1.2, 0.22, 0.85, 0.56, 1.2], left=0.06, right=0.985, top=0.965,
                          bottom=0.07, hspace=0.5, wspace=0.04)
    for i, src in enumerate(srcs):
        e = cs[src]
        bbs = sorted({k.split("/")[0] for k in e["runs"]} & set(BB_ORDER), key=BB_ORDER.index)
        cmean = e["case_mean"]
        if style.NO_VOCO and len(bbs) < len({k.split("/")[0] for k in e["runs"]}):
            # the selection averaged VoCo too; the printed mean is over the backbones shown (exact, from the per-case files)
            lab = e["label"].split(":")[-1]
            cmean = {}
            for c in e["cases"].values():
                v = [float(case_scores(cls_percase(run_row("cls", e["dataset"], bb, "pertask", mt, label=lab)))[c])
                     for bb in bbs for mt, _ in METHODS]
                cmean[c] = float(np.mean(v))
                CHECKS.append(f"fig_cases_cls {src} {c}: mean over {', '.join(bbs)} = {cmean[c]:.4f} "
                              f"(selection mean with VoCo {e['case_mean'][c]:.4f})")
        for j, role in enumerate(["success", "failure"]):
            case = e["cases"][role]
            axi = fig.add_subplot(gs[i, 0 if j == 0 else 4])
            show(axi, image_path(src, case), src)
            if j == 0:
                axi.set_ylabel(SOURCES[src][0].replace("CVPR26: ", "CVPR26\n").replace(" (CT)", "").replace("RSNA 20", "RSNA\n20"),
                               fontsize=FS, color=INK, linespacing=1.0)
            axd = fig.add_subplot(gs[i, 2 if j == 0 else 6])
            for k, (mt, mlab) in enumerate(METHODS):
                y = len(METHODS) - 1 - k
                for b, bb in enumerate(bbs):
                    v = e["case_score"][f"{bb}/{mt}"][case]
                    axd.scatter([v], [y + (b - (len(bbs) - 1) / 2) * 0.28], s=12, color=BB_COL[bb], marker=BB_MARK[bb], zorder=3,
                                edgecolor="white", linewidths=0.4)
            axd.axvline(0.5, color=INK2, lw=0.6, ls=":", zorder=1)
            axd.set_xlim(-0.03, 1.03); axd.set_ylim(-0.6, len(METHODS) - 0.4)
            axd.set_yticks(range(len(METHODS))); axd.set_yticklabels([m[1] for m in METHODS[::-1]], fontsize=FS)
            axd.set_xticks([0, 0.5, 1]); axd.set_xticklabels(["0", "0.5", "1"]); axd.tick_params(length=1.5, labelsize=FS, pad=1)
            axd.grid(axis="x", color=GRID, lw=0.5, zorder=0)
            if i == len(srcs) - 1:
                axd.set_xlabel("negatives outranked by the case", fontsize=FS, labelpad=1)
            axd.set_title(f"{role}: {pretty(e['label'].split(':')[-1])} (mean {cmean[case]:.2f})", fontsize=FS, color=INK, loc="left", pad=2)
    handles = [Line2D([], [], marker=BB_MARK[b], ls="", color=BB_COL[b], markersize=4, label=BB_TEX[b]) for b in BB_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(BB_ORDER), frameon=False, fontsize=FS, bbox_to_anchor=(0.5, 0.0))
    style.save(fig, out, "fig_cases_cls")


# ------------------------------------------------------------------ segmentation success / failure
SEG_TEX = {"amos22": "AMOS22", "kits23": "KiTS23", "lits": "LiTS", "msd_pancreas": "MSD Pancreas", "totalseg": "TotalSegmentator"}
SEG_METHODS = [("lin", "Lin"), ("dec", "Dec"), ("lora", "LoRA"), ("fullft", "Full FT")]


def _seg_hu(img: np.ndarray, bb: str) -> np.ndarray:
    """Model input to HU: TAP-CT receives HU; SPECTRE [-1000, 1000] HU mapped to [0, 1]; VoCo [-175, 250] HU mapped to
    [0, 1]. The RAS grids of SPECTRE and VoCo are flipped in both axes to match TAP-CT's radiological display."""
    x = img.astype(np.float32)
    if bb == "tapct":
        return x
    if bb == "spectre":
        return (x * 2000.0 - 1000.0)[::-1, ::-1]
    return (x * 425.0 - 175.0)[::-1, ::-1]


def _label_cols(n: int = 256) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    cols = np.zeros((n, 4))
    for k in range(1, n):
        cols[k] = cmap((k - 1) % 20)
    return cols


def file_dice(ds: str, bb: str, mt: str, case: str) -> np.ndarray:
    """Per-structure held-out Dice of a case from the per-case file of the joint run (NaN: not valid, i.e. absent from
    reference and prediction), in class_index order."""
    row = run_row("seg", ds, bb, "joint", mt)
    d = seg_percase(row)
    d = d[d.case_id == case]
    names = names_of(row)
    out = np.full(len(names), np.nan)
    for ci, v in zip(d.class_index, d.dice):
        out[int(ci)] = v
    return out


def fig_seg_cases(sel: dict, out: Path, ds: str) -> None:
    root = RES / "examples" / "seg_cases"
    cols = _label_cols()
    rows = [(role, bb) for role in ("success", "failure") for bb in SEG_BACKBONES]
    fig, axes = plt.subplots(len(rows), 5, figsize=(style.TEXTWIDTH_IN, 1.13 * len(rows) + 0.1))
    maxd = 0.0
    for r, (role, bb) in enumerate(rows):
        case = sel["seg"][ds]["cases"][role]
        for c, (mt, mlab) in enumerate([("ref", "reference")] + SEG_METHODS):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            f = root / ds / case / f"{bb}-{mt if mt != 'ref' else 'lora'}.npz"
            d = np.load(f)
            hu = _seg_hu(d["image"], bb)
            im = np.clip((hu + 160.0) / 400.0, 0, 1)
            lab = d["reference"] if mt == "ref" else d["prediction"]
            if bb != "tapct":
                lab = lab[::-1, ::-1]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            ax.imshow(cols[lab], alpha=np.where(lab > 0, 0.5, 0.0), interpolation="nearest")
            if mt == "ref":
                ax.set_title(f"{BB_TEX[bb]}: reference", fontsize=FS, color=INK, pad=1.5)
                if bb == "tapct":
                    ax.set_ylabel(f"{SEG_TEX[ds]}\n{role}, case {case.split('_')[-1] if ds != 'totalseg' else case}", fontsize=FS, color=INK)
                continue
            fd, nd = file_dice(ds, bb, mt, case), np.asarray(d["dice"], float)
            same_nan = np.array_equal(np.isnan(fd), np.isnan(nd))
            dd = float(np.nanmax(np.abs(fd - nd))) if same_nan else float("nan")
            maxd = max(maxd, dd) if not np.isnan(dd) else maxd
            if len(fd) <= 3:  # few structures: one Dice each (–: absent from reference and prediction)
                per = " / ".join("–" if np.isnan(x) else f"{x:.2f}" for x in fd)
                per_n = " / ".join("–" if np.isnan(x) else f"{x:.2f}" for x in nd)
                ttl = f"{mlab}: {per}"
                changed = per != per_n
            else:  # many structures: the case mean and how many structures it averages in at Dice 0
                dice, nz = float(np.nanmean(fd)), int((fd == 0).sum())
                ttl = f"{mlab}: Dice {dice:.2f}" + (f", {nz} at 0" if nz else "")
                old_t = f"{mlab}: Dice {float(np.nanmean(nd)):.2f}" + (f", {int((nd == 0).sum())} at 0" if (nd == 0).sum() else "")
                changed = ttl != old_t
            zero_names = [names_of(run_row('seg', ds, bb, 'joint', mt))[k] for k in np.where(fd == 0)[0]]
            CHECKS.append(f"fig_cases_seg_{ds} {role} {case} {bb}/{mt}: title '{ttl}'; max |saved prediction - per-case file| "
                          f"{'n/a (valid sets differ)' if not same_nan else f'{dd:.1e}'}" + ("  (2-decimal title differs from the saved prediction's)" if changed else "")
                          + (f"; Dice 0: {', '.join(zero_names)}" if zero_names else ""))
            ax.set_title(ttl, fontsize=FS, color=INK, pad=1.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.005, hspace=0.2, wspace=0.12)
    style.save(fig, out, f"fig_cases_seg_{ds}")
    CHECKS.append(f"fig_cases_seg_{ds}: max |saved prediction - per-case file| over shown Dice = {maxd:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="cohorts,cls,seg")
    a = ap.parse_args()
    style.setup()
    sel = json.loads((OUT_SEL / "case_selection.json").read_text())
    todo = a.only.split(",")
    names = []
    if "cohorts" in todo:
        fig_cohorts(OUT); names.append("fig_examples_cls")
    if "cls" in todo:
        fig_cls_cases(sel, OUT); names.append("fig_cases_cls")
    if "seg" in todo:
        for ds in ["amos22", "kits23", "lits", "msd_pancreas", "totalseg"]:
            fig_seg_cases(sel, OUT, ds); names.append(f"fig_cases_seg_{ds}")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy(OUT / f"{n}.pdf", PAPER_FIG / f"{n}.pdf")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    (CHECK_DIR / f"case_checks_{'_'.join(todo)}.txt").write_text("\n".join(CHECKS) + "\n")
    print("\n".join(CHECKS))


if __name__ == "__main__":
    main()
