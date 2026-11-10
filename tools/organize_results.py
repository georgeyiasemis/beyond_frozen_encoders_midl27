#!/usr/bin/env python3
"""Collect every input the paper's numbers and figures are computed from into one structured folder (copies; the
sources are only read). VoCo, the quarantined runs and the breast MRI cohort (breastdivider) are left out.

    python organize_results.py [--dest /projects/cvpr2026_challenge_fomo4ct/beyond_frozen_encoders] [--dry-run]

Layout of <dest>:
  runs/classification/<cohort>/<backbone>/<mode>/<method>/<config | label/label>/   one folder per run: val_case_metrics/
  runs/segmentation/<dataset>/<backbone>/<mode>/<method>/<config | label/label>/      (per-case outputs at every
                                                                                       validation), logs/, checkpoints/best/
  configs/classification/..., configs/segmentation/...   the run YAMLs, same tree as runs/ (<run>.yaml, eval/<run>.yaml)
  examples/                saved predictions of the example and case figures, selection files
  data/                    the CT images and label maps shown in the figures, split manifests (public / cvpr26 / segmentation)
  metrics/run_inventory.csv, metrics/cost/, metrics/trainable_params.csv, metrics/cohort_sizes.json   inputs of the scripts
  metrics/derived/         snapshot of the recomputed metrics (midl_paper_v2/metrics/out) and the paper's number macros
  index.csv                one row per run: new path, source path, config, design attributes
Paths in run_inventory.csv and cost/runs_used.csv are rewritten to the new layout (relative to <dest>).
"""
from __future__ import annotations

import argparse
import glob
import re
import shutil
from pathlib import Path

import pandas as pd

PROJ = Path("/home/g.yiasemis/projects/cvpr_ct_challenge")
V2 = PROJ / "midl_paper_v2"
RES = PROJ / "midl_paper" / "results"
FM = PROJ / "aifofomo" / "examples" / "fm_probes"
CVPR = Path("/projects/cvpr2026_challenge_fomo4ct/CVPR26-3DCTFMCompetition")
PUBLIC = Path("/mnt/kronos/data_public")
E1 = {"lin_lr1e2", "lin_lr1e4", "attn_lr3e5", "attn_lr3e4", "attn_lr1e3"}


def rel_of(path: str) -> str:
    """'raw/<site>/<rel>' or 'runs/<task>/...' -> '<rel>' (classification: <cohort>/..., segmentation: seg/<dataset>/...)."""
    path = re.sub(r"^runs/classification/", "", path)
    path = re.sub(r"^runs/segmentation/", "seg/", path)
    return re.sub(r"^raw/[a-z]+/", "", path)


def new_path(rel: str) -> str:
    return f"runs/segmentation/{rel[4:]}" if rel.startswith("seg/") else f"runs/classification/{rel}"


def config_of(rel: str) -> tuple[Path, str]:
    """Source YAML of a run and its path under configs/."""
    p = rel.split("/")
    if p[0] == "seg":
        return FM / "fm_seg" / "/".join(p[1:5]) / f"{p[-1]}.yaml", f"configs/segmentation/{'/'.join(p[1:5])}/{p[-1]}.yaml"
    return FM / "fm_adapt" / "/".join(p[0:4]) / f"{p[-1]}.yaml", f"configs/classification/{'/'.join(p[0:4])}/{p[-1]}.yaml"


def _cp(src: str, dst: str) -> str:
    """copy2 unless dst already holds the same file (size and mtime), so a rerun only adds what is missing."""
    s, d = Path(src), Path(dst)
    if d.is_file() and d.stat().st_size == s.stat().st_size and int(d.stat().st_mtime) == int(s.stat().st_mtime):
        return dst
    return shutil.copy2(src, dst)


def copy(src: Path, dst: Path, dry: bool) -> int:
    if dry:
        return sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) if src.is_dir() else src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_cp)
        return sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    _cp(str(src), str(dst))
    return dst.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("/projects/cvpr2026_challenge_fomo4ct/beyond_frozen_encoders"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    D, dry = a.dest, a.dry_run
    total = {}

    # ---- runs: the paper's run table (metrics/out/runs.csv) plus the E1 probe learning-rate runs
    runs = pd.read_csv(V2 / "metrics" / "out" / "runs.csv")
    runs["part"] = "paper"
    e1 = pd.read_csv(V2 / "metrics" / "out" / "e1_probe_lr_runs.csv")
    e1 = e1[e1.method.isin(E1)].copy()
    e1["path"] = e1.percase_file.str.replace(r"/val_case_metrics/.*", "", regex=True)
    e1 = e1.assign(task="cls", mode="joint", seed=42, label="joint", part="e1_probe_lr", site="snellius")
    inv = pd.read_csv(V2 / "report" / "run_inventory.csv")
    inv = inv[inv.cohort != "breastdivider"]   # breast MRI cohort: not part of the study
    runs = runs[runs.cohort != "breastdivider"]
    rows = []
    for r in pd.concat([runs, e1], ignore_index=True).itertuples():
        rel = rel_of(r.path)
        src = f"raw/{r.site}/{rel}"   # location in the original mirror midl_paper/results
        ysrc, ydst = config_of(rel)
        ev = ysrc.parent / "eval" / ysrc.name
        dc = inv.loc[inv.path == src, "design_class"]
        rows.append(dict(task=r.task, cohort=r.cohort, backbone=r.backbone, mode=r.mode, method=r.method, label=r.label,
                         seed=int(r.seed), best_iter=int(r.best_iter), final_iter=int(r.final_iter), part=r.part,
                         design_class=dc.iloc[0] if len(dc) else "", path=new_path(rel), config=ydst,
                         eval_config=str(Path(ydst).parent / "eval" / Path(ydst).name) if ev.is_file() else "",
                         source_path=f"midl_paper/results/{src}", source_config=str(ysrc.relative_to(PROJ))))
    index = pd.DataFrame(rows)
    assert index.path.is_unique, "two runs map to one folder"
    n = 0
    for r in index.itertuples():
        n += copy(PROJ / r.source_path, D / r.path, dry)
        n += copy(PROJ / r.source_config, D / r.config, dry)
        if r.eval_config:
            n += copy(PROJ / r.source_config.replace(f"/{Path(r.source_config).name}", f"/eval/{Path(r.source_config).name}"), D / r.eval_config, dry)
    total["runs + configs"] = n

    # ---- examples: the saved predictions the figure scripts read, and the selection files
    n = 0
    for f in ["case_selection.json", "seg/selection.json", "kits_pancreas_nested_selection.json", "lits_representative_selection.json"]:
        n += copy(RES / "examples" / f, D / "examples" / f, dry)
    for f in sorted(glob.glob(str(RES / "examples" / "seg_cases" / "*" / "*" / "*.npz")) + glob.glob(str(RES / "examples" / "seg" / "*" / "*" / "*.npz"))
                    + glob.glob(str(RES / "examples" / "*.npz"))):
        if "voco" in Path(f).name:
            continue
        n += copy(Path(f), D / "examples" / Path(f).relative_to(RES / "examples"), dry)
    total["examples"] = n

    # ---- data: images and label maps shown in the figures, split manifests (behind the cohort table counts)
    n = 0
    data = [CVPR / "AMOS-clf-tr-val/images/amos_6795.nii.gz", CVPR / "COVID-CT/images/patient10_SR3.nii.gz",
            CVPR / "LUNA25/images/1.2.840.113654.2.55.125403043383466225171814830534052775879.nii.gz",
            PUBLIC / "FDG-PET-CT-Lesions/PETCT_04ab5c61c9/11-22-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42241/CTres.nii.gz",
            PUBLIC / "RSNA-2019-Intracranial-Hemorrhage-Detection/niftis/train/ID_026c4d95e8/ID_982be5bd0f/ct_scan.nii.gz",
            PUBLIC / "RSNA-2022-Cervical-Spine-Fracture-Detection/images/1.2.826.0.1.3680043.28753.nii.gz",  # link into niftis/
            PUBLIC / "RSNA-2023-Abdominal-Trauma-Detection/niftis/10300/31085/ct_scan.nii.gz"]
    data += [Path(p) for p in sorted(glob.glob(str(PUBLIC / "MSD/Task03_Liver/labelsTr/liver_*.nii.gz")))
             if Path(p).name in {f"liver_{i}.nii.gz" for i in (8, 24, 28, 33, 41, 47, 49, 57, 59, 69, 85, 86, 90, 94, 98, 99, 100, 110, 118, 119)}]
    data += [Path(p) for p in sorted(glob.glob(str(CVPR / "AMOS-clf-tr-val/labels/*.csv")))]
    data += [CVPR / "COVID-CT/labels/covid.csv", CVPR / "LUNA25/labels/lung_nodule_malignancy.csv",
             PUBLIC / "FDG-PET-CT-Lesions/cls_labels.csv", PUBLIC / "RSNA-2022-Cervical-Spine-Fracture-Detection/labels.csv",
             PUBLIC / "RSNA-2023-Abdominal-Trauma-Detection/labels.csv",
             PUBLIC / "RSNA-2019-Intracranial-Hemorrhage-Detection/niftis/patient_table_stratified_5k.csv"]
    for f in data:
        dst = D / "data" / ("cvpr26" if str(f).startswith(str(CVPR)) else "public") / f.relative_to(CVPR if str(f).startswith(str(CVPR)) else PUBLIC)
        n += copy(f.resolve() if f.is_symlink() else f, dst, dry)
    for f in sorted((FM / "fm_seg" / "manifests").glob("*.csv")):
        n += copy(f, D / "data" / "segmentation_manifests" / f.name, dry)
    total["data"] = n

    # ---- metrics: script inputs (paths rewritten to the new layout) and a snapshot of the derived metrics
    n = 0
    m = {r.source_path.removeprefix("midl_paper/results/"): r.path for r in index.itertuples()}
    if not dry:
        (D / "metrics" / "cost").mkdir(parents=True, exist_ok=True)
        inv2 = inv.copy()
        inv2["path"] = [m.get(p, p) for p in inv2.path]
        inv2.to_csv(D / "metrics" / "run_inventory.csv", index=False)
        ru = pd.read_csv(V2 / "report" / "cost" / "runs_used.csv")
        ru["percase_file"] = [m[p] + f[len(p):] for p, f in zip(ru.path, ru.percase_file)]
        ru["path"] = [m[p] for p in ru.path]
        ru.to_csv(D / "metrics" / "cost" / "runs_used.csv", index=False)
        index.to_csv(D / "index.csv", index=False)
    for f in ["steps.csv", "memory.csv", "measure_memory_14649.out", "gpu_factors.csv"]:
        n += copy(V2 / "report" / "cost" / f, D / "metrics" / "cost" / f, dry)
    n += copy(RES / "trainable_params.csv", D / "metrics" / "trainable_params.csv", dry)
    n += copy(RES / "cohort_sizes.json", D / "metrics" / "cohort_sizes.json", dry)
    for f in sorted((V2 / "metrics" / "out").iterdir()):
        if f.is_file():
            n += copy(f, D / "metrics" / "derived" / f.name, dry)
    for f in sorted((V2 / "paper" / "tables").glob("*numbers.tex")):
        n += copy(f, D / "metrics" / "derived" / "paper_macros" / f.name, dry)
    total["metrics"] = n

    print(f"{len(index)} runs ({(index.task == 'cls').sum()} classification, {(index.task == 'seg').sum()} segmentation; "
          f"{(index.part == 'e1_probe_lr').sum()} E1) -> {D}")
    for k, v in total.items():
        print(f"  {k:16s} {v / 1e9:7.3f} GB")
    print(f"  {'total':16s} {sum(total.values()) / 1e9:7.3f} GB")


if __name__ == "__main__":
    main()
