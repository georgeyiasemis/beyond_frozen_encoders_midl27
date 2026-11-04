#!/usr/bin/env python3
"""Recompute every metric of the paper from the per-case prediction files of the finished sweep.

    python recompute.py [--inventory ../report/run_inventory.csv] [--results ../../midl_paper/results] [--out out]
                        [--workers 6] [--no-cache]

Inputs
  * ../report/run_inventory.csv: one row per run directory; rows with status == finished, in_design == yes, excluded
    empty and aux empty are used (quarantined copies are excluded by the `excluded` column).
  * <results>/<path>/checkpoints/best/best_val_metric_val_<auroc|dice>.txt: selected iteration (best validation
    macro metric) and its logged value.
  * <results>/<path>/val_case_metrics/iter_<best>/logits.csv (classification: case_id, split, task, task_index, label,
    logit, probability) or dice.csv (segmentation: case_id, split, dataset, class_index, dice, valid).
  * Label names: the run YAML's `task_names` list (fm_adapt) in task_index order; structure names: the dataset's
    channel order of fm_seg/generate.py (DATASETS[ds]["channels"], copied below) by class_index.

Outputs (see SCHEMA.md; every file goes to --out): runs.csv, metrics_cls.csv, metrics_seg.csv, cohort_means.csv,
paired_cls.csv, paired_seg.csv, label_level_values.csv, label_level.csv, ranks.csv, spread.csv, convergence.csv,
seeds.csv, seeds_summary.csv, revision.csv, numbers.tex, changes_vs_draft.csv, NAMING.md (one directory up),
checks.txt (the verification printout). The statistical procedure is described in PROCEDURE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
DEFAULT_INVENTORY = HERE.parent / "report" / "run_inventory.csv"
DEFAULT_RESULTS = HERE.parents[1] / "midl_paper" / "results"
DEFAULT_DRAFT_NUMBERS = HERE.parents[1] / "midl_paper" / "paper" / "tables" / "numbers.tex"
FM_ADAPT = Path("/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo/examples/fm_probes/fm_adapt")

# ----------------------------------------------------------------------------- design constants
CLS_METHODS = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
CLS_REVISION_METHODS = ["fullft_lowlr", "fullft_llrd", "fullft_accum", "mlpft", "rslora_lowlr"]
SEG_REVISION_METHODS = ["scratch"]
CLS_BASELINES = ["lin", "lora", "attn", "fullft"]
SEG_BASELINES = ["lin", "dec", "lora"]
CLS_SEED_METHODS = ["lin", "lora", "attnft", "fullft"]
SEG_SEED_METHODS = ["dec", "lora", "fullft"]
FIXES = ["fullft_lowlr", "fullft_llrd", "fullft_accum"]
MAIN_BACKBONES = ["tapct", "spectre", "pillar0"]
ALL_BACKBONES = MAIN_BACKBONES + ["voco"]
EXCLUDED_COHORTS = {"breastdivider"}
EXCLUDED_LABELS = {"lesion_malignant"}
SEG_LONG_JOINT = {"amos22", "totalseg"}  # 15-structure datasets: the 24 000-step joint_lm run is the paper's "joint"
SEG_N_STRUCT = {"amos22": 15, "kits23": 3, "lits": 2, "msd_pancreas": 2, "totalseg": 15}
SPREAD_MIN = 0.01
N_BOOT = 4000  # paired segmentation bootstrap and the clustered label-level bootstrap
N_CASE = 1000  # case-level resamples per label (third stage)
SEED_PAIRED = 20261031
SEED_LABEL = 20261025
SEED_LABEL_VOCO = 20261027
MAIN_MODES = ("pertask", "joint")

METHOD_TEX = {"lin": "Lin", "attn": "Attn", "lora": "LoRA", "lorag": "LoRA$_{\\text{late}}$", "babmil": "LoRA+ABMIL",
              "fullft": "Full FT", "lin_then_lorag": "Lin$\\to$LoRA", "attn_then_lorag": "Attn$\\to$LoRA", "dora": "DoRA",
              "rslora": "rsLoRA", "attnft": "Attn-only FT"}
SEG_METHOD_TEX = {"lin": "Lin", "dec": "Dec", "lora": "LoRA", "lorag": "LoRA$_{\\text{late}}$", "dora": "DoRA",
                  "fullft": "Full FT", "dec_then_lora": "Dec$\\to$LoRA"}
FIX_TEX = {"fullft": "default", "fullft_lowlr": "lower LR", "fullft_llrd": "layer decay", "fullft_accum": "accum.\\ + clip"}

AMOS_ORGANS = ["spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "postcava",
               "pancreas", "right_adrenal", "left_adrenal", "duodenum", "bladder", "prostate_uterus"]
SEG_CHANNELS = {
    "amos22": AMOS_ORGANS,
    "kits23": ["kidney", "tumor", "cyst"],
    "lits": ["liver", "tumor"],
    "msd_pancreas": ["pancreas", "tumor"],
    "totalseg": ["lungs_airway", "heart", "arteries", "veins", "liver", "spleen", "pancreas", "kidneys", "gi_tract",
                 "bladder_prostate", "glands", "brain_spinal_cord", "spine", "other_bones", "muscles"],
}
CANONICAL = ["amos:adrenal_hyperplasia", "amos:ascites", "amos:atherosclerosis", "amos:cholecystitis", "amos:colorectal_cancer",
             "amos:fatty_liver", "amos:gallstone", "amos:hydronephrosis", "amos:kidney_stone", "amos:liver_calcifications",
             "amos:liver_cyst", "amos:liver_lesion", "amos:lymphadenopathy", "amos:renal_cyst", "amos:splenomegaly",
             "covid_ct:covid", "luna25:lung_nodule_malignancy"]
CLS_TASKS = {  # fallback label lists when a run YAML is missing
    "amos_covid_luna": CANONICAL,
    "fdg": ["NEGATIVE", "MELANOMA", "LUNG_CANCER", "LYMPHOMA"],
    "rsna2022": ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    "rsna2023": ["any_injury", "kidney_injury", "liver_injury", "spleen_injury"],
    "rsna2019_ich": ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"],
    "breastdivider": ["lesion_presence", "lesion_malignant"],
}
COHORT_SHORT = {"amos_covid_luna": "amos_covid_luna", "rsna2019_ich": "ich"}  # cohort_* macro keys of the draft

# label-level comparisons: key -> (task, spec); spec = ("diff", a, b, modes) or ("joint",)
LABEL_KEYS: list[tuple[str, str, tuple]] = []
for _k, _a, _b, _m in [("cls_lora_lin", "lora", "lin", MAIN_MODES), ("cls_attnft_lora", "attnft", "lora", MAIN_MODES),
                       ("cls_fullft_lin", "fullft", "lin", MAIN_MODES), ("cls_attn_lin", "attn", "lin", MAIN_MODES)]:
    LABEL_KEYS.append((_k, "cls", ("diff", _a, _b, _m)))
for _mt in CLS_METHODS:
    if _mt != "lin" and f"cls_{_mt}_lin" not in {k for k, _, _ in LABEL_KEYS}:
        LABEL_KEYS.append((f"cls_{_mt}_lin", "cls", ("diff", _mt, "lin", MAIN_MODES)))
LABEL_KEYS.append(("cls_joint_pertask", "cls", ("joint",)))
for _k, _a, _b, _m in [("seg_dec_lin", "dec", "lin", MAIN_MODES), ("seg_lora_dec", "lora", "dec", MAIN_MODES),
                       ("seg_lora_dec_pertask", "lora", "dec", ("pertask",)), ("seg_lora_dec_joint", "lora", "dec", ("joint",)),
                       ("seg_fullft_dec", "fullft", "dec", MAIN_MODES)]:
    LABEL_KEYS.append((_k, "seg", ("diff", _a, _b, _m)))
for _mt in SEG_METHODS:
    if _mt != "dec" and f"seg_{_mt}_dec" not in {k for k, _, _ in LABEL_KEYS}:
        LABEL_KEYS.append((f"seg_{_mt}_dec", "seg", ("diff", _mt, "dec", MAIN_MODES)))
LABEL_KEYS.append(("seg_joint_pertask", "seg", ("joint",)))
# macro key of the draft for each label-level key (ci_<oldkey>_<backbone>_<suffix>)
OLD_CI_KEY = {"cls_lora_lin": "clsloralin", "cls_attnft_lora": "clsattnftlora", "cls_fullft_lin": "clsfullftlin",
              "cls_joint_pertask": "clsjoint", "seg_dec_lin": "segdeclin", "seg_lora_dec": "segloradec",
              "seg_lora_dec_pertask": "segloradecpt", "seg_lora_dec_joint": "segloradecjt", "seg_fullft_dec": "segfullftdec",
              "seg_joint_pertask": "segjoint"}
LABEL_KEY_TEX = {"cls_lora_lin": "LoRA $-$ Lin", "cls_attnft_lora": "Attn-only FT $-$ LoRA", "cls_fullft_lin": "Full FT $-$ Lin",
                 "cls_joint_pertask": "Joint $-$ per-task", "seg_dec_lin": "Dec $-$ Lin", "seg_lora_dec": "LoRA $-$ Dec",
                 "seg_lora_dec_pertask": "LoRA $-$ Dec, per-structure models", "seg_lora_dec_joint": "LoRA $-$ Dec, joint models",
                 "seg_fullft_dec": "Full FT $-$ Dec", "seg_joint_pertask": "Joint $-$ per-task"}

_DIGITS = str.maketrans("0123456789", "ZABCDEFGHI")


def macro_name(key: str) -> str:
    """'paired_pillar0_lin_lora_meandiff' -> 'numPairedPillarZLinLoraMeandiff' (make_tables.macro_name)."""
    return "num" + "".join(ch for ch in key.title() if ch.isalnum()).translate(_DIGITS)


def crc(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- metrics
def auroc_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC per row of (B, n) arrays with midranks (ties count 0.5); NaN with a single class."""
    r = rankdata(p, axis=1)
    n_pos = y.sum(axis=1)
    n_neg = y.shape[1] - n_pos
    s = (r * y).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    auc = np.asarray(auc, dtype=float)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def auroc1(y: np.ndarray, p: np.ndarray) -> float:
    return float(auroc_rows(y[None, :], p[None, :])[0])


def auprc1(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (sklearn average_precision_score: sum over distinct thresholds of dRecall x Precision)."""
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    n_pos = ys.sum()
    if n_pos == 0:
        return float("nan")
    recall = tp / n_pos
    last = np.ones(len(ps), dtype=bool)
    last[:-1] = ps[:-1] != ps[1:]
    rec_block = np.where(last, recall, 0.0)
    cm = np.maximum.accumulate(rec_block)
    prev = np.concatenate([[0.0], cm[:-1]])
    delta = np.where(last, recall - prev, 0.0)
    return float((delta * precision).sum())


def _midrank_components(pos: np.ndarray, neg: np.ndarray):
    m, n = len(pos), len(neg)
    tz = rankdata(np.concatenate([pos, neg]))
    tx = rankdata(pos)
    ty = rankdata(neg)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    return float(auc), v10, v01


def delong_paired(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Two-sided DeLong test of AUROC(p1) - AUROC(p2) on the same cases (Sun & Xu 2014 fast midrank version)."""
    pos, neg = y == 1, y == 0
    a1, v10_1, v01_1 = _midrank_components(p1[pos], p1[neg])
    a2, v10_2, v01_2 = _midrank_components(p2[pos], p2[neg])
    m, n = int(pos.sum()), int(neg.sum())
    s10 = np.cov(np.vstack([v10_1, v10_2])) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.vstack([v01_1, v01_2])) if n > 1 else np.zeros((2, 2))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = a1 - a2
    if var <= 0:
        z = 0.0 if diff == 0 else math.inf * np.sign(diff)
    else:
        z = diff / math.sqrt(var)
    pval = float(2 * norm.sf(abs(z))) if np.isfinite(z) else 0.0
    return {"auc1": a1, "auc2": a2, "diff": diff, "se": math.sqrt(max(var, 0.0)), "p": pval}


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone)."""
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        running = min(running, p[order[rank]] * n / (rank + 1))
        adj[order[rank]] = running
    return np.minimum(adj, 1.0)


def boot_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


# ----------------------------------------------------------------------------- inventory and per-case files
def report_mode(task: str, cohort: str, mode: str) -> tuple[str, int | None]:
    """Reporting mode and the seed implied by the mode (None = keep the inventory seed)."""
    if mode == "joint_s43":
        return "joint", 43
    if mode == "joint_s44":
        return "joint", 44
    if task == "seg" and cohort in SEG_LONG_JOINT:
        if mode == "joint_lm":
            return "joint", None
        if mode == "joint":
            return "joint_8k", None
    return mode, None


def load_inventory(path: Path) -> pd.DataFrame:
    inv = pd.read_csv(path)
    use = inv[(inv.status == "finished") & (inv.in_design == "yes") & inv.excluded.isna() & inv.aux.isna()].copy()
    use = use.sort_values(["task", "cohort", "backbone", "mode", "method", "label", "site"]).reset_index(drop=True)
    use["rel"] = use.path.str.replace(r"^raw/[a-z]+/", "", regex=True)
    dup = use[use.rel.duplicated(keep=False)]
    if len(dup):
        log(f"WARNING: {dup.rel.nunique()} runs finished on both sites; keeping the first in site order (gaia, snellius)")
        use = use.sort_values("site").drop_duplicates("rel").sort_values(["task", "cohort", "backbone", "mode", "method", "label"])
    rm = [report_mode(r.task, r.cohort, r["mode"]) for _, r in use.iterrows()]
    use["mode_raw"] = use["mode"]
    use["mode"] = [m for m, _ in rm]
    use["seed"] = [s if s is not None else int(r) for (_, s), r in zip(rm, use.seed)]
    use["run_id"] = np.arange(len(use))
    use["label_run"] = use.label  # "joint" for joint runs, the task for per-task runs
    use["percase_file"] = [f"{r.path}/val_case_metrics/iter_{int(r.best_iter):06d}/{'logits' if r.task == 'cls' else 'dice'}.csv"
                           for _, r in use.iterrows()]
    return use


def yaml_task_names(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    lines = path.read_text().splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == "task_names:")
    except StopIteration:
        return None
    out = []
    for ln in lines[i + 1:]:
        if not ln.strip().startswith("- "):
            break
        out.append(ln.strip()[2:])
    return out


def label_names(row) -> list[str]:
    """Names of the labels (task_index order) or structures (class_index order) of a run."""
    if row.task == "cls":
        parts = row.rel.split("/")  # <cohort>/<bb>/<mode_raw>/<method>/<config | task/task>
        stem = parts[-1]
        names = yaml_task_names(FM_ADAPT / parts[0] / parts[1] / parts[2] / parts[3] / f"{stem}.yaml")
        if names is None:
            names = CLS_TASKS[row.cohort] if row.label_run == "joint" else [t for t in (CANONICAL if row.cohort == "amos_covid_luna" else CLS_TASKS[row.cohort]) if t.split(":")[-1] == row.label_run]
            names = list(names)
        return names
    return list(SEG_CHANNELS[row.cohort]) if row.label_run == "joint" else [row.label_run]


def read_best(path: Path) -> tuple[int, float]:
    txt = path.read_text()
    it = int(re.search(r"iteration:\s*(\d+)", txt).group(1))
    bv = float(re.search(r"best_value:\s*([-0-9.eE+]+)", txt).group(1))
    return it, bv


def _parse_one(args) -> tuple[int, dict]:
    """Worker: per-case arrays of one run, keyed by label name; splits val/test."""
    run_id, task, file, names, results = args
    df = pd.read_csv(Path(results) / file)
    out: dict = {}
    if task == "cls":
        df["task_index"] = df.task_index.fillna(0).astype(int)
        for (ti, sp), g in df.groupby(["task_index", "split"], sort=True):
            if ti >= len(names):
                out.setdefault("__warn__", []).append(f"task_index {ti} beyond {len(names)} names")
                continue
            d = out.setdefault(names[ti], {})
            d[sp] = (g.case_id.astype(str).to_numpy(), g.label.astype(float).astype(np.int8).to_numpy(),
                     g.probability.astype(np.float64).to_numpy(), g.task.astype(str).iloc[0])
    else:
        for (ci, sp), g in df.groupby(["class_index", "split"], sort=True):
            ci = int(ci)
            if ci >= len(names):
                out.setdefault("__warn__", []).append(f"class_index {ci} beyond {len(names)} names")
                continue
            d = out.setdefault(names[ci], {})
            d[sp] = (g.case_id.astype(str).to_numpy(), g.dice.astype(np.float64).to_numpy(), g.valid.astype(int).to_numpy() == 1)
    return run_id, out


def load_percase(runs: pd.DataFrame, results: Path, cache: Path | None, workers: int) -> dict[int, dict]:
    """Per-case arrays of every run (cached by file path, size and mtime)."""
    names = {r.run_id: label_names(r) for r in runs.itertuples()}
    sig = {}
    for r in runs.itertuples():
        st = (results / r.percase_file).stat()
        sig[r.run_id] = (r.percase_file, st.st_size, int(st.st_mtime), tuple(names[r.run_id]))
    store: dict = {}
    if cache is not None and cache.is_file():
        try:
            with cache.open("rb") as f:
                old = pickle.load(f)
            store = {rid: old[s] for rid, s in sig.items() if s in old}
        except Exception as e:  # noqa: BLE001
            log(f"cache unreadable ({e}); re-reading every file")
    todo = [(r.run_id, r.task, r.percase_file, names[r.run_id], str(results)) for r in runs.itertuples() if r.run_id not in store]
    log(f"per-case files: {len(store)} cached, {len(todo)} to read with {workers} workers")
    if todo:
        if workers > 1:
            with Pool(workers) as pool:
                for rid, out in pool.imap_unordered(_parse_one, todo, chunksize=8):
                    store[rid] = out
        else:
            for t in todo:
                rid, out = _parse_one(t)
                store[rid] = out
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as f:
                pickle.dump({sig[rid]: v for rid, v in store.items()}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return store


# ----------------------------------------------------------------------------- per-run metrics
def per_run_metrics(runs: pd.DataFrame, store: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cls_rows, seg_rows, notes = [], [], []
    for r in runs.itertuples():
        data = store[r.run_id]
        for w in data.get("__warn__", []):
            notes.append(f"{r.rel}: {w}")
        base = dict(task=r.task, backbone=r.backbone, cohort=r.cohort, mode=r.mode, method=r.method, seed=int(r.seed))
        for lab, splits in data.items():
            if lab == "__warn__":
                continue
            if r.task == "cls":
                if "val" not in splits or "test" not in splits:
                    notes.append(f"{r.rel} {lab}: missing split ({sorted(splits)}); label skipped")
                    continue
                yv, yt = splits["val"][1], splits["test"][1]
                if len(yv) < 2 or len(yt) < 2 or len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
                    notes.append(f"{r.rel} {lab}: single class (val n={len(yv)} pos={int(yv.sum())}; test n={len(yt)} pos={int(yt.sum())}); label skipped")
                    continue
                for sp in ("val", "test"):
                    ids, y, p, _ = splits[sp]
                    cls_rows.append({**base, "label": lab, "split": sp, "auroc": auroc1(y, p), "auprc": auprc1(y, p),
                                     "n": int(len(y)), "n_pos": int(y.sum()), "pred_sd": float(p.std())})
            else:
                for sp in ("val", "test"):
                    if sp not in splits:
                        continue
                    ids, dice, valid = splits[sp]
                    nv = int(valid.sum())
                    seg_rows.append({**base, "label": lab, "split": sp, "dice": float(dice[valid].mean()) if nv else float("nan"), "n_valid": nv})
    return pd.DataFrame(cls_rows), pd.DataFrame(seg_rows), notes


def exclude(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.cohort.isin(EXCLUDED_COHORTS) & ~df.label.isin(EXCLUDED_LABELS)].copy()


# ----------------------------------------------------------------------------- paired tests
def align(a, b):
    """Common ordering of two held-out arrays (case ids, labels): positional when the sequences are identical, by
    sorted case id when both are unique sets; None when the case sets or labels differ."""
    ids_a, ids_b = a[0], b[0]
    if len(ids_a) == len(ids_b) and np.array_equal(ids_a, ids_b):
        idx_a = idx_b = np.arange(len(ids_a))
    else:
        if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b) or set(ids_a) != set(ids_b):
            return None
        idx_a, idx_b = np.argsort(ids_a, kind="stable"), np.argsort(ids_b, kind="stable")
    return idx_a, idx_b


def paired_cls(runs: pd.DataFrame, store: dict, mcls: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """DeLong test of every method against each baseline per (backbone, cohort, mode, label), held-out split."""
    scored = {(r.backbone, r.cohort, r.mode, r.method, r.label) for r in mcls[(mcls.seed == 42) & (mcls.split == "test")].itertuples()}
    groups: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for r in runs[(runs.task == "cls") & (runs.seed == 42) & runs["mode"].isin(MAIN_MODES) & runs.method.isin(CLS_METHODS)].itertuples():
        for lab, splits in store[r.run_id].items():
            if lab == "__warn__" or "test" not in splits or (r.backbone, r.cohort, r.mode, r.method, lab) not in scored:
                continue
            groups[(r.backbone, r.cohort, r.mode, lab)][r.method] = splits["test"]
    rows, notes = [], []
    for key in sorted(groups):
        bym = groups[key]
        for base in CLS_BASELINES:
            if base not in bym:
                continue
            b = bym[base]
            for mt in CLS_METHODS:
                if mt == base or mt not in bym:
                    continue
                m = bym[mt]
                al = align(m, b)
                if al is None or not np.array_equal(m[1][al[0]], b[1][al[1]]):
                    notes.append(f"paired cls {'/'.join(key)} {mt} vs {base}: case sets or labels differ; skipped")
                    continue
                y = b[1][al[1]]
                dl = delong_paired(y, m[2][al[0]], b[2][al[1]])
                rows.append(dict(backbone=key[0], cohort=key[1], mode=key[2], label=key[3], method=mt, baseline=base,
                                 value_method=dl["auc1"], value_baseline=dl["auc2"], diff=dl["diff"], se=dl["se"], p=dl["p"],
                                 n=int(len(y)), n_pos=int(y.sum())))
    df = pd.DataFrame(rows)
    df["p_bh"] = np.nan
    for (_, _), idx in df.groupby([df.baseline, df.backbone == "voco"]).groups.items():
        df.loc[idx, "p_bh"] = bh_adjust(df.loc[idx, "p"].to_numpy())
    return exclude(df), notes


def paired_seg(runs: pd.DataFrame, store: dict) -> tuple[pd.DataFrame, list[str]]:
    """Paired case bootstrap (N_BOOT resamples, RNG keyed on the group) of the mean held-out Dice difference against
    each baseline per (backbone, cohort, mode, structure); cases valid for both methods enter."""
    groups: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for r in runs[(runs.task == "seg") & (runs.seed == 42) & runs["mode"].isin(MAIN_MODES) & runs.method.isin(SEG_METHODS)].itertuples():
        for lab, splits in store[r.run_id].items():
            if lab != "__warn__" and "test" in splits:
                groups[(r.backbone, r.cohort, r.mode, lab)][r.method] = splits["test"]
    rows, notes = [], []
    for key in sorted(groups):
        bym = groups[key]
        rng = np.random.default_rng([SEED_PAIRED, crc(*key)])
        n_cases = None
        idx_boot = None
        for base in SEG_BASELINES:
            if base not in bym:
                continue
            b = bym[base]
            for mt in SEG_METHODS:
                if mt == base or mt not in bym:
                    continue
                m = bym[mt]
                al = align(m, b)
                if al is None:
                    notes.append(f"paired seg {'/'.join(key)} {mt} vs {base}: case sets differ; skipped")
                    continue
                dm, db = m[1][al[0]], b[1][al[1]]
                vm, vb = m[2][al[0]], b[2][al[1]]
                both = vm & vb
                if n_cases is None:
                    n_cases = len(both)
                    idx_boot = rng.integers(0, n_cases, size=(N_BOOT, n_cases))  # shared by every comparison of the group
                if both.sum() < 2:
                    notes.append(f"paired seg {'/'.join(key)} {mt} vs {base}: fewer than 2 cases valid for both; skipped")
                    continue
                diffs = np.where(both, dm - db, np.nan)
                with np.errstate(all="ignore"):
                    boot = np.nanmean(diffs[idx_boot], axis=1)
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                rows.append(dict(backbone=key[0], cohort=key[1], mode=key[2], label=key[3], method=mt, baseline=base,
                                 value_method=float(dm[both].mean()), value_baseline=float(db[both].mean()), diff=float(np.nanmean(diffs)),
                                 ci_lo=float(lo), ci_hi=float(hi), p=boot_p(boot), n=int(both.sum()),
                                 note="" if both.all() else f"valid for both: {int(both.sum())} of {int(vm.sum())}/{int(vb.sum())}"))
    df = pd.DataFrame(rows)
    df["p_bh"] = np.nan
    for (_, _), idx in df.groupby([df.baseline, df.backbone == "voco"]).groups.items():
        df.loc[idx, "p_bh"] = bh_adjust(df.loc[idx, "p"].to_numpy())
    return df, notes


# ----------------------------------------------------------------------------- label-level estimates
def _auroc_boot(s: np.ndarray, n_pos: int) -> np.ndarray:
    r = rankdata(s, axis=1)
    n_neg = s.shape[1] - n_pos
    return (r[:, :n_pos].sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def label_boot_cls(arrs: dict, terms: list, rng: np.random.Generator) -> np.ndarray | None:
    """Case-level bootstrap of sum_k w_k AUROC_k for one label: positives and negatives resampled separately with the
    same row indices for every term (rows aligned on case id)."""
    ref = arrs[terms[0][0]]
    aligned = {}
    for k, _ in terms:
        al = align(arrs[k], ref)
        if al is None or not np.array_equal(arrs[k][1][al[0]], ref[1][al[1]]):
            return None
        aligned[k] = (arrs[k][2][al[0]], al[1])
    y = ref[1][aligned[terms[0][0]][1]]
    # every term is now in the order of ref[al[1]]; al[1] is identical for all (same ref)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return None
    idx = np.concatenate([pos[rng.integers(0, len(pos), (N_CASE, len(pos)))], neg[rng.integers(0, len(neg), (N_CASE, len(neg)))]], 1)
    out = np.zeros(N_CASE)
    for k, w in terms:
        out += w * _auroc_boot(aligned[k][0][idx], len(pos))
    return out


def label_boot_seg(arrs: dict, terms: list, rng: np.random.Generator) -> np.ndarray | None:
    """Case-level bootstrap of sum_k w_k meanDice_k for one structure (invalid cases are NaN and skipped per term)."""
    ref = arrs[terms[0][0]]
    vals = {}
    for k, _ in terms:
        al = align(arrs[k], ref)
        if al is None:
            return None
        d = arrs[k][1][al[0]].copy()
        d[~arrs[k][2][al[0]]] = np.nan
        vals[k] = d
    n = len(ref[0])
    idx = rng.integers(0, n, (N_CASE, n))
    out = np.zeros(N_CASE)
    for k, w in terms:
        with np.errstate(all="ignore"):
            out += w * np.nanmean(vals[k][idx], axis=1)
    return out if np.isfinite(out).all() else None


def clustered_ci(vals: list[np.ndarray], rng: np.random.Generator, case: list[np.ndarray] | None) -> tuple[float, float]:
    """Percentile interval of the mean over labels from N_BOOT two-stage resamples (clusters with replacement, then
    labels within each drawn cluster with replacement); with `case` (per cluster: labels x N_CASE case-level draws) a
    third stage replaces each drawn label's value by one of its case-level draws."""
    G = len(vals)
    C = rng.integers(0, G, (N_BOOT, G))  # the clusters drawn in every resample
    total = np.zeros(N_BOOT)
    count = np.zeros(N_BOOT)
    for i in range(G):
        n_i = len(vals[i])
        j = rng.integers(0, n_i, (N_BOOT, G, n_i))  # label draws for every slot in which cluster i could appear
        if case is None:
            drawn = vals[i][j]
        else:
            drawn = case[i][j, rng.integers(0, N_CASE, (N_BOOT, G, n_i))]
        hit = C == i  # slots that drew cluster i
        total += (drawn.sum(axis=2) * hit).sum(axis=1)
        count += hit.sum(axis=1) * n_i
    bs = total / count
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


_LL: dict = {}  # per-label values and arrays shared with the worker processes (fork)


def _label_recs(item) -> tuple[str, list[dict], list[str]]:
    """Worker: point value and case-level draws of every label of one label-level key."""
    key, task, spec = item
    val, arr, dnt = _LL["val"], _LL["arr"], _LL["dnt"]
    keep_dnt = key == "cls_fullft_lin"
    recs, notes = [], []
    for (tk, bb, ds, lab), rows_ in sorted(val.items()):
        if tk != task:
            continue
        avail = {k for k in rows_ if keep_dnt or task != "cls" or (bb not in MAIN_BACKBONES) or ((bb, k[0]) not in dnt)}
        if spec[0] == "diff":
            _, a, b, modes = spec
            if any((x, mo) not in avail for x in (a, b) for mo in modes):
                continue
            terms = [((a, mo), 1 / len(modes)) for mo in modes] + [((b, mo), -1 / len(modes)) for mo in modes]
        else:
            meths = sorted({mt for mt, _ in avail if (mt, "pertask") in avail and (mt, "joint") in avail})
            if not meths:
                continue
            terms = [((mt, "joint"), 1 / len(meths)) for mt in meths] + [((mt, "pertask"), -1 / len(meths)) for mt in meths]
        v = float(sum(w * rows_[k] for k, w in terms))
        rng = np.random.default_rng([SEED_LABEL_VOCO if bb == "voco" else SEED_LABEL, crc(key, bb, ds, lab)])
        boot = (label_boot_cls if task == "cls" else label_boot_seg)(arr[(tk, bb, ds, lab)], terms, rng)
        if boot is None:
            notes.append(f"label-level {key} {bb}/{ds}/{lab}: case-level bootstrap not possible; label left out")
            continue
        recs.append(dict(backbone=bb, cohort=ds, label=lab, v=v, boot=boot - boot.mean() + v))
    return key, recs, notes


def label_level(runs: pd.DataFrame, store: dict, mcls: pd.DataFrame, mseg: pd.DataFrame, dnt: set, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # held-out values and arrays of the main runs, keyed by (task, backbone, cohort, label) -> (method, mode)
    val: dict[tuple, dict] = defaultdict(dict)
    arr: dict[tuple, dict] = defaultdict(dict)
    for task, m in (("cls", mcls), ("seg", mseg)):
        d = m[(m.seed == 42) & (m.split == "test") & m["mode"].isin(MAIN_MODES) & m.method.isin(CLS_METHODS if task == "cls" else SEG_METHODS)]
        col = "auroc" if task == "cls" else "dice"
        for r in d.itertuples():
            val[(task, r.backbone, r.cohort, r.label)][(r.method, r.mode)] = getattr(r, col)
    for r in runs[(runs.seed == 42) & runs["mode"].isin(MAIN_MODES)].itertuples():
        meths = CLS_METHODS if r.task == "cls" else SEG_METHODS
        if r.method not in meths:
            continue
        for lab, splits in store[r.run_id].items():
            if lab != "__warn__" and "test" in splits and (r.method, r.mode) in val.get((r.task, r.backbone, r.cohort, lab), {}):
                arr[(r.task, r.backbone, r.cohort, lab)][(r.method, r.mode)] = splits["test"]
    _LL.update(val=dict(val), arr=dict(arr), dnt=dnt)
    order = sorted(LABEL_KEYS, key=lambda k: (k[0] != "cls_joint_pertask", k[1] != "cls"))  # heavy keys first
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            results = dict((k, (recs, nts)) for k, recs, nts in pool.imap_unordered(_label_recs, order))
    else:
        results = {k: (recs, nts) for k, recs, nts in map(_label_recs, order)}
    log(f"  case-level draws of {len(order)} keys: {time.time() - t0:.1f} s")
    rows_out, vals_out, notes = [], [], []
    for key, task, spec in LABEL_KEYS:
        recs, nts = results[key]
        notes += nts
        x = pd.DataFrame(recs)
        if x.empty:
            continue
        for r in x.itertuples():
            vals_out.append(dict(key=key, task=task, backbone=r.backbone, cohort=r.cohort, label=r.label, value=r.v))
        x["cluster"] = x.cohort + "/" + x.backbone
        for bb in ["all"] + [b for b in ALL_BACKBONES if (x.backbone == b).any()]:
            g = x[x.backbone.isin(MAIN_BACKBONES)] if bb == "all" else x[x.backbone == bb]
            if g.empty:
                continue
            rng = np.random.default_rng([SEED_LABEL_VOCO if bb == "voco" else SEED_LABEL, crc(key, bb)])
            groups = [gg for _, gg in g.groupby("cluster", sort=True)]
            vals = [gg.v.to_numpy() for gg in groups]
            case = [np.stack(gg.boot.tolist()) for gg in groups]
            lo2, hi2 = clustered_ci(vals, rng, None)
            lo3, hi3 = clustered_ci(vals, rng, case)
            cm = g.groupby("cluster").v.mean()
            rows_out.append(dict(key=key, task=task, backbone=bb, n_labels=int(len(g)), mean=float(g.v.mean()), lo2=lo2, hi2=hi2,
                                 lo3=lo3, hi3=hi3, clusters_pos=int((cm > 0).sum()), clusters_n=int(len(cm))))
    return pd.DataFrame(rows_out), pd.DataFrame(vals_out), notes


# ----------------------------------------------------------------------------- ranks
def ranks(m: pd.DataFrame, task: str, methods: list[str], col: str, dnt: set) -> tuple[pd.DataFrame, dict]:
    """Slots = backbone x cohort x label x mode with every expected method (minus did-not-train pairs on the main
    backbones); average ranks on the held-out metric; per backbone and pooled over the main backbones."""
    d = m[(m.seed == 42) & (m.split == "test") & m["mode"].isin(MAIN_MODES) & m.method.isin(methods)]
    piv = d.pivot_table(index=["backbone", "cohort", "label", "mode"], columns="method", values=col).reindex(columns=methods)
    for bb, mt in dnt:  # pairs that did not train leave the ranking
        piv.loc[piv.index.get_level_values("backbone") == bb, mt] = np.nan
    expected = pd.Series([sum((bb, mt) not in dnt for mt in methods) if bb in MAIN_BACKBONES else len(methods)
                          for bb in piv.index.get_level_values("backbone")], index=piv.index)
    piv = piv[piv.notna().sum(axis=1) == expected]
    rk = piv.rank(axis=1, ascending=False)
    best = piv.max(axis=1)
    near = piv.ge(best.values[:, None] - 0.01).where(piv.notna()).astype(float).where(piv.notna())
    win = piv.eq(best.values[:, None]).where(piv.notna()).astype(float).where(piv.notna())
    regret = best.values[:, None] - piv
    rows = []
    info = {"slots": piv, "rk": rk, "near": near, "win": win, "regret": regret}
    for bb in ALL_BACKBONES + ["all"]:
        sel = piv.index.get_level_values("backbone").isin(MAIN_BACKBONES) if bb == "all" else piv.index.get_level_values("backbone") == bb
        if sel.sum() == 0:
            continue
        for mt in methods:
            r_ = rk.loc[sel, mt]
            if r_.notna().sum() == 0:
                rows.append(dict(task=task, backbone=bb, method=mt, mean_rank=np.nan, n_slots=0, within_001_pct=np.nan, regret=np.nan))
                continue
            rows.append(dict(task=task, backbone=bb, method=mt, mean_rank=float(r_.mean()), n_slots=int(r_.notna().sum()),
                             within_001_pct=float(near.loc[sel, mt].mean() * 100), regret=float(regret.loc[sel, mt].mean())))
    return pd.DataFrame(rows), info


# ----------------------------------------------------------------------------- formatting of macros
def fmt_macro(k: str, v) -> str | None:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if k.startswith("segrecomp_") and k.endswith("diff"):
        return f"{v:.4f}"
    if k.startswith(("rank_", "segrank_", "vocorank_", "vocosegrank_")) and k.endswith("_mean"):
        return f"{v:.2f}"
    if "better" in k or "worse" in k or k.endswith("_n") or k.endswith("_sig") or k.startswith("cohort_") or k.startswith("bestrows_") or k.endswith("_ntest") or k.endswith("_hit"):
        return f"{v:.0f}"
    if k.endswith("_spearman") or k == "gaincorr":
        return f"{v:.2f}"
    if "meandiff" in k or "delta" in k or k.endswith("diff"):
        s = f"{v:+.3f}"
        return "0.000" if s in ("-0.000", "+0.000") else s
    return f"{v:.3f}"


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--draft-numbers", type=Path, default=DEFAULT_DRAFT_NUMBERS)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []

    def check(msg: str) -> None:
        checks.append(msg)
        log(msg)

    # ------------------------------------------------------------------ runs
    runs = load_inventory(args.inventory)
    log(f"runs used: {len(runs)} ({runs.groupby('task').size().to_dict()})")
    store = load_percase(runs, args.results, None if args.no_cache else out / "_cache" / "percase.pkl", args.workers)
    runs_out = runs[["task", "site", "cohort", "backbone", "mode", "method", "seed", "label_run", "best_iter", "n_validations_logged",
                     "last_iter_logged", "path", "percase_file"]].rename(columns={"label_run": "label", "n_validations_logged": "n_validations",
                                                                                "last_iter_logged": "final_iter"}).copy()
    runs_out["best_iter"] = runs_out.best_iter.astype(int)
    runs_out["final_iter"] = runs_out.final_iter.astype(int)
    runs_out["n_validations"] = runs_out.n_validations.astype(int)
    runs_out.to_csv(out / "runs.csv", index=False)

    # ------------------------------------------------------------------ per-run metrics
    mcls_all, mseg_all, notes = per_run_metrics(runs, store)
    mcls_all.to_csv(out / "metrics_cls.csv", index=False, float_format="%.10g")
    mseg_all.to_csv(out / "metrics_seg.csv", index=False, float_format="%.10g")
    mcls, mseg = exclude(mcls_all), exclude(mseg_all)
    log(f"metrics_cls: {len(mcls_all)} rows; metrics_seg: {len(mseg_all)} rows; {len(notes)} notes")
    # label name consistency: per-task labels must be joint labels of the same cohort
    for coh, g in mcls.groupby("cohort"):
        jl = set(g[g["mode"].isin(["joint", "joint_lm"])].label)
        pl = set(g[g["mode"] == "pertask"].label)
        if pl - jl:
            check(f"WARNING: cohort {coh}: per-task labels not among joint labels: {sorted(pl - jl)}")

    # ------------------------------------------------------------------ spread / did not train
    sp_rows = []
    dnt: set = set()
    d = mcls[(mcls.seed == 42) & (mcls.split == "test") & mcls["mode"].isin(MAIN_MODES) & mcls.method.isin(CLS_METHODS)]
    for (bb, mt), g in d.groupby(["backbone", "method"]):
        med = float(g.pred_sd.median())
        flag = bool(med < SPREAD_MIN and bb in MAIN_BACKBONES)
        if flag:
            dnt.add((bb, mt))
        sp_rows.append(dict(backbone=bb, method=mt, median_sd=med, did_not_train=flag, n_pairs=int(len(g))))
    spread = pd.DataFrame(sp_rows)
    spread.to_csv(out / "spread.csv", index=False, float_format="%.6g")
    log(f"did not train: {sorted(dnt)}")

    def drop_dnt(df: pd.DataFrame) -> pd.DataFrame:
        bad = pd.Series(list(zip(df.backbone, df.method)), index=df.index).isin(dnt)
        if "baseline" in df.columns:
            bad |= pd.Series(list(zip(df.backbone, df.baseline)), index=df.index).isin(dnt)
        return df[~bad]

    # ------------------------------------------------------------------ paired tests
    t0 = time.time()
    pcls, n1 = paired_cls(runs, store, mcls)
    pseg, n2 = paired_seg(runs, store)
    pseg = exclude(pseg)
    notes += n1 + n2
    pcls.to_csv(out / "paired_cls.csv", index=False, float_format="%.10g")
    pseg.to_csv(out / "paired_seg.csv", index=False, float_format="%.10g")
    log(f"paired_cls: {len(pcls)} rows; paired_seg: {len(pseg)} rows ({time.time() - t0:.1f} s)")

    # ------------------------------------------------------------------ label-level
    t0 = time.time()
    ll, llv, n3 = label_level(runs, store, mcls, mseg, dnt, args.workers)
    notes += n3
    ll.to_csv(out / "label_level.csv", index=False, float_format="%.10g")
    llv.to_csv(out / "label_level_values.csv", index=False, float_format="%.10g")
    log(f"label_level: {len(ll)} rows, {len(llv)} label values ({time.time() - t0:.1f} s)")

    # ------------------------------------------------------------------ ranks
    rk_cls, info_cls = ranks(mcls, "cls", CLS_METHODS, "auroc", dnt)
    rk_seg, info_seg = ranks(mseg, "seg", SEG_METHODS, "dice", set())
    rk = pd.concat([rk_cls, rk_seg], ignore_index=True)
    rk.to_csv(out / "ranks.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------------ convergence
    cv = runs.copy()
    cv["interval"] = cv.last_iter_logged / cv.n_validations_logged
    cv["is_last"] = cv.best_iter >= cv.last_iter_logged
    cv["is_last_two"] = cv.best_iter >= cv.last_iter_logged - cv.interval
    conv_rows = []
    for (task, bb, mt, mode), g in cv.groupby(["task", "backbone", "method", "mode"]):
        conv_rows.append(dict(task=task, backbone=bb, method=mt, mode=mode, n_runs=len(g), frac_last=g.is_last.mean(), frac_last_two=g.is_last_two.mean()))
    for task, g in cv.groupby("task"):
        conv_rows.append(dict(task=task, backbone="all", method="all", mode="all", n_runs=len(g), frac_last=g.is_last.mean(), frac_last_two=g.is_last_two.mean()))
    pd.DataFrame(conv_rows).to_csv(out / "convergence.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------------ cohort means (complete cells)
    cm_rows = []
    for task, m, col in (("cls", mcls, "auroc"), ("seg", mseg, "dice")):
        t = m[m.split == "test"]
        n_exp = t[t.backbone.isin(MAIN_BACKBONES)].groupby("cohort").label.nunique().to_dict()
        for (bb, ds, mode, mt, seed), g in t.groupby(["backbone", "cohort", "mode", "method", "seed"]):
            if g.label.nunique() == n_exp.get(ds, -1):
                cm_rows.append(dict(task=task, backbone=bb, cohort=ds, mode=mode, method=mt, seed=seed, metric=col, value=float(g[col].mean()), n_labels=int(g.label.nunique())))
    cohort_means = pd.DataFrame(cm_rows)
    cohort_means.to_csv(out / "cohort_means.csv", index=False, float_format="%.10g")

    # ------------------------------------------------------------------ seeds
    seed_rows, ss_rows = [], []
    for task, m, col, meths, pairs in (("cls", mcls, "auroc", CLS_SEED_METHODS, [("lora", "lin"), ("attnft", "lora")]),
                                       ("seg", mseg, "dice", SEG_SEED_METHODS, [("lora", "dec"), ("fullft", "dec")])):
        t = m[(m.split == "test") & (m["mode"] == "joint") & m.method.isin(meths) & m.backbone.isin(MAIN_BACKBONES)]
        cells = t.groupby(["backbone", "cohort", "method", "seed"])[col].mean()
        for (bb, ds, mt, seed), v in cells.items():
            seed_rows.append(dict(task=task, backbone=bb, cohort=ds, method=mt, mode="joint", seed=seed, value=float(v)))
        sd = cells.groupby(level=["backbone", "cohort", "method"]).agg(["std", "count"])
        sd = sd[sd["count"] >= 2]
        for bb in MAIN_BACKBONES + ["all"]:
            for mt in meths:
                x = sd.xs(mt, level="method")["std"] if mt in sd.index.get_level_values("method") else pd.Series(dtype=float)
                if bb != "all":
                    x = x[x.index.get_level_values("backbone") == bb]
                ss_rows.append(dict(task=task, backbone=bb, method=mt, comparison="", sd_median=float(x.median()) if len(x) else np.nan,
                                    sd_max=float(x.max()) if len(x) else np.nan, n_cells=int(len(x)), pairs_pos=np.nan, pairs_n=np.nan,
                                    extra_pairs_pos=np.nan, extra_pairs_n=np.nan))
        lab = t.pivot_table(index=["backbone", "cohort", "seed", "label"], columns="method", values=col)
        for a, b in pairs:
            if a not in lab or b not in lab:
                continue
            s = (lab[a] - lab[b]).dropna().groupby(level=["backbone", "cohort", "seed"]).mean()
            for bb in MAIN_BACKBONES + ["all"]:
                x = s if bb == "all" else s[s.index.get_level_values("backbone") == bb]
                extra = x[x.index.get_level_values("seed") != 42]
                ss_rows.append(dict(task=task, backbone=bb, method=a, comparison=f"{a}>{b}", sd_median=np.nan, sd_max=np.nan, n_cells=np.nan,
                                    pairs_pos=int((x > 0).sum()), pairs_n=int(len(x)), extra_pairs_pos=int((extra > 0).sum()), extra_pairs_n=int(len(extra))))
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(out / "seeds.csv", index=False, float_format="%.10g")
    seeds_summary = pd.DataFrame(ss_rows)
    seeds_summary.to_csv(out / "seeds_summary.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------------ revision runs
    rev_c = mcls[mcls.method.isin(CLS_REVISION_METHODS)].rename(columns={"auroc": "value"})
    rev_s = mseg[mseg.method.isin(SEG_REVISION_METHODS) | (mseg["mode"] == "pertask_cm")].rename(columns={"dice": "value"})
    revision = pd.concat([rev_c[["task", "backbone", "cohort", "mode", "method", "label", "split", "value"]],
                          rev_s[["task", "backbone", "cohort", "mode", "method", "label", "split", "value"]]], ignore_index=True)
    revision.to_csv(out / "revision.csv", index=False, float_format="%.10g")

    # ================================================================== macros
    macros: dict = {}
    doc: dict[str, str] = {}  # family prefix -> description (NAMING.md)

    def fam(prefix: str, text: str) -> None:
        doc.setdefault(prefix, text)

    C = mcls[(mcls.seed == 42) & mcls.method.isin(CLS_METHODS)]  # main classification runs, all modes
    S = mseg[(mseg.seed == 42) & mseg.method.isin(SEG_METHODS)]
    Ct = C[(C.split == "test") & C.backbone.isin(MAIN_BACKBONES)]
    Ct_nd = drop_dnt(Ct)
    St = S[(S.split == "test") & S.backbone.isin(MAIN_BACKBONES)]
    Cv = C[(C.split == "test") & (C.backbone == "voco")]
    Sv = S[(S.split == "test") & (S.backbone == "voco")]
    P = pcls[pcls.backbone.isin(MAIN_BACKBONES)]
    P_nd = drop_dnt(P)
    PS = pseg[pseg.backbone.isin(MAIN_BACKBONES)]

    # --- pooled means, spreads
    fam("pooled<bb><method>", "held-out AUROC pooled over labels and the per-task and joint modes, per backbone and method (no num prefix)")
    pooled = Ct[Ct["mode"].isin(MAIN_MODES)].groupby(["backbone", "method"]).auroc.mean()
    pooled_lines = [f"\\newcommand{{\\pooled{bb.translate(_DIGITS)}{mt.replace('_', '')}}}{{{v:.3f}}}" for (bb, mt), v in pooled.items()]
    fam("spread_<bb>_<method>", "median over label-mode pairs of the SD of the held-out probabilities; spread_othermin = smallest value among pairs that trained; spread_min_value = threshold; dnt_cells_n = label-mode rows of pairs that did not train")
    for r in spread.itertuples():
        macros[f"spread_{r.backbone}_{r.method}"] = r.median_sd
    sp_main = spread[spread.backbone.isin(MAIN_BACKBONES)]
    macros["spread_othermin"] = float(sp_main[~sp_main.did_not_train].median_sd.min())
    macros["spread_min_value"] = SPREAD_MIN
    macros["dnt_cells_n"] = int(len(Ct[Ct["mode"].isin(MAIN_MODES) & pd.Series(list(zip(Ct.backbone, Ct.method)), index=Ct.index).isin(dnt)]))
    macros["dnt_pairs_n"] = len(dnt)
    for bb, mt in sorted(dnt):
        macros[f"dnt_{bb}_{mt}_n"] = 1

    # --- ladder cells (complete cells only)
    fam("cls_auroc_<cohort>_<bb>_<mode>_<method>", "mean held-out AUROC over the labels of a cohort (complete cells; modes pertask, joint, joint_lm)")
    fam("seg_<dataset>_<bb>_<mode>_<method>", "mean held-out Dice over the structures of a dataset (complete cells; modes pertask, joint)")
    fam("segbudget_<dataset>_<bb>_<pertask|joint_8k|joint>_<method>", "AMOS22 / TotalSegmentator: per-structure 8000-step, joint 8000-step and joint 24000-step means")
    for r in cohort_means[(cohort_means.seed == 42)].itertuples():
        if r.task == "cls" and r.method in CLS_METHODS and r.mode in ("pertask", "joint", "joint_lm"):
            macros[f"cls_auroc_{r.cohort}_{r.backbone}_{r.mode}_{r.method}"] = r.value
        if r.task == "seg" and r.method in SEG_METHODS:
            if r.mode in MAIN_MODES:
                macros[f"seg_{r.cohort}_{r.backbone}_{r.mode}_{r.method}"] = r.value
            if r.cohort in SEG_LONG_JOINT and r.mode in ("pertask", "joint_8k", "joint") and r.backbone != "voco":
                macros[f"segbudget_{r.cohort}_{r.backbone}_{r.mode}_{r.method}"] = r.value

    # --- paired summaries (classification)
    fam("paired_<bb>_<baseline>_<method>_{meandiff,better,sigbetter,sigworse,n}", "DeLong comparisons per label-mode pair: mean difference, % pairs with the method ahead, % BH-significant (p_bh < 0.05) ahead / behind, number of pairs")
    for (bb, base, mt), g in P.groupby(["backbone", "baseline", "method"]):
        k = f"paired_{bb}_{base}_{mt}"
        macros.update({f"{k}_meandiff": g["diff"].mean(), f"{k}_better": (g["diff"] > 0).mean() * 100,
                       f"{k}_sigbetter": ((g["diff"] > 0) & (g.p_bh < 0.05)).mean() * 100, f"{k}_sigworse": ((g["diff"] < 0) & (g.p_bh < 0.05)).mean() * 100, f"{k}_n": len(g)})
    fam("head_<bb>_*", "headline row per backbone: n = label-mode pairs of Lin; LoRA vs Lin and vs Full FT (mean difference, % ahead, % significant); pooled AUROC per method")
    for bb in MAIN_BACKBONES:
        vl = P[(P.backbone == bb) & (P.baseline == "lin") & (P.method == "lora")]
        vf = P[(P.backbone == bb) & (P.baseline == "fullft") & (P.method == "lora")]
        macros[f"head_{bb}_n"] = int(len(Ct[(Ct.backbone == bb) & (Ct.method == "lin") & Ct["mode"].isin(MAIN_MODES)]))
        macros[f"head_{bb}_lora_vs_lin_meandiff"] = vl["diff"].mean()
        macros[f"head_{bb}_lora_vs_lin_better"] = (vl["diff"] > 0).mean() * 100
        macros[f"head_{bb}_lora_vs_lin_sig"] = ((vl["diff"] > 0) & (vl.p_bh < 0.05)).mean() * 100
        macros[f"head_{bb}_lora_vs_fullft_meandiff"] = vf["diff"].mean()
        macros[f"head_{bb}_lora_vs_fullft_better"] = (vf["diff"] > 0).mean() * 100
        macros[f"head_{bb}_fullft_sig_better"] = ((vf["diff"] < 0) & (vf.p_bh < 0.05)).mean() * 100
        for mt in CLS_METHODS:
            macros[f"head_{bb}_pooled_{mt}"] = float(pooled.get((bb, mt), np.nan))

    # --- ranks
    fam("rank_*, near_*, win_*, regret_*, regretmax_*, bestarm_*", "classification ranks: rank_slots_n / rank_<bb>_slots_n slots; rank_<bb>_<method>_mean and rank_<method>_mean (pooled over main backbones); near = % of slots within 0.01 of the best; win = % of slots best; regret = mean AUROC deficit to the best; regretmax = largest deficit; bestarm = method with the lowest mean rank (name)")
    piv = info_cls["slots"]
    main_sel = piv.index.get_level_values("backbone").isin(MAIN_BACKBONES)
    macros["rank_slots_n"] = int(main_sel.sum())
    for bb in MAIN_BACKBONES:
        sel = piv.index.get_level_values("backbone") == bb
        macros[f"rank_{bb}_slots_n"] = int(sel.sum())
        rb = info_cls["rk"].loc[sel]
        for mt in CLS_METHODS:
            v = float(rb[mt].mean())
            macros[f"rank_{bb}_{mt}_mean"] = v
            if np.isfinite(v):
                macros[f"near_{bb}_{mt}_better"] = float(info_cls["near"].loc[sel, mt].mean() * 100)
                macros[f"win_{bb}_{mt}_better"] = float(info_cls["win"].loc[sel, mt].mean() * 100)
                macros[f"regret_{bb}_{mt}"] = float(info_cls["regret"].loc[sel, mt].mean())
        macros[f"bestarm_{bb}"] = METHOD_TEX[rb.mean().idxmin()]
        macros[f"bestarmregret_{bb}"] = METHOD_TEX[info_cls["regret"].loc[sel].mean().idxmin()]
    rk_main = info_cls["rk"].loc[main_sel]
    for mt in CLS_METHODS:
        macros[f"rank_{mt}_mean"] = float(rk_main[mt].mean())
        macros[f"near_{mt}_better"] = float(info_cls["near"].loc[main_sel, mt].mean() * 100)
        macros[f"win_{mt}_better"] = float(info_cls["win"].loc[main_sel, mt].mean() * 100)
        macros[f"regret_{mt}"] = float(info_cls["regret"].loc[main_sel, mt].mean())
        macros[f"regretmax_{mt}"] = float(info_cls["regret"].loc[main_sel, mt].max())
    macros["bestarm_all"] = METHOD_TEX[rk_main.mean().idxmin()]
    macros["bestarmregret_all"] = METHOD_TEX[info_cls["regret"].loc[main_sel].mean().idxmin()]
    fam("segrank_*, segbestarm_*", "segmentation ranks over structure-mode slots with all seven methods, per backbone and pooled")
    piv_s = info_seg["slots"]
    sel_s = piv_s.index.get_level_values("backbone").isin(MAIN_BACKBONES)
    macros["segrank_slots_n"] = int(sel_s.sum())
    for mt in SEG_METHODS:
        macros[f"segrank_{mt}_mean"] = float(info_seg["rk"].loc[sel_s, mt].mean())
        macros[f"segnear_{mt}_better"] = float(info_seg["near"].loc[sel_s, mt].mean() * 100)
        macros[f"segregret_{mt}"] = float(info_seg["regret"].loc[sel_s, mt].mean())
    for bb in ["tapct", "spectre"]:
        sel = piv_s.index.get_level_values("backbone") == bb
        macros[f"segrank_{bb}_slots_n"] = int(sel.sum())
        for mt in SEG_METHODS:
            macros[f"segrank_{bb}_{mt}_mean"] = float(info_seg["rk"].loc[sel, mt].mean())
        macros[f"segbestarm_{bb}"] = SEG_METHOD_TEX[info_seg["rk"].loc[sel].mean().idxmin()]
    macros["segbestarm_all"] = SEG_METHOD_TEX[info_seg["rk"].loc[sel_s].mean().idxmin()]

    # --- direct differences against reference methods (did-not-train pairs dropped)
    fam("vs<ref>_<bb|all>_<method>_{meandiff,better,n}", "method minus reference (lora, lorag, babmil, fullft) on the same backbone x cohort x label x mode, pairs that trained")
    cp = Ct_nd[Ct_nd["mode"].isin(MAIN_MODES)].pivot_table(index=["backbone", "cohort", "label", "mode"], columns="method", values="auroc")
    for ref in ("lora", "lorag", "babmil", "fullft"):
        for mt in CLS_METHODS:
            if mt == ref or mt not in cp or ref not in cp:
                continue
            x = cp[[mt, ref]].dropna()
            for bb, g in x.groupby(level="backbone"):
                dd = g[mt] - g[ref]
                macros.update({f"vs{ref}_{bb}_{mt}_meandiff": dd.mean(), f"vs{ref}_{bb}_{mt}_better": (dd > 0).mean() * 100, f"vs{ref}_{bb}_{mt}_n": len(dd)})
            dd = x[mt] - x[ref]
            macros[f"vs{ref}_all_{mt}_meandiff"] = dd.mean()
            macros[f"vs{ref}_all_{mt}_better"] = (dd > 0).mean() * 100
    fam("fullftsens_tapct_{all,nofdgjoint}_meandiff", "Full FT minus Lin on TAP-CT at label level, with and without the joint FDG run")
    cpa = Ct[(Ct.backbone == "tapct") & Ct["mode"].isin(MAIN_MODES)].pivot_table(index=["cohort", "label", "mode"], columns="method", values="auroc").dropna(subset=["fullft", "lin"])
    dd = cpa["fullft"] - cpa["lin"]
    macros["fullftsens_tapct_all_meandiff"] = float(dd.groupby(level=["cohort", "label"]).mean().mean())
    keep = ~((dd.index.get_level_values("cohort") == "fdg") & (dd.index.get_level_values("mode") == "joint"))
    macros["fullftsens_tapct_nofdgjoint_meandiff"] = float(dd[keep].groupby(level=["cohort", "label"]).mean().mean())

    # --- segmentation paired summaries and structure-level deltas
    fam("segpaired_<bb>_<baseline>_<method>_{meandiff,better,sigbetter,sigworse,n}; segpairedmode_..._<mode>_{meandiff,n}", "paired-bootstrap comparisons per structure-mode pair (BH within baseline family)")
    for (bb, base, mt), g in PS[PS["mode"].isin(MAIN_MODES)].groupby(["backbone", "baseline", "method"]):
        k = f"segpaired_{bb}_{base}_{mt}"
        macros.update({f"{k}_meandiff": g["diff"].mean(), f"{k}_better": (g["diff"] > 0).mean() * 100,
                       f"{k}_sigbetter": ((g["diff"] > 0) & (g.p_bh < 0.05)).mean() * 100, f"{k}_sigworse": ((g["diff"] < 0) & (g.p_bh < 0.05)).mean() * 100, f"{k}_n": len(g)})
        for mode_, g_ in g.groupby("mode"):
            macros[f"segpairedmode_{bb}_{base}_{mt}_{mode_}_meandiff"] = g_["diff"].mean()
            macros[f"segpairedmode_{bb}_{base}_{mt}_{mode_}_n"] = len(g_)
    fam("segjoint_*, segjointds_*, segcollapse_*, segjointsmall_min_meandiff, seg_runs_n", "joint minus per-structure Dice per backbone, method and dataset; collapsed structures (joint < 0.01 while per-structure >= 0.4) and the means without them")
    pj = St[St["mode"].isin(MAIN_MODES)].pivot_table(index=["backbone", "cohort", "method", "label"], columns="mode", values="dice").dropna()
    pj["dd"] = pj["joint"] - pj["pertask"]
    for (bb, mt), g in pj.reset_index().groupby(["backbone", "method"]):
        macros[f"segjoint_{bb}_{mt}_meandiff"] = g.dd.mean()
    for bb, g in pj.reset_index().groupby("backbone"):
        macros[f"segjoint_{bb}_meandiff"] = g.dd.mean()
    for (bb, ds), g in pj.reset_index().groupby(["backbone", "cohort"]):
        macros[f"segjointds_{ds}_{bb}_meandiff"] = g.dd.mean()
    small = pj.reset_index()
    small = small[~small.cohort.isin(SEG_LONG_JOINT)]
    macros["segjointsmall_min_meandiff"] = float(small.groupby(["backbone", "cohort"]).dd.mean().min())
    col = (pj["joint"] < 0.01) & (pj["pertask"] >= 0.4)
    macros["segcollapse_n"] = int(col.sum())
    macros["segcollapse_cells_n"] = int(len(pj))
    macros["segcollapse_pertaskmin_value"] = float(pj.loc[col, "pertask"].min()) if col.any() else np.nan
    macros["segcollapse_pertaskmax_value"] = float(pj.loc[col, "pertask"].max()) if col.any() else np.nan
    macros["segcollapse_structs_n"] = int(col[col].reset_index()[["cohort", "label"]].drop_duplicates().shape[0])
    rest = pj[~col].reset_index()
    for bb, g in rest.groupby("backbone"):
        macros[f"segjoint_{bb}_nocollapse_meandiff"] = g.dd.mean()
        macros[f"segcollapse_{bb}_n"] = int(col.xs(bb, level="backbone").sum())
    for (bb, ds), g in rest.groupby(["backbone", "cohort"]):
        macros[f"segjointds_{ds}_{bb}_nocollapse_meandiff"] = g.dd.mean()
    macros["seg_runs_n"] = int(len(runs[(runs.task == "seg") & (runs.seed == 42) & runs.method.isin(SEG_METHODS) & runs.backbone.isin(MAIN_BACKBONES) & runs["mode"].isin(["pertask", "joint", "joint_8k"])]))
    fam("segjointeightk_<bb>_meandiff, segbudgetgain_<bb>_meandiff", "AMOS22/TotalSegmentator: 8000-step joint minus per-structure; 24000-step joint minus 8000-step joint")
    b8 = St[St.cohort.isin(SEG_LONG_JOINT) & St["mode"].isin(["pertask", "joint_8k", "joint"])].pivot_table(index=["backbone", "cohort", "method", "label"], columns="mode", values="dice")
    for bb, g in b8.reset_index().groupby("backbone"):
        macros[f"segjointeightk_{bb}_meandiff"] = float((g["joint_8k"] - g["pertask"]).mean())
        macros[f"segbudgetgain_{bb}_meandiff"] = float((g["joint"] - g["joint_8k"]).mean())
    fam("segmode_<dataset>_<bb>_<mode>_<lora|fullft>_meandiff, segmodepos_<bb>_<method>_<mode>_n, segmodecells_...", "LoRA minus Dec and Full FT minus Dec per dataset, backbone and mode (joint_8k written jointeightk); number of positive cells")
    sp_ = St.pivot_table(index=["backbone", "cohort", "mode", "label"], columns="method", values="dice")
    res = {mt: (sp_[mt] - sp_["dec"]).groupby(level=["backbone", "cohort", "mode"]).mean() for mt in ("lora", "fullft")}
    for mt in ("lora", "fullft"):
        for (bb, ds, mode), v in res[mt].items():
            if mode in ("pertask", "joint", "joint_8k") and np.isfinite(v):
                macros[f"segmode_{ds}_{bb}_{mode.replace('_8k', 'eightk')}_{mt}_meandiff"] = v
        for mode in MAIN_MODES:
            x = res[mt].xs(mode, level="mode")
            for bb in ("tapct", "spectre"):
                xb = x.xs(bb, level="backbone").dropna()
                macros[f"segmodepos_{bb}_{mt}_{mode}_n"] = int((xb > 0).sum())
                macros[f"segmodecells_{bb}_{mt}_{mode}_n"] = int(len(xb))
    fam("seggap_<method>_*, segdelta_<bb>_<baseline>_<method>_*, segdeltamode_*, segvslora_*", "SPECTRE minus TAP-CT Dice per method; method minus Lin/Dec per structure-mode pair (mean, % ahead, median, min, drops below -0.1 and the mean without them); method minus LoRA")
    sd_ = St[St["mode"].isin(MAIN_MODES)]
    bp = sd_.pivot_table(index=["cohort", "label", "mode", "method"], columns="backbone", values="dice").dropna()
    bp["gap"] = bp["spectre"] - bp["tapct"]
    for mt, g in bp.reset_index().groupby("method"):
        per_ds = g.groupby("cohort").gap.mean()
        macros.update({f"seggap_{mt}_meandiff": g.gap.mean(), f"seggap_{mt}_ahead_n": int((per_ds > 0).sum()), f"seggap_{mt}_datasets_n": len(per_ds),
                       f"seggap_{mt}_min_meandiff": per_ds.min(), f"seggap_{mt}_max_meandiff": per_ds.max()})
    spm = sd_.pivot_table(index=["backbone", "cohort", "label", "mode"], columns="method", values="dice")
    for bb, g in spm.groupby(level="backbone"):
        for base in ["lin", "dec"]:
            for mt in [c for c in g.columns if c != base]:
                x = (g[mt] - g[base]).dropna()
                if len(x):
                    k = f"segdelta_{bb}_{base}_{mt}"
                    macros.update({f"{k}_meandiff": x.mean(), f"{k}_better": (x > 0).mean() * 100, f"{k}_mediandiff": x.median(), f"{k}_mindiff": x.min(),
                                   f"{k}_drop_n": int((x < -0.1).sum()), f"{k}_nodrop_meandiff": x[x >= -0.1].mean(), f"{k}_n": len(x)})
                    for md, xm in x.groupby(level="mode"):
                        macros[f"segdeltamode_{bb}_{base}_{mt}_{md}_meandiff"] = xm.mean()
        for mt in ["dora", "lorag", "dec_then_lora", "fullft"]:
            x = (g[mt] - g["lora"]).dropna()
            if len(x):
                macros[f"segvslora_{bb}_{mt}_meandiff"] = x.mean()
                macros[f"segvslora_{bb}_{mt}_better"] = (x > 0).mean() * 100
    fam("clshead_*", "attentive minus linear probe (classification) per label-mode pair")
    x = (cp["attn"] - cp["lin"]).dropna()
    macros["clshead_attn_vs_lin_meandiff"] = x.mean()
    macros["clshead_attn_vs_lin_better"] = (x > 0).mean() * 100
    for bb, g in x.groupby(level="backbone"):
        macros[f"clshead_{bb}_attn_vs_lin_meandiff"] = g.mean()

    # --- convergence macros (as the draft defined them)
    fam("conv_*", "share of runs (main backbones, seed 42, two-stage methods excluded) whose selected checkpoint is the last validation (last_better) or one of the last two (late_better); conv_seg_<pertask|joint|jointlong|jointeightk>_runs_n = runs counted")
    cvm = cv[(cv.seed == 42) & cv.backbone.isin(MAIN_BACKBONES) & ~cv.method.str.contains("then") & ~cv.cohort.isin(EXCLUDED_COHORTS)]
    c_ = cvm[(cvm.task == "cls") & cvm["mode"].isin(MAIN_MODES) & cvm.method.isin(CLS_METHODS)]
    macros["conv_cls_last_better"] = c_.is_last.mean() * 100
    macros["conv_cls_late_better"] = c_.is_last_two.mean() * 100
    macros["conv_cls_runs_n"] = len(c_)
    s_ = cvm[(cvm.task == "seg") & cvm.method.isin(SEG_METHODS)]
    for tag, g in (("pertask", s_[s_["mode"] == "pertask"]), ("joint", s_[(s_["mode"] == "joint") & ~s_.cohort.isin(SEG_LONG_JOINT)]),
                   ("jointlong", s_[(s_["mode"] == "joint") & s_.cohort.isin(SEG_LONG_JOINT)]), ("jointeightk", s_[s_["mode"] == "joint_8k"])):
        macros[f"conv_seg_{tag}_last_better"] = g.is_last.mean() * 100
        macros[f"conv_seg_{tag}_late_better"] = g.is_last_two.mean() * 100
        macros[f"conv_seg_{tag}_runs_n"] = len(g)
    eight = s_[s_["mode"].isin(["pertask", "joint_8k"]) | ((s_["mode"] == "joint") & ~s_.cohort.isin(SEG_LONG_JOINT))]
    macros["conv_seg_eightk_last_better"] = eight.is_last.mean() * 100
    macros["conv_seg_eightk_late_better"] = eight.is_last_two.mean() * 100
    for task in ("cls", "seg"):
        g = cv[cv.task == task]
        macros[f"conv_{task}_all_last_better"] = g.is_last.mean() * 100
        macros[f"conv_{task}_all_late_better"] = g.is_last_two.mean() * 100
        macros[f"conv_{task}_all_runs_n"] = len(g)

    # --- average precision
    fam("ap_*", "LoRA minus Lin in held-out average precision per label-mode pair (pairs that trained); rare = held-out prevalence below 10%")
    apv = Ct_nd[Ct_nd["mode"].isin(MAIN_MODES) & Ct_nd.method.isin(["lora", "lin"])].pivot_table(index=["backbone", "cohort", "label", "mode"], columns="method", values=["auprc", "n", "n_pos"])
    a = pd.DataFrame({"diff": apv[("auprc", "lora")] - apv[("auprc", "lin")], "lin": apv[("auprc", "lin")], "prev": apv[("n_pos", "lin")] / apv[("n", "lin")]}).dropna()
    macros["ap_all_meandiff"] = a["diff"].mean()
    macros["ap_all_better"] = (a["diff"] > 0).mean() * 100
    for bb, g in a.groupby(level="backbone"):
        macros[f"ap_{bb}_meandiff"] = g["diff"].mean()
    r_ = a[a.prev < 0.10]
    macros.update({"ap_rare_meandiff": r_["diff"].mean(), "ap_rare_n": len(r_), "ap_rare_better": (r_["diff"] > 0).mean() * 100, "ap_rare_lin": r_.lin.mean(), "ap_rare_prev": r_.prev.mean()})

    # --- label-level intervals
    fam("ci_<key>_<bb|all>_{meandiff,lodiff,hidiff,lotwodiff,hitwodiff,pos_n,clusters_n,labels_n}", "label-level estimate: mean over labels, three-stage interval (lo/hi), two-stage interval (lotwo/hitwo), cohort-backbone clusters with a positive mean and their number, labels. Keys: the draft's names (clsloralin, clsattnftlora, clsfullftlin, clsjoint, segdeclin, segloradec, segloradecpt, segloradecjt, segfullftdec, segjoint) and the new keys without underscores (e.g. clsattnlin, clsloraglin, seglorageddec)")
    for r in ll.itertuples():
        k = f"ci_{OLD_CI_KEY.get(r.key, r.key.replace('_', ''))}_{r.backbone}"
        macros.update({f"{k}_meandiff": r.mean, f"{k}_lodiff": r.lo3, f"{k}_hidiff": r.hi3, f"{k}_lotwodiff": r.lo2, f"{k}_hitwodiff": r.hi2,
                       f"{k}_pos_n": r.clusters_pos, f"{k}_clusters_n": r.clusters_n, f"{k}_labels_n": r.n_labels})

    # --- VoCo
    fam("voco_*, vocorank_*, vocosegrank_*, vocovslin_*, vocopaired_*", "VoCo: run counts, ranks over label-mode slots with all methods, every method minus Lin at label level (per cohort too), DeLong summaries")
    inv_all = pd.read_csv(args.inventory)
    for tag, meths in (("cls", CLS_METHODS), ("seg", SEG_METHODS)):
        iv = inv_all[(inv_all.backbone == "voco") & (inv_all.task == tag) & inv_all.method.isin(meths) & inv_all.excluded.isna() & inv_all.aux.isna() & (inv_all.in_design == "yes")]
        macros[f"voco_runs_{tag}_n"] = int(len(iv))
        macros[f"voco_runs_{tag}_done_n"] = int((iv.status == "finished").sum())
    macros["voco_dnt_n"] = int(spread[(spread.backbone == "voco") & (spread.median_sd < SPREAD_MIN)].shape[0])  # below the threshold; the rule is not applied to VoCo
    for task, info, meths, pre in (("cls", info_cls, CLS_METHODS, "vocorank"), ("seg", info_seg, SEG_METHODS, "vocosegrank")):
        sel = info["slots"].index.get_level_values("backbone") == "voco"
        macros[f"{pre}_slots_n"] = int(sel.sum())
        for mt in meths:
            macros[f"{pre}_{mt}_mean"] = float(info["rk"].loc[sel, mt].mean()) if sel.sum() else np.nan
    vpiv = Sv[Sv["mode"].isin(MAIN_MODES)].pivot_table(index=["cohort", "label", "mode"], columns="method", values="dice")
    if len(vpiv):
        x = (vpiv["fullft"] - vpiv["dec"]).dropna()
        macros.update({"segdelta_voco_dec_fullft_better": (x > 0).mean() * 100, "segdelta_voco_dec_fullft_mediandiff": x.median(), "segdelta_voco_dec_fullft_mindiff": x.min(), "segdelta_voco_dec_fullft_n": len(x)})
    vc = Cv[Cv["mode"].isin(MAIN_MODES)].pivot_table(index=["cohort", "label", "mode"], columns="method", values="auroc")
    if len(vc):
        lab_ = vc.groupby(level=[0, 1]).mean()
        for mt in CLS_METHODS:
            if mt == "lin":
                continue
            x = (lab_[mt] - lab_["lin"]).dropna()
            cm_ = x.groupby(level=0).mean()
            macros.update({f"vocovslin_{mt}_meandiff": x.mean(), f"vocovslin_{mt}_pos_n": int((x > 0).sum()), f"vocovslin_{mt}_labels_n": len(x),
                           f"vocovslin_{mt}_cohorts_pos_n": int((cm_ > 0).sum()), f"vocovslin_{mt}_cohorts_n": len(cm_)})
            for ds, v in cm_.items():
                macros[f"vocovslin_{mt}_{ds}_meandiff"] = v
    pv = pcls[pcls.backbone == "voco"]
    for mt, base in (("lora", "lin"), ("attnft", "lora"), ("fullft", "lin"), ("lin_then_lorag", "lin")):
        x = pv[(pv.method == mt) & (pv.baseline == base)]
        if len(x):
            macros.update({f"vocopaired_{mt}_vs_{base}_n": len(x), f"vocopaired_{mt}_vs_{base}_better": (x["diff"] > 0).mean() * 100,
                           f"vocopaired_{mt}_vs_{base}_sigbetter": ((x["diff"] > 0) & (x.p_bh < 0.05)).mean() * 100, f"vocopaired_{mt}_vs_{base}_sigworse": ((x["diff"] < 0) & (x.p_bh < 0.05)).mean() * 100})

    # --- prose macros of the draft (write_macros)
    fam("jointdelta_*, jointlmdelta_*, jointcohort_*, jointpairs_n, jointneg_n", "joint minus per-task AUROC over (method, label) pairs: pooled, per backbone, per backbone and cohort (jointlm: 26000-step joint), per cohort and method; number of (backbone, cohort) pairs and how many are negative")
    d = Ct_nd
    pivm = d.pivot_table(index=["backbone", "cohort", "method", "label"], columns="mode", values="auroc")
    allb = pivm.dropna(subset=["pertask", "joint"])
    macros["jointdelta_all"] = float((allb["joint"] - allb["pertask"]).mean())
    macros["jointdelta_all_better"] = float(((allb["joint"] - allb["pertask"]) > 0).mean() * 100)
    macros["jointdelta_all_n"] = int(len(allb))
    for bb, g in pivm.groupby(level="backbone"):
        both = g.dropna(subset=["pertask", "joint"])
        macros[f"jointdelta_{bb}"] = float((both["joint"] - both["pertask"]).mean())
        for ds, g2 in both.groupby(level="cohort"):
            macros[f"jointdelta_{bb}_{ds}"] = float((g2["joint"] - g2["pertask"]).mean())
        if "joint_lm" in g.columns:
            lm = g.dropna(subset=["pertask", "joint_lm"])
            for ds, g2 in lm.groupby(level="cohort"):
                macros[f"jointlmdelta_{bb}_{ds}"] = float((g2["joint_lm"] - g2["pertask"]).mean())
    jd = {k: v for k, v in macros.items() if k.startswith("jointdelta_") and len(k.split("_")) > 2 and k.split("_")[1] in MAIN_BACKBONES}
    macros["jointpairs_n"] = len(jd)
    macros["jointneg_n"] = sum(v < 0 for v in jd.values())
    macros["jointdelta_min"] = float(min(jd.values()))
    macros["jointdelta_max"] = float(max(jd.values()))
    for (ds, mt), g in allb.reset_index().groupby(["cohort", "method"]):
        macros[f"jointcohort_{ds}_{mt}_meandiff"] = float((g["joint"] - g["pertask"]).mean())
    fam("sigcohort_<cohort>_*, sigbysize_<small|mid|large>_*", "LoRA vs Lin DeLong rows (pairs that trained) by cohort (% significant ahead, % ahead, mean difference, median held-out n) and by held-out size (<100, 100-400, >=400)")
    pt = P_nd[(P_nd.baseline == "lin") & (P_nd.method == "lora")]
    for ds, g in pt.groupby("cohort"):
        macros.update({f"sigcohort_{ds}_sigbetter": ((g["diff"] > 0) & (g.p_bh < 0.05)).mean() * 100, f"sigcohort_{ds}_better": (g["diff"] > 0).mean() * 100,
                       f"sigcohort_{ds}_meandiff": g["diff"].mean(), f"sigcohort_{ds}_ntest": int(g.n.median())})
    for tag, lo, hi in (("small", 0, 100), ("mid", 100, 400), ("large", 400, 10**9)):
        g = pt[(pt.n >= lo) & (pt.n < hi)]
        macros[f"sigbysize_{tag}_n"] = len(g)
        macros[f"sigbysize_{tag}_sig"] = ((g["diff"] > 0) & (g.p_bh < 0.05)).mean() * 100 if len(g) else np.nan
        macros[f"sigbysize_{tag}_meandiff"] = g["diff"].mean() if len(g) else np.nan
    fam("bestrows_*", "how often each method has the highest cohort mean of a (cohort, backbone, mode) row (pairs that trained; modes pertask, joint, joint_lm)")
    ladder = d.groupby(["cohort", "backbone", "mode", "method"]).auroc.mean().unstack()
    best = ladder.idxmax(axis=1)
    macros["bestrows_total"] = int(len(best))
    for mt in ladder.columns:
        macros[f"bestrows_{mt}"] = int((best == mt).sum())
    macros["bestrows_anylora"] = int(best.isin(["lora", "lorag", "babmil", "lin_then_lorag", "attn_then_lorag"]).sum())
    fam("gainbin_<low|mid|high|top>_*, gainbinattn_*, gaincorr", "LoRA (attn) minus Lin by the linear probe's selection-split AUROC bin (<=0.6, 0.6-0.7, 0.7-0.8, >0.8); Spearman correlation of probe AUROC and LoRA gain")
    probe_val = {(r.backbone, r.cohort, r["mode"], r.label): r.auroc for _, r in C[(C.split == "val") & (C.method == "lin")].iterrows()}
    for meth, pre in (("lora", "gainbin"), ("attn", "gainbinattn")):
        g0 = P_nd[(P_nd.baseline == "lin") & (P_nd.method == meth)].copy()
        g0["probe_val"] = [probe_val.get(k, np.nan) for k in zip(g0.backbone, g0.cohort, g0["mode"], g0.label)]
        g0 = g0.dropna(subset=["probe_val"])
        for lo, hi, tag in ((0.0, 0.6, "low"), (0.6, 0.7, "mid"), (0.7, 0.8, "high"), (0.8, 1.01, "top")):
            g = g0[(g0.probe_val > lo) & (g0.probe_val <= hi)]
            macros[f"{pre}_{tag}_n"] = len(g)
            macros[f"{pre}_{tag}_meandiff"] = g["diff"].mean() if len(g) else np.nan
            macros[f"{pre}_{tag}_better"] = (g["diff"] > 0).mean() * 100 if len(g) else np.nan
        if meth == "lora":
            macros["gaincorr"] = float(spearmanr(g0.probe_val, g0["diff"]).statistic)
    fam("bbdiff_*", "SPECTRE minus TAP-CT held-out AUROC per shared cohort and method (modes pooled; pairs that trained)")
    sh = d[d.backbone.isin(["tapct", "spectre"]) & d["mode"].isin(MAIN_MODES)]
    bk = sh.groupby(["cohort", "method", "backbone"]).auroc.mean().unstack().dropna()
    for (ds, mt), row in bk.iterrows():
        macros[f"bbdiff_{ds}_{mt}_meandiff"] = float(row["spectre"] - row["tapct"])
    for mt in [x for x in ["lin", "attn", "lora", "babmil", "fullft"] if x in bk.index.get_level_values("method")]:
        sub = bk.xs(mt, level="method")
        diffs = sub["spectre"] - sub["tapct"]
        macros.update({f"bbdiff_min_{mt}_meandiff": diffs.min(), f"bbdiff_max_{mt}_meandiff": diffs.max(), f"bbdiff_{mt}_spectre_ahead_n": int((diffs > 0).sum()), f"bbdiff_{mt}_cohorts_n": len(diffs)})
    fam("select_*, valtest_<cohort>_spearman", "per slot (cohort, backbone, mode, label) with >= 4 methods: is the method with the best selection-split AUROC also best held-out (hit, %), its held-out deficit (regret), the deficit of always choosing LoRA, and the Spearman correlation of selection and held-out AUROC per cohort")
    vt = drop_dnt(C[C.backbone.isin(MAIN_BACKBONES) & C["mode"].isin(MAIN_MODES)]).pivot_table(index=["cohort", "backbone", "mode", "method", "label"], columns="split", values="auroc").dropna().reset_index()
    hits = []
    for key, g in vt.groupby(["cohort", "backbone", "mode", "label"]):
        if len(g) < 4:
            continue
        vb = g.loc[g.val.idxmax()]
        lora = g[g.method == "lora"]
        hits.append((key[0], vb.method == g.loc[g.test.idxmax(), "method"], g.test.max() - vb.test, (g.test.max() - lora.test.values[0]) if len(lora) else np.nan))
    hdf = pd.DataFrame(hits, columns=["cohort", "hit", "regret", "lora_regret"])
    macros.update({"select_slots_n": len(hdf), "select_hit": hdf.hit.mean() * 100, "select_regret": hdf.regret.mean(), "select_lora_regret": hdf.lora_regret.mean(),
                   "select_gain": hdf.lora_regret.mean() - hdf.regret.mean()})
    for ds, g in hdf.groupby("cohort"):
        macros[f"select_{ds}_hit"] = g.hit.mean() * 100
        macros[f"select_{ds}_regret"] = g.regret.mean()
    for ds, g in vt.groupby("cohort"):
        macros[f"valtest_{ds}_spearman"] = float(spearmanr(g.val, g.test).statistic)
    fam("early_<bb>_<method>_better, early_<bb>_<method>_median_n", "% of runs selected at <= 2000 iterations and the median selected iteration (classification, main backbones, seed 42, every mode)")
    e_ = runs[(runs.task == "cls") & (runs.seed == 42) & runs.backbone.isin(MAIN_BACKBONES) & runs.method.isin(CLS_METHODS) & ~runs.cohort.isin(EXCLUDED_COHORTS)]
    for (bb, mt), g in e_.groupby(["backbone", "method"]):
        macros[f"early_{bb}_{mt}_better"] = float((g.best_iter <= 2000).mean() * 100)
        macros[f"early_{bb}_{mt}_median_n"] = int(g.best_iter.median())
    fam("cohortmode_<bb>_<cohort>_<mode>_<method>_{meandiff,n}, cohortall_<bb>_<cohort>_<method>_meandiff, worstcohort_*", "mean gain over Lin (DeLong rows, pairs that trained) per backbone, cohort, mode and method; per backbone, cohort and method; the worst (backbone, cohort) mean per method and over every method but Full FT")
    allp = P_nd[P_nd.baseline == "lin"]
    for (bb, ds, mode, mt), g in allp.groupby(["backbone", "cohort", "mode", "method"]):
        macros[f"cohortmode_{bb}_{ds}_{mode}_{mt}_meandiff"] = g["diff"].mean()
        macros[f"cohortmode_{bb}_{ds}_{mode}_{mt}_n"] = len(g)
    for (bb, ds, mt), g in allp.groupby(["backbone", "cohort", "method"]):
        macros[f"cohortall_{bb}_{ds}_{mt}_meandiff"] = g["diff"].mean()
    cmeans = allp.groupby(["backbone", "cohort", "method"])["diff"].mean()
    for mt in CLS_METHODS[1:]:
        if mt in cmeans.index.get_level_values("method"):
            macros[f"worstcohort_{mt}_meandiff"] = float(cmeans.xs(mt, level="method").min())
    macros["worstcohort_other_meandiff"] = float(min(macros[f"worstcohort_{mt}_meandiff"] for mt in CLS_METHODS[1:] if mt != "fullft" and f"worstcohort_{mt}_meandiff" in macros))
    fam("secondorder_max", "largest |method minus LoRA| among the head and schedule variants (not Full FT, not Attn), and |Attn minus Lin|")
    second = [abs(v) for k, v in macros.items() if k.startswith("vslora_") and k.endswith("_meandiff") and not k.endswith("_fullft_meandiff") and "_attn_meandiff" not in k and k.split("_")[1] in MAIN_BACKBONES]
    second += [abs(macros[k]) for k in macros if k.startswith("paired_") and k.endswith("_lin_attn_meandiff")]
    macros["secondorder_max"] = float(max(second))

    # --- revision macros
    fam("seeds<cls|seg>_<a>_<b>_<bb|all>_{pos_n,pairs_n,extrapos_n,extrapairs_n}, seeds<task>_<method>_{sdmedian,sdmax,cells_n}, seeds<task>_runs_n", "seed repeats of the joint models: (cohort, seed) pairs with A > B (all seeds; seeds 43/44 only), SD of the cohort mean across seeds (median and max over cells with >= 2 seeds), number of extra-seed runs")
    for task in ("cls", "seg"):
        ss = seeds_summary[seeds_summary.task == task]
        for r in ss[ss.comparison != ""].itertuples():
            a_, b_ = r.comparison.split(">")
            macros.update({f"seeds{task}_{a_}_{b_}_{r.backbone}_pos_n": r.pairs_pos, f"seeds{task}_{a_}_{b_}_{r.backbone}_pairs_n": r.pairs_n,
                           f"seeds{task}_{a_}_{b_}_{r.backbone}_extrapos_n": r.extra_pairs_pos, f"seeds{task}_{a_}_{b_}_{r.backbone}_extrapairs_n": r.extra_pairs_n})
        for r in ss[(ss.comparison == "") & (ss.backbone == "all")].itertuples():
            macros.update({f"seeds{task}_{r.method}_sdmedian": r.sd_median, f"seeds{task}_{r.method}_sdmax": r.sd_max, f"seeds{task}_{r.method}_cells_n": r.n_cells})
        macros[f"seeds{task}_runs_n"] = int(len(runs[(runs.task == task) & (runs.seed != 42)]))
    fam("fix_<bb>_*, fixsel_<bb>_*", "full fine-tuning settings (lower LR, layer decay, accumulation + clipping): mean selection-split AUROC, cohorts whose run trained (spread >= 0.01), the setting chosen per backbone (highest selection AUROC) and its label-level difference to Lin and LoRA")
    rc = mcls[(mcls["mode"] == "joint") & (mcls.seed == 42) & mcls.backbone.isin(MAIN_BACKBONES)]
    test_r = rc[(rc.split == "test") & rc.method.isin(FIXES)]
    val_r = rc[(rc.split == "val") & rc.method.isin(FIXES)]
    lab = rc[(rc.split == "test") & rc.method.isin(["lin", "lora", "fullft", *FIXES])].pivot_table(index=["backbone", "cohort", "label"], columns="method", values="auroc")
    spread_run = test_r.groupby(["backbone", "cohort", "method"]).pred_sd.median()
    vmean = val_r.groupby(["backbone", "method"]).auroc.mean()
    vcoh = val_r.groupby(["backbone", "method"]).cohort.nunique()
    chosen = {}
    for bb in MAIN_BACKBONES:
        n_coh = rc[(rc.backbone == bb) & (rc.method == "lin")].cohort.nunique()
        ready = [f for f in FIXES if vcoh.get((bb, f), 0) == n_coh]
        if len(ready) == len(FIXES):
            chosen[bb] = max(ready, key=lambda f: vmean[(bb, f)])
            macros[f"fix_{bb}_chosen"] = FIX_TEX[chosen[bb]]
        for f in FIXES:
            macros[f"fix_{bb}_{f}_val"] = float(vmean.get((bb, f), np.nan))
            ks = [k for k in spread_run.index if k[0] == bb and k[2] == f]
            macros[f"fix_{bb}_{f}_trained_n"] = int(sum(spread_run[k] >= SPREAD_MIN for k in ks))
            macros[f"fix_{bb}_{f}_cohorts_n"] = len(ks)
    for bb, f in chosen.items():
        for ref in ("lin", "lora"):
            dd = (lab[f] - lab[ref]).dropna().xs(bb, level="backbone")
            per = dd.groupby(level="cohort").mean()
            macros.update({f"fixsel_{bb}_{ref}_meandiff": dd.mean(), f"fixsel_{bb}_{ref}_pos_n": int((per > 0).sum()), f"fixsel_{bb}_{ref}_cohorts_n": len(per)})
        macros[f"fixsel_{bb}_mean"] = float(lab[f].xs(bb, level="backbone").groupby(level="cohort").mean().mean())
    fam("ctrl_<bb>_<a>_<b>_{meandiff,pos_n,cohorts_n}", "controls: MLP-only FT vs Attn-only FT and LoRA; rsLoRA at LR/4 vs LoRA and rsLoRA (label-level differences of the joint models; cohorts with a positive mean)")
    labc = rc[(rc.split == "test") & rc.method.isin(["lora", "rslora", "attnft", "mlpft", "rslora_lowlr"])].pivot_table(index=["backbone", "cohort", "label"], columns="method", values="auroc")
    for a_, b_ in [("mlpft", "attnft"), ("mlpft", "lora"), ("rslora_lowlr", "lora"), ("rslora_lowlr", "rslora")]:
        if a_ not in labc or b_ not in labc:
            continue
        dd_all = (labc[a_] - labc[b_]).dropna()
        for bb in MAIN_BACKBONES:
            dd = dd_all[dd_all.index.get_level_values("backbone") == bb]
            if dd.empty:
                continue
            per = dd.groupby(level="cohort").mean()
            macros.update({f"ctrl_{bb}_{a_}_{b_}_meandiff": dd.mean(), f"ctrl_{bb}_{a_}_{b_}_pos_n": int((per > 0).sum()), f"ctrl_{bb}_{a_}_{b_}_cohorts_n": len(per)})
    fam("cm_<dataset>_<bb>_<method>_{jointdiff,pertaskdiff,pos_n,struct_n}", "compute-matched per-structure models (1600 steps): joint minus matched and 8000-step per-structure minus matched, structures with the joint model ahead")
    cmr = mseg[(mseg["mode"] == "pertask_cm") & (mseg.split == "test")]
    ref = St[St["mode"].isin(MAIN_MODES) & St.method.isin(SEG_SEED_METHODS)]
    for ds in sorted(SEG_LONG_JOINT):
        for bb in ("tapct", "spectre"):
            for mt in SEG_SEED_METHODS:
                g = cmr[(cmr.backbone == bb) & (cmr.cohort == ds) & (cmr.method == mt)].set_index("label").dice
                if g.empty:
                    continue
                rr = ref[(ref.backbone == bb) & (ref.cohort == ds) & (ref.method == mt)].pivot_table(index="label", columns="mode", values="dice").reindex(g.index)
                dj = rr["joint"] - g
                k = f"cm_{ds}_{bb}_{mt}"
                macros.update({f"{k}_jointdiff": dj.mean(), f"{k}_pertaskdiff": (rr["pertask"] - g).mean(), f"{k}_pos_n": int((dj > 0).sum()), f"{k}_struct_n": int(dj.notna().sum()),
                               f"{k}_matched_mean": g.mean()})
    fam("scratch_<dataset>_<bb>_{meandiff,mean}", "from-scratch joint models (TAP-CT architecture, random init) minus the Full FT joint model, mean over structures")
    scr = mseg[(mseg.method == "scratch") & (mseg.split == "test")]
    for (ds, bb), g in scr.groupby(["cohort", "backbone"]):
        f_ = St[(St.backbone == bb) & (St.cohort == ds) & (St.method == "fullft") & (St["mode"] == "joint")].set_index("label").dice
        x = (g.set_index("label").dice - f_).dropna()
        macros[f"scratch_{ds}_{bb}_meandiff"] = x.mean()
        macros[f"scratch_{ds}_{bb}_mean"] = g.dice.mean()

    # --- counts (run, label, structure and case counts)
    fam("runs_<task>_<bb>_n, runs_<task>_n, labels_<bb>_n, structs_<bb>_n, labels_cls_n, structs_seg_n, delong_tests_n, segpaired_tests_n, numRunsCls, numLabels, numDelongTests, numSegRunsN", "counts: runs used per task and backbone (main design, seed 42, main methods, no breast MRI cohort), labels / structures per backbone, DeLong and paired-bootstrap comparisons; the draft's names numRunsCls (main cls runs without VoCo), numLabels (distinct cohort-label pairs), numDelongTests (vs Lin and Full FT, main backbones, pairs that trained), numSegRunsN")
    mainruns = runs[(runs.seed == 42) & ~runs.cohort.isin(EXCLUDED_COHORTS) & (((runs.task == "cls") & runs.method.isin(CLS_METHODS) & runs["mode"].isin(["pertask", "joint", "joint_lm"]))
                                                                                  | ((runs.task == "seg") & runs.method.isin(SEG_METHODS) & runs["mode"].isin(["pertask", "joint", "joint_8k"])))]
    for (task, bb), g in mainruns.groupby(["task", "backbone"]):
        macros[f"runs_{task}_{bb}_n"] = len(g)
    for task, g in mainruns.groupby("task"):
        macros[f"runs_{task}_n"] = len(g)
        macros[f"runs_{task}_main_n"] = int(g.backbone.isin(MAIN_BACKBONES).sum())
    macros["runs_all_n"] = len(runs)
    macros["runs_used_n"] = len(mainruns)
    for bb, g in Ct.groupby("backbone"):
        macros[f"labels_{bb}_n"] = int(g[["cohort", "label"]].drop_duplicates().shape[0])
    for bb, g in Cv.groupby("backbone"):
        macros[f"labels_{bb}_n"] = int(g[["cohort", "label"]].drop_duplicates().shape[0])
    for bb, g in pd.concat([St, Sv]).groupby("backbone"):
        macros[f"structs_{bb}_n"] = int(g[["cohort", "label"]].drop_duplicates().shape[0])
    macros["labels_cls_n"] = int(Ct[["cohort", "label"]].drop_duplicates().shape[0])
    macros["structs_seg_n"] = int(St[["cohort", "label"]].drop_duplicates().shape[0])
    macros["delong_tests_n"] = len(pcls)
    macros["segpaired_tests_n"] = len(pseg)
    special = {"numRunsCls": int(len(mainruns[(mainruns.task == "cls") & mainruns.backbone.isin(MAIN_BACKBONES)])),
               "numLabels": macros["labels_cls_n"],
               "numDelongTests": int(len(P_nd[P_nd.baseline.isin(["lin", "fullft"])])),
               "numSegRunsN": macros["seg_runs_n"]}
    fam("cohort_<cohort>_<val|test>, cohort_<cohort>_<split>_ids_n, cases_<dataset>_<split>_n", "held-out and selection cases per cohort from the per-case files (rows of the joint Lin run; FDG rows are studies, ids are patients; CVPR26 split into amos / covid / luna by label source); segmentation cases per dataset and split")
    rows_by, ids_by = defaultdict(int), defaultdict(set)  # (cohort, split) -> most rows of any label / union of case ids
    for r in runs[(runs.task == "cls") & (runs.seed == 42) & (runs["mode"] == "joint") & (runs.method == "lin") & (runs.backbone == "tapct") & ~runs.cohort.isin(EXCLUDED_COHORTS)].itertuples():
        for lab, splits in store[r.run_id].items():
            if lab == "__warn__":
                continue
            coh = {"amos": "amos", "covid_ct": "covid", "luna25": "luna"}[lab.split(":")[0]] if r.cohort == "amos_covid_luna" else COHORT_SHORT.get(r.cohort, r.cohort)
            for sp in ("val", "test"):
                if sp in splits:
                    rows_by[(coh, sp)] = max(rows_by[(coh, sp)], len(splits[sp][0]))
                    ids_by[(coh, sp)] |= set(splits[sp][0])
    for (coh, sp) in sorted(ids_by):
        macros[f"cohort_{coh}_{sp}"] = len(ids_by[(coh, sp)]) if coh != "fdg" else rows_by[(coh, sp)]  # FDG: studies
        macros[f"cohort_{coh}_{sp}_ids_n"] = len(ids_by[(coh, sp)])
    macros["cohort_fdg_patients_test"] = len(ids_by[("fdg", "test")])
    te = [macros[f"cohort_{coh}_test"] for (coh, sp) in ids_by if sp == "test"]
    macros["cohort_min_test"], macros["cohort_max_test"] = int(min(te)), int(max(te))
    for r in runs[(runs.task == "seg") & (runs.seed == 42) & (runs["mode"] == "joint") & (runs.method == "dec") & (runs.backbone == "tapct")].itertuples():
        for lab, splits in store[r.run_id].items():
            if lab != "__warn__":
                for sp in ("val", "test"):
                    if sp in splits:
                        macros[f"cases_{r.cohort}_{sp}_n"] = int(len(set(splits[sp][0])))
                break
    for ds, n in SEG_N_STRUCT.items():
        macros[f"structs_{ds}_n"] = n
    for coh in CLS_TASKS:
        if coh not in EXCLUDED_COHORTS:
            macros[f"labels_{coh}_n"] = int(Ct[Ct.cohort == coh].label.nunique())

    # ================================================================== verification
    check("\n== Verification ==")
    rng_check = np.random.default_rng(0)
    rec_rows = []
    for r in runs.itertuples():
        data = store[r.run_id]
        if r.task == "cls":
            vals = []
            for lab, splits in data.items():
                if lab == "__warn__" or "val" not in splits:
                    continue
                y, p = splits["val"][1], splits["val"][2]
                if len(y) >= 2 and np.isfinite(p).all() and len(np.unique(y)) == 2:
                    vals.append(auroc1(y, p))
            rec = float(np.mean(vals)) if vals else 0.0
        else:
            vals = [float(s["val"][1][s["val"][2]].mean()) for lab, s in data.items() if lab != "__warn__" and "val" in s and s["val"][2].any()]
            rec = float(np.mean(vals)) if vals else float("nan")
        it, logged = read_best(args.results / r.path / "checkpoints" / "best" / f"best_val_metric_val_{'auroc' if r.task == 'cls' else 'dice'}.txt")
        rec_rows.append(dict(task=r.task, rel=r.rel, best_iter=int(r.best_iter), logged=logged, recomputed=rec, absdiff=abs(rec - logged), iter_ok=it == int(r.best_iter)))
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(out / "selection_check.csv", index=False, float_format="%.6g")
    for task in ("cls", "seg"):
        g = rec_df[rec_df.task == task]
        pick = g.iloc[rng_check.choice(len(g), 5, replace=False)]
        check(f"(a) {task}: 5 random runs, recomputed selection-split macro metric vs logged best value:")
        for x in pick.itertuples():
            check(f"    {x.rel}: iter {x.best_iter}, logged {x.logged:.6f}, recomputed {x.recomputed:.6f}, |diff| {x.absdiff:.2e} {'OK' if x.absdiff <= 2e-3 else 'MISMATCH'}")
        check(f"    all {len(g)} {task} runs: max |diff| {g.absdiff.max():.2e}; runs above 2e-3: {int((g.absdiff > 2e-3).sum())}; best-iteration file mismatch: {int((~g.iter_ok).sum())}")
        for x in g[g.absdiff > 2e-3].itertuples():
            check(f"      above tolerance: {x.rel}: logged {x.logged:.6f}, recomputed {x.recomputed:.6f}")
    single = {r.rel for r in runs[runs.task == "seg"].itertuples() if len(list((args.results / r.path / "val_case_metrics").glob("iter_*"))) == 1}
    sg = rec_df[rec_df.rel.isin(single)]
    macros.update({"segrecomp_runs_n": int(len(sg)), "segrecomp_maxdiff": float(sg.absdiff.max()), "segrecomp_mediandiff": float(sg.absdiff.median())})
    for task in ("cls", "seg"):
        g = rec_df[rec_df.task == task]
        macros.update({f"selcheck_{task}_runs_n": int(len(g)), f"selcheck_{task}_maxdiff": float(g.absdiff.max()), f"selcheck_{task}_above_n": int((g.absdiff > 2e-3).sum())})
    fam("segrecomp_*, selcheck_<task>_*", "agreement of the recomputed selection-split macro metric with the logged best value: segrecomp over the segmentation runs whose per-case file was re-inferred at the selected checkpoint (single per-case folder), selcheck over every run used")
    check("(b) labels / structures per backbone (held-out rows of the main runs):")
    for bb in ALL_BACKBONES:
        nl = macros.get(f"labels_{bb}_n", 0)
        ns = macros.get(f"structs_{bb}_n", 0)
        per_coh = Ct[Ct.backbone == bb].groupby("cohort").label.nunique().to_dict() if bb in MAIN_BACKBONES else Cv.groupby("cohort").label.nunique().to_dict()
        check(f"    {bb}: {nl} classification labels {per_coh}; {ns} segmentation structures")
    check(f"    pooled main backbones: {macros['labels_cls_n']} distinct cohort-label pairs; label-level n per backbone: " +
          str({r.backbone: r.n_labels for r in ll[ll.key == 'cls_lora_lin'].itertuples()}))
    # mapping check of joint classification runs against the logged per-label validation AUROC
    mism, nolog, checked = 0, 0, 0
    for r in runs[(runs.task == "cls") & (runs.label_run == "joint")].itertuples():
        mj = args.results / r.path / "logs" / "metrics.json"
        if not mj.is_file():
            nolog += 1
            continue
        logged = {}
        with mj.open() as f:
            for line in f:
                if f'"iteration": {int(r.best_iter)},' in line or f'"iteration": {int(r.best_iter)}}}' in line:
                    try:
                        import json
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if row.get("iteration") == int(r.best_iter):
                        logged.update({k: v for k, v in row.items() if isinstance(v, (int, float))})
        for lab, splits in store[r.run_id].items():
            if lab == "__warn__" or "val" not in splits:
                continue
            key = f"val/metric/val_auroc/{re.sub(r'[^0-9A-Za-z_]+', '_', lab)}"
            if key not in logged:
                continue
            y, p = splits["val"][1], splits["val"][2]
            if len(np.unique(y)) < 2:
                continue
            checked += 1
            if abs(auroc1(y, p) - logged[key]) > 2e-3:
                mism += 1
                notes.append(f"mapping check {r.rel} {lab}: recomputed val AUROC {auroc1(y, p):.4f} vs logged {logged[key]:.4f}")
    check(f"(d) task_index -> label mapping of joint classification runs: {checked} labels compared with the logged per-label validation AUROC, {mism} mismatches > 2e-3, {nolog} runs without metrics.json")

    # ================================================================== numbers.tex
    lines = list(pooled_lines)
    skipped_nan = []
    for k in sorted(macros):
        s = fmt_macro(k, macros[k])
        if s is None:
            skipped_nan.append(k)
            continue
        lines.append(f"\\newcommand{{\\{macro_name(k)}}}{{{s}}}")
    for name, v in special.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    log(f"numbers.tex: {len(lines)} macros ({len(skipped_nan)} undefined values left out: {skipped_nan[:10]}{'...' if len(skipped_nan) > 10 else ''})")

    # NAMING.md
    nm = ["# Macro names in metrics/out/numbers.tex (written by recompute.py)", "",
          "Every macro is `\\newcommand{\\<name>}{<value>}`. Names follow the draft pipeline (`make_tables.macro_name`): the underscore key is title-cased, "
          "non-alphanumerics are dropped, the prefix `num` is added and digits are replaced letter by letter with `str.maketrans('0123456789', 'ZABCDEFGHI')` "
          "(`pillar0` -> `PillarZ`, `rsna2019_ich` -> `RsnaBZAIIch`, `joint_8k` -> `JointHK`). Exceptions without the `num` prefix: `\\pooled<bb><method>` (method without underscores) "
          "and the draft's `\\numRunsCls`, `\\numLabels`, `\\numDelongTests`, `\\numSegRunsN`.", "",
          "Formatting (from the key): `*meandiff*`, `*delta*` and keys ending in `diff` are signed with three decimals (`+0.034`; a value that rounds to zero is `0.000`); "
          "keys containing `better` or `worse`, ending in `_n`, `_sig`, `_ntest`, `_hit`, or starting with `cohort_` / `bestrows_` are integers (percentages rounded to whole numbers); "
          "`rank_*_mean`, `segrank_*_mean`, `vocorank_*_mean`, `vocosegrank_*_mean`, `*_spearman` and `gaincorr` have two decimals; `segrecomp_*diff` four; everything else three. "
          "`bestarm_*`, `segbestarm_*` and `fix_<bb>_chosen` are method names (LaTeX). Macros whose value is undefined (no runs) are not written.", "",
          "Backbone tokens: tapct, spectre, pillar0, voco, all (= pooled over tapct, spectre, pillar0). Cohort tokens: amos_covid_luna, fdg, rsna2022, rsna2023, rsna2019_ich; "
          "datasets amos22, kits23, lits, msd_pancreas, totalseg. Mode tokens: pertask, joint, joint_lm (26 000-step classification joint), joint_8k / jointeightk "
          "(8000-step joint of AMOS22 / TotalSegmentator), pertask_cm.", "", "## Families", ""]
    for prefix, text in doc.items():
        nm.append(f"- `{prefix}`: {text}")
    nm += ["", "## All macro names", ""]
    for ln in lines:
        mm = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", ln)
        nm.append(f"- `\\{mm.group(1)}` = `{mm.group(2)}`")
    (HERE / "NAMING.md").write_text("\n".join(nm) + "\n")

    # ================================================================== changes vs draft
    changes = []
    if args.draft_numbers.is_file():
        old = {}
        for m_ in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", args.draft_numbers.read_text()):
            old[m_.group(1)] = m_.group(2)
        new = {}
        for ln in lines:
            m_ = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", ln)
            new[m_.group(1)] = m_.group(2)

        def num(s: str) -> float | None:
            s = s.strip().replace("$", "")
            try:
                return float(s.replace("+", "")) if re.fullmatch(r"[+-]?\d+(\.\d+)?", s) else None
            except ValueError:
                return None

        for name in sorted(set(old) & set(new)):
            o, n_ = num(old[name]), num(new[name])
            if o is None or n_ is None:
                changes.append(dict(macro=name, old_value=old[name], new_value=new[name], abs_change=np.nan, flag="non-numeric" if old[name] != new[name] else ""))
                continue
            ch = abs(n_ - o)
            is_int = re.fullmatch(r"[+-]?\d+", old[name].strip()) is not None
            changes.append(dict(macro=name, old_value=o, new_value=n_, abs_change=ch, flag="changed" if (ch > 0.0005 or (is_int and ch >= 1)) else ""))
        ch_df = pd.DataFrame(changes).sort_values(["abs_change", "macro"], ascending=[False, True], na_position="last")
        ch_df.to_csv(out / "changes_vs_draft.csv", index=False, float_format="%.6g")
        only_old = sorted(set(old) - set(new))
        only_new = sorted(set(new) - set(old))
        (out / "macros_only_in_draft.txt").write_text("\n".join(only_old) + "\n")
        (out / "macros_only_in_recompute.txt").write_text("\n".join(only_new) + "\n")
        check(f"changes_vs_draft: {len(ch_df)} shared macros, {int((ch_df.flag == 'changed').sum())} changed, {len(only_old)} only in the draft, {len(only_new)} new")
    (out / "notes.txt").write_text("\n".join(notes) + "\n")

    # ================================================================== summary
    check("\n== Summary ==")
    check("runs used per task/backbone (all finished design runs incl. seeds, revision, compute-matched):")
    check(runs.groupby(["task", "backbone"]).size().to_string())
    check("main-design runs (seed 42, main methods):")
    check(mainruns.groupby(["task", "backbone"]).size().to_string())
    check("rows per output file:")
    for f in sorted(out.glob("*.csv")):
        n_ = sum(1 for _ in f.open()) - 1
        check(f"    {f.name}: {n_}")
    check(f"    numbers.tex: {len(lines)} macros")
    check(f"notes (skipped labels / pairs): {len(notes)} -> notes.txt")
    if changes:
        top = ch_df[ch_df.abs_change.notna()].head(20)
        check("20 largest changes vs the draft:")
        for r in top.itertuples():
            check(f"    {r.macro}: {r.old_value} -> {r.new_value} (|d| = {r.abs_change:.4g})")
    (out / "checks.txt").write_text("\n".join(checks) + "\n")  # written before the timing line, so out/ is byte-identical across runs
    log(f"total time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
