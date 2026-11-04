#!/usr/bin/env python3
"""Table generator of the MIDL paper (v2): reads metrics/out/*.csv (written by metrics/recompute.py, columns in
metrics/SCHEMA.md) and writes LaTeX tabulars to paper/tables/*.tex.  Every number in a table comes from the CSVs;
nothing is computed from raw run files here.  Conventions: booktabs, no vertical rules, \\addlinespace between
groups, 3 decimals for AUROC / Dice and their differences, 2 for ranks, whole percentages.

Document-side macros used: \\mci{mean}{lo}{hi} (mean with the interval stacked below) and the colour `modebest`
(\\cellcolor) for the better of the per-task / joint rows.  Marks: $^\\dagger$ = backbone-method pair that did not
train (spread.csv did_not_train); such cells are shown plain, never bolded or shaded, and leave ranks / averages.

Usage: python make_tables.py [--check]   (--check compiles every table in a tiny standalone document with tectonic)

The tables in the previous draft's layouts and file names are written by paper_tables.py, called at the end of main().
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
OUT_CSV = V2 / "metrics" / "out"
# VoCo is quarantined (2026-11-02, ../quarantine_voco_20261102/): the tables leave out the VoCo backbone (VoCo-only tables skipped) unless
# WITH_VOCO=1, which needs the VoCo runs restored in the inventory. Written to ../paper/tables (PAPER_DIR overrides).
NO_VOCO = os.environ.get("WITH_VOCO", "") in ("", "0")
VOCO = [] if NO_VOCO else ["voco"]
OUT_TEX = Path(os.environ.get("PAPER_DIR") or (V2 / "paper")) / "tables"
TECTONIC = Path("/home/g.yiasemis/miniconda3/envs/tex/bin/tectonic")

# ----------------------------------------------------------------------------------------------- names and orders
METHODS = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
METHOD_TEX = {
    "lin": "Lin", "attn": "Attn", "lora": "LoRA", "lorag": "LoRA$_{\\text{late}}$", "babmil": "LoRA+ABMIL", "dora": "DoRA",
    "rslora": "rsLoRA", "lin_then_lorag": "Lin$\\to$LoRA", "attn_then_lorag": "Attn$\\to$LoRA", "attnft": "Attn-only FT",
    "fullft": "Full FT",
}
# two-line headers for the wide per-cohort tables
METHOD_STACK = {
    "lorag": ("LoRA$_{\\text{late}}$", ""), "babmil": ("LoRA+", "ABMIL"), "lin_then_lorag": ("Lin$\\to$", "LoRA"),
    "attn_then_lorag": ("Attn$\\to$", "LoRA"), "attnft": ("Attn-only", "FT"), "fullft": ("Full", "FT"),
    "dec_then_lora": ("Dec$\\to$", "LoRA"),
}
SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
SEG_METHOD_TEX = {
    "lin": "Lin", "dec": "Dec", "lora": "LoRA", "lorag": "LoRA$_{\\text{late}}$", "dora": "DoRA", "fullft": "Full FT",
    "dec_then_lora": "Dec$\\to$LoRA",
}
REV_METHODS = ["fullft_lowlr", "fullft_llrd", "fullft_accum", "mlpft", "rslora_lowlr"]
REV_TEX = {
    "fullft_lowlr": "Full FT, lower LR", "fullft_llrd": "Full FT, layer decay", "fullft_accum": "Full FT, accumulation",
    "mlpft": "MLP-only FT", "rslora_lowlr": "rsLoRA, matched LR",
}
REV_STACK = {
    "fullft_lowlr": ("Full FT,", "lower LR"), "fullft_llrd": ("Full FT,", "layer decay"), "fullft_accum": ("Full FT,", "accumulation"),
    "mlpft": ("MLP-only", "FT"), "rslora_lowlr": ("rsLoRA,", "matched LR"), "fullft": ("Full", "FT"),
}
COHORTS = ["amos_covid_luna", "fdg", "rsna2022", "rsna2023", "rsna2019_ich"]
COHORT_TEX = {
    "amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2022": "RSNA 2022 C-spine", "rsna2023": "RSNA 2023 trauma",
    "rsna2019_ich": "RSNA 2019 hemorrhage",
}
DATASETS = ["amos22", "kits23", "lits", "msd_pancreas", "totalseg"]
DATASET_TEX = {"amos22": "AMOS22", "kits23": "KiTS23", "lits": "LiTS", "msd_pancreas": "MSD Pancreas", "totalseg": "TotalSegmentator"}
BACKBONE_TEX = {"tapct": "TAP-CT", "spectre": "SPECTRE", "pillar0": "Pillar-0", "voco": "VoCo", "all": "All"}
MAIN_BB = ["tapct", "spectre", "pillar0"]
SEG_BB = ["tapct", "spectre"]
MODE_TEX = {"pertask": "per-task", "joint": "joint", "joint_lm": "joint (26\\,000 steps)", "joint_8k": "joint (8000 steps)",
            "pertask_cm": "per-task, compute-matched"}
EXCLUDED_COHORTS = {"breastdivider"}  # not part of the study (see metrics/PROCEDURE.md); present in metrics_cls.csv only
DNT = "$^{\\dagger}$"
KEY_TEX = {
    "cls_lora_lin": "LoRA $-$ Lin", "cls_lora_attn": "LoRA $-$ Attn", "cls_attnft_lora": "Attn-only FT $-$ LoRA", "cls_fullft_lin": "Full FT $-$ Lin",
    "cls_joint_pertask": "joint $-$ per-task", "seg_dec_lin": "Dec $-$ Lin", "seg_lora_dec": "LoRA $-$ Dec",
    "seg_lora_dec_pertask": "LoRA $-$ Dec, per-task", "seg_lora_dec_joint": "LoRA $-$ Dec, joint", "seg_fullft_dec": "Full FT $-$ Dec",
    "seg_joint_pertask": "joint $-$ per-task", "seg_lin_dec": "Lin $-$ Dec",
}
KEY_ROWS = {
    "cls": ["cls_lora_lin", "cls_lora_attn", "cls_attnft_lora", "cls_fullft_lin", "cls_joint_pertask"],
    "seg": ["seg_dec_lin", "seg_lora_dec", "seg_lora_dec_pertask", "seg_lora_dec_joint", "seg_fullft_dec", "seg_joint_pertask"],
}

WRITTEN: list[tuple[str, int]] = []
ISSUES: list[str] = []


# ----------------------------------------------------------------------------------------------- helpers
def key_tex(key: str) -> str:
    if key in KEY_TEX:
        return KEY_TEX[key]
    m = re.fullmatch(r"cls_(.+)_lin", key)
    if m and m.group(1) in METHOD_TEX:
        return f"{METHOD_TEX[m.group(1)]} $-$ Lin"
    m = re.fullmatch(r"seg_(.+)_dec", key)
    if m and m.group(1) in SEG_METHOD_TEX:
        return f"{SEG_METHOD_TEX[m.group(1)]} $-$ Dec"
    return key.replace("_", "\\_")


def isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


def f3(v) -> str:
    return "--" if isnan(v) else f"{float(v):.3f}"


def s3(v) -> str:
    """signed, three decimals; a value that rounds to zero is printed 0.000"""
    if isnan(v):
        return "--"
    r = round(float(v), 3)
    return "0.000" if r == 0 else f"{r:+.3f}"


def f2(v) -> str:
    return "--" if isnan(v) else f"{float(v):.2f}"


def pct(v) -> str:
    return "--" if isnan(v) else f"{float(v):.0f}"


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def label_tex(label: str) -> str:
    """classification label or segmentation structure as printed in the per-label tables"""
    src = None
    if ":" in label:
        src, label = label.split(":", 1)
    txt = label.replace("_", " ")
    if txt.isupper() and len(txt) > 2:
        txt = txt.lower()
    if src:
        txt += " (" + {"amos": "AMOS", "covid_ct": "COVID-CT", "luna25": "LUNA25"}.get(src, src) + ")"
    return esc(txt)


def stack(top: str, bottom: str) -> str:
    if not bottom:
        return top
    return "\\begin{tabular}[b]{@{}c@{}}%s\\\\%s\\end{tabular}" % (top, bottom)


def header_cell(mt: str, tex: dict, stacks: dict) -> str:
    return stack(*stacks[mt]) if mt in stacks else tex[mt]


def mark_row(vals: dict, dnt: set = frozenset(), fmt=f3, higher: bool = True) -> dict:
    """bold the best and underline the second best (compared at the printed precision); dnt methods plain + mark"""
    pool = sorted({round(float(v), 3) for k, v in vals.items() if not isnan(v) and k not in dnt}, reverse=higher)
    out = {}
    for k, v in vals.items():
        if isnan(v):
            out[k] = "--"
            continue
        c = fmt(v)
        if k in dnt:
            out[k] = c + DNT
            continue
        r = round(float(v), 3)
        if pool and r == pool[0]:
            c = f"\\textbf{{{c}}}"
        elif len(pool) > 1 and r == pool[1]:
            c = f"\\underline{{{c}}}"
        out[k] = c
    return out


def shade(cell: str) -> str:
    return "\\cellcolor{modebest}" + cell


def tabular(colspec: str, header: str | list[str], rows: list[str], pre: str = "") -> str:
    head = header if isinstance(header, list) else [header]
    return "\n".join([*( [pre] if pre else []), f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *head, "\\midrule", *rows,
                      "\\bottomrule", "\\end{tabular}"]) + "\n"


def write(name: str, tex: str) -> None:
    p = OUT_TEX / name
    p.write_text(tex)
    WRITTEN.append((name, tex.count("\\\\")))


def issue(msg: str) -> None:
    ISSUES.append(msg)


# ----------------------------------------------------------------------------------------------- inputs
def load() -> dict:
    d = {name: pd.read_csv(OUT_CSV / f"{name}.csv") for name in
         ["cohort_means", "label_level", "ranks", "spread", "convergence", "seeds", "seeds_summary", "revision", "paired_cls",
          "paired_seg", "metrics_cls", "metrics_seg"]}
    sp = d["spread"]
    # spread.csv is computed on classification probabilities: the flag applies to classification tables only
    d["dnt"] = {(r.backbone, r.method) for r in sp.itertuples() if bool(r.did_not_train)}
    return d


def cm_lookup(cm: pd.DataFrame, task: str, seed: int = 42) -> dict:
    """(backbone, cohort, mode, method) -> (value, n_labels)"""
    t = cm[(cm.task == task) & (cm.seed == seed)]
    return {(r.backbone, r.cohort, r["mode"], r.method): (float(r.value), int(r.n_labels)) for _, r in t.iterrows()}


# ----------------------------------------------------------------------------------------------- tab_key
def table_key(ll: pd.DataFrame, dnt: set, backbones: list[str], out: str, with_all: bool) -> None:
    cols = backbones + (["all"] if with_all else [])
    ci_bb = "all" if with_all else backbones[0]
    rows = []
    for task, title in (("cls", "Classification (AUROC)"), ("seg", "Segmentation (Dice)")):
        rows.append(f"\\multicolumn{{{len(cols) + 2}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for key in KEY_ROWS[task]:
            sub = ll[ll.key == key].set_index("backbone")
            cells = []
            for bb in cols:
                if bb not in sub.index:
                    cells.append("--")
                    if bb != "pillar0" or task != "seg":
                        issue(f"label_level: {key} has no row for backbone {bb}")
                    continue
                r = sub.loc[bb]
                c = f"\\mci{{{s3(r['mean'])}}}{{{s3(r.lo3)}}}{{{s3(r.hi3)}}}"
                # the Full FT - Lin estimate on SPECTRE includes a pair that did not train
                if key == "cls_fullft_lin" and (bb, "fullft") in dnt:
                    c += DNT
                cells.append(c)
            if ci_bb in sub.index:
                r = sub.loc[ci_bb]
                clus = f"{int(r.clusters_pos)}/{int(r.clusters_n)}"
            else:
                clus = "--"
            rows.append(f"{key_tex(key)} & " + " & ".join(cells) + f" & {clus} \\\\")
        if task == "cls":
            rows.append("\\addlinespace")
    header = "Comparison & " + " & ".join(BACKBONE_TEX[b] for b in cols) + " & Cohorts $>0$ \\\\"
    write(out, tabular("@{}l" + "c" * len(cols) + "c@{}", header, rows))


# ----------------------------------------------------------------------------------------------- per-cohort tables
def table_cohort(cm: pd.DataFrame, task: str, backbones: list[str], modes: tuple, methods: list[str], method_tex: dict, dnt: set,
                 out: str) -> None:
    look = cm_lookup(cm, task)
    groups = COHORTS if task == "cls" else DATASETS
    group_tex = COHORT_TEX if task == "cls" else DATASET_TEX
    lead_name = "Cohort (labels)" if task == "cls" else "Dataset (structures)"
    rows = []
    for gi, g in enumerate(groups):
        bbs = [bb for bb in backbones if any((bb, g, mo, mt) in look for mo in modes for mt in methods)]
        if not bbs:
            continue
        first_group = True
        for bb in bbs:
            n_labels = max(n for (b, c, mo, mt), (v, n) in look.items() if b == bb and c == g)
            vals = {mo: {mt: look.get((bb, g, mo, mt), (np.nan, 0))[0] for mt in methods} for mo in ("pertask", "joint", "joint_lm")}
            present = [mo for mo in modes if any(not isnan(v) for v in vals[mo].values())]
            if not present:
                continue
            for mi, mo in enumerate(present):
                dn = {mt for mt in methods if (bb, mt) in dnt}
                cells = mark_row(vals[mo], dn)
                other = {"pertask": "joint", "joint": "pertask"}.get(mo)
                for mt in methods:
                    v, o = vals[mo][mt], vals[other][mt] if other else np.nan
                    if other and not isnan(v) and not isnan(o) and mt not in dn and round(v, 3) > round(o, 3):
                        cells[mt] = shade(cells[mt])
                    if isnan(v) and not (task == "seg" and g == "totalseg" and mo == "pertask" and mt in {"lorag", "dora", "dec_then_lora"}):
                        issue(f"cohort_means: missing {task} {bb} {g} {mo} {mt}")
                lead = (f"{group_tex[g]} ({n_labels})" if first_group else "") + " & " + (BACKBONE_TEX[bb] if mi == 0 else "")
                first_group = False
                rows.append(f"{lead} & {MODE_TEX[mo]} & " + " & ".join(cells[mt] for mt in methods) + " \\\\")
            if bb != bbs[-1]:
                rows.append("\\addlinespace[1.5pt]")
        if gi < len(groups) - 1:
            rows.append("\\addlinespace")
    if rows and rows[-1] == "\\addlinespace":
        rows.pop()
    header = f"{lead_name} & Backbone & Mode & " + " & ".join(header_cell(mt, method_tex, METHOD_STACK) for mt in methods) + " \\\\"
    write(out, tabular("@{}lll" + "c" * len(methods) + "@{}", header, rows, pre="\\setlength{\\tabcolsep}{3pt}"))


# ----------------------------------------------------------------------------------------------- tab_seg_modes
def table_seg_modes(cm: pd.DataFrame, out: str) -> None:
    look = cm_lookup(cm, "seg")
    rows = []
    mode_cols = ["pertask", "joint", "joint_8k"]
    for di, ds in enumerate(DATASETS):
        for bi, bb in enumerate(SEG_BB):
            cells = []
            for mo in mode_cols:
                dec = look.get((bb, ds, mo, "dec"), (np.nan, 0))[0]
                for mt in ("lora", "fullft"):
                    v = look.get((bb, ds, mo, mt), (np.nan, 0))[0]
                    cells.append(s3(v - dec) if not isnan(v) and not isnan(dec) else "--")
            n = look[(bb, ds, "joint", "dec")][1]
            lead = f"{DATASET_TEX[ds]} ({n})" if bi == 0 else ""
            rows.append(f"{lead} & {BACKBONE_TEX[bb]} & " + " & ".join(cells) + " \\\\")
        if di < len(DATASETS) - 1:
            rows.append("\\addlinespace")
    head1 = ("Dataset (structures) & Backbone & \\multicolumn{2}{c}{per-task} & \\multicolumn{2}{c}{joint} & "
             "\\multicolumn{2}{c}{joint, 8000 steps} \\\\")
    cm_rules = "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}"
    head2 = " & & " + " & ".join(["LoRA $-$ Dec & Full FT $-$ Dec"] * 3) + " \\\\"
    write(out, tabular("@{}ll" + "cc" * 3 + "@{}", [head1, cm_rules, head2], rows))


# ----------------------------------------------------------------------------------------------- ranks
def table_ranks(rk: pd.DataFrame, task: str, backbones: list[str], methods: list[str], method_tex: dict, dnt: set, out: str) -> None:
    t = rk[rk.task == task]
    cols = backbones + ["all"]
    rows = []
    best = {}
    for bb in cols:
        s = t[(t.backbone == bb) & t.method.isin(methods) & t.mean_rank.notna()]
        s = s[[(bb_ not in dnt) for bb_ in zip(s.backbone, s.method)]] if bb != "all" else s
        best[bb] = round(float(s.mean_rank.min()), 2) if len(s) else None
    s_all = t[(t.backbone == "all") & t.method.isin(methods)]
    best_near = round(float(s_all.within_001_pct.max()), 0)
    best_regret = round(float(s_all.regret.min()), 3)
    n_slots = {bb: int(t[(t.backbone == bb) & t.n_slots.notna()].n_slots.max()) for bb in cols}
    for mt in methods:
        cells = []
        for bb in cols:
            r = t[(t.backbone == bb) & (t.method == mt)]
            if r.empty or isnan(r.mean_rank.iloc[0]):
                cells.append("--" + (DNT if (bb, mt) in dnt else ""))
                if not r.empty and (bb, mt) not in dnt:
                    issue(f"ranks: {task} {bb} {mt} has no mean rank")
                continue
            v = float(r.mean_rank.iloc[0])
            c = f2(v)
            if round(v, 2) == best[bb]:
                c = f"\\textbf{{{c}}}"
            if int(r.n_slots.iloc[0]) != n_slots[bb]:
                c += DNT  # pooled over fewer slots because a backbone-method pair did not train
            cells.append(c)
        r = s_all[s_all.method == mt]
        if r.empty:
            cells += ["--", "--"]
        else:
            near, reg = float(r.within_001_pct.iloc[0]), float(r.regret.iloc[0])
            cells.append(f"\\textbf{{{pct(near)}}}" if round(near, 0) == best_near else pct(near))
            cells.append(f"\\textbf{{{f3(reg)}}}" if round(reg, 3) == best_regret else f3(reg))
        rows.append(f"{method_tex[mt]} & " + " & ".join(cells) + " \\\\")
    head1 = "Method & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{stack(BACKBONE_TEX[bb], f'{{\\scriptsize {n_slots[bb]} slots}}')}}}" for bb in cols) \
        + " & " + stack("Within 0.01", "{\\scriptsize of best (\\%)}") + " & " + stack("Regret", "{\\scriptsize all}") + " \\\\"
    write(out, tabular("@{}l" + "c" * len(cols) + "cc@{}", head1, rows))


# ----------------------------------------------------------------------------------------------- paired tests
def table_paired(p: pd.DataFrame, backbones: list[str], baselines: list[str], methods: list[str], method_tex: dict, dnt: set,
                 out: str) -> None:
    # one column group per baseline; a baseline that did not train on a backbone leaves its group as -- in those rows
    all_bases = baselines
    rows = []
    for bi, bb in enumerate(backbones):
        t = p[p.backbone == bb]
        if t.empty:
            continue
        for mt in methods:
            cells = []
            ns = set()
            for b in all_bases:
                s = t[(t.baseline == b) & (t.method == mt)]
                if mt == b or (bb, b) in dnt or s.empty:
                    cells += ["--"] * 4
                    if s.empty and mt != b and (bb, b) not in dnt:
                        issue(f"paired: no rows for {bb} {mt} vs {b}")
                    continue
                ns.add(len(s))
                sig = s.p_bh < 0.05
                cells += [s3(s["diff"].mean()), pct(100 * (s["diff"] > 0).mean()), pct(100 * (sig & (s["diff"] > 0)).mean()),
                          pct(100 * (sig & (s["diff"] < 0)).mean())]
            n_txt = "/".join(str(n) for n in sorted(ns)) if ns else "--"
            mark = DNT if (bb, mt) in dnt else ""
            rows.append(f"{BACKBONE_TEX[bb] if mt == methods[0] else ''} & {method_tex[mt]}{mark} & {n_txt} & " + " & ".join(cells) + " \\\\")
        if bi < len(backbones) - 1:
            rows.append("\\addlinespace")
    head1 = "Backbone & Method & $n$ & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{vs {method_tex[b]}}}" for b in all_bases) + " \\\\"
    rules = "".join(f"\\cmidrule(lr){{{4 + 4 * i}-{7 + 4 * i}}}" for i in range(len(all_bases)))
    head2 = " & & & " + " & ".join(["$\\Delta$ & $>0$ & $+$ & $-$"] * len(all_bases)) + " \\\\"
    write(out, tabular("@{}llr" + "rrrr" * len(all_bases) + "@{}", [head1, rules, head2], rows, pre="\\setlength{\\tabcolsep}{3.5pt}"))


# ----------------------------------------------------------------------------------------------- tab_cis
def table_cis(ll: pd.DataFrame, task: str, out: str) -> None:
    t = ll[ll.task == task]
    keys = KEY_ROWS[task] + [k for k in t.key.unique() if k not in KEY_ROWS[task]]
    bbs = [b for b in ["tapct", "spectre", "pillar0", *VOCO, "all"] if b in set(t.backbone)]
    rows = []
    for ki, key in enumerate(keys):
        sub = t[t.key == key].set_index("backbone")
        first = True
        for bb in bbs:
            if bb not in sub.index:
                continue
            r = sub.loc[bb]
            rows.append(f"{key_tex(key) if first else ''} & {BACKBONE_TEX[bb]} & {int(r.n_labels)} & {s3(r['mean'])} & "
                        f"[{s3(r.lo2)}, {s3(r.hi2)}] & [{s3(r.lo3)}, {s3(r.hi3)}] & {int(r.clusters_pos)}/{int(r.clusters_n)} \\\\")
            first = False
        if ki < len(keys) - 1:
            rows.append("\\addlinespace[2pt]")
    header = "Comparison & Backbone & $n$ & Mean & 95\\% CI (two-stage) & 95\\% CI (three-stage) & Cohorts $>0$ \\\\"
    write(out, tabular("@{}llrrccc@{}", header, rows))


# ----------------------------------------------------------------------------------------------- seeds
def table_seeds(sd: pd.DataFrame, task: str, methods: list[str], method_tex: dict, dnt: set, out: str) -> None:
    t = sd[sd.task == task]
    groups = COHORTS if task == "cls" else DATASETS
    gtex = COHORT_TEX if task == "cls" else DATASET_TEX
    seeds = [42, 43, 44]
    rows = []
    bbs = [b for b in MAIN_BB if b in set(t.backbone)]
    for bi, bb in enumerate(bbs):
        first = True
        for g in groups:
            s = t[(t.backbone == bb) & (t.cohort == g)]
            if s.empty:
                continue
            cells = []
            for mt in methods:
                vals = {int(r.seed): float(r.value) for r in s[s.method == mt].itertuples()}
                mark = DNT if (bb, mt) in dnt else ""
                for sdd in seeds:
                    cells.append((f3(vals[sdd]) + mark) if sdd in vals else "--")
                    if sdd not in vals and (bb, mt) not in dnt:
                        issue(f"seeds: missing {task} {bb} {g} {mt} seed {sdd}")
                cells.append(f3(np.std(list(vals.values()), ddof=1)) if len(vals) >= 2 else "--")
            rows.append(f"{BACKBONE_TEX[bb] if first else ''} & {gtex[g]} & " + " & ".join(cells) + " \\\\")
            first = False
        if bi < len(bbs) - 1:
            rows.append("\\addlinespace")
    k = len(seeds) + 1
    head1 = "Backbone & " + ("Cohort" if task == "cls" else "Dataset") + " & " + \
        " & ".join(f"\\multicolumn{{{k}}}{{c}}{{{method_tex[mt]}}}" for mt in methods) + " \\\\"
    rules = "".join(f"\\cmidrule(lr){{{3 + k * i}-{2 + k * (i + 1)}}}" for i in range(len(methods)))
    head2 = " & & " + " & ".join(["42 & 43 & 44 & SD"] * len(methods)) + " \\\\"
    write(out, tabular("@{}ll" + "c" * (k * len(methods)) + "@{}", [head1, rules, head2], rows, pre="\\setlength{\\tabcolsep}{3.5pt}"))


def table_seeds_summary(ss: pd.DataFrame, dnt: set, out: str) -> None:
    rows = []
    comp_tex = {"lora>lin": "LoRA $>$ Lin", "attnft>lora": "Attn-only FT $>$ LoRA", "lora>dec": "LoRA $>$ Dec", "fullft>dec": "Full FT $>$ Dec"}
    for ti, (task, title, methods, mtex) in enumerate((("cls", "Classification (AUROC)", ["lin", "lora", "attnft", "fullft"], METHOD_TEX),
                                                       ("seg", "Segmentation (Dice)", ["dec", "lora", "fullft"], SEG_METHOD_TEX))):
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        t = ss[ss.task == task]
        bbs = [b for b in MAIN_BB + ["all"] if b in set(t.backbone)]
        for bb in bbs:
            first = True
            for mt in methods:
                sdr = t[(t.backbone == bb) & (t.method == mt) & (t.comparison.isna() | (t.comparison == ""))]
                cmp_ = t[(t.backbone == bb) & (t.method == mt) & t.comparison.notna() & (t.comparison != "")]
                if sdr.empty and cmp_.empty:
                    continue
                if not cmp_.empty and int(cmp_.pairs_n.iloc[0]) == 0:
                    cmp_ = cmp_.iloc[0:0]
                if (sdr.empty or int(sdr.n_cells.iloc[0]) == 0) and cmp_.empty:
                    if task == "cls" and (bb, mt) not in dnt:
                        issue(f"seeds_summary: {task} {bb} {mt} has no cells")
                    if task == "seg" or (bb, mt) not in dnt:
                        continue
                sd_med = f3(sdr.sd_median.iloc[0]) if not sdr.empty else "--"
                sd_max = f3(sdr.sd_max.iloc[0]) if not sdr.empty else "--"
                n_cells = str(int(sdr.n_cells.iloc[0])) if not sdr.empty else "--"
                if task == "cls" and (bb, mt) in dnt:
                    sd_med += DNT
                if cmp_.empty:
                    c_tex, pairs, extra = "", "", ""
                else:
                    r = cmp_.iloc[0]
                    c_tex = comp_tex.get(r.comparison, esc(r.comparison))
                    pairs = f"{int(r.pairs_pos)}/{int(r.pairs_n)}"
                    extra = f"{int(r.extra_pairs_pos)}/{int(r.extra_pairs_n)}"
                rows.append(f"{BACKBONE_TEX[bb] if first else ''} & {mtex[mt]} & {sd_med} & {sd_max} & {n_cells} & {c_tex} & {pairs} & {extra} \\\\")
                first = False
            if not first:
                rows.append("\\addlinespace[2pt]")
        rows.pop()
        if ti == 0:
            rows.append("\\addlinespace")
    head1 = "Backbone & Method & \\multicolumn{3}{c}{SD over seeds 42--44} & Comparison & \\multicolumn{2}{c}{Cohort pairs ahead} \\\\"
    rules = "\\cmidrule(lr){3-5}\\cmidrule(lr){7-8}"
    head2 = " & & median & max & cells & & all seeds & seeds 43, 44 \\\\"
    write(out, tabular("@{}llrrrlcc@{}", [head1, rules, head2], rows))


# ----------------------------------------------------------------------------------------------- revision and controls
def table_revision_cls(cm: pd.DataFrame, dnt: set, out: str) -> None:
    look = cm_lookup(cm, "cls")
    cols = ["lin", "lora", "fullft", "fullft_lowlr", "fullft_llrd", "fullft_accum", "mlpft", "rslora", "rslora_lowlr"]
    tex = {**METHOD_TEX, **REV_TEX}
    stacks = {**{k: v for k, v in METHOD_STACK.items()}, **REV_STACK}
    rows = []
    bbs = [bb for bb in MAIN_BB if any(b == bb and mt in REV_METHODS for (b, c, mo, mt) in look)]
    for bi, bb in enumerate(bbs):
        first = True
        for g in COHORTS:
            vals = {mt: look.get((bb, g, "joint", mt), (np.nan, 0))[0] for mt in cols}
            if all(isnan(v) for v in vals.values()):
                continue
            for mt in cols:
                if isnan(vals[mt]):
                    issue(f"revision: missing cls {bb} {g} joint {mt}")
            n = look[(bb, g, "joint", "lin")][1]
            cells = mark_row(vals, {mt for mt in cols if (bb, mt) in dnt})
            rows.append(f"{BACKBONE_TEX[bb] if first else ''} & {COHORT_TEX[g]} ({n}) & " + " & ".join(cells[mt] for mt in cols) + " \\\\")
            first = False
        if bi < len(bbs) - 1:
            rows.append("\\addlinespace")
    header = "Backbone & Cohort (labels) & " + " & ".join(header_cell(mt, tex, stacks) for mt in cols) + " \\\\"
    write(out, tabular("@{}ll" + "c" * len(cols) + "@{}", header, rows, pre="\\setlength{\\tabcolsep}{3.5pt}"))


def table_compute_matched(cm: pd.DataFrame, out: str) -> None:
    look = cm_lookup(cm, "seg")
    rows = []
    for di, ds in enumerate(["amos22", "totalseg"]):
        for bi, bb in enumerate(SEG_BB):
            for mi, mt in enumerate(["dec", "lora", "fullft"]):
                j = look.get((bb, ds, "joint", mt), (np.nan, 0))[0]
                cmv = look.get((bb, ds, "pertask_cm", mt), (np.nan, 0))[0]
                pt = look.get((bb, ds, "pertask", mt), (np.nan, 0))[0]
                if isnan(j) or isnan(cmv) or isnan(pt):
                    issue(f"compute-matched: missing {bb} {ds} {mt}")
                n = look[(bb, ds, "joint", "dec")][1]
                lead = (f"{DATASET_TEX[ds]} ({n})" if bi == 0 and mi == 0 else "") + " & " + (BACKBONE_TEX[bb] if mi == 0 else "")
                diff = s3(j - cmv) if not isnan(j) and not isnan(cmv) else "--"
                rows.append(f"{lead} & {SEG_METHOD_TEX[mt]} & {f3(j)} & {f3(cmv)} & {diff} & {f3(pt)} \\\\")
            if bi == 0:
                rows.append("\\addlinespace[2pt]")
        if di == 0:
            rows.append("\\addlinespace")
    header = ["Dataset (structures) & Backbone & Method & " + stack("Joint", "{\\scriptsize 24\\,000 steps}") + " & "
              + stack("Per-task, matched", "{\\scriptsize 1600 steps each}") + " & Joint $-$ matched & "
              + stack("Per-task", "{\\scriptsize 8000 steps each}") + " \\\\"]
    write(out, tabular("@{}lllcccc@{}", header, rows))


def table_scratch(cm: pd.DataFrame, out: str) -> None:
    look = cm_lookup(cm, "seg")
    rows = []
    for ds in DATASETS:
        sc = look.get(("tapct", ds, "joint", "scratch"), (np.nan, 0))[0]
        dec = look.get(("tapct", ds, "joint", "dec"), (np.nan, 0))[0]
        ft = look.get(("tapct", ds, "joint", "fullft"), (np.nan, 0))[0]
        if isnan(sc):
            issue(f"scratch: missing tapct {ds}")
        n = look[("tapct", ds, "joint", "dec")][1]
        d1 = s3(sc - dec) if not isnan(sc) and not isnan(dec) else "--"
        d2 = s3(sc - ft) if not isnan(sc) and not isnan(ft) else "--"
        rows.append(f"{DATASET_TEX[ds]} ({n}) & {f3(sc)} & {f3(dec)} & {f3(ft)} & {d1} & {d2} \\\\")
    header = "Dataset (structures) & From scratch & Dec & Full FT & Scratch $-$ Dec & Scratch $-$ Full FT \\\\"
    write(out, tabular("@{}lccccc@{}", header, rows))


# ----------------------------------------------------------------------------------------------- per-label tables
def block(title: str, tex: str) -> str:
    return ("\\noindent\\begin{minipage}{\\textwidth}\\centering\\footnotesize\\textit{" + title + "}\\\\[2pt]\n"
            "\\adjustbox{max width=\\textwidth}{%\n" + tex + "}\\end{minipage}\n\\par\\bigskip\n")


def tables_perlabel(mc: pd.DataFrame, ms: pd.DataFrame, dnt: set) -> None:
    # classification: held-out AUROC per label, one block per cohort, modes as row groups
    d = mc[(mc.split == "test") & (mc.seed == 42) & mc.method.isin(METHODS) & ~mc.cohort.isin(EXCLUDED_COHORTS)]
    for bb in MAIN_BB + VOCO:
        parts = []
        for g in COHORTS:
            s = d[(d.backbone == bb) & (d.cohort == g)]
            if s.empty:
                continue
            labels = list(dict.fromkeys(s.label))  # file order
            rows = []
            modes = [mo for mo in ("pertask", "joint", "joint_lm") if mo in set(s["mode"])]
            for mi, mo in enumerate(modes):
                rows.append(f"\\multicolumn{{{len(METHODS) + 1}}}{{@{{}}l}}{{\\textit{{{MODE_TEX[mo]}}}}} \\\\")
                piv = s[s["mode"] == mo].pivot_table(index="label", columns="method", values="auroc")
                for lab in labels:
                    vals = {mt: (float(piv.loc[lab, mt]) if lab in piv.index and mt in piv.columns and not isnan(piv.loc[lab, mt]) else np.nan)
                            for mt in METHODS}
                    for mt, v in vals.items():
                        if isnan(v):
                            issue(f"metrics_cls: missing {bb} {g} {mo} {lab} {mt}")
                    cells = mark_row(vals, {mt for mt in METHODS if (bb, mt) in dnt})
                    rows.append(f"{label_tex(lab)} & " + " & ".join(cells[mt] for mt in METHODS) + " \\\\")
                if mi < len(modes) - 1:
                    rows.append("\\addlinespace")
            header = "Label & " + " & ".join(header_cell(mt, METHOD_TEX, METHOD_STACK) for mt in METHODS) + " \\\\"
            tex = tabular("@{}l" + "c" * len(METHODS) + "@{}", header, rows, pre="\\setlength{\\tabcolsep}{3pt}")
            parts.append(block(f"{COHORT_TEX[g]}, {BACKBONE_TEX[bb]}: held-out AUROC per label", tex))
        if parts:
            write(f"tab_perlabel_cls_{bb}.tex", "".join(parts))
    # segmentation: held-out Dice per structure
    d = ms[(ms.split == "test") & (ms.seed == 42) & ms.method.isin(SEG_METHODS)]
    for bb in SEG_BB + VOCO:
        parts = []
        for g in DATASETS:
            s = d[(d.backbone == bb) & (d.cohort == g)]
            if s.empty:
                continue
            labels = list(dict.fromkeys(s.label))
            rows = []
            modes = [mo for mo in ("pertask", "joint", "joint_8k") if mo in set(s["mode"])]
            for mi, mo in enumerate(modes):
                rows.append(f"\\multicolumn{{{len(SEG_METHODS) + 1}}}{{@{{}}l}}{{\\textit{{{MODE_TEX[mo]}}}}} \\\\")
                piv = s[s["mode"] == mo].pivot_table(index="label", columns="method", values="dice")
                for lab in labels:
                    vals = {mt: (float(piv.loc[lab, mt]) if lab in piv.index and mt in piv.columns and not isnan(piv.loc[lab, mt]) else np.nan)
                            for mt in SEG_METHODS}
                    for mt, v in vals.items():
                        if isnan(v) and not (g == "totalseg" and mo == "pertask" and mt in {"lorag", "dora", "dec_then_lora"}):
                            issue(f"metrics_seg: missing {bb} {g} {mo} {lab} {mt}")
                    cells = mark_row(vals)
                    rows.append(f"{label_tex(lab)} & " + " & ".join(cells[mt] for mt in SEG_METHODS) + " \\\\")
                if mi < len(modes) - 1:
                    rows.append("\\addlinespace")
            header = "Structure & " + " & ".join(header_cell(mt, SEG_METHOD_TEX, METHOD_STACK) for mt in SEG_METHODS) + " \\\\"
            tex = tabular("@{}l" + "c" * len(SEG_METHODS) + "@{}", header, rows, pre="\\setlength{\\tabcolsep}{3pt}")
            parts.append(block(f"{DATASET_TEX[g]}, {BACKBONE_TEX[bb]}: held-out Dice per structure", tex))
        if parts:
            write(f"tab_perstructure_seg_{bb}.tex", "".join(parts))


# ----------------------------------------------------------------------------------------------- convergence
def table_convergence(cv: pd.DataFrame, out: str, out_methods: str) -> None:
    t = cv[(cv.backbone != "all")]
    rows = []
    for ti, (task, title, modes, bbs) in enumerate((("cls", "Classification", ["pertask", "joint", "joint_lm"], MAIN_BB + VOCO),
                                                    ("seg", "Segmentation", ["pertask", "joint", "joint_8k", "pertask_cm"], SEG_BB + VOCO))):
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for bb in bbs:
            first = True
            for mo in modes:
                s = t[(t.task == task) & (t.backbone == bb) & (t["mode"] == mo)]
                if s.empty:
                    continue
                n = int(s.n_runs.sum())
                last = 100 * float((s.frac_last * s.n_runs).sum() / n)
                two = 100 * float((s.frac_last_two * s.n_runs).sum() / n)
                rows.append(f"{BACKBONE_TEX[bb] if first else ''} & {MODE_TEX[mo]} & {n} & {pct(last)} & {pct(two)} \\\\")
                first = False
            rows.append("\\addlinespace[2pt]")
        a = cv[(cv.task == task) & (cv.backbone == "all")]
        if not a.empty:
            r = a.iloc[0]
            rows.append(f"All & all & {int(r.n_runs)} & {pct(100 * r.frac_last)} & {pct(100 * r.frac_last_two)} \\\\")
        if ti == 0:
            rows.append("\\addlinespace")
    header = ["Backbone & Mode & Runs & " + stack("Last validation", "{\\scriptsize selected (\\%)}") + " & "
              + stack("One of last two", "{\\scriptsize selected (\\%)}") + " \\\\"]
    write(out, tabular("@{}llrcc@{}", header, rows))
    # per method: share of runs whose selected checkpoint is one of the last two validations, per backbone and mode
    rows = []
    for ti, (task, title, methods, mtex, modes, bbs) in enumerate((
            ("cls", "Classification", METHODS + REV_METHODS, {**METHOD_TEX, **REV_TEX}, ["pertask", "joint"], MAIN_BB + VOCO),
            ("seg", "Segmentation", SEG_METHODS + ["scratch"], {**SEG_METHOD_TEX, "scratch": "From scratch"}, ["pertask", "joint"], MAIN_BB + VOCO))):
        ncol = 1 + 2 * len(bbs)
        rows.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for mt in methods:
            cells = []
            for bb in bbs:
                for mo in modes:
                    s = t[(t.task == task) & (t.backbone == bb) & (t["mode"] == mo) & (t.method == mt)]
                    cells.append("--" if s.empty else f"{pct(100 * float(s.frac_last_two.iloc[0]))} ({int(s.n_runs.iloc[0])})")
            rows.append(f"{mtex[mt]} & " + " & ".join(cells) + " \\\\")
        if ti == 0:
            rows.append("\\addlinespace")
    bbs = MAIN_BB + VOCO
    head1 = "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{BACKBONE_TEX[bb]}}}" for bb in bbs) + " \\\\"
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(bbs)))
    head2 = " & " + " & ".join(["per-task & joint"] * len(bbs)) + " \\\\"
    write(out_methods, tabular("@{}l" + "c" * (2 * len(bbs)) + "@{}", [head1, rules, head2], rows, pre="\\setlength{\\tabcolsep}{4pt}"))


# ----------------------------------------------------------------------------------------------- check
PREAMBLE = r"""\documentclass{article}
\usepackage[table]{xcolor}
\usepackage{booktabs,colortbl,adjustbox,amsmath,multirow}
\usepackage[margin=1cm,landscape]{geometry}
\definecolor{modebest}{HTML}{E8F1FA}
\newcommand{\mci}[3]{\begin{tabular}[c]{@{}c@{}}#1\\[-1.5pt]{\scriptsize[#2, #3]}\end{tabular}}
\begin{document}\small
"""


def check(names: list[str]) -> list[str]:
    bad = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for name in names:
            src = (OUT_TEX / name).read_text()
            wrapped = src if "\\begin{minipage}" in src else "\\noindent\\adjustbox{max width=\\textwidth}{%\n" + src + "}\n"
            doc = tdp / (name.replace(".tex", "") + "_check.tex")
            doc.write_text(PREAMBLE + wrapped + "\\end{document}\n")
            r = subprocess.run([str(TECTONIC), "--chatter", "minimal", "-o", td, str(doc)], capture_output=True, text=True)
            if r.returncode != 0:
                bad.append(name)
                print(f"  LaTeX error in {name}:\n" + "\n".join(line for line in (r.stdout + r.stderr).splitlines() if "error" in line.lower() or "!" in line)[:2000])
    return bad


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compile every written table in a standalone document")
    args = ap.parse_args()
    OUT_TEX.mkdir(parents=True, exist_ok=True)
    D = load()
    cm, ll, rk, dnt = D["cohort_means"], D["label_level"], D["ranks"], D["dnt"]

    table_key(ll, dnt, MAIN_BB, "tab_key.tex", with_all=True)
    if not NO_VOCO:
        table_key(ll, dnt, ["voco"], "tab_key_voco.tex", with_all=False)

    table_cohort(cm, "cls", MAIN_BB, ("pertask", "joint"), METHODS, METHOD_TEX, dnt, "tab_cls_main.tex")
    if not NO_VOCO:
        table_cohort(cm, "cls", ["voco"], ("pertask", "joint"), METHODS, METHOD_TEX, dnt, "tab_cls_voco.tex")
    table_cohort(cm, "cls", MAIN_BB + VOCO, ("pertask", "joint", "joint_lm"), METHODS, METHOD_TEX, dnt, "tab_cls_ladder.tex")
    table_cohort(cm, "seg", SEG_BB, ("pertask", "joint"), SEG_METHODS, SEG_METHOD_TEX, set(), "tab_seg_main.tex")
    if not NO_VOCO:
        table_cohort(cm, "seg", ["voco"], ("pertask", "joint"), SEG_METHODS, SEG_METHOD_TEX, set(), "tab_seg_voco.tex")
    table_seg_modes(cm, "tab_seg_modes.tex")

    table_ranks(rk, "cls", MAIN_BB, METHODS, METHOD_TEX, dnt, "tab_ranks_cls.tex")
    table_ranks(rk, "seg", SEG_BB, SEG_METHODS, SEG_METHOD_TEX, set(), "tab_ranks_seg.tex")

    table_paired(D["paired_cls"], MAIN_BB, ["lin", "lora", "attn", "fullft"], METHODS, METHOD_TEX, dnt, "tab_paired_cls.tex")
    if not NO_VOCO:
        table_paired(D["paired_cls"], ["voco"], ["lin", "lora", "attn", "fullft"], METHODS, METHOD_TEX, dnt, "tab_paired_cls_voco.tex")
    table_paired(D["paired_seg"], SEG_BB, ["lin", "dec", "lora"], SEG_METHODS, SEG_METHOD_TEX, set(), "tab_paired_seg.tex")
    if not NO_VOCO:
        table_paired(D["paired_seg"], ["voco"], ["lin", "dec", "lora"], SEG_METHODS, SEG_METHOD_TEX, set(), "tab_paired_seg_voco.tex")

    table_cis(ll, "cls", "tab_cis_cls.tex")
    table_cis(ll, "seg", "tab_cis_seg.tex")

    table_seeds(D["seeds"], "cls", ["lin", "lora", "attnft", "fullft"], METHOD_TEX, dnt, "tab_seeds_cls.tex")
    table_seeds(D["seeds"], "seg", ["dec", "lora", "fullft"], SEG_METHOD_TEX, set(), "tab_seeds_seg.tex")
    table_seeds_summary(D["seeds_summary"], dnt, "tab_seeds_summary.tex")

    table_revision_cls(cm, dnt, "tab_revision_cls.tex")
    table_compute_matched(cm, "tab_compute_matched.tex")
    table_scratch(cm, "tab_scratch.tex")

    tables_perlabel(D["metrics_cls"], D["metrics_seg"], dnt)
    table_convergence(D["convergence"], "tab_convergence.tex", "tab_convergence_methods.tex")

    # ------------------------------------------------------------------ tables in the previous draft's layouts
    # (tables/paper_tables.py; written last, so its tab_key, tab_cls_main, tab_cls_ladder, tab_seg_modes and
    # tab_seeds_* replace the v2 tables of the same name; the v2 versions are kept in paper_compact_20261101/)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paper_tables
    legacy = paper_tables.write_all()
    print(f"paper_tables.py (previous layouts): wrote {len(legacy)} files: {' '.join(legacy)}")

    print(f"wrote {len(WRITTEN)} files to {OUT_TEX}:")
    for name, n in WRITTEN:
        print(f"  {name:36s} {n:4d} rows")
    uniq = list(dict.fromkeys(ISSUES))
    if uniq:
        print(f"{len(uniq)} data notes:")
        for m in uniq:
            print("  " + m)
    if args.check:
        if not TECTONIC.exists():
            raise SystemExit(f"tectonic not found at {TECTONIC}")
        bad = check([n for n, _ in WRITTEN])
        print("LaTeX check: " + ("all tables compile" if not bad else f"{len(bad)} failing: {bad}"))
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
