#!/usr/bin/env python3
"""Tables of the previous paper draft in their previous layouts and file names, filled from the verified CSVs.

Called by make_tables.py (after its own tables, so the files below replace v2 files of the same name); can also be run
alone: python paper_tables.py.  Reads metrics/out/*.csv only (schemas in metrics/SCHEMA.md) and writes to
paper/tables/:

    tab_key, tab_cls_main, tab_cls_ladder, tab_seg_ladder, tab_methods, tab_headline, tab_cis, tab_cls_paired,
    tab_seg_paired, tab_seg_budget, tab_seg_modes, tab_seeds_cls, tab_seeds_seg, tab_fullft_fix, tab_controls,
    tab_seg_cm, tab_app_all (+ tab_app_<cohort>[_<backbone>_<i>]), tab_seg_app_all (+ tab_seg_app_<dataset>),
    tab_cohorts, tab_spread

VoCo is quarantined: tables are written without it to ../paper/tables/ (WITH_VOCO=1 restores the VoCo variant).

Layout conventions of the previous draft: bold = best, underline = second best at the printed precision;
\\cellcolor{modebest} = the better of per-task and joint for a method (no shade on a printed tie); \\mci{mean}{lo}{hi}
in tab_key.  The layouts follow midl_paper/results/scripts/make_tables.py and revision_tables.py; the differences in
definition from those scripts are listed in tables/PAPER_TABLES.md.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
OUT_CSV = V2 / "metrics" / "out"
# VoCo is quarantined (2026-11-02, ../quarantine_voco_20261102/): every table leaves out the VoCo backbone unless WITH_VOCO=1 (needs the
# VoCo runs restored in the inventory). Written to ../paper/tables (PAPER_DIR overrides the paper folder).
NO_VOCO = os.environ.get("WITH_VOCO", "") in ("", "0")
OUT_TEX = Path(os.environ.get("PAPER_DIR") or (V2 / "paper")) / "tables"

METHODS = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
METHOD_TEX = {"lin": "Lin", "attn": "Attn", "lora": "LoRA", "lorag": "LoRA$_{\\text{late}}$", "babmil": "LoRA+ABMIL",
              "fullft": "Full FT", "lin_then_lorag": "Lin$\\to$LoRA", "attn_then_lorag": "Attn$\\to$LoRA", "dora": "DoRA",
              "rslora": "rsLoRA", "attnft": "Attn-only FT"}
SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
SEG_METHOD_TEX = {"lin": "Lin", "dec": "Dec", "lora": "LoRA", "lorag": "LoRA$_{\\text{late}}$", "dora": "DoRA",
                  "fullft": "Full FT", "dec_then_lora": "Dec$\\to$LoRA"}
COHORTS = ["amos_covid_luna", "fdg", "rsna2022", "rsna2023", "rsna2019_ich"]
DATASETS = ["amos22", "kits23", "lits", "msd_pancreas", "totalseg"]
DATASET_TEX = {"amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2022": "RSNA22 C-spine", "rsna2023": "RSNA23 trauma",
               "rsna2019_ich": "RSNA19 ICH", "amos22": "AMOS22", "kits23": "KiTS23", "lits": "LiTS", "msd_pancreas": "MSD Pancreas",
               "totalseg": "TotalSegmentator"}
SHORT_MAIN = {"amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2022": "RSNA22 C-spine", "rsna2023": "RSNA23 trauma",
              "rsna2019_ich": "RSNA19 ICH"}  # cohort short names of Section 3 (E9)
SHORT = {"amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2022": "RSNA22 C-spine", "rsna2023": "RSNA23 trauma",
         "rsna2019_ich": "RSNA19 ICH", "amos22": "AMOS22", "kits23": "KiTS23", "lits": "LiTS", "msd_pancreas": "MSD Pancreas",
         "totalseg": "TotalSegmentator"}
BACKBONE_TEX = {"tapct": "TAP-CT", "spectre": "SPECTRE", "pillar0": "Pillar-0", "voco": "VoCo"}
MAIN_BB = ["tapct", "spectre", "pillar0"]
PER_BB = MAIN_BB + ([] if NO_VOCO else ["voco"])
MODE_TEX = {"pertask": "per-task", "joint": "joint", "joint_lm": "joint (26\\,000 steps)"}
SEG_N_STRUCT = {"amos22": 15, "kits23": 3, "lits": 2, "msd_pancreas": 2, "totalseg": 15}
EXCLUDED_COHORTS = {"breastdivider"}
EXCLUDED_LABELS = {"lesion_malignant"}
SPREAD_MIN = 0.01
FIXES = ["fullft_lowlr", "fullft_llrd", "fullft_accum"]
FIX_TEX = {"fullft": "default", "fullft_lowlr": "lower LR", "fullft_llrd": "layer decay", "fullft_accum": "accum.\\ + clip"}
CLS_ORDER = [("tapct", "amos_covid_luna"), ("tapct", "fdg"), ("tapct", "rsna2023"), ("tapct", "rsna2022"), ("tapct", "rsna2019_ich"),
             ("spectre", "amos_covid_luna"), ("spectre", "fdg"), ("spectre", "rsna2023"), ("spectre", "rsna2022"),
             ("pillar0", "rsna2019_ich")]
SEG_ORDER = [(bb, ds) for bb in ("tapct", "spectre") for ds in DATASETS]
# label-level comparisons of tab_key / tab_cis: (label_level key, task, row label); LoRA - Attn (matched head) added
KEY_ROWS = [("cls_lora_lin", "cls", "LoRA $-$ Lin"), ("cls_lora_attn", "cls", "LoRA $-$ Attn"),
            ("cls_attnft_lora", "cls", "Attn-only FT $-$ LoRA"), ("cls_fullft_lin", "cls", "Full FT $-$ Lin"),
            ("cls_joint_pertask", "cls", "Joint $-$ per-task"), ("seg_dec_lin", "seg", "Dec $-$ Lin"),
            ("seg_lora_dec", "seg", "LoRA $-$ Dec"), ("seg_lora_dec_pertask", "seg", "\\quad per-structure models"),
            ("seg_lora_dec_joint", "seg", "\\quad joint models"), ("seg_fullft_dec", "seg", "Full FT $-$ Dec"),
            ("seg_joint_pertask", "seg", "Joint $-$ per-task")]
# extra classification rows of tab_cis only, so that every label-level estimate quoted in Section 4.1 has a table row
CIS_EXTRA_ROWS = [("cls_attn_lin", "cls", "Attn $-$ Lin"), ("cls_attnft_lin", "cls", "Attn-only FT $-$ Lin"),
                  ("cls_attnft_attn", "cls", "Attn-only FT $-$ Attn"),
                  ("cls_lora_lin_nosc", "cls", "LoRA $-$ Lin, without FDG-PET/CT and COVID-CTset")]
# Training-split sizes are not in the per-case files; they are the manifest counts of the v2 appendix table
# "Cohorts and datasets" (paper/sections/appendix.tex) and report/code_map_classification.md section 1.
# AMOS train = distinct case ids over the 15 per-label CSVs. Selection / held-out = manifest counts (same source).
COHORT_SPLITS = {  # key: (name, labels, train, selection, held-out)
    "amos": ("AMOS (disease labels)", 15, 1107, 178, 178),
    "covid": ("COVID-CTset", 1, 308, 32, 32),
    "luna": ("LUNA25 (nodule malignancy)", 1, 1000, 100, 100),
    "fdg": ("FDG-PET/CT (CT only)", 4, 776, 116, 122),
    "rsna2022": ("RSNA22 C-spine (cervical-spine fractures)", 8, 1615, 202, 202),
    "rsna2023": ("RSNA23 trauma (abdominal trauma)", 4, 2517, 315, 315),
    "ich": ("RSNA19 ICH (intracranial hemorrhage)", 6, 5000, 3269, 3264),
}

WRITTEN: list[str] = []


# ----------------------------------------------------------------------------------------------- helpers
def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _mark(v: float, pool, higher: bool = True, dec: int = 3) -> str:
    """bold best, underline second best of pool, compared at the printed precision"""
    vals = sorted({round(float(x), dec) for x in pool if not _isnan(x)}, reverse=higher)
    s = f"{v:.{dec}f}"
    if vals and round(v, dec) == vals[0]:
        return f"\\textbf{{{s}}}"
    if len(vals) > 1 and round(v, dec) == vals[1]:
        return f"\\underline{{{s}}}"
    return s


def _mode_shade(cell: str, v: float, other: float, dec: int = 3) -> str:
    if _isnan(v) or _isnan(other) or round(v, dec) <= round(other, dec):
        return cell
    return "\\cellcolor{modebest}" + cell


def _f3s(v: float) -> str:
    """signed three decimals; a value that rounds to zero is 0.000"""
    s = f"{v:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s


def _two(a: str, b: str) -> str:
    return f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{a}\\\\{b}\\end{{tabular}}"


def _group(rows: list[tuple[str, str]], ncol: int) -> list[str]:
    out, last = [], None
    for g, line in rows:
        if g != last:
            out.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\")
            last = g
        out.append(line)
    return out


def _esc(s: str) -> str:
    return s.replace("_", "\\_")


def _fmt_params(n: float) -> str:
    n = float(n)
    if n < 1e4:
        return f"{int(n)}"
    if n < 1e6:
        return f"{n / 1e6:.2f}\\,M"
    if n < 1e8:
        return f"{n / 1e6:.1f}\\,M"
    return f"{n / 1e6:.0f}\\,M"


def _write(name: str, lines: list[str] | str) -> None:
    txt = lines if isinstance(lines, str) else "\n".join(lines) + "\n"
    txt = re.sub(r"(?<![$\w\\{])-(?=\d*\.\d+(?!\d*pt))", "$-$", txt)  # true minus in text mode, as make_tables.true_minus
    (OUT_TEX / name).write_text(txt)
    WRITTEN.append(name)


def load() -> dict:
    d = {n: pd.read_csv(OUT_CSV / f"{n}.csv") for n in
         ["metrics_cls", "metrics_seg", "cohort_means", "label_level", "ranks", "spread", "paired_cls", "paired_seg", "seeds",
          "label_ci", "params"]}
    for n in ("metrics_cls", "metrics_seg", "cohort_means", "paired_cls", "paired_seg", "seeds", "label_ci"):
        t = d[n]
        keep = ~t.cohort.isin(EXCLUDED_COHORTS)
        if "label" in t.columns:
            keep &= ~t.label.isin(EXCLUDED_LABELS)
        d[n] = t[keep].copy()
    d["dnt"] = {(r.backbone, r.method) for r in d["spread"].itertuples() if bool(r.did_not_train)}
    return d


def _cm(cm: pd.DataFrame, task: str) -> dict:
    t = cm[(cm.task == task) & (cm.seed == 42)]
    return {(r.backbone, r.cohort, r["mode"], r.method): (float(r.value), int(r.n_labels)) for _, r in t.iterrows()}


# ----------------------------------------------------------------------------------------------- per-cohort tables
def legacy_cls_ladder(D: dict, out: str, modes: tuple, compact: bool) -> None:
    look = _cm(D["cohort_means"], "cls")
    rows = []
    for ds in COHORTS:
        for bb in PER_BB:
            allv = {mo: {mt: look.get((bb, ds, mo, mt), (np.nan, 0))[0] for mt in METHODS} for mo in ("pertask", "joint", "joint_lm")}
            present = [mo for mo in modes if any(not _isnan(v) for v in allv[mo].values())]
            if not present:
                continue
            for mode in present:
                vals = allv[mode]
                n_labels = max(n for (b, c, mo, mt), (v, n) in look.items() if b == bb and c == ds and mo == mode)
                other = {"pertask": "joint", "joint": "pertask"}.get(mode)
                cells = []
                for mt in METHODS:
                    v = vals[mt]
                    if _isnan(v):
                        cells.append("--")
                        continue
                    c = _mark(v, list(vals.values()))
                    cells.append(_mode_shade(c, v, allv[other][mt]) if other else c)
                if compact:
                    lead = f"{SHORT_MAIN[ds]} ({n_labels}) & {BACKBONE_TEX[bb]}" if mode == present[0] else " & "
                    rows.append(f"{lead} & {MODE_TEX[mode]} & " + " & ".join(cells) + " \\\\")
                else:
                    rows.append(f"{DATASET_TEX[ds]} & {BACKBONE_TEX[bb]} & {MODE_TEX[mode]} ({n_labels}) & " + " & ".join(cells) + " \\\\")
            rows.append("\\addlinespace[2pt]")
    stack = {"babmil": ("LoRA+", "ABMIL"), "attnft": ("Attn-only", "FT"), "fullft": ("Full", "FT"),
             "lin_then_lorag": ("Lin$\\to$", "LoRA"), "attn_then_lorag": ("Attn$\\to$", "LoRA")}
    if compact:
        header = " & ".join(_two(*stack[mt]) if mt in stack else METHOD_TEX[mt] for mt in METHODS)
        head = f"Cohort (labels) & Backbone & Mode & {header} \\\\"
    else:
        head = "Cohort & Backbone & Mode (labels) & " + " & ".join(METHOD_TEX[mt] for mt in METHODS) + " \\\\"
    _write(out, [*(["\\setlength{\\tabcolsep}{3pt}"] if compact else []), "\\begin{tabular}{lll" + "c" * len(METHODS) + "}",
                 "\\toprule", head, "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


def legacy_seg_ladder(D: dict) -> None:
    look = _cm(D["cohort_means"], "seg")
    rows = []
    for ds in DATASETS:
        for bb in ["tapct", "spectre"] + ([] if NO_VOCO else ["voco"]):
            full = {mo: {mt: look[(bb, ds, mo, mt)][0] for mt in SEG_METHODS if (bb, ds, mo, mt) in look} for mo in ("pertask", "joint")}
            first = True
            for mode in ("pertask", "joint"):
                if not full[mode]:
                    continue
                other = {"pertask": "joint", "joint": "pertask"}[mode]
                cells = []
                for mt in SEG_METHODS:
                    if mt not in full[mode]:
                        cells.append("--")
                        continue
                    v = full[mode][mt]
                    cells.append(_mode_shade(_mark(v, list(full[mode].values())), v, full[other].get(mt, np.nan)))
                lead = f"{DATASET_TEX[ds]} ({SEG_N_STRUCT[ds]}) & {BACKBONE_TEX[bb]}" if first else " & "
                first = False
                mode_tex = "per-structure" if mode == "pertask" else MODE_TEX[mode]  # segmentation wording (E6)
                rows.append(f"{lead} & {mode_tex} & " + " & ".join(cells) + " \\\\")
            rows.append("\\addlinespace[2pt]")
    _write("tab_seg_ladder.tex", ["\\begin{tabular}{lll" + "c" * len(SEG_METHODS) + "}", "\\toprule",
                                  "Dataset (structures) & Backbone & Mode & " + " & ".join(SEG_METHOD_TEX[mt] for mt in SEG_METHODS) + " \\\\",
                                  "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


# ----------------------------------------------------------------------------------------------- tab_key, tab_cis
def legacy_key(D: dict) -> None:
    """Table 1: classification (left block) and segmentation (right block) side by side, one column pair per backbone.

    Each estimate is a stacked \\mci cell (mean over its interval); the joint-minus-per-task rows of both blocks share
    the last row, below a rule."""
    ll = D["label_level"].set_index(["key", "backbone"])
    short = {"seg_lora_dec_pertask": "\\quad per-structure", "seg_lora_dec_joint": "\\quad joint"}

    def cells(key: str, bb: str) -> str:
        if (key, bb) not in ll.index:
            return "-- & "
        r = ll.loc[(key, bb)]
        # dagger: the first method of the row did not train on this backbone (spread.csv; caption of Table 1)
        dag = "^{\\dagger}" if key.startswith("cls_") and (bb, key.split("_")[1]) in D["dnt"] else ""
        return f"\\mci{{${_f3s(r['mean'])}{dag}$}}{{${_f3s(r.lo3)}$}}{{${_f3s(r.hi3)}$}} & {int(r.clusters_pos)}/{int(r.clusters_n)}"

    blocks = {}
    for task in ("cls", "seg"):
        bbs = [bb for bb in PER_BB if any((k, bb) in ll.index for k, t, _ in KEY_ROWS if t == task)]
        rows = [(key, short.get(key, label)) for key, t, label in KEY_ROWS if t == task]
        blocks[task] = (bbs, [r for r in rows if not r[1].startswith("Joint")], [r for r in rows if r[1].startswith("Joint")])

    def line(task: str, row) -> str:
        bbs = blocks[task][0]
        if row is None:
            return " & " + " & ".join(" & " for _ in bbs)
        key, label = row
        return f"{label} & " + " & ".join(cells(key, bb) for bb in bbs)

    (cb, cm, cj), (sb, sm, sj) = blocks["cls"], blocks["seg"]
    nc, ns = 1 + 2 * len(cb), 1 + 2 * len(sb)
    body = []
    for i in range(max(len(cm), len(sm))):
        body.append(line("cls", cm[i] if i < len(cm) else None) + " & " + line("seg", sm[i] if i < len(sm) else None) + " \\\\")
    body.append(f"\\cmidrule(r){{1-{nc}}}\\cmidrule(l){{{nc + 1}-{nc + ns}}}")
    for i in range(max(len(cj), len(sj))):
        body.append(line("cls", cj[i] if i < len(cj) else None) + " & " + line("seg", sj[i] if i < len(sj) else None) + " \\\\")
    top = (f"\\multicolumn{{{nc}}}{{c}}{{\\textit{{Classification}} (AUROC)}} & "
           f"\\multicolumn{{{ns}}}{{c}}{{\\textit{{Segmentation}} (Dice)}} \\\\")
    toprule = f"\\cmidrule(r){{1-{nc}}}\\cmidrule(l){{{nc + 1}-{nc + ns}}}"
    head = (" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{BACKBONE_TEX[bb]}}}" for bb in cb) + " & & "
            + " & ".join(f"\\multicolumn{{2}}{{c}}{{{BACKBONE_TEX[bb]}}}" for bb in sb) + " \\\\")
    rule = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cb))) + \
        "".join(f"\\cmidrule(lr){{{nc + 2 + 2 * i}-{nc + 3 + 2 * i}}}" for i in range(len(sb)))
    sub = "Comparison" + " & Mean & $>0$" * len(cb) + " & Comparison" + " & Mean & $>0$" * len(sb) + " \\\\"
    spec = "@{}l" + "cc" * len(cb) + "@{\\hspace{10pt}}l" + "cc" * len(sb) + "@{}"
    _write("tab_key.tex", ["\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{" + spec + "}", "\\toprule", top, toprule, head, rule,
                           sub, "\\midrule", *body, "\\bottomrule", "\\end{tabular}"])


def legacy_cis(D: dict) -> None:
    ll = D["label_level"]
    rows = []
    rows_def = [r for r in KEY_ROWS if r[1] == "cls"] + CIS_EXTRA_ROWS + [r for r in KEY_ROWS if r[1] == "seg"]
    for key, task, label in rows_def:
        sub = ll[ll.key == key].set_index("backbone")
        for bb in ["all"] + PER_BB:
            if bb not in sub.index:
                continue
            r = sub.loc[bb]
            m = lambda v: f"${_f3s(v)}$"  # math mode, so negative values print with a true minus (E2)
            rows.append(f"{label} & {BACKBONE_TEX.get(bb, 'all')} & {int(r.n_labels)} & {m(r['mean'])} & [{m(r.lo2)}, {m(r.hi2)}] & "
                        f"[{m(r.lo3)}, {m(r.hi3)}] & {int(r.clusters_pos)}/{int(r.clusters_n)} \\\\")
        rows.append("\\addlinespace[2pt]")
    _write("tab_cis.tex", ["\\begin{tabular}{llrrccc}", "\\toprule", " & & & & \\multicolumn{2}{c}{95\\% interval, resampling} & \\\\",
                           "\\cmidrule(lr){5-6}", "Comparison & Backbone & $n$ & Mean & cohorts, labels & + cases & Cohorts $>0$ \\\\", "\\midrule",
                           *rows[:-1], "\\bottomrule", "\\end{tabular}"])


# ----------------------------------------------------------------------------------------------- paired, headline, methods
def _paired_summary(s: pd.DataFrame) -> tuple[int, float, float, float, float]:
    sig = s.p_bh < 0.05
    return (len(s), float(s["diff"].mean()), float((s["diff"] > 0).mean() * 100), float(((s["diff"] > 0) & sig).mean() * 100),
            float(((s["diff"] < 0) & sig).mean() * 100))


def legacy_cls_paired(D: dict) -> None:
    p = D["paired_cls"]
    p = p[p["mode"].isin(["pertask", "joint"])]
    rows = []
    for bb in MAIN_BB:
        for base in ["lin", "fullft"]:
            sub = p[(p.backbone == bb) & (p.baseline == base)]
            if sub.empty:
                continue
            first = True
            for mt in METHODS:
                s2 = sub[sub.method == mt]
                if s2.empty:
                    continue
                n, md, bt, sb, sw = _paired_summary(s2)
                lead = f"{BACKBONE_TEX[bb]} & vs {METHOD_TEX[base]}" if first else " & "
                first = False
                rows.append(f"{lead} & {METHOD_TEX[mt]} & {n} & {md:+.3f} & {bt:.0f} & {sb:.0f} & {sw:.0f} \\\\")
            rows.append("\\addlinespace[2pt]")
    _write("tab_cls_paired.tex", ["\\begin{tabular}{lllrrrrr}", "\\toprule",
                                  "Backbone & Baseline & Method & Label--mode pairs & Mean $\\Delta$AUROC & Better (\\%) & Sig.\\ better (\\%) & Sig.\\ worse (\\%) \\\\",
                                  "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


def legacy_seg_paired(D: dict) -> None:
    p = D["paired_seg"]
    p = p[p["mode"].isin(["pertask", "joint"])]
    rows = []
    for bb in ["tapct", "spectre"]:
        for base in ["lin", "dec"]:
            sub = p[(p.backbone == bb) & (p.baseline == base)]
            first = True
            for mt in SEG_METHODS:
                s2 = sub[sub.method == mt]
                if s2.empty:
                    continue
                n, md, bt, sb, sw = _paired_summary(s2)
                lead = f"{BACKBONE_TEX[bb]} & vs {SEG_METHOD_TEX[base]}" if first else " & "
                first = False
                rows.append(f"{lead} & {SEG_METHOD_TEX[mt]} & {n} & {md:+.3f} & {bt:.0f} & {sb:.0f} & {sw:.0f} \\\\")
            rows.append("\\addlinespace[2pt]")
    _write("tab_seg_paired.tex", ["\\begin{tabular}{lllrrrrr}", "\\toprule",
                                  "Backbone & Baseline & Method & Pairs & Mean $\\Delta$Dice & Better (\\%) & Sig.\\ better (\\%) & Sig.\\ worse (\\%) \\\\",
                                  "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


def legacy_headline(D: dict) -> None:
    m = D["metrics_cls"]
    d = m[(m.seed == 42) & (m.split == "test") & m["mode"].isin(["pertask", "joint"]) & m.method.isin(METHODS)]
    pooled = d.groupby(["backbone", "method"]).auroc.mean()
    p = D["paired_cls"]
    p = p[p["mode"].isin(["pertask", "joint"])]
    rows = []
    for bb in MAIN_BB:
        vs_lin = p[(p.backbone == bb) & (p.baseline == "lin") & (p.method == "lora")]
        vs_ft = p[(p.backbone == bb) & (p.baseline == "fullft") & (p.method == "lora")]
        n_cmp = int(((d.backbone == bb) & (d.method == "lin")).sum())
        cells = [f"{pooled.get((bb, mt), float('nan')):.3f}" for mt in ("lin", "lora", "attnft", "fullft")]
        better = float((vs_lin["diff"] > 0).mean() * 100)
        sig = float(((vs_lin["diff"] > 0) & (vs_lin.p_bh < 0.05)).mean() * 100)
        ft_better = float(((vs_ft["diff"] < 0) & (vs_ft.p_bh < 0.05)).mean() * 100)
        lora_better_ft = float((vs_ft["diff"] > 0).mean() * 100)
        dag = "$^{\\dagger}$" if (bb, "fullft") in D["dnt"] else ""   # Full FT did not train (as in Table 1)
        cells[-1] += dag
        rows.append(f"{BACKBONE_TEX[bb]} & {n_cmp} & " + " & ".join(cells)
                    + f" & {vs_lin['diff'].mean():+.3f} & {better:.0f} & {sig:.0f} & {vs_ft['diff'].mean():+.3f}{dag} & {lora_better_ft:.0f} & {ft_better:.0f} \\\\")
    _write("tab_headline.tex", [
        "\\begin{tabular}{lrccccrrrrrr}", "\\toprule",
        " & & \\multicolumn{4}{c}{Pooled AUROC} & \\multicolumn{3}{c}{LoRA vs.\\ linear probe} & \\multicolumn{3}{c}{LoRA vs.\\ full fine-tuning} \\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
        "Backbone & $n$ & Lin & LoRA & Attn-only FT & Full FT & Mean $\\Delta$ & LoRA better (\\%) & sig.\\ (\\%) & Mean $\\Delta$ & LoRA better (\\%) & FT sig.\\ better (\\%) \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


def legacy_methods(D: dict) -> None:
    rk = D["ranks"]
    rk = rk[rk.task == "cls"].set_index(["backbone", "method"])
    prm = D["params"].set_index(["backbone", "method"])
    # parameters that receive a gradient, as in Figure 6c: drop the Pillar-0 per-scale blocks that never run
    ng = pd.read_csv(V2 / "figures" / "data" / "no_grad_cls.csv").set_index(["backbone", "method"])["no_grad_backbone"]
    prm = prm.assign(trainable=[t - int(ng.get(k, 0)) for k, t in zip(prm.index, prm.trainable)])
    partial = {mt for _, mt in D["dnt"]}  # pooled values over fewer slots: shown, not marked
    full = [mt for mt in METHODS if mt not in partial]
    rows = []
    for mt in METHODS:
        cells = [_fmt_params(prm.loc[(bb, mt), "trainable"]) if (bb, mt) in prm.index else "--" for bb in MAIN_BB]
        for bb in MAIN_BB:
            v = rk.loc[(bb, mt), "mean_rank"]
            pool = [rk.loc[(bb, x), "mean_rank"] for x in METHODS]
            cells.append("--" if _isnan(v) else _mark(float(v), pool, higher=False, dec=2))
        for col, higher, dec, scale in (("mean_rank", False, 2, 1), ("within_001_pct", True, 0, 1), ("regret", False, 3, 1)):
            v = float(rk.loc[("all", mt), col])
            pool = [float(rk.loc[("all", x), col]) for x in full]
            cells.append(f"{v:.{dec}f}" if mt in partial else _mark(v, pool, higher=higher, dec=dec))
        rows.append(f"{METHOD_TEX[mt]} & " + " & ".join(cells) + " \\\\")
    _write("tab_methods.tex", [
        "\\begin{tabular}{lrrrccccrr}", "\\toprule",
        " & \\multicolumn{3}{c}{Trainable parameters} & \\multicolumn{4}{c}{Mean rank (1 = best)} & Within 0.01 & $\\Delta$ to best \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}",
        "Method & TAP-CT & SPECTRE & Pillar-0 & TAP-CT & SPECTRE & Pillar-0 & All & of best (\\%) & (AUROC) \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


# ----------------------------------------------------------------------------------------------- segmentation budget / modes
def legacy_seg_budget(D: dict) -> None:
    look = _cm(D["cohort_means"], "seg")
    rows = []
    for ds in ("amos22", "totalseg"):
        for bb in ("tapct", "spectre"):
            for mode, lab in (("pertask", "per-structure, 8000 steps"), ("joint_8k", "joint, 8000 steps"), ("joint", "joint, 24\\,000 steps")):
                cells = [f"{look[(bb, ds, mode, mt)][0]:.3f}" if (bb, ds, mode, mt) in look else "--" for mt in SEG_METHODS]
                rows.append(f"{DATASET_TEX[ds]} & {BACKBONE_TEX[bb]} & {lab} & " + " & ".join(cells) + " \\\\")
            rows.append("\\addlinespace[2pt]")
    _write("tab_seg_budget.tex", ["\\begin{tabular}{lll" + "c" * len(SEG_METHODS) + "}", "\\toprule",
                                  "Dataset & Backbone & Training & " + " & ".join(SEG_METHOD_TEX[mt] for mt in SEG_METHODS) + " \\\\",
                                  "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


def legacy_seg_modes(D: dict) -> None:
    m = D["metrics_seg"]
    d = m[(m.seed == 42) & (m.split == "test") & m.backbone.isin(["tapct", "spectre"]) & m["mode"].isin(["pertask", "joint", "joint_8k"])]
    sp = d.pivot_table(index=["backbone", "cohort", "mode", "label"], columns="method", values="dice")
    res = {mt: (sp[mt] - sp["dec"]).groupby(level=["backbone", "cohort", "mode"]).mean() for mt in ("lora", "fullft")}
    mode_tex = {"pertask": "per-structure", "joint": "joint", "joint_8k": "joint, 8000 steps"}
    rows = []
    for ds in DATASETS:
        first = True
        for mode in ("pertask", "joint", "joint_8k"):
            if ("tapct", ds, mode) not in res["lora"].index:
                continue
            cells = []
            for mt in ("lora", "fullft"):
                for bb in ("tapct", "spectre"):
                    v = res[mt].get((bb, ds, mode), np.nan)
                    cells.append("--" if _isnan(v) else f"${v:+.3f}$")
            lead = f"{DATASET_TEX[ds]} ({SEG_N_STRUCT[ds]})" if first else ""
            first = False
            rows.append(f"{lead} & {mode_tex[mode]} & " + " & ".join(cells) + " \\\\")
    _write("tab_seg_modes.tex", ["\\begin{tabular}{llrrrr}", "\\toprule",
                                 " & & \\multicolumn{2}{c}{LoRA $-$ Dec} & \\multicolumn{2}{c}{Full FT $-$ Dec} \\\\",
                                 "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "Dataset & Mode & TAP-CT & SPECTRE & TAP-CT & SPECTRE \\\\",
                                 "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])


# ----------------------------------------------------------------------------------------------- seeds and revision runs
def _seed_table(D: dict, task: str, methods: list[str], tex: dict, order: list, pairs: list, out: str) -> None:
    sd = D["seeds"]
    sd = sd[(sd.task == task) & sd.method.isin(methods)]
    cm = sd.groupby(["backbone", "cohort", "method"]).value.apply(list)
    m = D["metrics_cls" if task == "cls" else "metrics_seg"]
    col = "auroc" if task == "cls" else "dice"
    t = m[(m.split == "test") & (m["mode"] == "joint") & m.method.isin(methods) & m.backbone.isin(MAIN_BB)]
    lab = t.pivot_table(index=["backbone", "cohort", "seed", "label"], columns="method", values=col)
    sign = {(a, b): (lab[a] - lab[b]).dropna().groupby(level=["backbone", "cohort", "seed"]).mean() for a, b in pairs if a in lab and b in lab}
    rows = []
    for bb, ds in order:
        if not any((bb, ds, mt) in cm.index for mt in methods):
            continue
        cells = []
        for mt in methods:
            v = pd.Series(cm.get((bb, ds, mt), []), dtype=float)
            if len(v) == 0:
                cells.append("--")
            elif len(v) == 1:
                cells.append(f"{v.iloc[0]:.3f} (1)")
            else:
                cells.append(f"{v.mean():.3f} $\\pm$ {f'{v.std():.3f}'[1:]} ({len(v)})")
        for s in sign.values():
            x = s[(s.index.get_level_values(0) == bb) & (s.index.get_level_values(1) == ds)]
            cells.append(f"{int((x > 0).sum())}/{len(x)}" if len(x) else "--")
        rows.append((BACKBONE_TEX[bb], f"{SHORT[ds]} & " + " & ".join(cells) + " \\\\"))
    head = " & ".join([tex[mt] for mt in methods] + [_two(tex[a], f"$>$ {tex[b]}") for a, b in sign])
    _write(out, [f"\\begin{{tabular}}{{l{'r' * len(methods)}{'c' * len(sign)}}}", "\\toprule",
                 f"{'Cohort' if task == 'cls' else 'Dataset'} & {head} \\\\", "\\midrule", *_group(rows, 1 + len(methods) + len(sign)),
                 "\\bottomrule", "\\end{tabular}"])


def _joint_lab(m: pd.DataFrame, methods: list[str], col: str) -> pd.DataFrame:
    """held-out label values of the seed-42 joint models: (backbone, cohort, label) x method"""
    d = m[(m.seed == 42) & (m.split == "test") & (m["mode"] == "joint") & m.method.isin(methods)]
    return d.pivot_table(index=["backbone", "cohort", "label"], columns="method", values=col)


def legacy_fullft_fix(D: dict) -> None:
    m = D["metrics_cls"]
    cols = ["lin", "lora", "fullft", *FIXES]
    lab = _joint_lab(m, cols, "auroc")
    t = m[(m.seed == 42) & (m["mode"] == "joint") & m.method.isin(["fullft", *FIXES])]
    # run-level spread: median over the run's labels of the SD of its held-out probabilities
    spread = t[t.split == "test"].groupby(["backbone", "cohort", "method"]).pred_sd.median().to_dict()
    val = t[(t.split == "val") & t.method.isin(FIXES)]
    vmean = val.groupby(["backbone", "method"]).auroc.mean()
    vcoh = val.groupby(["backbone", "method"]).cohort.nunique()
    chosen = {}
    for bb in MAIN_BB:
        n_coh = lab.xs(bb, level="backbone").index.get_level_values("cohort").nunique() if bb in lab.index.get_level_values(0) else 0
        ready = [f for f in FIXES if vcoh.get((bb, f), 0) == n_coh]
        if n_coh and len(ready) == len(FIXES):
            chosen[bb] = max(ready, key=lambda f: vmean[(bb, f)])
    rows = []
    for bb, ds in CLS_ORDER:
        x = lab.xs((bb, ds), level=["backbone", "cohort"]).mean()
        cells = []
        for mt in cols:
            v = x.get(mt, np.nan)
            if _isnan(v):
                cells.append("--")
                continue
            s = f"{v:.3f}"
            if spread.get((bb, ds, mt), 1.0) < SPREAD_MIN:
                s = f"\\textcolor{{gray}}{{{s}}}"
            if chosen.get(bb) == mt:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        rows.append((BACKBONE_TEX[bb], f"{SHORT[ds]} & " + " & ".join(cells) + " \\\\"))
    head = " & ".join(["Lin", "LoRA"] + [FIX_TEX[k] for k in ["fullft", *FIXES]])
    _write("tab_fullft_fix.tex", ["\\begin{tabular}{lrrrrrr}", "\\toprule", " & & & \\multicolumn{4}{c}{Full FT} \\\\", "\\cmidrule(lr){4-7}",
                                  f"Cohort & {head} \\\\", "\\midrule", *_group(rows, 7), "\\bottomrule", "\\end{tabular}"])


def legacy_controls(D: dict) -> None:
    m = D["metrics_cls"]
    cols = ["lora", "rslora", "rslora_lowlr", "attnft", "mlpft"]
    lab = _joint_lab(m, cols, "auroc")
    tex = {**METHOD_TEX, "rslora_lowlr": "rsLoRA (LR$/4$)", "mlpft": "MLP-only FT"}
    have = set(zip(*[m[m.method.isin(["mlpft", "rslora_lowlr"])][c] for c in ("backbone", "cohort")]))
    rows = []
    for bb, ds in CLS_ORDER:
        if (bb, ds) not in have:
            continue
        x = lab.xs((bb, ds), level=["backbone", "cohort"]).mean()
        rows.append((BACKBONE_TEX[bb], f"{SHORT[ds]} & " + " & ".join("--" if _isnan(x.get(c, np.nan)) else f"{x[c]:.3f}" for c in cols) + " \\\\"))
    _write("tab_controls.tex", ["\\begin{tabular}{lrrrrr}", "\\toprule", "Cohort & " + " & ".join(tex[c] for c in cols) + " \\\\", "\\midrule",
                                *_group(rows, 1 + len(cols)), "\\bottomrule", "\\end{tabular}"])


def legacy_seg_cm(D: dict) -> None:
    m = D["metrics_seg"]
    d = m[(m.seed == 42) & (m.split == "test")]
    rows = []
    for ds in ("amos22", "totalseg"):
        for bb in ("tapct", "spectre"):
            for mt in ("dec", "lora", "fullft"):
                g = d[(d.backbone == bb) & (d.cohort == ds) & (d.method == mt) & (d["mode"] == "pertask_cm")].set_index("label").dice
                if g.empty:
                    continue
                rr = d[(d.backbone == bb) & (d.cohort == ds) & (d.method == mt) & d["mode"].isin(["pertask", "joint"])].pivot_table(
                    index="label", columns="mode", values="dice").reindex(g.index)
                diff = rr["joint"] - g
                rows.append((f"{DATASET_TEX[ds]} ({SEG_N_STRUCT[ds]} structures)",
                             f"{BACKBONE_TEX[bb]} & {SEG_METHOD_TEX[mt]} & {len(g)} & {g.mean():.3f} & {rr['pertask'].mean():.3f} & "
                             f"{rr['joint'].mean():.3f} & ${diff.mean():+.3f}$ & {int((diff > 0).sum())}/{int(diff.notna().sum())} \\\\"))
    _write("tab_seg_cm.tex", ["\\begin{tabular}{llrrrrrc}", "\\toprule",
                              " & & & \\multicolumn{2}{c}{per-structure} & & \\multicolumn{2}{c}{joint $-$ matched} \\\\", "\\cmidrule(lr){4-5}\\cmidrule(lr){7-8}",
                              f"Backbone & Method & {_two('Struct.', 'finished')} & matched & 8000 steps & joint & mean & $>0$ \\\\", "\\midrule",
                              *_group(rows, 8), "\\bottomrule", "\\end{tabular}"])


# ----------------------------------------------------------------------------------------------- per-label appendix
def _block(title: str, path: str) -> str:
    return (f"\\begin{{center}}\\begin{{minipage}}{{\\textwidth}}\\centering\\footnotesize\\textit{{{title}}}\\\\[2pt]"
            f"\\adjustbox{{max width=\\textwidth}}{{\\input{{tables/{path}}}}}\\end{{minipage}}\\end{{center}}")


def legacy_app_cls(D: dict) -> None:
    ci = D["label_ci"]
    d = ci[(ci.task == "cls") & (ci.seed == 42) & ci.method.isin(METHODS)]
    parts = []
    for ds in COHORTS:
        sub = d[d.cohort == ds]
        blocks = []
        for bb in MAIN_BB:
            s1 = sub[sub.backbone == bb]
            if s1.empty:
                continue
            rows = []
            for mode in ["pertask", "joint", "joint_lm"]:
                s2 = s1[s1["mode"] == mode]
                if s2.empty:
                    continue
                rows.append(f"\\multicolumn{{{len(METHODS) + 1}}}{{l}}{{\\textit{{{BACKBONE_TEX[bb]}, {MODE_TEX[mode]}}}}} \\\\")
                for lab in sorted(s2.label.unique()):
                    s3 = s2[s2.label == lab].set_index("method")
                    cells = [f"{s3.loc[mt].value:.2f} [{s3.loc[mt].ci_lo:.2f},{s3.loc[mt].ci_hi:.2f}]" if mt in s3.index else "--" for mt in METHODS]
                    rows.append(f"{_esc(lab.split(':')[-1])} & " + " & ".join(cells) + " \\\\")
            blocks.append((bb, rows))
        if sum(len(r) for _, r in blocks) <= 36:
            groups = [blocks]
        else:
            groups = []
            for bb, rows in blocks:
                if len(rows) <= 36:
                    groups.append([(bb, rows)])
                    continue
                starts = [i for i, r in enumerate(rows) if r.startswith("\\multicolumn")] + [len(rows)]
                for a, b in zip(starts[:-1], starts[1:]):
                    groups.append([(bb, rows[a:b])])
        for gi, grp in enumerate(groups):
            rows = [r for _, rr in grp for r in rr]
            tag = ds if len(groups) == 1 else f"{ds}_{grp[0][0]}_{gi}"
            _write(f"tab_app_{tag}.tex", ["\\begin{tabular}{l" + "c" * len(METHODS) + "}", "\\toprule",
                                          "Label & " + " & ".join(METHOD_TEX[mt] for mt in METHODS) + " \\\\", "\\midrule", *rows,
                                          "\\bottomrule", "\\end{tabular}"])
            title = DATASET_TEX[ds] if len(groups) == 1 else f"{DATASET_TEX[ds]}, {BACKBONE_TEX[grp[0][0]]}"
            parts.append(_block(title, f"tab_app_{tag}.tex"))
    _write("tab_app_all.tex", parts)


def legacy_app_seg(D: dict) -> None:
    ci = D["label_ci"]
    d = ci[(ci.task == "seg") & (ci.seed == 42) & ci["mode"].isin(["pertask", "joint"])]
    parts = []
    for ds in DATASETS:
        rows = []
        for bb in ["tapct", "spectre"]:
            for mode in ["pertask", "joint"]:
                s2 = d[(d.cohort == ds) & (d.backbone == bb) & (d["mode"] == mode)]
                if s2.empty:
                    continue
                rows.append(f"\\multicolumn{{{len(SEG_METHODS) + 1}}}{{l}}{{\\textit{{{BACKBONE_TEX[bb]}, {'per-structure' if mode == 'pertask' else 'joint'}}}}} \\\\")
                for lab in sorted(s2.label.unique()):
                    s3 = s2[s2.label == lab].set_index("method")
                    cells = [f"{s3.loc[mt].value:.2f} [{s3.loc[mt].ci_lo:.2f},{s3.loc[mt].ci_hi:.2f}]" if mt in s3.index else "--" for mt in SEG_METHODS]
                    rows.append(f"{_esc(lab)} & " + " & ".join(cells) + " \\\\")
        _write(f"tab_seg_app_{ds}.tex", ["\\begin{tabular}{l" + "c" * len(SEG_METHODS) + "}", "\\toprule",
                                         "Structure & " + " & ".join(SEG_METHOD_TEX[mt] for mt in SEG_METHODS) + " \\\\", "\\midrule", *rows,
                                         "\\bottomrule", "\\end{tabular}"])
        parts.append(_block(DATASET_TEX[ds], f"tab_seg_app_{ds}.tex"))
    _write("tab_seg_app_all.tex", parts)


def legacy_cohorts() -> None:
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Cohort & Labels & Train & Selection & Held-out \\\\", "\\midrule"]
    cvpr = [COHORT_SPLITS[k] for k in ("amos", "covid", "luna")]
    tot = [sum(c[i] for c in cvpr) for i in (1, 2, 3, 4)]
    lines.append(f"\\textbf{{CVPR26}} (CVPR 2026 CT foundation-model challenge) & \\textbf{{{tot[0]}}} & \\textbf{{{tot[1]}}} & "
                 f"\\textbf{{{tot[2]}}} & \\textbf{{{tot[3]}}} \\\\")
    for name, n, tr, va, te in cvpr:
        lines.append(f"\\quad {name} & {n} & {tr} & {va} & {te} \\\\")
    for k in ("fdg", "rsna2022", "rsna2023", "ich"):
        name, n, tr, va, te = COHORT_SPLITS[k]
        lines.append(f"{name} & {n} & {tr} & {va} & {te} \\\\")
    _write("tab_cohorts.tex", lines + ["\\bottomrule", "\\end{tabular}"])


LABEL_TEX = {"covid_ct:covid": "COVID-19 (COVID-CTset)", "luna25:lung_nodule_malignancy": "nodule malignancy (LUNA25)",
             "patient_overall": "any fracture (patient)"}


def legacy_label_counts() -> None:
    """tab_label_counts: selection and held-out cases and positives per classification label (metrics_cls.csv; the counts
    are the same in every run)."""
    import pandas as pd
    m = pd.read_csv(OUT_CSV / "metrics_cls.csv")
    m = m[(m.backbone != "voco") & (m.cohort != "breastdivider") & (m.seed == 42)]
    g = m.groupby(["cohort", "label", "split"])[["n", "n_pos"]]
    assert (g.nunique() == 1).all().all(), "case counts differ between runs"
    c = g.first().unstack("split")
    rows = []
    for ds in ("amos_covid_luna", "fdg", "rsna2022", "rsna2023", "rsna2019_ich"):
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{DATASET_TEX[ds]}}}}} \\\\")
        for lab in c.loc[ds].index:
            raw = lab.split(":")[-1]
            name = LABEL_TEX.get(lab, raw if raw[:1] == "C" and raw[1:].isdigit() else raw.replace("_", " ").lower())
            r = c.loc[(ds, lab)]
            rows.append(f"{_esc(name)} & {int(r[('n', 'val')])} & {int(r[('n_pos', 'val')])} & {int(r[('n', 'test')])} & {int(r[('n_pos', 'test')])} \\\\")
    _write("tab_label_counts.tex", ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
                                    " & \\multicolumn{2}{c}{Selection} & \\multicolumn{2}{c}{Held-out} \\\\",
                                    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
                                    "Label & Cases & Positives & Cases & Positives \\\\", "\\midrule", *rows,
                                    "\\bottomrule", "\\end{tabular}"])


# ----------------------------------------------------------------------------------------------- main
def legacy_spread(D: dict) -> None:
    """tab_spread: median over label-mode pairs of the SD of the held-out probabilities (spread.csv); bold + dagger = the
    pair that did not train."""
    sp = D["spread"].set_index(["backbone", "method"])
    rows = []
    for bb in PER_BB:
        cells = []
        for mt in METHODS:
            r = sp.loc[(bb, mt)]
            c = f"{float(r.median_sd):.3f}"
            cells.append(f"\\textbf{{{c}}}$^{{\\dagger}}$" if bool(r.did_not_train) else c)
        rows.append(f"{BACKBONE_TEX[bb]} & " + " & ".join(cells) + " \\\\")
    _write("tab_spread.tex", ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "c" * len(METHODS) + "}", "\\toprule",
                              "Backbone & " + " & ".join(METHOD_TEX[mt] for mt in METHODS) + " \\\\", "\\midrule", *rows,
                              "\\bottomrule", "\\end{tabular}"])


def write_all() -> list[str]:
    WRITTEN.clear()
    D = load()
    legacy_key(D)
    legacy_cls_ladder(D, "tab_cls_main.tex", ("pertask", "joint"), compact=True)
    legacy_cls_ladder(D, "tab_cls_ladder.tex", ("pertask", "joint", "joint_lm"), compact=False)
    legacy_seg_ladder(D)
    legacy_methods(D)
    legacy_headline(D)
    legacy_cis(D)
    legacy_cls_paired(D)
    legacy_seg_paired(D)
    legacy_seg_budget(D)
    legacy_seg_modes(D)
    _seed_table(D, "cls", ["lin", "lora", "attnft", "fullft"], METHOD_TEX, CLS_ORDER, [("lora", "lin"), ("attnft", "lora")], "tab_seeds_cls.tex")
    _seed_table(D, "seg", ["dec", "lora", "fullft"], SEG_METHOD_TEX, SEG_ORDER, [("lora", "dec"), ("fullft", "dec")], "tab_seeds_seg.tex")
    legacy_fullft_fix(D)
    legacy_controls(D)
    legacy_seg_cm(D)
    legacy_app_cls(D)
    legacy_app_seg(D)
    legacy_cohorts()
    legacy_label_counts()
    legacy_spread(D)
    return list(WRITTEN)


if __name__ == "__main__":
    names = write_all()
    print(f"paper_tables: wrote {len(names)} files to {OUT_TEX}: {' '.join(names)}")
