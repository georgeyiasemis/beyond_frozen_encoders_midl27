#!/usr/bin/env python3
"""Results figures of the paper in the designs of the first draft, regenerated from the recomputed metrics.

    ~/miniconda3/envs/aifofomo/bin/python paper_figures.py

Inputs (read-only):
  * ``../metrics/out/*.csv`` and ``../metrics/out/numbers.tex`` (written by ``metrics/recompute.py``);
  * the training logs of the runs listed in ``../metrics/out/runs.csv`` (``<path>/logs/metrics.json`` for the
    validation curves, ``<path>/logs/log_0_0.txt`` for the logged parameter counts), under
    ``midl_paper/results/`` (the paths in runs.csv are relative to that folder).

VoCo is quarantined: every figure is drawn without the VoCo backbone into ``out/`` (``WITH_VOCO=1`` restores the VoCo
variant once its runs are back in the inventory) and copied to ``../paper/figures/`` (style.py; ``PAPER_DIR`` / ``FIG_OUT`` override);
the rank panels of fig_overview then use the main-backbone ranks (\\numRank*, \\numSegrank*).

Outputs: ``out/<name>.pdf|png`` for fig_overview, fig_key, fig_mechanisms, fig_seg, fig_bbgap, fig_joint, fig_methods,
fig_seg_methods, fig_seg_budget, fig_seg_convergence; the PDFs are copied to ``../paper/figures/``;
``../paper/tables/figure_numbers.tex`` (macros read off the validation curves); ``out/paper_figures_checks.txt``
(every printed number compared with metrics/out).

Conventions (metrics/PROCEDURE.md): held-out split, seed 42, modes per-task and joint (segmentation "joint" of
AMOS22 / TotalSegmentator = the 24 000-step run, "joint_8k" = the 8000-step run); VoCo is drawn per backbone and
never pooled; SPECTRE Full FT (classification) did not train (spread.csv) and leaves every average over methods,
the ranks and the summary panels.
"""
from __future__ import annotations

import os
import json
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

import style
from style import OI, BACKBONE_COL, BACKBONE_MARK, BACKBONE_TEX, INK, INK2, GRID, TEXTWIDTH_IN

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS = HERE.parent / "metrics" / "out"
OUT = style.FIG_OUT                                      # figures/out
PAPER_FIG = style.PAPER_DIR / "figures"                  # ../paper/figures
TABLES = style.PAPER_DIR / "tables"
NO_VOCO = style.NO_VOCO                                  # True unless WITH_VOCO=1: every figure without the VoCo backbone
VOCO = [] if NO_VOCO else ["voco"]
RESULTS = Path(os.environ.get("BFE_RESULTS") or "/projects/cvpr2026_challenge_fomo4ct/beyond_frozen_encoders")   # root of the run paths in runs.csv (training logs only); BFE_RESULTS overrides

W = TEXTWIDTH_IN                                         # every figure is drawn at the printed width (6.0 in)
FONT_FLOOR = 7.0                                         # no text below 7 pt at print size
MAIN = ["tapct", "spectre", "pillar0"]
CLS_COHORTS = ["amos_covid_luna", "fdg", "rsna2022", "rsna2023", "rsna2019_ich"]
SEG_DATASETS = ["amos22", "kits23", "lits", "msd_pancreas", "totalseg"]
CLS_METHODS = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
LONG_JOINT = {"amos22", "totalseg"}                      # 15-structure joint models: "joint" is the 24 000-step run
SEG_STEPS_PER_ITER = 0.5                                 # segmentation: accumulation 2, the engine counts micro-batches

# Method colours and markers: one palette for every figure (style.METHOD_COL: greys and purples, none of the backbone
# hues, no yellow), so that a hue never stands for a method in one figure and a backbone in another.
MCOL = {m: style.METHOD_COL[m] for m in ("lin", "attn", "dec", "lora", "babmil", "attnft", "fullft", "dora")}
MEDGE = {m: ("#7a7a7a" if m in ("attn", "babmil", "dora") else "white") for m in MCOL}   # light fills need a rim on white
MMARK = dict(style.METHOD_MARK)
MTEX = {"lin": "Lin", "attn": "Attn", "lora": "LoRA", "lorag": r"LoRA$_\mathrm{late}$", "babmil": "LoRA+ABMIL",
        "dora": "DoRA", "rslora": "rsLoRA", "lin_then_lorag": "Lin\u2192LoRA", "attn_then_lorag": "Attn\u2192LoRA",
        "attnft": "Attn-only FT", "fullft": "Full FT", "dec": "Dec", "dec_then_lora": "Dec\u2192LoRA"}
COHORT_SHORT = {"amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2022": "RSNA22 C-spine",
                "rsna2023": "RSNA23 trauma", "rsna2019_ich": "RSNA19 ICH"}
DS_SHORT = {"amos22": "AMOS22", "kits23": "KiTS23", "lits": "LiTS", "msd_pancreas": "MSD Pancreas", "totalseg": "TotalSeg."}
BAND = "#f4f3ef"

_savefig = matplotlib.figure.Figure.savefig


def _savefig_legible(self, *a, **k):
    for t in self.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_fontsize() < FONT_FLOOR:
            t.set_fontsize(FONT_FLOOR)
    return _savefig(self, *a, **k)


matplotlib.figure.Figure.savefig = _savefig_legible


# ============================================================================ data layer
@lru_cache(None)
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS / f"{name}.csv")


@lru_cache(None)
def macros() -> dict[str, str]:
    txt = (METRICS / "numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt))


def dnt() -> set[tuple[str, str]]:
    """(backbone, method) classification pairs that did not train (spread.csv; applied to the main backbones)."""
    sp = load("spread")
    sp = sp[sp.backbone.isin(MAIN) & (sp.did_not_train.astype(str).str.lower() == "true")]
    return set(zip(sp.backbone, sp.method))


def cls_test(drop_dnt: bool = True) -> pd.DataFrame:
    d = load("metrics_cls")
    d = d[(d.split == "test") & (d.seed == 42) & d["mode"].isin(["pertask", "joint"]) & d.cohort.isin(CLS_COHORTS)
          & d.method.isin(CLS_METHODS)]
    if drop_dnt:
        bad = dnt()
        d = d[[(b, m) not in bad for b, m in zip(d.backbone, d.method)]]
    return d


def seg_split(split: str = "test") -> pd.DataFrame:
    d = load("metrics_seg")
    return d[(d.split == split) & (d.seed == 42) & d["mode"].isin(["pertask", "joint"]) & d.method.isin(SEG_METHODS)]


def paired_cls(baseline: str = "lin", drop_dnt: bool = False) -> pd.DataFrame:
    p = load("paired_cls")
    p = p[(p.baseline == baseline) & p["mode"].isin(["pertask", "joint"]) & p.cohort.isin(CLS_COHORTS) & p.method.isin(CLS_METHODS)]
    if drop_dnt:
        bad = dnt()
        p = p[[(b, m) not in bad for b, m in zip(p.backbone, p.method)]]
    return p


def runs(task: str, **kw) -> pd.DataFrame:
    r = load("runs")
    r = r[(r.task == task) & (r.seed == kw.pop("seed", 42))]
    for k, v in kw.items():
        r = r[r[k].isin(v if isinstance(v, (list, set, tuple)) else [v])]
    return r


@lru_cache(None)
def val_curve(run_path: str, key: str) -> dict[int, float]:
    """Validation metric per evaluation from a run's logs/metrics.json (the reader of the draft's make_figures.py)."""
    f = RESULTS / run_path / "logs" / "metrics.json"
    vals: dict[int, float] = {}
    if f.is_file():
        tag = f'"val/metric/{key}"'
        for line in open(f):
            if tag in line:
                r = json.loads(line)
                vals[int(r["iteration"])] = float(r[f"val/metric/{key}"])
    return vals


_PARAM_RE = re.compile(r"(\w+Model) \[[^\]]*\]: (\d+) / (\d+) trainable \([^)]*\); backbone(?:\+combiner)? (\d+) / (\d+)")


@lru_cache(None)
def logged_params(task: str, bb: str, method: str) -> tuple[int, int, str]:
    """(encoder parameters trained, encoder parameters in total, model class) from the training log of the first run
    of (backbone, method) in runs.csv (classification: CT cohorts) that has the line."""
    r = runs(task, backbone=bb, method=method)
    if task == "cls":  # the CT cohorts only (the breast MRI cohort loads Pillar-0's breast weights)
        r = r[r.cohort.isin(CLS_COHORTS)]
    for p in r.path:
        f = RESULTS / p / "logs" / "log_0_0.txt"
        if not f.is_file():
            continue
        with open(f, errors="ignore") as fh:
            for line in fh:
                m = _PARAM_RE.search(line)
                if m:
                    return int(m.group(4)), int(m.group(5)), m.group(1)
    raise RuntimeError(f"no parameter line for {task} {bb} {method}")


DIAG_DATA = HERE / "data"   # written by figures/diagrams.py (gradient check of the method diagrams; see DIAGRAMS.md)


@lru_cache(None)
def no_grad_cls() -> dict[tuple[str, str], int]:
    """Encoder parameters that are marked trainable but never receive a gradient (Pillar-0: per-scale sub-blocks that
    never run and the text projection), from the gradient check of diagrams.py; 0 for every other pair."""
    d = pd.read_csv(DIAG_DATA / "no_grad_cls.csv")
    return {(b, m): int(v) for b, m, v in zip(d.backbone, d.method, d.no_grad_backbone)}


def updated_params(bb: str, method: str) -> int:
    """Encoder parameters updated (with a gradient) = logged trainable encoder parameters - those without a gradient;
    the same counts as fig_arms (d)."""
    return logged_params("cls", bb, method)[0] - no_grad_cls().get((bb, method), 0)


# ============================================================================ checks
CHECKS: list[str] = []


def fmt_signed(v: float) -> str:
    s = f"{v:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s


def um(s: str) -> str:
    """Printed numbers use the true minus sign (U+2212); the checks compare the ASCII form."""
    return s.replace("-", "\u2212")


def check(fig: str, what: str, value: str, macro: str | None = None, ref: str | None = None) -> None:
    """Compare a printed / plotted value (as printed) with a macro of numbers.tex or with a reference value."""
    if macro is not None:
        ref = macros().get(macro)
        src = f"\\{macro}"
    else:
        src = "metrics/out"
    status = "OK " if ref is not None and str(ref) == str(value) else ("MISSING" if ref is None else "DIFF")
    CHECKS.append(f"{status:7s} {fig:20s} {what:55s} figure={value:>9s}  {src}={ref}")


def write_checks() -> None:
    n_ok = sum(c.startswith("OK") for c in CHECKS)
    n_bad = sum(c.startswith("DIFF") for c in CHECKS)
    n_miss = sum(c.startswith("MISSING") for c in CHECKS)
    head = f"paper_figures.py number check: {len(CHECKS)} comparisons, {n_ok} equal, {n_bad} different, {n_miss} without a reference\n"
    (OUT / "paper_figures_checks.txt").write_text(head + "\n".join(CHECKS) + "\n")
    print(head.strip())
    for c in CHECKS:
        if c.startswith(("DIFF", "MISSING")):
            print("  ", c)


def mname(*parts: str) -> str:
    """Macro name of numbers.tex for an underscore key (metrics/NAMING.md)."""
    key = "_".join(parts)
    s = "".join(w.title() for w in re.split(r"[^0-9A-Za-z]+", key) if w)
    return "num" + s.translate(str.maketrans("0123456789", "ZABCDEFGHI"))


# ============================================================================ drawing helpers
def save(fig, name: str) -> None:
    style.save(fig, OUT, name)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / f"{name}.pdf", PAPER_FIG / f"{name}.pdf")
    print("wrote", name)


def mscatter(ax, x, y, mt, s=16, **kw):
    ax.scatter(x, y, s=s, color=MCOL[mt], marker=MMARK.get(mt, "o"), edgecolor=MEDGE[mt], linewidths=0.6, zorder=3, **kw)


def mhandle(mt, ms=4.2, line=False, marker=None, **kw):
    return Line2D([], [], color=MCOL[mt], marker=marker or MMARK.get(mt, "o"), ms=ms, mec=MEDGE[mt] if MEDGE[mt] != "white" else MCOL[mt],
                  mew=0.6, ls="-" if line else "", lw=1.4, label=kw.pop("label", MTEX[mt]), **kw)


def tidy(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.tick_params(length=2, labelsize=6.3)


# ============================================================================ fig_overview
GROUPS = [("frozen", ["lin", "attn"]), ("LoRA\nlate\nblocks", ["lorag", "lin_then_lorag", "attn_then_lorag"]),
          ("LoRA\nall\nblocks", ["lora", "babmil", "dora", "rslora"]), ("Attn-\nonly\nFT", ["attnft"]), ("Full\nFT", ["fullft"])]
BB_LS = {"tapct": "-", "spectre": "-", "pillar0": ":", "voco": "--"}      # dotted / dashed: backbones with 1 / 3 cohorts


def _fmt_p(n: float) -> str:
    """Millions of parameters: two significant digits below 1 M (0.2, 0.55), one decimal below 10 M, none above."""
    m = n / 1e6
    return f"{m:.2g}" if m < 1 else (f"{m:.1f}" if m < 10 else f"{m:.0f}")


def _fmt_m(n: float) -> str:
    return f"{n / 1e6:.0f} M"


def pooled_ranks(rk: pd.DataFrame, bbs: list[str]) -> pd.Series:
    """Mean rank over the slots of the given backbones = per-backbone mean ranks weighted by their slot counts
    (each slot is ranked within its own backbone, ranks.csv)."""
    r = rk[rk.backbone.isin(bbs) & (rk.n_slots > 0)]
    return r.groupby("method").apply(lambda g: float((g.mean_rank * g.n_slots).sum() / g.n_slots.sum()), include_groups=False)


def rank_panel(ax, rk: pd.DataFrame, bbs: list[str], fam: str, xmax: int, panel: str) -> None:
    """Mean rank per method: small markers per backbone (VoCo included), diamond pooled over every backbone's slots."""
    allr = pooled_ranks(rk, bbs)
    nslots = int(rk[rk.backbone.isin(bbs)].groupby("backbone").n_slots.max().sum())
    check("fig_overview", f"({panel}) slots pooled over {'+'.join(bbs)}", str(nslots), mname(fam, "slots", "n"))
    order = allr.sort_values().index.tolist()
    per = rk[rk.backbone.isin(bbs)].pivot(index="method", columns="backbone", values="mean_rank")
    pre_bb = {"rankall": "rank", "segrankall": "segrank", "rank": "rank", "segrank": "segrank"}[fam]
    for i, mt in enumerate(order):
        y = len(order) - 1 - i
        for bb in bbs:
            v = per.loc[mt, bb]
            if np.isnan(v):
                continue
            ax.scatter([v], [y], s=10, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], alpha=0.85, linewidths=0, zorder=3)
            mac = mname("voco" + pre_bb, mt, "mean") if bb == "voco" else mname(pre_bb, bb, mt, "mean")
            check("fig_overview", f"({panel}) mean rank {bb} {mt}", f"{v:.2f}", mac)
        ax.scatter([allr[mt]], [y], s=48, color=INK, marker="|", linewidths=1.8, zorder=4)   # pooled: a bar (VoCo owns the diamond)
        check("fig_overview", f"({panel}) mean rank {mt}, all backbones", f"{allr[mt]:.2f}", mname(fam, mt, "mean"))
    ax.set_yticks(range(len(order))); ax.set_yticklabels([MTEX[mt] for mt in order[::-1]], fontsize=6.2)
    ax.set_xlim(1, xmax); ax.set_xticks(list(range(1, xmax + 1, 2)))
    ax.set_xlabel("mean rank (1 = best)", fontsize=6.4)
    tidy(ax)


def fig_overview() -> None:
    d = cls_test()
    pooled = d.groupby(["backbone", "method"]).auroc.mean()
    for (bb, mt), v in pooled.items():
        if bb not in MAIN:
            continue
        check("fig_overview", f"(b) pooled AUROC {bb} {mt}", f"{v:.3f}", f"pooled{bb.replace('0', 'Z')}{mt.replace('_', '')}")
    # VoCo (no pooled macro): its pooled differences to Lin vs the label-level VoCo - Lin macros (every VoCo label has both modes)
    vl = d[d.backbone == "voco"].pivot_table(index=["cohort", "label", "mode"], columns="method", values="auroc")
    for mt in CLS_METHODS:
        if mt != "lin" and not NO_VOCO:
            full = vl[[mt, "lin"]].dropna()
            ref_ok = len(full) == len(vl)
            check("fig_overview", f"(b) VoCo pooled {mt} - Lin (label-mode pairs {len(full)}/{len(vl)})",
                  fmt_signed(pooled[("voco", mt)] - pooled[("voco", "lin")]), mname("vocovslin", mt, "meandiff") if ref_ok else None,
                  ref=None if ref_ok else "n/a")
    rk = load("ranks")
    rcls, rseg = rk[rk.task == "cls"], rk[rk.task == "seg"]
    # design counts (from the CSVs)
    allcls = cls_test(drop_dnt=False)
    n_labels = allcls.drop_duplicates(["cohort", "label"]).shape[0]
    n_cohorts = allcls.cohort.nunique()
    sg = seg_split()
    n_struct = sg[sg["mode"] == "pertask"].drop_duplicates(["cohort", "label"]).shape[0]
    n_segds = sg.cohort.nunique()
    n_cls_m, n_seg_m = allcls.method.nunique(), sg.method.nunique()
    for what, v, mac in (("labels", n_labels, "numLabels"), ("structures", n_struct, "numStructsSegN"),
                         ("classification methods", n_cls_m, None), ("segmentation methods", n_seg_m, None)):
        if mac:
            check("fig_overview", f"(a) {what}", str(v), mac)
    dn = dict(re.findall(r"\\providecommand\{\\(\w+)\}\{([^}]*)\}", (TABLES / "design_numbers.tex").read_text()))
    for what, v, mac in (("labels", n_labels, "numLabelsN"), ("structures", n_struct, "numStructuresN"),
                         ("classification methods", n_cls_m, "numClsMethodsN"), ("segmentation methods", n_seg_m, "numSegMethodsN")):
        check("fig_overview", f"(a) {what} vs design_numbers.tex", str(v), ref=dn.get(mac))
    check("fig_overview", "(a) classification cohorts", str(n_cohorts), ref=str(len(CLS_COHORTS)))
    check("fig_overview", "(a) segmentation datasets", str(n_segds), ref=str(len(SEG_DATASETS)))
    tot, cls_name = {}, {}
    for bb in MAIN + VOCO:
        _, tot[bb], cls_name[bb] = logged_params("cls", bb, "lin")
    # Pillar-0: the logged encoder total includes the CLIP text projection, which never runs; (a) gives the vision
    # encoder (Atlas with its stem), as for TAP-CT and SPECTRE (figures/data/pillar0_parts.csv, from diagrams.py)
    p0 = pd.read_csv(DIAG_DATA / "pillar0_parts.csv").set_index("part").numel
    check("fig_overview", "(a) Pillar-0 logged total = checkpoint - BatchNorm buffers <= checkpoint", str(tot["pillar0"] <= p0["checkpoint_total"]), ref="True")
    tot["pillar0"] -= int(p0["text_projection"])
    print("  logged encoder sizes:", {b: f"{v:,} ({cls_name[b]})" for b, v in tot.items()})

    fig = plt.figure(figsize=(W, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 0.86], height_ratios=[1.0, 0.85], wspace=0.42, hspace=0.5,
                          left=0.005, right=0.985, top=0.905, bottom=0.10)
    # (a) design: three stacked cards, equal gaps, identical arrows
    ax = fig.add_subplot(gs[:, 0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-1.62, 10)
    # family tints: greys and purples (blue, orange and green are reserved for the backbones)
    FAM = {"frozen": ("#eceef1", "#a4abb4"), "low": ("#efedf5", style.METHOD_COL["lorag"]), "full": ("#f7e8f1", style.METHOD_COL["attnft"])}
    GAP = 0.42                                              # vertical gap between cards (holds one arrow)

    def cardbox(y, h, title):
        ax.add_patch(FancyBboxPatch((0.12, y - 0.06), 9.8, h, boxstyle="round,pad=0.02,rounding_size=0.3", fc="#000000", ec="none", alpha=0.06))
        ax.add_patch(FancyBboxPatch((0.05, y), 9.8, h, boxstyle="round,pad=0.02,rounding_size=0.3", fc="white", ec="#d9d7d2", lw=0.7))
        ax.add_patch(Rectangle((0.05, y + h - 0.62), 0.12, 0.42, fc=INK2, ec="none"))
        ax.text(0.35, y + h - 0.41, title, ha="left", va="center", fontsize=7.0, color=INK, fontweight="bold")

    def link(y_upper_bottom):                               # one arrow, same length and style for every gap
        ax.annotate("", xy=(5, y_upper_bottom - GAP + 0.08), xytext=(5, y_upper_bottom - 0.1),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))

    h1, h2, h3 = (2.05, 6.27, 2.3) if NO_VOCO else (2.45, 6.27, 2.3)
    row1, row3 = 0.42, 0.40                                 # line spacing in the backbone and task cards
    y1 = 9.95 - h1; y2 = y1 - GAP - h2; y3 = y2 - GAP - h3
    cardbox(y1, h1, "3D CT backbones")
    ax.text(9.55, y1 + h1 - 0.41, "parameters", ha="right", va="center", fontsize=6.2, color=INK2)
    arch = {"tapct": "ViT-B", "spectre": "ViT-L", "pillar0": "Atlas", "voco": "Swin"}
    for k, bb in enumerate(["tapct", "spectre", "pillar0"] + VOCO):
        yy = y1 + h1 - 0.88 - k * row1
        ax.scatter([0.6], [yy], s=13, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], linewidths=0)
        ax.text(1.0, yy, BACKBONE_TEX[bb], ha="left", va="center", fontsize=6.2, color=INK)
        ax.text(4.2, yy, arch[bb], ha="left", va="center", fontsize=6.2, color=INK2)
        ax.text(9.55, yy, _fmt_m(tot[bb]), ha="right", va="center", fontsize=6.2, color=INK)
    cardbox(y2, h2, "What is trained")
    cols = [("frozen", "frozen", ["lin", "attn"], ["lin", "dec"]),
            ("low", "low-rank", ["lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag"],
             ["lora", "lorag", "dora", "dec_then_lora"]),
            ("full", "full-rank", ["attnft", "fullft"], ["fullft"])]
    fam_cls = sum(len(c[2]) for c in cols); fam_seg = sum(len(c[3]) for c in cols)
    check("fig_overview", "(a) classification methods listed", str(fam_cls), ref=str(n_cls_m))
    check("fig_overview", "(a) segmentation methods listed", str(fam_seg), ref=str(n_seg_m))
    step = 0.40
    top2 = y2 + h2
    ax.text(0.35, top2 - 0.85, f"classification, {n_cls_m} methods", ha="left", va="center", fontsize=7.0, color=INK2)
    ax.text(0.35, top2 - 4.37, f"segmentation, {n_seg_m} methods", ha="left", va="center", fontsize=7.0, color=INK2)
    for (fam, head, cls_m, seg_m), x0, wc in zip(cols, [0.2, 2.4, 6.25], [2.0, 3.95, 3.75]):
        fc, ec = FAM[fam]
        ax.text(x0 + 0.05, top2 - 1.22, head, ha="left", va="center", fontsize=7.0, color=INK, fontweight="bold")
        for items, ytop in ((cls_m, top2 - 1.60), (seg_m, top2 - 4.76)):
            for j, it in enumerate(items):
                y = ytop - j * step
                ax.add_patch(FancyBboxPatch((x0, y - 0.18), wc - 0.2, 0.36, boxstyle="round,pad=0,rounding_size=0.1", fc=fc, ec=ec, lw=0.6))
                ax.text(x0 + 0.12, y, MTEX[it], ha="left", va="center", fontsize=6.0, color=INK)
    cardbox(y3, h3, "Two tasks, per label and joint")
    for k, t in enumerate((f"classification: {n_labels} binary labels", f"in {n_cohorts} cohorts, AUROC",
                           f"segmentation: {n_struct} structures", f"in {n_segds} datasets, Dice")):
        ax.text(0.35 + (0.4 if k % 2 else 0), y3 + h3 - 0.75 - k * row3, t, ha="left", va="center", fontsize=6.2, color=INK2)
    link(y1); link(y2)
    ax.set_title("(a) design", fontsize=7, color=INK, loc="left")

    # (b) classification: change vs the backbone's frozen probes, per group of methods (categorical axis)
    ax = fig.add_subplot(gs[0, 1])
    bbs_b = MAIN + VOCO
    off = {bb: (k - (len(bbs_b) - 1) / 2) * 0.15 for k, bb in enumerate(bbs_b)}
    for bb in bbs_b:
        xs, ys = [], []
        ref = np.mean([pooled[(bb, mt)] for mt in GROUPS[0][1]])
        for gi, (_, meths) in enumerate(GROUPS):
            vals = [pooled[(bb, mt)] - ref for mt in meths if (bb, mt) in pooled.index]
            if not vals:
                continue
            for j, v in enumerate(vals):
                ax.scatter([gi + off[bb] + (j - (len(vals) - 1) / 2) * 0.05], [v], s=6, color=BACKBONE_COL[bb],
                           marker=BACKBONE_MARK[bb], alpha=0.7, linewidths=0, zorder=3)
            xs.append(gi + off[bb]); ys.append(np.mean(vals))
        ax.plot(xs, ys, BB_LS[bb], color=BACKBONE_COL[bb], lw=1.2, zorder=2)
        ax.scatter(xs, ys, s=20, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], edgecolor="white", linewidths=0.7, zorder=4)
    # encoder parameters updated per group (for the caption / appendix; no longer printed on the axis)
    for name, meths in GROUPS:
        vals = [updated_params(bb, mt) for bb in MAIN for mt in meths if (bb, mt) in pooled.index]  # runs that trained
        lo, hi = min(vals), max(vals)
        CHECKS.append(f"INFO    fig_overview         (b) encoder parameters updated, {name.replace(chr(10), ' ')}: "
                      f"{'0' if hi == 0 else _fmt_p(lo) + '-' + _fmt_p(hi)} M (TAP-CT, SPECTRE, Pillar-0; caption)")
    tp = pd.read_csv(DIAG_DATA / "trainable_params_cls.csv").set_index(["backbone", "method"]).backbone_trainable
    for _, meths in GROUPS:
        for bb in MAIN:
            for mt in meths:
                if (bb, mt) in pooled.index and (bb, mt) in tp.index:
                    check("fig_overview", f"(b) logged encoder trainable {bb} {mt} vs diagrams table", str(logged_params("cls", bb, mt)[0]), ref=str(int(tp[(bb, mt)])))
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([g for g, _ in GROUPS], fontsize=6.2, linespacing=1.0)
    ax.set_ylabel("\u0394 mean AUROC vs. frozen probes", fontsize=6.4)
    ax.axhline(0, color=INK2, lw=0.7, zorder=1)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_ylim(*((-0.012, 0.056) if NO_VOCO else (-0.075, 0.056)))   # the lower range held VoCo's line
    tidy(ax, "y")
    ax.set_title("(b) classification: what is trained", fontsize=7, color=INK, loc="left")

    # (c) classification rank
    ax = fig.add_subplot(gs[0, 2])
    rank_panel(ax, rcls, MAIN + VOCO, "rank" if NO_VOCO else "rankall", 11, "c")
    ax.set_title("(c) classification rank", fontsize=7, color=INK, loc="left")

    # (d) segmentation: mean Dice per method minus that of Lin, over the structure-mode pairs with all seven methods
    #     (a difference, as in (b), because absolute Dice is computed on each backbone's own grid)
    smeth = ["lin", "dec", "lorag", "lora", "dora", "dec_then_lora", "fullft"]
    sbbs = ["tapct", "spectre"] + VOCO
    sp = sg[sg.backbone.isin(sbbs)].pivot_table(index=["backbone", "cohort", "label", "mode"], columns="method", values="dice")[smeth].dropna()
    lci = load("label_ci")
    lci = lci[(lci.task == "seg") & lci["mode"].isin(["pertask", "joint"]) & lci.method.isin(smeth)].pivot_table(
        index=["backbone", "cohort", "label", "mode"], columns="method", values="value")[smeth].dropna()
    ax = fig.add_subplot(gs[1, 1])
    dmax = 0.0
    for k, bb in enumerate(sbbs):
        g = sp.xs(bb, level="backbone")
        nslot = len(g)
        check("fig_overview", f"(d) slots with all seven methods, {bb}", str(nslot),
              ref=str(int(rseg[(rseg.backbone == bb)].n_slots.iloc[0])))
        xs = np.arange(len(smeth)) + (k - (len(sbbs) - 1) / 2) * 0.16
        ys = np.array([g[mt].mean() for mt in smeth])
        for mt, v in zip(smeth, ys):   # cross-file check: label_ci.csv holds the same per-structure held-out values
            check("fig_overview", f"(d) mean Dice {bb} {mt} vs label_ci.csv", f"{v:.3f}", ref=f"{lci.xs(bb, level='backbone')[mt].mean():.3f}")
        ys = ys - ys[smeth.index("lin")]
        dmax = max(dmax, float(ys.max()))
        # markers only: the methods are unordered, so no connecting line
        ax.scatter(xs, ys, s=18, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], edgecolor="white", linewidths=0.7, zorder=3)
        CHECKS.append(f"INFO    fig_overview         (d) {bb} mean Dice minus Lin: " + ", ".join(f"{mt} {v:+.3f}" for mt, v in zip(smeth, ys)))
    ax.axvspan(-0.5, 1.5, color=BAND, zorder=0)
    ax.text(0.5, 0.965, "frozen\nencoder", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color=INK2, linespacing=1.0)
    ax.set_xticks(range(len(smeth))); ax.set_xticklabels([MTEX[mt] for mt in smeth], fontsize=6.0, rotation=35, ha="right", rotation_mode="anchor")
    ax.axhline(0, color=INK2, lw=0.7, zorder=1)
    ax.set_xlim(-0.5, len(smeth) - 0.5); ax.set_ylim(-0.012, dmax + 0.03)
    ax.set_ylabel("\u0394 mean Dice vs. Lin", fontsize=6.4)
    tidy(ax, "y")
    ax.set_title("(d) segmentation: what is trained", fontsize=7, color=INK, loc="left")

    # (e) segmentation rank
    ax = fig.add_subplot(gs[1, 2])
    rank_panel(ax, rseg, sbbs, "segrank" if NO_VOCO else "segrankall", 7, "e")
    ax.set_title("(e) segmentation rank", fontsize=7, color=INK, loc="left")
    # one backbone legend for (b)-(e), above the data panels
    hs = [Line2D([], [], color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], markersize=3.6, lw=1.2, ls=BB_LS[bb],
                 label={"pillar0": "Pillar-0 (ICH only)"}.get(bb, BACKBONE_TEX[bb])) for bb in MAIN + VOCO]
    hs.append(Line2D([], [], color=INK, marker="|", ms=6, mew=1.8, ls="", label="mean rank, all backbones"))
    fig.legend(handles=hs, frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(0.335, 1.0), ncol=len(hs),
               handlelength=2.0, columnspacing=1.1, handletextpad=0.4, borderaxespad=0.0)
    save(fig, "fig_overview")


# ============================================================================ fig_key
KEY_ROWS = {
    "cls": [("cls_lora_lin", "LoRA \u2212 Lin", "clsloralin"), ("cls_lora_attn", "LoRA \u2212 Attn", "clsloraattn"),
            ("cls_attnft_lora", "Attn-only FT \u2212 LoRA", "clsattnftlora"), ("cls_fullft_lin", "Full FT \u2212 Lin", "clsfullftlin"),
            ("cls_joint_pertask", "joint \u2212 per-task", "clsjoint")],
    "seg": [("seg_dec_lin", "Dec \u2212 Lin", "segdeclin"), ("seg_lora_dec", "LoRA \u2212 Dec", "segloradec"),
            ("seg_lora_dec_pertask", "LoRA \u2212 Dec, per-structure", "segloradecpt"), ("seg_lora_dec_joint", "LoRA \u2212 Dec, joint", "segloradecjt"),
            ("seg_fullft_dec", "Full FT \u2212 Dec", "segfullftdec"), ("seg_joint_pertask", "joint \u2212 per-task", "segjoint")],
}


def fig_key() -> None:
    ll = load("label_level")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(width_ratios=[1, 1], wspace=1.0))
    xlims = {"cls": (-0.21, 0.13), "seg": (-0.13, 0.17)}
    for ax, (task, title, unit) in zip(axes, (("cls", "(a) classification", "AUROC"), ("seg", "(b) segmentation", "Dice"))):
        rows = KEY_ROWS[task]
        bbs = [bb for bb in ("tapct", "spectre", "pillar0", *VOCO) if ((ll.task == task) & (ll.backbone == bb)).any()]
        lo_x, hi_x = xlims[task]
        for i, (key, lab, mkey) in enumerate(rows):
            y0 = len(rows) - 1 - i
            if i % 2 == 0:
                ax.axhspan(y0 - 0.5, y0 + 0.5, color=BAND, zorder=0, lw=0)
            for j, bb in enumerate(bbs):
                y = y0 + 0.3 - j * 0.6 / max(len(bbs) - 1, 1)
                r = ll[(ll.key == key) & (ll.backbone == bb)]
                if r.empty:
                    continue
                r = r.iloc[0]
                col, mk = BACKBONE_COL[bb], BACKBONE_MARK[bb]
                est, lo, hi = r["mean"], r["lo3"], r["hi3"]
                tag = bb.title().replace("0", "Z")
                check("fig_key", f"{key} {bb} mean", fmt_signed(est), f"numCi{mkey.title()}{tag}Meandiff")
                check("fig_key", f"{key} {bb} lo3", fmt_signed(lo), f"numCi{mkey.title()}{tag}Lodiff")
                check("fig_key", f"{key} {bb} hi3", fmt_signed(hi), f"numCi{mkey.title()}{tag}Hidiff")
                if est > hi_x or est < lo_x:
                    edge = hi_x if est > hi_x else lo_x
                    ax.scatter([edge], [y], s=18, marker=">" if est > hi_x else "<", color=col, zorder=3, linewidths=0, clip_on=False)
                    ax.text(edge - 0.008 if est > hi_x else edge + 0.008, y, um(f"{est:+.2f}"), fontsize=6.0, color=INK,
                            ha="right" if est > hi_x else "left", va="center")
                    check("fig_key", f"{key} {bb} printed value", f"{est:+.2f}", ref=f"{r['mean']:+.2f}")
                    continue
                if lo < lo_x or hi > hi_x:
                    raise SystemExit(f"fig_key: interval of {key}/{bb} [{lo:.3f}, {hi:.3f}] leaves the axis; widen xlims")
                ax.plot([lo, hi], [y, y], color=col, lw=1.0, zorder=2, solid_capstyle="butt")
                ax.scatter([est], [y], s=16, marker=mk, color=col, edgecolor=col, linewidths=0.5, zorder=3)
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for _, lab, _ in rows][::-1], fontsize=6.4)
        ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlim(lo_x, hi_x)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.set_xlabel(f"mean difference in held-out {unit}", fontsize=6.5)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.tick_params(length=2, labelsize=6.2); ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=7, color=INK, loc="left")
    hs = [Line2D([], [], color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], lw=1.0, markersize=3.8, label=BACKBONE_TEX[bb])
          for bb in ("tapct", "spectre", "pillar0", *VOCO)]
    fig.legend(handles=hs, frameon=False, fontsize=6.4, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.55, 1.0), handlelength=1.8, columnspacing=1.6)
    fig.subplots_adjust(left=0.21, right=0.985, top=0.85, bottom=0.16)
    save(fig, "fig_key")


# ============================================================================ fig_mechanisms
ITS8 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]


def trajectories() -> tuple[dict, dict]:
    """Median validation-AUROC trajectory per (backbone, method) over the per-task and joint runs (seed 42) of
    runs.csv with all eight evaluations (1000, ..., 8000 iterations = optimizer steps)."""
    r = runs("cls", mode=["pertask", "joint"], backbone=MAIN)
    out: dict = {}
    for bb, mt, p in zip(r.backbone, r.method, r.path):
        v = val_curve(p, "val_auroc")
        if all(k in v for k in ITS8):
            out.setdefault((bb, mt), []).append([v[k] for k in ITS8])
    return {k: np.median(np.array(v), axis=0) for k, v in out.items()}, {k: len(v) for k, v in out.items()}


def joint_trajectories(bb: str, methods: list[str]) -> tuple[dict, list[str]]:
    """Median validation-AUROC trajectory of the joint models of one backbone per method, over the cohorts where every
    method has all eight evaluations (one population for every curve)."""
    r = runs("cls", mode="joint", backbone=bb, method=methods)
    per = {}
    for ds, mt, p in zip(r.cohort, r.method, r.path):
        v = val_curve(p, "val_auroc")
        if all(k in v for k in ITS8):
            per[(mt, ds)] = [v[k] for k in ITS8]
    common = sorted(set.intersection(*[{ds for (m_, ds) in per if m_ == mt} for mt in methods]))
    return {mt: np.median(np.array([per[(mt, ds)] for ds in common]), axis=0) for mt in methods}, common


def fig_mechanisms() -> None:
    v = load("metrics_cls")
    v = v[(v.split == "val") & (v.seed == 42) & (v.method == "lin")]
    probe_val = {k: x for k, x in zip(zip(v.backbone, v.cohort, v["mode"], v.label), v.auroc)}
    p = paired_cls("lin", drop_dnt=True)
    p = p[p.backbone.isin(MAIN)]

    def gains(meth):  # held-out gain over the probe, at the probe's selection-split AUROC (as the gainbin_* macros)
        g = p[p.method == meth].copy()
        g["x"] = [probe_val.get(k) for k in zip(g.backbone, g.cohort, g["mode"], g.label)]
        return g.dropna(subset=["x"])
    d, dp = gains("lora"), gains("attn")
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35))
    # (a)
    ax = axes[0]
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.scatter(d.x, d["diff"], s=5, color=OI["grey"], alpha=0.7, linewidths=0, zorder=2)
    edges = [0.0, 0.6, 0.7, 0.8, 1.01]
    for bi, (lo, hi, tag) in enumerate(zip(edges[:-1], edges[1:], ["low", "mid", "high", "top"])):
        g = d[(d.x > lo) & (d.x <= hi)]
        gp = dp[(dp.x > lo) & (dp.x <= hi)]
        xa, xb = max(lo, 0.35), min(hi, 1.0)
        ax.plot([xa, xb], [g["diff"].mean()] * 2, color=MCOL["lora"], lw=2.2, zorder=3, solid_capstyle="butt")
        ax.plot([xa, xb], [gp["diff"].mean()] * 2, color=INK2, lw=1.2, ls=(0, (1, 1)), zorder=3)
        yl = 0.40 if bi % 2 == 0 else 0.32
        ax.text((xa + xb) / 2, yl, fmt_signed(g["diff"].mean()), ha="center", va="center", fontsize=6.0, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=4)
        if hi < 1.0:
            ax.axvline(hi, color=GRID, lw=0.8, ls=":", zorder=1)
        check("fig_mechanisms", f"(a) LoRA-Lin bin {tag} mean", fmt_signed(g["diff"].mean()), mname("gainbin", tag, "meandiff"))
        check("fig_mechanisms", f"(a) LoRA-Lin bin {tag} n", str(len(g)), mname("gainbin", tag, "n"))
        check("fig_mechanisms", f"(a) Attn-Lin bin {tag} mean (line)", fmt_signed(gp["diff"].mean()), mname("gainbinattn", tag, "meandiff"))
        check("fig_mechanisms", f"(a) Attn-Lin bin {tag} n", str(len(gp)), mname("gainbinattn", tag, "n"))
    ax.set_xlabel("linear-probe AUROC, selection split", fontsize=6.4)
    ax.set_ylabel("Δ AUROC vs. linear probe", fontsize=6.4)
    ax.set_title("(a) gain vs. probe level", fontsize=7, color=INK, loc="left")
    ax.set_xlim(0.35, 1.0); ax.set_ylim(-0.25, 0.45); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    tidy(ax, "both")
    # (b) learning curves
    ax = axes[1]
    med, n = trajectories()
    its = np.arange(1, 9)
    styles = {"tapct": "-", "spectre": "--"}
    meths = ["lin", "lora", "attnft", "fullft"]
    for bb in ["tapct", "spectre"]:
        for mt in meths:
            ax.plot(its, med[(bb, mt)], styles[bb], color=MCOL[mt], lw=1.3, marker="o", ms=2.0, mec=MEDGE[mt] if mt == "attnft" else MCOL[mt], mew=0.4)
    ax.set_xlabel("optimizer steps (×1000)", fontsize=6.4)
    ax.set_ylabel("median val. AUROC", fontsize=6.4)
    ax.set_title("(b) learning curves", fontsize=7, color=INK, loc="left")
    ax.set_ylim(0.45, 0.85); ax.set_xticks([2, 4, 6, 8])
    tidy(ax, "both")
    # (c) SPECTRE joint models, one population of cohorts for every curve
    ax = axes[2]
    jt, common = joint_trajectories("spectre", ["lin", "lora", "attnft", "fullft", "fullft_llrd"])
    print("  (c) SPECTRE joint cohorts:", common)
    for mt, tr in jt.items():
        if mt == "fullft_llrd":
            ax.plot(its, tr, ":", color=MCOL["fullft"], lw=1.3, marker="o", ms=2.4, mfc="white", mew=0.9)
        else:
            ax.plot(its, tr, "--", color=MCOL[mt], lw=1.3, marker="o", ms=2.0, mec=MEDGE[mt] if mt == "attnft" else MCOL[mt], mew=0.4)
    ax.set_xlabel("optimizer steps (×1000)", fontsize=6.4)
    ax.set_title("(c) SPECTRE joint models", fontsize=7, color=INK, loc="left")
    ax.set_ylim(0.45, 0.85); ax.set_xticks([2, 4, 6, 8]); ax.set_yticklabels([])
    tidy(ax, "both")
    fig.tight_layout(w_pad=0.9, rect=(0, 0.15, 1, 1))
    hs = [Line2D([], [], color=MCOL["lora"], lw=2.2, label="LoRA − Lin, bin mean (a)"),
          Line2D([], [], color=INK2, lw=1.2, ls=(0, (1, 1)), label="Attn − Lin, bin mean (a)")]
    hs += [mhandle(mt, line=True, ms=3.0, marker="o") for mt in meths]
    hs += [Line2D([], [], color=INK2, lw=1.3, ls="-", label="TAP-CT (b)"), Line2D([], [], color=INK2, lw=1.3, ls="--", label="SPECTRE (b, c)"),
           Line2D([], [], color=MCOL["fullft"], lw=1.3, ls=":", marker="o", ms=3, mfc="white", mew=0.9, label="Full FT, layer decay (c)")]
    fig.legend(handles=hs, frameon=False, fontsize=6.0, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, handletextpad=0.4, columnspacing=1.0)
    save(fig, "fig_mechanisms")
    write_traj_macros(med, n, jt, common)


def write_traj_macros(med: dict, n: dict, jt: dict, common: list[str]) -> None:
    """figure_numbers.tex: macros read off the learning curves (fig_fullft_spectre; formerly fig_mechanisms b, c)."""
    lines = ["% written by figures/paper_figures.py: median validation AUROC of the per-task and joint runs (seed 42) with all",
             "% eight evaluations, after the first evaluation (1000 steps), at the last (8000) and the maximum; N = runs."]
    for (bb, mt) in sorted(med):
        tr = med[(bb, mt)]
        tag = (bb.replace("0", "Z") + mt.replace("_", "")).title().replace("_", "")
        lines += [f"\\providecommand{{\\numTraj{tag}First}}{{{tr[0]:.3f}}}", f"\\providecommand{{\\numTraj{tag}Last}}{{{tr[-1]:.3f}}}",
                  f"\\providecommand{{\\numTraj{tag}Max}}{{{tr.max():.3f}}}", f"\\providecommand{{\\numTraj{tag}N}}{{{n[(bb, mt)]}}}"]
    lines.append("% SPECTRE joint models (fig_fullft_spectre b), cohorts where every curve has all eight evaluations")
    lines.append(f"\\providecommand{{\\numTrajJointSpectreCohortsN}}{{{len(common)}}}")
    for mt, tr in jt.items():
        tag = mt.replace("_", "").title()
        lines += [f"\\providecommand{{\\numTrajJointSpectre{tag}First}}{{{tr[0]:.3f}}}", f"\\providecommand{{\\numTrajJointSpectre{tag}Last}}{{{tr[-1]:.3f}}}"]
    (TABLES / "figure_numbers.tex").write_text("\n".join(lines) + "\n")
    for k in [("spectre", "fullft"), ("spectre", "attnft")]:
        print(f"  trajectory {k}: first {med[k][0]:.3f} last {med[k][-1]:.3f} n={n[k]}")


# ============================================================================ fig_gain_vs_probe (appendix; was fig_mechanisms a)
def fig_gain_vs_probe() -> None:
    """LoRA - Lin held-out AUROC per label-mode pair against the linear probe's selection-split AUROC, all four backbones;
    bin means as the \\numGainbin* macros (TAP-CT, SPECTRE, Pillar-0 pooled; VoCo drawn but not in the bins)."""
    v = load("metrics_cls")
    v = v[(v.split == "val") & (v.seed == 42) & (v.method == "lin")]
    probe_val = {k: x for k, x in zip(zip(v.backbone, v.cohort, v["mode"], v.label), v.auroc)}
    p = paired_cls("lin", drop_dnt=True)

    def gains(meth, bbs):
        g = p[(p.method == meth) & p.backbone.isin(bbs)].copy()
        g["x"] = [probe_val.get(k) for k in zip(g.backbone, g.cohort, g["mode"], g.label)]
        return g.dropna(subset=["x"])
    d, dp = gains("lora", MAIN), gains("attn", MAIN)
    dv = gains("lora", ["voco"]) if not NO_VOCO else None
    fig = plt.figure(figsize=(W, 1.95))
    ax = fig.add_axes([0.095, 0.2, 0.65, 0.77])
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    for bb in MAIN + VOCO:
        g = dv if bb == "voco" else d[d.backbone == bb]
        ax.scatter(g.x, g["diff"], s=9 if bb != "voco" else 11, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb],
                   alpha=0.6 if bb != "voco" else 0.85, linewidths=0, zorder=2 if bb != "voco" else 3)
        CHECKS.append(f"INFO    fig_gain_vs_probe    {bb}: {len(g)} label-mode pairs drawn (paired_cls.csv, LoRA vs Lin)")
        if g["diff"].min() < -0.36 or g["diff"].max() > 0.46 or g.x.min() < 0.35:
            raise SystemExit(f"fig_gain_vs_probe: {bb} points outside the axis")
    edges = [0.0, 0.6, 0.7, 0.8, 1.01]
    for bi, (lo, hi, tag) in enumerate(zip(edges[:-1], edges[1:], ["low", "mid", "high", "top"])):
        g = d[(d.x > lo) & (d.x <= hi)]
        gp = dp[(dp.x > lo) & (dp.x <= hi)]
        xa, xb = max(lo, 0.35), min(hi, 1.0)
        ax.plot([xa, xb], [g["diff"].mean()] * 2, color=INK, lw=2.0, zorder=4, solid_capstyle="butt")
        ax.plot([xa, xb], [gp["diff"].mean()] * 2, color=style.CONTROL_COL, lw=1.4, ls=(0, (1, 1)), zorder=4)
        # per bin, top to bottom: LoRA - Lin mean, Attn - Lin mean (control colour), number of pairs
        for yt, s_, c_ in ((0.55, um(fmt_signed(g["diff"].mean())), INK),
                           (0.478, um(fmt_signed(gp["diff"].mean())), style.CONTROL_COL), (0.406, f"n = {len(g)}", INK)):
            ax.text((xa + xb) / 2, yt, s_, ha="center", va="center", fontsize=7.0, color=c_,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=5)
        if hi < 1.0:   # bin edge
            ax.axvline(hi, color="#c4c4c4", lw=0.7, zorder=1)
        check("fig_gain_vs_probe", f"LoRA-Lin bin {tag} mean", fmt_signed(g["diff"].mean()), mname("gainbin", tag, "meandiff"))
        check("fig_gain_vs_probe", f"LoRA-Lin bin {tag} n", str(len(g)), mname("gainbin", tag, "n"))
        check("fig_gain_vs_probe", f"Attn-Lin bin {tag} mean (line)", fmt_signed(gp["diff"].mean()), mname("gainbinattn", tag, "meandiff"))
        check("fig_gain_vs_probe", f"Attn-Lin bin {tag} n", str(len(gp)), mname("gainbinattn", tag, "n"))
    ax.set_xlabel("linear-probe AUROC on the selection split", fontsize=6.5)
    ax.set_ylabel("Δ held-out AUROC\nvs. linear probe", fontsize=6.5, linespacing=1.0)
    ax.set_xlim(0.35, 1.0); ax.set_ylim(-0.36, 0.60); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.tick_params(length=2, labelsize=7.0)   # x: bin edges only
    hs = [Line2D([], [], color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], ls="", ms=4, label=BACKBONE_TEX[bb])
          for bb in MAIN + VOCO]
    hs += [Line2D([], [], color=INK, lw=2.0, label="LoRA − Lin,\nbin mean"),
           Line2D([], [], color=style.CONTROL_COL, lw=1.4, ls=(0, (1, 1)), label="Attn − Lin,\nbin mean (control)")]
    fig.legend(handles=hs, frameon=False, fontsize=7.0, loc="center left", bbox_to_anchor=(0.765, 0.56), handlelength=2.0,
               labelspacing=0.7)
    save(fig, "fig_gain_vs_probe")


# ============================================================================ fig_fullft_spectre (appendix; was fig_mechanisms b, c)
def fig_fullft_spectre() -> None:
    med, n = trajectories()
    its = np.arange(1, 9)
    meths = ["lin", "lora", "attnft", "fullft"]
    ls_bb = {"tapct": "-", "spectre": "--"}
    fig = plt.figure(figsize=(W, 2.3))
    axa = fig.add_axes([0.075, 0.18, 0.3, 0.7])
    axb = fig.add_axes([0.415, 0.18, 0.3, 0.7], sharey=axa)
    for bb in ["tapct", "spectre"]:
        for mt in meths:
            axa.plot(its, med[(bb, mt)], ls_bb[bb], color=MCOL[mt], lw=1.3, marker="o", ms=2.0, mec=MEDGE[mt] if mt == "attnft" else MCOL[mt], mew=0.4)
            CHECKS.append(f"INFO    fig_fullft_spectre   (a) {bb} {mt}: median over {n[(bb, mt)]} runs, first {med[(bb, mt)][0]:.3f} last {med[(bb, mt)][-1]:.3f}")
    axa.set_title("(a) TAP-CT and SPECTRE, all runs", fontsize=7, color=INK, loc="left")
    axa.set_ylabel("median validation AUROC", fontsize=6.5)
    jt, common = joint_trajectories("spectre", ["lin", "lora", "attnft", "fullft", "fullft_llrd"])
    print("  (b) SPECTRE joint cohorts:", common)
    for mt, tr in jt.items():
        if mt == "fullft_llrd":
            axb.plot(its, tr, ":", color=MCOL["fullft"], lw=1.4, marker="o", ms=2.6, mfc="white", mew=0.9)
        else:
            axb.plot(its, tr, "--", color=MCOL[mt], lw=1.3, marker="o", ms=2.0, mec=MEDGE[mt] if mt == "attnft" else MCOL[mt], mew=0.4)
    axb.set_title(f"(b) SPECTRE joint models, {len(common)} cohorts", fontsize=7, color=INK, loc="left")
    plt.setp(axb.get_yticklabels(), visible=False)
    for ax in (axa, axb):
        ax.set_xlabel("optimizer steps (×1000)", fontsize=6.5)
        ax.set_ylim(0.45, 0.85); ax.set_xticks([2, 4, 6, 8])
        tidy(ax, "both")
    hm = [mhandle(mt, line=True, ms=3.0, marker="o") for mt in meths]
    hm.append(Line2D([], [], color=MCOL["fullft"], lw=1.4, ls=":", marker="o", ms=3, mfc="white", mew=0.9,
                     label="Full FT, layer-wise LR decay\n(SPECTRE only: re-run because\nits default Full FT did not train)"))
    hb = [Line2D([], [], color="#000000", lw=1.3, ls="-", label="TAP-CT"), Line2D([], [], color="#000000", lw=1.3, ls="--", label="SPECTRE")]
    lg = fig.legend(handles=hm, frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(0.74, 0.92), handlelength=2.2,
                    title="color: method", title_fontsize=6.4, alignment="left", labelspacing=0.55)
    fig.add_artist(lg)
    fig.legend(handles=hb, frameon=False, fontsize=6.3, loc="lower left", bbox_to_anchor=(0.74, 0.08), handlelength=2.2,
               title="line style: backbone", title_fontsize=6.4, alignment="left")
    save(fig, "fig_fullft_spectre")
    write_traj_macros(med, n, jt, common)


# ============================================================================ fig_joint_main (main text)
def _joint_rows(task: str):
    """Per (backbone, cohort): the label-level joint - per-task value averaged over labels (label_level_values.csv; per label
    the mean over the methods trained in both modes), and per method the mean over labels of joint - per-task from the
    metrics files (grey dots)."""
    key = f"{task}_joint_pertask"
    ll = load("label_level_values")
    lv = ll[ll.key == key]
    if task == "cls":
        t = cls_test()
        piv = t.pivot_table(index=["backbone", "cohort", "method", "label"], columns="mode", values="auroc").dropna()
    else:
        t = seg_split()
        piv = t.pivot_table(index=["backbone", "cohort", "method", "label"], columns="mode", values="dice").dropna()
    per_m = (piv["joint"] - piv["pertask"]).groupby(level=["backbone", "cohort", "method"]).mean()
    main = lv.groupby(["backbone", "cohort"]).value.mean()
    return main, per_m, lv


def fig_joint_main() -> None:
    fig = plt.figure(figsize=(W, 1.85))
    axes = [fig.add_axes([0.105, 0.215, 0.37, 0.545]), fig.add_axes([0.615, 0.215, 0.37, 0.545])]
    specs = [("cls", CLS_COHORTS, COHORT_SHORT, MAIN + VOCO, "(a) classification: joint − per-task", "held-out AUROC", (-0.18, 0.12)),
             ("seg", SEG_DATASETS, DS_SHORT, ["tapct", "spectre"] + VOCO, "(b) segmentation: joint − per-structure", "held-out Dice", (-0.16, 0.04))]
    ll_ci = load("label_level")
    for ax, (task, rows, names, bbs, title, unit, xl) in zip(axes, specs):
        main, per_m, lv = _joint_rows(task)
        nb = len(bbs)
        for j, ds in enumerate(rows):
            if task == "seg" and ds in LONG_JOINT:
                ax.axhspan(j - 0.5, j + 0.5, color=BAND, zorder=0, lw=0)
            for k, bb in enumerate(bbs):
                if (bb, ds) not in main.index:
                    continue
                y = j + (k - (nb - 1) / 2) * (0.62 / (nb - 1))
                pm = per_m.loc[(bb, ds)]
                ax.scatter(pm.values, np.full(len(pm), y), s=7, color=BACKBONE_COL[bb], alpha=0.35, linewidths=0, zorder=2)   # one method
                ax.scatter([main[(bb, ds)]], [y], s=22, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], edgecolor="white",
                           linewidths=0.6, zorder=4)
                if pm.min() < xl[0] or pm.max() > xl[1]:
                    raise SystemExit(f"fig_joint_main: {task} {bb} {ds} per-method value outside {xl}")
                check("fig_joint_main", f"({'ab'[task == 'seg']}) {bb} {ds} mean over labels vs mean over method dots",
                      fmt_signed(main[(bb, ds)]), ref=fmt_signed(pm.mean()))
                CHECKS.append(f"INFO    fig_joint_main       ({'ab'[task == 'seg']}) {bb} {ds}: {fmt_signed(main[(bb, ds)])} over "
                              f"{int((lv.backbone == bb).mul(lv.cohort == ds).sum())} labels, {len(pm)} methods "
                              f"({', '.join(pm.index)}), method range {fmt_signed(pm.min())} to {fmt_signed(pm.max())}")
        for bb in bbs:   # the backbone's mean over all labels = the label-level estimate quoted in the text
            v = lv[lv.backbone == bb].value.mean()
            tag = bb.title().replace("0", "Z")
            check("fig_joint_main", f"({'ab'[task == 'seg']}) {bb} mean over all labels", fmt_signed(v),
                  f"numCi{'Clsjoint' if task == 'cls' else 'Segjoint'}{tag}Meandiff")
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[r] for r in rows], fontsize=6.3)
        ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(*xl)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.set_xlabel(f"Δ {unit}", fontsize=6.5)
        ax.set_title(title, fontsize=7, color=INK, loc="left")
        tidy(ax)
        ax.tick_params(axis="y", length=0)
    hs = [Line2D([], [], color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], ls="", ms=4.2, mec="white", mew=0.5,
                 label=BACKBONE_TEX[bb]) for bb in MAIN + VOCO]
    hs.append(Line2D([], [], color=INK2, alpha=0.35, marker="o", ls="", ms=3.0, label="one method (light)"))
    fig.legend(handles=hs, frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=len(hs),
               handletextpad=0.3, columnspacing=1.4)
    save(fig, "fig_joint_main")


# ============================================================================ fig_seg
def fig_seg() -> None:
    d = seg_split()
    piv = d.pivot_table(index=["backbone", "cohort", "label", "mode"], columns="method", values="dice")
    ll = load("label_level")
    meths = ["lin", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
    bbs = ["tapct", "spectre"] + VOCO
    fig = plt.figure(figsize=(W, 2.0))
    gs = fig.add_gridspec(1, len(bbs), wspace=0.36 if NO_VOCO else 0.42, left=0.115, right=0.915 if NO_VOCO else 0.925, top=0.88, bottom=0.2)
    xl = (-0.3, 0.3)   # clipped: pairs beyond it are open triangles at the edge they exceed
    for gi, bb in enumerate(bbs):
        ax = fig.add_subplot(gs[0, gi])
        sub = piv.xs(bb, level="backbone")
        for i, mt in enumerate(meths):
            x = (sub[mt] - sub["dec"]).dropna().values
            y = len(meths) - 1 - i
            jit = (np.random.default_rng(i).random(len(x)) - 0.5) * 0.45
            out_l, out_r = x < xl[0] + 0.015, x > xl[1] - 0.015
            inn = ~(out_l | out_r)
            ax.scatter(x[inn], (y + jit)[inn], s=4, color=OI["grey"], alpha=0.75, linewidths=0, zorder=2)
            ax.scatter(np.full(out_l.sum(), xl[0] + 0.015), (y + jit)[out_l], s=8, marker="<", facecolor="white", edgecolor=OI["grey"], linewidths=0.6, zorder=2)
            ax.scatter(np.full(out_r.sum(), xl[1] - 0.015), (y + jit)[out_r], s=8, marker=">", facecolor="white", edgecolor=OI["grey"], linewidths=0.6, zorder=2)
            CHECKS.append(f"INFO    fig_seg              ({'ab'[gi]}) {bb} {mt}: {int(out_l.sum())} pairs below {xl[0]}, {int(out_r.sum())} above {xl[1]} (edge triangles)")
            if not (xl[0] <= x.mean() <= xl[1]):  # mean beyond the axis: a filled triangle at the edge (value printed on the right)
                ax.scatter([xl[0] if x.mean() < xl[0] else xl[1]], [y], s=34, color=BACKBONE_COL[bb], marker="<" if x.mean() < xl[0] else ">",
                           edgecolor="white", linewidths=0.6, zorder=4, clip_on=False)
            else:
                ax.scatter([x.mean()], [y], s=28, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], edgecolor="white", linewidths=0.9, zorder=3)
            ms = fmt_signed(x.mean())
            ax.text(1.07, y, um(ms), va="center", ha="left", fontsize=7.0, color=INK2, zorder=4, transform=ax.get_yaxis_transform(), clip_on=False)   # clear of the edge triangles
            if bb in ("tapct", "spectre"):
                check("fig_seg", f"({'ab'[gi]}) {bb} {mt} - Dec mean", ms, mname("segdelta", bb, "dec", mt, "meandiff"))
                check("fig_seg", f"({'ab'[gi]}) {bb} {mt} - Dec n", str(len(x)), mname("segdelta", bb, "dec", mt, "n"))
            else:  # VoCo: no per-pair macro for every method; check against the label-level mean where every structure has both modes
                mac = mname("segdelta", bb, "dec", mt, "n")
                if mac in macros():
                    check("fig_seg", f"(c) {bb} {mt} - Dec n", str(len(x)), mac)
                if mt == "lin":
                    r = ll[(ll.key == "seg_dec_lin") & (ll.backbone == bb)].iloc[0]
                    check("fig_seg", f"(c) {bb} Lin - Dec mean vs -(Dec - Lin) label level", ms, ref=fmt_signed(-r["mean"]))
                else:
                    r = ll[(ll.key == f"seg_{mt}_dec") & (ll.backbone == bb)]
                    if len(r) and len(x) == 2 * int(r.n_labels.iloc[0]):
                        check("fig_seg", f"(c) {bb} {mt} - Dec mean vs label level", ms, ref=fmt_signed(r["mean"].iloc[0]))
                    else:
                        CHECKS.append(f"INFO    fig_seg              (c) {bb} {mt} - Dec mean {ms}: structures lack a mode; computed from metrics_seg.csv")
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_yticks(range(len(meths)))
        if gi == 0:
            ax.set_yticklabels([MTEX[mt] for mt in meths[::-1]], fontsize=6.3)
        else:
            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
        ax.set_xlim(*xl); ax.set_xticks([-0.3, -0.15, 0, 0.15, 0.3]); ax.set_xticklabels(["\u22120.3", "", "0", "", "0.3"])
        ax.set_xlabel("Δ held-out Dice", fontsize=6.4)
        ax.set_title(f"({'abc'[gi]}) {BACKBONE_TEX[bb]} vs. Dec", fontsize=7, color=INK, loc="left")
        tidy(ax)
    save(fig, "fig_seg")


# ============================================================================ fig_bbgap (appendix)
def fig_bbgap() -> None:
    """SPECTRE minus TAP-CT held-out metric per cohort / dataset and method (same labels / structures and modes)."""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw=dict(wspace=0.55))
    # (a) classification, pairs that trained
    ax = axes[0]
    t = cls_test()
    t = t[t.backbone.isin(["tapct", "spectre"])]
    bk = t.groupby(["cohort", "method", "backbone"]).auroc.mean().unstack().dropna()
    cohorts = [c for c in COHORT_SHORT if c in bk.index.get_level_values(0)]
    cm = ["lin", "lora", "babmil", "attnft", "fullft"]
    vals_a = []
    for j, ds in enumerate(cohorts):
        for k, mt in enumerate(cm):
            if (ds, mt) in bk.index:
                val = bk.loc[(ds, mt), "spectre"] - bk.loc[(ds, mt), "tapct"]
                vals_a.append(val)
                mscatter(ax, [val], [j + (k - 2) * 0.15], mt, s=15)
                check("fig_bbgap", f"(a) SPECTRE-TAP-CT {ds} {mt}", fmt_signed(val), mname("bbdiff", ds, mt, "meandiff"))
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(len(cohorts))); ax.set_yticklabels([COHORT_SHORT[c] for c in cohorts], fontsize=6.3)
    ax.set_ylim(len(cohorts) - 0.5, -0.5)
    ax.set_xlabel("SPECTRE − TAP-CT, held-out AUROC", fontsize=6.4)
    ax.set_title("(a) between-backbone difference per method, classification", fontsize=7, color=INK, loc="right")
    ax.set_xlim(np.floor(min(vals_a) * 20 - 0.4) / 20, np.ceil(max(vals_a) * 20 + 0.4) / 20)   # the data, padded to 0.05
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    tidy(ax)
    # (b) segmentation
    ax = axes[1]
    d = seg_split()
    bp = d[d.backbone.isin(["tapct", "spectre"])].pivot_table(index=["cohort", "label", "mode", "method"], columns="backbone", values="dice").dropna()
    bp["gap"] = bp["spectre"] - bp["tapct"]
    gap = bp.groupby(["cohort", "method"]).gap.mean()
    gm = ["lin", "dec", "lora", "dora", "fullft"]
    for j, ds in enumerate(SEG_DATASETS):
        for k, mt in enumerate(gm):
            mscatter(ax, [gap.loc[(ds, mt)]], [j + (k - 2) * 0.15], mt, s=15)
    for mt in gm:  # the per-dataset values summarised by the seggap_* macros (mean over all pairs, min / max over datasets)
        vals = gap.xs(mt, level="method")
        check("fig_bbgap", f"(b) gap {mt} mean over structure-mode pairs", fmt_signed(bp.xs(mt, level="method").gap.mean()), mname("seggap", mt, "meandiff"))
        check("fig_bbgap", f"(b) gap {mt} max over datasets", fmt_signed(vals.max()), mname("seggap", mt, "max", "meandiff"))
        check("fig_bbgap", f"(b) gap {mt} min over datasets", fmt_signed(vals.min()), mname("seggap", mt, "min", "meandiff"))
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(len(SEG_DATASETS))); ax.set_yticklabels([DS_SHORT[x] for x in SEG_DATASETS], fontsize=6.3)
    ax.set_ylim(len(SEG_DATASETS) - 0.5, -0.5)
    ax.set_xlabel("SPECTRE − TAP-CT, held-out Dice", fontsize=6.4)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_title("(b) between-backbone difference per method, segmentation", fontsize=7, color=INK, loc="right")
    tidy(ax)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.3)
    for axx, ms_ in ((axes[0], [m for m in cm if any((ds, m) in bk.index for ds in cohorts)]), (axes[1], gm)):  # one legend per task
        axx.legend(handles=[mhandle(mt) for mt in ms_], frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.3),
                   ncol=len(ms_), handletextpad=0.1, columnspacing=0.8)
    save(fig, "fig_bbgap")


# ============================================================================ fig_joint
def fig_joint() -> None:
    t = cls_test()
    piv = t.pivot_table(index=["backbone", "cohort", "method", "label"], columns="mode", values="auroc").dropna()
    piv["dd"] = piv["joint"] - piv["pertask"]
    g = piv.reset_index().groupby(["backbone", "cohort", "method"]).dd.mean()
    cm = load("cohort_means")
    cm = cm[(cm.task == "cls") & (cm.seed == 42)].pivot_table(index=["backbone", "cohort", "method"], columns="mode", values="value")
    meths = ["lin", "lora", "babmil", "attnft", "fullft"]
    for (bb, ds, mt), v in g.items():
        if mt in meths and (bb, ds, mt) in cm.index:
            check("fig_joint", f"{bb} {ds} {mt} joint-pertask", fmt_signed(v), ref=fmt_signed(cm.loc[(bb, ds, mt), "joint"] - cm.loc[(bb, ds, mt), "pertask"]))
    # \numJointcohort*: the same rows pooled over TAP-CT, SPECTRE and Pillar-0
    pv = piv.loc[MAIN]
    pooled_c = pv.dd.groupby(level=["cohort", "method"]).mean()
    for (ds, mt), v in pooled_c.items():
        if mt in meths:
            check("fig_joint", f"rows behind the figure, pooled over backbones: {ds} {mt}", fmt_signed(v), mname("jointcohort", ds, mt, "meandiff"))
    lim = max(0.12, float(np.ceil((g.loc[[i for i in g.index if i[2] in meths]].abs().max() + 0.01) * 50) / 50))
    fig, axes = plt.subplots(1, len(MAIN + VOCO), figsize=(W, 2.05), sharex=True, sharey=True)
    cohorts = list(COHORT_SHORT)                     # same rows in every panel; a backbone without a cohort leaves it empty
    for ax, bb in zip(axes, MAIN + VOCO):
        for j, ds in enumerate(cohorts):
            if not any((bb, ds, m) in g.index for m in meths):
                ax.axhspan(j - 0.5, j + 0.5, color=BAND, zorder=2, lw=0)   # above the zero line and grid
                ax.text(0, j, "no runs", ha="center", va="center", fontsize=7.0, color=INK2, zorder=3)
            for k, mt in enumerate(meths):
                if (bb, ds, mt) in g.index:
                    mscatter(ax, [g.loc[(bb, ds, mt)]], [j + (k - 2) * 0.16], mt, s=15)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_title(BACKBONE_TEX[bb], fontsize=7, color=INK)
        tidy(ax)
        ax.set_xlim(-lim, lim); ax.set_xticks([-0.1, 0, 0.1]); ax.set_xticklabels(["\u22120.1", "0", "0.1"])
    axes[0].set_yticks(range(len(cohorts))); axes[0].set_yticklabels([COHORT_SHORT[c] for c in cohorts], fontsize=6.3)
    axes[0].set_ylim(len(cohorts) - 0.5, -0.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("joint \u2212 per-task held-out AUROC (mean over the cohort's labels)", fontsize=6.5, y=0.1)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.12, 1, 1))
    fig.legend(handles=[mhandle(mt) for mt in meths], frameon=False, fontsize=6.3, loc="lower center", ncol=len(meths),
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.2, columnspacing=1.2)
    save(fig, "fig_joint")


# ============================================================================ fig_methods
def fig_methods() -> None:
    d = paired_cls("lin")
    bbs = MAIN + VOCO
    d = d[d.backbone.isin(bbs)]
    bad = dnt()
    meths = [m for m in CLS_METHODS if m != "lin"]
    fig, axes = plt.subplots(3, 4, figsize=(W, 4.45), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, mt in zip(axes, meths):
        sub = d[d.method == mt]
        for j, ds in enumerate(CLS_COHORTS):
            here = [bb for bb in bbs if len(sub[(sub.cohort == ds) & (sub.backbone == bb)])]   # backbones in this row
            for k, bb in enumerate(bbs):
                s = sub[(sub.cohort == ds) & (sub.backbone == bb)]["diff"].values
                if len(s) == 0:
                    continue
                y = j + (here.index(bb) - (len(here) - 1) / 2) * 0.36   # backbones of a row offset vertically
                jit = (np.random.default_rng(j * 3 + k).random(len(s)) - 0.5) * 0.14
                off = (bb, mt) in bad
                ax.scatter(s, y + jit, s=3.5, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], alpha=0.45, linewidths=0, zorder=2)
                ax.scatter([s.mean()], [y], s=17, color="white" if off else BACKBONE_COL[bb], marker=BACKBONE_MARK[bb],
                           edgecolor=BACKBONE_COL[bb] if off else "white", linewidths=0.8, zorder=3)
        trained = sub[[(b, mt) not in bad for b in sub.backbone]]
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.set_title(f"{MTEX[mt]}\nmean {um(fmt_signed(trained['diff'].mean()))}, {100 * (trained['diff'] > 0).mean():.0f}% > 0",
                     fontsize=6.4, color=INK, loc="left")
        tidy(ax)
        for bb in MAIN:
            sb = trained[trained.backbone == bb]
            if len(sb):
                check("fig_methods", f"{mt} {bb} mean (part of title)", fmt_signed(sb["diff"].mean()), mname("paired", bb, "lin", mt, "meandiff"))
                check("fig_methods", f"{mt} {bb} % ahead (part of title)", f"{100 * (sb['diff'] > 0).mean():.0f}", mname("paired", bb, "lin", mt, "better"))
                check("fig_methods", f"{mt} {bb} n", str(len(sb)), mname("paired", bb, "lin", mt, "n"))
        sv = trained[trained.backbone == "voco"]
        if not NO_VOCO:
            CHECKS.append(f"INFO    fig_methods          {mt} voco: mean {fmt_signed(sv['diff'].mean())} over {len(sv)} label-mode pairs (no label-mode macro for VoCo)")
        CHECKS.append(f"INFO    fig_methods          {mt} title: mean {fmt_signed(trained['diff'].mean())}, "
                      f"{100 * (trained['diff'] > 0).mean():.0f}% > 0 over {len(trained)} label-mode pairs of paired_cls.csv (no pooled macro)")
    axes[0].set_yticks(range(len(CLS_COHORTS))); axes[0].set_yticklabels([COHORT_SHORT[c] for c in CLS_COHORTS], fontsize=6.2)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.45, 0.45); axes[0].set_xticks([-0.25, 0, 0.25])
    for k in (6, 7):   # lowest panel of the last two columns (the bottom row there holds the legend)
        axes[k].tick_params(labelbottom=True)
    axes[9].set_xlabel("\u0394 held-out AUROC vs. linear probe of the same backbone, cohort and mode", fontsize=6.4)
    axes[9].xaxis.set_label_coords(0.0, -0.24)
    axes[11].axis("off")
    ax = axes[10]; ax.axis("off")
    hs = [Line2D([], [], color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], ls="", ms=4.2, label=BACKBONE_TEX[bb]) for bb in bbs]
    hs.append(Line2D([], [], color=BACKBONE_COL["spectre"], marker=BACKBONE_MARK["spectre"], mfc="white", ls="", ms=4.2, label="did not train"))
    ax.legend(handles=hs, frameon=False, fontsize=6.3, loc="upper left", title="backbone", title_fontsize=6.4)
    axes[11].text(0.02, 0.98, "small: one label\u2013mode pair\nlarge: cohort mean\ntitle: mean \u0394, share above 0\n(all backbones)", transform=axes[11].transAxes,
            fontsize=6.0, color=INK2, va="top")
    fig.tight_layout(w_pad=0.5, h_pad=0.8)
    save(fig, "fig_methods")


# ============================================================================ fig_cls (main text)
def fig_cls() -> None:
    """Main-text counterpart of fig_seg: per-label change in held-out AUROC relative to the linear probe for the frozen
    attentive probe, LoRA and attention-only fine-tuning (full fine-tuning, which did not train on SPECTRE, is shown with
    all methods in fig_methods); same data and checks as the matching fig_methods panels."""
    d = paired_cls("lin")
    bbs = MAIN + VOCO
    d = d[d.backbone.isin(bbs)]
    bad = dnt()
    meths = ["attn", "lora", "attnft"]
    fig, axes = plt.subplots(1, len(meths), figsize=(W, 1.8), sharex=True, sharey=True)
    for pi, (ax, mt) in enumerate(zip(axes, meths)):
        sub = d[d.method == mt]
        for j, ds in enumerate(CLS_COHORTS):
            here = [bb for bb in bbs if len(sub[(sub.cohort == ds) & (sub.backbone == bb)])]   # backbones in this row
            for k, bb in enumerate(bbs):
                s = sub[(sub.cohort == ds) & (sub.backbone == bb)]["diff"].values
                if len(s) == 0:
                    continue
                y = j + (here.index(bb) - (len(here) - 1) / 2) * 0.36   # backbones of a row offset vertically
                jit = (np.random.default_rng(j * 3 + k).random(len(s)) - 0.5) * 0.14
                off = (bb, mt) in bad
                ax.scatter(s, y + jit, s=3.5, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], alpha=0.45, linewidths=0, zorder=2)
                ax.scatter([s.mean()], [y], s=17, color="white" if off else BACKBONE_COL[bb], marker=BACKBONE_MARK[bb],
                           edgecolor=BACKBONE_COL[bb] if off else "white", linewidths=0.8, zorder=3)
        trained = sub[[(b, mt) not in bad for b in sub.backbone]]
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.set_title(f"({'abcd'[pi]}) {MTEX[mt]} vs. Lin\nmean {um(fmt_signed(trained['diff'].mean()))}, {100 * (trained['diff'] > 0).mean():.0f}% > 0",
                     fontsize=6.6, color=INK, loc="left")
        tidy(ax)
        for bb in MAIN:
            sb = trained[trained.backbone == bb]
            if len(sb):
                check("fig_cls", f"{mt} {bb} mean", fmt_signed(sb["diff"].mean()), mname("paired", bb, "lin", mt, "meandiff"))
                check("fig_cls", f"{mt} {bb} n", str(len(sb)), mname("paired", bb, "lin", mt, "n"))
        CHECKS.append(f"INFO    fig_cls              {mt} title: mean {fmt_signed(trained['diff'].mean())}, "
                      f"{100 * (trained['diff'] > 0).mean():.0f}% > 0 over {len(trained)} label-mode pairs of paired_cls.csv")
    axes[0].set_yticks(range(len(CLS_COHORTS))); axes[0].set_yticklabels([COHORT_SHORT[c] for c in CLS_COHORTS], fontsize=6.3)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.45, 0.45); axes[0].set_xticks([-0.25, 0, 0.25])
    for ax in axes:
        ax.set_xlabel("\u0394 held-out AUROC", fontsize=6.4)
    hs = [Line2D([], [], color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], ls="", ms=4.2, label=BACKBONE_TEX[bb]) for bb in bbs]
    if any(mt in meths for _, mt in bad):
        hs.append(Line2D([], [], color=BACKBONE_COL["spectre"], marker=BACKBONE_MARK["spectre"], mfc="white", ls="", ms=4.2, label="did not train"))
    fig.legend(handles=hs, frameon=False, fontsize=6.3, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.08, 1, 1))
    save(fig, "fig_cls")


# ============================================================================ fig_seg_methods
SBBS = ["tapct", "spectre"] + VOCO   # segmentation backbones drawn


def fig_seg_methods() -> None:
    d = seg_split()
    piv = d.pivot_table(index=["backbone", "cohort", "label", "mode"], columns="method", values="dice")
    meths = ["dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.6), sharex=True, sharey=True)
    for ax, mt in zip(axes.ravel(), meths):
        allx = []
        for j, ds in enumerate(SEG_DATASETS):
            for k, bb in enumerate(SBBS):
                sub = piv.xs((bb, ds), level=("backbone", "cohort"))
                x = (sub[mt] - sub["lin"]).dropna().values
                if not len(x):
                    continue
                allx.extend(x)
                y = j + (k - (len(SBBS) - 1) / 2) * 0.26
                jit = (np.random.default_rng(j * 2 + k).random(len(x)) - 0.5) * 0.18
                ax.scatter(x, y + jit, s=3.5, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], alpha=0.45, linewidths=0, zorder=2)
                ax.scatter([x.mean()], [y], s=17, color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], edgecolor="white", linewidths=0.8, zorder=3)
        for bb in ["tapct", "spectre"]:
            x = (piv.xs(bb, level="backbone")[mt] - piv.xs(bb, level="backbone")["lin"]).dropna()
            check("fig_seg_methods", f"{mt} {bb} - Lin mean (part of title)", fmt_signed(x.mean()), mname("segdelta", bb, "lin", mt, "meandiff"))
            check("fig_seg_methods", f"{mt} {bb} - Lin % ahead (part of title)", f"{100 * (x > 0).mean():.0f}", mname("segdelta", bb, "lin", mt, "better"))
        if not NO_VOCO:
            xv = (piv.xs("voco", level="backbone")[mt] - piv.xs("voco", level="backbone")["lin"]).dropna()
            CHECKS.append(f"INFO    fig_seg_methods      {mt} voco - Lin: mean {fmt_signed(xv.mean())} over {len(xv)} structure-mode pairs (no segdelta macro for VoCo)")
        allx = np.array(allx)
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.set_title(f"{MTEX[mt]}\nmean {um(fmt_signed(allx.mean()))}, {100 * (allx > 0).mean():.0f}% > 0", fontsize=7.0, color=INK, loc="left")
        CHECKS.append(f"INFO    fig_seg_methods      {mt} title: mean {fmt_signed(allx.mean())}, {100 * (allx > 0).mean():.0f}% > 0 over "
                      f"{len(allx)} structure-mode pairs of metrics_seg.csv ({'TAP-CT and SPECTRE' if NO_VOCO else 'TAP-CT, SPECTRE and VoCo'} pooled; no pooled macro)")
        tidy(ax)
    axes[0, 0].set_yticks(range(len(SEG_DATASETS))); axes[0, 0].set_yticklabels([DS_SHORT[x] for x in SEG_DATASETS], fontsize=6.2)
    axes[0, 0].invert_yaxis()
    if NO_VOCO:   # without VoCo's large gains over its probe the axis ends at the largest TAP-CT / SPECTRE difference
        axes[0, 0].set_xlim(-0.62, 0.62); axes[0, 0].set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    else:
        axes[0, 0].set_xlim(-0.62, 1.0); axes[0, 0].set_xticks([-0.4, 0.0, 0.4, 0.8])
    axes[1, 1].set_xlabel("\u0394 held-out Dice vs. linear probe of the same backbone, dataset and mode", fontsize=6.4)
    fig.tight_layout(w_pad=1.0, h_pad=0.8, rect=(0, 0.06, 1, 1))
    fig.legend(handles=[Line2D([], [], color=BACKBONE_COL[bb], marker=BACKBONE_MARK[bb], ls="", ms=4.2, label=BACKBONE_TEX[bb])
                        for bb in SBBS], frameon=False, fontsize=6.4, loc="lower center", ncol=len(SBBS), bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_seg_methods")


# ============================================================================ fig_seg_budget
def fig_seg_budget() -> None:
    fig, ax = plt.subplots(figsize=(0.62 * W, 2.3))  # included at 0.62 of the text width
    meths = ["dec", "lora", "fullft"]
    for bb, ls in (("tapct", "-"), ("spectre", "--")):
        for mt in meths:
            for mode in ("joint_8k", "joint"):
                r = runs("seg", cohort="amos22", backbone=bb, method=mt, mode=mode)
                if len(r) != 1:
                    raise SystemExit(f"fig_seg_budget: {len(r)} runs for amos22 {bb} {mt} {mode}")
                t = val_curve(r.path.iloc[0], "val_dice")
                its = np.array(sorted(t)) * SEG_STEPS_PER_ITER / 1000
                thick = mode == "joint_8k"
                ax.plot(its, [t[k] for k in sorted(t)], ls, color=MCOL[mt], lw=1.6 if thick else 0.9, alpha=1.0 if thick else 0.75)
                CHECKS.append(f"INFO    fig_seg_budget       amos22 {bb} {mt} {mode}: {len(t)} evaluations, last at {its[-1]:.0f}k steps")
    ax.axvline(8, color=INK2, lw=0.6, ls=":")
    ax.text(8.4, 0.83, "8000-step budget", fontsize=6.0, color=INK2, va="top")
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel("optimizer steps (\u00d71000)", fontsize=6.5); ax.set_ylabel("validation Dice", fontsize=6.5)
    ax.set_title("AMOS22 joint model (15 structures)", fontsize=7, color=INK, loc="left")
    hs = [Line2D([], [], color=MCOL[mt], lw=1.6, label=MTEX[mt]) for mt in meths]
    hs += [Line2D([], [], color=INK, lw=1.0, ls="-", label="TAP-CT"), Line2D([], [], color=INK, lw=1.0, ls="--", label="SPECTRE")]
    hs += [Line2D([], [], color=INK, lw=1.6, label="8000 steps"), Line2D([], [], color=INK, lw=0.9, alpha=0.75, label="24 000 steps")]
    ax.legend(handles=hs, frameon=False, fontsize=7.0, ncol=3, loc="lower right", handlelength=1.6, columnspacing=0.8)
    tidy(ax, "both")
    fig.tight_layout()
    save(fig, "fig_seg_budget")


# ============================================================================ fig_seg_convergence
def fig_seg_convergence() -> None:
    meths = ["lin", "dec", "lora", "fullft"]
    r = runs("seg", backbone=["tapct", "spectre"], method=meths, mode=["pertask", "joint", "joint_8k"])
    groups = [("per-structure\n8000 steps", lambda m, ds: m == "pertask"),
              ("joint, 2\u20133 structures\n8000 steps", lambda m, ds: m == "joint" and ds not in LONG_JOINT),
              ("joint, 15 structures\n8000 steps", lambda m, ds: m == "joint_8k"),
              ("joint, 15 structures\n24 000 steps", lambda m, ds: m == "joint" and ds in LONG_JOINT)]
    curves: dict = {}
    for _, row in r.iterrows():
        vals = val_curve(row.path, "val_dice")
        if len(vals) < 4:
            continue
        its = sorted(vals)
        for gi, (_, sel) in enumerate(groups):
            if sel(row["mode"], row.cohort):
                curves.setdefault((gi, row.backbone, row.method), []).append(
                    (np.array(its) * SEG_STEPS_PER_ITER / 1000, np.array([vals[k] - vals[its[-1]] for k in its])))
    for (gi, bb, mt), rr in sorted(curves.items()):
        CHECKS.append(f"INFO    fig_seg_convergence  group {gi} {bb} {mt}: {len(rr)} runs")
    med = {k: np.median(np.stack([y[:min(len(x) for x, _ in rr)] for _, y in rr]), axis=0) for k, rr in curves.items()}
    ylo = min(-0.3, float(np.floor(min(v.min() for v in med.values()) * 20) / 20) - 0.02)
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.15), sharey=True)
    for gi, (title, _) in enumerate(groups):
        ax = axes[gi]
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        for (g, bb, mt), rr in curves.items():
            if g != gi:
                continue
            nmin = min(len(x) for x, _ in rr)
            ax.plot(rr[0][0][:nmin], med[(g, bb, mt)], "-" if bb == "tapct" else "--", color=MCOL[mt], lw=1.2)
        ax.set_title(title, fontsize=6.4, color=INK, loc="left")
        ax.set_xlabel("optimizer steps (\u00d71000)", fontsize=6.3)
        ax.set_ylim(ylo, 0.03)
        tidy(ax, "both")
    axes[0].set_ylabel("val. Dice \u2212 last evaluation", fontsize=6.3)
    hs = [Line2D([], [], color=MCOL[mt], lw=1.5, label=MTEX[mt]) for mt in meths]
    hs += [Line2D([], [], color="#000000", lw=1.2, ls="-", label="TAP-CT"), Line2D([], [], color="#000000", lw=1.2, ls="--", label="SPECTRE")]
    fig.tight_layout(w_pad=0.5, rect=(0, 0.1, 1, 1))
    fig.legend(handles=hs, frameon=False, fontsize=6.3, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, 0.0), handlelength=1.8)
    save(fig, "fig_seg_convergence")


def main() -> None:
    style.setup()
    plt.rcParams.update({"axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    OUT.mkdir(parents=True, exist_ok=True)
    figs = [fig_overview, fig_key, fig_gain_vs_probe, fig_fullft_spectre, fig_seg, fig_cls, fig_joint_main, fig_bbgap, fig_joint, fig_methods,
            fig_seg_methods, fig_seg_budget, fig_seg_convergence]
    extra = [fig_mechanisms]   # kept on disk, no longer in the paper: redrawn only when named on the command line
    only = set(sys.argv[1:])  # optional: names of the figures to redraw (the check file then covers only those)
    for f in figs:
        if not only or f.__name__ in only:
            f()
    for f in extra:
        if f.__name__ in only:
            f()
    write_checks()


if __name__ == "__main__":
    main()
