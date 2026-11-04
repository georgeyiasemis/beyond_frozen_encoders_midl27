"""Results figures of the paper, drawn with matplotlib from the recomputed CSV files in ``../metrics/out``.

Every number drawn comes from those CSVs (see ``RESULTS_FIGURES.md`` for the columns and filters per figure); nothing
is typed by hand. Outputs ``out/fig_key``, ``out/fig_methods_cls_rank``, ``out/fig_methods_seg_rank``,
``out/fig_perlabel_cls``, ``out/fig_perstructure_seg``, ``out/fig_joint``, ``out/fig_seeds``, ``out/fig_convergence``,
``out/fig_revision``, ``out/fig_voco`` (PDF + PNG).

Run:  /home/g.yiasemis/miniconda3/envs/aifofomo/bin/python results_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style
from style import (BACKBONE_COL, BACKBONE_MARK, BACKBONE_TEX, COHORT_COL, COHORT_MARK, COHORT_TEX, FAMILY_COL,
                   HALFWIDTH_IN, INK, INK2, GRID, METHOD_COL, METHOD_FAMILY, METHOD_TEX, TEXTWIDTH_IN)

HERE = Path(__file__).resolve().parent
METRICS = HERE.parent / "metrics" / "out"
OUT = HERE / "out"
MAIN = ["tapct", "spectre", "pillar0"]           # main backbones; voco is a check, never pooled
FS = 6.0                                           # smallest text on any figure
POOLED_COL, POOLED_MARK = INK, "D"
CLS_COHORTS = ["amos_covid_luna", "fdg", "rsna2019_ich", "rsna2022", "rsna2023"]
SEG_DATASETS = ["amos22", "kits23", "lits", "msd_pancreas", "totalseg"]
INTERVAL_NOTE = "Marker: mean over labels (classification) or structures (segmentation); line: three-stage 95% interval (label_level.csv lo3–hi3)."


# ----------------------------------------------------------------------------- data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS / f"{name}.csv")


def did_not_train() -> set[tuple[str, str]]:
    """(backbone, method) pairs of classification runs whose held-out probabilities did not spread (spread.csv)."""
    sp = load("spread")
    return set(map(tuple, sp.loc[sp["did_not_train"].astype(str).str.lower() == "true", ["backbone", "method"]].values))


DNT = did_not_train()


def dnt(task: str, bb: str, method: str) -> bool:
    """spread.csv describes classification only; no segmentation run is flagged."""
    return task == "cls" and (bb, method) in DNT


def main_cls(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out rows of the main design: seed 42, modes pertask/joint, the five classification cohorts."""
    return df[(df["split"] == "test") & (df["seed"] == 42) & df["mode"].isin(["pertask", "joint"])
              & df["cohort"].isin(CLS_COHORTS)]


def main_seg(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["split"] == "test") & (df["seed"] == 42) & df["mode"].isin(["pertask", "joint"])]


def short_label(label: str) -> str:
    s = label.split(":", 1)[-1]
    s = {"lung_nodule_malignancy": "lung nodule", "patient_overall": "overall", "any_injury": "any",
         "NEGATIVE": "negative", "MELANOMA": "melanoma", "LUNG_CANCER": "lung cancer", "LYMPHOMA": "lymphoma",
         "liver_calcifications": "liver calcif."}.get(s, s)
    return s.replace("_injury", "").replace("_", " ")


# ----------------------------------------------------------------------------- drawing helpers
def mark(ax, x, y, color, marker, ms=4.0, open_=False, zorder=4, mew=0.7, alpha=1.0):
    ax.plot([x], [y], marker=marker, ms=ms, mfc="white" if open_ else color, mec=color, mew=mew, ls="none",
            zorder=zorder, alpha=alpha, clip_on=False)


def interval(ax, y, mean, lo, hi, color, marker, xlim, ms=4.0, open_=False, lw=0.8, zorder=4, fmt="{:+.2f}"):
    """Point estimate with interval; parts beyond ``xlim`` are clipped and flagged with a small triangle, an estimate
    beyond the axis is replaced by an arrow at the edge with the value printed."""
    x0, x1 = xlim
    pad = 0.012 * (x1 - x0)
    if mean > x1 or mean < x0:
        at_right = mean > x1
        xe = x1 if at_right else x0
        txt = f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"
        ax.annotate("", xy=(xe, y), xytext=(xe - (pad * 4 if at_right else -pad * 4), y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, mutation_scale=6, shrinkA=0, shrinkB=0))
        ax.text(xe - (pad * 5 if at_right else -pad * 5), y, txt, fontsize=FS, color=color,
                ha="right" if at_right else "left", va="center", zorder=zorder + 1)
        return
    lo_c, hi_c = max(lo, x0), min(hi, x1)
    ax.plot([lo_c, hi_c], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder - 1)
    if lo < x0:
        ax.plot([x0], [y], marker="<", ms=3, mfc=color, mec=color, ls="none", zorder=zorder, clip_on=False)
    if hi > x1:
        ax.plot([x1], [y], marker=">", ms=3, mfc=color, mec=color, ls="none", zorder=zorder, clip_on=False)
    mark(ax, mean, y, color, marker, ms=ms, open_=open_, zorder=zorder)


def beyond(ax, xs, ys, xlim, ms=3.0):
    """Open triangles at the axis edge for values outside ``xlim``; returns the mask of values inside."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    inset = 0.015 * (xlim[1] - xlim[0])
    for side, mk_, xe in [(xs < xlim[0], "<", xlim[0] + inset), (xs > xlim[1], ">", xlim[1] - inset)]:
        if side.any():
            ax.plot([xe] * int(side.sum()), ys[side], marker=mk_, ms=ms, mfc="white", mec=INK2, mew=0.6, ls="none", zorder=3, clip_on=False)
    return (xs >= xlim[0]) & (xs <= xlim[1])


def zero_line(ax, x=0.0):
    ax.axvline(x, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)


def row_bands(ax, n: int, ys=None):
    for i, y in enumerate(ys if ys is not None else range(n)):
        if i % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f4f4", zorder=0, lw=0)


def hide_left(ax):
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def panel_label(ax, text: str, x=0.0, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7.5, fontweight="bold", ha="left", va="bottom", color=INK)


def backbone_handles(bbs, ms=4.0):
    return [Line2D([], [], marker=BACKBONE_MARK[b], color=BACKBONE_COL[b], ls="none", ms=ms, mew=0.7) for b in bbs]


def top_legend(fig, handles, labels, ncol, y=0.99, **kw):
    opts = dict(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=FS, handletextpad=0.3,
                columnspacing=1.0, borderaxespad=0)
    opts.update(kw)
    fig.legend(handles, labels, **opts)


def footnote(fig, text: str, y=-0.02):
    return  # explanations are in the LaTeX captions; figures carry no file or column names
    fig.text(0.0, y, text, fontsize=FS, color=INK2, ha="left", va="top", wrap=True, linespacing=1.25)


# ============================================================================= Fig 1: key comparisons
KEY_ROWS = {
    "cls": [("cls_lora_lin", "LoRA − Lin"), ("cls_lora_attn", "LoRA − Attn"), ("cls_attnft_lora", "Attn-only FT − LoRA"),
            ("cls_fullft_lin", "Full FT − Lin"), ("cls_joint_pertask", "Joint − per-task")],
    "seg": [("seg_dec_lin", "Dec − Lin"), ("seg_lora_dec", "LoRA − Dec"), ("seg_lora_dec_pertask", "LoRA − Dec, per-task"),
            ("seg_lora_dec_joint", "LoRA − Dec, joint"), ("seg_fullft_dec", "Full FT − Dec"),
            ("seg_joint_pertask", "Joint − per-task")],
}


def key_is_dnt(key: str, bb: str) -> bool:
    """A label-level key compares <method>_<baseline>; the estimate is void when either did not train on ``bb``."""
    task, *parts = key.split("_")
    return any(dnt(task, bb, m) for m in parts)


def fig_key():
    ll = load("label_level")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.55), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.95))
    xlims = {"cls": (-0.21, 0.12), "seg": (-0.12, 0.17)}
    xlabels = {"cls": "Δ held-out AUROC", "seg": "Δ held-out Dice"}
    print("\n=== fig_key: rows plotted (label_level.csv) ===")
    for ax, task, lab in zip(axes, ["cls", "seg"], ["a", "b"]):
        rows = KEY_ROWS[task]
        n = len(rows)
        bbs_all = [b for b in ["tapct", "spectre", "pillar0", "voco"] if ((ll["task"] == task) & (ll["backbone"] == b)).any()]
        order = ["all"] + bbs_all
        step = 0.72 / len(order)
        offs = {b: 0.36 - step / 2 - i * step for i, b in enumerate(order)}
        row_bands(ax, n)
        for i, (key, name) in enumerate(rows):
            y = n - 1 - i
            sub = ll[ll["key"] == key].set_index("backbone")
            for b in order:
                if b not in sub.index:
                    continue
                r = sub.loc[b]
                col = POOLED_COL if b == "all" else BACKBONE_COL[b]
                mk = POOLED_MARK if b == "all" else BACKBONE_MARK[b]
                ms = 4.6 if b == "all" else 3.6
                open_ = b != "all" and key_is_dnt(key, b)
                interval(ax, y + offs[b], r["mean"], r["lo3"], r["hi3"], col, mk, xlims[task], ms=ms, open_=open_)
                print(f"{key:22s} {b:8s} n={int(r['n_labels']):3d} mean={r['mean']:+.4f} lo3={r['lo3']:+.4f} hi3={r['hi3']:+.4f}"
                      + ("  (did not train)" if open_ else ""))
        zero_line(ax)
        ax.set_xlim(*xlims[task]); ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticks(range(n)); ax.set_yticklabels([name for _, name in rows][::-1])
        ax.set_xlabel(xlabels[task])
        hide_left(ax)
        panel_label(ax, f"({lab}) {'classification' if task == 'cls' else 'segmentation'}")
    handles = [Line2D([], [], marker=POOLED_MARK, color=POOLED_COL, ls="none", ms=4.6)] + backbone_handles(["tapct", "spectre", "pillar0", "voco"], 3.6)
    labels = ["pooled (TAP-CT, SPECTRE, Pillar-0)"] + [BACKBONE_TEX[b] for b in ["tapct", "spectre", "pillar0", "voco"]]
    handles.append(Line2D([], [], marker="s", color=INK2, mfc="white", ls="none", ms=3.6)); labels.append("did not train")
    top_legend(fig, handles, labels, 6)
    footnote(fig, INTERVAL_NOTE + " The pooled Full FT − Lin estimate includes SPECTRE, where Full FT did not train.", y=-0.03)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.9, bottom=0.17)
    style.save(fig, OUT, "fig_key")


# ============================================================================= Fig 2/3: methods by rank and by difference to the reference
def methods_rank_fig(task: str, ref: str, methods: list[str], name: str, xlabel_b: str, xlim_b: tuple[float, float]):
    ranks = load("ranks"); ll = load("label_level"); llv = load("label_level_values")
    rk = ranks[(ranks["task"] == task)]
    pooled = rk[rk["backbone"] == "all"].set_index("method")["mean_rank"]
    order = sorted(methods, key=lambda m: pooled[m])          # best (lowest mean rank) first -> top
    n = len(order)
    bbs = [b for b in MAIN if ((rk["backbone"] == b)).any()]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.235 * n + 1.0), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.1))
    notes = []
    # (a) mean rank
    ax = axes[0]
    row_bands(ax, n)
    for i, m in enumerate(order):
        y = n - 1 - i
        fam = FAMILY_COL[METHOD_FAMILY[m]]
        dnt_bbs = [b for b in bbs if dnt(task, b, m)]
        for b in bbs:
            r = rk[(rk["backbone"] == b) & (rk["method"] == m)]
            if len(r) and pd.notna(r["mean_rank"].iloc[0]):
                mark(ax, r["mean_rank"].iloc[0], y, BACKBONE_COL[b], BACKBONE_MARK[b], ms=2.8, zorder=5, mew=0.6)
        mark(ax, pooled[m], y, fam, "o", ms=6.0, open_=bool(dnt_bbs), zorder=4, mew=0.9)
        if dnt_bbs:
            ns = int(rk[(rk["backbone"] == "all") & (rk["method"] == m)]["n_slots"].iloc[0])
            notes.append((m, dnt_bbs, ns))
    ax.set_yticks(range(n)); ax.set_yticklabels([METHOD_TEX[m] for m in order][::-1])
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("mean rank over label–mode slots (1 = best)")
    hide_left(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    panel_label(ax, "(a) rank")
    # (b) label-level difference to the reference
    ax = axes[1]
    row_bands(ax, n)
    nl_note = None
    for i, m in enumerate(order):
        y = n - 1 - i
        fam = FAMILY_COL[METHOD_FAMILY[m]]
        if m == ref:
            ax.text(0.0, y, "reference", fontsize=FS, color=INK2, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
            continue
        key = f"{task}_{m}_{ref}"
        dnt_bbs = [b for b in MAIN if dnt(task, b, m)]
        if dnt_bbs:   # pooled estimate without the backbones that did not train: mean over the remaining label values, no interval
            v = llv[(llv["key"] == key) & llv["backbone"].isin([b for b in MAIN if b not in dnt_bbs])]["value"]
            mark(ax, v.mean(), y, fam, "o", ms=6.0, open_=True, zorder=4, mew=0.9)
            nl_note = (m, len(v))
            continue
        r = ll[(ll["key"] == key) & (ll["backbone"] == "all")].iloc[0]
        interval(ax, y, r["mean"], r["lo3"], r["hi3"], fam, "o", xlim_b, ms=6.0, lw=1.0)
    zero_line(ax)
    ax.set_xlim(*xlim_b); ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([])
    hide_left(ax)
    ax.set_xlabel(xlabel_b)
    panel_label(ax, f"(b) difference to {METHOD_TEX[ref]}")
    fam_h = [Line2D([], [], marker="o", color=FAMILY_COL[f], ls="none", ms=6.0, mec=FAMILY_COL[f]) for f in ["frozen", "lowrank", "fullrank"]]
    fam_l = ["frozen encoder", "low-rank adapter", "trained in full"]
    bb_h = backbone_handles(bbs, 2.8); bb_l = [BACKBONE_TEX[b] for b in bbs]
    top_legend(fig, fam_h + bb_h, fam_l + bb_l, 6)
    txt = INTERVAL_NOTE
    if notes:
        m, dbb, ns = notes[0]
        txt += (f" Open marker: {METHOD_TEX[m]} did not train on {', '.join(BACKBONE_TEX[b] for b in dbb)} (spread.csv); its pooled rank uses "
                f"the other main backbones ({ns} slots)")
        if nl_note:
            txt += f" and its difference is the mean over their {nl_note[1]} labels without an interval."
    footnote(fig, txt, y=-0.03)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.14)
    style.save(fig, OUT, name)


def fig_methods_cls_rank():
    methods_rank_fig("cls", "lin", style.CLS_METHODS, "fig_methods_cls_rank", "Δ held-out AUROC vs Lin", (-0.03, 0.08))


def fig_methods_seg_rank():
    methods_rank_fig("seg", "dec", style.SEG_METHODS, "fig_methods_seg_rank", "Δ held-out Dice vs Dec", (-0.11, 0.08))


# ============================================================================= Fig 4: per-label Lin vs LoRA
def callouts(ax, pts, lim, side: str, step: float):
    """Label points along the left (``side='above'``) or right (``'below'``) margin, stacked without overlap, with leaders."""
    if not pts:
        return
    pts = sorted(pts, key=lambda t: t[1])
    ys = []
    for _, y, _ in pts:
        ys.append(y if not ys else max(y, ys[-1] + step))
    overflow = ys[-1] - (lim[1] - 0.03)
    if overflow > 0:
        ys = [y - overflow for y in ys]
    xt = lim[0] + 0.03 if side == "above" else lim[1] - 0.03
    for (x, y, s), yt in zip(pts, ys):
        ax.annotate(s, xy=(x, y), xytext=(xt, yt), fontsize=FS, color=INK, ha="left" if side == "above" else "right", va="center",
                    zorder=6, arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=2.0))


def fig_perlabel_cls():
    mc = main_cls(load("metrics_cls"))
    pc = load("paired_cls")
    pc = pc[(pc["method"] == "lora") & (pc["baseline"] == "lin") & pc["mode"].isin(["pertask", "joint"])]
    bbs = ["tapct", "spectre", "pillar0", "voco"]
    piv = mc[mc["method"].isin(["lin", "lora"])].pivot_table(index=["backbone", "cohort", "label"], columns="method", values="auroc", aggfunc="mean")
    piv = piv.dropna()
    lo = np.floor(min(piv["lin"].min(), piv["lora"].min()) * 20) / 20 - 0.02
    lim = (lo, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 2.35), gridspec_kw=dict(wspace=0.12))
    for ax, b in zip(axes, bbs):
        d = piv.loc[b]
        ax.plot(lim, lim, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        above, below = [], []
        for (coh, lab), r in d.iterrows():
            mark(ax, r["lin"], r["lora"], COHORT_COL[coh], COHORT_MARK[coh], ms=3.4, zorder=3, mew=0.5)
            if abs(r["lora"] - r["lin"]) > 0.05:
                (above if r["lora"] > r["lin"] else below).append((r["lin"], r["lora"], short_label(lab)))
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_title(BACKBONE_TEX[b], color=BACKBONE_COL[b], fontsize=7.5, fontweight="bold", pad=3)
        ax.set_xlabel("Lin held-out AUROC")
        ticks = np.arange(0.4, 1.001, 0.2)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        if ax is axes[0]:
            ax.set_ylabel("LoRA held-out AUROC")
        else:
            ax.set_yticklabels([])
        # significance strip from paired_cls.csv (BH-adjusted DeLong, label-mode pairs)
        p = pc[pc["backbone"] == b]
        sig = p["p_bh"] < 0.05
        better, worse, n = int((sig & (p["diff"] > 0)).sum()), int((sig & (p["diff"] < 0)).sum()), len(p)
        ax.text(0.5, -0.3, f"{n} pairs; LoRA significantly\nahead {100 * better / n:.0f}%, behind {100 * worse / n:.0f}%",
                transform=ax.transAxes, fontsize=FS, ha="center", va="top", color=INK, linespacing=1.3)
    handles = [Line2D([], [], marker=COHORT_MARK[c], color=COHORT_COL[c], ls="none", ms=3.4, mew=0.5) for c in CLS_COHORTS]
    top_legend(fig, handles, [COHORT_TEX[c] for c in CLS_COHORTS], 5, y=1.0)
    footnote(fig, "One marker per label: held-out AUROC averaged over the per-task and joint runs (seed 42); labels with |Δ| > 0.05 named. "
                  "Strip: share of label–mode pairs whose paired DeLong test (paired_cls.csv, p_bh) favours LoRA or Lin.", y=-0.03)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.3)
    style.save(fig, OUT, "fig_perlabel_cls")


# ============================================================================= Fig 5: per-structure Δ Dice vs Dec
def perstructure_panel(ax, d: pd.DataFrame, methods: list[str], xlim, show_labels: bool, rng):
    """``d``: held-out rows of one backbone (modes pertask/joint, seed 42). Grey points per structure–mode pair, the mean as a
    coloured marker with its value printed."""
    piv = d.pivot_table(index=["cohort", "label", "mode"], columns="method", values="dice")
    n = len(methods)
    row_bands(ax, n)
    for i, m in enumerate(methods):
        y = n - 1 - i
        if m not in piv.columns:
            continue
        delta = (piv[m] - piv["dec"]).dropna()
        jit = rng.uniform(-0.22, 0.22, len(delta))
        inside = beyond(ax, delta.values, y + jit, xlim)
        ax.plot(delta.values[inside], y + jit[inside], marker="o", ms=2.0, mfc="#bbbbbb", mec="none", ls="none", alpha=0.8, zorder=2)
        mu = delta.mean()
        mark(ax, min(max(mu, xlim[0]), xlim[1]), y, METHOD_COL[m], "D", ms=4.6, zorder=5, mew=0.8)
        ax.text(xlim[1] + 0.03 * (xlim[1] - xlim[0]), y, ("0.000" if abs(mu) < 5e-4 else f"{mu:+.3f}"), fontsize=FS, color=INK, ha="left", va="center")
    zero_line(ax)
    ax.set_xlim(*xlim); ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([METHOD_TEX[m] for m in methods][::-1] if show_labels else [])
    hide_left(ax)


def fig_perstructure_seg():
    ms = main_seg(load("metrics_seg"))
    methods = ["lin", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
    bbs = ["tapct", "spectre", "voco"]
    xlim = (-0.6, 0.6)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 1.9), gridspec_kw=dict(wspace=0.42))
    rng = np.random.default_rng(0)
    for k, (ax, b) in enumerate(zip(axes, bbs)):
        perstructure_panel(ax, ms[ms["backbone"] == b], methods, xlim, show_labels=(k == 0), rng=rng)
        ax.set_title(BACKBONE_TEX[b], color=BACKBONE_COL[b], fontsize=7.5, fontweight="bold", pad=3)
        ax.set_xlabel("Δ held-out Dice vs Dec")
        ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    handles = [Line2D([], [], marker="o", color="#bbbbbb", ls="none", ms=2.0), Line2D([], [], marker="D", color=INK2, ls="none", ms=4.6),
               Line2D([], [], marker=">", color=INK2, mfc="white", ls="none", ms=3.0)]
    labels = ["one structure–mode pair (per-task and joint)", "mean (value printed)", "beyond the axis"]
    top_legend(fig, handles, labels, 3, y=1.0, columnspacing=1.2)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.22)
    style.save(fig, OUT, "fig_perstructure_seg")


# ============================================================================= Fig 6: joint minus per-task
def fig_joint():
    cm = load("cohort_means")
    cm = cm[(cm["seed"] == 42) & cm["mode"].isin(["pertask", "joint"])]
    bbs = ["tapct", "spectre", "pillar0", "voco"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.3), gridspec_kw=dict(wspace=0.75))
    xlims = {"cls": (-0.16, 0.16), "seg": (-0.28, 0.12)}
    for ax, task, rows, lab, xl in [(axes[0], "cls", CLS_COHORTS, "(a) classification", "Δ held-out AUROC, joint − per-task"),
                                    (axes[1], "seg", SEG_DATASETS, "(b) segmentation", "Δ held-out Dice, joint − per-task")]:
        n = len(rows)
        row_bands(ax, n)
        d = cm[cm["task"] == task]
        for i, coh in enumerate(rows):
            y = n - 1 - i
            present = [b for b in bbs if ((d["cohort"] == coh) & (d["backbone"] == b)).any()]
            step = 0.72 / max(len(present), 1)
            for j, b in enumerate(present):
                yy = y + 0.36 - step / 2 - j * step
                p = d[(d["cohort"] == coh) & (d["backbone"] == b)].pivot_table(index="method", columns="mode", values="value")
                p = p.dropna()
                p = p[[not dnt(task, b, m) for m in p.index]]
                if p.empty:
                    continue
                delta = p["joint"] - p["pertask"]
                inside = beyond(ax, delta.values, [yy] * len(delta), xlims[task])
                ax.plot(delta.values[inside], [yy] * int(inside.sum()), marker=BACKBONE_MARK[b], ms=2.2, mfc=BACKBONE_COL[b], mec="none",
                        ls="none", alpha=0.45, zorder=3)
                mark(ax, delta.mean(), yy, BACKBONE_COL[b], BACKBONE_MARK[b], ms=4.6, zorder=5)
        zero_line(ax)
        ax.set_xlim(*xlims[task])
        ax.set_ylim(-0.5, n - 0.5); ax.set_yticks(range(n)); ax.set_yticklabels([COHORT_TEX[c] for c in rows][::-1])
        hide_left(ax)
        ax.set_xlabel(xl)
        panel_label(ax, lab)
    handles = backbone_handles(bbs, 4.6) + [Line2D([], [], marker="o", color=INK2, mfc=INK2, mec="none", alpha=0.45, ls="none", ms=2.2),
                                           Line2D([], [], marker=">", color=INK2, mfc="white", ls="none", ms=3.0)]
    labels = [BACKBONE_TEX[b] for b in bbs] + ["one method", "beyond the axis"]
    top_legend(fig, handles, labels, 6, columnspacing=1.2)
    footnote(fig, "Large marker: mean over the methods with both modes (cohort_means.csv, seed 42; classification pairs that did not train excluded). "
                  "Segmentation 'joint' of AMOS22 and TotalSegmentator is the 24 000-step model.", y=-0.03)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.88, bottom=0.17)
    style.save(fig, OUT, "fig_joint")


# ============================================================================= Fig 7: seeds
def fig_seeds():
    sd = load("seeds")
    panels = [("cls", "tapct"), ("cls", "spectre"), ("cls", "pillar0"), None, ("seg", "tapct"), ("seg", "spectre")]
    methods = {"cls": ["lin", "lora", "attnft", "fullft"], "seg": ["dec", "lora", "fullft"]}
    fig, axes = plt.subplots(1, 6, figsize=(TEXTWIDTH_IN, 2.3), gridspec_kw=dict(wspace=0.22, width_ratios=[4, 4, 4, 1.0, 3, 3]))
    axes[3].axis("off")
    single_seed = []
    for ax, pan in zip(axes, panels):
        if pan is None:
            continue
        task, b = pan
        d = sd[(sd["task"] == task) & (sd["backbone"] == b)]
        ms_ = methods[task]
        cohorts = [c for c in (CLS_COHORTS if task == "cls" else SEG_DATASETS) if (d["cohort"] == c).any()]
        for coh in cohorts:
            dc = d[d["cohort"] == coh]
            means = []
            for k, m in enumerate(ms_):
                v = dc[dc["method"] == m]["value"].values
                if len(v) == 0:
                    means.append(np.nan); continue
                open_ = dnt(task, b, m)
                if open_:
                    single_seed.append((BACKBONE_TEX[b], METHOD_TEX[m], len(v)))
                ax.plot([k] * len(v), v, marker=COHORT_MARK[coh], ms=2.8, mfc="white" if open_ else COHORT_COL[coh],
                        mec=INK2 if open_ else COHORT_COL[coh], mew=0.6, ls="none", zorder=4)
                means.append(v.mean())
            ax.plot(range(len(ms_)), means, color=COHORT_COL[coh], lw=0.7, zorder=3)
        ax.set_xticks(range(len(ms_))); ax.set_xticklabels([METHOD_TEX[m] for m in ms_], rotation=50, ha="right", rotation_mode="anchor", fontsize=FS)
        ax.set_xlim(-0.4, len(ms_) - 0.6)
        ax.set_title(BACKBONE_TEX[b], color=BACKBONE_COL[b], fontsize=7.5, fontweight="bold", pad=3)
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    for ax in axes[:3]:
        ax.set_ylim(0.5, 0.95)
    for ax in axes[4:]:
        ax.set_ylim(0.42, 0.85)
    for ax in (axes[1], axes[2], axes[5]):
        ax.set_yticklabels([])
    axes[0].set_ylabel("held-out AUROC (cohort mean)")
    axes[4].set_ylabel("held-out Dice (dataset mean)")
    panel_label(axes[0], "(a) classification, joint models, seeds 42/43/44", y=1.14)
    panel_label(axes[4], "(b) segmentation, joint models", y=1.14)
    hc = [Line2D([], [], marker=COHORT_MARK[c], color=COHORT_COL[c], ls="-", lw=0.7, ms=2.8, mew=0.6) for c in CLS_COHORTS]
    hs = [Line2D([], [], marker=COHORT_MARK[c], color=COHORT_COL[c], ls="-", lw=0.7, ms=2.8, mew=0.6) for c in SEG_DATASETS]
    fig.canvas.draw()
    xa = (axes[0].get_position().x0 + axes[2].get_position().x1) / 2
    xb = (axes[4].get_position().x0 + axes[5].get_position().x1) / 2
    fig.legend(hc, [COHORT_TEX[c] for c in CLS_COHORTS], loc="lower center", bbox_to_anchor=(xa, 1.04), ncol=3, fontsize=FS,
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0, handlelength=1.6, frameon=False)
    fig.legend(hs, [COHORT_TEX[c] for c in SEG_DATASETS], loc="lower center", bbox_to_anchor=(xb, 1.04), ncol=2, fontsize=FS,
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0, handlelength=1.6, frameon=False)
    txt = "Small markers: the cohort mean of each seed (seeds.csv); lines join the seed means per cohort."
    if single_seed:
        b_, m_, k_ = single_seed[0]
        txt += f" Open markers: {m_} on {b_} did not train ({k_} seed in seeds.csv)."
    footnote(fig, txt, y=-0.12)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.2)
    style.save(fig, OUT, "fig_seeds")


# ============================================================================= Fig 8: convergence
def fig_convergence():
    cv = load("convergence")
    main_methods = set(style.CLS_METHODS) | set(style.SEG_METHODS)
    cv = cv[cv["method"].isin(main_methods) & cv["mode"].isin(["pertask", "joint"])]
    rows = []
    for task, bbs in [("cls", ["tapct", "spectre", "pillar0", "voco"]), ("seg", ["tapct", "spectre", "voco"])]:
        for b in bbs:
            for mode in ["pertask", "joint"]:
                d = cv[(cv["task"] == task) & (cv["backbone"] == b) & (cv["mode"] == mode)]
                if d.empty:
                    continue
                w = d["n_runs"].values
                rows.append((task, b, mode, int(w.sum()), np.average(d["frac_last"], weights=w), np.average(d["frac_last_two"], weights=w)))
    n_cls = sum(1 for r in rows if r[0] == "cls")
    gap = 1.0
    ys = [len(rows) - 1 - i + (gap if r[0] == "cls" else 0) for i, r in enumerate(rows)]
    top = ys[0]
    fig, ax = plt.subplots(figsize=(HALFWIDTH_IN, 0.16 * (len(rows) + gap) + 0.95))
    row_bands(ax, len(rows), ys)
    for (task, b, mode, nr, f1, f2), y in zip(rows, ys):
        ax.plot([f1, f2], [y, y], color=BACKBONE_COL[b], lw=0.8, zorder=2)
        mark(ax, f1, y, BACKBONE_COL[b], BACKBONE_MARK[b], ms=3.8, zorder=4)
        mark(ax, f2, y, BACKBONE_COL[b], BACKBONE_MARK[b], ms=3.8, open_=True, zorder=4)
        ax.text(1.03, y, f"{nr}", fontsize=FS, color=INK2, ha="left", va="center")
    ax.text(1.03, top + 0.6, "runs", fontsize=FS, color=INK2, ha="left", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{BACKBONE_TEX[b]}, {'per-task' if mode == 'pertask' else 'joint'}" for _, b, mode, *_ in rows])
    ax.set_ylim(-0.5, top + 0.5); ax.set_xlim(0, 1)
    hide_left(ax)
    ax.set_xlabel("share of runs selecting the last (filled)\nor one of the last two (open) validations")
    ax.text(-0.02, top + 0.6, "classification", transform=ax.get_yaxis_transform(), fontsize=FS + 0.5, ha="right", va="center", color=INK, fontweight="bold")
    ax.text(-0.02, ys[n_cls] + 1.0, "segmentation", transform=ax.get_yaxis_transform(), fontsize=FS + 0.5, ha="right", va="center", color=INK, fontweight="bold")
    footnote(fig, "convergence.csv: main methods, per-task and joint runs, weighted by n_runs.", y=-0.03)
    fig.subplots_adjust(left=0.42, right=0.9, top=0.93, bottom=0.2)
    style.save(fig, OUT, "fig_convergence")


# ============================================================================= Fig 9: revision experiments
def cohort_mean(df: pd.DataFrame, value: str) -> pd.Series:
    """Mean over labels/structures of a held-out metric per (backbone, cohort, mode, method); complete cells only."""
    return df[df["split"] == "test"].groupby(["backbone", "cohort", "mode", "method"])[value].mean()


def check_against_cohort_means(means: pd.Series, cm: pd.Series, task: str, where: str):
    for (b, c, mo, m), v in means.items():
        ref = cm.get((task, b, c, mo, m))
        if ref is not None and abs(ref - v) > 1e-6:
            print(f"WARNING {where}: {b}/{c}/{mo}/{m} mean {v:.5f} differs from cohort_means {ref:.5f}")


def two_line(b: str, c: str) -> str:
    return f"{BACKBONE_TEX[b]}\n{COHORT_TEX[c]}"


def fig_revision():
    rv = load("revision"); mc = load("metrics_cls"); ms = load("metrics_seg"); cm = load("cohort_means")
    cm = cm[cm["seed"] == 42].set_index(["task", "backbone", "cohort", "mode", "method"])["value"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.9), gridspec_kw=dict(width_ratios=[1.3, 1.0, 1.0], wspace=1.0))
    # (a) full fine-tuning variants vs Lin and LoRA, joint classification models
    ax = axes[0]
    mcj = mc[(mc["split"] == "test") & (mc["seed"] == 42) & (mc["mode"] == "joint") & mc["method"].isin(["lin", "lora", "fullft"]) & mc["cohort"].isin(CLS_COHORTS)]
    rvj = rv[(rv["task"] == "cls") & (rv["mode"] == "joint") & rv["method"].isin(["fullft_lowlr", "fullft_llrd", "fullft_accum"])].rename(columns={"value": "auroc"})
    cols = ["backbone", "cohort", "mode", "method", "split", "auroc"]
    means = cohort_mean(pd.concat([mcj[cols], rvj[cols]]), "auroc")
    check_against_cohort_means(means, cm, "cls", "fig_revision(a)")
    rows = [(b, c) for b in ["tapct", "spectre"] for c in CLS_COHORTS if (b, c, "joint") in means.index.droplevel(3)]
    n = len(rows)
    row_bands(ax, n)
    spec = [("lin", METHOD_COL["lin"], "o", False), ("lora", METHOD_COL["lora"], "o", False), ("fullft", METHOD_COL["fullft"], "o", False),
            ("fullft_lowlr", METHOD_COL["fullft"], "s", True), ("fullft_llrd", METHOD_COL["fullft"], "^", True), ("fullft_accum", METHOD_COL["fullft"], "D", True)]
    for i, (b, c) in enumerate(rows):
        y = n - 1 - i
        for m, col, mk_, open_ in spec:
            if (b, c, "joint", m) in means.index:
                mark(ax, means[(b, c, "joint", m)], y, col, mk_, ms=3.8, open_=open_, zorder=5 if m in ("lin", "lora") else 4)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{BACKBONE_TEX[b]}: {COHORT_TEX[c]}" for b, c in rows][::-1])
    ax.set_ylim(-0.5, n - 0.5); ax.set_xlim(0.45, 1.0)
    hide_left(ax)
    ax.set_xlabel("held-out AUROC, joint model\n(cohort mean)")
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    panel_label(ax, "(a) full fine-tuning settings")
    h = [Line2D([], [], marker=mk_, color=col, mfc="white" if open_ else col, ls="none", ms=3.8, mew=0.7) for _, col, mk_, open_ in spec]
    l = ["Lin", "LoRA", "Full FT (default)", "Full FT, lower LR", "Full FT, layer decay", "Full FT, accumulation"]
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 1.1), ncol=2, fontsize=FS, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    # (b) compute-matched segmentation: joint (24 000 steps) vs per-structure at 1600 steps
    ax = axes[1]
    msj = ms[(ms["split"] == "test") & (ms["seed"] == 42) & (ms["mode"] == "joint") & ms["method"].isin(["dec", "lora", "fullft"]) & ms["cohort"].isin(["amos22", "totalseg"])]
    rvc = rv[(rv["task"] == "seg") & (rv["mode"] == "pertask_cm")].rename(columns={"value": "dice"})
    cols = ["backbone", "cohort", "mode", "method", "split", "dice"]
    means_b = cohort_mean(pd.concat([msj[cols], rvc[cols]]), "dice")
    check_against_cohort_means(means_b, cm, "seg", "fig_revision(b)")
    rows = [(b, c) for b in ["tapct", "spectre"] for c in ["amos22", "totalseg"]]
    n = len(rows)
    row_bands(ax, n)
    meths = ["dec", "lora", "fullft"]
    for i, (b, c) in enumerate(rows):
        y = n - 1 - i
        step = 0.7 / len(meths)
        for j, m in enumerate(meths):
            yy = y + 0.35 - step / 2 - j * step
            xj, xc = means_b[(b, c, "joint", m)], means_b[(b, c, "pertask_cm", m)]
            ax.plot([xc, xj], [yy, yy], color=METHOD_COL[m], lw=0.7, zorder=2)
            mark(ax, xj, yy, METHOD_COL[m], "o", ms=3.8, zorder=4)
            mark(ax, xc, yy, METHOD_COL[m], "o", ms=3.8, open_=True, zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels([two_line(b, c) for b, c in rows][::-1])
    ax.set_ylim(-0.5, n - 0.5)
    hide_left(ax)
    ax.set_xlabel("held-out Dice\n(dataset mean)")
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    panel_label(ax, "(b) compute-matched")
    h = [Line2D([], [], marker="o", color=METHOD_COL[m], ls="none", ms=3.8) for m in meths] + \
        [Line2D([], [], marker="o", color=INK2, ls="none", ms=3.8), Line2D([], [], marker="o", color=INK2, mfc="white", ls="none", ms=3.8)]
    l = [METHOD_TEX[m] for m in meths] + ["joint, 24 000 steps", "per-structure, 1600 steps"]
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 1.1), ncol=1, fontsize=FS, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    # (c) from-scratch control (TAP-CT architecture, random init) vs Full FT and Dec, joint models
    ax = axes[2]
    msj = ms[(ms["split"] == "test") & (ms["seed"] == 42) & (ms["mode"] == "joint") & (ms["backbone"] == "tapct") & ms["method"].isin(["dec", "fullft"])]
    rvs = rv[(rv["task"] == "seg") & (rv["method"] == "scratch")].rename(columns={"value": "dice"})
    means_c = cohort_mean(pd.concat([msj[cols], rvs[cols]]), "dice")
    check_against_cohort_means(means_c, cm, "seg", "fig_revision(c)")
    n = len(SEG_DATASETS)
    row_bands(ax, n)
    spec_c = [("dec", METHOD_COL["dec"], "o", False), ("fullft", METHOD_COL["fullft"], "o", False), ("scratch", INK, "x", False)]
    for i, c in enumerate(SEG_DATASETS):
        y = n - 1 - i
        for m, col, mk_, open_ in spec_c:
            if ("tapct", c, "joint", m) in means_c.index:
                mark(ax, means_c[("tapct", c, "joint", m)], y, col, mk_, ms=3.8 if mk_ != "x" else 4.5, open_=open_, zorder=4, mew=0.9)
    ax.set_yticks(range(n)); ax.set_yticklabels([COHORT_TEX[c] for c in SEG_DATASETS][::-1])
    ax.set_ylim(-0.5, n - 0.5)
    hide_left(ax)
    ax.set_xlabel("held-out Dice, joint model\n(dataset mean)")
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    panel_label(ax, "(c) from scratch, TAP-CT")
    h = [Line2D([], [], marker=mk_, color=col, ls="none", ms=3.8 if mk_ != "x" else 4.5, mew=0.9) for _, col, mk_, _ in spec_c]
    ax.legend(h, ["Dec", "Full FT", "from scratch (random init)"], loc="lower left", bbox_to_anchor=(0.0, 1.1), ncol=1, fontsize=FS,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.72, bottom=0.18)
    style.save(fig, OUT, "fig_revision")


# ============================================================================= Fig 10: VoCo check
def fig_voco():
    ll = load("label_level")
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.9))
    attn_head = ["lora", "lorag", "babmil", "dora", "rslora", "attn_then_lorag", "attnft", "fullft"]
    panels = [("cls", "lin", [m for m in style.CLS_METHODS if m != "lin"], "(a) classification vs Lin", "Δ held-out AUROC vs Lin", (-0.2, 0.12)),
              ("cls", "attn", attn_head, "(b) classification vs Attn", "Δ held-out AUROC vs Attn", (-0.2, 0.12)),
              ("seg", "dec", [m for m in style.SEG_METHODS if m != "dec"], "(c) segmentation vs Dec", "Δ held-out Dice vs Dec", (-0.12, 0.12))]
    for ax, (task, ref, methods, lab, xl, xlim) in zip(axes, panels):
        n = len(methods)
        row_bands(ax, n)
        for i, m in enumerate(methods):
            y = n - 1 - i
            key = f"{task}_{m}_{ref}"
            sub = ll[ll["key"] == key].set_index("backbone")
            r = sub.loc["all"]
            open_ = any(dnt(task, b, m) for b in MAIN)
            interval(ax, y - 0.17, r["mean"], r["lo3"], r["hi3"], POOLED_COL, POOLED_MARK, xlim, ms=4.2, open_=open_, lw=0.8)
            r = sub.loc["voco"]
            interval(ax, y + 0.17, r["mean"], r["lo3"], r["hi3"], BACKBONE_COL["voco"], BACKBONE_MARK["voco"], xlim, ms=4.2, lw=0.8)
        zero_line(ax)
        ax.set_xlim(*xlim); ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticks(range(n)); ax.set_yticklabels([METHOD_TEX[m] for m in methods][::-1])
        hide_left(ax)
        ax.set_xlabel(xl)
        panel_label(ax, lab)
    h = [Line2D([], [], marker=BACKBONE_MARK["voco"], color=BACKBONE_COL["voco"], ls="none", ms=4.2),
         Line2D([], [], marker=POOLED_MARK, color=POOLED_COL, ls="none", ms=4.2),
         Line2D([], [], marker=POOLED_MARK, color=POOLED_COL, mfc="white", ls="none", ms=4.2)]
    l = ["VoCo", "pooled main backbones (TAP-CT, SPECTRE, Pillar-0)", "pooled estimate includes SPECTRE Full FT (did not train)"]
    top_legend(fig, h, l, 3, columnspacing=1.2)
    footnote(fig, INTERVAL_NOTE, y=-0.03)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.86, bottom=0.17)
    style.save(fig, OUT, "fig_voco")


if __name__ == "__main__":
    style.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_key, fig_methods_cls_rank, fig_methods_seg_rank, fig_perlabel_cls, fig_perstructure_seg, fig_joint, fig_seeds,
              fig_convergence, fig_revision, fig_voco):
        f()
        print("wrote", f.__name__)
