#!/usr/bin/env python3
"""Segmentation example figure and the example-prediction macros (ported from the draft's make_example_figures.py).

    python example_figures.py   -> out/fig_examples_seg.{pdf,png}, out/fig_examples_pred.{pdf,png},
                                   ../paper/tables/example_numbers.tex, out/selection/example_checks.txt

fig_examples_seg   (a) LiTS liver tumor, per-structure Lin / Dec / LoRA / Full FT, and (b) AMOS22, one joint model for
                   15 structures (24 000 steps), per backbone (TAP-CT, SPECTRE, VoCo); cases chosen by
                   select_seg_examples.py (out/selection/seg_example_selection.json). Slices are rendered from the saved
                   predictions under midl_paper/results/examples/seg/<dataset>/<case>/<backbone>-<method>.npz (read-only).
fig_examples_pred  held-out cases with Lin vs LoRA+ABMIL probabilities and the attended depth window (saved predictions
                   midl_paper/results/examples/<cohort>_{lin,babmil}.npz); it also writes the \\numEx* macros.
Check: the Dice printed in each panel title (from the saved prediction) is compared with the per-case Dice of the same
run in the per-case file the metrics are computed from (runs.csv); the probabilities of the prediction dumps are
compared with the per-case file of the run with the same configuration.
Intensity window: soft tissue [-160, 240] HU (ICH: brain [0, 80] HU).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import style
from select_cases import OUT_SEL, RES, V2, cls_percase, run_row, seg_percase

# NO_VOCO (default; WITH_VOCO=1 turns it off): no VoCo row in fig_examples_seg; fig_examples_pred and example_numbers.tex are not written. Outputs go to
# style.FIG_OUT and style.PAPER_DIR (defaults: out/ and ../paper/); the selection is read from OUT_SEL (out/selection).
OUT = style.FIG_OUT
PAPER_FIG = style.PAPER_DIR / "figures"
PAPER_TAB = style.PAPER_DIR / "tables"
CHECK_DIR = OUT / "selection"
INK, INK2 = style.INK, style.INK2
FS = 6.0
CHECKS: list[str] = []

SEG_BACKBONES = [("tapct", "TAP-CT"), ("spectre", "SPECTRE"), ("voco", "VoCo")]
if style.NO_VOCO:
    SEG_BACKBONES = SEG_BACKBONES[:2]
# Display window per backbone input: TAP-CT receives HU (normalised inside the model); SPECTRE receives HU
# [-1000, 1000] mapped to [0, 1]; VoCo receives [-175, 250] HU mapped to [0, 1]. All shown at [-160, 240] HU.
SEG_WINDOW = {"tapct": (-160.0, 240.0), "spectre": ((-160.0 + 1000.0) / 2000.0, (240.0 + 1000.0) / 2000.0),
              "voco": ((-160.0 + 175.0) / 425.0, (240.0 + 175.0) / 425.0)}
SEG_FLIP = {"tapct": (False, False), "spectre": (True, True), "voco": (True, True)}  # radiological display
SEG_BLOCKS = [
    # (selection key, saved-prediction directory, dataset, mode, structure, methods, title)
    ("lits", "lits", "lits", "pertask", "tumor", [("lin", "Lin"), ("dec", "Dec"), ("lora", "LoRA"), ("fullft", "Full FT")],
     "(a) LiTS liver tumor, one model per structure"),
    ("amos", "amos22_long", "amos22", "joint", None, [("lin", "Lin"), ("dec", "Dec"), ("lora", "LoRA"), ("fullft", "Full FT")],
     "(b) AMOS22, one joint model for 15 structures"),
]
REF_EDGE = "#ffd400"
PRED_FILL = np.array([0.835, 0.369, 0.0, 0.55])  # Okabe-Ito vermillion, semi-transparent


def _seg_colors(n: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    cols = np.zeros((n + 1, 4)); cols[1:] = [cmap(k % 20) for k in range(n)]
    return cols


def _display(a: np.ndarray, bb: str) -> np.ndarray:
    fr, fc = SEG_FLIP[bb]
    if fr: a = a[::-1]
    if fc: a = a[:, ::-1]
    return a


def check_seg_title(fig: str, ds: str, mode: str, struct: str | None, bb: str, mt: str, case: str, npz) -> tuple[float, float]:
    """Dice of the saved prediction (a re-inference of the selected checkpoint) vs the per-case file of the run the
    metrics use; returns (per-case-file Dice, which is printed, |difference|)."""
    shown = float(np.nanmean(npz["dice"]))
    row = run_row("seg", ds, bb, mode, mt, label=struct if (struct and mode == "pertask") else "joint")
    d = seg_percase(row)
    if struct:
        d = d[d.structure == struct]
    d = d[d.case_id == case]
    ref = float(d.dice.mean()) if len(d) else float("nan")
    it = json.loads(str(npz["meta"])).get("iteration")
    flag = "" if f"{shown:.2f}" == f"{ref:.2f}" else f"  (2-decimal value differs: saved prediction {shown:.2f}, printed {ref:.2f})"
    CHECKS.append(f"{fig} {ds}/{case} {bb}/{mt}: saved prediction {shown:.4f}, per-case file {ref:.4f} (printed), |diff| {abs(shown - ref):.2e}; "
                  f"checkpoint iteration {it} vs selected {int(row.best_iter)}{flag}")
    return ref, abs(shown - ref)


def fig_examples_seg(out: Path) -> None:
    sel = json.loads((OUT_SEL / "seg_example_selection.json").read_text())
    old = json.loads((RES / "examples" / "seg" / "selection.json").read_text())
    seg_root = RES / "examples" / "seg"
    fig = plt.figure(figsize=(style.TEXTWIDTH_IN, 5.75 if len(SEG_BACKBONES) == 3 else 4.0))
    subs = fig.subfigures(2, 1, height_ratios=[1.0, 1.0], hspace=0.03)
    diffs = []
    for sub, (key, sdir, ds, mode, struct, methods, title) in zip(subs, SEG_BLOCKS):
        case = sel[key]["case"]
        if not (seg_root / sdir / case).is_dir():
            CHECKS.append(f"fig_examples_seg {key}: selected case {case} has no saved predictions; drawing the draft's case {old[key]['case']}")
            case = old[key]["case"]
        axes = sub.subplots(len(SEG_BACKBONES), len(methods) + 1, gridspec_kw={"wspace": 0.04, "hspace": 0.24, "top": 0.89, "bottom": 0.01,
                                                                                "left": 0.0, "right": 1.0})
        cid = case.replace("case_", "").replace("amos_", "").replace("liver_", "")
        sub.suptitle(f"{title}; held-out case {cid}", fontsize=7.0, color=INK, x=0.0, ha="left", y=0.995)
        for r, (bb, bbl) in enumerate(SEG_BACKBONES):
            first = np.load(seg_root / sdir / case / f"{bb}-{methods[0][0]}.npz")
            img, ref = first["image"].astype(np.float32), first["reference"]
            labels = json.loads(str(first["meta"]))["labels"]
            single = len(labels) == 1
            tgt = np.isin(ref, labels) if single else ref > 0
            if single:  # slice with the largest reference area of the structure
                z = int(np.argmax(tgt.reshape(tgt.shape[0], -1).sum(1)))
            else:  # slice with the most reference structures (ties: most foreground)
                z = max(range(ref.shape[0]), key=lambda k: (len(np.unique(ref[k])) - 1, int((ref[k] > 0).sum())))
            ys, xs = np.nonzero(tgt[z])
            if single:  # square crop centred on the structure, at least 72 voxels wide
                half = max(36, int(0.8 * max(ys.max() - ys.min(), xs.max() - xs.min())))
                cy, cx = (ys.max() + ys.min()) // 2, (xs.max() + xs.min()) // 2
                y0, y1 = max(0, cy - half), min(ref.shape[1], cy + half)
                x0, x1 = max(0, cx - half), min(ref.shape[2], cx + half)
            else:
                py, px = (ys.max() - ys.min()) // 3, (xs.max() - xs.min()) // 3
                y0, y1 = max(0, ys.min() - py), min(ref.shape[1], ys.max() + py + 1)
                x0, x1 = max(0, xs.min() - px), min(ref.shape[2], xs.max() + px + 1)
            lo, hi = SEG_WINDOW[bb]
            sl = _display(np.clip((img[z, y0:y1, x0:x1] - lo) / (hi - lo), 0, 1), bb)
            cols = _seg_colors(15)

            def panel(ax, lab, ttl, is_ref=False):
                ax.imshow(sl, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                if single:
                    t = _display(np.isin(ref[z, y0:y1, x0:x1], labels), bb)
                    if not is_ref:
                        pr = _display(lab[z, y0:y1, x0:x1] > 0, bb)
                        ax.imshow(np.where(pr[..., None], PRED_FILL, 0.0))
                    ax.contour(t, levels=[0.5], colors=[REF_EDGE], linewidths=0.8)
                else:
                    lb = _display(lab[z, y0:y1, x0:x1], bb)
                    ax.imshow(cols[lb], alpha=np.where(lb > 0, 0.55, 0.0), interpolation="nearest")
                ax.set_title(ttl, fontsize=FS, color=INK, pad=1.5)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

            panel(axes[r, 0], ref, f"{bbl}: reference", is_ref=True)
            for c, (mt, mtl) in enumerate(methods, start=1):
                d = np.load(seg_root / sdir / case / f"{bb}-{mt}.npz")
                val, dif = check_seg_title("fig_examples_seg", ds, mode, struct, bb, mt, case, d)
                diffs.append(dif)
                panel(axes[r, c], d["prediction"], f"{mtl}: Dice {val:.2f}")
    style.save(fig, out, "fig_examples_seg")
    CHECKS.append(f"fig_examples_seg: titles print the per-case-file Dice; max |saved prediction - per-case file| = {max(diffs):.2e} over {len(diffs)} panels")


def _window(sl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def fig_examples_pred(examples_dir: Path, out: Path) -> None:
    """Held-out cases with predicted probabilities (Lin vs LoRA+ABMIL) and the attended depth window; writes the
    \\numEx* macros (cases A-D in figure order)."""
    cohorts = [("amos", "AMOS: fatty liver"), ("fdg", "FDG-PET/CT: lung cancer"), ("rsna2023", "RSNA 2023: any injury"), ("ich", "RSNA 2019: hemorrhage")]
    rows = []
    for key, title in cohorts:
        fl, fb = examples_dir / f"{key}_lin.npz", examples_dir / f"{key}_babmil.npz"
        zl, zb = np.load(fl, allow_pickle=True), np.load(fb, allow_pickle=True)
        jl, jb = json.loads(str(zl["records"])), json.loads(str(zb["records"]))
        rl, rb = jl["cases"], jb["cases"]
        file_p = {}
        # check against the per-case files of the runs with the same configuration (cohort/backbone/mode/method/label)
        for j, recs in ((jl, rl), (jb, rb)):
            parts = j["yaml"].split("fm_adapt/")[1].split("/")
            cohort, bb, mode, mt, lab = parts[0], parts[1], parts[2], parts[3], parts[4].replace(".yaml", "")
            pc = cls_percase(run_row("cls", cohort, bb, mode, mt, label=lab)).drop_duplicates("case_id", keep="last").set_index("case_id")
            dd = [abs(r["probability"] - float(pc.probability.get(r["case_id"], np.nan))) for r in recs[:4]]
            CHECKS.append(f"fig_examples_pred {key} {bb}/{mode}/{mt}/{lab}: |saved prob - per-case file prob| over the 4 macro cases: "
                          + ", ".join(f"{x:.3f}" for x in dd) + " (the per-case file values are printed and written)")
            file_p[mt] = {r["case_id"]: float(pc.probability[r["case_id"]]) for r in recs}
        # printed probabilities: the per-case files the metrics use (the saved predictions are a re-inference)
        lin_p = file_p["lin"]
        rb = [dict(r, probability=file_p["babmil"][r["case_id"]]) for r in rb]
        rows.append((title, key, rb, zb["slices"], zb["attention"], lin_p))
    n_cases = 3
    rows_d = [(t, k, recs[:n_cases], sl[:n_cases], at[:n_cases], lp) for (t, k, recs, sl, at, lp) in rows]
    fig, axes = plt.subplots(len(rows_d), n_cases, figsize=(style.TEXTWIDTH_IN * 0.75, 1.4 * len(rows_d)))
    for r, (title, key, recs, slices, attns, lin_p) in enumerate(rows_d):
        for c in range(n_cases):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            rec = recs[c]; sl = np.asarray(slices[c], dtype=np.float32)
            lo, hi = (0.0, 80.0) if key == "ich" else (-160.0, 240.0)
            if sl.max() <= 1.5:
                im = np.clip(sl, 0, 1); body = sl > 0.12
            else:
                im = _window(sl, lo, hi); body = sl > -500.0
            ys, xs_ = np.where(body)
            if ys.size > 100:
                m = 6
                im = im[max(0, ys.min() - m):min(im.shape[0], ys.max() + m), max(0, xs_.min() - m):min(im.shape[1], xs_.max() + m)]
            h, w = im.shape
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="bilinear", extent=(0, w, h, 0))
            a = np.asarray(attns[c], dtype=np.float32)
            if a.size:
                xs = np.linspace(0, w, len(a) + 1)
                base, strip = h * 1.03, h * 0.12
                for k in range(len(a)):
                    ax.add_patch(plt.Rectangle((xs[k], base), xs[k + 1] - xs[k], strip * (a[k] / a.max() if a.max() > 0 else 0), color=style.OI["blue"], lw=0))
                ax.set_ylim(base + strip, 0)
            else:
                ax.set_ylim(h, 0)
            ax.set_xlim(0, w)
            ax.set_title(f"label {rec['label']}", fontsize=FS, color=INK, pad=2)
            ax.set_xlabel(f"Lin {lin_p.get(rec['case_id'], float('nan')):.2f} | LoRA+ABMIL {rec['probability']:.2f}", fontsize=FS, color=INK2, labelpad=1)
        axes[r, 0].set_ylabel(title.replace(": ", ":\n"), fontsize=FS, color=INK)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    style.save(fig, out, "fig_examples_pred")
    letters = "ABCD"
    lines = ["% written by figures/example_figures.py (held-out example predictions, Lin vs LoRA+ABMIL)"]
    for title, key, recs, _s, _a, lin_p in rows:
        for c, rec in enumerate(recs[:4]):
            tag = key.capitalize().replace("2023", "BZBC") + letters[c]
            lines.append(f"\\providecommand{{\\numEx{tag}Label}}{{{rec['label']}}}")
            lines.append(f"\\providecommand{{\\numEx{tag}Lin}}{{{lin_p.get(rec['case_id'], float('nan')):.2f}}}")
            lines.append(f"\\providecommand{{\\numEx{tag}Abmil}}{{{rec['probability']:.2f}}}")
    (PAPER_TAB / "example_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    style.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_examples_seg(OUT)
    if not style.NO_VOCO:
        fig_examples_pred(RES / "examples", OUT)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(OUT / "fig_examples_seg.pdf", PAPER_FIG / "fig_examples_seg.pdf")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    (CHECK_DIR / "example_checks.txt").write_text("\n".join(CHECKS) + "\n")
    print("\n".join(CHECKS))


if __name__ == "__main__":
    main()
