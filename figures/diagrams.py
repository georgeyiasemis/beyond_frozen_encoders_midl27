#!/usr/bin/env python3
"""Architecture / method diagrams in the design of the previous draft, corrected against the code.

    ~/miniconda3/envs/aifofomo/bin/python diagrams.py [--refresh-data] [--only pipeline,arms,arms_seg]

fig_pipeline  (a) classification with TAP-CT: volume -> 12-slice windows -> 4x8x8 patches -> [CLS, 4 registers,
              2352 patch tokens] -> 12 blocks -> bag of window CLS tokens -> linear or attentive head;
              (b) segmentation with TAP-CT: crop -> taps at blocks 3/6/9/12 -> multi-depth decoder with its tensor
              shapes; (c) inside one encoder block: what each method trains; (d) per-task vs joint training.
fig_arms      classification: how SPECTRE, Pillar-0 and VoCo turn a volume into tokens, and the method x component
              matrix with the encoder parameters each method trains.
fig_arms_seg  segmentation: decoder tensor shapes per backbone, and the method x component matrix.

Outputs: out/<name>.pdf + .png; ../paper/tables/param_numbers.tex (parameter macros).
Every drawn fact and its source is listed in DIAGRAMS.md. Numbers come from
  data/trainable_params_cls.csv  (copy of the logged classification counts, one per-task run per backbone x method)
  data/trainable_params_seg.csv  (logged segmentation counts, harvested from the run logs by --refresh-data, plus the
                                  SPECTRE feature-combiner share that receives no gradient in segmentation)
and from the code (aifofomo/), cited next to each constant.
"""
from __future__ import annotations

import os
import argparse
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402
from style import BACKBONE_TEX, TEXTWIDTH_IN  # noqa: E402

ROOT = HERE.parents[1]                                     # .../cvpr_ct_challenge
OLD_RESULTS = Path(os.environ.get("BFE_RESULTS") or "/projects/cvpr2026_challenge_fomo4ct/beyond_frozen_encoders")   # read only; BFE_RESULTS overrides
DATA = HERE / "data"
OUT = style.FIG_OUT                                       # figures/out
PAPER_FIG = style.PAPER_DIR / "figures"
PAPER_TAB = style.PAPER_DIR / "tables"
NO_VOCO = style.NO_VOCO                                    # fig_arms / fig_arms_seg without VoCo unless WITH_VOCO=1

# ------------------------------------------------------------------ palette. No greys: frozen weights are cold blue,
# weights trained in full warm orange, low-rank adapters green; tokens are shades of the frozen blue; all text is black
# (secondary labels in italic). BLUE / ORANGE keep their names as the trained / adapter edges.
NAVY = "#2f5f8a"
BLUE, ORANGE = "#c2571a", "#3b8a2e"
INK, INK2, MUTED = "#000000", "#000000", "#000000"
RAIL_C = NAVY                                             # residual rail of a block
FROZEN_FC, FROZEN_EC = "#dceaf7", NAVY                    # frozen pretrained weights
TRAIN_FC, TRAIN_EC = "#fbd9b8", BLUE                      # trained in full
ADP_FC = "#d8eecf"                                        # low-rank adapter (edge ORANGE)
DATA_EC = NAVY                                            # frames of the CT thumbnails
CARD_FC, CARD_EC = "#ffffff", "#7fa7cf"
BAND_C, EMPTY_EC = "#eef5fb", "#c6dcf0"                   # table row bands, edges of empty table cells
LABEL_C = ["#000000", "#7b3fa0", "#d4a017"]               # three labels in panel (d)
CLS_C = NAVY           # class token
REG_C = "#9dbfe0"      # register tokens
PATCH_C = "#c6dcf0"    # patch tokens
PATCH2_C = "#7fa7cf"   # pooled / combined tokens
GEO_C = "#ffffff"      # data geometry (crop grids), drawn on the CT thumbnails
WIN_C = "#8C510A"      # slices of one window / crop in a slice stack (brown: not a method colour)
HEAD_C = TRAIN_FC

# ------------------------------------------------------------------ fonts. Figures are drawn 7.0 in wide and placed at
# the 6.0 in text width (scale 6/7); FS_MIN = 8.2 pt drawn is 7.0 pt printed (include them at \textwidth).
DRAW_W = 7.0
FS = 5.4
FSCALE = 1.28
FS_MIN = 8.2
PRINT_MIN = 7.0
assert abs(TEXTWIDTH_IN - 6.0) < 1e-6 and FS_MIN * TEXTWIDTH_IN / DRAW_W >= PRINT_MIN - 0.01
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, **style.FONT_RC})

LATE = r"LoRA$_\mathrm{late}$"


# ================================================================== data
def _harvest_seg_logs() -> pd.DataFrame:
    """Logged `SegProbeModel [...]: T / N trainable ...; backbone b / B` line of the AMOS22 joint run (15 structures)
    per backbone x method."""
    pat = re.compile(r"SegProbeModel \[mode=(\S+), head=(\S+), classes=(\d+), roi=\(([^)]*)\)\]: (\d+) / (\d+) trainable "
                     r"\([^)]*\); backbone (\d+) / (\d+)")
    rows = []
    for bb in ("tapct", "spectre", "voco"):
        for m in ("lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"):
            hit = None
            for site_dir in [OLD_RESULTS / "runs" / "segmentation"]:
                for run in sorted(site_dir.glob(f"amos22/{bb}/joint/{m}")):
                    for log in sorted(run.rglob("log_0_0.txt")):
                        for line in log.read_text(errors="ignore").splitlines():
                            g = pat.search(line)
                            if g:
                                hit = g
                                break
                        if hit:
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit is None:
                raise RuntimeError(f"no logged parameter line for segmentation {bb}/{m}")
            mode, head, C, roi, tr, tot, btr, btot = hit.groups()
            rows.append(dict(backbone=bb, method=m, mode=mode, head=head, classes=int(C), trainable=int(tr), total=int(tot),
                             backbone_trainable=int(btr), backbone_total=int(btot)))
    return pd.DataFrame(rows)


def _spectre_combiner_numel() -> int:
    """Parameters of SPECTRE's feature combiner, from the safetensors header (rope `periods` are buffers)."""
    snap = sorted((Path.home() / ".cache/huggingface/hub/models--cclaess--SPECTRE-Large/snapshots").glob("*/model.safetensors"))[0]
    with open(snap, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return sum(math.prod(v["shape"]) for k, v in hdr.items()
               if k.startswith("model.feature_combiner.") and not k.endswith("rope.periods"))


PILLAR0_DIR = Path("/projects/cvpr2026_challenge_fomo4ct/pretrained/Pillar0-HeadCT")


def _pillar0_no_grad() -> dict[str, int]:
    """Pillar-0 backbone parameters that the classification methods wrap or unfreeze but that receive no gradient.

    MultiModalAtlas.forward (remote multimodal_atlas.py l.219-305; same loop in pillar0_cls.py `_extract_spatial`) runs
    stage k on the scales that have not yet left the encoder: stage 1 on 32^3, 8^3, 2^3; stage 2 on 8^3, 2^3; stage 3 on
    2^3 only; MultiScaleAttentionBlock uses `blocks[S]` / `posemb[S]` for S < number of scales present
    (multimodal_msa.py l.558-597). Every block nevertheless holds 3 per-scale CrossWindowBlocks, so per-scale block 2 of
    stage 2 and blocks 1, 2 of stage 3 never run; nor does the text projection (`model.text.*`, `logit_scale`).
    Confirmed numerically: a forward/backward of the randomly initialised model gives gradients to 44,843,456 of
    94,144,193 parameters (check_pillar0_grads.py). Counts below come from the safetensors header shapes."""
    with open(PILLAR0_DIR / "model.safetensors", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    r = 16

    def unused(k: str) -> bool:
        if k.startswith("model.text.") or k == "model.logit_scale":
            return True
        g = re.search(r"atlas_models\.(\d+)\.blocks\.\d+\.(blocks|posemb)\.(\d+)\.", k)
        return bool(g) and int(g.group(3)) >= 3 - int(g.group(1))   # stage k (0-based) holds 3 - k scales

    def late(k: str) -> bool:
        return ".atlas_models.2." in k

    qkv = [(k, v["shape"]) for k, v in hdr.items() if re.search(r"\.attn\.(q|kv)\.weight$", k) and unused(k)]
    lora_all = sum(r * (o + i) for _, (o, i) in qkv)
    lora_late = sum(r * (o + i) for k, (o, i) in qkv if late(k))
    dora_all = lora_all + sum(o for _, (o, i) in qkv)
    attn_all = sum(math.prod(v["shape"]) for k, v in hdr.items() if ".attn." in k and unused(k))
    # buffers in the header (BatchNorm statistics) are not parameters: full = logged backbone total - used parameters
    full = sum(math.prod(v["shape"]) for k, v in hdr.items() if unused(k))
    return {"lora": lora_all, "babmil": lora_all, "rslora": lora_all, "dora": dora_all, "lorag": lora_late,
            "lin_then_lorag": lora_late, "attn_then_lorag": lora_late, "attnft": attn_all, "fullft": full}


def refresh_data() -> None:
    """Copy / harvest every number the diagrams draw into data/ (run once; reads the previous draft's results read-only)."""
    DATA.mkdir(exist_ok=True)
    shutil.copyfile(OLD_RESULTS / "metrics" / "trainable_params.csv", DATA / "trainable_params_cls.csv")
    seg = _harvest_seg_logs()
    # SPECTRE segmentation: `module` is the whole SpectreImageFeatureExtractor (backbones.py l.147-149), so all-block
    # LoRA/DoRA also wrap the combiner's q/kv and Full FT unfreezes it, but features() calls only the ViT
    # (forward_intermediates), so the combiner never receives a gradient. Its share of the logged count:
    #   LoRA: 4 blocks x r x (1080+1080 [q] + 1080+2160 [kv]); DoRA adds the magnitudes 4 x (1080 + 2160);
    #   Full FT: every combiner parameter (safetensors header). LoRA_late's pattern excludes the combiner.
    r, d = 16, 1080
    comb_lora = 4 * r * ((d + d) + (d + 2 * d))
    comb = {"lora": comb_lora, "dec_then_lora": comb_lora, "dora": comb_lora + 4 * (d + 2 * d),
            "fullft": _spectre_combiner_numel()}
    seg["no_grad_backbone"] = [comb.get(m, 0) if bb == "spectre" else 0 for bb, m in zip(seg.backbone, seg.method)]
    seg.to_csv(DATA / "trainable_params_seg.csv", index=False)
    pd.DataFrame([dict(backbone="pillar0", method=m, no_grad_backbone=v) for m, v in _pillar0_no_grad().items()]
                 ).to_csv(DATA / "no_grad_cls.csv", index=False)
    # example slices (AMOS22 case drawn on the TAP-CT grid, and a head CT slice for Pillar-0)
    ex = OLD_RESULTS / "examples" / "seg" / "amos22_long" / "amos_0284" / "tapct-lora.npz"
    dd = np.load(ex)
    img, ref, pred = dd["image"].astype(np.float32), dd["reference"], dd["prediction"]
    z = max(range(ref.shape[0]), key=lambda k: (len(np.unique(ref[k])) - 1, int((ref[k] > 0).sum())))
    sl0 = np.clip((img[z] + 160.0) / 400.0, 0, 1)
    ys, xs = np.nonzero(sl0 > 0.05)
    c = ((ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2)
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 2
    sel = (slice(max(0, c[0] - half), c[0] + half), slice(max(0, c[1] - half), c[1] + half))
    n, step = 6, 7
    zs = [min(img.shape[0] - 1, z + step * (n - 1 - k)) for k in range(n)]
    vol = np.stack([np.clip((img[k] + 160.0) / 400.0, 0, 1)[sel] for k in zs]).astype(np.float32)
    hct = np.load(OLD_RESULTS / "examples" / "ich_lin.npz", allow_pickle=True)
    hsl = np.asarray(hct["slices"][0], dtype=np.float32)
    hsl = np.clip(hsl, 0, 1) if hsl.max() <= 1.5 else np.clip(hsl / 80.0, 0, 1)
    np.savez_compressed(DATA / "diagram_examples.npz", vol=vol, pred=np.stack([pred[k][sel] for k in zs]).astype(np.uint8),
                        head_ct=hsl)
    print("refreshed", DATA)


def load_data():
    if not (DATA / "diagram_examples.npz").exists():
        refresh_data()
    ex = np.load(DATA / "diagram_examples.npz")
    cls = pd.read_csv(DATA / "trainable_params_cls.csv").set_index(["backbone", "method"])
    ng = pd.read_csv(DATA / "no_grad_cls.csv").set_index(["backbone", "method"])["no_grad_backbone"]
    cls["no_grad_backbone"] = [int(ng.get(i, 0)) for i in cls.index]
    seg = pd.read_csv(DATA / "trainable_params_seg.csv").set_index(["backbone", "method"])
    return list(ex["vol"]), list(ex["pred"]), ex["head_ct"], cls, seg


# ------------------------------------------------------------------ parameter formatting
def fmt_m(n: float, sep: str = " ") -> str:
    """Two significant figures in millions (0.59 M, 2.4 M, 28 M); 0 -> '0'."""
    if n == 0:
        return "0"
    x = n / 1e6
    s = f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:.0f}")
    return f"{s}{sep}M"


# Classification: the two-stage methods use the late-block pattern of LoRA_late (generate.py `_model`, l.491-640), so
# their encoder count equals LoRA_late's; VoCo has no logged line for them (no VoCo two-stage log in the copied table).
CLS_SAME_AS = {"lin_then_lorag": "lorag", "attn_then_lorag": "lorag"}


def cls_bb(cls: pd.DataFrame, bb: str, m: str, grad_only: bool = True) -> tuple[int, bool]:
    """(backbone parameters trained [that receive a gradient], derived?) for classification."""
    key, derived = ((bb, m), False) if (bb, m) in cls.index else ((bb, CLS_SAME_AS[m]), True)
    n = int(cls.loc[key, "backbone_trainable"])
    return (n - int(cls.loc[key, "no_grad_backbone"]) if grad_only else n), derived


# ================================================================== drawing primitives
def box(ax, x, y, w, h, fc, ec, lw=0.7, r=0.4, z=3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))   # r: unused


def card(ax, x, y, w, h, fc=CARD_FC, ec=CARD_EC, lw=0.7, r=0.6, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=lw, zorder=z))   # r: unused


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.9, z=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=7), zorder=z)


def txt(ax, x, y, s, fs=FS, c=INK2, ha="center", va="center", bold=False, z=6, **kw):
    ax.text(x, y, s, fontsize=max(fs * FSCALE, FS_MIN), color=c, ha=ha, va=va, fontweight="bold" if bold else "normal",
            zorder=z, linespacing=1.15, **kw)


def head(ax, x, y, title):
    ax.text(x, y, title, fontsize=7 * FSCALE, color=INK, ha="left", va="center", fontweight="bold")


def token(ax, x, y, s, kind, z=4):
    fc = {"cls": CLS_C, "reg": REG_C, "patch": PATCH_C, "patch2": PATCH2_C}[kind]
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec="white", lw=0.4, zorder=z))


def _shade(c, f):
    return tuple(np.clip(np.array(matplotlib.colors.to_rgb(c)) * f, 0, 1))


def _mix(c, e, f):
    """c moved a fraction f towards e (cuboid faces are shaded towards their own edge colour, not towards black)."""
    a, b = np.array(matplotlib.colors.to_rgb(c)), np.array(matplotlib.colors.to_rgb(e))
    return tuple(a + (b - a) * f)


def cuboid(ax, x, y, w, h, d, fc, ec, lw=0.6, z=3):
    """Oblique cuboid: front face w x h at (x, y), depth d drawn up and to the right."""
    dx, dy = d * 0.6, d * 0.45
    front = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z)
    top = Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, fc=_mix(fc, ec, 0.15), ec=ec, lw=lw, zorder=z)
    side = Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, fc=_mix(fc, ec, 0.35), ec=ec, lw=lw, zorder=z)
    for p in (side, top, front):
        ax.add_patch(p)
    return dx, dy


def data_units(fig, ax, s_in):
    p0 = ax.transData.inverted().transform((0, 0))
    p1 = ax.transData.inverted().transform((s_in * fig.dpi, s_in * fig.dpi))
    return p1[0] - p0[0], p1[1] - p0[1]


def _label_colors() -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    cols = np.zeros((16, 4)); cols[1:] = [cmap(k % 20) for k in range(15)]
    return cols


def thumb(fig, ax, x, y, s_in, img, overlay=None, alpha=0.55, grid=None):
    fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    W, H = fig.get_size_inches()
    iax = fig.add_axes([fx, fy, s_in / W, s_in / H])
    iax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    if overlay is not None:
        iax.imshow(_label_colors()[overlay], alpha=np.where(overlay > 0, alpha, 0.0), interpolation="nearest")
    if grid:
        ny, nx = grid
        for k in range(1, nx):
            iax.axvline(k * img.shape[1] / nx - 0.5, color="white", lw=0.7, alpha=0.9)
        for k in range(1, ny):
            iax.axhline(k * img.shape[0] / ny - 0.5, color="white", lw=0.7, alpha=0.9)
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_edgecolor(DATA_EC); sp.set_linewidth(0.6)
    return iax


def stack(fig, ax, x, y, s_in, slices, overlays=None, alpha=0.5, mark=None, mark_c=WIN_C, step_in=0.035):
    """Real axial slices drawn back to front; mark=(i0, i1) outlines a run of slices (a window)."""
    n = len(slices)
    for k in range(n):
        dxu, dyu = data_units(fig, ax, step_in * (n - 1 - k))
        img = slices[k] * (0.6 + 0.4 * (k + 1) / n)
        iax = thumb(fig, ax, x + dxu, y + dyu, s_in, img, overlay=overlays[k] if overlays is not None else None, alpha=alpha)
        if mark is not None and mark[0] <= k <= mark[1]:
            for sp in iax.spines.values():
                sp.set_edgecolor(mark_c); sp.set_linewidth(1.0)
    return data_units(fig, ax, s_in + step_in * (n - 1))


def patch_slabs(fig, ax, x, y, s_in, fronts, n_slab=3, depth=4, grid=(7, 7), step_in=0.012, gap_in=0.07,
                mark_c=WIN_C):
    """One window cut into n_slab depth slabs of `depth` slices each, drawn back to front; each slab's front slice
    carries the in-plane patch grid, so a patch spans `depth` slices in depth and one grid cell in-plane."""
    n = n_slab * depth
    off = lambda i: (n_slab - 1 - i // depth) * gap_in + (n - 1 - i) * step_in
    for i in range(n):
        dxu, dyu = data_units(fig, ax, off(i))
        img = fronts[i // depth] * (0.6 + 0.4 * (i + 1) / n)
        iax = thumb(fig, ax, x + dxu, y + dyu, s_in, img, grid=grid if i % depth == depth - 1 else None)
        for sp in iax.spines.values():
            sp.set_edgecolor(mark_c); sp.set_linewidth(0.8)
    return data_units(fig, ax, s_in + off(0))


def mini_block(ax, x, ytop, w, qkv="frozen", proj="frozen", mlp="frozen", norm="frozen", lora=False, title=None):
    """One pre-norm transformer block, read bottom to top: LN, W_qkv, attention, W_proj, residual add, LN, MLP,
    residual add. lora=True draws the rank-16 side branch A, B next to W_qkv. The block hangs from ytop; every
    sub-box is CH units (10 pt drawn) tall so that its 6.1 pt label (with subscript) keeps a margin. Returns the
    bottom of the block."""
    col = {"frozen": (FROZEN_FC, FROZEN_EC), "train": (TRAIN_FC, TRAIN_EC)}
    CH, GAP, ADD, ATT = 2.4, 0.4, 1.6, 1.7        # sub-box height (fits W_qkv with its subscript), gap, gap holding a residual add, attention row
    bx, bw = x + 1.25, 7.6                      # same sub-box width in every card; a LoRA card is wider for A, B
    cx, sx = bx + bw / 2, x + 0.85              # residual rail and its adds inside the card
    # rows bottom to top: (name, state, height, gap above)
    rows = [("LN", norm, CH, GAP), ("$W_{qkv}$", qkv, CH, 0.3), ("attention", None, ATT, 0.3), ("$W_{proj}$", proj, CH, ADD),
            ("LN", norm, CH, GAP), ("MLP", mlp, CH, ADD)]
    pad_b = 0.7
    h = pad_b + sum(r[2] + r[3] for r in rows)
    y = ytop - h
    box(ax, x, y, w, h, CARD_FC, CARD_EC, lw=0.6, r=0.6, z=2)
    if title:
        txt(ax, x + w / 2, ytop + 1.1, title, fs=5.8, c=INK, bold=True)
    ax.plot([sx, sx], [y + 0.35, ytop - 0.35], color=RAIL_C, lw=0.5, zorder=2)
    yy = y + pad_b
    centres = {}
    for k, (name, st, hh, gap) in enumerate(rows):
        yc = yy + hh / 2
        if st is None:
            txt(ax, cx, yc, name, fs=4.4, c=MUTED, style="italic")
        else:
            fc, ec = col[st]
            box(ax, bx, yy, bw, hh, fc, ec, lw=0.9 if st == "train" else 0.5, r=0.25, z=3)
            txt(ax, cx, yc, name, fs=4.8, c=INK)
        centres[k] = yc
        yy += hh + gap
        if gap == ADD:                               # residual add on the rail, centred in the gap
            ya = yy - gap / 2
            ax.add_patch(plt.Circle((sx, ya), 0.62, fc="white", ec=INK2, lw=0.5, zorder=4))
            txt(ax, sx, ya - 0.04, "+", fs=4.6, c=INK2)
    if lora:
        yq = centres[1]
        aw, ag = 2.3, 0.7
        lx = bx + bw + 0.8
        for k, nm in enumerate("AB"):
            box(ax, lx + k * (aw + ag), yq - CH / 2, aw, CH, ADP_FC, ORANGE, lw=0.8, r=0.2, z=4)
            txt(ax, lx + k * (aw + ag) + aw / 2, yq, nm, fs=4.8, c=INK)
        ax.plot([bx + bw, lx], [yq, yq], color=ORANGE, lw=0.6, zorder=3)
        ax.plot([lx + aw, lx + aw + ag], [yq, yq], color=ORANGE, lw=0.6, zorder=3)
        txt(ax, lx + aw + ag / 2, yq + CH / 2 + 1.0, "r = 16", fs=4.4, c=ORANGE)
    return y


def canvas(H: float, ylo: float = 0.0):
    fig = plt.figure(figsize=(DRAW_W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(ylo, ylo + 100 * H / DRAW_W)
    ax.axis("off"); ax.set_aspect("equal")
    return fig, ax


def save(fig, name: str) -> None:
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    printed = min(sizes) * TEXTWIDTH_IN / fig.get_size_inches()[0]
    assert printed >= PRINT_MIN - 0.01, (name, printed)
    print(f"{name}: {fig.get_size_inches()[0]:.1f} x {fig.get_size_inches()[1]:.2f} in drawn, smallest text {min(sizes):.2f} pt"
          f" -> {printed:.2f} pt at {TEXTWIDTH_IN} in")
    for msg in style.text_box_overflows(fig):
        print(f"TEXT-BOX OVERFLOW {name}: {msg}")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


# ================================================================== Figure: pipeline (TAP-CT)
def fig_pipeline(vol, vpred, seg) -> None:
    LINE = 1.95                                # height of one label line (8.2 pt drawn) in drawing units
    A = 3.7                                    # (b) moves up: tighter gap under the (a) labels
    U = 4 * LINE + A + 0.5                     # (c)/(d) move up: one label line fewer in (a), three in (b), tighter gaps
    L = U + 2 * LINE - 0.8 - 1.4               # legend and bottom: two lines fewer under the (c) blocks; blocks 1.4 taller
    H = 6.17 - L * DRAW_W / 100                # 7.0 pt printed text; was 6.17 with the longer labels
    top = 100 * 5.35 / DRAW_W - 1.0 + 7.9
    fig, ax = canvas(H, top - 100 * H / DRAW_W)
    s_in, ar = 0.5, 2.4

    def go(x, y, ar=ar):
        arrow(ax, x + 0.5, y, x + 0.5 + ar, y)
        return x + 1.0 + ar

    # ---------------- (a) classification, TAP-CT (tapct_cls.py; code map CLS 1.5, 2.1, 2.5)
    y0 = 68.4
    dx, dy = stack(fig, ax, 1.0, y0, s_in, vol)
    head(ax, 0.8, y0 + dy + 2.6, "(a) Classification")
    ym, lab_y = y0 + dy / 2, y0 - 1.3
    txt(ax, 0.8, lab_y, "CT volume,\n224² in-plane", va="top", ha="left")
    AR = 4.0                                   # (a) has no token strip: longer arrows spread it over the width
    x = go(1.0 + dx, ym, AR)
    stack(fig, ax, x, y0, s_in, vol, mark=(2, 4))
    txt(ax, x + dx / 2 + 0.8, lab_y, "W windows of 12\nslices, stride 3", va="top")
    x = go(x + dx, ym, AR)
    ps = patch_slabs(fig, ax, x, y0, s_in, vol[2:5])[0]
    txt(ax, x + ps / 2, lab_y, "2352 patches of\n4×8×8 per window", va="top")
    x = go(x + ps + 1.0, ym, AR)
    ew = 12.0
    box(ax, x, y0 + 0.6, ew, dy - 1.2, FROZEN_FC, FROZEN_EC, lw=0.7, r=0.5)
    for k in range(12):
        ax.plot([x + 0.7 + k * (ew - 1.4) / 11] * 2, [y0 + 1.4, y0 + dy - 2.6], color=FROZEN_EC, lw=0.6, zorder=4)
    txt(ax, x + ew / 2, y0 + dy - 1.4, "ViT-B", fs=4.8, c=INK)
    txt(ax, x + ew / 2, lab_y, "12 blocks\n(panel c)", va="top")
    x = go(x + ew, ym, AR)
    for k in range(5):
        token(ax, x, y0 + 0.7 + k * 1.65, 1.4, "cls")
    txt(ax, x + 0.7, lab_y, "W CLS\ntokens", va="top")
    x += 1.4
    hx = x + 2.6
    hw = 99.0 - hx
    arrow(ax, x + 0.4, ym, hx - 0.2, y0 + dy * 0.83)
    arrow(ax, x + 0.4, ym, hx - 0.2, y0 + dy * 0.33)
    box(ax, hx, y0 + dy * 0.70, hw, dy * 0.26, HEAD_C, BLUE, lw=0.9, r=0.8)
    txt(ax, hx + hw / 2, y0 + dy * 0.83, "Lin: mean → linear", fs=4.8, c=INK)
    box(ax, hx, y0 + dy * 0.02, hw, dy * 0.62, HEAD_C, BLUE, lw=0.9, r=0.8)
    txt(ax, hx + hw / 2, y0 + dy * 0.33, "Attn: two attention\nblocks → linear", fs=4.6, c=INK)
    txt(ax, hx + hw / 2, lab_y, "head: one logit\nper label", va="top")

    # ---------------- (b) segmentation, TAP-CT (backbones.py TapCTDenseBackbone, heads.py MultiTapSegDecoder)
    y1 = 42.6 + LINE + A
    head(ax, 0.8, y1 + dy + 3.6, "(b) Segmentation")
    ym, lab_y = y1 + dy / 2, y1 - 1.3
    stack(fig, ax, 1.0, y1, s_in, vol, mark=(2, 4))
    txt(ax, 0.8, lab_y, "crop of 12\nwhole slices", va="top", ha="left")
    x = go(1.0 + dx, ym)
    ew2 = 30.0                                  # blocks wide enough for the two-digit label 12
    by, bh = y1 + 0.8, dy - 1.6
    bw = (ew2 - 11 * 0.3) / 12
    tap_x = []
    for k in range(12):
        bx = x + k * (bw + 0.3)
        box(ax, bx, by, bw, bh, FROZEN_FC, FROZEN_EC, lw=0.5, r=0.2)
        if k in (2, 5, 8, 11):                      # TAPCT_DEFAULT_TAPS = (2, 5, 8, 11)
            tap_x.append(bx + bw / 2)
            txt(ax, bx + bw / 2, by + bh / 2, str(k + 1), fs=4.6, c=BLUE, bold=True)
    txt(ax, x + ew2 / 2 + 0.8, lab_y, "blocks 3, 6, 9, 12\n(Lin: block 12)", va="top")
    dx0 = x + ew2 + 4.0
    for j, tx_ in enumerate(reversed(tap_x)):     # j = 0: block 12, lowest route, nearest column, bottom conv
        lev, cx_ = by + bh + 0.8 + j * 0.6, x + ew2 + 0.7 + j * 0.8
        yy = by + 0.4 + j * (bh - 0.8) / 4
        ax.plot([tx_, tx_, cx_, cx_, dx0], [by + bh, lev, lev, yy + 0.8, yy + 0.8], color=BLUE, lw=0.6, zorder=2)
        ax.add_patch(plt.Circle((tx_, by + bh), 0.32, color=BLUE, zorder=3))
        box(ax, dx0, yy, 2.6, 1.6, HEAD_C, BLUE, lw=0.6, r=0.2)
    txt(ax, dx0 + 0.3, lab_y + 0.4, "1×1×1\nconv", va="top")
    # TAP-CT stride (4, 8, 8): 3 stages with scales (2,2,2), (2,2,2), (1,2,2); channels 128 -> 64 -> 32 -> 16
    stages = [("128", "3×28²", 128, 28, 3), ("64", "6×56²", 64, 56, 6), ("32", "12×112²", 32, 112, 12), ("16", "12×224²", 16, 224, 12)]
    fx = dx0 + 2.6 + 0.6
    arrow(ax, fx, ym, fx + 1.6, ym); fx += 2.0
    for k, (c_, g_, ch, hw_, dd) in enumerate(stages):
        w_ = 0.8 + 1.4 * np.log2(ch / 8) / 4
        h_ = 2.4 + 6.6 * np.log2(hw_ / 28) / 3
        ddx, _ = cuboid(ax, fx, ym - h_ / 2, w_, h_, d=1.0 + dd / 5, fc=HEAD_C, ec=BLUE, lw=0.6)
        txt(ax, fx + w_ / 2, lab_y, f"{c_}\n{g_}", va="top", fs=4.6)
        nx = fx + w_ + ddx + 0.4
        if k < len(stages) - 1:
            arrow(ax, nx + 0.3, ym, nx + 2.5, ym)
        fx = nx + 4.0
    dec_tap = seg.loc[("tapct", "dec")]
    dec_n = int(dec_tap["trainable"])           # logged, joint AMOS22 model (15 structures)
    txt(ax, 44.0, y1 + dy + 3.6, f"multi-depth decoder ({dec_n / 1e6:.2f} M parameters for {int(dec_tap['classes'])} structures)",
        fs=4.5, ha="left", va="center")
    arrow(ax, fx - 2.4, ym, fx + 1.2, ym)
    fx += 1.6
    stack(fig, ax, fx, y1, s_in, vol, overlays=vpred, alpha=0.55)
    txt(ax, min(fx + dx / 2, 99.0 - 6.6), lab_y, "one sigmoid map\nper structure", va="top")

    # ---------------- (c) what is trained inside one block (lora.py, dora.py, unfreeze_attention_modules)
    y2 = 4.0
    head(ax, 0.8, 29.4 + U, "(c) What each method trains in every block")
    specs = [("frozen", dict(), 9.4, "Lin, Attn,\nDec"),
             ("LoRA", dict(lora=True), 16.1, "all LoRA variants;\n" + LATE + ", Lin→LoRA,\nAttn→LoRA: blocks 9–12"),
             ("Attn-only FT", dict(qkv="train", proj="train"), 9.4, "attention\nprojections\nand biases"),
             ("Full FT", dict(qkv="train", proj="train", mlp="train", norm="train"), 9.4, "all, incl. patch\nembedding")]
    bx = 1.0
    for t, kw, w_, sub in specs:
        yb = mini_block(ax, bx, 25.9 + U, w_, title=t, **kw)
        txt(ax, bx + w_ / 2, yb - 0.6, sub, fs=4.4, va="top")
        bx += w_ + 5.3
    items = [(FROZEN_FC, FROZEN_EC, "frozen"), (TRAIN_FC, TRAIN_EC, "trained"),
             (ADP_FC, ORANGE, "low-rank adapter: $W_0x + \\frac{\\alpha}{r}BAx$, $\\alpha$ = 32")]
    for k, (fc, ec, t) in enumerate(items):
        xx = 1.0 + [0, 10.0, 21.0][k]
        box(ax, xx, -3.65 + L, 1.8, 1.3, fc, ec, lw=0.7, r=0.2)
        txt(ax, xx + 2.4, -3.0 + L, t, fs=4.6, ha="left")

    # ---------------- (d) per-task vs joint (code map CLS 4, SEG 5)
    x3 = 64.5
    head(ax, x3, 29.4 + U, "(d) Two training modes")
    card(ax, x3, 2.5 + L, 34.5, 25.3 - (L - U))
    labs = LABEL_C

    def model(xm, ym_, outs, w=5.4, h=2.4):
        box(ax, xm, ym_, w, h, FROZEN_FC, FROZEN_EC, lw=0.6, r=0.3)
        for q in range(4):
            ax.plot([xm + 0.9 + q * 1.2] * 2, [ym_ + 0.5, ym_ + h - 0.5], color=FROZEN_EC, lw=0.6, zorder=4)
        box(ax, xm + w + 0.5, ym_ + 0.35, 2.2, h - 0.7, HEAD_C, BLUE, lw=0.6, r=0.3)
        for o, c in enumerate(outs):
            oy = ym_ + h / 2 + (o - (len(outs) - 1) / 2) * 1.25
            ax.plot([xm + w + 2.7, xm + w + 3.8], [ym_ + h / 2, oy], color=INK2, lw=0.5, zorder=2)
            ax.add_patch(plt.Circle((xm + w + 4.25, oy), 0.45, color=c, zorder=4))
    txt(ax, x3 + 1.2, 25.6 + U, "per-task: one model per label", fs=4.9, c=INK, ha="left", bold=True)
    for q, c in enumerate(labs):
        model(x3 + 1.5, 21.8 + U - q * 2.7, [c])
    txt(ax, x3 + 13.4, 19.4 + U, "K runs; each sees\nits own label only", fs=4.6, ha="left")
    ax.plot([x3 + 1.0, x3 + 33.5], [15.7 + U, 15.7 + U], color=CARD_EC, lw=0.6)
    txt(ax, x3 + 1.2, 14.2 + U, "joint: one model per cohort", fs=4.9, c=INK, ha="left", bold=True)
    model(x3 + 1.5, 9.1 + U, labs)
    txt(ax, x3 + 13.4, 10.3 + U, "1 run; encoder,\nadapters and head\nshared; K outputs", fs=4.6, ha="left")
    # in the caption: "Segmentation: label = structure, cohort = dataset; 15-structure joint models train for 3 × the steps."
    save(fig, "fig_pipeline")


# ================================================================== method x component matrices
CELL = {"f": (FROZEN_FC, FROZEN_EC), "a": (ADP_FC, ORANGE), "t": (TRAIN_FC, TRAIN_EC), "-": ("white", EMPTY_EC)}


def cell(ax, x, y, w, h, st, mark=None):
    fc, ec = CELL[st]
    box(ax, x, y, w, h, fc, ec, lw=0.8 if st in "at" else 0.45, r=0.2, z=3)
    if mark:
        txt(ax, x + w / 2, y + h / 2, mark, fs=4.3, c=INK)


def matrix(ax, x0, ytop, rows, cols, row_h=2.25, name_w=17.0, groups=()):
    """rows: (name, sub, {col: state or text}). cols: (key, label, width, kind) with kind 'cell' or 'text'."""
    x = x0 + name_w
    pos = {}
    for key, lab, w, kind in cols:
        pos[key] = (x, w, kind)
        txt(ax, x + w / 2, ytop + 1.2, lab, fs=4.5, c=INK, va="bottom")
        x += w + 0.5
    for g_lab, k0, k1, *al in groups:   # optional 4th item "right": label right-aligned to the group (longer than the group)
        xa = pos[k0][0]; xb = pos[k1][0] + pos[k1][1]
        ax.plot([xa, xb], [ytop + 5.0, ytop + 5.0], color=INK2, lw=0.5)
        if al and al[0] == "right":
            txt(ax, xb, ytop + 5.9, g_lab, fs=4.8, c=INK, bold=True, va="bottom", ha="right")
        else:
            txt(ax, (xa + xb) / 2, ytop + 5.9, g_lab, fs=4.8, c=INK, bold=True, va="bottom")
    ax.plot([x0, x], [ytop + 0.6, ytop + 0.6], color=INK2, lw=0.5)
    for i, (name, sub, vals) in enumerate(rows):
        y = ytop - (i + 1) * row_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((x0, y - 0.15), x - x0, row_h, fc=BAND_C, ec="none", zorder=0))
        txt(ax, x0 + 0.4, y + row_h / 2 + 0.05, name, fs=5.0, c=INK, ha="left", bold=True)
        if sub:
            txt(ax, x0 + 9.0, y + row_h / 2, sub, fs=4.1, c=MUTED, ha="left", style="italic")
        for key, (cx, w, kind) in pos.items():
            v = vals.get(key, "-" if kind == "cell" else "")
            if kind == "cell":
                st, mark = (v[0], v[1:] or None) if v else ("-", None)
                cell(ax, cx + 0.15, y + 0.25, w - 0.3, row_h - 0.6, st, mark)
            else:
                txt(ax, cx + w / 2, y + row_h / 2, v, fs=4.5, c=INK2)
    return x


def legend_row(ax, x, y, items, fs=4.6):
    for fc, ec, t in items:
        box(ax, x, y - 0.65, 1.8, 1.3, fc, ec, lw=0.7, r=0.2)
        txt(ax, x + 2.3, y, t, fs=fs, ha="left")
        x += 2.3 + 0.105 * max(fs * FSCALE, FS_MIN) * len(t) + 2.0


ENC_COLS = [("pe", "patch\nemb.", 4.6, "cell"),
            ("eq", "$W_{qkv}$", 4.6, "cell"), ("ep", "$W_{proj}$", 4.6, "cell"), ("em", "MLP,\nnorms", 4.6, "cell"),
            ("lq", "$W_{qkv}$", 4.6, "cell"), ("lp", "$W_{proj}$", 4.6, "cell"), ("lm", "MLP,\nnorms", 4.6, "cell")]
CELLS = {  # which encoder components each method trains (code map CLS 3.1-3.4, SEG 4)
    "lin": {}, "attn": {}, "dec": {},
    "lora": {"eq": "a", "lq": "a"}, "lorag": {"lq": "a"}, "babmil": {"eq": "a", "lq": "a"},
    "dora": {"eq": "am", "lq": "am"}, "rslora": {"eq": "a", "lq": "a"},
    "lin_then_lorag": {"lq": "a2"}, "attn_then_lorag": {"lq": "a2"}, "dec_then_lora": {"eq": "a2", "lq": "a2"},
    "attnft": {"eq": "t", "ep": "t", "lq": "t", "lp": "t"},
    "fullft": {k: "t" for k in ("pe", "eq", "ep", "em", "lq", "lp", "lm")},
}
NAME = {"lin": "Lin", "attn": "Attn", "dec": "Dec", "lora": "LoRA", "lorag": LATE, "babmil": "LoRA+ABMIL", "dora": "DoRA",
        "rslora": "rsLoRA", "lin_then_lorag": "Lin→LoRA", "attn_then_lorag": "Attn→LoRA", "dec_then_lora": "Dec→LoRA",
        "attnft": "Attn-only FT", "fullft": "Full FT"}
SUB = {"dora": "(+ m)", "rslora": r"($\alpha/\sqrt{r}$)"}


def _cells(m):
    d = dict(CELLS[m])
    for k in ("pe", "eq", "ep", "em", "lq", "lp", "lm"):
        d.setdefault(k, "f")
    return d


# ================================================================== Figure: classification backbones + method matrix
def fig_arms(vol, head_ct, cls) -> None:
    sh = 27.5 if NO_VOCO else 0.0       # NO_VOCO: panel (c) VoCo is dropped; (a), (b) move down onto the matrix
    yb = -3.4 if NO_VOCO else 100 * 0.45 / DRAW_W   # NO_VOCO: the notes are in the caption, the canvas ends below the legend
    H = 9.3 - sh * DRAW_W / 100 - (100 * 0.45 / DRAW_W - yb) * DRAW_W / 100
    fig, ax = canvas(H, -yb)
    s_in, ar = 0.5, 2.6

    def go(x, y):
        arrow(ax, x + 0.4, y, x + 0.4 + ar, y)
        return x + 0.8 + ar

    # ---------------- (a) SPECTRE (spectre_cls.py; code map CLS 1.5, 2.2)
    y0 = 112.5 - sh
    dx, dy = data_units(fig, ax, s_in + 0.035 * 5)
    head(ax, 0.8, y0 + dy + 2.4, "(a) SPECTRE: crops, per-crop ViT-L, feature combiner")
    ym, lab_y = y0 + dy / 2, y0 - 1.2
    stack(fig, ax, 1.0, y0, s_in, vol)
    front = vol[-1]
    fax = thumb(fig, ax, 1.0, y0, s_in, front)
    n0, n1 = front.shape
    for k in range(1, 4):
        fax.axvline(k * n1 / 4 - 0.5, color=GEO_C, lw=0.6); fax.axhline(k * n0 / 4 - 0.5, color=GEO_C, lw=0.6)
    fax.add_patch(Rectangle((n1 / 4 - 0.5, n0 / 4 - 0.5), n1 / 2, n0 / 2, fc="none", ec=GEO_C, lw=1.5))
    txt(ax, 1.0, lab_y, "512²×256,\n0.5×0.5×1 mm,\n4³ = 64 crops\n(train: 2³ = 8)", va="top", fs=4.6, ha="left")
    x = go(1.0 + dx + 3.0, ym)
    ts = 1.3
    token(ax, x, ym - ts / 2, ts, "cls")
    for k in range(4):
        token(ax, x + (k + 1) * ts, ym - ts / 2, ts, "patch")
    txt(ax, x + 5.5 * ts, ym, "…", fs=6)
    token(ax, x + 6 * ts, ym - ts / 2, ts, "patch")
    txt(ax, x + 3.9 * ts, lab_y, "128²×64 crop:\n16×16×8\npatches,\n512 + CLS", va="top", fs=4.6)
    x = go(x + 7 * ts, ym)
    ew = 8.0
    box(ax, x, y0 + 0.6, ew, dy - 1.2, FROZEN_FC, FROZEN_EC, lw=0.7, r=0.5)
    for k in range(12):
        ax.plot([x + 0.6 + k * (ew - 1.2) / 11] * 2, [y0 + 1.3, y0 + dy - 2.4], color=FROZEN_EC, lw=0.5, zorder=4)
    txt(ax, x + ew / 2, y0 + dy - 1.3, "ViT-L", fs=4.8, c=INK)
    txt(ax, x + ew / 2, lab_y, "24 blocks,\n1080-d,\nper crop", va="top", fs=4.6)
    x = go(x + ew, ym)
    token(ax, x, ym - ts / 2, ts, "cls"); token(ax, x + ts, ym - ts / 2, ts, "patch2")
    txt(ax, x + ts, lab_y, "[CLS ‖\nmean],\n2160-d", va="top", fs=4.6)
    x = go(x + 2 * ts, ym)
    box(ax, x, y0 + 1.6, 6.5, dy - 3.2, FROZEN_FC, FROZEN_EC, lw=0.7, r=0.5)
    for k in range(4):
        ax.plot([x + 1.2 + k * 1.4] * 2, [y0 + 2.3, y0 + dy - 2.3], color=FROZEN_EC, lw=0.5, zorder=4)
    txt(ax, x + 3.25, lab_y, "combiner:\n4 blocks\nover crops", va="top", fs=4.6)
    x = go(x + 6.5, ym)
    token(ax, x, ym - ts / 2 + 2.2, ts, "cls")
    for k in range(3):
        token(ax, x, ym - ts / 2 - k * ts + 0.4, ts, "patch2")
    txt(ax, x + ts / 2 + 2.4, lab_y, "scan CLS\n+ N crop\ntokens", va="top", fs=4.6)
    x += ts + 0.6
    hx = x + 2.8
    hw = 99.0 - hx
    for t, yy in [("Lin: scan CLS → linear", 0.82), ("Attn: scan CLS + N crop tokens", 0.5),
                  ("LoRA+ABMIL: crop tokens → ABMIL → Attn", 0.18)]:
        arrow(ax, x + 0.2, ym, hx - 0.2, y0 + dy * yy)
        box(ax, hx, y0 + dy * yy - 1.15, hw, 2.3, HEAD_C, BLUE, lw=0.8, r=0.6)
        txt(ax, hx + hw / 2, y0 + dy * yy, t, fs=4.5, c=INK)

    # ---------------- (b) Pillar-0 (pillar0_cls.py; remote multimodal_atlas.py / multimodal_msa.py; CLS 2.3)
    y1 = 90.5 - sh
    head(ax, 0.8, y1 + dy + 2.4, "(b) Pillar-0 (head CT): multi-window input, conv stem, three-stage multi-scale encoder")
    ym, lab_y = y1 + dy / 2, y1 - 1.2
    chans = [np.clip((head_ct - 0.5) * g + 0.5, 0, 1) for g in (0.6, 0.8, 1.0, 1.3, 1.7, 2.2)]
    stack(fig, ax, 1.0, y1, s_in, chans)
    txt(ax, 1.0, lab_y, "128×256×256,\n0.5×0.5×1.25 mm;\n11 channels (10 HU\nwindows, min–max)", va="top", fs=4.6, ha="left")
    x = go(1.0 + dx + 3.5, ym)
    box(ax, x, y1 + 1.5, 6.0, dy - 3.0, FROZEN_FC, FROZEN_EC, lw=0.7, r=0.5)
    txt(ax, x + 3.0, ym, "conv\nstem", fs=4.8, c=INK)
    txt(ax, x + 3.6, lab_y, "8×8×4\npatches,\n32³ grid", va="top", fs=4.6)
    x = go(x + 6.0, ym)
    gx = x + 0.3
    for k, sz in enumerate([6.6, 4.2, 2.2]):   # sizes are schematic: scale k = stem grid max-pooled by 4^k per axis
        ddx, _ = cuboid(ax, gx, ym - sz / 2, sz, sz, d=sz * 0.5, fc=PATCH_C if k == 0 else PATCH2_C, ec=NAVY, lw=0.5)
        txt(ax, gx + sz / 2, ym - 4.6, ["32³", "8³", "2³"][k], fs=4.6, c=INK)
        gx += sz + ddx + 0.8
    txt(ax, x + 9.0, lab_y - 0.6, "tokens at 3 scales\n(max-pool by 4\nper axis), attention\nwithin and across", va="top", fs=4.6)
    x = go(x + 20.0, ym)
    ew = 13.0
    box(ax, x, y1 + 0.6, ew, dy - 1.2, FROZEN_FC, FROZEN_EC, lw=0.7, r=0.5)
    for k, (nb, w_, sc) in enumerate([(2, 2.4, "32³ 8³ 2³"), (2, 2.4, "8³ 2³"), (8, 6.0, "2³")]):
        sx = x + 0.6 + [0, 3.0, 6.0][k]
        box(ax, sx, y1 + 1.4, w_, dy - 3.6, "white", FROZEN_EC, lw=0.5, r=0.3, z=4)
        txt(ax, sx + w_ / 2, ym - 0.2, f"{nb}", fs=5.0, c=INK)
    txt(ax, x + ew / 2, y1 + dy - 1.3, "384-d", fs=4.6, c=INK)
    txt(ax, x + ew / 2, lab_y, "3 stages of 2, 2, 8\nblocks; scales 32³,\n8³, 2³ leave after\nstages 1, 2, 3", va="top", fs=4.6)
    x = go(x + ew, ym)
    hx = x
    hw = 99.0 - hx
    for t, dyc in [("Lin: per-scale max-pool,\nL2-normalized, 1152-d → linear", 4.25),
                   ("Attn: tokens of all scales\n(32³ avg-pooled to 16³)", 0.0),
                   ("LoRA+ABMIL: same tokens\n→ ABMIL → Attn", -4.25)]:
        box(ax, hx, ym + dyc - 1.95, hw, 3.9, HEAD_C, BLUE, lw=0.8, r=0.6)
        txt(ax, hx + hw / 2, ym + dyc, t, fs=4.5, c=INK)

    if not NO_VOCO:
        # ---------------- (c) VoCo (voco_cls.py: SwinUNETR(use_v2).swinViT, feature_size 96, depths 2/2/2/2; CLS 2.4)
        y2 = 67.0
        head(ax, 0.8, y2 + dy + 2.4, "(c) VoCo: 96³ windows, four-stage Swin encoder, one token per window")
        ym, lab_y = y2 + dy / 2, y2 - 1.2
        stack(fig, ax, 1.0, y2, s_in, vol)
        fax = thumb(fig, ax, 1.0, y2, s_in, front)
        for k in range(1, 3):
            fax.axvline(k * n1 / 3 - 0.5, color=GEO_C, lw=0.6); fax.axhline(k * n0 / 3 - 0.5, color=GEO_C, lw=0.6)
        fax.add_patch(Rectangle((n1 / 3 - 0.5, n0 / 3 - 0.5), n1 / 3, n0 / 3, fc="none", ec=GEO_C, lw=1.5))
        txt(ax, 1.0, lab_y, "RAS, 1.5 mm,\nHU −175…250,\n96³ windows\n(train: ≤ 16)", va="top", fs=4.6, ha="left")
        x = go(1.0 + dx + 3.0, ym)
        box(ax, x, y2 + 1.5, 6.0, dy - 3.0, FROZEN_FC, FROZEN_EC, lw=0.7, r=0.5)
        txt(ax, x + 3.0, ym, "patch\nembed", fs=4.8, c=INK)
        txt(ax, x + 3.0, lab_y, "2³ patches,\n96 × 48³", va="top", fs=4.6)
        x = go(x + 6.0, ym)
        ew = 22.0
        box(ax, x, y2 + 0.6, ew, dy - 1.2, FROZEN_FC, FROZEN_EC, lw=0.7, r=0.5)
        for k, (dim, out_) in enumerate([(96, 192), (192, 384), (384, 768), (768, 1536)]):
            sx = x + 0.6 + k * 5.3
            box(ax, sx, y2 + 1.4, 4.6, dy - 3.6, "white", FROZEN_EC, lw=0.5, r=0.3, z=4)
            txt(ax, sx + 2.3, ym + 0.9, "2", fs=5.0, c=INK)
            txt(ax, sx + 2.3, ym - 1.3, f"{dim}", fs=3.6, c=INK2)
        txt(ax, x + ew / 2, lab_y, "2 blocks per stage, widths\n96, 192, 384, 768 (shifted\n7³ windows); stage outputs\nafter merging: 192, 384,\n768, 1536 at strides 4–32", va="top", fs=4.6)
        x = go(x + ew, ym)
        for k in range(3):
            token(ax, x, ym - ts / 2 + (1 - k) * ts, ts, "patch2")
        txt(ax, x + ts / 2 + 5.0, lab_y, "window token: means\nof stage-2–4 outputs,\nconcatenated: 384 +\n768 + 1536 = 2688-d", va="top", fs=4.6)
        x += ts + 0.6
        hx = x + 3.2
        hw = 99.0 - hx
        for t, yy in [("Lin: mean window token → linear", 0.82), ("Attn: all window tokens (projected to 768-d)", 0.5),
                      ("LoRA+ABMIL: window tokens → ABMIL → Attn", 0.18)]:
            arrow(ax, x + 0.2, ym, hx - 0.2, y2 + dy * yy)
            box(ax, hx, y2 + dy * yy - 1.15, hw, 2.3, HEAD_C, BLUE, lw=0.8, r=0.6)
            txt(ax, hx + hw / 2, y2 + dy * yy, t, fs=4.5, c=INK)

    # ---------------- (d) the eleven classification methods
    head(ax, 0.8, 52.0, "(c) What each classification method trains" if NO_VOCO else "(d) What each classification method trains")
    bbs = ("tapct", "spectre", "pillar0") if NO_VOCO else ("tapct", "spectre", "pillar0", "voco")
    pkeys = ("p1", "p2", "p3", "p4")[:len(bbs)]
    cols = ENC_COLS + [("hd", "head", 14.0, "text"), ("sc", "steps", 10.0, "text"),
                       ("p1", "TAP-CT", 6.6, "text"), ("p2", "SPECTRE", 7.4, "text"), ("p3", "Pillar-0", 7.0, "text"), ("p4", "VoCo", 6.4, "text")]
    cols = cols[:len(cols) - 4 + len(bbs)]
    hd = {"lin": "pooled → linear", "babmil": "ABMIL → Attn", "lin_then_lorag": "linear"}
    order = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
    rows = []
    for m in order:
        vals = _cells(m)
        vals["hd"] = hd.get(m, "attentive probe")
        vals["sc"] = "4000 + 4000" if m.endswith("_then_lorag") else "8000"
        for key, bb in zip(pkeys, bbs):
            n, derived = cls_bb(cls, bb, m)
            vals[key] = fmt_m(n) + ("†" if derived else "")   # never-run blocks: stated in the caption, no mark
        rows.append((NAME[m], SUB.get(m, ""), vals))
    matrix(ax, 0.8, 41.5, rows, cols, row_h=2.95, name_w=15.0,
           groups=[("early blocks", "eq", "em"), ("late blocks", "lq", "lm"), ("encoder parameters updated", "p1", pkeys[-1], "right")])
    legend_row(ax, 1.0, 5.2, [(FROZEN_FC, FROZEN_EC, "frozen"),
                              (ADP_FC, ORANGE, "rank-16 adapter (m: DoRA magnitude; 2: added in stage 2)"),
                              (TRAIN_FC, TRAIN_EC, "trained in full")])
    lin = {bb: int(cls.loc[(bb, "lin"), "trainable"]) for bb in bbs}
    att = {bb: int(cls.loc[(bb, "attn"), "trainable"]) for bb in bbs}
    pl = {m: fmt_m(cls_bb(cls, "pillar0", m, grad_only=False)[0]) for m in ("lora", "lorag", "attnft", "fullft")}
    if NO_VOCO:   # the note is part of the caption (printed here for the main text, not drawn)
        print("CAPTION NOTE fig_arms:",
              "W_qkv: fused qkv (TAP-CT) or q and kv (SPECTRE, Pillar-0); Attn-only FT trains the whole attention module (with biases). "
              "Late blocks: TAP-CT 9-12, SPECTRE 17-24 and its combiner, the last stage of Pillar-0 (8 of 12 blocks). CVPR26 joint models "
              "also ran 26 000 steps (13 000 + 13 000). Head parameters for one label (TAP-CT / SPECTRE / Pillar-0): linear "
              f"{' / '.join(str(lin[b]) for b in lin)}; attentive probe {' / '.join(f'{att[b] / 1e6:.1f}' for b in att)} M. "
              "*Pillar-0: the per-scale blocks of scales that have already left the encoder, and the text projection, are also wrapped or "
              f"unfrozen but never run and receive no gradient; counting them: LoRA {pl['lora']}, LoRA_late {pl['lorag']}, "
              f"Attn-only FT {pl['attnft']}, Full FT {pl['fullft']}.")
        save(fig, "fig_arms")
        return
    txt(ax, 1.0, 3.3,
        "$W_{qkv}$: fused qkv (TAP-CT, VoCo) or q and kv (SPECTRE, Pillar-0); Attn-only FT trains the whole attention module (with biases;\n"
        "VoCo: also the relative position bias). Late blocks: TAP-CT 9–12, SPECTRE 17–24 and its combiner, the last stage of Pillar-0 (8 of 12\n"
        "blocks) and of VoCo (2 of 8). CVPR26 joint models also ran 26 000 steps (13 000 + 13 000). Head parameters for one label (TAP-CT /\n"
        f"SPECTRE / Pillar-0 / VoCo): linear {' / '.join(str(lin[b]) for b in lin)}; attentive probe "
        f"{' / '.join(f'{att[b] / 1e6:.1f}' for b in att)} M. †: same late blocks as {LATE}.\n"
        "*Pillar-0: the per-scale blocks of scales that have already left the encoder, and the text projection, are also wrapped or unfrozen\n"
        f"but never run and receive no gradient; counting them: LoRA {pl['lora']}, {LATE} {pl['lorag']}, Attn-only FT {pl['attnft']}, Full FT {pl['fullft']}.",
        fs=4.3, c=MUTED, ha="left", va="top")
    save(fig, "fig_arms")


# ================================================================== Figure: segmentation decoder + method matrix
def fig_arms_seg(seg) -> None:
    sh = 15.5 if NO_VOCO else 0.0       # NO_VOCO: no VoCo shape row, notes in the caption; panel (a) moves down
    ylo = -1.4 if NO_VOCO else -5.4
    H = 5.42 - sh * DRAW_W / 100 - (ylo + 5.4) * DRAW_W / 100
    fig, ax = canvas(H, ylo)
    head(ax, 0.8, 70.3 - sh, "(a) Segmentation decoder: tensor shapes (channels × depth × height × width)")
    y = 58.8 - sh
    # TAP-CT stride (4,8,8) on a 12x224x224 crop; SPECTRE (8,16,16) on 64x128x128; VoCo taps at strides 4-32 on 96^3,
    # each projected and resized to the stride-4 grid (fuse_index 0), then two decoder stages (heads.py l.51-100)
    stages = [("block tokens\n(4 taps)", "768×3×28²", "1080×8³", "192×24³…\n1536×3³", 64, 28, 1),
              ("1×1×1 conv,\nconcatenate", "4·128×3×28²", "4·128×8³", "4·128×24³", 512, 28, 1),
              ("3³ conv", "128×3×28²", "128×8³", "128×24³", 128, 28, 1),
              ("upsample,\n3³ conv", "64×6×56²", "64×16³", "64×48³", 64, 56, 2),
              ("upsample,\n3³ conv", "32×12×112²", "32×32³", "32×96³", 32, 112, 3),
              ("upsample,\n3³ conv", "16×12×224²", "16×64³", "–", 16, 224, 4),
              ("upsample,\n3³ conv", "–", "16×64×128²", "–", 16, 224, 4),
              ("1×1×1 conv", "C×12×224²", "C×64×128²", "C×96³", 8, 224, 4)]
    sbbs = ("tapct", "spectre") if NO_VOCO else ("tapct", "spectre", "voco")
    for name, yy in (("TAP-CT", y - 8.6), ("SPECTRE", y - 11.0), ("VoCo", y - 13.4))[:len(sbbs)]:
        txt(ax, 1.0, yy, name, fs=4.7, c=INK, ha="left", bold=True)
    xs = 13.0
    for k, (nm, t1, t2, t3, ch, hw_, dd) in enumerate(stages):
        w_ = 0.8 + 1.4 * np.log2(max(ch, 8) / 8) / 6
        h_ = 2.6 + 6.0 * np.log2(hw_ / 28) / 3
        fc = PATCH_C if k == 0 else HEAD_C
        ddx, _ = cuboid(ax, xs, y - h_ / 2 + 2.6, w_, h_, d=1.0 + dd * 0.8, fc=fc, ec=BLUE if k else FROZEN_EC, lw=0.6)
        cx = xs + (w_ + ddx) / 2
        txt(ax, cx, y - 4.0, nm, fs=4.4, c=INK, va="top")
        txt(ax, cx, y - 8.6, t1, fs=4.1)
        txt(ax, cx, y - 11.0, t2, fs=4.1)
        if not NO_VOCO:
            txt(ax, cx, y - 13.4, t3, fs=4.1)
        if k < len(stages) - 1:
            arrow(ax, xs + w_ + ddx + 0.6, y + 2.6, xs + 11.2 - 0.6, y + 2.6)
        xs += 11.2
    dec = {bb: seg.loc[(bb, "dec")] for bb in sbbs}
    linp = {bb: int(seg.loc[(bb, "lin"), "trainable"]) // int(seg.loc[(bb, "lin"), "classes"]) for bb in sbbs}
    C = {int(d["classes"]) for d in dec.values()}
    assert len(C) == 1
    dec_txt = " / ".join(f"{d['trainable'] / 1e6:.2f}" for d in dec.values()) + \
        f" M ({' / '.join(BACKBONE_TEX[b] for b in sbbs)}, C = {C.pop()})"
    if NO_VOCO:   # the note is part of the caption (printed here for the main text, not drawn)
        print("CAPTION NOTE fig_arms_seg (a):",
              "Taps: blocks 3, 6, 9, 12 (TAP-CT) or 6, 12, 18, 24 (SPECTRE) after the final norm. Each 3³ conv is followed by instance norm "
              "and LeakyReLU; trilinear upsampling along the axes whose stride is still above 1; no skips from the input. SPECTRE: per-crop "
              f"ViT-L only. Decoder parameters: {dec_txt}. Lin: a 1×1×1 conv on the deepest tap (block 12 or "
              f"block 24) with trilinear upsampling; {' / '.join(str(linp[b]) for b in linp)} parameters per structure. "
              "C = 1 per structure, or the number of structures (joint).")
    else:
        txt(ax, 1.0, y - 15.0,
            "Taps: blocks 3, 6, 9, 12 (TAP-CT) or 6, 12, 18, 24 (SPECTRE) after the final norm; for VoCo the four stage outputs\n"
            "192×24³, 384×12³, 768×6³ and 1536×3³ (strides 4 to 32, channel-normalized), resized to the stride-4 grid before\n"
            "concatenation. Each 3³ conv is followed by instance norm and LeakyReLU; trilinear upsampling along the axes whose\n"
            "stride is still above 1; no skips from the input. SPECTRE: per-crop ViT-L only. Decoder parameters:\n"
            f"{dec_txt}. Lin: a 1×1×1 conv on the deepest tap\n"
            "(block 12, block 24, or the VoCo stage-4 output at stride 32, i.e. 48 mm) with trilinear upsampling; "
            f"{' / '.join(str(linp[b]) for b in linp)}\nparameters per structure. C = 1 per structure, or the number of structures (joint).",
            fs=4.3, c=MUTED, ha="left", va="top")
    # ---------------- (b) the seven methods
    head(ax, 0.8, 29.0, "(b) What each segmentation method trains")
    cols = ENC_COLS + [("hd", "head", 13.0, "text"), ("sc", "steps", 10.5, "text"),
                       ("p1", "TAP-CT", 7.0, "text"), ("p2", "SPECTRE", 7.6, "text"), ("p3", "VoCo", 6.0, "text")]
    cols = cols[:len(cols) - 3 + len(sbbs)]
    order = ["lin", "dec", "lora", "lorag", "dora", "dec_then_lora", "fullft"]
    rows = []
    for m in order:
        vals = _cells(m)
        vals["hd"] = "1×1×1 conv" if m == "lin" else "decoder"
        vals["sc"] = "4000 + 4000" if m == "dec_then_lora" else "8000"
        for key, bb in zip(("p1", "p2", "p3"), sbbs):
            r = seg.loc[(bb, m)]
            n = int(r["backbone_trainable"]) - int(r["no_grad_backbone"])
            vals[key] = fmt_m(n)   # combiner without gradient: stated in the caption, no mark
        rows.append((NAME[m], SUB.get(m, ""), vals))
    matrix(ax, 0.8, 19.8, rows, cols, row_h=2.45, name_w=15.0,
           groups=[("early blocks", "eq", "em"), ("late blocks", "lq", "lm"), ("encoder parameters updated", "p1", ("p1", "p2", "p3")[len(sbbs) - 1], "right")])
    legend_row(ax, 1.0, 0.4, [(FROZEN_FC, FROZEN_EC, "frozen"), (ADP_FC, ORANGE, "rank-16 adapter (m: DoRA magnitude; 2: added in stage 2)"),
                               (TRAIN_FC, TRAIN_EC, "trained in full")])
    sp = {m: seg.loc[("spectre", m)] for m in ("lora", "dora", "dec_then_lora", "fullft")}
    starred = [m for m in sp if int(sp[m]["no_grad_backbone"])]
    print("CAPTION NOTE fig_arms_seg (b):",
          "AMOS22 and TotalSegmentator joint models: 24 000 steps (12 000 + 12 000). *SPECTRE: "
          f"{', '.join(NAME[m] for m in starred[:-1])} and {NAME[starred[-1]]} also wrap or unfreeze "
          "the feature combiner, which never enters the segmentation features and receives no gradient; counting it, "
          f"{' / '.join(fmt_m(sp[m]['backbone_trainable']) for m in starred)}.")
    save(fig, "fig_arms_seg")


# ================================================================== macros
def write_param_macros(cls: pd.DataFrame) -> None:
    """\\numParamsbb<Backbone><Method>: encoder parameters trained (classification, logged), as in fig_arms (d)."""
    lines = ["% generated by figures/diagrams.py from figures/data/trainable_params_cls.csv (logged training counts)"]
    for bb, tag in (("tapct", "Tapct"), ("spectre", "Spectre"), ("pillar0", "PillarZ")):
        ns = {}
        for m, mt in (("lora", "Lora"), ("attnft", "Attnft")):
            n, derived = cls_bb(cls, bb, m)
            assert not derived
            ns[m] = n
            lines.append(f"\\providecommand{{\\numParamsbb{tag}{mt}}}{{{fmt_m(n, sep=chr(92) + ',')}}}")
        # encoder parameters of Attn-only FT per encoder parameter of LoRA, rounded to an integer
        lines.append(f"\\providecommand{{\\numParamsratio{tag}AttnftLora}}{{{round(ns['attnft'] / ns['lora'])}}}")
    PAPER_TAB.mkdir(parents=True, exist_ok=True)
    (PAPER_TAB / "param_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--only", default="pipeline,arms,arms_seg")
    a = ap.parse_args()
    if a.refresh_data:
        refresh_data()
    vol, vpred, head_ct, cls, seg = load_data()
    todo = a.only.split(",")
    if "pipeline" in todo:
        fig_pipeline(vol, vpred, seg)
    if "arms" in todo:
        fig_arms(vol, head_ct, cls)
    if "arms_seg" in todo:
        fig_arms_seg(seg)
    write_param_macros(cls)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for t in todo:
        name = {"pipeline": "fig_pipeline", "arms": "fig_arms", "arms_seg": "fig_arms_seg"}[t]
        shutil.copyfile(OUT / f"{name}.pdf", PAPER_FIG / f"{name}.pdf")
    print("wrote", todo, "to", OUT, "and", PAPER_FIG)


if __name__ == "__main__":
    main()
