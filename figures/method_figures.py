"""Method diagrams of the paper, drawn with matplotlib from the code audit.

Every number comes from ``report/code_map_classification.md`` / ``report/code_map_segmentation.md`` or from the code
under ``aifofomo`` (see ``METHOD_FIGURES.md`` for the fact-by-fact sources). Outputs: ``out/fig_pipeline_cls``,
``out/fig_pipeline_seg``, ``out/fig_methods_cls``, ``out/fig_methods_seg``, ``out/fig_modes`` (PDF + PNG).

Run:  /home/g.yiasemis/miniconda3/envs/aifofomo/bin/python method_figures.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

import style
from style import BACKBONE_COL, BACKBONE_TEX, FAMILY_COL, INK, INK2, METHOD_TEX, TEXTWIDTH_IN

OUT = Path(__file__).resolve().parent / "out"
FROZEN, LOWRANK, FULL = FAMILY_COL["frozen"], FAMILY_COL["lowrank"], FAMILY_COL["fullrank"]
PAPER = "#ffffff"
FS = 5.5          # body text (minimum size on every figure)
FS_H = 6.5        # headers
LEGEND_HANDLES = [Patch(fc=FROZEN, ec=INK2, lw=0.4), Patch(fc=LOWRANK, ec=INK2, lw=0.4), Patch(fc=FULL, ec=INK2, lw=0.4)]
LEGEND_LABELS = ["frozen (pretrained weights)", "low-rank adapter, trained", "trained in full"]


# ----------------------------------------------------------------------------- helpers
def canvas(w: float, h: float):
    """Figure of ``w`` x ``h`` inches with one axes in inch coordinates."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def wrap(text: str, width_in: float, fs: float) -> str:
    cpi = 72.0 / (0.58 * fs)                      # average glyph width ~ 0.58 em (DejaVu Sans; conservative for LM Roman)
    n = max(6, int(width_in * cpi))
    out = []
    for para in text.split("\n"):
        out += textwrap.wrap(para, n, break_long_words=False) or [""]
    return "\n".join(out)


def box(ax, x, y, w, h, text="", fc=PAPER, ec=INK2, lw=0.5, fs=FS, pad=0.035, title=None, title_fs=None,
        ha="left", va="top", color=INK, round_=True, ls="-"):
    """Rounded box with wrapped text. ``title`` is a bold first line."""
    if round_:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03", fc=fc, ec=ec, lw=lw, ls=ls))
    else:
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls))
    ty = y + h - pad
    if title:
        ax.text(x + pad, ty, title, fontsize=title_fs or fs, fontweight="bold", va="top", ha="left", color=color)
        ty -= (title_fs or fs) * 1.3 / 72.0
    if text:
        body = wrap(text, w - 2 * pad, fs)
        if ha == "center":
            ax.text(x + w / 2, y + h / 2, body, fontsize=fs, va="center", ha="center", color=color, linespacing=1.12)
        else:
            ax.text(x + pad, ty, body, fontsize=fs, va="top", ha="left", color=color, linespacing=1.12)


def arrow(ax, x0, y0, x1, y1, lw=0.6, color=INK2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=5))


def legend(ax, x, y, handles=LEGEND_HANDLES, labels=LEGEND_LABELS, ncol=3, fs=FS, extra=None):
    hs, ls = list(handles), list(labels)
    if extra:
        for h, l in extra:
            hs.append(h); ls.append(l)
    ax.legend(hs, ls, loc="lower left", bbox_to_anchor=(x, y), bbox_transform=ax.transData, ncol=ncol, fontsize=fs,
              handlelength=1.3, handleheight=0.8, columnspacing=1.0, handletextpad=0.5, borderaxespad=0, borderpad=0)


def column_header(ax, x, y, w, text, fs=FS_H):
    ax.text(x + w / 2, y, text, fontsize=fs, fontweight="bold", ha="center", va="bottom", color=INK)


def row_label(ax, x, y, h, key):
    ax.text(x, y + h / 2, BACKBONE_TEX[key], fontsize=FS_H, fontweight="bold", color=BACKBONE_COL[key],
            ha="left", va="center", rotation=90, rotation_mode="anchor")


# ============================================================================= Fig 1: classification pipeline
def fig_pipeline_cls():
    W = TEXTWIDTH_IN
    row_h, gap, top, bot = 1.30, 0.07, 0.32, 0.40
    H = top + 4 * row_h + 3 * gap + bot
    fig, ax = canvas(W, H)
    x_lab = 0.02
    cols = [  # (x, w)
        (0.14, 1.07),   # input volume
        (1.30, 1.00),   # coverage
        (2.39, 1.38),   # encoder
        (3.86, 1.07),   # feature read by the head
        (5.02, 0.96),   # head
    ]
    heads = ["Input volume", "Coverage (windows / crops)", "Encoder", "Feature read by the head", "Head"]
    y_head = H - top + 0.03
    for (x, w), t in zip(cols, heads):
        column_header(ax, x, y_head, w, t)

    rows = {
        "tapct": [
            "Orientation LPS. Native slice spacing; in-plane trilinear resize to 224×224, depth D kept. "
            "In the model: HU clamp [−1008, 822], z-score (μ −86.81, σ 322.63); depth padded with air to ≥ 12 and to a multiple of 4.",
            "12×224×224 slabs along depth: starts 0, 3, 6, … (overlap 0.75 → stride 3) plus one slab ending at the last slice. "
            "Train: random subset of ≤ 32 slabs; eval: all. 16 slabs per forward.",
            "3-D ViT-B: 12 blocks, 768-d, 12 heads; patch 4×8×8 → 3×28×28 = 2352 patch tokens + CLS + 4 register tokens per slab. "
            "Fused qkv Linear + proj; MLP fc1/fc2. SDPA attention.",
            "CLS of the last block (after the final norm), one per slab: K × 768.\n"
            "Lin: mean over the K slabs → 768-d.\n"
            "Attn: the K CLS tokens (768-d).",
            "Lin: Dropout 0.2 → Linear(768, T).\n"
            "Attn: attentive probe, hidden 768, 2 self-attention blocks (8 heads) over [CLS]+tokens → LayerNorm → Dropout 0.2 → Linear(768, T).",
        ],
        "spectre": [
            "Orientation RAS. Resampled to 0.5×0.5×1.0 mm (x, y, z) straight into a centred 512×512×256 (X,Y,Z) region, "
            "padded with the scan minimum. HU [−1000, 1000] → [0, 1], clipped.",
            "Grid of 128×128×64 crops, no overlap: 4×4×4 = 64 crops at eval. "
            "Train: one random contiguous 2×2×2 sub-grid (8 crops). 16 crops per forward.",
            "Per crop ViT-L: 24 blocks, 1080-d, 12 heads; patch 16×16×8 → 8×8×8 = 512 tokens + CLS; separate q and kv Linears + proj; RoPE. "
            "Per-crop [CLS ‖ mean patch] (2160-d) → feature combiner: ViT of 4 blocks, 1080-d, 12 heads, over the crop grid.",
            "Combiner output: CLS + one token per crop, 1080-d (N = 8 train / 64 eval).\n"
            "Lin: combiner CLS → 1080-d.\n"
            "Attn: the N + 1 combiner tokens.",
            "Lin: Dropout 0.2 → Linear(1080, T).\n"
            "Attn: attentive probe, hidden 1080.",
        ],
        "pillar0": [
            "Orientation LPI. Resampled to 0.5×0.5×1.25 mm into a centred 128×256×256 (D,H,W) region. "
            "In the model: 11 channels = 10 HU centre/width windows (lung, mediastinum, abdomen, liver, bone, brain, subdural, stroke, "
            "temporal bone, soft tissue) + per-volume min–max, each in [0, 1].",
            "Whole volume in one forward pass (no windows, no crops).",
            "Conv stem (Conv3d + BatchNorm), patch 8×8×4 (H,W,D) → 32³ grid; 3 Atlas stages of 2, 2, 8 multi-scale attention blocks "
            "(dim 384, 6 heads, 3 scales, merge ratio 4 → grids 32³, 8³, 2³), no class token; separate q and kv Linears + proj.",
            "Lin: pooled image embedding, 1152-d, L2-normalised.\n"
            "Attn: tokens of the 3 scales (scale 1 average-pooled 32³ → 16³ to stay ≤ 4096 tokens): 16³ + 8³ + 2³ tokens, "
            "each scale in its own 384-d block of a 1152-d token.",
            "Lin: Dropout 0.2 → Linear(1152, T).\n"
            "Attn: attentive probe, hidden 1152.",
        ],
        "voco": [
            "Orientation RAS. Resampled to 1.5 mm isotropic, padded (never cropped) to ≥ 96³. HU [−175, 250] → [0, 1], clipped.",
            "96³ windows, no overlap (last window shifted to the border), every corner combination. "
            "Train: random subset of ≤ 16 windows; eval: all. 4 windows per forward.",
            "Swin encoder (SwinUNETR v2): patch 2, 4 stages × 2 shifted-window blocks, heads 3/6/12/24, window 7, widths 96/192/384/768; "
            "stage outputs 192/384/768/1536 ch at strides 4/8/16/32 (channel LayerNorm). Fused qkv + proj; MLP linear1/linear2.",
            "Per window: concat of the spatial means of stage 2, 3, 4 outputs = 384 + 768 + 1536 = 2688-d, "
            "then running (EMA) per-channel standardisation.\n"
            "Lin: mean over windows → 2688-d.\n"
            "Attn: the K window tokens (2688-d, projected to 768).",
            "Lin: Dropout 0.2 → Linear(2688, T).\n"
            "Attn: attentive probe, hidden 768.",
        ],
    }
    enc_tag = {"tapct": "LoRA target: fused qkv", "spectre": "LoRA: q, kv (and combiner)",
               "pillar0": "LoRA target: q, kv", "voco": "LoRA target: fused qkv"}
    for i, key in enumerate(style.BACKBONES):
        y = H - top - (i + 1) * row_h - i * gap
        row_label(ax, x_lab + 0.06, y, row_h, key)
        ax.plot([x_lab + 0.09, x_lab + 0.09], [y + 0.05, y + row_h - 0.05], color=BACKBONE_COL[key], lw=1.4, solid_capstyle="butt")
        fcs = [PAPER, PAPER, FROZEN, PAPER, FULL]
        for (x, w), txt, fc in zip(cols, rows[key], fcs):
            box(ax, x, y, w, row_h, txt, fc=fc)
        # adapter tag inside the encoder box (bottom-right)
        ex, ew = cols[2]
        tw, th = ew - 0.08, 0.13
        box(ax, ex + 0.04, y + 0.04, tw, th, enc_tag[key], fc=LOWRANK, fs=5.5, ha="center", lw=0.4)
        for (x0, w0), (x1, _) in zip(cols[:-1], cols[1:]):
            arrow(ax, x0 + w0, y + row_h / 2, x1, y + row_h / 2)
    legend(ax, 0.14, 0.03)
    ax.text(0.14, 0.17, wrap(
            "T = number of outputs (1 per-task, K joint). LoRA+ABMIL: gated-attention MIL pooling (hidden 128) over the token set, "
            "then the probe on the pooled token. Encoder: frozen under Lin/Attn, adapted or trained under the other methods.", W - 0.3, FS),
            fontsize=FS, ha="left", va="bottom", color=INK2, linespacing=1.12)
    style.save(fig, OUT, "fig_pipeline_cls")


# ============================================================================= Fig 2: segmentation pipeline
def fig_pipeline_seg():
    W = TEXTWIDTH_IN
    row_h, gap, top = 0.96, 0.07, 0.32
    chain_h, chain_top_gap, bot = 0.62, 0.34, 0.40
    H = top + 3 * row_h + 2 * gap + chain_top_gap + chain_h + bot
    fig, ax = canvas(W, H)
    cols = [(0.14, 1.20), (1.43, 1.42), (2.94, 0.92), (3.95, 1.10), (5.14, 0.84)]
    heads = ["Training crop", "Tapped feature maps", "Lin head", "Dec head", "Evaluation"]
    y_head = H - top + 0.03
    for (x, w), t in zip(cols, heads):
        column_header(ax, x, y_head, w, t)
    rows = {
        "tapct": [
            "LPS; in-plane resize to 224, native depth. Crop 12×224×224 (D,H,W), centre on a foreground voxel with p = 0.67, else uniform. "
            "In the model: HU clamp [−1008, 822], z-score. Batch 2 crops × 2 accumulation = 4 crops per step.",
            "Blocks 3, 6, 9, 12 of 12 (final LayerNorm on each; CLS and register tokens dropped). "
            "4 maps × 768 ch on one grid: 3×28×28 tokens, stride (4, 8, 8).",
            "Conv 1³ (768 → C) on the block-12 map, trilinear ×(4, 8, 8) to the crop.",
            "Fuse grid stride (4, 8, 8). N = 3 up-stages: ×(2,2,2), ×(2,2,2), ×(1,2,2); channels 128 → 64 → 32 → 16. "
            "2.45 M parameters (C = 15).",
            "ROI 12×224×224, overlap 0.5, Gaussian weights, 4 windows per batch; border fill −1008 HU; logits stitched on the CPU.",
        ],
        "spectre": [
            "RAS; 1.0 mm isotropic (image linear, label nearest). HU [−1000, 1000] → [0, 1]. Crop 64×128×128 (D,H,W), same foreground sampling.",
            "Blocks 6, 12, 18, 24 of 24 (final norm on each; prefix token dropped). 4 maps × 1080 ch: 8×8×8 tokens, stride (8, 16, 16). "
            "The feature combiner is not used.",
            "Conv 1³ (1080 → C) on the block-24 map, trilinear ×(8, 16, 16).",
            "Fuse grid stride (8, 16, 16). N = 4: ×(2,2,2) three times, then ×(1,2,2); channels 128 → 64 → 32 → 16 → 16. "
            "2.62 M parameters.",
            "ROI 64×128×128, overlap 0.25, Gaussian, 8 windows per batch; border fill 0.",
        ],
        "voco": [
            "RAS; 1.5 mm isotropic. HU [−175, 250] → [0, 1]. Crop 96³, same foreground sampling.",
            "Swin stage outputs 1–4 (parameter-free channel LayerNorm): 192 / 384 / 768 / 1536 ch at strides 4 / 8 / 16 / 32 "
            "(grids 24³, 12³, 6³, 3³).",
            "Conv 1³ (1536 → C) on the stage-4 map, trilinear ×32.",
            "Taps resized to the stride-4 grid (24³). N = 2: ×2, ×2; channels 128 → 64 → 32. 2.42 M parameters.",
            "ROI 96³, overlap 0.5, Gaussian, 4 windows per batch; border fill 0.",
        ],
    }
    for i, key in enumerate(["tapct", "spectre", "voco"]):
        y = H - top - (i + 1) * row_h - i * gap
        row_label(ax, 0.08, y, row_h, key)
        ax.plot([0.11, 0.11], [y + 0.05, y + row_h - 0.05], color=BACKBONE_COL[key], lw=1.4, solid_capstyle="butt")
        fcs = [PAPER, FROZEN, FULL, FULL, PAPER]
        for (x, w), txt, fc in zip(cols, rows[key], fcs):
            box(ax, x, y, w, row_h, txt, fc=fc)
        ex, ew = cols[1]
        tag = {"tapct": "LoRA target: fused qkv", "spectre": "LoRA target: q, kv", "voco": "LoRA target: fused qkv"}[key]
        box(ax, ex + 0.04, y + 0.04, ew - 0.08, 0.13, tag, fc=LOWRANK, ha="center", lw=0.4)
        for (x0, w0), (x1, _) in zip(cols[:-1], cols[1:]):
            arrow(ax, x0 + w0, y + row_h / 2, x1, y + row_h / 2)

    # Dec head, layer by layer (shared by the three backbones)
    yc = bot + 0.02
    ax.text(0.14, yc + chain_h + 0.03, "Dec head, layer by layer (N = 3 / 4 / 2 for TAP-CT / SPECTRE / VoCo; no skip connections from the input volume)",
            fontsize=FS_H, fontweight="bold", va="bottom", ha="left", color=INK)
    steps = [
        ("4 tapped maps", PAPER),
        ("4 × Conv 1³ → 128 ch, each resized (trilinear) to the fuse grid", FULL),
        ("concat → 512 ch", PAPER),
        ("Conv 3³ + InstanceNorm + LeakyReLU → 128 ch", FULL),
        ("N × [ ×2 trilinear up (axes with stride > 1) → Conv 3³ + IN + LeakyReLU, channels halved, min 16 ]", FULL),
        ("Conv 1³ → C logits", FULL),
        ("trilinear to crop size; sigmoid per structure (BCE + Dice); threshold 0.5 for Dice", PAPER),
    ]
    widths = [0.50, 0.98, 0.50, 0.84, 1.28, 0.50, 0.90]
    gapc = 0.05
    x = 0.14
    for (txt, fc), w in zip(steps, widths):
        box(ax, x, yc, w, chain_h, txt, fc=fc, ha="center")
        x += w
        if w is not widths[-1]:
            pass
        x_next = x + gapc
        if txt != steps[-1][0]:
            arrow(ax, x, yc + chain_h / 2, x_next, yc + chain_h / 2)
        x = x_next
    legend(ax, 0.14, 0.03)
    ax.text(0.14, 0.17, wrap("Lin head = Conv 1³ on the last tap + trilinear upsampling to the crop. C = number of structures (1 per-task, K joint). "
            "Encoder: frozen under Lin/Dec, adapted or trained under the other methods.", W - 0.3, FS),
            fontsize=FS, ha="left", va="bottom", color=INK2, linespacing=1.12)
    style.save(fig, OUT, "fig_pipeline_seg")


# ============================================================================= matrices (Figs 3, 4)
ENC_COLS = ["patch\nembed /\nstem", "attn\nqkv", "attn\nout proj", "MLP,\nnorms", "attn\nqkv", "attn\nout proj", "MLP,\nnorms", "head"]
KIND_FC = {"frozen": FROZEN, "lora": LOWRANK, "dora": LOWRANK, "rslora": LOWRANK, "full": FULL}
KIND_LABEL = {"frozen": "", "lora": "A, B", "dora": "A, B, m", "rslora": "A, B\nα/√r", "full": ""}


def draw_matrix(ax, x0, y_top, rows, text_cols, cell_w, cell_h, lab_w, text_w, head_label):
    """rows: list of (method_key, [8 kinds], marks); text_cols: list of (header, [values], width)."""
    n = len(rows)
    y0 = y_top - n * cell_h
    # group headers
    gx = x0 + lab_w
    yh = y_top + 0.02
    sub_h = 0.36
    for lbl, c0, c1 in [("early blocks", 1, 3), ("late blocks", 4, 6)]:
        xa, xb = gx + c0 * cell_w, gx + (c1 + 1) * cell_w
        ax.plot([xa + 0.02, xb - 0.02], [yh + sub_h + 0.02] * 2, color=INK2, lw=0.5)
        ax.text((xa + xb) / 2, yh + sub_h + 0.04, lbl, fontsize=FS_H, fontweight="bold", ha="center", va="bottom", color=INK)
    for j, c in enumerate(ENC_COLS):
        ax.text(gx + (j + 0.5) * cell_w, yh, c, fontsize=FS, ha="center", va="bottom", color=INK, linespacing=1.05)
    ax.text(x0 + lab_w - 0.05, yh, head_label, fontsize=FS_H, fontweight="bold", ha="right", va="bottom", color=INK)
    # text column headers
    tx = gx + 8 * cell_w + 0.06
    for hdr, _vals, w in text_cols:
        ax.text(tx + w / 2, yh, hdr, fontsize=FS, fontweight="bold", ha="center", va="bottom", color=INK, linespacing=1.05)
        tx += w
    # rows
    for i, (key, kinds, marks) in enumerate(rows):
        y = y_top - (i + 1) * cell_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((x0, y), lab_w + 8 * cell_w + 0.06 + sum(w for *_, w in text_cols), cell_h, fc="#f4f4f4", ec="none", zorder=0))
        ax.text(x0 + lab_w - 0.05, y + cell_h / 2, METHOD_TEX[key], fontsize=FS_H, ha="right", va="center", color=INK)
        for j, kind in enumerate(kinds):
            cx = gx + j * cell_w
            ax.add_patch(Rectangle((cx + 0.015, y + 0.015), cell_w - 0.03, cell_h - 0.03, fc=KIND_FC[kind], ec=INK2, lw=0.35, zorder=1))
            lbl = KIND_LABEL[kind]
            if marks and j in marks:
                lbl = (lbl + "\n" if lbl else "") + marks[j]
            if lbl:
                ax.text(cx + cell_w / 2, y + cell_h / 2, lbl, fontsize=5.5, ha="center", va="center", color=INK, zorder=2, linespacing=1.0)
        tx = gx + 8 * cell_w + 0.06
        for _hdr, vals, w in text_cols:
            ax.text(tx + w / 2, y + cell_h / 2, vals[i], fontsize=FS, ha="center", va="center", color=INK, linespacing=1.0)
            tx += w
    return y0


def notes_panel(ax, x, y, w, h, columns):
    """Three note columns under a matrix; each column is a list of (title, text)."""
    cw = (w - 0.1 * (len(columns) - 1)) / len(columns)
    for k, col in enumerate(columns):
        cx = x + k * (cw + 0.1)
        ax.add_patch(FancyBboxPatch((cx, y), cw, h, boxstyle="round,pad=0,rounding_size=0.03", fc="#fafafa", ec=INK2, lw=0.4))
        ty = y + h - 0.04
        for title, text in col:
            ax.text(cx + 0.04, ty, title, fontsize=FS, fontweight="bold", va="top", ha="left", color=INK)
            ty -= FS * 1.35 / 72.0
            body = wrap(text, cw - 0.08, FS)
            ax.text(cx + 0.04, ty, body, fontsize=FS, va="top", ha="left", color=INK, linespacing=1.12)
            ty -= (body.count("\n") + 1) * FS * 1.12 / 72.0 + 0.07


def fig_methods_cls():
    W = TEXTWIDTH_IN
    cell_w, cell_h, lab_w = 0.32, 0.235, 0.72
    text_cols_spec = [("head\ntype", 0.72), ("warm\nstart", 0.72), ("steps", 0.70)]
    n = 11
    top_h, bot = 0.62, 0.30
    panel_h = 1.34
    mat_h = n * cell_h
    H = top_h + mat_h + 0.12 + panel_h + bot
    fig, ax = canvas(W, H)
    F, L, D, R, U = "frozen", "lora", "dora", "rslora", "full"
    rows = [
        ("lin",             [F, F, F, F, F, F, F, U], None),
        ("attn",            [F, F, F, F, F, F, F, U], None),
        ("lora",            [F, L, F, F, L, F, F, U], None),
        ("lorag",           [F, F, F, F, L, F, F, U], None),
        ("babmil",          [F, L, F, F, L, F, F, U], None),
        ("dora",            [F, D, F, F, D, F, F, U], None),
        ("rslora",          [F, R, F, F, R, F, F, U], None),
        ("lin_then_lorag",  [F, F, F, F, L, F, F, U], None),
        ("attn_then_lorag", [F, F, F, F, L, F, F, U], None),
        ("attnft",          [F, U, U, F, U, U, F, U], {1: "†", 2: "†", 4: "†", 5: "†"}),
        ("fullft",          [U, U, U, U, U, U, U, U], None),
    ]
    head = ["linear", "attentive", "attentive", "attentive", "ABMIL →\nattentive", "attentive", "attentive",
            "linear\n(from Lin)", "attentive\n(from Attn)", "attentive", "attentive"]
    warm = ["–", "–", "–", "–", "–", "–", "–", "Lin,\nstep 4000", "Attn,\nstep 4000", "–", "–"]
    steps = ["8000"] * 7 + ["4000 + 4000", "4000 + 4000", "8000", "8000"]
    text_cols = [(text_cols_spec[0][0], head, text_cols_spec[0][1]), (text_cols_spec[1][0], warm, text_cols_spec[1][1]),
                 (text_cols_spec[2][0], steps, text_cols_spec[2][1])]
    y_top = H - top_h
    draw_matrix(ax, 0.02, y_top, rows, text_cols, cell_w, cell_h, lab_w, None, "Method")
    columns = [
        [("Late blocks per backbone",
          "TAP-CT: blocks 9–12 of 12. SPECTRE: blocks 17–24 of 24 plus the 4 feature-combiner blocks. "
          "Pillar-0: stage 3 of 3 (8 of 12 multi-scale blocks, 3 scales each). VoCo: stage 4 of 4 (2 of 8 blocks)."),
         ("LoRA target per backbone",
          "Fused qkv Linear: TAP-CT, VoCo. Separate q and kv Linears: SPECTRE (backbone and combiner), Pillar-0. "
          "r 16, α 32, dropout 0.1 on the low-rank branch; scale α/r = 2; rsLoRA α/√r = 8.")],
        [("DoRA", "W = m · (W₀ + 2BA) / ‖W₀ + 2BA‖ per output row; m initialised to the row norms of W₀ and trained with A, B."),
         ("† Attn-only FT",
          "Every parameter of each matched attention module: qkv (or q, kv), out proj, their biases, and for VoCo the "
          "relative-position bias table. MLPs, norms and the patch embedding stay frozen.")],
        [("Warm start (two-stage)",
          "Head (for VoCo also the running token standardiser) copied from the parent's step-4000 checkpoint of its 8000-step run; "
          "adapters start fresh (B = 0); new optimizer, 4000 steps."),
         ("Learning rate / weight decay",
          "Lin 1e-3 / 0; Attn and all adapter methods 1e-4 / 0.01; Attn-only FT and Full FT backbone 2e-5 / 0.05, head 1e-4 / 0.01. "
          "Head dropout 0.2; batch 1.")],
    ]
    notes_panel(ax, 0.02, bot, W - 0.04, panel_h, columns)
    legend(ax, 0.02, 0.04, ncol=3)
    style.save(fig, OUT, "fig_methods_cls")


def fig_methods_seg():
    W = TEXTWIDTH_IN
    cell_w, cell_h, lab_w = 0.32, 0.27, 0.72
    text_cols_spec = [("head", 0.50), ("warm\nstart", 0.80), ("steps (8000 modes /\n15-structure joint / matched)", 1.30)]
    n = 7
    top_h, bot = 0.62, 0.30
    panel_h = 1.24
    H = top_h + n * cell_h + 0.12 + panel_h + bot
    fig, ax = canvas(W, H)
    F, L, D, U = "frozen", "lora", "dora", "full"
    rows = [
        ("lin",           [F, F, F, F, F, F, F, U], None),
        ("dec",           [F, F, F, F, F, F, F, U], None),
        ("lora",          [F, L, F, F, L, F, F, U], None),
        ("lorag",         [F, F, F, F, L, F, F, U], None),
        ("dora",          [F, D, F, F, D, F, F, U], None),
        ("fullft",        [U, U, U, U, U, U, U, U], None),
        ("dec_then_lora", [F, L, F, F, L, F, F, U], None),
    ]
    head = ["Lin", "Dec", "Dec", "Dec", "Dec", "Dec", "Dec\n(from Dec)"]
    warm = ["–", "–", "–", "–", "–", "–", "Dec run, half-way\nstep; fresh LoRA"]
    steps = ["8000 / 24000 / –", "8000 / 24000 / 1600", "8000 / 24000 / 1600", "8000 / 24000 / –", "8000 / 24000 / –",
             "8000 / 24000 / 1600", "4000 + 4000 /\n12000 + 12000 / –"]
    text_cols = [(h, v, w) for (h, w), v in zip(text_cols_spec, [head, warm, steps])]
    y_top = H - top_h
    draw_matrix(ax, 0.02, y_top, rows, text_cols, cell_w, cell_h, lab_w, None, "Method")
    columns = [
        [("Late blocks per backbone", "TAP-CT: blocks 9–12 of 12. SPECTRE: blocks 17–24 of 24 (combiner excluded). VoCo: stage 4 of 4 (2 of 8 blocks)."),
         ("LoRA target per backbone",
          "Fused qkv: TAP-CT, VoCo; q and kv: SPECTRE. r 16, α 32, dropout 0.1, scale 2. SPECTRE all-block LoRA / DoRA / Full FT also "
          "wrap the unused feature combiner (it receives no gradient).")],
        [("Warm start (Dec → LoRA)",
          "head.* copied from the Dec run's periodic checkpoint at half its steps (4000 of 8000; 12000 of 24000); all-block LoRA "
          "starts fresh (B = 0); new optimizer; the second stage trains the other half."),
         ("Steps", "8000 per run; 24000 for the 15-structure joint models (AMOS22, TotalSegmentator); 1600 for the compute-matched "
                   "per-structure runs (Dec, LoRA, Full FT of the 15-structure datasets only).")],
        [("Learning rate / weight decay",
          "Lin 1e-2 / 0; Dec 1e-3 / 1e-4; LoRA, DoRA 1e-4 / 1e-4; Full FT backbone 2e-5, head 1e-3, wd 1e-4."),
         ("Batch", "2 crops × 2 accumulation = 4 crops per step. DoRA as in classification (m trained with A, B).")],
    ]
    notes_panel(ax, 0.02, bot, W - 0.04, panel_h, columns)
    legend(ax, 0.02, 0.04, ncol=3)
    style.save(fig, OUT, "fig_methods_seg")


# ============================================================================= Fig 5: per-task vs joint, selection
def fig_modes():
    W = TEXTWIDTH_IN
    H = 2.62
    fig, ax = canvas(W, H)
    top = H - 0.30
    # ---- per-task (left)
    lx, lw_ = 0.05, 2.75
    ax.text(lx, top + 0.04, "Per-task: one model per label (K runs)", fontsize=FS_H, fontweight="bold", va="bottom", color=INK)
    bw, bh = 0.62, 0.30
    y = top - bh
    for k, lbl in enumerate(["label 1", "label 2", "label K"]):
        yy = y - k * (bh + 0.09)
        if k == 2:
            ax.text(lx + 0.33, yy + bh + 0.045, "⋮", fontsize=7, ha="center", va="center", color=INK2)
        box(ax, lx, yy, bw, bh, "encoder", fc=FROZEN, ha="center")
        ax.add_patch(Rectangle((lx, yy), bw, 0.06, fc=LOWRANK, ec="none"))
        arrow(ax, lx + bw, yy + bh / 2, lx + bw + 0.14, yy + bh / 2)
        box(ax, lx + bw + 0.14, yy, 0.46, bh, "head", fc=FULL, ha="center")
        arrow(ax, lx + bw + 0.60, yy + bh / 2, lx + bw + 0.74, yy + bh / 2)
        box(ax, lx + bw + 0.74, yy + 0.04, 0.70, bh - 0.08, f"1 output: {lbl}", fc=PAPER, ha="center", lw=0.4)
        box(ax, lx + bw + 1.52, yy + 0.04, 0.58, bh - 0.08, "8000 steps", fc=PAPER, ha="center", lw=0.4, ls="--")
    y_bottom_left = y - 2 * (bh + 0.09)
    ax.text(lx, y_bottom_left - 0.06, "Budget: K × 8000 steps in total, 8000 per label.", fontsize=FS, va="top", color=INK)
    # ---- joint (right)
    rx = 3.10
    ax.text(rx, top + 0.04, "Joint: one shared model, K outputs (1 run)", fontsize=FS_H, fontweight="bold", va="bottom", color=INK)
    jh = 3 * bh + 2 * 0.09
    box(ax, rx, top - jh, bw, jh, "encoder\n+ adapters", fc=FROZEN, ha="center")
    ax.add_patch(Rectangle((rx, top - jh), bw, 0.06, fc=LOWRANK, ec="none"))
    arrow(ax, rx + bw, top - jh / 2, rx + bw + 0.14, top - jh / 2)
    box(ax, rx + bw + 0.14, top - jh, 0.46, jh, "head", fc=FULL, ha="center")
    for k, lbl in enumerate(["label 1", "label 2", "label K"]):
        yy = y - k * (bh + 0.09)
        arrow(ax, rx + bw + 0.60, yy + bh / 2, rx + bw + 0.74, yy + bh / 2)
        box(ax, rx + bw + 0.74, yy + 0.04, 0.70, bh - 0.08, f"output {k + 1 if k < 2 else 'K'}: {lbl}", fc=PAPER, ha="center", lw=0.4)
        if k == 2:
            ax.text(rx + bw + 1.09, yy + bh + 0.045, "⋮", fontsize=7, ha="center", va="center", color=INK2)
    box(ax, rx + bw + 1.52, top - jh, 0.58, jh, "8000 steps shared by K labels (1/K per label); 24000 for the 15-structure segmentation joint models",
        fc=PAPER, ha="center", lw=0.4, ls="--")
    ax.text(rx, y_bottom_left - 0.06, "Budget: 8000 steps in total; same tokens feed every output.", fontsize=FS, va="top", color=INK)
    # ---- losses and selection (bottom)
    yb = 0.04
    bh2 = 0.80
    box(ax, 0.05, yb, 2.30, bh2, fc="#fafafa", lw=0.4, title="Classification loss (masked, positive-weighted BCE)",
        text="L = (1/B) Σ_b Σ_t m_bt · BCE(z_bt, y_bt; w_t): summed over the known labels of each case (m_bt = 1 if label t is known), "
             "mean over the batch (B = 1). w_t = clip(neg_t / pos_t, 0.05, 20) on the training manifest. "
             "Per-task: T = 1; joint: T = K.")
    box(ax, 2.42, yb, 1.76, bh2, fc="#fafafa", lw=0.4, title="Segmentation loss (Dice + BCE)",
        text="Per case b and structure c: Dice_bc (soft, smooth 1) + BCE_bc (mean over the crop's voxels), sigmoid per structure; "
             "L = mean over all (b, c) of the batch. Per-task: C = 1; joint: C = K structures, each a separate sigmoid channel.")
    box(ax, 4.25, yb, W - 4.30, bh2, fc="#fafafa", lw=0.4, title="Model selection (both modes)",
        text="Validate every 1000 steps (8 times per 8000-step run) on the selection split; keep the checkpoint with the best "
             "validation macro AUROC (classification) or mean Dice over structures (segmentation), strict improvement only; "
             "report the held-out metric at that checkpoint.")
    legend(ax, 0.05, yb + bh2 + 0.05, ncol=3)
    style.save(fig, OUT, "fig_modes")


if __name__ == "__main__":
    style.setup()
    # the canvas fills the figure exactly, so save at the nominal size (no tight bbox, no padding)
    plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})
    fig_pipeline_cls()
    fig_pipeline_seg()
    fig_methods_cls()
    fig_methods_seg()
    fig_modes()
    print("written to", OUT)
