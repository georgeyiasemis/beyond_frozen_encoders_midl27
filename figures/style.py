"""Shared figure style for the paper (import this in every figure script).

Colour-blind-safe Okabe-Ito palette; one colour and one marker per backbone, one colour per method family, one
colour per method where methods are distinguished; LM Roman 12 (the paper's font, fonts/), 7 pt base; figures sized
for the MIDL text width.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# VoCo is quarantined (2026-11-02, ../quarantine_voco_20261102/): every figure is drawn without the VoCo backbone. The VoCo branches stay
# for WITH_VOCO=1, which needs the VoCo runs restored in report/run_inventory.csv and metrics/out recomputed with them.
# Outputs: figures/out/ and ../paper/{figures,tables}/ unless PAPER_DIR / FIG_OUT say otherwise.
import os as _os
from pathlib import Path as _Path
NO_VOCO = _os.environ.get("WITH_VOCO", "") in ("", "0")
_V2 = _Path(__file__).resolve().parents[1]
PAPER_DIR = _Path(_os.environ.get("PAPER_DIR") or (_V2 / "paper"))
FIG_OUT = _Path(_os.environ.get("FIG_OUT") or (_Path(__file__).resolve().parent / "out"))

# LM Roman 12 (GUST Font License) ships in fonts/, so that every machine draws the same text. Glyphs it lacks (a few
# Greek letters and relations in plain text) fall back to DejaVu Serif; math ($...$) uses Computer Modern.
# Matplotlib embeds CFF-outline (.otf) fonts in a form that PDF checkers flag, so each .otf is converted once to
# TrueType outlines in fonts/_ttf/ (not in git; the original files stay unchanged) and the converted files are used.
from matplotlib import font_manager as _fm


def _otf_to_ttf(src: _Path, dst: _Path) -> None:
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    f = TTFont(str(src))
    order, gs = f.getGlyphOrder(), f.getGlyphSet()
    glyf = newTable("glyf"); glyf.glyphOrder, glyf.glyphs = order, {}
    for g in order:
        pen = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf[g] = pen.glyph()
    f["loca"], f["glyf"] = newTable("loca"), glyf
    maxp = newTable("maxp"); maxp.tableVersion = 0x00010000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions", "maxComponentElements", "maxComponentDepth"):
        setattr(maxp, k, 1 if k == "maxZones" else 0)
    f["maxp"] = maxp
    for t in ("CFF ", "VORG"):
        if t in f:
            del f[t]
    post = f["post"]; post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    f.sfntVersion = "\x00\x01\x00\x00"
    dst.parent.mkdir(exist_ok=True)
    f.save(str(dst))


for _f in sorted((_Path(__file__).resolve().parent / "fonts").glob("lmroman12-*.otf")):
    _t = _f.parent / "_ttf" / (_f.stem + ".ttf")
    if not _t.exists() or _t.stat().st_mtime < _f.stat().st_mtime:
        _otf_to_ttf(_f, _t)
    _fm.fontManager.addfont(str(_t))
FONT = ["Latin Modern Roman", "DejaVu Serif"]
FONT_RC = {"font.family": FONT, "mathtext.fontset": "cm", "axes.unicode_minus": True}

TEXTWIDTH_IN = 6.0   # MIDL (jmlr) text width is ~ 6.0 in; figures are saved at this width or at half of it
HALFWIDTH_IN = 2.95

# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}

BACKBONES = ["tapct", "spectre", "pillar0", "voco"]
BACKBONE_TEX = {"tapct": "TAP-CT", "spectre": "SPECTRE", "pillar0": "Pillar-0", "voco": "VoCo"}
BACKBONE_COL = {"tapct": OI["blue"], "spectre": OI["vermillion"], "pillar0": OI["green"], "voco": OI["purple"]}
BACKBONE_MARK = {"tapct": "o", "spectre": "s", "pillar0": "^", "voco": "D"}

# method families: what is trained
FAMILY_COL = {"frozen": "#bdbdbd", "lowrank": OI["skyblue"], "fullrank": OI["orange"]}
CLS_METHODS = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
METHOD_TEX = {"lin": "Lin", "attn": "Attn", "lora": "LoRA", "lorag": r"LoRA$_{\mathrm{late}}$", "babmil": "LoRA+ABMIL",
              "dora": "DoRA", "rslora": "rsLoRA", "lin_then_lorag": r"Lin$\rightarrow$LoRA",
              "attn_then_lorag": r"Attn$\rightarrow$LoRA", "attnft": "Attn-only FT", "fullft": "Full FT",
              "dec": "Dec", "dec_then_lora": r"Dec$\rightarrow$LoRA"}
METHOD_FAMILY = {"lin": "frozen", "attn": "frozen", "dec": "frozen", "lora": "lowrank", "lorag": "lowrank", "babmil": "lowrank",
                 "dora": "lowrank", "rslora": "lowrank", "lin_then_lorag": "lowrank", "attn_then_lorag": "lowrank",
                 "dec_then_lora": "lowrank", "attnft": "fullrank", "fullft": "fullrank"}
# per-method colours where a figure colours methods: greys and purples only, so that blue, orange/vermillion and green
# always mean a backbone, and no yellow (poor contrast on white). Methods that share a figure also differ in marker
# (METHOD_MARK). LoRA+ABMIL (classification) and DoRA (segmentation) never share a figure and share a colour.
METHOD_COL = {"lin": "#999999", "attn": "#c8c8c8", "dec": "#000000",
              "lora": "#5E3C99", "lorag": "#9E9AC8", "babmil": "#B2ABD2", "dora": "#B2ABD2", "rslora": "#9E9AC8",
              "lin_then_lorag": "#9E9AC8", "attn_then_lorag": "#9E9AC8", "dec_then_lora": "#9E9AC8",
              "attnft": "#CC79A7", "fullft": "#882255"}
METHOD_MARK = {"lin": "o", "attn": "o", "dec": "s", "lora": "o", "babmil": "D", "dora": "v", "attnft": "P", "fullft": "^"}
CONTROL_COL = "#8C510A"   # brown: negative-control lines (not a backbone or method colour)
COHORT_TEX = {"amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2022": "RSNA22 C-spine", "rsna2023": "RSNA23 trauma",
              "rsna2019_ich": "RSNA19 ICH", "amos22": "AMOS22", "kits23": "KiTS23", "lits": "LiTS",
              "msd_pancreas": "MSD Pancreas", "totalseg": "TotalSegmentator"}
INK, INK2, GRID = "#222222", "#555555", "#e6e6e6"

# cohorts / datasets: a muted blend of the Okabe-Ito hues (65 % colour + 35 % white) plus one marker each, so that
# cohort colours stay distinguishable from the saturated backbone colours
def _mute(hex_, f=0.65):
    r, g, b = (int(hex_[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(f * c + (1 - f) * 255)) for c in (r, g, b))
COHORT_COL = {"amos_covid_luna": _mute(OI["blue"]), "fdg": _mute(OI["orange"]), "rsna2019_ich": _mute(OI["green"]),
              "rsna2022": _mute(OI["purple"]), "rsna2023": _mute(OI["vermillion"]),
              "amos22": _mute(OI["blue"]), "kits23": _mute(OI["orange"]), "lits": _mute(OI["green"]),
              "msd_pancreas": _mute(OI["purple"]), "totalseg": _mute(OI["vermillion"])}
COHORT_MARK = {"amos_covid_luna": "o", "fdg": "s", "rsna2019_ich": "^", "rsna2022": "D", "rsna2023": "v",
               "amos22": "o", "kits23": "s", "lits": "^", "msd_pancreas": "D", "totalseg": "v"}


def setup() -> None:
    plt.rcParams.update({
        **FONT_RC, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.prop_cycle": matplotlib.cycler(color=[OI[k] for k in ("blue", "vermillion", "green", "purple", "orange", "skyblue", "yellow", "grey")]),
    })


def save(fig, out_dir, name: str) -> None:
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for msg in text_box_overflows(fig):
        print(f"TEXT-BOX OVERFLOW {name}: {msg}")
    fig.savefig(out_dir / f"{name}.pdf"); fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def text_box_overflows(fig, tol_pt: float = 0.3) -> list[str]:
    """Texts drawn inside a box (FancyBboxPatch / Rectangle of the same axes whose extent contains the text anchor)
    whose rendered extent leaves that box; overflow reported in printed pt (figure scaled to TEXTWIDTH_IN)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72 * fig.get_size_inches()[0] / TEXTWIDTH_IN   # display px per printed pt
    out = []
    for ax in fig.axes:
        def is_box(p):   # a drawn box: visible fill or visible edge (drop-shadows and empty frames excluded)
            if not isinstance(p, (FancyBboxPatch, Rectangle)) or not p.get_visible():
                return False
            fill = p.get_fill() and p.get_facecolor()[3] >= 0.2
            edge = p.get_edgecolor()[3] > 0 and p.get_linewidth() > 0
            return fill or edge
        boxes = [(p, p.get_window_extent(r)) for p in ax.patches if is_box(p)]
        for t in ax.texts:
            if not t.get_visible() or not t.get_text().strip():
                continue
            x, y = t.get_transform().transform(t.get_position())
            inside = [(p, bb) for p, bb in boxes if bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1]
            if not inside:
                continue
            p, bb = min(inside, key=lambda pb: pb[1].width * pb[1].height)   # innermost box
            tb = t.get_window_extent(r)
            over = max(bb.x0 - tb.x0, tb.x1 - bb.x1, bb.y0 - tb.y0, tb.y1 - bb.y1) / px_per_pt
            if over > tol_pt:
                side = [n for n, v in (("left", bb.x0 - tb.x0), ("right", tb.x1 - bb.x1), ("bottom", bb.y0 - tb.y0),
                                       ("top", tb.y1 - bb.y1)) if v / px_per_pt > tol_pt]
                out.append(f"{t.get_text()!r} leaves its box by {over:.1f} pt ({', '.join(side)})")
    return out
