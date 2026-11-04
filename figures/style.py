"""Shared figure style for the paper (import this in every figure script).

Colour-blind-safe Okabe-Ito palette; one colour and one marker per backbone, one colour per method family, one
colour per method where methods are distinguished; DejaVu Sans, 7 pt base; figures sized for the MIDL text width.
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
# per-method colours when methods are distinguished on one axis (families stay recognisable by hue)
METHOD_COL = {"lin": "#7f7f7f", "attn": "#bdbdbd", "dec": "#7f7f7f",
              "lora": OI["blue"], "lorag": OI["skyblue"], "babmil": "#1f5fa0", "dora": OI["purple"], "rslora": "#8b5fa8",
              "lin_then_lorag": OI["green"], "attn_then_lorag": "#66c2a4", "dec_then_lora": OI["green"],
              "attnft": OI["yellow"], "fullft": OI["orange"]}
COHORT_TEX = {"amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2022": "RSNA 2022 C-spine", "rsna2023": "RSNA 2023 trauma",
              "rsna2019_ich": "RSNA 2019 hemorrhage", "amos22": "AMOS22", "kits23": "KiTS23", "lits": "LiTS",
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
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.prop_cycle": matplotlib.cycler(color=[OI[k] for k in ("blue", "vermillion", "green", "purple", "orange", "skyblue", "yellow", "grey")]),
    })


def save(fig, out_dir, name: str) -> None:
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf"); fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
