#!/usr/bin/env python3
"""Native draw.io versions of the method diagrams: Figures 2, 6 and 7 (fig_pipeline, fig_arms, fig_arms_seg).

Built cell by cell, not converted from the matplotlib drawing, so that they are easy to edit in draw.io: every label
sits inside its shape, arrows are attached to the shapes they connect, each slice stack is a single image, and all
positions are whole pixels. Colours, method tables and every number come from diagrams.py (the same data as the
paper figures); the font is LM Roman 12 (fonts/lmroman12-*.otf; install it to see it in draw.io).

Canvas 700 px wide (placed at the 6.0 in text width: 12 px = 7.4 pt printed).

    python drawio_build.py   -> drawio/<name>.drawio and drawio/<name>_preview.png (a redraw of the cells, for checking)
"""
from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from PIL import Image, ImageDraw

import diagrams as D  # noqa: E402  (registers the LM Roman fonts through style.py)
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "drawio"
W = 700
FONT = "LM Roman 12"
FS, FS_T = 12, 14                       # body text, panel titles (px)
BLACK, WHITE = "#000000", "#ffffff"
NAVY = D.NAVY
FROZEN = (D.FROZEN_FC, D.FROZEN_EC)
TRAIN = (D.TRAIN_FC, D.TRAIN_EC)
ADAPT = (D.ADP_FC, D.ORANGE)
CARD = (WHITE, D.CARD_EC)


# ================================================================== document
class Doc:
    def __init__(self, name: str, h: int):
        self.name, self.h = name, h
        self.cells: list[str] = []
        self.geo: dict[str, tuple] = {}
        self.prev: list[tuple] = []
        self.n = 1

    def _id(self) -> str:
        self.n += 1
        return f"{self.name}-{self.n}"

    def vertex(self, x, y, w, h, style, value="", prev=None) -> str:
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
                          f'<mxGeometry x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" as="geometry"/></mxCell>')
        self.geo[i] = (x, y, w, h)
        if prev:
            self.prev.append(prev)
        return i

    # ---- shapes
    def rect(self, x, y, w, h, fill, stroke, label="", fs=FS, bold=False, italic=False, lw=1.0, align="center",
             valign="middle", color=BLACK) -> str:
        st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={lw:g};"
              f"fontFamily={FONT};fontSize={fs};fontColor={color};fontStyle={int(bold) + 2 * int(italic)};"
              f"align={align};verticalAlign={valign};spacing=2;")
        return self.vertex(x, y, w, h, st, label, ("rect", x, y, w, h, fill, stroke, lw, label, fs, bold, italic, align, valign, color))

    def text(self, x, y, w, h, label, fs=FS, align="center", valign="top", bold=False, italic=False, color=BLACK) -> str:
        st = (f"text;html=1;whiteSpace=nowrap;fillColor=none;strokeColor=none;fontFamily={FONT};fontSize={fs};"
              f"fontColor={color};fontStyle={int(bold) + 2 * int(italic)};align={align};verticalAlign={valign};spacing=0;")
        return self.vertex(x, y, w, h, st, label, ("rect", x, y, w, h, "none", "none", 0, label, fs, bold, italic, align, valign, color))

    def title(self, x, y, label, w=690) -> str:
        return self.text(x, y, w, 18, label, fs=FS_T, align="left", valign="middle", bold=True)

    def circle(self, cx, cy, r, fill, stroke="none", plus=False, lw=1.0) -> str:
        shape = "shape=orEllipse;perimeter=ellipsePerimeter;" if plus else "ellipse;"
        return self.vertex(cx - r, cy - r, 2 * r, 2 * r, f"{shape}html=1;fillColor={fill};strokeColor={stroke};strokeWidth={lw:g};",
                           "", ("circle", cx, cy, r, fill, stroke, plus, lw))

    def cube(self, x, y, w, h, depth, fill, stroke) -> str:
        """Front face w x h at (x, y + depth*0.75), depth drawn up and to the right; (x, y) is the top-left of the bounding box."""
        return self.vertex(x, y, w + depth, h + 0.75 * depth,
                           f"shape=cube;whiteSpace=wrap;html=1;flipH=1;size={depth:g};darkOpacity=0.06;darkOpacity2=0.14;"
                           f"fillColor={fill};strokeColor={stroke};strokeWidth=1;",
                           "", ("cube", x, y, w, h, depth, fill, stroke))

    def image(self, x, y, png: bytes, scale: float) -> str:
        im = Image.open(io.BytesIO(png))
        w, h = im.width / scale, im.height / scale
        b64 = base64.b64encode(png).decode()
        return self.vertex(x, y, round(w, 1), round(h, 1),
                           f"shape=image;html=1;imageAspect=0;aspect=fixed;image=data:image/png,{b64};", "",
                           ("image", x, y, w, h, np.asarray(im.convert("RGBA"))))

    # ---- connectors
    def edge(self, src, tgt, exit_, entry, pts=(), arrow=True, color=BLACK, lw=1.2) -> str:
        """Arrow attached to two shapes; exit_ / entry are (fx, fy) fractions of the source / target box."""
        i = self._id()
        st = (f"edgeStyle=none;html=1;rounded=0;endArrow={'block' if arrow else 'none'};endFill=1;endSize=5;"
              f"strokeColor={color};strokeWidth={lw:g};exitX={exit_[0]:g};exitY={exit_[1]:g};exitDx=0;exitDy=0;"
              f"entryX={entry[0]:g};entryY={entry[1]:g};entryDx=0;entryDy=0;")
        arr = ("<Array as=\"points\">" + "".join(f'<mxPoint x="{px:g}" y="{py:g}"/>' for px, py in pts) + "</Array>") if pts else ""
        self.cells.append(f'<mxCell id="{i}" style={quoteattr(st)} edge="1" parent="1" source="{src}" target="{tgt}">'
                          f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
        sx, sy, sw, sh = self.geo[src]; tx, ty, tw, th = self.geo[tgt]
        p0 = (sx + exit_[0] * sw, sy + exit_[1] * sh); p1 = (tx + entry[0] * tw, ty + entry[1] * th)
        self.prev.append(("line", [p0, *pts, p1], color, lw, arrow))
        return i

    def line(self, pts, arrow=False, color=BLACK, lw=1.0) -> str:
        """Free line (not attached): separators, rails, arrows into or out of a group of shapes."""
        i = self._id()
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        mid = "".join(f'<mxPoint x="{px:g}" y="{py:g}"/>' for px, py in pts[1:-1])
        st = (f"edgeStyle=none;html=1;rounded=0;endArrow={'block' if arrow else 'none'};endFill=1;endSize=5;"
              f"strokeColor={color};strokeWidth={lw:g};")
        self.cells.append(f'<mxCell id="{i}" style={quoteattr(st)} edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{x0:g}" y="{y0:g}" as="sourcePoint"/><mxPoint x="{x1:g}" y="{y1:g}" as="targetPoint"/>'
                          + (f'<Array as="points">{mid}</Array>' if mid else "") + '</mxGeometry></mxCell>')
        self.prev.append(("line", list(pts), color, lw, arrow))
        return i

    def arrow(self, x0, y, x1, lw=1.2) -> str:
        return self.line([(x0, y), (x1, y)], arrow=True, lw=lw)

    # ---- output
    def xml(self) -> str:
        return (f'<mxfile host="drawio_build.py"><diagram name="{self.name}" id="{self.name}"><mxGraphModel grid="1" gridSize="5" '
                f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" '
                f'pageHeight="{self.h}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                + "".join(self.cells) + "</root></mxGraphModel></diagram></mxfile>")


# ================================================================== slice-stack images
SC = 4                                     # image pixels per canvas px


def _rgb(sl, f=1.0, overlay=None, alpha=0.55):
    g = np.clip(sl * f, 0, 1)
    rgb = np.repeat(g[..., None], 3, axis=2)
    if overlay is not None:
        cols = D._label_colors()[overlay][..., :3]
        a = np.where(overlay > 0, alpha, 0.0)[..., None]
        rgb = rgb * (1 - a) + cols * a
    return (rgb * 255).astype(np.uint8)


def _tile(sl, size, f=1.0, overlay=None, frame=NAVY, frame_w=1.0, grid=None, centre=None):
    """One slice padded to a square, resized to size x size canvas px (at SC), framed; optional white grid / centre box."""
    a = _rgb(sl, f, overlay)
    h, w = a.shape[:2]
    m = max(h, w)
    sq = np.zeros((m, m, 3), np.uint8); sq[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = a
    S = int(round(size * SC))
    im = Image.fromarray(sq).resize((S, S), Image.BILINEAR).convert("RGBA")
    d = ImageDraw.Draw(im)
    if grid:
        for k in range(1, grid):
            p = round(k * S / grid)
            d.line([(p, 0), (p, S)], fill=WHITE, width=SC); d.line([(0, p), (S, p)], fill=WHITE, width=SC)
    if centre:
        q0, q1 = round(S * centre[0]), round(S * centre[1])
        d.rectangle([q0, q0, q1, q1], outline=WHITE, width=int(2.2 * SC))
    fw = max(1, int(round(frame_w * SC)))
    d.rectangle([0, 0, S - 1, S - 1], outline=frame, width=fw)
    return im


def stack_png(slices, size=46, step=3.5, mark=None, overlays=None, front_grid=None, front_centre=None) -> bytes:
    """Real slices drawn back to front, up and to the right; mark=(i0, i1) frames a run of slices (one window) in brown."""
    n = len(slices)
    L = int(round((size + step * (n - 1)) * SC))
    canvas = Image.new("RGBA", (L, L), (0, 0, 0, 0))
    for k in range(n):
        marked = mark is not None and mark[0] <= k <= mark[1]
        last = k == n - 1
        t = _tile(slices[k], size, 0.6 + 0.4 * (k + 1) / n, overlays[k] if overlays is not None else None,
                  frame=D.WIN_C if marked else NAVY, frame_w=1.6 if marked else 1.0,
                  grid=front_grid if last else None, centre=front_centre if last else None)
        canvas.alpha_composite(t, (int(round(step * (n - 1 - k) * SC)), int(round(step * k * SC))))
    buf = io.BytesIO(); canvas.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def slabs_png(fronts, size=46, n_slab=3, depth=4, step=1.2, gap=7.0, grid=7) -> bytes:
    """One window cut into n_slab depth slabs of `depth` slices; each slab's front slice carries the patch grid."""
    n = n_slab * depth
    off = lambda i: (n_slab - 1 - i // depth) * gap + (n - 1 - i) * step
    L = int(round((size + off(0)) * SC))
    canvas = Image.new("RGBA", (L, L), (0, 0, 0, 0))
    for i in range(n):
        t = _tile(fronts[i // depth], size, 0.6 + 0.4 * (i + 1) / n, frame=D.WIN_C, frame_w=1.0,
                  grid=grid if i % depth == depth - 1 else None)
        o = off(i)
        canvas.alpha_composite(t, (int(round(o * SC)), int(round((off(0) - o) * SC))))
    buf = io.BytesIO(); canvas.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def img_size(png: bytes) -> tuple[float, float]:
    im = Image.open(io.BytesIO(png))
    return im.width / SC, im.height / SC


# ================================================================== reusable pieces
def tokens(doc, x, y, kinds, s=10):
    """A row of tokens (left to right); kind 'dots' leaves a gap with an ellipsis."""
    col = {"cls": D.CLS_C, "reg": D.REG_C, "patch": D.PATCH_C, "patch2": D.PATCH2_C}
    for k, kind in enumerate(kinds):
        if kind == "dots":
            doc.text(x + k * s, y - 4, s, s + 4, "…", fs=FS, align="center", valign="middle")
        else:
            doc.rect(x + k * s, y, s, s, col[kind], WHITE, lw=1)
    return x + len(kinds) * s


def encoder_box(doc, x, y, w, h, title, n_lines):
    """Frozen encoder: a box with its name on top and n_lines thin block lines."""
    b = doc.rect(x, y, w, h, *FROZEN, label=title, valign="top")
    for k in range(n_lines):
        lx = x + 8 + k * (w - 16) / max(n_lines - 1, 1)
        doc.line([(round(lx, 1), y + 18), (round(lx, 1), y + h - 6)], color=FROZEN[1], lw=1)
    return b


def mini_block(doc, x, top, w, qkv="f", proj="f", mlp="f", norm="f", lora=False):
    """One pre-norm transformer block, read bottom to top: LN, W_qkv, attention, W_proj, add, LN, MLP, add."""
    st = {"f": FROZEN, "t": TRAIN}
    CH, GAP, ADD, ATT = 17, 4, 14, 14
    bx, bw = x + 20, 52
    rows = [("LN", norm, CH, GAP), ("<i>W</i><sub>qkv</sub>", qkv, CH, 3), ("attention", None, ATT, 3),
            ("<i>W</i><sub>proj</sub>", proj, CH, ADD), ("LN", norm, CH, GAP), ("MLP", mlp, CH, ADD)]
    h = 6 + sum(r[2] + r[3] for r in rows)
    doc.rect(x, top, w, h, *CARD)
    rail = x + 9
    doc.line([(rail, top + 4), (rail, top + h - 4)], color=NAVY, lw=1)
    yy = top + h - 6
    centres = {}
    for k, (name, s, hh, gap) in enumerate(rows):
        yb = yy - hh
        if s is None:
            doc.text(bx, yb, bw, hh, name, italic=True, valign="middle")
        else:
            fc, ec = st[s]
            centres[k] = doc.rect(bx, yb, bw, hh, fc, ec, label=name, lw=1.5 if s == "t" else 1)
        yy = yb - gap
        if gap == ADD:
            doc.circle(rail, yb - gap / 2, 5, WHITE, BLACK, plus=True, lw=1)
    if lora:
        q = centres[1]
        qx, qy, qw, qh = doc.geo[q]
        a = doc.rect(qx + qw + 10, qy, 18, qh, *ADAPT, label="A", lw=1.5)
        b = doc.rect(qx + qw + 34, qy, 18, qh, *ADAPT, label="B", lw=1.5)
        doc.edge(q, a, (1, 0.5), (0, 0.5), arrow=False, color=ADAPT[1], lw=1)
        doc.edge(a, b, (1, 0.5), (0, 0.5), arrow=False, color=ADAPT[1], lw=1)
        doc.text(qx + qw + 10, qy - 15, 42, 14, "<i>r</i> = 16", color=ADAPT[1], valign="middle")
    return top + h


def legend(doc, x, y, items):
    for fill, stroke, lab, wtxt in items:
        doc.rect(x, y + 2, 16, 11, fill, stroke, lw=1.5 if stroke != D.FROZEN_EC else 1)
        doc.text(x + 21, y, wtxt, 15, lab, align="left", valign="middle")
        x += 21 + wtxt + 16


def method_table(doc, x0, top, rows, cols, groups, row_h=21, name_w=112):
    """rows: (name html, sub html, {key: state or text}); cols: (key, label html, width, kind 'cell' | 'text')."""
    pos, x = {}, x0 + name_w
    for key, lab, w, kind in cols:
        pos[key] = (x, w, kind)
        doc.text(x, top + 18, w, 30, lab, valign="bottom")
        x += w + 4
    right = x - 4
    for lab, k0, k1 in groups:
        xa, xb = pos[k0][0], pos[k1][0] + pos[k1][1]
        doc.text(xa - 20, top - 2, xb - xa + 40, 16, lab, bold=True, valign="middle")
        doc.line([(xa, top + 16), (xb, top + 16)], lw=1)
    hy = top + 51
    doc.line([(x0, hy), (right, hy)], lw=1)
    for i, (name, sub, vals) in enumerate(rows):
        y = hy + 2 + i * row_h
        if i % 2 == 0:
            doc.vertex(x0, y, right - x0, row_h, f"rounded=0;html=1;fillColor={D.BAND_C};strokeColor=none;", "",
                       ("rect", x0, y, right - x0, row_h, D.BAND_C, "none", 0, "", FS, False, False, "center", "middle", BLACK))
        doc.text(x0 + 4, y, name_w - 8, row_h, name, bold=True, align="left", valign="middle")
        if sub:
            doc.text(x0 + name_w - 50, y, 46, row_h, sub, italic=True, align="right", valign="middle")
        for key, (cx, w, kind) in pos.items():
            v = vals.get(key, "")
            if kind == "cell":
                st, mark = v[0], v[1:]
                fc, ec = {"f": FROZEN, "a": ADAPT, "t": TRAIN}[st]
                doc.rect(cx + 2, y + 3, w - 4, row_h - 6, fc, ec, label=mark, lw=1.5 if st in "at" else 1)
            else:
                doc.text(cx, y, w, row_h, v, valign="middle")
    return hy + 2 + len(rows) * row_h


CELL_COLS = [("pe", "patch<br>emb.", 30, "cell"),
             ("eq", "<i>W</i><sub>qkv</sub>", 30, "cell"), ("ep", "<i>W</i><sub>proj</sub>", 30, "cell"), ("em", "MLP,<br>norms", 30, "cell"),
             ("lq", "<i>W</i><sub>qkv</sub>", 30, "cell"), ("lp", "<i>W</i><sub>proj</sub>", 30, "cell"), ("lm", "MLP,<br>norms", 30, "cell")]
NAME = {**D.NAME, "lorag": "LoRA<sub>late</sub>"}
SUB = {"dora": "(+ <i>m</i>)", "rslora": "(<i>α</i>/√<i>r</i>)"}


# ================================================================== Figure 2: pipeline (TAP-CT)
def fig_pipeline(vol, vpred, seg) -> Doc:
    doc = Doc("fig_pipeline", 580)
    def lab(xc, y, w, s, align="center"):
        x = min(max(xc - w / 2, 2), W - 2 - w) if align == "center" else xc
        return doc.text(x, y, w, 30, s, align=align)

    # (a) classification: CT volume -> windows -> patches -> ViT-B -> W CLS tokens -> head
    doc.title(4, 2, "(a) Classification")
    yc = 62
    s1 = stack_png(vol); w1, h1 = img_size(s1)
    doc.image(4, yc - h1 / 2, s1, SC); lab(4, yc + 40, 110, "CT volume,<br>224² in-plane", align="left")
    x = 4 + w1
    doc.arrow(x + 6, yc, x + 26); x += 32
    s2 = stack_png(vol, mark=(2, 4)); doc.image(x, yc - h1 / 2, s2, SC); lab(x + w1 / 2, yc + 40, 110, "<i>W</i> windows of 12<br>slices, stride 3")
    x += w1
    doc.arrow(x + 6, yc, x + 26); x += 32
    s3 = slabs_png(vol[2:5]); w3, h3 = img_size(s3)
    doc.image(x, yc - h3 / 2, s3, SC); lab(x + w3 / 2, yc + 40, 120, "2352 patches of<br>4×8×8 per window")
    x += w3
    doc.arrow(x + 6, yc, x + 30); x += 36
    vit = encoder_box(doc, x, yc - 28, 96, 56, "ViT-B", 12); lab(x + 48, yc + 40, 100, "12 blocks<br>(panel c)")
    x += 96
    doc.arrow(x + 6, yc, x + 30); x += 36
    for k in range(5):
        doc.rect(x, yc - 29 + k * 12, 10, 10, D.CLS_C, WHITE)
    clsc = doc.vertex(x, yc - 29, 10, 58, "rounded=0;html=1;fillColor=none;strokeColor=none;")   # anchor of the head arrows
    lab(x + 5, yc + 40, 60, "<i>W</i> CLS<br>tokens")
    hx = x + 46
    lin = doc.rect(hx, yc - 30, W - 4 - hx, 20, *TRAIN, label="Lin: mean → linear", lw=1.5)
    att = doc.rect(hx, yc - 4, W - 4 - hx, 34, *TRAIN, label="Attn: two attention<br>blocks → linear", lw=1.5)
    doc.edge(clsc, lin, (1, 0.5), (0, 0.5)); doc.edge(clsc, att, (1, 0.5), (0, 0.5))
    lab((hx + W - 4) / 2, yc + 40, 140, "head: one logit<br>per label")

    # (b) segmentation: crop -> 12 blocks, taps 3, 6, 9, 12 -> 1x1x1 convs -> decoder stages -> maps
    yt = 140
    doc.title(4, yt, "(b) Segmentation", w=200)
    dec = seg.loc[("tapct", "dec")]
    doc.text(300, yt, W - 4 - 300, 18, f"multi-depth decoder ({dec['trainable'] / 1e6:.2f} M parameters for {int(dec['classes'])} structures)",
             align="right", valign="middle")
    yc = 224
    doc.image(4, yc - h1 / 2, s2, SC); lab(4, yc + 40, 110, "crop of 12<br>whole slices", align="left")
    x = 4 + w1
    doc.arrow(x + 6, yc, x + 26); x += 32
    bw, bh, gap = 14, 64, 3
    blocks = []
    for k in range(12):
        tap = k in (2, 5, 8, 11)                              # TAPCT_DEFAULT_TAPS = (2, 5, 8, 11)
        blocks.append(doc.rect(x + k * (bw + gap), yc - bh / 2, bw, bh, *FROZEN, label=str(k + 1) if tap else "",
                               bold=True, color=TRAIN[1]))
    xe = x + 12 * bw + 11 * gap
    lab((x + xe) / 2, yc + 40, 130, "blocks 3, 6, 9, 12<br>(Lin: block 12)")
    cx0 = xe + 32
    convs = []
    for j in range(4):                                         # j = 0: block 12, lowest route, bottom conv
        convs.append(doc.rect(cx0, yc + 22 - j * 16 - 11, 22, 11, *TRAIN, lw=1.5))
    for j, k in enumerate((11, 8, 5, 2)):
        bx = x + k * (bw + gap) + bw / 2
        lev, col_x = yc - bh / 2 - 7 - 6 * j, xe + 8 + 6 * j
        cy = doc.geo[convs[j]][1] + 5.5
        doc.edge(blocks[k], convs[j], (0.5, 0), (0, 0.5), pts=[(bx, lev), (col_x, lev), (col_x, cy)], arrow=False, color=TRAIN[1], lw=1)
        doc.circle(bx, yc - bh / 2, 3, TRAIN[1])
    lab(cx0 + 11, yc + 40, 50, "1×1×1<br>conv")
    x = cx0 + 22
    doc.arrow(x + 4, yc, x + 18); x += 24
    stages = [("128", "3×28²", 128, 28, 3), ("64", "6×56²", 64, 56, 6), ("32", "12×112²", 32, 112, 12), ("16", "12×224²", 16, 224, 12)]
    for k, (c_, g_, ch, hw_, dd) in enumerate(stages):
        w_ = round(6 + 10 * np.log2(ch / 8) / 4)
        h_ = round(17 + 46 * np.log2(hw_ / 28) / 3)
        d_ = round(7 + dd * 1.4)
        doc.cube(x, yc - (h_ + 0.75 * d_) / 2, w_, h_, d_, *TRAIN)
        lab(x + (w_ + d_) / 2, yc + 40, 60, f"{c_}<br>{g_}")
        x += w_ + d_
        if k < 3:
            doc.arrow(x + 4, yc, x + 16); x += 20
    doc.arrow(x + 5, yc, x + 25); x += 30
    s4 = stack_png(vol, overlays=vpred); doc.image(x, yc - h1 / 2, s4, SC)
    lab(x + w1 / 2, yc + 40, 110, "one sigmoid map<br>per structure")

    # (c) what each method trains inside one block
    yt = 300
    doc.title(4, yt, "(c) What each method trains in every block", w=440)
    top = yt + 40
    specs = [("frozen", {}, 80, "Lin, Attn,<br>Dec"),
             ("LoRA", dict(lora=True), 124, "all LoRA variants;<br>LoRA<sub>late</sub>, Lin→LoRA,<br>Attn→LoRA: blocks 9–12"),
             ("Attn-only FT", dict(qkv="t", proj="t"), 80, "attention<br>projections<br>and biases"),
             ("Full FT", dict(qkv="t", proj="t", mlp="t", norm="t"), 80, "all, incl. patch<br>embedding")]
    x = 4
    for t, kw, w_, sub in specs:
        doc.text(x - 10, top - 18, w_ + 20, 16, t, bold=True, valign="middle")
        bot = mini_block(doc, x, top, w_, **kw)
        doc.text(x + w_ / 2 - 80, bot + 4, 160, 44, sub)
        x += w_ + 22

    # (d) per-task vs joint
    x3 = 456
    doc.title(x3, yt, "(d) Two training modes", w=240)
    doc.rect(x3, top - 18, W - 4 - x3, 168, *CARD)
    labs = D.LABEL_C

    def model(xm, ym, outs):
        doc.rect(xm, ym, 44, 18, *FROZEN)
        for q in range(4):
            doc.line([(xm + 8 + q * 9.3, ym + 4), (xm + 8 + q * 9.3, ym + 14)], color=FROZEN[1], lw=1)
        hd = doc.rect(xm + 50, ym + 3, 16, 12, *TRAIN, lw=1.5)
        for o, c in enumerate(outs):
            oy = ym + 9 + (o - (len(outs) - 1) / 2) * 9
            dot = doc.circle(xm + 82, oy, 4, c)
            doc.edge(hd, dot, (1, 0.5), (0, 0.5), arrow=False, lw=1)
    doc.text(x3 + 8, top - 12, 220, 16, "per-task: one model per label", bold=True, align="left", valign="middle")
    for q, c in enumerate(labs):
        model(x3 + 10, top + 10 + q * 24, [c])
    doc.text(x3 + 108, top + 26, 120, 30, "<i>K</i> runs; each sees<br>its own label only", align="left")
    doc.line([(x3 + 8, top + 86), (W - 12, top + 86)], color=D.CARD_EC, lw=1)
    doc.text(x3 + 8, top + 92, 220, 16, "joint: one model per cohort", bold=True, align="left", valign="middle")
    model(x3 + 10, top + 120, labs)
    doc.text(x3 + 108, top + 110, 120, 44, "1 run; encoder,<br>adapters and head<br>shared; <i>K</i> outputs", align="left")

    legend(doc, 4, 560, [(*FROZEN, "frozen", 44), (*TRAIN, "trained", 50),
                         (*ADAPT, "low-rank adapter: <i>W</i><sub>0</sub><i>x</i> + (<i>α</i>/<i>r</i>) <i>BAx</i>, <i>α</i> = 32", 280)])
    return doc


# ================================================================== Figure 6: classification with SPECTRE and Pillar-0
def fig_arms(vol, head_ct, cls) -> Doc:
    doc = Doc("fig_arms", 684)
    lab = lambda xc, y, w, s, h=58, align="center": doc.text(xc - w / 2 if align == "center" else xc, y, w, h, s, align=align)

    # (a) SPECTRE
    doc.title(4, 2, "(a) SPECTRE: crops, per-crop ViT-L, feature combiner")
    yc = 62
    s1 = stack_png(vol, front_grid=4, front_centre=(0.25, 0.75)); w1, h1 = img_size(s1)
    doc.image(4, yc - h1 / 2, s1, SC)
    lab(4, yc + 34, 110, "512²×256,<br>0.5×0.5×1 mm,<br>4³ = 64 crops<br>(train: 2³ = 8)", align="left")
    x = 4 + w1
    doc.arrow(x + 8, yc, x + 26); x += 32
    xe = tokens(doc, x, yc - 5, ["cls", "patch", "patch", "patch", "patch", "dots", "patch"])
    lab((x + xe) / 2, yc + 34, 90, "128²×64 crop:<br>16×16×8<br>patches,<br>512 + CLS")
    x = xe
    doc.arrow(x + 5, yc, x + 21); x += 26
    encoder_box(doc, x, yc - 28, 56, 56, "ViT-L", 12); lab(x + 28, yc + 34, 70, "24 blocks,<br>1080-d,<br>per crop")
    x += 56
    doc.arrow(x + 5, yc, x + 21); x += 26
    xe = tokens(doc, x, yc - 5, ["cls", "patch2"]); lab((x + xe) / 2, yc + 34, 60, "[CLS ‖<br>mean],<br>2160-d")
    x = xe
    doc.arrow(x + 5, yc, x + 21); x += 26
    encoder_box(doc, x, yc - 20, 46, 40, "", 4); lab(x + 23, yc + 34, 70, "combiner:<br>4 blocks<br>over crops")
    x += 46
    doc.arrow(x + 5, yc, x + 29); x += 34
    doc.rect(x, yc - 22, 10, 10, D.CLS_C, WHITE)
    for k in range(3):
        doc.rect(x, yc - 8 + k * 10, 10, 10, D.PATCH2_C, WHITE)
    anchor = doc.vertex(x, yc - 22, 10, 44, "rounded=0;html=1;fillColor=none;strokeColor=none;")
    lab(x + 5, yc + 34, 64, "scan CLS<br>+ <i>N</i> crop<br>tokens")
    hx = x + 36
    for t, dy in [("Lin: scan CLS → linear", -28), ("Attn: scan CLS + <i>N</i> crop tokens", 0),
                  ("LoRA+ABMIL: crop tokens → ABMIL → Attn", 28)]:
        h_ = doc.rect(hx, yc + dy - 10, W - 4 - hx, 20, *TRAIN, label=t, lw=1.5)
        doc.edge(anchor, h_, (1, 0.5), (0, 0.5))

    # (b) Pillar-0
    yt = 166
    doc.title(4, yt, "(b) Pillar-0 (head CT): multi-window input, conv stem, three-stage multi-scale encoder")
    yc = yt + 72
    chans = [np.clip((head_ct - 0.5) * g + 0.5, 0, 1) for g in (0.6, 0.8, 1.0, 1.3, 1.7, 2.2)]
    s2 = stack_png(chans); doc.image(4, yc - h1 / 2, s2, SC)
    lab(4, yc + 34, 120, "128×256×256,<br>0.5×0.5×1.25 mm;<br>11 channels (10 HU<br>windows, min–max)", align="left")
    x = 4 + w1
    doc.arrow(x + 8, yc, x + 48); x += 54
    doc.rect(x, yc - 20, 44, 40, *FROZEN, label="conv<br>stem"); lab(x + 22, yc + 34, 60, "8×8×4<br>patches,<br>32³ grid")
    x += 44
    doc.arrow(x + 6, yc, x + 26); x += 32
    gx = x
    for k, sz in enumerate([40, 26, 14]):               # schematic sizes: scale k = stem grid max-pooled by 4^k per axis
        d_ = round(sz * 0.5)
        doc.cube(gx, yc - (sz + 0.75 * d_) / 2, sz, sz, d_, D.PATCH_C if k == 0 else D.PATCH2_C, NAVY)
        doc.text(gx, yc + (sz + 0.75 * d_) / 2 + 2, sz, 14, ["32³", "8³", "2³"][k], valign="middle")
        gx += sz + d_ + 8
    lab((x + gx - 8) / 2, yc + 44, 130, "tokens at 3 scales<br>(max-pool by 4<br>per axis), attention<br>within and across")
    x = gx - 8
    doc.arrow(x + 6, yc, x + 26); x += 32
    enc = doc.rect(x, yc - 28, 92, 56, *FROZEN, label="384-d", valign="top")
    for k, (nb, w_, ox) in enumerate([(2, 16, 6), (2, 16, 26), (8, 38, 48)]):
        doc.rect(x + ox, yc - 10, w_, 32, WHITE, FROZEN[1], label=str(nb))
    lab(x + 46, yc + 34, 120, "3 stages of 2, 2, 8<br>blocks; scales 32³,<br>8³, 2³ leave after<br>stages 1, 2, 3")
    x += 92
    hx = x + 28
    for t, dy in [("Lin: per-scale max-pool,<br>L2-normalized, 1152-d → linear", -34),
                  ("Attn: tokens of all scales<br>(32³ avg-pooled to 16³)", 0),
                  ("LoRA+ABMIL: same tokens<br>→ ABMIL → Attn", 34)]:
        h_ = doc.rect(hx, yc + dy - 15, W - 4 - hx, 30, *TRAIN, label=t, lw=1.5)
        doc.edge(enc, h_, (1, 0.5), (0, 0.5))

    # (c) the eleven classification methods
    yt = 360
    doc.title(4, yt, "(c) What each classification method trains")
    bbs = ("tapct", "spectre", "pillar0")
    hd = {"lin": "pooled → linear", "babmil": "ABMIL → Attn", "lin_then_lorag": "linear"}
    order = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
    rows = []
    for m in order:
        v = D._cells(m)
        v["hd"] = hd.get(m, "attentive probe")
        v["sc"] = "4000 + 4000" if m.endswith("_then_lorag") else "8000"
        for key, bb in zip(("p1", "p2", "p3"), bbs):
            v[key] = D.fmt_m(D.cls_bb(cls, bb, m)[0])
        rows.append((NAME[m], SUB.get(m, ""), v))
    cols = CELL_COLS + [("hd", "head", 104, "text"), ("sc", "steps", 66, "text"),
                        ("p1", "TAP-CT", 48, "text"), ("p2", "SPECTRE", 52, "text"), ("p3", "Pillar-0", 50, "text")]
    bot = method_table(doc, 4, yt + 24, rows, cols, [("early blocks", "eq", "em"), ("late blocks", "lq", "lm"),
                                                     ("encoder parameters updated", "p1", "p3")])
    legend(doc, 4, bot + 10, [(*FROZEN, "frozen", 44), (*ADAPT, "rank-16 adapter (<i>m</i>: DoRA magnitude; 2: added in stage 2)", 318),
                              (*TRAIN, "trained in full", 84)])
    doc.h = bot + 30
    return doc


# ================================================================== Figure 7: segmentation decoder and methods
def fig_arms_seg(seg) -> Doc:
    doc = Doc("fig_arms_seg", 420)
    doc.title(4, 2, "(a) Segmentation decoder: tensor shapes (channels × depth × height × width)")
    stages = [("block tokens<br>(4 taps)", "768×3×28²", "1080×8³", 64, 28, 1),
              ("1×1×1 conv,<br>concatenate", "4·128×3×28²", "4·128×8³", 512, 28, 1),
              ("3³ conv", "128×3×28²", "128×8³", 128, 28, 1),
              ("upsample,<br>3³ conv", "64×6×56²", "64×16³", 64, 56, 2),
              ("upsample,<br>3³ conv", "32×12×112²", "32×32³", 32, 112, 3),
              ("upsample,<br>3³ conv", "16×12×224²", "16×64³", 16, 224, 4),
              ("upsample,<br>3³ conv", "–", "16×64×128²", 16, 224, 4),
              ("1×1×1 conv", "C×12×224²", "C×64×128²", 8, 224, 4)]
    yc, slot, x0 = 66, 76, 92
    doc.text(4, 140, 80, 18, "TAP-CT", bold=True, align="left", valign="middle")
    doc.text(4, 160, 80, 18, "SPECTRE", bold=True, align="left", valign="middle")
    prev_r = None
    for k, (nm, t1, t2, ch, hw_, dd) in enumerate(stages):
        w_ = round(6 + 10 * np.log2(max(ch, 8) / 8) / 6)
        h_ = round(18 + 42 * np.log2(hw_ / 28) / 3)
        d_ = round(7 + dd * 5)
        cx = x0 + k * slot + slot / 2
        bx = cx - (w_ + d_) / 2
        fill, stroke = (D.PATCH_C, NAVY) if k == 0 else TRAIN
        doc.cube(bx, yc - (h_ + 0.75 * d_) / 2, w_, h_, d_, fill, stroke)
        if prev_r is not None:
            doc.arrow(prev_r + 6, yc, bx - 6)
        prev_r = bx + w_ + d_
        doc.text(cx - slot / 2, 104, slot, 30, nm)
        doc.text(cx - slot / 2, 140, slot, 18, t1, valign="middle")
        doc.text(cx - slot / 2, 160, slot, 18, t2, valign="middle")

    yt = 190
    doc.title(4, yt, "(b) What each segmentation method trains")
    order = ["lin", "dec", "lora", "lorag", "dora", "dec_then_lora", "fullft"]
    rows = []
    for m in order:
        v = D._cells(m)
        v["hd"] = "1×1×1 conv" if m == "lin" else "decoder"
        v["sc"] = "4000 + 4000" if m == "dec_then_lora" else "8000"
        for key, bb in zip(("p1", "p2"), ("tapct", "spectre")):
            r = seg.loc[(bb, m)]
            v[key] = D.fmt_m(int(r["backbone_trainable"]) - int(r["no_grad_backbone"]))
        rows.append((NAME[m], SUB.get(m, ""), v))
    cols = CELL_COLS + [("hd", "head", 104, "text"), ("sc", "steps", 76, "text"), ("p1", "TAP-CT", 66, "text"), ("p2", "SPECTRE", 66, "text")]
    bot = method_table(doc, 4, yt + 24, rows, cols, [("early blocks", "eq", "em"), ("late blocks", "lq", "lm"),
                                                     ("encoder parameters updated", "p1", "p2")])
    legend(doc, 4, bot + 10, [(*FROZEN, "frozen", 44), (*ADAPT, "rank-16 adapter (<i>m</i>: DoRA magnitude; 2: added in stage 2)", 318),
                              (*TRAIN, "trained in full", 84)])
    doc.h = bot + 30
    return doc


# ================================================================== preview (redraw of the cells) + overflow check
def _mt(s: str) -> str:
    """HTML label -> matplotlib text (mathtext for sub/superscripts) for the preview."""
    s = s.replace("<br>", "\n")
    s = re.sub(r"<sub>(.*?)</sub>", lambda m: "$_{\\mathrm{" + m.group(1) + "}}$", s)
    s = re.sub(r"<sup>(.*?)</sup>", lambda m: "$^{" + m.group(1) + "}$", s)
    s = html.unescape(re.sub(r"</?[bi]>", "", s))
    return s.replace("α", r"$\alpha$")


def preview(doc: Doc, path: Path) -> list[str]:
    from matplotlib.patches import Circle, Polygon, Rectangle
    fig = plt.figure(figsize=(W / 100, doc.h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(doc.h, 0); ax.axis("off")
    pt = 72 / 100
    texts = []
    for p in doc.prev:
        k = p[0]
        if k == "rect":
            _, x, y, w, h, fc, ec, lw, label, fs, bold, italic, al, va, color = p
            ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw * pt, zorder=2))
            if label:
                tx = {"left": x + 2, "right": x + w - 2}.get(al, x + w / 2)
                ty = {"top": y + 2, "bottom": y + h - 2}.get(va, y + h / 2)
                t = ax.text(tx, ty, _mt(label), fontsize=fs * pt, ha=al, va={"middle": "center"}.get(va, va), color=color,
                            fontweight="bold" if bold else "normal", style="italic" if italic else "normal", linespacing=1.15, zorder=6)
                texts.append((t, (x, y, w, h), ec != "none", label))
        elif k == "circle":
            _, cx, cy, r, fc, ec, plus, lw = p
            ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw * pt, zorder=4))
            if plus:
                ax.plot([cx - r, cx + r], [cy, cy], color=ec, lw=lw * pt, zorder=5); ax.plot([cx, cx], [cy - r, cy + r], color=ec, lw=lw * pt, zorder=5)
        elif k == "cube":
            _, x, y, w, h, d, fc, ec = p
            dx, dy = d, 0.75 * d
            ax.add_patch(Polygon([(x, y + dy), (x + dx, y), (x + w + dx, y), (x + w, y + dy)], fc=D._mix(fc, ec, 0.15), ec=ec, lw=pt, zorder=3))
            ax.add_patch(Polygon([(x + w, y + dy), (x + w + dx, y), (x + w + dx, y + h), (x + w, y + h + dy)], fc=D._mix(fc, ec, 0.35), ec=ec, lw=pt, zorder=3))
            ax.add_patch(Rectangle((x, y + dy), w, h, fc=fc, ec=ec, lw=pt, zorder=3))
        elif k == "image":
            _, x, y, w, h, arr = p
            ax.imshow(arr, extent=(x, x + w, y + h, y), zorder=5, interpolation="bilinear")
        elif k == "line":
            _, pts, c, lw, arrow = p
            xs, ys = zip(*pts)
            if arrow:
                ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=c, lw=lw * pt, zorder=4)
                ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=c, lw=lw * pt, mutation_scale=8, shrinkA=0, shrinkB=0), zorder=4)
            else:
                ax.plot(xs, ys, color=c, lw=lw * pt, zorder=4)
    ax.set_xlim(0, W); ax.set_ylim(doc.h, 0)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    issues = []
    for t, (x, y, w, h), boxed, label in texts:
        bb = t.get_window_extent(r)
        (x0, y1), (x1, y0) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        tol = 1.0 if boxed else 3.0
        if x0 < x - tol or x1 > x + w + tol or (boxed and (y0 < y - tol or y1 > y + h + tol)):
            issues.append(f"{doc.name}: '{label[:40]}' {x1 - x0:.0f}x{y1 - y0:.0f} px in a {w:.0f}x{h:.0f} box at ({x:.0f}, {y:.0f})")
        if x1 > W - 1 or x0 < 1:
            issues.append(f"{doc.name}: '{label[:40]}' leaves the canvas")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return issues


def main() -> None:
    vol, vpred, head_ct, cls, seg = D.load_data()
    OUT.mkdir(exist_ok=True)
    for doc in (fig_pipeline(vol, vpred, seg), fig_arms(vol, head_ct, cls), fig_arms_seg(seg)):
        (OUT / f"{doc.name}.drawio").write_text(doc.xml())
        for msg in preview(doc, OUT / f"{doc.name}_preview.png"):
            print("CHECK", msg)
        print(f"{doc.name}: {len(doc.cells)} cells, {W}x{doc.h} px -> {OUT / (doc.name + '.drawio')}")


if __name__ == "__main__":
    main()
