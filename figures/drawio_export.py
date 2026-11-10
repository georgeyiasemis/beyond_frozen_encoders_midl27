#!/usr/bin/env python3
"""Editable draw.io versions of the method diagrams (Figures 2, 6 and 7: fig_pipeline, fig_arms, fig_arms_seg).

The layout and every number come from diagrams.py: its figure functions are run unchanged, and each primitive they
draw is converted to a draw.io (mxGraph) cell instead of being saved as a PDF:
  * boxes, cards, tokens, table shading (FancyBboxPatch / Rectangle) -> rectangles (rounded where the box is rounded),
  * circles (residual adds, tap points, output dots)                   -> ellipses,
  * cuboids (decoder tensors, Pillar-0 token grids)                    -> draw.io cube shapes,
  * lines and arrows                                                   -> edges (arrows keep their block head),
  * text, including the $W_{qkv}$-style subscripts                     -> HTML labels,
  * CT thumbnails (inset image axes)                                   -> embedded PNG images.
Units: the 7.0 in drawn width becomes 700 px (1 px = 1/100 in), so fonts keep their drawn size (pt * 100/72 px).

    python drawio_export.py    -> drawio/fig_pipeline.drawio, drawio/fig_arms.drawio, drawio/fig_arms_seg.drawio
                                  (+ drawio/<name>_preview.png, a redraw of the draw.io cells for checking)
"""
from __future__ import annotations

import base64
import html
import io
import re
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagrams as D  # noqa: E402

OUT = HERE / "drawio"
DPI = 100            # 1 px = 1/100 in of the drawn figure
IMG_DPI = 220        # resolution of the embedded CT thumbnails

CUBES: list = []     # (ax, x, y, w, h, d, fc, ec, lw, z) recorded by the patched cuboid()
CAPTURED: dict = {}  # name -> figure, filled by the patched save()


def _cuboid(ax, x, y, w, h, d, fc, ec, lw=0.6, z=3):
    CUBES.append((ax, x, y, w, h, d, fc, ec, lw, z))
    return d * 0.6, d * 0.45


def _save(fig, name):
    CAPTURED[name] = fig


D.cuboid = _cuboid
D.save = _save


# ------------------------------------------------------------------ mathtext -> HTML
GREEK = {r"\alpha": "α", r"\beta": "β", r"\sqrt ": "√", r"\sqrt": "√"}


def _math(s: str) -> str:
    """Small mathtext subset used in the diagrams: sub/superscripts, \\mathrm, \\frac, Greek letters."""
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1/\2)", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\\up{\1}", s)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"\\sqrt \1", s)
    for k, v in GREEK.items():
        s = s.replace(k, v)
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in "_^":
            tag = "sub" if c == "_" else "sup"
            if s[i + 1] == "{":
                j = s.index("}", i + 2)
                body, i = s[i + 2:j], j + 1
            else:
                body, i = s[i + 1], i + 2
            body = re.sub(r"\\up\{([^}]*)\}", r"\1", body)
            out.append(f"<{tag}>{html.escape(body)}</{tag}>")
        elif s.startswith(r"\up{", i):
            j = s.index("}", i)
            out.append(html.escape(s[i + 4:j]))
            i = j + 1
        elif c.isalpha():
            out.append(f"<i>{c}</i>")
            i += 1
        else:
            out.append(html.escape(c))
            i += 1
    return "".join(out)


def to_html(s: str) -> str:
    parts = re.split(r"(\$[^$]*\$)", s)
    h = "".join(_math(p[1:-1]) if p.startswith("$") and p.endswith("$") and len(p) > 1 else html.escape(p) for p in parts)
    return h.replace("\n", "<br>")


# ------------------------------------------------------------------ cells
class Doc:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.cells: list[tuple[float, str]] = []   # (z, xml)
        self.n = 2
        self.preview: list = []                   # primitives for the check redraw

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def vertex(self, z, x, y, w, h, style, value=""):
        self.cells.append((z, f'<mxCell id="{self._id()}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
                              f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>'))

    def edge(self, z, pts, style):
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        mid = "".join(f'<mxPoint x="{px:.2f}" y="{py:.2f}"/>' for px, py in pts[1:-1])
        arr = f'<Array as="points">{mid}</Array>' if mid else ""
        self.cells.append((z, f'<mxCell id="{self._id()}" value="" style={quoteattr(style)} edge="1" parent="1">'
                              f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x0:.2f}" y="{y0:.2f}" as="sourcePoint"/>'
                              f'<mxPoint x="{x1:.2f}" y="{y1:.2f}" as="targetPoint"/>{arr}</mxGeometry></mxCell>'))

    def xml(self):
        body = "".join(c for _, c in sorted(self.cells, key=lambda t: t[0]))
        return ('<mxfile host="drawio_export.py"><diagram name="{n}" id="{n}"><mxGraphModel dx="{w}" dy="{h}" grid="0" gridSize="10" '
                'guides="1" tooltips="1" connect="0" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{w}" pageHeight="{h}" '
                'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{b}</root></mxGraphModel></diagram></mxfile>'
                ).format(n=self.name, w=int(self.w + 1), h=int(self.h + 1), b=body)


# Colours come from diagrams.py (the paper figures); only the font and a minimum outline width are set here.
FONT = "LM Roman 12"     # as in the paper figures (fonts/lmroman12-*.otf; install them to see it in draw.io)
MIN_LW = 1.0             # draw.io px; thinner outlines look faint on screen


def col(c):
    return "none" if c is None or (len(c) == 4 and c[3] == 0) else to_hex(c, keep_alpha=False)


def convert(name: str, fig) -> Doc:
    fig.set_dpi(DPI)
    fig.canvas.draw()
    W, H = fig.bbox.width, fig.bbox.height
    doc = Doc(name, W, H)
    main = fig.axes[0]
    T = main.transData
    flip = lambda x, y: (x, H - y)

    def rect_px(x, y, w, h):
        (ax0, ay0), (ax1, ay1) = T.transform((x, y)), T.transform((x + w, y + h))
        return min(ax0, ax1), H - max(ay0, ay1), abs(ax1 - ax0), abs(ay1 - ay0)

    # patches
    for p in main.patches:
        z = p.get_zorder()
        fc, ec, lw = col(p.get_facecolor()), col(p.get_edgecolor()), p.get_linewidth() * DPI / 72
        if ec not in ("none", "#ffffff"):
            lw = max(lw, MIN_LW)
        if isinstance(p, FancyBboxPatch):
            x, y, w, h = rect_px(p.get_x(), p.get_y(), p.get_width(), p.get_height())
            r = getattr(p.get_boxstyle(), "rounding_size", 0) or 0
            rpx = 0.0 * abs(T.transform((r, 0))[0] - T.transform((0, 0))[0])   # square corners
            style = (f"rounded=0;fillColor={fc};strokeColor={ec};strokeWidth={lw:.2f};"
                     + ("dashed=1;" if p.get_linestyle() not in ("-", "solid") else ""))
            doc.vertex(z, x, y, w, h, style)
            doc.preview.append(("rect", x, y, w, h, fc, ec, lw, rpx))
        elif isinstance(p, Circle):
            cx, cy = T.transform(p.center)
            rr = abs(T.transform((p.radius, 0))[0] - T.transform((0, 0))[0])
            doc.vertex(z, cx - rr, H - cy - rr, 2 * rr, 2 * rr, f"ellipse;fillColor={fc};strokeColor={ec};strokeWidth={lw:.2f};")
            doc.preview.append(("ellipse", cx - rr, H - cy - rr, 2 * rr, 2 * rr, fc, ec, lw, 0))
        elif isinstance(p, Rectangle):
            x, y, w, h = rect_px(p.get_x(), p.get_y(), p.get_width(), p.get_height())
            doc.vertex(z, x, y, w, h, f"rounded=0;fillColor={fc};strokeColor={ec};strokeWidth={lw:.2f};")
            doc.preview.append(("rect", x, y, w, h, fc, ec, lw, 0))
        else:
            raise TypeError(f"{name}: unhandled patch {type(p).__name__}")
    # cuboids: draw.io cube, depth drawn up and to the right (flipH), as diagrams.cuboid
    for ax, x, y, w, h, d, fc, ec, lw, z in CUBES:
        if ax is not main:
            continue
        dx, dy = d * 0.6, d * 0.45
        X, Y, Wd, Hd = rect_px(x, y, w + dx, h + dy)
        depth = abs(T.transform((dx, 0))[0] - T.transform((0, 0))[0])
        doc.vertex(z, X, Y, Wd, Hd, f"shape=cube;flipH=1;size={depth:.1f};darkOpacity=0.08;darkOpacity2=0.18;"
                                    f"fillColor={col(matplotlib.colors.to_rgba(fc))};strokeColor={col(matplotlib.colors.to_rgba(ec))};"
                                    f"strokeWidth={lw * DPI / 72:.2f};")
        doc.preview.append(("cube", X, Y, Wd, Hd, col(matplotlib.colors.to_rgba(fc)), col(matplotlib.colors.to_rgba(ec)), lw, depth))
    # lines
    for ln in main.lines:
        xy = T.transform(ln.get_xydata())
        pts = [flip(px, py) for px, py in xy]
        lw = ln.get_linewidth() * DPI / 72
        doc.edge(ln.get_zorder(), pts, f"endArrow=none;html=1;strokeColor={col(matplotlib.colors.to_rgba(ln.get_color()))};"
                                       f"strokeWidth={lw:.2f};rounded=0;")
        doc.preview.append(("line", pts, col(matplotlib.colors.to_rgba(ln.get_color())), lw, False))
    # arrows (annotations with an arrow patch) and text
    texts = []
    for t in main.texts:
        if isinstance(t, Annotation) and t.arrow_patch is not None:
            x1, y1 = T.transform(t.xy)
            x0, y0 = T.transform(t.xyann)
            c = col(t.arrow_patch.get_edgecolor())
            lw = t.arrow_patch.get_linewidth() * DPI / 72
            pts = [flip(x0, y0), flip(x1, y1)]
            doc.edge(t.get_zorder(), pts, f"endArrow=block;endFill=1;endSize=5;html=1;strokeColor={c};strokeWidth={lw:.2f};")
            doc.preview.append(("line", pts, c, lw, True))
        elif t.get_text().strip():
            texts.append(t)
    r = fig.canvas.get_renderer()
    for t in texts:
        bb = t.get_window_extent(r)
        fs = t.get_fontsize() * DPI / 72
        ha, va = t.get_horizontalalignment(), t.get_verticalalignment()
        bold = t.get_fontweight() in ("bold", 700)
        italic = t.get_fontstyle() == "italic"
        c = col(matplotlib.colors.to_rgba(t.get_color()))
        pad = 2.0
        x, y, w, h = bb.x0 - pad, H - bb.y1 - pad, bb.width + 2 * pad, bb.height + 2 * pad
        style = (f"text;html=1;whiteSpace=nowrap;align={ {'left': 'left', 'right': 'right'}.get(ha, 'center') };"
                 f"verticalAlign=middle;spacing=0;fontFamily={FONT};fontSize={fs:.1f};fontColor={c};"
                 f"fontStyle={(1 if bold else 0) + (2 if italic else 0)};")
        doc.vertex(t.get_zorder() + 10, x, y, w, h, style, to_html(t.get_text()))
        doc.preview.append(("text", x, y, w, h, t.get_text(), fs, c, ha, bold, italic))
    # CT thumbnails: each inset axes rendered on its own and embedded as a PNG (drawn above the main axes, as in matplotlib)
    others = fig.axes[1:]
    vis = {a: a.get_visible() for a in fig.axes}
    for k, ia in enumerate(others):
        for a in fig.axes:
            a.set_visible(a is ia)
        fig.set_dpi(IMG_DPI)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        import numpy as np
        arr = np.asarray(buf)
        bb = ia.get_window_extent(fig.canvas.get_renderer())
        Hh = arr.shape[0]
        e = 2  # keep the spine
        x0, x1 = int(max(bb.x0 - e, 0)), int(min(bb.x1 + e, arr.shape[1]))
        y0, y1 = int(max(Hh - bb.y1 - e, 0)), int(min(Hh - bb.y0 + e, Hh))
        crop = arr[y0:y1, x0:x1, :3].copy()   # the canvas buffer is reused by the next draw
        png = io.BytesIO()
        plt.imsave(png, crop, format="png")
        b64 = base64.b64encode(png.getvalue()).decode()
        s = DPI / IMG_DPI
        doc.vertex(50 + k * 0.001, x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s,
                   f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;image=data:image/png,{b64};")
        doc.preview.append(("image", x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s, crop))
    for a, v in vis.items():
        a.set_visible(v)
    fig.set_dpi(DPI)
    return doc


def preview(doc: Doc, path: Path) -> None:
    """Redraw the converted cells (not the original figure) so that the conversion can be checked by eye."""
    import numpy as np
    from matplotlib.patches import Ellipse, FancyBboxPatch as FB, Polygon
    fig = plt.figure(figsize=(doc.w / DPI, doc.h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, doc.w); ax.set_ylim(doc.h, 0); ax.axis("off")
    for p in doc.preview:
        k = p[0]
        if k == "rect":
            _, x, y, w, h, fc, ec, lw, r = p
            ax.add_patch(FB((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw * 72 / DPI))
        elif k == "ellipse":
            _, x, y, w, h, fc, ec, lw, _ = p
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, fc=fc, ec=ec, lw=lw * 72 / DPI))
        elif k == "cube":
            _, x, y, w, h, fc, ec, lw, d = p
            dy = d * 0.75
            ax.add_patch(Polygon([(x, y + dy), (x + d, y), (x + w, y), (x + w - d, y + dy)], fc=fc, ec=ec, lw=lw))
            ax.add_patch(Polygon([(x + w - d, y + dy), (x + w, y), (x + w, y + h - dy), (x + w - d, y + h)], fc=fc, ec=ec, lw=lw))
            ax.add_patch(Rectangle((x, y + dy), w - d, h - dy, fc=fc, ec=ec, lw=lw))
        elif k == "line":
            _, pts, c, lw, arrow = p
            xs, ys = zip(*pts)
            if arrow:
                ax.annotate("", xy=pts[-1], xytext=pts[0], arrowprops=dict(arrowstyle="-|>", color=c, lw=lw * 72 / DPI, mutation_scale=7))
            else:
                ax.plot(xs, ys, color=c, lw=lw * 72 / DPI)
        elif k == "image":
            _, x, y, w, h, img = p
            ax.imshow(img, extent=(x, x + w, y + h, y), zorder=5, aspect="auto")
        elif k == "text":
            _, x, y, w, h, s, fs, c, ha, bold, italic = p
            xx = {"left": x, "right": x + w}.get(ha, x + w / 2)
            ax.text(xx, y + h / 2, s, fontsize=fs * 72 / DPI, color=c, ha=ha, va="center", fontweight="bold" if bold else "normal",
                    style="italic" if italic else "normal",
                    linespacing=1.15, zorder=10, family=D.style.FONT)
    ax.set_xlim(0, doc.w); ax.set_ylim(doc.h, 0)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    vol, vpred, head_ct, cls, seg = D.load_data()
    D.fig_pipeline(vol, vpred, seg)
    D.fig_arms(vol, head_ct, cls)
    D.fig_arms_seg(seg)
    OUT.mkdir(exist_ok=True)
    for name in ("fig_pipeline", "fig_arms", "fig_arms_seg"):
        doc = convert(name, CAPTURED[name])
        (OUT / f"{name}.drawio").write_text(doc.xml())
        preview(doc, OUT / f"{name}_preview.png")
        kinds = {}
        for p in doc.preview:
            kinds[p[0]] = kinds.get(p[0], 0) + 1
        print(f"{name}: {len(doc.cells)} cells {kinds}, {doc.w:.0f}x{doc.h:.0f} px -> {OUT / (name + '.drawio')}")


if __name__ == "__main__":
    main()
