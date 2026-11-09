"""Main-text length of a compiled MIDL PDF: pages before the References page, the last one as a fraction of the
text block (body words' lowest edge, from pdftotext -bbox; running head and page number excluded)."""
import re, subprocess, sys
pdf = sys.argv[1]; T = "/home/g.yiasemis/miniconda3/envs/tex/bin/"
def words(p):
    x = subprocess.run([T + "pdftotext", "-bbox", "-f", str(p), "-l", str(p), pdf, "-"], capture_output=True, text=True).stdout
    return [(float(a), float(b), w) for a, b, w in re.findall(r'yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)">([^<]*)<', x)]
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run([T + "pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
ref = next(p for p in range(2, n + 1) if any(w == "References" for _, _, w in words(p)[:15]))
def body(p):
    ws = words(p); ys = sorted(ws, key=lambda t: t[0])
    return [t for t in ws if t[0] > ys[0][1] + 5 and t[1] < max(b for _, b, _ in ws) - 5]   # drop running head, page number
full = body(2)
top, bottom = min(t[0] for t in full), max(t[1] for t in full)
last = body(ref - 1)
frac = (max(t[1] for t in last) - top) / (bottom - top)
print(f"main text: {ref - 2 + frac:.2f} pages (ends on p{ref - 1}, {100 * frac:.0f}% of that page; References on p{ref})")
