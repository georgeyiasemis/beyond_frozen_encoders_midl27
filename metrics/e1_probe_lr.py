#!/usr/bin/env python3
"""E1 (experiments_plan_20261106.md): LoRA against frozen probes whose learning rate is selected on the selection split.

Joint classification models, seed 42. For each cohort-backbone pair, the learning rate of Lin (1e-3 default, 1e-2,
1e-4) and of Attn (1e-4 default, 3e-5, 3e-4, 1e-3) is selected by the mean selection-split AUROC over the cohort's
labels, at each run's own selected checkpoint (as for every run of the paper). LoRA - probe is then computed per label
on the held-out split, for the default and for the selected probe, and summarized per backbone and pooled with the
paper's three-stage bootstrap (recompute.clustered_ci with recompute.label_boot_cls).

    python e1_probe_lr.py            -> out/e1_probe_lr_{runs,selection,labels,summary}.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recompute as rc  # noqa: E402

RES = HERE.parent.parent / "midl_paper" / "results"
OUT = HERE / "out"
FAMILY = {"lin": ["lin", "lin_lr1e2", "lin_lr1e4"], "attn": ["attn", "attn_lr3e5", "attn_lr3e4", "attn_lr1e3"]}
LR = {"lin": 1e-3, "lin_lr1e2": 1e-2, "lin_lr1e4": 1e-4, "attn": 1e-4, "attn_lr3e5": 3e-5, "attn_lr3e4": 3e-4, "attn_lr1e3": 1e-3}
PAIRS = [("tapct", "amos_covid_luna"), ("tapct", "fdg"), ("tapct", "rsna2023"), ("tapct", "rsna2022"), ("tapct", "rsna2019_ich"),
         ("spectre", "amos_covid_luna"), ("spectre", "fdg"), ("spectre", "rsna2023"), ("spectre", "rsna2022"), ("pillar0", "rsna2019_ich")]


def run_table() -> pd.DataFrame:
    """Default lin / attn / lora joint runs from the paper's run table plus the finished E1 runs."""
    runs = pd.read_csv(HERE / "out" / "runs.csv")
    d = runs[(runs.task == "cls") & (runs["mode"] == "joint") & (runs.seed == 42) & runs.method.isin(["lin", "attn", "lora"])]
    rows = [dict(cohort=r.cohort, backbone=r.backbone, method=r.method, path=r.path, best_iter=int(r.best_iter), final_iter=int(r.final_iter))
            for r in d.itertuples() if (r.backbone, r.cohort) in PAIRS]
    for bb, ds in PAIRS:
        for m in [x for f in FAMILY.values() for x in f if x not in ("lin", "attn")]:
            path = f"raw/snellius/{ds}/{bb}/joint/{m}/config"
            best = RES / path / "checkpoints" / "best" / "best_val_metric_val_auroc.txt"
            if not best.is_file():
                continue
            it, _ = rc.read_best(best)
            iters = sorted(int(p.name.split("_")[1]) for p in (RES / path / "val_case_metrics").glob("iter_*"))
            rows.append(dict(cohort=ds, backbone=bb, method=m, path=path, best_iter=it, final_iter=iters[-1] if iters else 0))
    t = pd.DataFrame(rows)
    t["finished"] = t.final_iter >= 8000
    t["percase_file"] = [f"{p}/val_case_metrics/iter_{b:06d}/logits.csv" for p, b in zip(t.path, t.best_iter)]
    return t


def load(t: pd.DataFrame) -> dict:
    out = {}
    for i, r in t.iterrows():
        ns = type("R", (), dict(task="cls", rel=r.path.split("/", 2)[2], label_run="joint", cohort=r.cohort))
        _, arr = rc._parse_one((i, "cls", r.percase_file, rc.label_names(ns), str(RES)))
        out[i] = {k: v for k, v in arr.items() if k != "__warn__" and k not in rc.EXCLUDED_LABELS}
    return out


def main() -> None:
    t = run_table()
    t = t[t.finished].reset_index(drop=True)
    store = load(t)
    # per-run selection-split mean AUROC and held-out AUROC per label
    t["sel_auroc"] = [np.mean([rc.auroc1(v["val"][1], v["val"][2]) for v in store[i].values()
                               if "val" in v and 0 < v["val"][1].sum() < len(v["val"][1])]) for i in t.index]
    sel = []
    for (bb, ds), g in t.groupby(["backbone", "cohort"]):
        for fam, ms in FAMILY.items():
            gg = g[g.method.isin(ms)]
            best = gg.loc[gg.sel_auroc.idxmax()]
            sel.append(dict(backbone=bb, cohort=ds, probe=fam, n_lr=len(gg), selected=best.method, lr=LR[best.method],
                            sel_auroc_default=float(gg[gg.method == fam].sel_auroc.iloc[0]), sel_auroc_selected=float(best.sel_auroc)))
    sel = pd.DataFrame(sel)
    # label-level LoRA - probe on the held-out split, default and selected probe
    rng = np.random.default_rng(rc.SEED_LABEL)
    recs, boots = [], {}
    for r in sel.itertuples():
        g = t[(t.backbone == r.backbone) & (t.cohort == r.cohort)]
        i_lora = g[g.method == "lora"].index[0]
        for kind, m in (("default", r.probe), ("selected", r.selected)):
            i_p = g[g.method == m].index[0]
            for lab in store[i_lora]:
                a, b = store[i_lora][lab].get("test"), store[i_p].get(lab, {}).get("test")
                if a is None or b is None:
                    continue
                arrs = {"lora": a, "probe": b}
                al = rc.align(a, b)
                if al is None:
                    continue
                v = rc.auroc1(a[1][al[0]], a[2][al[0]]) - rc.auroc1(b[1][al[1]], b[2][al[1]])
                bt = rc.label_boot_cls(arrs, [("lora", 1.0), ("probe", -1.0)], rng)
                recs.append(dict(backbone=r.backbone, cohort=r.cohort, probe=r.probe, kind=kind, label=lab, diff=v))
                boots[(r.backbone, r.cohort, r.probe, kind, lab)] = bt
    lab = pd.DataFrame(recs)
    summ = []
    for probe in FAMILY:
        for kind in ("default", "selected"):
            for scope in ("tapct", "spectre", "pillar0", "all"):
                d = lab[(lab.probe == probe) & (lab.kind == kind) & ((lab.backbone == scope) if scope != "all" else True)]
                clusters = [g for _, g in d.groupby(["backbone", "cohort"])]
                vals = [g["diff"].to_numpy() for g in clusters]
                case = [np.stack([boots[(r.backbone, r.cohort, probe, kind, r.label)] for r in g.itertuples()]) for g in clusters]
                lo, hi = rc.clustered_ci(vals, rng, case)
                summ.append(dict(probe=probe, kind=kind, scope=scope, mean=float(d["diff"].mean()), lo=lo, hi=hi,
                                 n_labels=len(d), n_clusters=len(clusters), clusters_pos=int(sum(v.mean() > 0 for v in vals))))
    summ = pd.DataFrame(summ)
    OUT.mkdir(exist_ok=True)
    t.drop(columns=["path"]).to_csv(OUT / "e1_probe_lr_runs.csv", index=False)
    sel.to_csv(OUT / "e1_probe_lr_selection.csv", index=False)
    lab.to_csv(OUT / "e1_probe_lr_labels.csv", index=False)
    summ.to_csv(OUT / "e1_probe_lr_summary.csv", index=False)
    write_paper(sel, summ, int((~t.method.isin(['lin', 'attn', 'lora'])).sum()))
    pd.set_option("display.width", 200)
    print(sel.to_string(index=False))
    print(summ.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


def _s(x: float) -> str:
    return "$0.000$" if round(x, 3) == 0 else f"${x:+.3f}$"


def write_paper(sel: pd.DataFrame, summ: pd.DataFrame, n_runs_e1: int) -> None:
    """paper/tables/probe_lr_numbers.tex (macros) and paper/tables/tab_probe_lr.tex (Appendix D table)."""
    tex = HERE.parent / "paper" / "tables"
    piv = summ.set_index(["probe", "kind", "scope"])
    change = max(abs(piv.loc[(p, "selected", "all")]["mean"] - piv.loc[(p, "default", "all")]["mean"]) for p in FAMILY)
    lin_kept = int((sel[sel.probe == "lin"].selected == "lin").sum())
    attn_low = int((sel[sel.probe == "attn"].selected == "attn_lr3e5").sum())
    n_pairs = int((sel.probe == "lin").sum())
    change_bb = max(abs(piv.loc[(p, "selected", bb)]["mean"] - piv.loc[(p, "default", bb)]["mean"])
                    for p in FAMILY for bb in ("tapct", "spectre", "pillar0"))
    n_e1 = int(n_runs_e1)
    src = "".join(f.read_text() for f in sorted(tex.glob("*.tex")) if f.name != "probe_lr_numbers.tex")
    rev = int(re.search(r"numRunsRevisionN\}\{(\d+)\}", src).group(1))
    macros = {"numProbelrRunsN": str(n_e1), "numRunsRevisionAllN": str(rev + n_e1), "numProbelrMaxchange": f"{change:.3f}", "numProbelrMaxchangeBb": f"{change_bb:.3f}", "numProbelrPairsN": str(n_pairs),
              "numProbelrLinKeptN": str(lin_kept), "numProbelrAttnLowN": str(attn_low)}
    (tex / "probe_lr_numbers.tex").write_text("% generated by metrics/e1_probe_lr.py\n" + "".join(
        f"\\providecommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    name = {"tapct": "TAP-CT", "spectre": "SPECTRE", "pillar0": "Pillar-0", "all": "all"}
    rows = []
    for probe, lab in (("lin", "LoRA $-$ Lin"), ("attn", "LoRA $-$ Attn")):
        for sc in ("all", "tapct", "spectre", "pillar0"):
            d, e = piv.loc[(probe, "default", sc)], piv.loc[(probe, "selected", sc)]
            rows.append(f"{lab} & {name[sc]} & {int(d.n_labels)} & {_s(d['mean'])} & [{_s(d.lo)}, {_s(d.hi)}] & "
                        f"{_s(e['mean'])} & [{_s(e.lo)}, {_s(e.hi)}] & {int(e.clusters_pos)}/{int(e.n_clusters)} \\\\")
    (tex / "tab_probe_lr.tex").write_text("\n".join([
        "\\begin{tabular}{llrrcrcc}", "\\toprule",
        " & & & \\multicolumn{2}{c}{Default learning rate} & \\multicolumn{2}{c}{Selected learning rate} & \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Comparison & Backbone & $n$ & Mean & 95\\% interval & Mean & 95\\% interval & Cohorts $>0$ \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}"]) + "\n")


if __name__ == "__main__":
    main()
