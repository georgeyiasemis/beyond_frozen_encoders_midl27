#!/usr/bin/env python3
"""E2 (experiments_plan_20261106.md): learning-rate ablation of LoRA in the joint classification models, seed 42, on the
10 cohort-backbone pairs of E1. LoRA was trained at 3e-5, 1e-4 (default) and 3e-4; each run is evaluated at its own
selected checkpoint. Reported per pair: held-out AUROC averaged over the cohort's labels for Lin, Attn, LoRA at the three
learning rates and Attn-only FT (all other methods at their default setting), grey when the run's predictions barely
vary (recompute.SPREAD_MIN). Pooled over the 78 labels: LoRA - Attn-only FT at each LoRA learning rate, with the paper's three-stage bootstrap (recompute.clustered_ci with
recompute.label_boot_cls); every comparison has its own generator seeded with recompute.SEED_LABEL.
    python e2_lora_lr.py   -> out/e2_lora_lr_{runs,labels,summary}.csv, paper/tables/{tab_lora_lr,lora_lr_numbers}.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e1_probe_lr as e1  # noqa: E402
import recompute as rc  # noqa: E402

OUT = HERE / "out"
TEX = HERE.parent / "paper" / "tables"
LORA = ["lora_lr3e5", "lora", "lora_lr3e4"]           # 3e-5, 1e-4 (default), 3e-4
REF = ["lin", "attn", "attnft"]
COLS = ["lin", "attn", *LORA, "attnft"]
HEAD = ["Lin", "Attn", "$3\\times10^{-5}$", "$10^{-4}$", "$3\\times10^{-4}$", "Attn-only FT"]
BB_TEX = {"tapct": "TAP-CT", "spectre": "SPECTRE", "pillar0": "Pillar-0"}
DS_TEX = {"amos_covid_luna": "CVPR26", "fdg": "FDG-PET/CT", "rsna2023": "RSNA23 trauma", "rsna2022": "RSNA22 C-spine",
          "rsna2019_ich": "RSNA19 ICH"}
LR_TAG = {"lora_lr3e5": "Low", "lora": "Def", "lora_lr3e4": "High"}


def run_table() -> pd.DataFrame:
    """Default lin / attn / lora (E1 table) + the E2 LoRA runs + default joint Attn-only FT, for the 10 pairs."""
    t = e1.run_table()
    t = t[t.method.isin(["lin", "attn", "lora"])]
    rows = []
    for bb, ds in e1.PAIRS:
        for m in ("lora_lr3e5", "lora_lr3e4"):
            path = f"runs/classification/{ds}/{bb}/joint/{m}/config"
            best = e1.RES / path / "checkpoints" / "best" / "best_val_metric_val_auroc.txt"
            if not best.is_file():
                continue
            it, _ = rc.read_best(best)
            iters = sorted(int(p.name.split("_")[1]) for p in (e1.RES / path / "val_case_metrics").glob("iter_*"))
            rows.append(dict(cohort=ds, backbone=bb, method=m, path=path, best_iter=it, final_iter=iters[-1] if iters else 0))
    runs = pd.read_csv(HERE / "out" / "runs.csv")
    d = runs[(runs.task == "cls") & (runs["mode"] == "joint") & (runs.seed == 42) & (runs.method == "attnft")]
    rows += [dict(cohort=r.cohort, backbone=r.backbone, method="attnft", path=r.path, best_iter=int(r.best_iter),
                  final_iter=int(r.final_iter)) for r in d.itertuples() if (r.backbone, r.cohort) in e1.PAIRS]
    t2 = pd.DataFrame(rows)
    t2["finished"] = t2.final_iter >= 8000
    t2["percase_file"] = [f"{p}/val_case_metrics/iter_{b:06d}/logits.csv" for p, b in zip(t2.path, t2.best_iter)]
    return pd.concat([t, t2], ignore_index=True)


def _s(x: float) -> str:
    return "$0.000$" if round(x, 3) == 0 else f"${x:+.3f}$"


def main() -> None:
    t = run_table()
    if (~t.finished).any():
        sys.exit("unfinished runs:\n" + t[~t.finished][["cohort", "backbone", "method", "final_iter"]].to_string(index=False))
    t = t.reset_index(drop=True)
    store = e1.load(t)
    idx = {(r.backbone, r.cohort, r.method): i for i, r in t.iterrows()}
    # held-out AUROC per label and method
    auc = pd.DataFrame([dict(backbone=bb, cohort=ds, method=m, label=lab, auroc=rc.auroc1(v["test"][1], v["test"][2]),
                             pred_sd=float(np.std(v["test"][2])))
                        for (bb, ds, m), i in idx.items() for lab, v in store[i].items() if "test" in v])
    # pooled differences LoRA(lr) - reference, label level, three-stage bootstrap
    recs, summ = [], []
    for lr in LORA:
        for ref in ("attnft",):
            rng = np.random.default_rng(rc.SEED_LABEL)
            vals, case = [], []
            for bb, ds in sorted(e1.PAIRS):
                A, B = store[idx[(bb, ds, lr)]], store[idx[(bb, ds, ref)]]
                v, c = [], []
                for lab in A:
                    a, b = A[lab].get("test"), B.get(lab, {}).get("test")
                    if a is None or b is None or (al := rc.align(a, b)) is None:
                        continue
                    v.append(rc.auroc1(a[1][al[0]], a[2][al[0]]) - rc.auroc1(b[1][al[1]], b[2][al[1]]))
                    c.append(rc.label_boot_cls({"x": a, "y": b}, [("x", 1.0), ("y", -1.0)], rng))
                    recs.append(dict(backbone=bb, cohort=ds, lora=lr, ref=ref, label=lab, diff=v[-1]))
                vals.append(np.array(v))
                case.append(np.stack(c))
            lo, hi = rc.clustered_ci(vals, rng, case)
            allv = np.concatenate(vals)
            summ.append(dict(lora=lr, ref=ref, mean=float(allv.mean()), lo=lo, hi=hi, n_labels=len(allv),
                             n_clusters=len(vals), clusters_pos=int(sum(x.mean() > 0 for x in vals))))
    lab, summ = pd.DataFrame(recs), pd.DataFrame(summ)
    OUT.mkdir(exist_ok=True)
    t.drop(columns=["path"]).to_csv(OUT / "e2_lora_lr_runs.csv", index=False)
    lab.to_csv(OUT / "e2_lora_lr_labels.csv", index=False)
    summ.to_csv(OUT / "e2_lora_lr_summary.csv", index=False)
    (OUT / "e2_lora_lr_selection.csv").unlink(missing_ok=True)
    write_paper(auc, summ, int(t.method.isin(["lora_lr3e5", "lora_lr3e4"]).sum()))
    piv = auc.groupby(["backbone", "cohort", "method"]).auroc.mean().unstack()[COLS]
    pd.set_option("display.width", 200)
    print(piv.round(3).to_string())
    print(summ.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


def write_paper(auc: pd.DataFrame, summ: pd.DataFrame, n_runs: int) -> None:
    """paper/tables/lora_lr_numbers.tex (macros) and paper/tables/tab_lora_lr.tex (Appendix D table)."""
    piv = auc.groupby(["backbone", "cohort", "method"]).auroc.mean().unstack()
    spread = auc.groupby(["backbone", "cohort", "method"]).pred_sd.median()   # run-level spread, as recompute.py
    low = {k for k, v in spread.items() if v < rc.SPREAD_MIN}
    allm = auc.groupby("method").auroc.mean()
    s = summ.set_index(["lora", "ref"])
    n = len(COLS) + 1
    rows = []
    for bb in ("tapct", "spectre", "pillar0"):
        rows.append(f"\\multicolumn{{{n}}}{{@{{}}l}}{{\\textit{{{BB_TEX[bb]}}}}} \\\\")
        for bb2, ds in e1.PAIRS:
            if bb2 == bb:
                rows.append(f"{DS_TEX[ds]} & " + " & ".join(f"\\textcolor{{gray}}{{{piv.loc[(bb, ds), c]:.3f}}}" if (bb, ds, c) in low else f"{piv.loc[(bb, ds), c]:.3f}" for c in COLS) + " \\\\")
    rows.append("\\midrule")
    rows.append(f"All ({int(s.iloc[0].n_labels)} labels) & " + " & ".join(f"{allm[c]:.3f}" for c in COLS) + " \\\\")
    for ref, name in (("attnft", "LoRA $-$ Attn-only FT"),):
        cells = [f"{_s(s.loc[(lr, ref)]['mean'])}" for lr in LORA]
        ci = [f"[{_s(s.loc[(lr, ref)].lo)}, {_s(s.loc[(lr, ref)].hi)}]" for lr in LORA]
        rows.append(f"{name} & & & " + " & ".join(cells) + " & \\\\")
        rows.append(" & & & " + " & ".join(f"\\scriptsize {c}" for c in ci) + " & \\\\")
    (TEX / "tab_lora_lr.tex").write_text("\n".join([
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        " & & & \\multicolumn{3}{c}{LoRA learning rate} & \\\\", "\\cmidrule(lr){4-6}",
        "Cohort & " + " & ".join(HEAD) + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n")
    macros = {"numLoralrRunsN": str(n_runs)}
    for lr in LORA:
        for ref, rt in (("attnft", "Attnft"),):
            r = s.loc[(lr, ref)]
            for suf, v in (("Meandiff", r["mean"]), ("Lodiff", r.lo), ("Hidiff", r.hi)):
                macros[f"numLoralr{LR_TAG[lr]}{rt}{suf}"] = _s(v).strip("$")
            macros[f"numLoralr{LR_TAG[lr]}{rt}PosN"] = str(int(r.clusters_pos))
    macros["numLoralrPairsN"] = str(int(s.iloc[0].n_clusters))
    for lr in LORA:
        macros[f"numLoralr{LR_TAG[lr]}LowspreadN"] = str(sum(k[2] == lr for k in low))
        macros[f"numLoralr{LR_TAG[lr]}AllAuroc"] = f"{allm[lr]:.3f}"
    for c in REF:
        macros[f"numLoralrRef{c.capitalize()}AllAuroc"] = f"{allm[c]:.3f}"
    (TEX / "lora_lr_numbers.tex").write_text("% generated by metrics/e2_lora_lr.py\n" + "".join(
        f"\\providecommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))


if __name__ == "__main__":
    main()
