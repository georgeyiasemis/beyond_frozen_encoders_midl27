#!/usr/bin/env python3
"""Design counts quoted in the paper (methods, labels, runs, split sizes), computed from the run inventory and the
per-case files; written to paper/tables/design_numbers.tex. Metric numbers come from metrics/recompute.py."""
from pathlib import Path
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
RAW = V2.parent / "midl_paper" / "results"
inv = pd.read_csv(V2 / "report" / "run_inventory.csv")
inv = inv[(inv.status == "finished") & (inv.in_design == "yes") & inv.excluded.isna() & inv.aux.isna()]
inv = inv[inv.cohort != "breastdivider"]
# one row per designed run (drop the second site copy)
inv = inv.drop_duplicates(subset=["task", "cohort", "backbone", "mode", "method", "label", "seed"])
main = inv[inv.design_class == "main"]
rev = inv[inv.design_class != "main"]
macros = {
    "ClsMethodsN": 11, "SegMethodsN": 7,
    "LabelsN": 17 + 4 + 8 + 4 + 6, "StructuresN": 15 + 3 + 2 + 2 + 15,
    "RunsMainN": len(main), "RunsMainClsN": int((main.task == "cls").sum()), "RunsMainSegN": int((main.task == "seg").sum()),
    "RunsRevisionN": len(rev),
    "SeedsclsRunsN": int(((rev.task == "cls") & rev["mode"].isin(["joint_s43", "joint_s44"])).sum()),
    "SeedssegRunsN": int(((rev.task == "seg") & rev["mode"].isin(["joint_s43", "joint_s44"])).sum()),
}
for bb in ["tapct", "spectre", "pillar0"]:  # VoCo is quarantined (excluded in the inventory)
    for task in ["cls", "seg"]:
        macros[f"RunsMain{task.capitalize()}{bb.capitalize().replace("0", "Z")}N"] = int(((main.task == task) & (main.backbone == bb)).sum())
# CVPR26 per-source case counts of the selection / held-out halves, from one joint run's per-case file
row = inv[(inv.task == "cls") & (inv.cohort == "amos_covid_luna") & (inv.backbone == "tapct") & (inv["mode"] == "joint") & (inv.method == "lin")].iloc[0]
f = RAW / row.path / "val_case_metrics" / f"iter_{int(row.best_iter):06d}" / "logits.csv"
d = pd.read_csv(f)
d["src"] = d.task.str.split(":").str[0]
for src, name in (("amos", "Amos"), ("covid_ct", "Covid"), ("luna25", "Luna")):
    for split, tag in (("val", "Val"), ("test", "Test")):
        macros[f"CasesCvpr{name}{tag}"] = int(d[(d.src == src) & (d.split == split)].case_id.nunique())
out = V2 / "paper" / "tables" / "design_numbers.tex"
out.write_text("".join(f"\\providecommand{{\\num{k}}}{{{v}}}\n" for k, v in macros.items()))
print(out, len(macros)); print(macros)
