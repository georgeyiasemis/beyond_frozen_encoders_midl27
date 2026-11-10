#!/usr/bin/env python3
"""Choose the held-out cases shown in the segmentation example figure by a fixed rule (ported from the draft's
select_seg_examples.py to the recomputed run table and metrics).

    python select_seg_examples.py   -> out/selection/seg_example_selection.json, ../paper/tables/seg_example_numbers.tex

For each example (a set of runs on one target), every held-out case where all runs have a valid Dice is considered.
A case qualifies when, on both TAP-CT and SPECTRE, the per-case ranking of the runs agrees with the ranking of their mean
held-out Dice (LiTS tumor: on the best and the worst method; AMOS22: on the full ranking of the four methods). Among
qualifying cases (all cases if none qualifies) the chosen case minimises the squared distance between its pairwise
per-case Dice differences and the pairwise differences of the means. "lits_joint" (appendix LiTS figure) applies the
LiTS rule to the joint models' mean Dice over liver and tumor, among held-out cases whose reference has a tumor (label 2
in the MSD Task03 label file).
Per-case Dice: the run's per-case file (runs.csv), held-out split, valid cases. Means: metrics_seg.csv (held-out,
seed 42), mean over the structures of the run.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from select_cases import OUT_SEL, RES, V2, metrics, run_row, seg_percase

EXAMPLES = {
    # name: (dataset, structure or None for the mean over structures, [(method, mode), ...]); AMOS22 "joint" is the
    # 24 000-step joint model (the draft's joint_lm)
    "lits": ("lits", "tumor", [("lin", "pertask"), ("dec", "pertask"), ("lora", "pertask"), ("fullft", "pertask")]),
    "amos": ("amos22", None, [("lin", "joint"), ("dec", "joint"), ("lora", "joint"), ("fullft", "joint")]),
    "lits_joint": ("lits", None, [("lin", "joint"), ("dec", "joint"), ("lora", "joint"), ("fullft", "joint")]),
}
RULE = {"lits": "ends", "amos": "full", "lits_joint": "ends"}
BACKBONES = ("tapct", "spectre")
LITS_LABELS = RES / "data" / "public" / "MSD" / "Task03_Liver" / "labelsTr"   # LiTS = MSD Task03 (copies, read-only)


def has_tumor(case: str) -> bool:
    return bool((np.asanyarray(nib.load(LITS_LABELS / f"{case}.nii.gz").dataobj) == 2).any())


def per_case(ds: str, struct: str | None, bb: str, mt: str, md: str) -> tuple[pd.Series, float]:
    m = metrics("seg")
    m = m[(m.cohort == ds) & (m.backbone == bb) & (m.method == mt) & (m["mode"] == md) & (m.split == "test") & (m.seed == 42)]
    if struct is not None:
        row = run_row("seg", ds, bb, md, mt, label=struct if md == "pertask" else "joint")
        d = seg_percase(row)
        d = d[d.structure == struct]
        mm = m[m.label == struct]
        if len(mm) != 1:
            raise SystemExit(f"{ds}/{bb}/{md}/{mt}/{struct}: expected one metrics row, got {len(mm)}")
        return d.set_index("case_id").dice, float(mm.dice.iloc[0])
    d = seg_percase(run_row("seg", ds, bb, md, mt))
    return d.groupby("case_id").dice.mean(), float(m.dice.mean())


def select(ds: str, struct: str | None, runs: list[tuple[str, str]], rule: str = "full", keep=None) -> dict:
    series, means = {}, {}
    for bb in BACKBONES:
        for mt, md in runs:
            series[(bb, mt, md)], means[(bb, mt, md)] = per_case(ds, struct, bb, mt, md)
    df = pd.DataFrame(series).dropna()
    if keep is not None:   # case filter (the means stay those of the whole held-out split)
        df = df[[keep(c) for c in df.index]]
    rows = []
    for cid, row in df.iterrows():
        match, dist = True, 0.0
        for bb in BACKBONES:
            ks = [k for k in series if k[0] == bb]
            mean_rank = sorted(ks, key=lambda k: -means[k]); case_rank = sorted(ks, key=lambda k: -row[k])
            if rule == "full" and mean_rank != case_rank:
                match = False
            if rule == "ends" and (case_rank[0] != mean_rank[0] or case_rank[-1] != mean_rank[-1]):
                match = False
            dist += sum((row[a] - row[b] - (means[a] - means[b])) ** 2 for a, b in itertools.combinations(ks, 2))
        rows.append((cid, match, dist))
    r = pd.DataFrame(rows, columns=["case", "match", "dist"]).sort_values(["match", "dist"], ascending=[False, True])
    best = str(r.iloc[0].case)
    return {
        "dataset": ds, "structure": struct, "case": best, "n_cases": int(len(df)), "n_match": int(r.match.sum()),
        "case_dice": {"/".join(k): round(float(df.loc[best, k]), 4) for k in series},
        "mean_dice": {"/".join(k): round(v, 4) for k, v in means.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=OUT_SEL / "seg_example_selection.json")
    ap.add_argument("--out-tex", type=Path, default=V2 / "paper" / "tables" / "seg_example_numbers.tex")
    args = ap.parse_args()
    sel = {name: select(*spec, rule=RULE[name], keep=has_tumor if name == "lits_joint" else None) for name, spec in EXAMPLES.items()}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(sel, indent=2))
    lines = ["% written by figures/select_seg_examples.py (selection rule of the segmentation example figure)"]
    for name, s in sel.items():
        tag = {"lits": "Lits", "amos": "Amos", "lits_joint": "Litsjoint"}[name]
        lines += [f"\\providecommand{{\\numSegEx{tag}Case}}{{{s['case'].replace('_', '-')}}}",
                  f"\\providecommand{{\\numSegEx{tag}CasesN}}{{{s['n_cases']}}}",
                  f"\\providecommand{{\\numSegEx{tag}MatchN}}{{{s['n_match']}}}"]
    args.out_tex.write_text("\n".join(lines) + "\n")
    old_f = RES / "examples" / "seg" / "selection.json"
    old = json.loads(old_f.read_text()) if old_f.is_file() else {}
    for name, s in sel.items():
        o = old.get(name, {})
        print(f"{name}: {s['case']} ({s['n_match']} of {s['n_cases']} cases match the mean ranking on both backbones); "
              f"draft: {o.get('case')} ({o.get('n_match')} of {o.get('n_cases')})")
        od = {k.replace("joint_lm", "joint"): v for k, v in o.get("case_dice", {}).items()}
        dd = [abs(v - od[k]) for k, v in s["case_dice"].items() if k in od]
        if dd:
            print(f"    max |v2 - draft| per-case Dice of the chosen case: {max(dd):.4g}")


if __name__ == "__main__":
    main()
