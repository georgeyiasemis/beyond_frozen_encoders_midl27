#!/usr/bin/env python3
"""Choose the success and failure cases of the appendix case figures by fixed rules, from the per-case held-out
predictions of the runs the paper uses (ported from the draft's select_cases.py to the recomputed run table).

    python select_cases.py            -> out/selection/case_selection.json

Inputs: ../metrics/out/runs.csv (which run, which per-case file), ../metrics/out/metrics_cls.csv and metrics_seg.csv
(label names, n_pos), the per-case files under midl_paper/results/ (read-only). Label and structure names of a run's
task_index / class_index follow metrics/recompute.py (imported, not modified).

Segmentation (one model per dataset, the paper's joint mode; for AMOS22 and TotalSegmentator the 24 000-step run):
    case score = mean Dice over the valid structures of a held-out case, averaged over Lin, Dec, LoRA and Full FT on
    TAP-CT and SPECTRE; success = the case at the 90th percentile, failure = the case with the lowest score, among
    cases scored by all eight runs. VoCo (quarantined) is included only with WITH_VOCO=1 (style.NO_VOCO).
Classification (per-task runs): per source, the label with the most held-out positives (FDG: excluding NEGATIVE).
    case score of a positive = share of held-out negatives it outranks (ties count half), i.e. its contribution to
    AUROC; averaged over Lin, LoRA, Attn-only FT and Full FT on every backbone of the cohort; success / failure =
    the positive at the 90th percentile / with the lowest score.
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
METRICS = V2 / "metrics" / "out"
RES = Path(os.environ.get("BFE_RESULTS") or "/projects/cvpr2026_challenge_fomo4ct/beyond_frozen_encoders")   # per-case files and saved predictions (read-only); BFE_RESULTS overrides
OUT_SEL = HERE / "out" / "selection"
sys.path.insert(0, str(V2 / "metrics"))
import recompute as rc  # noqa: E402  (label-name mapping only)
from style import NO_VOCO  # noqa: E402

SEG_METHODS = ["lin", "dec", "lora", "fullft"]
CLS_METHODS = ["lin", "lora", "attnft", "fullft"]
SEG_DATASETS = ["amos22", "kits23", "lits", "msd_pancreas", "totalseg"]
# (source key, cohort, label or None = most held-out positives, backbones)
CLS_SOURCES = [("amos", "amos_covid_luna", None, ["tapct", "spectre"]),
               ("covid", "amos_covid_luna", "covid_ct:covid", ["tapct", "spectre"]),
               ("luna", "amos_covid_luna", "luna25:lung_nodule_malignancy", ["tapct", "spectre"]),
               ("fdg", "fdg", None, ["tapct", "spectre", "voco"]),
               ("rsna2022", "rsna2022", None, ["tapct", "spectre", "voco"]),
               ("rsna2023", "rsna2023", None, ["tapct", "spectre", "voco"]),
               ("ich", "rsna2019_ich", None, ["tapct", "pillar0"])]
SEG_BACKBONES = ["tapct", "spectre", "voco"]
if NO_VOCO:
    CLS_SOURCES = [(k, d, l, [b for b in bbs if b != "voco"]) for k, d, l, bbs in CLS_SOURCES]
    SEG_BACKBONES = [b for b in SEG_BACKBONES if b != "voco"]


# ----------------------------------------------------------------------------- shared data layer (also used by
# select_seg_examples.py, example_figures.py and case_figures.py)
@lru_cache(None)
def runs() -> pd.DataFrame:
    return pd.read_csv(METRICS / "runs.csv")


@lru_cache(None)
def metrics(task: str) -> pd.DataFrame:
    return pd.read_csv(METRICS / f"metrics_{task}.csv")


def run_row(task: str, cohort: str, bb: str, mode: str, method: str, label: str = "joint", seed: int = 42) -> pd.Series:
    r = runs()
    r = r[(r.task == task) & (r.cohort == cohort) & (r.backbone == bb) & (r["mode"] == mode) & (r.method == method)
          & (r.seed == seed) & (r.label == label)]
    if len(r) != 1:
        raise SystemExit(f"{task}/{cohort}/{bb}/{mode}/{method}/{label}: expected one run, got {len(r)}")
    return r.iloc[0]


def names_of(row: pd.Series) -> list[str]:
    """Label (task_index order) or structure (class_index order) names of a run, as in recompute.py."""
    rel = row.path.split("/", 2)[2]  # drop raw/<site>/
    if row.task == "seg":
        rel = rel.split("/", 1)[1]   # drop seg/
    ns = type("R", (), dict(task=row.task, rel=rel, label_run=row.label, cohort=row.cohort))
    return rc.label_names(ns)


def seg_percase(row: pd.Series) -> pd.DataFrame:
    """Held-out per-case Dice of a segmentation run, valid cases only, with a structure column."""
    d = pd.read_csv(RES / row.percase_file)
    d = d[(d.split == "test") & (d.valid == 1)].copy()
    nm = names_of(row)
    d["structure"] = [nm[int(c)] for c in d.class_index]
    return d


def cls_percase(row: pd.Series) -> pd.DataFrame:
    """Held-out per-case probabilities of a classification run, with a label-name column."""
    d = pd.read_csv(RES / row.percase_file)
    d = d[d.split == "test"].copy()
    nm = names_of(row)
    d["name"] = [nm[int(c)] if not pd.isna(c) else nm[0] for c in d.task_index]
    return d


def seg_case_mean(row: pd.Series) -> pd.Series:
    return seg_percase(row).groupby("case_id").dice.mean()


def pick(scores: pd.Series, q: float) -> str:
    """Case whose score is closest to the q-quantile (ties broken by case id)."""
    target = scores.quantile(q, interpolation="nearest")
    d = (scores - target).abs()
    return sorted(d[d == d.min()].index)[0]


def seg() -> dict:
    out = {}
    for ds in SEG_DATASETS:
        per_run, runs_ = {}, {}
        for bb in SEG_BACKBONES:
            for mt in SEG_METHODS:
                r = run_row("seg", ds, bb, "joint", mt)
                per_run[f"{bb}/{mt}"] = seg_case_mean(r)
                runs_[f"{bb}/{mt}"] = {"percase_file": r.percase_file, "best_iter": int(r.best_iter)}
        tab = pd.DataFrame(per_run).dropna()
        mean = tab.mean(axis=1)
        cases = {"success": pick(mean, 0.9), "failure": pick(mean, 0.0)}
        out[ds] = {"mode": "joint", "n_cases": int(len(tab)), "cases": cases, "runs": runs_,
                   "case_dice": {k: {c: round(float(tab.loc[c, k]), 4) for c in cases.values()} for k in tab.columns},
                   "case_mean": {c: round(float(mean[c]), 4) for c in cases.values()}}
    return out


def case_scores(d: pd.DataFrame) -> pd.Series:
    # FDG-PET/CT lists some patients with several studies under one case id; as in the draft (a dict keyed by case
    # id), the last row of a repeated id is kept
    d = d.drop_duplicates("case_id", keep="last")
    y = d.set_index("case_id").label.astype(float).astype(int)
    p = d.set_index("case_id").probability.astype(float)
    neg = np.sort(p[y == 0].values)
    pos_ids = y[y == 1].index
    return pd.Series({c: (np.searchsorted(neg, p[c], "left") + np.searchsorted(neg, p[c], "right")) / (2 * len(neg)) for c in pos_ids})


def cls() -> dict:
    out = {}
    t = metrics("cls")
    t = t[(t.split == "test") & (t.seed == 42)]
    for key, ds, label, bbs in CLS_SOURCES:
        if label is None:
            cand = t[(t.cohort == ds) & (t["mode"] == "pertask") & (t.method == "lin") & (t.backbone == bbs[0])]
            if key == "amos":
                cand = cand[cand.label.str.startswith("amos:")]
            cand = cand[cand.label != "NEGATIVE"].sort_values("label")
            label = cand.loc[cand.n_pos.idxmax(), "label"]
        per_run, runs_ = {}, {}
        for bb in bbs:
            for mt in CLS_METHODS:
                r = run_row("cls", ds, bb, "pertask", mt, label=label.split(":")[-1])
                per_run[f"{bb}/{mt}"] = case_scores(cls_percase(r))
                au = t[(t.cohort == ds) & (t.backbone == bb) & (t["mode"] == "pertask") & (t.method == mt) & (t.label == label)].auroc
                runs_[f"{bb}/{mt}"] = {"percase_file": r.percase_file, "auroc": round(float(au.iloc[0]), 4)}
        tab = pd.DataFrame(per_run).dropna()
        mean = tab.mean(axis=1)
        cases = {"success": pick(mean, 0.9), "failure": pick(mean, 0.0)}
        out[key] = {"dataset": ds, "label": label, "n_pos": int(len(tab)), "cases": cases, "runs": runs_,
                    "case_score": {k: {c: round(float(tab.loc[c, k]), 4) for c in cases.values()} for k in tab.columns},
                    "case_mean": {c: round(float(mean[c]), 4) for c in cases.values()}}
    return out


def compare_old(sel: dict) -> list[str]:
    old_f = RES / "examples" / "case_selection.json"
    if not old_f.is_file():
        return ["no draft selection to compare"]
    old = json.loads(old_f.read_text())
    lines = []
    for task in ("seg", "cls"):
        for k, v in sel[task].items():
            o = old[task].get(k)
            if o is None:
                lines.append(f"{task} {k}: not in draft selection"); continue
            same = o["cases"] == v["cases"]
            lines.append(f"{task} {k}: cases {'unchanged' if same else 'CHANGED'} {v['cases']}" + ("" if same else f" (draft {o['cases']})")
                         + (f"; label {v['label']}" + ("" if o.get('label') == v['label'] else f" (draft {o.get('label')})") if task == 'cls' else "")
                         + f"; n {v.get('n_cases', v.get('n_pos'))} (draft {o.get('n_cases', o.get('n_pos'))})")
            key = "case_dice" if task == "seg" else "case_score"
            diffs = [abs(v[key][r][c] - o[key][r][c]) for r in v[key] for c in v[key][r] if r in o[key] and c in o[key][r]]
            if diffs:
                lines.append(f"    max |v2 - draft| of the shown per-case values: {max(diffs):.4g} over {len(diffs)} values")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_SEL / "case_selection.json")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sel = {"seg": seg(), "cls": cls()}
    a.out.write_text(json.dumps(sel, indent=1))
    for k, v in sel["seg"].items():
        print("seg", k, v["cases"], v["case_mean"], "n", v["n_cases"])
    for k, v in sel["cls"].items():
        print("cls", k, v["label"], v["cases"], v["case_mean"], "n_pos", v["n_pos"])
    print("\n".join(compare_old(sel)))


if __name__ == "__main__":
    main()
