#!/usr/bin/env python3
"""Run inventory of the fm_adapt / fm_seg sweep from the files on disk (no cluster access, no trust in summaries).

Reads every run directory under results/raw/<site>/ (plus the two quarantine folders, listed but excluded), judges each
run from its own files (best-metric record, per-case file at the selected iteration, last checkpoint, metrics.json),
enumerates the intended design from the generated YAML trees (fm_adapt, fm_seg) and compares the two.

Outputs (midl_paper_v2/report/): run_inventory.csv, run_inventory.md, plus run_inventory_design.csv (design x found).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/g.yiasemis/projects/cvpr_ct_challenge")
RESULTS = PROJ / "midl_paper/results"
RAW = RESULTS / "raw"
QUARANTINE = [RESULTS / "raw_voco_stale_quarantine_20261027", RESULTS / "raw_dup_quarantine_20261029"]
FM_ADAPT = PROJ / "aifofomo/examples/fm_probes/fm_adapt"
FM_SEG = PROJ / "aifofomo/examples/fm_probes/fm_seg"
OUT = PROJ / "midl_paper_v2/report"
NUMBERS_TEX = PROJ / "midl_paper/paper/tables/numbers.tex"
SITES = ["snellius", "gaia"]
SEED_MODES = {"joint_s43": 43, "joint_s44": 44}
CLS_REVISION_METHODS = {"fullft_lowlr", "fullft_llrd", "fullft_accum", "mlpft", "rslora_lowlr"}
SEG_REVISION_METHODS = {"scratch"}
VOCO_RETIRED_BEFORE = datetime(2026, 10, 26)
MAIN_CLS_METHODS = ["lin", "attn", "lora", "lorag", "babmil", "fullft", "attnft", "rslora", "dora", "lin_then_lorag", "attn_then_lorag"]
MAIN_SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]


# ----------------------------------------------------------------------------- design (YAML trees)
def yaml_info(path: Path) -> dict:
    txt = path.read_text()
    m = re.search(r"^\s+num_iterations:\s*(\d+)", txt, re.M)
    s = re.search(r"^# seed:\s*(\d+)", txt, re.M)
    return {"num_iterations": int(m.group(1)) if m else None, "seed": int(s.group(1)) if s else 42}


def design_from_yaml() -> dict[tuple, dict]:
    """key = (task, cohort, backbone, mode, method, label) -> {seed, num_iterations, yaml, cls}"""
    design = {}
    for task, root in (("cls", FM_ADAPT), ("seg", FM_SEG)):
        for y in root.rglob("*.yaml"):
            rel = y.relative_to(root).parts
            if "eval" in rel or "sbatch" in rel or len(rel) != 5:
                continue  # totalseg eval/ yamls are the held-out re-evaluation configs of the same run, not runs
            cohort, backbone, mode, method, stem = rel
            label = "joint" if stem[:-5] == "config" else stem[:-5]
            info = yaml_info(y)
            seed = SEED_MODES.get(mode, info["seed"])
            cls = design_class(task, mode, method)
            design[(task, cohort, backbone, mode, method, label)] = {
                "seed": seed, "num_iterations": info["num_iterations"], "yaml": str(y.relative_to(PROJ)), "class": cls,
            }
    # fm_seg/generate.py enumerates runs whose YAML is absent from the tree (VoCo joint_lm/dec_then_lora); the runs exist and
    # the harvest uses them, so they belong to the design. Horizon = max num_iterations of the same (dataset, mode, method)
    # for the other backbones.
    try:
        sys.path.insert(0, str(FM_SEG))
        import generate as seg_gen  # noqa
        sys.path.pop(0)
        for d_, b_, m_, mt_, t_ in seg_gen.iter_runs():
            key = ("seg", d_, b_, m_, mt_, t_ or "joint")
            if key in design:
                continue
            sib = [v["num_iterations"] for k, v in design.items() if k[0] == "seg" and k[1] == d_ and k[3] == m_ and k[4] == mt_ and k[5] == key[5] and v["num_iterations"]]
            design[key] = {"seed": 42, "num_iterations": max(sib) if sib else None, "yaml": "MISSING (enumerated by fm_seg/generate.iter_runs, no YAML on disk)", "class": "main"}
    except Exception as e:  # noqa
        print("warning: fm_seg generate import failed:", e, file=sys.stderr)
    return design


def design_class(task: str, mode: str, method: str) -> str:
    if mode in SEED_MODES:
        return "seed_repeat"
    if mode == "pertask_cm":
        return "compute_matched"
    if (task == "cls" and method in CLS_REVISION_METHODS) or (task == "seg" and method in SEG_REVISION_METHODS):
        return "revision"
    return "main"


# ----------------------------------------------------------------------------- raw runs
def find_run_dirs(root: Path) -> list[Path]:
    """A run = directory that holds checkpoints/, val_case_metrics/ or logs/metrics.json."""
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if "checkpoints" in dirnames or "val_case_metrics" in dirnames or (p.name == "logs" and "metrics.json" in filenames):
            out.add(p.parent if p.name == "logs" else p)
            if p.name == "logs":
                continue
    return sorted(out)


def parse_rel(site_root: Path, run: Path) -> dict:
    parts = run.relative_to(site_root).parts
    d = {"group": "run", "task": None, "cohort": None, "backbone": None, "mode": None, "method": None, "label": None, "aux": ""}
    p = list(parts)
    if ".test_eval" in p:
        d["group"] = "aux"; d["aux"] = "test_eval_scratch"
        return d
    if p[0] == "voco_smoke":  # gaia/voco_smoke/cls/<cohort>/voco/<mode>/<method>/...
        d["group"] = "aux"; d["aux"] = "voco_smoke"; p = p[2:]; d["task"] = "cls"
    elif p[0] == "seg_gaia_ckpts":
        d["group"] = "aux"; d["aux"] = "seg_gaia_ckpts"; p = p[1:]; d["task"] = "seg"
    elif p[0] == "seg" and len(p) > 1 and p[1] == "voco_smoke":
        d["group"] = "aux"; d["aux"] = "seg_voco_smoke"; p = p[2:]; d["task"] = "seg"
    elif p[0] == "seg" and len(p) > 1 and p[1] == "examples":
        d["group"] = "aux"; d["aux"] = "seg_examples"; return d
    elif p[0] == "seg":
        d["task"] = "seg"; p = p[1:]
    elif p[0] == "revision":
        d["group"] = "aux"; d["aux"] = "revision_posthoc"; return d
    else:
        d["task"] = "cls"
    if len(p) < 5:
        d["group"] = "aux"; d["aux"] = d["aux"] or "unparsed"; return d
    d["cohort"], d["backbone"], d["mode"], d["method"] = p[:4]
    if len(p) == 5 and p[4] == "config":
        d["label"] = "joint"
    elif len(p) == 6 and p[4] == p[5]:
        d["label"] = p[4]
    else:
        d["group"] = "aux"; d["aux"] = d["aux"] or "unparsed"; d["label"] = "/".join(p[4:])
    return d


def read_best(exp: Path, metric: str) -> tuple[int | None, float | None, Path | None]:
    f = exp / "checkpoints" / "best" / f"best_val_metric_val_{metric}.txt"
    if not f.is_file():
        return None, None, None
    it = re.search(r"iteration:\s*(\d+)", f.read_text())
    bv = re.search(r"best_value:\s*([-0-9.eE+naN]+)", f.read_text())
    return (int(it.group(1)) if it else None), (float(bv.group(1)) if bv else None), f


def last_ckpt_iter(exp: Path) -> int | None:
    f = exp / "checkpoints" / "last_checkpoint.txt"
    if not f.is_file():
        return None
    m = re.search(r"model_(\d+)\.pt", f.read_text())
    return int(m.group(1)) if m else None


def metrics_json_summary(exp: Path, metric_key: str) -> tuple[int | None, int, int | None]:
    """(last iteration logged, number of validations = distinct iterations with the val metric, first val iteration)"""
    f = exp / "logs" / "metrics.json"
    if not f.is_file():
        return None, 0, None
    last = None
    vals = set()
    for line in f.open():
        m = re.search(r'"iteration":\s*(\d+)', line)
        if not m:
            continue
        it = int(m.group(1))
        last = it if last is None else max(last, it)
        if metric_key in line:
            vals.add(it)
    return last, len(vals), (min(vals) if vals else None)


def newest_mtime(exp: Path) -> tuple[datetime, datetime]:
    """(newest file mtime, oldest file mtime) over all files of the run directory."""
    ts = [Path(dp, fn).stat().st_mtime for dp, _, fns in os.walk(exp) for fn in fns]
    return datetime.fromtimestamp(max(ts)), datetime.fromtimestamp(min(ts))


def percase_summary(path: Path, task: str) -> dict:
    """Case / row counts per split at the per-case file + the headline test metric (macro AUROC or mean Dice)."""
    d = pd.read_csv(path)
    out = {"n_rows": len(d)}
    for sp in ("val", "test"):
        g = d[d.split == sp]
        out[f"n_{sp}_rows"] = len(g)
        out[f"n_{sp}_cases"] = g.case_id.nunique()
    if task == "cls":
        from sklearn.metrics import roc_auc_score

        aucs = []
        for _, g in d[d.split == "test"].groupby("task_index"):
            if g.label.nunique() == 2:
                aucs.append(roc_auc_score(g.label, g.probability))
        out["test_metric"] = float(np.mean(aucs)) if aucs else float("nan")
        out["n_labels_scored"] = len(aucs)
        out["n_labels"] = d.task_index.nunique()
    else:
        g = d[(d.split == "test") & (d.valid == 1)]
        out["test_metric"] = float(g.dice.mean()) if len(g) else float("nan")
        out["n_labels"] = d.class_index.nunique()
        out["n_labels_scored"] = g.class_index.nunique()
    return out


def inventory_run(site: str, site_root: Path, exp: Path, design: dict, excluded: bool) -> dict:
    meta = parse_rel(site_root, exp)
    task = meta["task"]
    metric = "auroc" if task == "cls" else "dice"
    percase_name = "logits.csv" if task == "cls" else "dice.csv"
    row = {
        "site": site, "excluded": "yes" if excluded else "", "group": meta["group"], "aux": meta["aux"], "task": task,
        "cohort": meta["cohort"], "backbone": meta["backbone"], "mode": meta["mode"], "method": meta["method"],
        "label": meta["label"], "seed": "", "design_class": "", "in_design": "", "num_iterations_design": "",
        "status": "", "status_detail": "", "best_iter": "", "best_val_metric": "", "n_validations_percase": 0,
        "n_validations_logged": 0, "last_iter_logged": "", "last_ckpt_iter": "", "percase_file_at_best": "",
        "n_test_cases": "", "n_val_cases": "", "n_test_rows": "", "n_labels_scored": "", "test_metric": "",
        "test_eval_done": "yes" if (exp / "test_eval_done.txt").is_file() else "",
        "newest_file": "", "oldest_file": "", "path": str(exp.relative_to(RESULTS)), "dup_other_site": "",
    }
    key = (task, meta["cohort"], meta["backbone"], meta["mode"], meta["method"], meta["label"])
    des = design.get(key) if meta["group"] == "run" else None
    if des:
        row["in_design"] = "yes"; row["design_class"] = des["class"]; row["seed"] = des["seed"]
        row["num_iterations_design"] = des["num_iterations"]
    else:
        row["in_design"] = "no" if meta["group"] == "run" else ""
        row["seed"] = SEED_MODES.get(meta["mode"] or "", 42)
    if task is None:
        row["status"] = "not_a_run"
        nf, of = newest_mtime(exp)
        row["newest_file"], row["oldest_file"] = nf.strftime("%Y-%m-%d %H:%M"), of.strftime("%Y-%m-%d %H:%M")
        return row
    best_it, best_val, best_f = read_best(exp, metric)
    last_ck = last_ckpt_iter(exp)
    last_logged, n_val_logged, _ = metrics_json_summary(exp, f"val/metric/val_{metric}")
    vcm = exp / "val_case_metrics"
    iters = sorted(int(p.name[5:]) for p in vcm.glob("iter_*") if (p / percase_name).is_file()) if vcm.is_dir() else []
    row.update({
        "best_iter": best_it if best_it is not None else "", "best_val_metric": f"{best_val:.6f}" if best_val is not None else "",
        "n_validations_percase": len(iters), "n_validations_logged": n_val_logged,
        "last_iter_logged": last_logged if last_logged is not None else "", "last_ckpt_iter": last_ck if last_ck is not None else "",
    })
    nf, of = newest_mtime(exp)
    row["newest_file"], row["oldest_file"] = nf.strftime("%Y-%m-%d %H:%M"), of.strftime("%Y-%m-%d %H:%M")
    if best_it is None:
        row["status"] = "failed/empty"
        row["status_detail"] = "no best-metric record" + (f"; {len(iters)} per-case files; last ckpt {last_ck}" if (iters or last_ck) else "; no checkpoints, no per-case files")
        return row
    pc = vcm / f"iter_{best_it:06d}" / percase_name
    row["percase_file_at_best"] = "yes" if pc.is_file() else "MISSING"
    if not pc.is_file():
        row["status"] = "partial"; row["status_detail"] = f"no {percase_name} at best iteration {best_it}"
        return row
    s = percase_summary(pc, task)
    row.update({"n_test_cases": s["n_test_cases"], "n_val_cases": s["n_val_cases"], "n_test_rows": s["n_test_rows"],
                "n_labels_scored": s["n_labels_scored"], "test_metric": f"{s['test_metric']:.6f}"})
    final = des["num_iterations"] if des else None
    problems = []
    if s["n_test_rows"] == 0:
        problems.append("no test rows at best iteration")
    if final is not None:
        if last_ck is None or last_ck < final:
            problems.append(f"last checkpoint {last_ck} < num_iterations {final}")
    else:
        if last_ck is None or last_logged is None or last_ck < last_logged:
            problems.append(f"horizon unknown (not in design); last ckpt {last_ck}, last logged {last_logged}")
    if problems:
        row["status"] = "partial"; row["status_detail"] = "; ".join(problems)
    else:
        row["status"] = "finished" if des else "finished(no yaml)"
    return row


# ----------------------------------------------------------------------------- main
def main() -> None:
    design = design_from_yaml()
    rows: list[dict] = []
    for site in SITES:
        root = RAW / site
        for exp in find_run_dirs(root):
            rows.append(inventory_run(site, root, exp, design, excluded=False))
    quarantine_rows: list[dict] = []
    for q in QUARANTINE:
        for site in SITES:
            root = q / site
            if root.is_dir():
                for exp in find_run_dirs(root):
                    quarantine_rows.append(inventory_run(site, root, exp, design, excluded=True))
    # duplicates across sites (same relative run path under raw/<site>/)
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["group"] == "run":
            by_rel["/".join(r["path"].split("/")[2:])].append(r)
    for rel, rs in by_rel.items():
        if len(rs) > 1:
            for r in rs:
                others = [o for o in rs if o is not r]
                r["dup_other_site"] = ";".join(f"{o['site']}:{o['status']}" for o in others)
    all_rows = rows + quarantine_rows
    cols = list(all_rows[0].keys())
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)

    # ---------------------------------------------------------------- design vs found
    found: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["group"] == "run":
            found[(r["task"], r["cohort"], r["backbone"], r["mode"], r["method"], r["label"])].append(r)
    design_rows = []
    for key, des in sorted(design.items()):
        rs = found.get(key, [])
        statuses = [r["status"] for r in rs]
        if any(s == "finished" for s in statuses):
            st = "finished"
        elif rs:
            st = "partial" if any(s == "partial" for s in statuses) else "failed/empty"
        else:
            st = "MISSING"
        design_rows.append({
            "task": key[0], "cohort": key[1], "backbone": key[2], "mode": key[3], "method": key[4], "label": key[5],
            "seed": des["seed"], "class": des["class"], "num_iterations": des["num_iterations"], "status": st,
            "n_copies": len(rs), "sites": "+".join(f"{r['site']}:{r['status']}" for r in rs),
            "n_finished_copies": sum(s == "finished" for s in statuses), "yaml": des["yaml"],
        })
    with (OUT / "run_inventory_design.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(design_rows[0].keys())); w.writeheader(); w.writerows(design_rows)
    extra = [r for r in rows if r["group"] == "run" and r["in_design"] == "no"]
    dups = [(rel, rs) for rel, rs in by_rel.items() if len(rs) > 1]
    failed = [r for r in rows if r["group"] == "run" and r["status"] in ("partial", "failed/empty")]

    # ---------------------------------------------------------------- eval-mismatch check (test case counts)
    mism = []
    fam = lambda m: "pertask_cm" if m == "pertask_cm" else ("pertask" if m == "pertask" else "joint")  # noqa: E731
    grp: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["group"] == "run" and r["status"].startswith("finished") and r["n_test_cases"] != "":
            grp[(r["task"], r["cohort"], fam(r["mode"]), r["label"] if fam(r["mode"]) != "joint" else "joint")].append(r)
    for key, rs in sorted(grp.items()):
        c = Counter(int(r["n_test_cases"]) for r in rs)
        modal = c.most_common(1)[0][0]
        for r in rs:
            if int(r["n_test_cases"]) != modal:
                mism.append((key, r["path"], int(r["n_test_cases"]), modal, len(rs)))
    # same check per label on the number of test rows (joint cls runs: rows = cases x labels)
    rowmism = []
    for key, rs in sorted(grp.items()):
        c = Counter(int(r["n_test_rows"]) for r in rs)
        modal = c.most_common(1)[0][0]
        for r in rs:
            if int(r["n_test_rows"]) != modal:
                rowmism.append((key, r["path"], int(r["n_test_rows"]), modal, len(rs)))

    # ---------------------------------------------------------------- cross-checks
    master_src = {}
    for fn in ("master_metrics.csv", "master_metrics_seg.csv"):
        p = RESULTS / fn
        if p.is_file():
            master_src[fn] = set(pd.read_csv(p, usecols=["source_file"]).source_file.unique())
    master_missing_files = {fn: sorted(s for s in srcs if not (RAW / s).is_file()) for fn, srcs in master_src.items()}
    master_runs = {fn: {"/".join(s.split("/")[1:-3]) for s in srcs} for fn, srcs in master_src.items()}  # strip site + val_case_metrics/iter/file
    finished_main = {("/".join(r["path"].split("/")[2:])) for r in rows if r["group"] == "run" and r["status"] == "finished" and r["design_class"] == "main"}
    finished_main_cls = {p for p in finished_main if not p.startswith("seg/")}
    finished_main_seg = {p for p in finished_main if p.startswith("seg/")}
    in_master_not_finished = {fn: sorted(master_runs[fn] - (finished_main_cls if "seg" not in fn else finished_main_seg)) for fn in master_runs}
    finished_not_in_master = {fn: sorted((finished_main_cls if "seg" not in fn else finished_main_seg) - master_runs[fn]) for fn in master_runs}
    # master runs that are only in revision classes (seed/revision) are expected to be absent from master
    rev_src = set()
    p = RESULTS / "revision_metrics.csv"
    if p.is_file():
        rev_src = set(pd.read_csv(p, usecols=["source_file"]).source_file.unique())
    rev_runs = {"/".join(s.split("/")[1:-3]) for s in rev_src}
    finished_rev = {("/".join(r["path"].split("/")[2:])) for r in rows if r["group"] == "run" and r["status"] == "finished" and r["design_class"] in ("seed_repeat", "revision", "compute_matched")}

    old_inv = {}
    old_disagree = {}
    mine_by_rel = {}
    for r in rows:
        if r["group"] == "run":
            rel = "/".join(r["path"].split("/")[2:])
            prev = mine_by_rel.get(rel)
            if prev is None or (r["status"] == "finished" and prev["status"] != "finished"):
                mine_by_rel[rel] = r
    for fn in ("runs_inventory.csv", "runs_inventory_seg.csv"):
        p = RESULTS / fn
        if p.is_file():
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
            old_inv[fn] = Counter(d.status)
            dis = []
            for _, o in d.iterrows():
                rel = o["rel"] if fn.endswith("seg.csv") else o["rel"]
                mine = mine_by_rel.get(rel)
                ms = mine["status"] if mine else "not on disk"
                ok = (o["status"] == "ok" and ms == "finished") or (o["status"] == "MISSING" and ms == "not on disk") or (o["status"] == "INCOMPLETE" and ms in ("partial", "failed/empty"))
                if not ok:
                    dis.append((rel, o["status"], o.get("reason", ""), ms, mine["site"] if mine else ""))
            old_disagree[fn] = dis
    fresh = {}
    for fn in ("master_metrics.csv", "master_metrics_seg.csv", "revision_metrics.csv", "runs_inventory.csv", "runs_inventory_seg.csv"):
        p = RESULTS / fn
        if p.is_file():
            fresh[fn] = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    if NUMBERS_TEX.is_file():
        fresh["paper/tables/numbers.tex"] = datetime.fromtimestamp(NUMBERS_TEX.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    newest_run = {}
    for r in rows:
        if r["group"] == "run":
            k = (r["task"], r["site"])
            newest_run[k] = max(newest_run.get(k, ""), r["newest_file"])
    macros = {}
    if NUMBERS_TEX.is_file():
        for m in re.finditer(r"\\newcommand\{\\(numRunsCls|numSegRunsN|numSeedsclsRunsN|numSeedssegRunsN|numVocoRunsClsN|numVocoRunsClsDoneN|numVocoRunsSegN|numVocoRunsSegDoneN)\}\{(\d+)\}", NUMBERS_TEX.read_text()):
            macros[m.group(1)] = int(m.group(2))

    # ---------------------------------------------------------------- date anomalies
    date_flags = []
    for r in rows:
        if r["group"] != "run":
            continue
        newest = datetime.strptime(r["newest_file"], "%Y-%m-%d %H:%M")
        if r["backbone"] == "voco" and newest < VOCO_RETIRED_BEFORE:
            date_flags.append(("voco run with files older than 2026-10-26 (retired design)", r["path"], r["newest_file"], r["status"]))
    # generator-implied design that has no YAML (fm_seg iter_revision_runs enumerates VoCo seed repeats / pertask_cm)
    gen_note = []
    try:
        sys.path.insert(0, str(FM_SEG))
        import generate as seg_gen  # noqa

        sys.path.pop(0)
        gen_rev = {("seg", d, b, m, mt, t or "joint") for d, b, m, mt, t in seg_gen.iter_revision_runs()}
        gen_main = {("seg", d, b, m, mt, t or "joint") for d, b, m, mt, t in seg_gen.iter_runs()}
        yaml_seg = {k for k in design if k[0] == "seg"}
        gen_note.append(f"fm_seg generate.iter_runs(): {len(gen_main)} runs; YAML tree (main class): {sum(1 for k, v in design.items() if k[0] == 'seg' and v['class'] == 'main')}; in generator but no YAML: {len(gen_main - yaml_seg)}; YAML but not in generator main: {len({k for k, v in design.items() if k[0] == 'seg' and v['class'] == 'main'} - gen_main)}")
        no_yaml = sorted(gen_rev - yaml_seg)
        gen_note.append(f"fm_seg generate.iter_revision_runs(): {len(gen_rev)} runs; with a YAML on disk: {len(gen_rev & yaml_seg)}; without YAML (never generated/submitted): {len(no_yaml)} -> " + Counter((k[2], k[3]) for k in no_yaml).__repr__())
    except Exception as e:  # noqa
        gen_note.append(f"could not import fm_seg/generate.py: {e}")

    write_report(rows, quarantine_rows, design, design_rows, extra, dups, failed, mism, rowmism, master_src, master_missing_files,
                 in_master_not_finished, finished_not_in_master, rev_runs, finished_rev, old_inv, macros, date_flags, gen_note,
                 old_disagree, fresh, newest_run)
    print(f"wrote {OUT/'run_inventory.csv'} ({len(all_rows)} rows), {OUT/'run_inventory_design.csv'} ({len(design_rows)} rows), {OUT/'run_inventory.md'}")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(rows, qrows, design, design_rows, extra, dups, failed, mism, rowmism, master_src, master_missing_files,
                 in_master_not_finished, finished_not_in_master, rev_runs, finished_rev, old_inv, macros, date_flags, gen_note,
                 old_disagree, fresh, newest_run) -> None:
    runs = [r for r in rows if r["group"] == "run"]
    aux = [r for r in rows if r["group"] != "run"]
    D = pd.DataFrame(design_rows)
    R = pd.DataFrame(runs)
    L = []
    L.append("# Run inventory of the fm_adapt / fm_seg sweep (files on disk, 2026-10-31)\n")
    L.append(f"Source: `{RAW}` (rsync mirror of both sites; never deletes). Design: YAML trees under `{FM_ADAPT}` and `{FM_SEG}` "
             "(one YAML per run; totalseg `eval/` YAMLs are held-out re-evaluations of the same run and are not counted). "
             "Script: `midl_paper_v2/report/scripts/build_run_inventory.py`. Tables: `run_inventory.csv` (one row per run directory, incl. quarantine), "
             "`run_inventory_design.csv` (one row per designed run).\n")
    L.append("Status rule (per run directory, from its own files): **finished** = `checkpoints/best/best_val_metric_val_<auroc|dice>.txt` exists, "
             "`val_case_metrics/iter_<best>/<logits|dice>.csv` exists and has test rows, and `checkpoints/last_checkpoint.txt` reached the YAML's "
             "`num_iterations`; **partial** = best record exists but one of those fails; **failed/empty** = no best record. "
             "This is the same selection the harvest uses (best validation macro metric, per-case file of that iteration).\n")

    # totals
    L.append("## 1. Totals\n")
    tot = R.groupby(["task", "status"]).size().unstack(fill_value=0)
    L.append(md_table(tot.reset_index()))
    L.append("")
    L.append(f"Run directories under raw/: {len(runs)} runs + {len(aux)} auxiliary directories (smoke tests, scratch evals; listed in section 6). "
             f"Quarantined run directories (excluded): {len(qrows)}.\n")
    L.append("Per site:\n")
    L.append(md_table(R.groupby(["site", "task", "status"]).size().unstack(fill_value=0).reset_index()))
    L.append("\nFinished runs by task, backbone and design class (finished copies on either site, deduplicated by run path):\n")
    Rf = R[R.status == "finished"].copy()
    Rf["rel"] = Rf.path.str.split("/").str[2:].str.join("/")
    Rfd = Rf.drop_duplicates("rel")
    L.append(md_table(Rfd.groupby(["task", "backbone", "design_class"]).size().unstack(fill_value=0).reset_index()))
    L.append("")

    # design vs found
    L.append("## 2. Design vs found\n")
    L.append(f"Designed runs (YAML): {len(D)} = cls {int((D.task=='cls').sum())} + seg {int((D.task=='seg').sum())}.\n")
    L.append(md_table(D.groupby(["task", "class", "status"]).size().unstack(fill_value=0).reset_index()))
    L.append("\nBy task / backbone / class:\n")
    L.append(md_table(D.groupby(["task", "backbone", "class", "status"]).size().unstack(fill_value=0).reset_index()))
    L.append("\n### 2a. Compact count table: finished / designed per (task, cohort, backbone, mode); columns = methods\n")
    for task in ("cls", "seg"):
        Dt = D[D.task == task]
        methods = [m for m in (MAIN_CLS_METHODS if task == "cls" else MAIN_SEG_METHODS) if m in set(Dt.method)]
        methods += sorted(set(Dt.method) - set(methods))
        g = Dt.groupby(["cohort", "backbone", "mode", "method"]).agg(n=("status", "size"), f=("status", lambda s: (s == "finished").sum()))
        tbl = []
        for (c, b, m), sub in g.groupby(level=[0, 1, 2]):
            row = {"cohort": c, "backbone": b, "mode": m}
            tot_n = tot_f = 0
            for mt in methods:
                if (c, b, m, mt) in sub.index:
                    n, f = sub.loc[(c, b, m, mt)]
                    row[mt] = f"{f}/{n}" if f != n else str(n)
                    tot_n += n; tot_f += f
                else:
                    row[mt] = "-"
            row["total"] = f"{tot_f}/{tot_n}"
            tbl.append(row)
        L.append(f"\n**{task}** (cell = finished/designed; a bare number means all finished; `-` = not in the design)\n")
        L.append(md_table(pd.DataFrame(tbl)))
    noyaml = D[D.yaml.str.startswith("MISSING")]
    L.append(f"\nDesign entries without a YAML on disk (enumerated by `fm_seg/generate.iter_runs()`, runs exist): {len(noyaml)}\n")
    for _, r in noyaml.iterrows():
        L.append(f"- {r.task} {r.cohort}/{r.backbone}/{r['mode']}/{r.method}/{r.label}: {r.status} ({r.sites}); horizon taken from sibling backbones = {r.num_iterations}")
    # missing
    miss = D[D.status == "MISSING"]
    L.append(f"\n### 2b. MISSING designed runs ({len(miss)})\n")
    if len(miss):
        L.append(md_table(miss.groupby(["task", "cohort", "backbone", "mode", "method", "class"]).size().reset_index(name="n_missing")))
        L.append("\nFull list:\n")
        for _, r in miss.iterrows():
            L.append(f"- {r.task} {r.cohort}/{r.backbone}/{r['mode']}/{r.method}/{r.label} (seed {r.seed}, {r['class']})")
    else:
        L.append("None.")
    notfin = D[D.status.isin(["partial", "failed/empty"])]
    L.append(f"\n### 2c. Designed runs present but not finished ({len(notfin)})\n")
    for _, r in notfin.iterrows():
        L.append(f"- {r.task} {r.cohort}/{r.backbone}/{r['mode']}/{r.method}/{r.label}: {r.status} ({r.sites})")
    if not len(notfin):
        L.append("None.")
    L.append(f"\n### 2d. EXTRA run directories not in the YAML design ({len(extra)})\n")
    for r in extra:
        L.append(f"- {r['path']}: status {r['status']} ({r['status_detail']}); best iter {r['best_iter']}, newest file {r['newest_file']}")
    if not extra:
        L.append("None.")
    L.append(f"\n### 2e. DUPLICATED runs (same run path on both sites) ({len(dups)})\n")
    L.append("| run | snellius | gaia | best iter (sn/ga) | best val (sn/ga) | test metric (sn/ga) | delta test | agree? |\n|---|---|---|---|---|---|---|---|")
    for rel, rs in sorted(dups):
        bs = {r["site"]: r for r in rs}
        sn, ga = bs.get("snellius"), bs.get("gaia")
        def tm(r):
            return r["test_metric"] if r and r["test_metric"] else "-"
        try:
            delta = abs(float(sn["test_metric"]) - float(ga["test_metric"]))
            agree = "identical" if delta < 1e-9 else ("same within 0.01" if delta < 0.01 else "DIFFERENT (separate trainings)")
            dtxt = f"{delta:.4f}"
        except (TypeError, ValueError, KeyError):
            delta, agree, dtxt = None, "n/a (one copy unfinished)", "-"
        L.append(f"| {rel} | {sn['status'] if sn else '-'} | {ga['status'] if ga else '-'} | {sn['best_iter'] if sn else '-'}/{ga['best_iter'] if ga else '-'} | "
                 f"{sn['best_val_metric'] if sn else '-'}/{ga['best_val_metric'] if ga else '-'} | {tm(sn)}/{tm(ga)} | {dtxt} | {agree} |")
    if not dups:
        L.append("| none | | | | | | | |")
    L.append(f"\n### 2f. FAILED / partial run directories ({len(failed)})\n")
    for r in failed:
        L.append(f"- {r['path']}: **{r['status']}** - {r['status_detail']}; validations {r['n_validations_percase']} (per-case) / {r['n_validations_logged']} (logged); "
                 f"last logged {r['last_iter_logged']}, last ckpt {r['last_ckpt_iter']}; newest file {r['newest_file']}; dup on other site: {r['dup_other_site'] or 'no'}")
    if not failed:
        L.append("None.")

    # per-case checks
    L.append("\n## 3. Per-case files behind the headline numbers\n")
    fin = R[R.status.str.startswith("finished")]
    L.append(f"Every finished run has `val_case_metrics/iter_<best>/<logits|dice>.csv` with test rows by construction of the status rule "
             f"({len(fin)} finished run directories). Runs whose best-iteration per-case file is missing are listed as partial in 2f "
             f"(count: {int((R.percase_file_at_best == 'MISSING').sum())}).\n")
    L.append("Test-case coverage: for every group (task, cohort, mode family, label) the number of unique test case ids of each finished run is compared "
             "with the group's modal value. Mode family: pertask / pertask_cm / joint (joint, joint_lm, joint_s43, joint_s44 share the case set).\n")
    cov = fin.copy()
    cov["fam"] = cov["mode"].map(lambda m: "pertask_cm" if m == "pertask_cm" else ("pertask" if m == "pertask" else "joint"))
    cov["lab"] = np.where(cov.fam == "joint", "joint", cov.label)
    cov_tbl = cov.groupby(["task", "cohort", "fam"]).agg(runs=("path", "size"), n_test_cases_min=("n_test_cases", lambda s: int(min(map(int, s)))),
                                                         n_test_cases_max=("n_test_cases", lambda s: int(max(map(int, s)))),
                                                         n_val_cases_min=("n_val_cases", lambda s: int(min(map(int, s)))),
                                                         n_val_cases_max=("n_val_cases", lambda s: int(max(map(int, s))))).reset_index()
    L.append(md_table(cov_tbl))
    L.append(f"\nRuns whose unique-test-case count differs from the modal value of their group: {len(mism)}\n")
    for key, path, n, modal, k in mism:
        L.append(f"- {path}: {n} test cases vs modal {modal} (group {key}, {k} runs)")
    L.append(f"\nRuns whose number of test rows differs from the modal value of their group: {len(rowmism)}\n")
    for key, path, n, modal, k in rowmism:
        L.append(f"- {path}: {n} test rows vs modal {modal} (group {key}, {k} runs)")
    if not rowmism:
        L.append("None.")
    L.append("\nCross-check against the harvest outputs actually used by the draft (`results/master_metrics.csv`, `results/master_metrics_seg.csv`, "
             "`results/revision_metrics.csv`): do their `source_file` per-case files exist, and does their run set match the finished main-design runs here?\n")
    for fn, srcs in master_src.items():
        L.append(f"- `{fn}`: {len(srcs)} source files; missing on disk: {len(master_missing_files[fn])}" + (" -> " + "; ".join(master_missing_files[fn][:10]) if master_missing_files[fn] else ""))
        nm = in_master_not_finished[fn]
        rev_in = [x for x in nm if x.split("/")[-2] in CLS_REVISION_METHODS | SEG_REVISION_METHODS or "/joint_s4" in x or "/pertask_cm/" in x]
        other = [x for x in nm if x not in rev_in]
        L.append(f"  - runs in `{fn}` that are not finished *main-design* runs here: {len(nm)}; of these {len(rev_in)} are revision-method runs the harvest "
                 f"also enumerates (fine), and {len(other)} are something else" + (" -> " + "; ".join(other) if other else ""))
        L.append(f"  - finished main-design runs here that are not in `{fn}`: {len(finished_not_in_master[fn])}" + (" -> " + "; ".join(finished_not_in_master[fn][:40]) if finished_not_in_master[fn] else ""))
    L.append(f"- `revision_metrics.csv`: {len(rev_runs)} runs; finished seed/revision/compute-matched runs here: {len(finished_rev)}; "
             f"in revision_metrics but not finished here: {len(rev_runs - finished_rev)}" + (" -> " + "; ".join(sorted(rev_runs - finished_rev)[:40]) if rev_runs - finished_rev else "") +
             f"; finished here but not in revision_metrics: {len(finished_rev - rev_runs)}" + (" -> " + "; ".join(sorted(finished_rev - rev_runs)[:40]) if finished_rev - rev_runs else ""))

    # macros
    L.append("\n## 4. Comparison with the draft's macros and the old inventories\n")
    Dm = D[D["class"] == "main"]
    cnt = {
        "numRunsCls (main cls design, no VoCo, no breastdivider)": (
            int(((Dm.task == "cls") & (Dm.backbone != "voco") & (Dm.cohort != "breastdivider")).sum()),
            int(((Dm.task == "cls") & (Dm.backbone != "voco") & (Dm.cohort != "breastdivider") & (Dm.status == "finished")).sum())),
        "numVocoRunsClsN (main cls VoCo)": (int(((Dm.task == "cls") & (Dm.backbone == "voco")).sum()), int(((Dm.task == "cls") & (Dm.backbone == "voco") & (Dm.status == "finished")).sum())),
        "breastdivider cls (not in the paper)": (int(((Dm.task == "cls") & (Dm.cohort == "breastdivider")).sum()), int(((Dm.task == "cls") & (Dm.cohort == "breastdivider") & (Dm.status == "finished")).sum())),
        "numSegRunsN (main seg design, no VoCo)": (int(((Dm.task == "seg") & (Dm.backbone != "voco")).sum()), int(((Dm.task == "seg") & (Dm.backbone != "voco") & (Dm.status == "finished")).sum())),
        "numVocoRunsSegN (main seg VoCo)": (int(((Dm.task == "seg") & (Dm.backbone == "voco")).sum()), int(((Dm.task == "seg") & (Dm.backbone == "voco") & (Dm.status == "finished")).sum())),
        "numSeedsclsRunsN (cls seed repeats)": (int(((D.task == "cls") & (D["class"] == "seed_repeat")).sum()), int(((D.task == "cls") & (D["class"] == "seed_repeat") & (D.status == "finished")).sum())),
        "numSeedssegRunsN (seg seed repeats)": (int(((D.task == "seg") & (D["class"] == "seed_repeat")).sum()), int(((D.task == "seg") & (D["class"] == "seed_repeat") & (D.status == "finished")).sum())),
        "cls revision methods (joint)": (int(((D.task == "cls") & (D["class"] == "revision")).sum()), int(((D.task == "cls") & (D["class"] == "revision") & (D.status == "finished")).sum())),
        "seg compute-matched (pertask_cm)": (int(((D.task == "seg") & (D["class"] == "compute_matched")).sum()), int(((D.task == "seg") & (D["class"] == "compute_matched") & (D.status == "finished")).sum())),
        "seg scratch": (int(((D.task == "seg") & (D["class"] == "revision")).sum()), int(((D.task == "seg") & (D["class"] == "revision") & (D.status == "finished")).sum())),
    }
    L.append("| quantity | designed (YAML) | finished | draft macro |\n|---|---|---|---|")
    mac = {"numRunsCls": "numRunsCls", "numVocoRunsClsN": "numVocoRunsClsN", "numSegRunsN": "numSegRunsN", "numVocoRunsSegN": "numVocoRunsSegN",
           "numSeedsclsRunsN": "numSeedsclsRunsN", "numSeedssegRunsN": "numSeedssegRunsN"}
    for k, (n, f) in cnt.items():
        name = k.split(" ")[0]
        mv = macros.get(name)
        extra_m = ""
        if name == "numVocoRunsClsN":
            extra_m = f" (Done: {macros.get('numVocoRunsClsDoneN')})"
        if name == "numVocoRunsSegN":
            extra_m = f" (Done: {macros.get('numVocoRunsSegDoneN')})"
        L.append(f"| {k} | {n} | {f} | {mv if mv is not None else '-'}{extra_m} |")
    L.append("\nOld inventories (`results/runs_inventory.csv`, `results/runs_inventory_seg.csv`), status counts as written there:\n")
    for fn, c in old_inv.items():
        L.append(f"- `{fn}`: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())) + f" (total {sum(c.values())})")
    for fn, dis in old_disagree.items():
        L.append(f"\nPer-run status of `{fn}` vs this inventory (same run path; ok<->finished, MISSING<->not on disk, INCOMPLETE<->partial/failed): {len(dis)} disagreements")
        shown = [x for x in dis if x[3] != "not on disk"]
        hidden = len(dis) - len(shown)
        for rel, ost, reason, ms, site in shown[:60]:
            L.append(f"- {rel}: old `{ost}` ({reason}) vs here `{ms}` ({site})")
        if hidden:
            L.append(f"- ... plus {hidden} rows the old inventory lists as MISSING and which exist neither on disk nor in the YAML design "
                     "(combinations the harvest enumerates from `generate.METHODS` incl. the revision methods for every mode/task)")
    L.append("\nFreshness: harvest outputs vs the newest run file on disk\n")
    for fn, t in fresh.items():
        L.append(f"- `{fn}`: written {t}")
    for (task, site), t in sorted(newest_run.items()):
        L.append(f"- newest run file, {task} on {site}: {t}")
    L.append("\nNotes on the differences: `runs_inventory.csv` enumerates `generate.METHODS`, which since 2026-10-26 includes the five revision methods, "
             "for every mode and task, so its 'expected' list (1792) contains 1792 - 1232 = 560 combinations that were never designed as YAMLs "
             "(revision methods per task, revision methods for VoCo/breastdivider/joint_lm); most of its 513 MISSING rows are those. It also knows no seed modes.\n")
    L.append("\n".join(f"- {n}" for n in gen_note))

    # dates
    L.append("\n## 5. File dates and anomalies\n")
    nd = R.copy()
    nd["day"] = nd.newest_file.str[:10]
    L.append("Newest-file date of the run directories, by backbone (number of runs):\n")
    L.append(md_table(nd.groupby(["backbone", "day"]).size().unstack(fill_value=0).T.reset_index().rename(columns={"index": "day"})))
    L.append(f"\nVoCo runs with files older than 2026-10-26 (retired design) under raw/: {len(date_flags)}\n")
    for why, path, when, st in date_flags:
        L.append(f"- {path}: newest file {when}, status {st}")
    if not date_flags:
        L.append("None under raw/. The stale VoCo runs are all in `raw_voco_stale_quarantine_20261027` (section 7).")
    # other anomalies
    L.append("\nOther anomalies:\n")
    odd = []
    single = Counter()
    partial_pc = []
    for r in runs:
        if not r["status"].startswith("finished"):
            continue
        if r["num_iterations_design"] != "" and r["last_ckpt_iter"] != "" and int(r["last_ckpt_iter"]) > int(r["num_iterations_design"]):
            odd.append(f"- {r['path']}: last checkpoint {r['last_ckpt_iter']} (best {r['best_iter']}, {r['n_validations_percase']} validations) beyond the YAML's num_iterations "
                       f"{r['num_iterations_design']}: the YAML on disk is older than the run (stale design file; the run follows the later two-stage joint_lm horizon)")
        if r["n_validations_percase"] != r["n_validations_logged"]:
            if r["n_validations_percase"] == 1:
                single[(r["task"], r["site"], r["cohort"])] += 1
            else:
                partial_pc.append(f"- {r['path']}: {r['n_validations_percase']} per-case folders vs {r['n_validations_logged']} validations logged (per-case files of some iterations absent at the source; the best iteration {r['best_iter']} is present)")
    L.append("\n".join(odd) if odd else "- none beyond the design horizon")
    L.append(f"- {sum(single.values())} finished seg runs keep the per-case dice.csv of a single iteration (the selected one) although metrics.json logs all validations: "
             "held-out re-inference at the selected checkpoint (`fm_seg/reinfer_dice.py`); by (task, site, dataset): " + ", ".join(f"{t}/{s}/{c}: {n}" for (t, s, c), n in sorted(single.items())))
    L.extend(partial_pc)
    nv = Counter((r["task"], r["mode"], r["n_validations_percase"]) for r in runs if r["status"] == "finished")
    L.append("\nNumber of per-case validation folders of finished runs, by (task, mode): " + "; ".join(f"{t}/{m}: {n}x{c}" for (t, m, n), c in sorted(nv.items())))

    # quarantine
    L.append("\n## 6. Auxiliary directories under raw/ (not runs of the design)\n")
    for r in aux:
        L.append(f"- {r['path']} [{r['aux']}]: status {r['status']}, best iter {r['best_iter']}, newest {r['newest_file']}")
    L.append("\nOther non-run content under raw/: `raw/<site>/revision/{features,nsd,nsd_scratch,test}` = post-hoc linear-probe features (.npz), NSD re-evaluations "
             "(nsd.csv per run) and their logs; `raw/gaia/seg_examples_scratch`, `raw/gaia/seg/examples` = example-figure scratch (logs only); "
             "`raw/<site>/seg/SITE_SPLIT.txt` = site split note (Gaia: amos22/tapct/pertask).")
    L.append(f"\n## 7. Quarantined run directories (excluded from everything above): {len(qrows)}\n")
    Q = pd.DataFrame(qrows)
    if len(Q):
        Q["root"] = Q.path.str.split("/").str[0]
        L.append(md_table(Q.groupby(["root", "site", "task", "cohort", "backbone", "status"]).size().reset_index(name="n")))
        rel_raw = {"/".join(r["path"].split("/")[2:]): r for r in runs}
        only_q, same_copy, superseded = [], [], []
        for _, q in Q.iterrows():
            rel = "/".join(q.path.split("/")[2:])
            r = rel_raw.get(rel)
            if r is None:
                only_q.append(q)
            elif r["newest_file"] == q.newest_file and r["site"] == q.site:
                same_copy.append(q)
            else:
                superseded.append((q, r))
        L.append(f"\nRelation to raw/: {len(only_q)} quarantined runs exist only in quarantine (retired VoCo-on-AMOS design, 2026-10-15/16), "
                 f"{len(same_copy)} have the same newest-file time as the copy under raw/ (same files; the run was re-mirrored by rsync after the quarantine move, so raw/ holds the authoritative copy), "
                 f"{len(superseded)} are older or partial copies superseded by a newer run of the same path under raw/.\n")
        L.append("Quarantine-only runs:\n")
        for q in only_q:
            L.append(f"- {q.path} (status {q.status}, newest {q.newest_file})")
        L.append("\nSuperseded copies (quarantine -> raw):\n")
        for q, r in superseded:
            L.append(f"- {q.path} ({q.status}, newest {q.newest_file}) -> raw {r['site']} {r['status']}, newest {r['newest_file']}")
    (OUT / "run_inventory.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
