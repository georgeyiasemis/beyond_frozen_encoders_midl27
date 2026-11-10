#!/usr/bin/env python3
"""F1 (third review): label-level intervals of the joint classification models that also resample training seeds.

Joint models of Lin, LoRA and Attn-only FT were trained with seeds 42, 43 and 44 on every cohort-backbone pair. For a
comparison A - B, every label gets a held-out AUROC difference and N_CASE case-level draws for each of the 3 x 3 seed
pairs (seed of A, seed of B). The bootstrap of recompute.clustered_ci gains one stage: each drawn cohort-backbone
pair also draws one seed for A and one for B (independently, as the runs are), which applies to all its labels, since
the seed acts on the whole joint model. The point estimate averages over the 9 seed pairs. For reference, the same
bootstrap with seed 42 only (as the main intervals, joint models only) is reported next to it.

    python f1_seed_ci.py   -> out/f1_seed_ci.csv, paper/tables/seed_ci_numbers.tex
"""
from __future__ import annotations

import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recompute as rc  # noqa: E402

RES = Path(os.environ.get("BFE_RESULTS") or "/projects/cvpr2026_challenge_fomo4ct/beyond_frozen_encoders")  # structured results folder (tools/organize_results.py); BFE_RESULTS overrides
SEEDS = (42, 43, 44)
COMPARISONS = {"loralin": ("lora", "lin"), "attnftlora": ("attnft", "lora")}


def runs() -> pd.DataFrame:
    r = pd.read_csv(HERE / "out" / "runs.csv")
    r = r[(r.task == "cls") & (r["mode"] == "joint") & r.method.isin(["lin", "lora", "attnft"]) & r.seed.isin(SEEDS)]
    r = r[r.backbone.isin(["tapct", "spectre", "pillar0"])].reset_index(drop=True)
    # keep cohort-backbone pairs with every method at every seed
    full = r.groupby(["backbone", "cohort"]).apply(lambda g: len(g.drop_duplicates(["method", "seed"])) == 9)
    return r[[full[(b, c)] for b, c in zip(r.backbone, r.cohort)]].reset_index(drop=True)


def load(t: pd.DataFrame) -> dict:
    out = {}
    for i, r in t.iterrows():
        ns = type("R", (), dict(task="cls", rel=r.path.split("/", 2)[2], label_run="joint", cohort=r.cohort))
        _, arr = rc._parse_one((i, "cls", r.percase_file, rc.label_names(ns), str(RES)))
        out[(r.backbone, r.cohort, r.method, int(r.seed))] = {k: v["test"] for k, v in arr.items()
                                                              if k != "__warn__" and k not in rc.EXCLUDED_LABELS and "test" in v}
    return out


def seed_ci(point: list[np.ndarray], case: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    """point[i]: labels x S (seed pairs); case[i]: labels x S x N_CASE. Clusters, then one seed pair per drawn cluster,
    then labels within it, then one case-level draw per drawn label."""
    G, B = len(point), rc.N_BOOT
    C = rng.integers(0, G, (B, G))
    total, count = np.zeros(B), np.zeros(B)
    for i in range(G):
        n_i, S = point[i].shape
        s = rng.integers(0, S, (B, G, 1))
        j = rng.integers(0, n_i, (B, G, n_i))
        drawn = case[i][j, np.broadcast_to(s, j.shape), rng.integers(0, rc.N_CASE, (B, G, n_i))]
        hit = C == i
        total += (drawn.sum(axis=2) * hit).sum(axis=1)
        count += hit.sum(axis=1) * n_i
    lo, hi = np.percentile(total / count, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    t = runs()
    store = load(t)
    rng = np.random.default_rng(rc.SEED_LABEL)
    rows, macros = [], {}
    for key, (a, b) in COMPARISONS.items():
        for scope in ("all", "tapct", "spectre", "pillar0"):
            pts, cases, pts42, cases42 = [], [], [], []
            for (bb, ds), _ in t.groupby(["backbone", "cohort"]):
                if scope != "all" and bb != scope:
                    continue
                labs = sorted(set(store[(bb, ds, a, 42)]) & set(store[(bb, ds, b, 42)]))
                P, K = [], []
                for lab in labs:
                    p_l, k_l = [], []
                    for sa, sb in product(SEEDS, SEEDS):
                        A, Bv = store[(bb, ds, a, sa)].get(lab), store[(bb, ds, b, sb)].get(lab)
                        al = rc.align(A, Bv)
                        p_l.append(rc.auroc1(A[1][al[0]], A[2][al[0]]) - rc.auroc1(Bv[1][al[1]], Bv[2][al[1]]))
                        k_l.append(rc.label_boot_cls({"a": A, "b": Bv}, [("a", 1.0), ("b", -1.0)], rng))
                    if any(k is None for k in k_l):
                        continue
                    P.append(p_l)
                    K.append(np.stack(k_l))
                P, K = np.array(P), np.stack(K)
                pts.append(P), cases.append(K)
                i42 = list(product(SEEDS, SEEDS)).index((42, 42))
                pts42.append(P[:, i42]), cases42.append(K[:, i42])
            lo, hi = seed_ci(pts, cases, rng)
            lo42, hi42 = rc.clustered_ci(pts42, rng, cases42)
            mean = float(np.mean(np.concatenate([p.mean(axis=1) for p in pts])))
            mean42 = float(np.mean(np.concatenate(pts42)))
            pos = int(sum(p.mean() > 0 for p in pts))
            rows.append(dict(comparison=key, scope=scope, n_labels=sum(len(p) for p in pts), n_clusters=len(pts), mean=mean,
                             lo=lo, hi=hi, mean_seed42=mean42, lo_seed42=lo42, hi_seed42=hi42, clusters_pos=pos))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "out" / "f1_seed_ci.csv", index=False)
    tag = {"all": "All", "tapct": "Tapct", "spectre": "Spectre", "pillar0": "PillarZ"}
    cmp_tag = {"loralin": "Loralin", "attnftlora": "Attnftlora"}
    for r in out.itertuples():
        base = f"numSeedci{cmp_tag[r.comparison]}{tag[r.scope]}"
        for suf, v in (("Meandiff", r.mean), ("Lodiff", r.lo), ("Hidiff", r.hi)):
            macros[base + suf] = f"{v:+.3f}" if round(v, 3) != 0 else "0.000"
    macros["numSeedciClustersN"] = str(int(out[out.scope == "all"].n_clusters.iloc[0]))
    (HERE.parent / "paper" / "tables" / "seed_ci_numbers.tex").write_text(
        "% generated by metrics/f1_seed_ci.py\n" + "".join(f"\\providecommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
