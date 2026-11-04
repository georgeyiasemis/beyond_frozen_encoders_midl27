#!/usr/bin/env python3
"""Training cost per backbone and method: time per 1000 steps on one reference GPU and peak GPU memory of a step.

    python cost_table.py

Time: report/cost/steps.csv (report/cost/parse_logs.py: median seconds per training step of every run of the paper,
validation excluded, and the run's GPU). The runs ran on five GPU models, so times are put on one scale by a
fixed-effects fit per task, log(s/step) = configuration + GPU, where a configuration is (backbone, method, cohort,
mode); configurations that ran on several GPU models tie the GPU effects together. The time of a method is the
geometric mean of its configurations' effects (over cohorts and modes) on the reference GPU (REF_GPU).
Memory and GPU time: report/cost/memory.csv (report/cost/measure_memory.py, one GPU job): peak allocated memory and
median time of one forward-backward-update on the training input, without data loading.
Writes ../paper/tables/tab_cost.tex, ../paper/tables/cost_numbers.tex and report/cost/gpu_factors.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
COST = V2 / "report" / "cost"
OUT = V2 / "paper" / "tables"
REF_GPU = "H100"
CLS_METHODS = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]
TEX = {"lin": "Lin", "attn": "Attn", "lora": "LoRA", "lorag": "LoRA$_{\\text{late}}$", "babmil": "LoRA+ABMIL", "dora": "DoRA",
       "rslora": "rsLoRA", "lin_then_lorag": "Lin$\\to$LoRA", "attn_then_lorag": "Attn$\\to$LoRA", "attnft": "Attn-only FT",
       "fullft": "Full FT", "dec": "Dec", "dec_then_lora": "Dec$\\to$LoRA"}
BB = {"tapct": ("TAP-CT", "Tapct"), "spectre": ("SPECTRE", "Spectre"), "pillar0": ("Pillar-0", "PillarZ")}
MACRO = {m: "".join(p.capitalize() for p in m.split("_")) for m in TEX}


def fit(d: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Alternating least squares for log(s) = config + gpu, gpu effect of REF_GPU = 0."""
    y = np.log(d.s_per_step.to_numpy()); cfg = d.cfg.to_numpy(); g = d.g.to_numpy()
    ge = pd.Series(0.0, index=sorted(set(g)))
    for _ in range(500):
        ce = pd.Series(y - ge[g].to_numpy()).groupby(cfg).mean()
        new = pd.Series(y - ce[cfg].to_numpy()).groupby(g).mean()
        new -= new[REF_GPU]
        if np.abs(new - ge).max() < 1e-10:
            ge = new; break
        ge = new
    ce = pd.Series(y - ge[g].to_numpy()).groupby(cfg).mean()
    res = y - ce[cfg].to_numpy() - ge[g].to_numpy()
    return ce, ge, float(res.std())


def main() -> None:
    d = pd.read_csv(COST / "steps.csv")
    d = d[d.s_per_step.notna() & d.method.isin(CLS_METHODS + SEG_METHODS)].copy()  # controls and full-FT settings add GPU overlap but are not in the table
    d = d[(d.task == "seg") | d.method.isin(CLS_METHODS)]
    d["g"] = d.gpu.str.replace(r" \(.*", "", regex=True).str.replace("NVIDIA ", "", regex=False)
    d["cfg"] = d.backbone + "|" + d.method + "|" + d.cohort + "|" + d["mode"].astype(str)
    mf = COST / "memory.csv"
    mem = pd.read_csv(mf) if mf.is_file() else pd.DataFrame(columns=["task", "backbone", "method", "peak_gib", "status"])
    mem = mem[mem.status == "ok"].set_index(["task", "backbone", "method"])
    t, macros, fac_rows = {}, {}, []
    for task, g in d.groupby("task"):
        ce, ge, sd = fit(g)
        for gpu, v in ge.items():
            fac_rows.append(dict(task=task, gpu=gpu, time_factor=float(np.exp(v)), runs=int((g.g == gpu).sum())))
        macros[f"CostResidSd{task.capitalize()}"] = f"{100 * (np.exp(sd) - 1):.0f}"
        cfg_bbm = ce.index.str.rsplit("|", n=2).str[0]
        for (bb, m), eff in ce.groupby(cfg_bbm.to_numpy()).mean().rename(lambda k: tuple(k.split("|"))).items():
            t[(task, bb, m)] = float(np.exp(eff)) * 1000 / 60  # minutes per 1000 steps on REF_GPU
    pd.DataFrame(fac_rows).to_csv(COST / "gpu_factors.csv", index=False, float_format="%.4f")
    macros["CostRefGpu"] = REF_GPU
    for (task, bb, m), v in t.items():
        macros[f"Cost{task.capitalize()}{BB[bb][1]}{MACRO[m]}Min"] = f"{v:.1f}"
        if (task, bb, m) in mem.index:
            macros[f"Cost{task.capitalize()}{BB[bb][1]}{MACRO[m]}Gib"] = f"{mem.loc[(task, bb, m), 'peak_gib']:.1f}"
            macros[f"Cost{task.capitalize()}{BB[bb][1]}{MACRO[m]}Gpus"] = f"{mem.loc[(task, bb, m), 's_per_step']:.2f}"
    macros["CostMemGpu"] = mem.gpu.iloc[0].replace("NVIDIA ", "") if len(mem) else ""
    # LoRA against Attn-only FT in classification: largest absolute difference in peak memory over the backbones
    dm = [abs(mem.loc[("cls", bb, "lora"), "peak_gib"] - mem.loc[("cls", bb, "attnft"), "peak_gib"]) for bb in BB if ("cls", bb, "lora") in mem.index]
    if dm:
        macros["CostClsLoraAttnftMaxGibdiff"] = f"{max(dm):.1f}"

    def cell(task, bb, m):
        gb = f"{mem.loc[(task, bb, m), 'peak_gib']:.1f}" if (task, bb, m) in mem.index else "--"
        gs = f"{mem.loc[(task, bb, m), 's_per_step']:.2f}" if (task, bb, m) in mem.index else "--"
        mm = f"{t[(task, bb, m)]:.1f}" if (task, bb, m) in t else "--"
        return f"{gb} & {gs} & {mm}"
    lines = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{TAP-CT} & \\multicolumn{3}{c}{SPECTRE} & \\multicolumn{3}{c}{Pillar-0} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
             "Method" + " & GiB & s & min" * 3 + " \\\\", "\\midrule",
             "\\multicolumn{10}{@{}l}{\\textit{Classification}} \\\\"]
    for m in CLS_METHODS:
        lines.append(f"{TEX[m]} & " + " & ".join(cell("cls", bb, m) for bb in BB) + " \\\\")
    lines.append("\\multicolumn{10}{@{}l}{\\textit{Segmentation}} \\\\")
    for m in SEG_METHODS:
        lines.append(f"{TEX[m]} & " + " & ".join(cell("seg", bb, m) for bb in ("tapct", "spectre")) + " & -- & -- & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_cost.tex").write_text("\n".join(lines) + "\n")
    (OUT / "cost_numbers.tex").write_text("% generated by tables/cost_table.py\n" + "".join(
        f"\\providecommand{{\\num{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(pd.DataFrame(fac_rows).to_string(index=False)); print("\n".join(lines))


if __name__ == "__main__":
    main()
