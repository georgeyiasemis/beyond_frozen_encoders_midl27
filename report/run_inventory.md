# Run inventory of the fm_adapt / fm_seg sweep (files on disk, 2026-10-31)

Source: `/home/g.yiasemis/projects/cvpr_ct_challenge/midl_paper/results/raw` (rsync mirror of both sites; never deletes). Design: YAML trees under `/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo/examples/fm_probes/fm_adapt` and `/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo/examples/fm_probes/fm_seg` (one YAML per run; totalseg `eval/` YAMLs are held-out re-evaluations of the same run and are not counted). Script: `midl_paper_v2/report/scripts/build_run_inventory.py`. Tables: `run_inventory.csv` (one row per run directory, incl. quarantine), `run_inventory_design.csv` (one row per designed run).

Status rule (per run directory, from its own files): **finished** = `checkpoints/best/best_val_metric_val_<auroc|dice>.txt` exists, `val_case_metrics/iter_<best>/<logits|dice>.csv` exists and has test rows, and `checkpoints/last_checkpoint.txt` reached the YAML's `num_iterations`; **partial** = best record exists but one of those fails; **failed/empty** = no best record. This is the same selection the harvest uses (best validation macro metric, per-case file of that iteration).

## 1. Totals

| task | failed/empty | finished | finished(no yaml) |
|---|---|---|---|
| cls | 2 | 1354 | 0 |
| seg | 0 | 1034 | 4 |

Run directories under raw/: 2394 runs + 24 auxiliary directories (smoke tests, scratch evals; listed in section 6). Quarantined run directories (excluded): 130.

Per site:

| site | task | failed/empty | finished | finished(no yaml) |
|---|---|---|---|---|
| gaia | cls | 2 | 332 | 0 |
| gaia | seg | 0 | 392 | 2 |
| snellius | cls | 0 | 1022 | 0 |
| snellius | seg | 0 | 642 | 2 |

Finished runs by task, backbone and design class (finished copies on either site, deduplicated by run path):

| task | backbone | compute_matched | main | revision | seed_repeat |
|---|---|---|---|---|---|
| cls | pillar0 | 0 | 110 | 5 | 8 |
| cls | spectre | 0 | 418 | 20 | 24 |
| cls | tapct | 0 | 495 | 25 | 40 |
| cls | voco | 0 | 209 | 0 | 0 |
| seg | spectre | 90 | 263 | 0 | 30 |
| seg | tapct | 90 | 263 | 5 | 30 |
| seg | voco | 0 | 263 | 0 | 0 |

## 2. Design vs found

Designed runs (YAML): 2388 = cls 1354 + seg 1034.

| task | class | finished |
|---|---|---|
| cls | main | 1232 |
| cls | revision | 50 |
| cls | seed_repeat | 72 |
| seg | compute_matched | 180 |
| seg | main | 789 |
| seg | revision | 5 |
| seg | seed_repeat | 60 |

By task / backbone / class:

| task | backbone | class | finished |
|---|---|---|---|
| cls | pillar0 | main | 110 |
| cls | pillar0 | revision | 5 |
| cls | pillar0 | seed_repeat | 8 |
| cls | spectre | main | 418 |
| cls | spectre | revision | 20 |
| cls | spectre | seed_repeat | 24 |
| cls | tapct | main | 495 |
| cls | tapct | revision | 25 |
| cls | tapct | seed_repeat | 40 |
| cls | voco | main | 209 |
| seg | spectre | compute_matched | 90 |
| seg | spectre | main | 263 |
| seg | spectre | seed_repeat | 30 |
| seg | tapct | compute_matched | 90 |
| seg | tapct | main | 263 |
| seg | tapct | revision | 5 |
| seg | tapct | seed_repeat | 30 |
| seg | voco | main | 263 |

### 2a. Compact count table: finished / designed per (task, cohort, backbone, mode); columns = methods


**cls** (cell = finished/designed; a bare number means all finished; `-` = not in the design)

| cohort | backbone | mode | lin | attn | lora | lorag | babmil | fullft | attnft | rslora | dora | lin_then_lorag | attn_then_lorag | fullft_accum | fullft_llrd | fullft_lowlr | mlpft | rslora_lowlr | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amos_covid_luna | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| amos_covid_luna | spectre | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | - | - | 11/11 |
| amos_covid_luna | spectre | joint_s43 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| amos_covid_luna | spectre | joint_s44 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| amos_covid_luna | spectre | pertask | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | - | - | - | - | - | 187/187 |
| amos_covid_luna | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| amos_covid_luna | tapct | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | - | - | 11/11 |
| amos_covid_luna | tapct | joint_s43 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| amos_covid_luna | tapct | joint_s44 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| amos_covid_luna | tapct | pertask | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | - | - | - | - | - | 187/187 |
| breastdivider | pillar0 | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | - | - | 11/11 |
| breastdivider | pillar0 | pertask | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | - | - | - | - | - | 22/22 |
| fdg | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| fdg | spectre | joint_s43 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| fdg | spectre | joint_s44 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| fdg | spectre | pertask | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | - | - | - | - | - | 44/44 |
| fdg | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| fdg | tapct | joint_s43 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| fdg | tapct | joint_s44 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| fdg | tapct | pertask | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | - | - | - | - | - | 44/44 |
| fdg | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | - | - | 11/11 |
| fdg | voco | pertask | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | - | - | - | - | - | 44/44 |
| rsna2019_ich | pillar0 | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| rsna2019_ich | pillar0 | joint_s43 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2019_ich | pillar0 | joint_s44 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2019_ich | pillar0 | pertask | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | - | - | - | - | - | 66/66 |
| rsna2019_ich | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| rsna2019_ich | tapct | joint_s43 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2019_ich | tapct | joint_s44 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2019_ich | tapct | pertask | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | - | - | - | - | - | 66/66 |
| rsna2022 | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| rsna2022 | spectre | joint_s43 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| rsna2022 | spectre | joint_s44 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| rsna2022 | spectre | pertask | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | - | - | - | - | - | 88/88 |
| rsna2022 | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| rsna2022 | tapct | joint_s43 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2022 | tapct | joint_s44 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2022 | tapct | pertask | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | - | - | - | - | - | 88/88 |
| rsna2022 | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | - | - | 11/11 |
| rsna2022 | voco | pertask | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | - | - | - | - | - | 88/88 |
| rsna2023 | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| rsna2023 | spectre | joint_s43 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| rsna2023 | spectre | joint_s44 | 1 | - | 1 | - | - | - | 1 | - | - | - | - | - | - | - | - | - | 3/3 |
| rsna2023 | spectre | pertask | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | - | - | - | - | - | 44/44 |
| rsna2023 | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 16/16 |
| rsna2023 | tapct | joint_s43 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2023 | tapct | joint_s44 | 1 | - | 1 | - | - | 1 | 1 | - | - | - | - | - | - | - | - | - | 4/4 |
| rsna2023 | tapct | pertask | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | - | - | - | - | - | 44/44 |
| rsna2023 | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | - | - | 11/11 |
| rsna2023 | voco | pertask | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | - | - | - | - | - | 44/44 |

**seg** (cell = finished/designed; a bare number means all finished; `-` = not in the design)

| cohort | backbone | mode | lin | dec | lora | lorag | dora | fullft | dec_then_lora | scratch | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amos22 | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| amos22 | spectre | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| amos22 | spectre | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| amos22 | spectre | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| amos22 | spectre | pertask | 15 | 15 | 15 | 15 | 15 | 15 | 15 | - | 105/105 |
| amos22 | spectre | pertask_cm | - | 15 | 15 | - | - | 15 | - | - | 45/45 |
| amos22 | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| amos22 | tapct | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 8/8 |
| amos22 | tapct | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| amos22 | tapct | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| amos22 | tapct | pertask | 15 | 15 | 15 | 15 | 15 | 15 | 15 | - | 105/105 |
| amos22 | tapct | pertask_cm | - | 15 | 15 | - | - | 15 | - | - | 45/45 |
| amos22 | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| amos22 | voco | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| amos22 | voco | pertask | 15 | 15 | 15 | 15 | 15 | 15 | 15 | - | 105/105 |
| kits23 | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| kits23 | spectre | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| kits23 | spectre | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| kits23 | spectre | pertask | 3 | 3 | 3 | 3 | 3 | 3 | 3 | - | 21/21 |
| kits23 | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 8/8 |
| kits23 | tapct | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| kits23 | tapct | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| kits23 | tapct | pertask | 3 | 3 | 3 | 3 | 3 | 3 | 3 | - | 21/21 |
| kits23 | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| kits23 | voco | pertask | 3 | 3 | 3 | 3 | 3 | 3 | 3 | - | 21/21 |
| lits | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| lits | spectre | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| lits | spectre | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| lits | spectre | pertask | 2 | 2 | 2 | 2 | 2 | 2 | 2 | - | 14/14 |
| lits | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 8/8 |
| lits | tapct | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| lits | tapct | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| lits | tapct | pertask | 2 | 2 | 2 | 2 | 2 | 2 | 2 | - | 14/14 |
| lits | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| lits | voco | pertask | 2 | 2 | 2 | 2 | 2 | 2 | 2 | - | 14/14 |
| msd_pancreas | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| msd_pancreas | spectre | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| msd_pancreas | spectre | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| msd_pancreas | spectre | pertask | 2 | 2 | 2 | 2 | 2 | 2 | 2 | - | 14/14 |
| msd_pancreas | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 8/8 |
| msd_pancreas | tapct | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| msd_pancreas | tapct | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| msd_pancreas | tapct | pertask | 2 | 2 | 2 | 2 | 2 | 2 | 2 | - | 14/14 |
| msd_pancreas | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| msd_pancreas | voco | pertask | 2 | 2 | 2 | 2 | 2 | 2 | 2 | - | 14/14 |
| totalseg | spectre | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| totalseg | spectre | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| totalseg | spectre | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| totalseg | spectre | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| totalseg | spectre | pertask | 15 | 15 | 15 | - | - | 15 | - | - | 60/60 |
| totalseg | spectre | pertask_cm | - | 15 | 15 | - | - | 15 | - | - | 45/45 |
| totalseg | tapct | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| totalseg | tapct | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 8/8 |
| totalseg | tapct | joint_s43 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| totalseg | tapct | joint_s44 | - | 1 | 1 | - | - | 1 | - | - | 3/3 |
| totalseg | tapct | pertask | 15 | 15 | 15 | - | - | 15 | - | - | 60/60 |
| totalseg | tapct | pertask_cm | - | 15 | 15 | - | - | 15 | - | - | 45/45 |
| totalseg | voco | joint | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| totalseg | voco | joint_lm | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 7/7 |
| totalseg | voco | pertask | 15 | 15 | 15 | - | - | 15 | - | - | 60/60 |

Design entries without a YAML on disk (enumerated by `fm_seg/generate.iter_runs()`, runs exist): 2

- seg amos22/voco/joint_lm/dec_then_lora/joint: finished (snellius:finished); horizon taken from sibling backbones = 8000
- seg totalseg/voco/joint_lm/dec_then_lora/joint: finished (snellius:finished); horizon taken from sibling backbones = 24000

### 2b. MISSING designed runs (0)

None.

### 2c. Designed runs present but not finished (0)

None.

### 2d. EXTRA run directories not in the YAML design (4)

- raw/snellius/seg/amos22/spectre/joint_lm/dec_then_lora_unscaled/config: status finished(no yaml) (); best iter 8000, newest file 2026-10-19 20:48
- raw/snellius/seg/amos22/tapct/joint_lm/dec_then_lora_unscaled/config: status finished(no yaml) (); best iter 8000, newest file 2026-10-18 19:39
- raw/gaia/seg/amos22/spectre/joint_lm/dec_stage1/config: status finished(no yaml) (); best iter 24000, newest file 2026-10-21 10:18
- raw/gaia/seg/amos22/tapct/joint_lm/dec_stage1/config: status finished(no yaml) (); best iter 24000, newest file 2026-10-21 01:58

### 2e. DUPLICATED runs (same run path on both sites) (2)

| run | snellius | gaia | best iter (sn/ga) | best val (sn/ga) | test metric (sn/ga) | delta test | agree? |
|---|---|---|---|---|---|---|---|
| amos_covid_luna/tapct/pertask/attn/atherosclerosis/atherosclerosis | finished | failed/empty | 5000/ | 0.820261/ | 0.871324/- | - | n/a (one copy unfinished) |
| amos_covid_luna/tapct/pertask/lin/liver_cyst/liver_cyst | finished | failed/empty | 2000/ | 0.610340/ | 0.641214/- | - | n/a (one copy unfinished) |

### 2f. FAILED / partial run directories (2)

- raw/gaia/amos_covid_luna/tapct/pertask/attn/atherosclerosis/atherosclerosis: **failed/empty** - no best-metric record; no checkpoints, no per-case files; validations 0 (per-case) / 0 (logged); last logged , last ckpt ; newest file 2026-10-16 03:44; dup on other site: snellius:finished
- raw/gaia/amos_covid_luna/tapct/pertask/lin/liver_cyst/liver_cyst: **failed/empty** - no best-metric record; 0 per-case files; last ckpt 500; validations 0 (per-case) / 0 (logged); last logged 995, last ckpt 500; newest file 2026-10-16 03:27; dup on other site: snellius:finished

## 3. Per-case files behind the headline numbers

Every finished run has `val_case_metrics/iter_<best>/<logits|dice>.csv` with test rows by construction of the status rule (2392 finished run directories). Runs whose best-iteration per-case file is missing are listed as partial in 2f (count: 0).

Test-case coverage: for every group (task, cohort, mode family, label) the number of unique test case ids of each finished run is compared with the group's modal value. Mode family: pertask / pertask_cm / joint (joint, joint_lm, joint_s43, joint_s44 share the case set).

| task | cohort | fam | runs | n_test_cases_min | n_test_cases_max | n_val_cases_min | n_val_cases_max |
|---|---|---|---|---|---|---|---|
| cls | amos_covid_luna | joint | 68 | 310 | 310 | 310 | 310 |
| cls | amos_covid_luna | pertask | 374 | 26 | 135 | 26 | 131 |
| cls | breastdivider | joint | 11 | 62 | 62 | 63 | 63 |
| cls | breastdivider | pertask | 22 | 22 | 62 | 22 | 63 |
| cls | fdg | joint | 57 | 98 | 98 | 99 | 99 |
| cls | fdg | pertask | 132 | 98 | 98 | 99 | 99 |
| cls | rsna2019_ich | joint | 48 | 3264 | 3264 | 3269 | 3269 |
| cls | rsna2019_ich | pertask | 132 | 3264 | 3264 | 3269 | 3269 |
| cls | rsna2022 | joint | 57 | 202 | 202 | 201 | 201 |
| cls | rsna2022 | pertask | 264 | 202 | 202 | 201 | 201 |
| cls | rsna2023 | joint | 57 | 315 | 315 | 315 | 315 |
| cls | rsna2023 | pertask | 132 | 315 | 315 | 315 | 315 |
| seg | amos22 | joint | 59 | 50 | 50 | 50 | 50 |
| seg | amos22 | pertask | 315 | 50 | 50 | 50 | 50 |
| seg | amos22 | pertask_cm | 90 | 50 | 50 | 50 | 50 |
| seg | kits23 | joint | 34 | 73 | 73 | 73 | 73 |
| seg | kits23 | pertask | 63 | 73 | 73 | 73 | 73 |
| seg | lits | joint | 34 | 20 | 20 | 20 | 20 |
| seg | lits | pertask | 42 | 20 | 20 | 20 | 20 |
| seg | msd_pancreas | joint | 34 | 42 | 42 | 42 | 42 |
| seg | msd_pancreas | pertask | 42 | 42 | 42 | 42 | 42 |
| seg | totalseg | joint | 55 | 109 | 109 | 109 | 109 |
| seg | totalseg | pertask | 180 | 109 | 109 | 109 | 109 |
| seg | totalseg | pertask_cm | 90 | 109 | 109 | 109 | 109 |

Runs whose unique-test-case count differs from the modal value of their group: 0


Runs whose number of test rows differs from the modal value of their group: 0

None.

Cross-check against the harvest outputs actually used by the draft (`results/master_metrics.csv`, `results/master_metrics_seg.csv`, `results/revision_metrics.csv`): do their `source_file` per-case files exist, and does their run set match the finished main-design runs here?

- `master_metrics.csv`: 1279 source files; missing on disk: 0
  - runs in `master_metrics.csv` that are not finished *main-design* runs here: 47; of these 47 are revision-method runs the harvest also enumerates (fine), and 0 are something else
  - finished main-design runs here that are not in `master_metrics.csv`: 0
- `master_metrics_seg.csv`: 789 source files; missing on disk: 0
  - runs in `master_metrics_seg.csv` that are not finished *main-design* runs here: 0; of these 0 are revision-method runs the harvest also enumerates (fine), and 0 are something else
  - finished main-design runs here that are not in `master_metrics_seg.csv`: 0
- `revision_metrics.csv`: 367 runs; finished seed/revision/compute-matched runs here: 367; in revision_metrics but not finished here: 0; finished here but not in revision_metrics: 0

## 4. Comparison with the draft's macros and the old inventories

| quantity | designed (YAML) | finished | draft macro |
|---|---|---|---|
| numRunsCls (main cls design, no VoCo, no breastdivider) | 990 | 990 | 990 |
| numVocoRunsClsN (main cls VoCo) | 209 | 209 | 209 (Done: 209) |
| breastdivider cls (not in the paper) | 33 | 33 | - |
| numSegRunsN (main seg design, no VoCo) | 526 | 526 | 526 |
| numVocoRunsSegN (main seg VoCo) | 263 | 263 | 263 (Done: 263) |
| numSeedsclsRunsN (cls seed repeats) | 72 | 72 | 72 |
| numSeedssegRunsN (seg seed repeats) | 60 | 60 | 60 |
| cls revision methods (joint) | 50 | 50 | - |
| seg compute-matched (pertask_cm) | 180 | 180 | - |
| seg scratch | 5 | 5 | - |

Old inventories (`results/runs_inventory.csv`, `results/runs_inventory_seg.csv`), status counts as written there:

- `runs_inventory.csv`: INCOMPLETE=2, MISSING=513, ok=1277 (total 1792)
- `runs_inventory_seg.csv`: ok=789 (total 789)

Per-run status of `runs_inventory.csv` vs this inventory (same run path; ok<->finished, MISSING<->not on disk, INCOMPLETE<->partial/failed): 5 disagreements
- rsna2022/tapct/joint/fullft_llrd/config: old `INCOMPLETE` () vs here `finished` (gaia)
- rsna2022/tapct/joint/fullft_accum/config: old `MISSING` (no best_val_metric_val_auroc.txt (no run dir)) vs here `finished` (gaia)
- rsna2022/tapct/joint/mlpft/config: old `MISSING` (no best_val_metric_val_auroc.txt (no run dir)) vs here `finished` (gaia)
- rsna2022/tapct/joint/rslora_lowlr/config: old `MISSING` (no best_val_metric_val_auroc.txt (no run dir)) vs here `finished` (gaia)
- rsna2019_ich/pillar0/joint/rslora_lowlr/config: old `INCOMPLETE` () vs here `finished` (gaia)

Per-run status of `runs_inventory_seg.csv` vs this inventory (same run path; ok<->finished, MISSING<->not on disk, INCOMPLETE<->partial/failed): 0 disagreements

Freshness: harvest outputs vs the newest run file on disk

- `master_metrics.csv`: written 2026-10-31 13:31
- `master_metrics_seg.csv`: written 2026-10-30 01:10
- `revision_metrics.csv`: written 2026-10-31 13:32
- `runs_inventory.csv`: written 2026-10-31 13:31
- `runs_inventory_seg.csv`: written 2026-10-30 01:10
- `paper/tables/numbers.tex`: written 2026-10-31 13:45
- newest run file, cls on gaia: 2026-10-30 08:47
- newest run file, cls on snellius: 2026-10-31 07:34
- newest run file, seg on gaia: 2026-10-27 22:15
- newest run file, seg on snellius: 2026-10-29 03:09

Notes on the differences: `runs_inventory.csv` enumerates `generate.METHODS`, which since 2026-10-26 includes the five revision methods, for every mode and task, so its 'expected' list (1792) contains 1792 - 1232 = 560 combinations that were never designed as YAMLs (revision methods per task, revision methods for VoCo/breastdivider/joint_lm); most of its 513 MISSING rows are those. It also knows no seed modes.

- fm_seg generate.iter_runs(): 789 runs; YAML tree (main class): 789; in generator but no YAML: 0; YAML but not in generator main: 0
- fm_seg generate.iter_revision_runs(): 365 runs; with a YAML on disk: 245; without YAML (never generated/submitted): 120 -> Counter({('voco', 'pertask_cm'): 90, ('voco', 'joint_s43'): 15, ('voco', 'joint_s44'): 15})

## 5. File dates and anomalies

Newest-file date of the run directories, by backbone (number of runs):

| day | pillar0 | spectre | tapct | voco |
|---|---|---|---|---|
| 2026-10-15 | 21 | 0 | 102 | 0 |
| 2026-10-16 | 38 | 25 | 49 | 0 |
| 2026-10-17 | 21 | 279 | 114 | 0 |
| 2026-10-18 | 0 | 29 | 149 | 0 |
| 2026-10-19 | 22 | 221 | 213 | 0 |
| 2026-10-20 | 8 | 50 | 56 | 0 |
| 2026-10-21 | 0 | 5 | 5 | 0 |
| 2026-10-23 | 0 | 17 | 68 | 0 |
| 2026-10-24 | 0 | 54 | 6 | 0 |
| 2026-10-25 | 0 | 3 | 0 | 0 |
| 2026-10-26 | 3 | 150 | 107 | 0 |
| 2026-10-27 | 9 | 14 | 74 | 208 |
| 2026-10-28 | 0 | 0 | 0 | 183 |
| 2026-10-29 | 1 | 0 | 8 | 52 |
| 2026-10-30 | 0 | 0 | 1 | 26 |
| 2026-10-31 | 0 | 0 | 0 | 3 |

VoCo runs with files older than 2026-10-26 (retired design) under raw/: 0

None under raw/. The stale VoCo runs are all in `raw_voco_stale_quarantine_20261027` (section 7).

Other anomalies:

- raw/snellius/seg/amos22/voco/joint_lm/dec_then_lora/config: last checkpoint 24000 (best 22000, 12 validations) beyond the YAML's num_iterations 8000: the YAML on disk is older than the run (stale design file; the run follows the later two-stage joint_lm horizon)
- raw/gaia/seg/amos22/spectre/joint_lm/dec_then_lora/config: last checkpoint 24000 (best 24000, 12 validations) beyond the YAML's num_iterations 8000: the YAML on disk is older than the run (stale design file; the run follows the later two-stage joint_lm horizon)
- raw/gaia/seg/amos22/tapct/joint_lm/dec_then_lora/config: last checkpoint 24000 (best 16000, 12 validations) beyond the YAML's num_iterations 8000: the YAML on disk is older than the run (stale design file; the run follows the later two-stage joint_lm horizon)
- 109 finished seg runs keep the per-case dice.csv of a single iteration (the selected one) although metrics.json logs all validations: held-out re-inference at the selected checkpoint (`fm_seg/reinfer_dice.py`); by (task, site, dataset): seg/gaia/amos22: 19, seg/snellius/amos22: 37, seg/snellius/kits23: 14, seg/snellius/lits: 4, seg/snellius/msd_pancreas: 35
- raw/snellius/seg/lits/spectre/joint/lin/config: 5 per-case folders vs 8 validations logged (per-case files of some iterations absent at the source; the best iteration 6000 is present)

Number of per-case validation folders of finished runs, by (task, mode): cls/joint: 4x28; cls/joint: 8x176; cls/joint_lm: 13x4; cls/joint_lm: 26x18; cls/joint_s43: 8x36; cls/joint_s44: 8x36; cls/pertask: 4x192; cls/pertask: 8x864; seg/joint: 1x24; seg/joint: 4x13; seg/joint: 5x1; seg/joint: 8x70; seg/joint_lm: 1x4; seg/joint_lm: 12x6; seg/joint_lm: 24x34; seg/joint_s43: 8x18; seg/joint_s43: 24x12; seg/joint_s44: 8x18; seg/joint_s44: 24x12; seg/pertask: 1x80; seg/pertask: 4x63; seg/pertask: 8x499; seg/pertask_cm: 8x180

## 6. Auxiliary directories under raw/ (not runs of the design)

- raw/snellius/seg/amos22/spectre/joint_lm/dec_then_lora_unscaled/dec_then_lora/config [unparsed]: status finished(no yaml), best iter 8000, newest 2026-10-19 20:48
- raw/snellius/seg/amos22/tapct/joint_lm/dec_then_lora_unscaled/dec_then_lora/config [unparsed]: status finished(no yaml), best iter 8000, newest 2026-10-18 19:39
- raw/snellius/seg/voco_smoke/amos22/voco/pertask/dec/spleen/spleen [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:52
- raw/snellius/seg/voco_smoke/amos22/voco/pertask/dec_then_lora/spleen/spleen [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:53
- raw/snellius/seg/voco_smoke/amos22/voco/pertask/lin/spleen/spleen [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:51
- raw/snellius/seg/voco_smoke/kits23/voco/joint/lora/config [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:54
- raw/snellius/seg/voco_smoke/lits/voco/pertask/lorag/tumor/tumor [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:55
- raw/snellius/seg/voco_smoke/msd_pancreas/voco/joint/dora/config [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:56
- raw/snellius/seg/voco_smoke/totalseg/voco/joint/fullft/.test_eval/fullft/config [test_eval_scratch]: status not_a_run, best iter , newest 2026-10-27 04:57
- raw/snellius/seg/voco_smoke/totalseg/voco/joint/fullft/config [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:57
- raw/snellius/seg/voco_smoke/totalseg/voco/pertask/dec/glands/.test_eval/glands/glands [test_eval_scratch]: status not_a_run, best iter , newest 2026-10-27 04:58
- raw/snellius/seg/voco_smoke/totalseg/voco/pertask/dec/glands/glands [seg_voco_smoke]: status finished(no yaml), best iter 20, newest 2026-10-27 04:58
- raw/snellius/seg_gaia_ckpts/amos22/tapct/pertask/lin/spleen/spleen [seg_gaia_ckpts]: status partial, best iter 14000, newest 2026-10-18 20:22
- raw/gaia/voco_smoke/cls/fdg/voco/joint/attn/config [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 03:27
- raw/gaia/voco_smoke/cls/fdg/voco/joint/attn_then_lorag/config [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 03:46
- raw/gaia/voco_smoke/cls/fdg/voco/joint/attnft/config [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 04:39
- raw/gaia/voco_smoke/cls/fdg/voco/pertask/lin/LUNG_CANCER/LUNG_CANCER [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 02:50
- raw/gaia/voco_smoke/cls/fdg/voco/pertask/lin_then_lorag/LUNG_CANCER/LUNG_CANCER [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 03:09
- raw/gaia/voco_smoke/cls/rsna2022/voco/joint/lora/config [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 03:50
- raw/gaia/voco_smoke/cls/rsna2022/voco/pertask/lorag/C5/C5 [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 03:54
- raw/gaia/voco_smoke/cls/rsna2023/voco/joint/babmil/config [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 04:02
- raw/gaia/voco_smoke/cls/rsna2023/voco/joint/fullft/config [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 04:47
- raw/gaia/voco_smoke/cls/rsna2023/voco/pertask/dora/spleen_injury/spleen_injury [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 04:11
- raw/gaia/voco_smoke/cls/rsna2023/voco/pertask/rslora/any_injury/any_injury [voco_smoke]: status finished(no yaml), best iter 10, newest 2026-10-27 04:20

Other non-run content under raw/: `raw/<site>/revision/{features,nsd,nsd_scratch,test}` = post-hoc linear-probe features (.npz), NSD re-evaluations (nsd.csv per run) and their logs; `raw/gaia/seg_examples_scratch`, `raw/gaia/seg/examples` = example-figure scratch (logs only); `raw/<site>/seg/SITE_SPLIT.txt` = site split note (Gaia: amos22/tapct/pertask).

## 7. Quarantined run directories (excluded from everything above): 130

| root | site | task | cohort | backbone | status | n |
|---|---|---|---|---|---|---|
| raw_dup_quarantine_20261029 | gaia | cls | fdg | voco | partial | 2 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | amos_covid_luna | voco | finished(no yaml) | 5 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | amos_covid_luna | voco | partial | 2 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | fdg | voco | finished | 6 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | fdg | voco | partial | 6 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | rsna2022 | voco | finished | 26 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | rsna2022 | voco | partial | 3 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | rsna2023 | voco | finished | 11 |
| raw_voco_stale_quarantine_20261027 | snellius | cls | rsna2023 | voco | partial | 4 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | amos22 | voco | finished | 27 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | amos22 | voco | partial | 5 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | kits23 | voco | failed/empty | 1 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | kits23 | voco | finished | 5 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | msd_pancreas | voco | finished | 10 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | msd_pancreas | voco | partial | 1 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | totalseg | voco | failed/empty | 1 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | totalseg | voco | finished | 13 |
| raw_voco_stale_quarantine_20261027 | snellius | seg | totalseg | voco | partial | 2 |

Relation to raw/: 7 quarantined runs exist only in quarantine (retired VoCo-on-AMOS design, 2026-10-15/16), 85 have the same newest-file time as the copy under raw/ (same files; the run was re-mirrored by rsync after the quarantine move, so raw/ holds the authoritative copy), 38 are older or partial copies superseded by a newer run of the same path under raw/.

Quarantine-only runs:

- raw_voco_stale_quarantine_20261027/snellius/amos_covid_luna/voco/joint/lin/config (status finished(no yaml), newest 2026-10-15 21:36)
- raw_voco_stale_quarantine_20261027/snellius/amos_covid_luna/voco/joint_lm/lin/config (status partial, newest 2026-10-16 00:00)
- raw_voco_stale_quarantine_20261027/snellius/amos_covid_luna/voco/pertask/lin/colorectal_cancer/colorectal_cancer (status finished(no yaml), newest 2026-10-15 21:40)
- raw_voco_stale_quarantine_20261027/snellius/amos_covid_luna/voco/pertask/lin/covid/covid (status finished(no yaml), newest 2026-10-15 22:09)
- raw_voco_stale_quarantine_20261027/snellius/amos_covid_luna/voco/pertask/lin/fatty_liver/fatty_liver (status finished(no yaml), newest 2026-10-15 22:48)
- raw_voco_stale_quarantine_20261027/snellius/amos_covid_luna/voco/pertask/lin/gallstone/gallstone (status finished(no yaml), newest 2026-10-15 23:29)
- raw_voco_stale_quarantine_20261027/snellius/amos_covid_luna/voco/pertask/lin/hydronephrosis/hydronephrosis (status partial, newest 2026-10-16 00:00)

Superseded copies (quarantine -> raw):

- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/joint/attn/config (partial, newest 2026-10-27 14:10) -> raw snellius finished, newest 2026-10-27 15:00
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/joint/lora/config (finished, newest 2026-10-15 23:04) -> raw snellius finished, newest 2026-10-27 20:57
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/joint/lorag/config (partial, newest 2026-10-16 00:00) -> raw snellius finished, newest 2026-10-28 02:06
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/pertask/attn/MELANOMA/MELANOMA (finished, newest 2026-10-15 22:55) -> raw snellius finished, newest 2026-10-29 09:29
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/pertask/attn/NEGATIVE/NEGATIVE (partial, newest 2026-10-15 23:54) -> raw snellius finished, newest 2026-10-29 14:35
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/pertask/babmil/LUNG_CANCER/LUNG_CANCER (partial, newest 2026-10-27 14:11) -> raw snellius finished, newest 2026-10-27 15:53
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/pertask/babmil/MELANOMA/MELANOMA (finished, newest 2026-10-15 23:06) -> raw snellius finished, newest 2026-10-28 03:51
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/pertask/babmil/NEGATIVE/NEGATIVE (partial, newest 2026-10-16 00:00) -> raw snellius finished, newest 2026-10-28 09:51
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/pertask/lorag/LUNG_CANCER/LUNG_CANCER (finished, newest 2026-10-15 23:00) -> raw snellius finished, newest 2026-10-30 19:36
- raw_voco_stale_quarantine_20261027/snellius/fdg/voco/pertask/lorag/LYMPHOMA/LYMPHOMA (partial, newest 2026-10-16 00:00) -> raw snellius finished, newest 2026-10-29 11:04
- raw_voco_stale_quarantine_20261027/snellius/rsna2022/voco/pertask/attnft/C5/C5 (partial, newest 2026-10-27 14:20) -> raw snellius finished, newest 2026-10-27 15:10
- raw_voco_stale_quarantine_20261027/snellius/rsna2022/voco/pertask/lin/C6/C6 (partial, newest 2026-10-27 14:21) -> raw snellius finished, newest 2026-10-27 14:35
- raw_voco_stale_quarantine_20261027/snellius/rsna2022/voco/pertask/lorag/C6/C6 (partial, newest 2026-10-27 14:21) -> raw snellius finished, newest 2026-10-27 14:47
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/joint/attn/config (finished, newest 2026-10-15 21:30) -> raw snellius finished, newest 2026-10-29 21:40
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/joint/fullft/config (partial, newest 2026-10-27 14:17) -> raw snellius finished, newest 2026-10-27 14:57
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/joint/lora/config (finished, newest 2026-10-15 23:25) -> raw snellius finished, newest 2026-10-30 01:33
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/attn/kidney_injury/kidney_injury (finished, newest 2026-10-15 21:31) -> raw snellius finished, newest 2026-10-28 15:31
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/attn/liver_injury/liver_injury (finished, newest 2026-10-15 23:16) -> raw gaia finished, newest 2026-10-28 03:31
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/attn/spleen_injury/spleen_injury (partial, newest 2026-10-16 00:00) -> raw gaia finished, newest 2026-10-28 06:52
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/babmil/kidney_injury/kidney_injury (finished, newest 2026-10-15 21:42) -> raw gaia finished, newest 2026-10-29 21:11
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/babmil/liver_injury/liver_injury (finished, newest 2026-10-15 23:37) -> raw snellius finished, newest 2026-10-27 21:20
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/babmil/spleen_injury/spleen_injury (partial, newest 2026-10-15 23:59) -> raw snellius finished, newest 2026-10-28 01:16
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/fullft/spleen_injury/spleen_injury (finished, newest 2026-10-15 22:11) -> raw snellius finished, newest 2026-10-28 17:14
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/lora/spleen_injury/spleen_injury (finished, newest 2026-10-15 21:41) -> raw gaia finished, newest 2026-10-28 23:15
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/lorag/any_injury/any_injury (finished, newest 2026-10-15 23:31) -> raw gaia finished, newest 2026-10-29 02:37
- raw_voco_stale_quarantine_20261027/snellius/rsna2023/voco/pertask/lorag/kidney_injury/kidney_injury (partial, newest 2026-10-16 00:00) -> raw gaia finished, newest 2026-10-29 06:07
- raw_voco_stale_quarantine_20261027/snellius/seg/amos22/voco/joint/dora/config (partial, newest 2026-10-27 14:19) -> raw snellius finished, newest 2026-10-27 14:24
- raw_voco_stale_quarantine_20261027/snellius/seg/amos22/voco/pertask/dec/gallbladder/gallbladder (partial, newest 2026-10-27 14:18) -> raw snellius finished, newest 2026-10-27 14:34
- raw_voco_stale_quarantine_20261027/snellius/seg/amos22/voco/pertask/dora/left_kidney/left_kidney (partial, newest 2026-10-27 14:21) -> raw snellius finished, newest 2026-10-27 15:40
- raw_voco_stale_quarantine_20261027/snellius/seg/amos22/voco/pertask/fullft/stomach/stomach (partial, newest 2026-10-27 14:18) -> raw snellius finished, newest 2026-10-27 14:46
- raw_voco_stale_quarantine_20261027/snellius/seg/amos22/voco/pertask/lora/right_adrenal/right_adrenal (partial, newest 2026-10-27 14:20) -> raw snellius finished, newest 2026-10-27 14:23
- raw_voco_stale_quarantine_20261027/snellius/seg/kits23/voco/pertask/fullft/tumor/tumor (failed/empty, newest 2026-10-27 14:19) -> raw snellius finished, newest 2026-10-27 16:06
- raw_voco_stale_quarantine_20261027/snellius/seg/msd_pancreas/voco/pertask/lora/pancreas/pancreas (partial, newest 2026-10-27 14:21) -> raw snellius finished, newest 2026-10-27 14:41
- raw_voco_stale_quarantine_20261027/snellius/seg/totalseg/voco/joint_lm/dec/config (partial, newest 2026-10-27 14:20) -> raw snellius finished, newest 2026-10-27 15:17
- raw_voco_stale_quarantine_20261027/snellius/seg/totalseg/voco/pertask/dec/other_bones/other_bones (failed/empty, newest 2026-10-27 14:15) -> raw snellius finished, newest 2026-10-27 15:29
- raw_voco_stale_quarantine_20261027/snellius/seg/totalseg/voco/pertask/fullft/brain_spinal_cord/brain_spinal_cord (partial, newest 2026-10-27 14:19) -> raw snellius finished, newest 2026-10-27 14:57
- raw_dup_quarantine_20261029/gaia/fdg/voco/pertask/attn/MELANOMA/MELANOMA (partial, newest 2026-10-29 11:48) -> raw snellius finished, newest 2026-10-29 09:29
- raw_dup_quarantine_20261029/gaia/fdg/voco/pertask/rslora/LYMPHOMA/LYMPHOMA (partial, newest 2026-10-29 11:54) -> raw snellius finished, newest 2026-10-29 16:47
