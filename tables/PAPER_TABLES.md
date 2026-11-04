# Tables in the previous draft's layouts (`tables/paper_tables.py`)

Date: 2026-11-01. `paper_tables.py` writes the tables of the previous draft (`midl_paper/paper/tables/`) with their
previous layouts and file names into `paper/tables/`, from `metrics/out/*.csv` only. It is called at the end of
`make_tables.py` (so its files replace the v2 files of the same name: `tab_key`, `tab_cls_main`, `tab_cls_ladder`,
`tab_seg_modes`, `tab_seeds_cls`, `tab_seeds_seg`; the v2 versions are in `paper_compact_20261101/`) and can be run
alone (`python paper_tables.py`). Python: `~/miniconda3/envs/aifofomo/bin/python`.

Layout sources: `midl_paper/results/scripts/make_tables.py` (table_ladder, table_paired, table_headline, table_methods,
clustered_cis, seg_tables, seg_mode_table, tables_appendix, tables_appendix_seg, table_cohorts) and
`revision_tables.py` (_seed_table, fullft_table, controls_table, compute_matched_table). Same marking rules: bold best,
underline second best at the printed precision (ties share the mark); `\cellcolor{modebest}` on the better of per-task
and joint (no shade on a printed tie); `\mci` cells in `tab_key`.

## Files and inputs

| File | Input (metrics/out) | Notes |
|---|---|---|
| tab_key.tex | label_level.csv (mean, lo3, hi3, clusters_pos/n) | old layout incl. VoCo column and "cohorts > 0"; row **LoRA − Attn** (`cls_lora_attn`, matched head) added after LoRA − Lin. The v2 "All" column was not added (it does not fit the old layout; pooled values are in tab_cis). |
| tab_cis.tex | label_level.csv (lo2/hi2, lo3/hi3) | LoRA − Attn rows added here as well, so the appendix table covers every row of tab_key |
| tab_cls_main.tex, tab_cls_ladder.tex | cohort_means.csv (cls, seed 42) | ladder adds the CVPR26 26 000-step joint rows (no shading on them) |
| tab_seg_ladder.tex | cohort_means.csv (seg, seed 42) | AMOS22 / TotalSegmentator joint = 24 000 steps |
| tab_methods.tex | params.csv, ranks.csv | ranks over slots with every method present; SPECTRE Full FT has no rank; Full FT pooled values (90 slots) shown unmarked |
| tab_headline.tex | metrics_cls.csv, paired_cls.csv | pooled = mean over label–mode pairs (per-task, joint) |
| tab_cls_paired.tex, tab_seg_paired.tex | paired_cls.csv, paired_seg.csv (modes per-task, joint) | |
| tab_seg_budget.tex | cohort_means.csv (pertask, joint_8k, joint) | |
| tab_seg_modes.tex | metrics_seg.csv (seed 42) | mean over structures of the per-structure difference |
| tab_seeds_cls.tex, tab_seeds_seg.tex | seeds.csv (means), metrics_*.csv (per-seed signs) | |
| tab_fullft_fix.tex | metrics_cls.csv (joint, seed 42; val split for the choice; pred_sd for grey) | |
| tab_controls.tex | metrics_cls.csv (joint, seed 42) | |
| tab_seg_cm.tex | metrics_seg.csv (pertask_cm, pertask, joint) | |
| tab_app_all.tex + 10 tab_app_*.tex | label_ci.csv (cls) | TAP-CT, SPECTRE, Pillar-0 (as before; VoCo per label is in the v2 file tab_perlabel_cls_voco.tex); breast MRI not written |
| tab_seg_app_all.tex + 5 tab_seg_app_*.tex | label_ci.csv (seg) | |
| tab_cohorts.tex | none (manifest counts) | training-split sizes are not in the per-case files; the counts are those of the v2 appendix table "Cohorts and datasets" and report/code_map_classification.md |

`label_ci.csv` and `params.csv` were added to `metrics/recompute.py` for these tables (PROCEDURE.md section 13); the
rerun with the cache left every other output byte-identical (`checks.txt` gains two row counts).

## Definitions that differ from the old scripts

1. **Per-label intervals (tab_app_*, tab_seg_app_*)**: 1000 case-bootstrap resamples as before, but classification
   now resamples positives and negatives separately (stratified; every resample has an AUROC) and treats an FDG-PET/CT
   study as a case; the old harvest resampled all cases (patients as clusters on FDG) and dropped single-class
   resamples. New random streams. Checked on RSNA 2022 C3, TAP-CT joint Attn (5 positives in 202): an unstratified
   bootstrap reproduces the old interval ([0.12, 0.80] vs old [0.11, 0.79]); the stratified one gives [0.21, 0.72].
   Segmentation: same procedure (valid cases), new random stream.
2. **BH families (tab_cls_paired, tab_seg_paired)**: as REPORT.md section 4. Verified for classification: all 3120
   DeLong differences and p-values (TAP-CT/SPECTRE/Pillar-0 vs Lin and vs Full FT) equal the old ones (to 5e-7, the
   old CSV precision); BH within (baseline, VoCo yes/no) over the old family (2306 tests per family, including the
   26 000-step joint and breast MRI rows) reproduces the old `p_bh` (to 2e-6), and over the new family (1560 tests)
   the new `p_bh` (to 2e-10). So the six changed classification rates come only from the family composition.
   Segmentation: 4000 instead of 1000 paired-bootstrap resamples and a new seed, so p-values near 0.05 move.
3. **Label-level intervals (tab_key, tab_cis)**: new random streams (REPORT.md section 4); means, label counts and
   cohort counts unchanged.
4. **Grey cells in tab_fullft_fix**: the caption's rule (a run whose held-out predictions have a median standard
   deviation over its labels below 0.01) is now applied to every Full FT run, the default setting included. Run-level
   medians: SPECTRE CVPR26 default 0.0014, TAP-CT FDG-PET/CT default 0.0053 (now grey), SPECTRE FDG / RSNA22 / RSNA23
   default 0.026 / 0.011 / 0.023 (not grey; the method-level SPECTRE Full FT median over all its per-task and joint
   label runs is 0.004, which is why it "did not train"), SPECTRE accumulation CVPR26 / RSNA22 / RSNA23 0.000 / 0.0035 /
   0.0024 (grey, as before). The old script only tested the revision runs: its intended method-level test
   (`(bb, mt) in DID_NOT_TRAIN`) never fired because `revision_tables` imported a second copy of `make_tables` with an
   empty set.
5. **tab_cohorts, AMOS training cases**: 1463 distinct cases over the 15 per-label CSVs (v2 definition) instead of
   1107 (the rows of one label file, `ascites.csv`); CVPR26 total 2771 instead of 2415. Selection and held-out counts
   are the manifest counts (FDG-PET/CT 122 held-out studies, RSNA 2022 202 selection cases); the per-case files score
   121 and 201 (`numCohortFdgTest`, `numCohortRsnaBZBBVal`).
6. **Parameter counts (tab_methods)**: read by recompute.py from the training logs of main-design runs; identical to
   the old `trainable_params.csv` (which used smoke runs for four pairs). The logs also hold VoCo counts (e.g. Full FT
   90.1 M trainable, encoder 73.9 M), which report/code_map_classification.md lists as missing.

## Cell-by-cell comparison with the old tables

Script: rows aligned by table, group and leading cells; every cell compared as printed. Full list (all 1772 differing
cells, including every appendix cell): `tables/paper_tables_diffs.csv` (columns table, row, column, old, new, kind).

Tables identical in every cell to the old ones: tab_cls_main, tab_cls_ladder, tab_seg_ladder, tab_methods,
tab_headline, tab_seg_budget, tab_seg_modes, tab_seeds_cls, tab_seeds_seg, tab_controls, tab_seg_cm, tab_app_all,
tab_seg_app_all (wrappers). No point estimate differs in any table. Differences:

| Table | Differing cells | What | Reason |
|---|---|---|---|
| tab_key | 27 + 1 new row | interval ends (by ≤ 0.007) | new random streams (3) |
| tab_cis | 67 + 5 new rows | interval ends (by ≤ 0.007) | new random streams (3) |
| tab_cls_paired | 6 | BH significance rates (1–2 points) | BH family (2) |
| tab_seg_paired | 8 | BH significance rates (1–4 points) | 4000 resamples, new seed (2) |
| tab_fullft_fix | 2 | grey added (TAP-CT FDG default, SPECTRE CVPR26 default) | rule applied to every run (4) |
| tab_cohorts | 2 | AMOS / CVPR26 training cases | distinct AMOS cases (5) |
| tab_app_* (10) | 1389 | interval ends (median 0.01, max 0.15; the 25 cells changed by 0.05 or more are all RSNA 2022 C3, 5 held-out positives) | stratified bootstrap, new stream (1) |
| tab_seg_app_* (5) | 265 | interval ends (max 0.02) | new stream (1) |

#### tab_key.tex (28 cells)

| Row | Column | Old | New |
|---|---|---|---|
| LoRA - Lin | TAP-CT | +0.032 [+0.008, +0.052] | +0.032 [+0.007, +0.053] |
| LoRA - Lin | Pillar-0 | +0.033 [+0.021, +0.045] | +0.033 [+0.021, +0.044] |
| LoRA - Lin | VoCo | -0.071 [-0.170, -0.006] | -0.071 [-0.168, -0.005] |
| LoRA - Attn | (row) |  | LoRA - Attn & +0.022 [+0.006, +0.042] & 5/5 & +0.033 [+0.003, +0.078] & 4/4 & +0.034 [+0.021, +0.052] & 1/1 & -0.031 [-0.145, +0.043] & 1/3 |
| Attn-only FT - LoRA | SPECTRE | +0.002 [-0.038, +0.026] | +0.002 [-0.037, +0.026] |
| Attn-only FT - LoRA | Pillar-0 | +0.001 [-0.001, +0.004] | +0.001 [-0.002, +0.004] |
| Attn-only FT - LoRA | VoCo | +0.036 [-0.026, +0.105] | +0.036 [-0.024, +0.102] |
| Full FT - Lin | TAP-CT | +0.022 [-0.063, +0.067] | +0.022 [-0.061, +0.068] |
| Full FT - Lin | SPECTRE | -0.117 [-0.201, -0.060] | -0.117 [-0.195, -0.059] |
| Full FT - Lin | VoCo | -0.034 [-0.132, +0.025] | -0.034 [-0.136, +0.027] |
| Joint - per-task | TAP-CT | +0.012 [-0.009, +0.035] | +0.012 [-0.014, +0.038] |
| Joint - per-task | SPECTRE | -0.003 [-0.037, +0.052] | -0.003 [-0.044, +0.051] |
| Joint - per-task | Pillar-0 | -0.008 [-0.013, -0.002] | -0.008 [-0.013, -0.001] |
| Joint - per-task | VoCo | -0.001 [-0.025, +0.019] | -0.001 [-0.022, +0.018] |
| Dec - Lin | TAP-CT | +0.085 [+0.055, +0.110] | +0.085 [+0.055, +0.109] |
| Dec - Lin | VoCo | +0.680 [+0.433, +0.747] | +0.680 [+0.433, +0.746] |
| LoRA - Dec | SPECTRE | +0.023 [-0.019, +0.077] | +0.023 [-0.018, +0.076] |
| LoRA - Dec | VoCo | -0.018 [-0.046, +0.015] | -0.018 [-0.047, +0.016] |
|   per-structure models | TAP-CT | -0.021 [-0.035, +0.003] | -0.021 [-0.036, +0.002] |
|   per-structure models | VoCo | -0.035 [-0.063, -0.008] | -0.035 [-0.062, -0.007] |
|   joint models | TAP-CT | +0.011 [-0.020, +0.065] | +0.011 [-0.021, +0.066] |
|   joint models | SPECTRE | +0.040 [-0.040, +0.147] | +0.040 [-0.039, +0.147] |
|   joint models | VoCo | 0.000 [-0.033, +0.048] | 0.000 [-0.034, +0.047] |
| Full FT - Dec | TAP-CT | 0.000 [-0.025, +0.015] | 0.000 [-0.024, +0.016] |
| Full FT - Dec | SPECTRE | -0.003 [-0.036, +0.009] | -0.003 [-0.035, +0.009] |
| Joint - per-task | TAP-CT | -0.046 [-0.092, -0.008] | -0.046 [-0.090, -0.008] |
| Joint - per-task | SPECTRE | -0.058 [-0.093, -0.010] | -0.058 [-0.093, -0.009] |
| Joint - per-task | VoCo | -0.041 [-0.081, +0.003] | -0.041 [-0.081, +0.004] |

#### tab_cis.tex (72 cells)

| Row | Column | Old | New |
|---|---|---|---|
| LoRA - Lin, all | cohorts, labels | [+0.020, +0.049] | [+0.019, +0.049] |
| LoRA - Lin, all | + cases | [+0.017, +0.051] | [+0.017, +0.052] |
| LoRA - Lin, TAP-CT | + cases | [+0.008, +0.052] | [+0.007, +0.053] |
| LoRA - Lin, SPECTRE | cohorts, labels | [+0.014, +0.068] | [+0.013, +0.068] |
| LoRA - Lin, Pillar-0 | + cases | [+0.021, +0.045] | [+0.021, +0.044] |
| LoRA - Lin, VoCo | cohorts, labels | [-0.161, -0.010] | [-0.162, -0.011] |
| LoRA - Lin, VoCo | + cases | [-0.170, -0.006] | [-0.168, -0.005] |
| LoRA - Attn, all | (row) |  | LoRA - Attn & all & 78 & +0.027 & [+0.013, +0.047] & [+0.012, +0.048] & 10/10 |
| LoRA - Attn, TAP-CT | (row) |  | LoRA - Attn & TAP-CT & 39 & +0.022 & [+0.009, +0.039] & [+0.006, +0.042] & 5/5 |
| LoRA - Attn, SPECTRE | (row) |  | LoRA - Attn & SPECTRE & 33 & +0.033 & [+0.007, +0.074] & [+0.003, +0.078] & 4/4 |
| LoRA - Attn, Pillar-0 | (row) |  | LoRA - Attn & Pillar-0 & 6 & +0.034 & [+0.021, +0.051] & [+0.021, +0.052] & 1/1 |
| LoRA - Attn, VoCo | (row) |  | LoRA - Attn & VoCo & 16 & -0.031 & [-0.150, +0.033] & [-0.145, +0.043] & 1/3 |
| Attn-only FT - LoRA, all | cohorts, labels | [-0.005, +0.027] | [-0.005, +0.026] |
| Attn-only FT - LoRA, all | + cases | [-0.007, +0.028] | [-0.007, +0.029] |
| Attn-only FT - LoRA, TAP-CT | cohorts, labels | [+0.001, +0.052] | [+0.002, +0.053] |
| Attn-only FT - LoRA, SPECTRE | cohorts, labels | [-0.037, +0.022] | [-0.037, +0.023] |
| Attn-only FT - LoRA, SPECTRE | + cases | [-0.038, +0.026] | [-0.037, +0.026] |
| Attn-only FT - LoRA, Pillar-0 | + cases | [-0.001, +0.004] | [-0.002, +0.004] |
| Attn-only FT - LoRA, VoCo | cohorts, labels | [-0.021, +0.096] | [-0.020, +0.097] |
| Attn-only FT - LoRA, VoCo | + cases | [-0.026, +0.105] | [-0.024, +0.102] |
| Full FT - Lin, all | cohorts, labels | [-0.101, +0.018] | [-0.101, +0.017] |
| Full FT - Lin, all | + cases | [-0.104, +0.020] | [-0.103, +0.019] |
| Full FT - Lin, TAP-CT | cohorts, labels | [-0.054, +0.066] | [-0.057, +0.064] |
| Full FT - Lin, TAP-CT | + cases | [-0.063, +0.067] | [-0.061, +0.068] |
| Full FT - Lin, SPECTRE | cohorts, labels | [-0.196, -0.065] | [-0.195, -0.063] |
| Full FT - Lin, SPECTRE | + cases | [-0.201, -0.060] | [-0.195, -0.059] |
| Full FT - Lin, Pillar-0 | cohorts, labels | [-0.011, +0.020] | [-0.011, +0.019] |
| Full FT - Lin, VoCo | cohorts, labels | [-0.127, +0.017] | [-0.133, +0.018] |
| Full FT - Lin, VoCo | + cases | [-0.132, +0.025] | [-0.136, +0.027] |
| Joint - per-task, all | cohorts, labels | [-0.017, +0.028] | [-0.017, +0.027] |
| Joint - per-task, all | + cases | [-0.018, +0.030] | [-0.021, +0.030] |
| Joint - per-task, TAP-CT | + cases | [-0.009, +0.035] | [-0.014, +0.038] |
| Joint - per-task, SPECTRE | + cases | [-0.037, +0.052] | [-0.044, +0.051] |
| Joint - per-task, Pillar-0 | cohorts, labels | [-0.013, -0.003] | [-0.012, -0.003] |
| Joint - per-task, Pillar-0 | + cases | [-0.013, -0.002] | [-0.013, -0.001] |
| Joint - per-task, VoCo | cohorts, labels | [-0.017, +0.013] | [-0.017, +0.012] |
| Joint - per-task, VoCo | + cases | [-0.025, +0.019] | [-0.022, +0.018] |
| Dec - Lin, all | + cases | [+0.049, +0.094] | [+0.049, +0.092] |
| Dec - Lin, TAP-CT | + cases | [+0.055, +0.110] | [+0.055, +0.109] |
| Dec - Lin, SPECTRE | cohorts, labels | [+0.028, +0.087] | [+0.028, +0.089] |
| Dec - Lin, VoCo | cohorts, labels | [+0.431, +0.746] | [+0.430, +0.747] |
| Dec - Lin, VoCo | + cases | [+0.433, +0.747] | [+0.433, +0.746] |
| LoRA - Dec, all | cohorts, labels | [-0.014, +0.045] | [-0.014, +0.046] |
| LoRA - Dec, TAP-CT | cohorts, labels | [-0.021, +0.024] | [-0.021, +0.023] |
| LoRA - Dec, SPECTRE | cohorts, labels | [-0.018, +0.078] | [-0.017, +0.079] |
| LoRA - Dec, SPECTRE | + cases | [-0.019, +0.077] | [-0.018, +0.076] |
| LoRA - Dec, VoCo | cohorts, labels | [-0.046, +0.016] | [-0.045, +0.017] |
| LoRA - Dec, VoCo | + cases | [-0.046, +0.015] | [-0.047, +0.016] |
|   per-structure models, all | + cases | [-0.022, +0.008] | [-0.022, +0.009] |
|   per-structure models, TAP-CT | cohorts, labels | [-0.036, +0.002] | [-0.035, +0.002] |
|   per-structure models, TAP-CT | + cases | [-0.035, +0.003] | [-0.036, +0.002] |
|   per-structure models, SPECTRE | cohorts, labels | [-0.001, +0.031] | [-0.001, +0.028] |
|   per-structure models, VoCo | cohorts, labels | [-0.061, -0.008] | [-0.062, -0.008] |
|   per-structure models, VoCo | + cases | [-0.063, -0.008] | [-0.062, -0.007] |
|   joint models, all | cohorts, labels | [-0.021, +0.094] | [-0.020, +0.092] |
|   joint models, TAP-CT | cohorts, labels | [-0.021, +0.066] | [-0.019, +0.067] |
|   joint models, TAP-CT | + cases | [-0.020, +0.065] | [-0.021, +0.066] |
|   joint models, SPECTRE | cohorts, labels | [-0.038, +0.146] | [-0.038, +0.148] |
|   joint models, SPECTRE | + cases | [-0.040, +0.147] | [-0.039, +0.147] |
|   joint models, VoCo | cohorts, labels | [-0.033, +0.048] | [-0.034, +0.047] |
|   joint models, VoCo | + cases | [-0.033, +0.048] | [-0.034, +0.047] |
| Full FT - Dec, all | cohorts, labels | [-0.018, +0.008] | [-0.017, +0.009] |
| Full FT - Dec, TAP-CT | + cases | [-0.025, +0.015] | [-0.024, +0.016] |
| Full FT - Dec, SPECTRE | + cases | [-0.036, +0.009] | [-0.035, +0.009] |
| Joint - per-task, all | cohorts, labels | [-0.081, -0.022] | [-0.082, -0.021] |
| Joint - per-task, all | + cases | [-0.082, -0.021] | [-0.082, -0.022] |
| Joint - per-task, TAP-CT | cohorts, labels | [-0.091, -0.009] | [-0.088, -0.009] |
| Joint - per-task, TAP-CT | + cases | [-0.092, -0.008] | [-0.090, -0.008] |
| Joint - per-task, SPECTRE | cohorts, labels | [-0.094, -0.011] | [-0.092, -0.010] |
| Joint - per-task, SPECTRE | + cases | [-0.093, -0.010] | [-0.093, -0.009] |
| Joint - per-task, VoCo | cohorts, labels | [-0.080, +0.004] | [-0.081, +0.002] |
| Joint - per-task, VoCo | + cases | [-0.081, +0.003] | [-0.081, +0.004] |

#### tab_cls_paired.tex (6 cells)

| Row | Column | Old | New |
|---|---|---|---|
| TAP-CT, vs Lin, rsLoRA | Sig.\ worse (%) | 4 | 5 |
| TAP-CT, vs Lin, Full FT | Sig.\ better (%) | 12 | 13 |
| TAP-CT, vs Lin, Full FT | Sig.\ worse (%) | 6 | 8 |
| TAP-CT, vs Full FT, Attn | Sig.\ worse (%) | 17 | 18 |
| TAP-CT, vs Full FT, DoRA | Sig.\ better (%) | 15 | 17 |
| SPECTRE, vs Full FT, DoRA | Sig.\ better (%) | 42 | 44 |

#### tab_seg_paired.tex (8 cells)

| Row | Column | Old | New |
|---|---|---|---|
| TAP-CT, vs Lin, Dec | Sig.\ better (%) | 95 | 93 |
| TAP-CT, vs Lin, DoRA | Sig.\ better (%) | 88 | 90 |
| TAP-CT, vs Dec, LoRA | Sig.\ better (%) | 30 | 31 |
| SPECTRE, vs Dec, LoRA | Sig.\ better (%) | 61 | 57 |
| SPECTRE, vs Dec, LoRA_{\text{late}} | Sig.\ better (%) | 44 | 46 |
| SPECTRE, vs Dec, DoRA | Sig.\ better (%) | 58 | 54 |
| SPECTRE, vs Dec, DoRA | Sig.\ worse (%) | 14 | 12 |
| SPECTRE, vs Dec, Dec\toLoRA | Sig.\ better (%) | 22 | 20 |

#### tab_fullft_fix.tex (2 cells)

| Row | Column | Old | New |
|---|---|---|---|
| FDG-PET/CT | default | 0.608 | \textcolor{gray}{0.608} |
| CVPR26 | default | 0.575 | \textcolor{gray}{0.575} |

#### tab_cohorts.tex (2 cells)

| Row | Column | Old | New |
|---|---|---|---|
| \textbf{CVPR26} (CVPR 2026 CT foundation-model challenge) | Train | \textbf{2415} | \textbf{2771} |
|   AMOS (disease labels) | Train | 1107 | 1463 |

#### Per-label appendix tables

| Table | Cells with a different interval | Point estimate differs | Largest change of an interval end |
|---|---|---|---|
| tab_app_amos_covid_luna_spectre_3.tex | 133 | 0 | 0.03 |
| tab_app_amos_covid_luna_spectre_4.tex | 132 | 0 | 0.03 |
| tab_app_amos_covid_luna_spectre_5.tex | 130 | 0 | 0.04 |
| tab_app_amos_covid_luna_tapct_0.tex | 141 | 0 | 0.03 |
| tab_app_amos_covid_luna_tapct_1.tex | 157 | 0 | 0.04 |
| tab_app_amos_covid_luna_tapct_2.tex | 152 | 0 | 0.04 |
| tab_app_fdg.tex | 133 | 0 | 0.05 |
| tab_app_rsna2019_ich.tex | 41 | 0 | 0.01 |
| tab_app_rsna2022.tex | 274 | 0 | 0.15 |
| tab_app_rsna2023.tex | 96 | 0 | 0.03 |
| tab_seg_app_amos22.tex | 96 | 0 | 0.01 |
| tab_seg_app_kits23.tex | 39 | 0 | 0.02 |
| tab_seg_app_lits.tex | 27 | 0 | 0.02 |
| tab_seg_app_msd_pancreas.tex | 26 | 0 | 0.01 |
| tab_seg_app_totalseg.tex | 77 | 0 | 0.01 |

## Suggested captions (adapted from the old captions to the corrected content)

- **tab_key**: Label-level comparisons of the main methods: mean difference, 95\% interval over cohorts, labels and held-out cases, and the number of cohorts (datasets in segmentation) with a positive mean difference, per backbone. Per-task and joint differences are averaged per label except in the two LoRA $-$ Dec sub-rows. LoRA $-$ Attn compares two methods with the same attentive head, so it isolates what LoRA trains in the encoder; Full FT $-$ Lin keeps the SPECTRE runs that did not train. VoCo covers three classification cohorts (\numCiClsloralinVocoLabelsN{} labels) and all segmentation datasets; Pillar-0 was run on classification only. Appendix Table~\ref{tab:cis} adds the interval without the case stage and the estimates pooled over TAP-CT, SPECTRE and Pillar-0.
- **tab_cls_main**: Held-out AUROC of the classification methods, mean over the labels of each cohort (number of labels in brackets), per backbone and training mode. Within each row, \textbf{bold} is the best and \underline{underlined} the second-best method; a \colorbox{modebest}{shaded} cell marks the better of per-task and joint training for that method, backbone and cohort. SPECTRE Full FT did not train (Section~\ref{sec:clsres}). VoCo was run on three cohorts.
- **tab_seg_ladder**: Held-out Dice of the segmentation methods, mean over the structures of each dataset (number of structures in brackets), per backbone and training mode; -- marks methods not run per structure on TotalSegmentator. The AMOS22 and TotalSegmentator joint models are trained for 24\,000 steps. Marks and shading as in Table~\ref{tab:cls}. VoCo's Lin reads stage-4 tokens at a stride of 32 voxels.
- **tab_methods**: Classification methods: trainable parameters of a single-label model (head included, from the training logs); mean rank within each of the \numRankSlotsN{} label--mode pairs ($\downarrow$; on SPECTRE among the ten methods that trained); share of pairs within 0.01 AUROC of the best method ($\uparrow$); mean AUROC deficit to the best method ($\downarrow$). \textbf{Bold}: best, \underline{underlined}: second best of each rank and score column. Full FT's pooled values exclude SPECTRE, where it did not train, so they are not comparable with the other methods' and are not marked. Choosing the method per label on the selection split finds the held-out best in \numSelectHit\% of pairs and lowers the mean deficit from $\numRegretAttnft$ (always attention-only fine-tuning) to $\numSelectRegret$, at eleven runs per label instead of one.
- **tab_headline**: Pooled held-out AUROC per backbone and paired per-label comparisons of LoRA with the linear probe and with full fine-tuning. Pooled AUROC is the mean over label--mode pairs ($n$ = labels $\times$ \{per-task, joint\}); the comparisons are DeLong tests with Benjamini--Hochberg correction within each baseline (``sig.'' = corrected $p<0.05$ in the stated direction). SPECTRE Full FT did not train. The mean gain of LoRA over the probe is similar for small and large held-out splits ($\numSigbysizeSmallMeandiff$ below 100 cases, $\numSigbysizeLargeMeandiff$ above 400) but significant in \numSigbysizeSmallSig\% and \numSigbysizeLargeSig\% of these comparisons; every comparison above 400 cases is from the hemorrhage cohort.
- **tab_cis**: Label-level differences with 95\% intervals. Each label (or structure) counts once, with its per-task and joint differences averaged (except the two LoRA $-$ Dec sub-rows); LoRA $-$ Attn compares methods with the same attentive head. Intervals are percentile bootstraps (4000 resamples) over cohort--backbone groups and labels within them, without and with a third stage that resamples the held-out cases of each drawn label (1000 resamples per label; positives and negatives separately in classification). ``all'' pools TAP-CT, SPECTRE and Pillar-0; Pillar-0 has one cohort, so its interval resamples labels and cases only. The last column counts the cohorts (datasets) with a positive mean difference.
- **tab_cls_paired**: Per-label paired comparisons of held-out AUROC against the linear probe and against full fine-tuning, over the per-task and joint modes (DeLong test; Benjamini--Hochberg correction within each baseline over the per-task and joint comparisons of TAP-CT, SPECTRE and Pillar-0). ``Better'' is the share of label--mode pairs in which the method's AUROC exceeds the baseline's; ``Sig.'' the share in which, in addition, the corrected $p<0.05$. SPECTRE Full FT did not train.
- **tab_seg_paired**: Per-structure paired comparisons of held-out Dice over the per-task and joint modes against the linear probe and the frozen decoder (paired bootstrap of the mean per-case difference, 4000 resamples; Benjamini--Hochberg correction within each baseline). Pairing keeps the cases on which a structure enters the Dice of both methods (present in the reference or in either prediction), so the mean differences here can differ from the differences of structure means in Table~\ref{tab:seg}.
- **tab_seg_budget**: Training budget of the 15-structure joint models (AMOS22, TotalSegmentator): held-out Dice, mean over structures, of one model per structure (8000 steps each), one joint model under the shared 8000 steps, and one joint model with three times the steps (Dec$\to$LoRA: 12\,000 + 12\,000).
- **tab_seg_modes**: Encoder adaptation on top of the frozen decoder per dataset, backbone and mode: mean over structures of the difference in held-out Dice. Joint rows of AMOS22 and TotalSegmentator use the 24\,000-step schedule; the 8000-step rows show the same models under the shared budget.
- **tab_seeds_cls**: Held-out AUROC of the joint classification models across seeds 42, 43 and 44: mean over the cohort's labels, given as mean $\pm$ standard deviation over the seeds (their number in brackets), and the number of seeds in which the mean label difference between two methods is positive. SPECTRE Full FT, which did not train, has seed 42 only.
- **tab_seeds_seg**: Held-out Dice of the joint segmentation models across seeds (AMOS22 and TotalSegmentator: 24\,000 steps), mean over structures, as in Table~\ref{tab:seedscls}.
- **tab_fullft_fix**: Held-out AUROC of the joint classification models under the default and three alternative full fine-tuning settings, mean over the cohort's labels, next to Lin and LoRA. \textbf{Bold}: the setting with the best mean selection-split AUROC on the backbone. Grey: runs whose held-out predictions barely vary (median over labels of the standard deviation of the predicted probabilities below \numSpreadMinValue{}).
- **tab_controls**: Controls for two confounds of the main comparison: rsLoRA at a quarter of the learning rate, which matches LoRA's update scale, and fine-tuning the MLP instead of the attention weights. Held-out AUROC of the joint models, mean over the cohort's labels.
- **tab_seg_cm**: Joint against compute-matched per-structure segmentation: held-out Dice, mean over the 15 structures, of per-structure models trained for 1600 steps (matched) or 8000 steps, and of the 24\,000-step joint model on the same structures; the difference joint $-$ matched and the number of structures in which it is positive.
- **tab_cls_ladder**: Held-out AUROC as in Table~\ref{tab:cls}, with the longer CVPR26 joint schedule (26\,000 steps) as an extra row. \textbf{Bold}: best, \underline{underlined}: second best of the row; \colorbox{modebest}{shaded}: the better of per-task and joint training (8000 steps). SPECTRE Full FT did not train.
- **tab_app_all** (lead-in text): Held-out AUROC with 95\% bootstrap intervals (1000 resamples of the held-out cases, positives and negatives resampled separately) for every label, mode and method on TAP-CT, SPECTRE and Pillar-0.
- **tab_seg_app_all** (lead-in text): Held-out Dice with 95\% bootstrap intervals (1000 resamples of the cases in which the structure is present in the reference or the prediction) for every structure, backbone, mode and segmentation method (AMOS22 and TotalSegmentator joint models: 24\,000 steps). In a case without the structure in the reference, any predicted voxel gives a Dice of 0, so small structures such as tumors carry false positives into their mean.
- **tab_cohorts**: Classification cohorts: labels and cases per split as listed in the manifests (training; selection = checkpoint selection; held-out = reported). CVPR26 is the CVPR 2026 CT foundation-model challenge release; its three sources are trained as one 17-label cohort, and its AMOS training count is the number of distinct cases over the 15 per-label files (each label is known for a subset). FDG-PET/CT counts are studies (1014 studies of 900 patients). The per-case prediction files hold \numCohortFdgTest{} FDG-PET/CT held-out studies and \numCohortRsnaBZBBVal{} RSNA 2022 selection cases, the cases every comparison uses.
