# Results pipeline and statistical procedure (as implemented)

Source: `midl_paper/results/scripts/{harvest_metrics.py, harvest_revision.py, case_level.py, make_tables.py, revision_tables.py}`.
Everything below is read from the code; line numbers refer to the files as of 2026-10-31.

## 1. What a "finished run" is and which prediction file is scored
- Classification (`harvest_metrics.harvest_run`, l.400-466): a run is scored only if `checkpoints/best/best_val_metric_val_auroc.txt` exists (selected iteration = best validation macro AUROC) and `val_case_metrics/iter_<best>/logits.csv` exists. Status `ok` requires `checkpoints/last_checkpoint.txt` to reach the YAML's final iteration (`copy_status`, l.391-397), otherwise `INCOMPLETE`; `make_tables.load` (l.96-112) drops INCOMPLETE rows.
- `logits.csv` rows: task, task_index, split (val/test), case_id, label, probability. Labels with a single class in val or test are skipped (l.439-441). A leakage check flags case ids present in both splits (l.444-446). The recomputed validation AUROC is compared with the logged in-training AUROC per task_index (tolerance 2e-3, l.448-455) to verify the task_index -> label-name mapping.
- Segmentation (`harvest_seg`, l.735-836): selected iteration from `best_val_metric_val_dice.txt`; per-case file `val_case_metrics/iter_<best>/dice.csv` (columns incl. split, class_index, case_id, dice, valid); only `valid == 1` rows enter; per-structure value = mean over valid held-out cases. When a run exists on both sites, the first `ok` copy in site order is used (l.757).

## 2. Point metrics
- AUROC: Mann-Whitney with midranks (ties 0.5) (`auroc_rows`, l.196-207); NaN if one class. Other metrics computed but not used in the paper: AUPRC, balanced accuracy, sensitivity, specificity, F1 (threshold chosen on the selection split, `best_threshold`), Brier, ECE (10 bins).
- Dice: read from the training code's per-case files (thresholding and empty-reference handling are defined in the aifofomo metric; see the segmentation code map). Harvest-level aggregation: mean over valid cases.
- Per-row 95% CIs in master_metrics: cluster bootstrap over case ids (FDG: patient id with several studies, rows cycled/truncated, `expand_units` l.343-366), `n_boot` resamples shared across methods of the same (dataset, mode, split) via a deterministic RNG keyed on the group (`resample_matrix`, l.333-340).

## 3. Paired tests per label--mode pair (`paired_rows`, l.545-621; seg l.800-831)
- Classification: DeLong two-sided test (Sun & Xu fast midrank version, l.290-330) of every method against each baseline in `BASELINES = ["lin", "fullft"]` on the held-out split; cases aligned on common ids (label/case mismatch -> skipped with a note). Paired bootstrap for the non-AUROC metrics with shared resamples.
- Segmentation: paired case bootstrap of the mean Dice difference against `SEG_BASELINES = ["lin", "dec"]`; p = two-sided bootstrap p (`boot_p`, l.376-381).
- Multiplicity: Benjamini-Hochberg within each (test, metric, baseline, is_voco) family (`add_bh`, l.624-644). VoCo is its own family.
- NOTE: paired comparisons exist only against Lin and Full FT (cls) and Lin and Dec (seg). Any "X vs LoRA" paired statement in the draft (e.g. Attn-only FT vs LoRA) is NOT backed by a paired test in `paired_tests.csv`; `make_tables.vs_lora_direct` computes plain differences instead.

## 4. Label-level estimates with intervals (`make_tables.clustered_cis`, l.656-764; `case_level.py`)
- Unit: one label (or structure) on one backbone; its value is a difference of two methods averaged over the per-task and joint modes (LoRA-Lin, Attn-only FT-LoRA, Full FT-Lin, Dec-Lin, LoRA-Dec, Full FT-Dec), or joint minus per-task averaged over the methods trained in both modes.
- Interval: percentile bootstrap, 4000 resamples, two stages (resample cohort--backbone clusters, then labels within the cluster) and a third stage that replaces each drawn label's value by one of its 1000 case-level resamples (`label_boot_cls`: positives and negatives resampled separately with identical case indices for every term; `label_boot_seg`: all cases, NaN-mean) (`_clustered_ci`, l.624-639). RNG seed 20261025; VoCo uses `default_rng([20261027, *key])` per comparison.
- Also reported: number of cohort--backbone clusters with positive mean difference. Pooled ("all") estimates use `MAIN_BACKBONES = [tapct, spectre, pillar0]` only (l.709). The Full FT-Lin row keeps runs that did not train (`m_all`, l.674).
- Exclusions: label `lesion_malignant` and cohort `breastdivider` (l.69-72); segmentation modes remapped so that for AMOS22/TotalSegmentator the 24000-step `joint_lm` run is the paper's "joint" and the 8000-step run is `joint_8k` (l.82-93).

## 5. Ranks (`table_methods`, l.975-1057)
- Slot = backbone x cohort x label x mode (per-task, joint) with every expected method present (`piv.notna().sum(axis=1) == expected`, where expected excludes `DID_NOT_TRAIN` pairs); methods ranked by held-out AUROC, ties share ranks (pandas average rank). Mean rank per backbone and overall over MAIN_BACKBONES; also "within 0.01 of best" and mean deficit. Segmentation ranks analogous in `seg_tables`.

## 6. "Did not train" rule (`spread_macros`, l.420-440; `case_level.prediction_spread`)
- Per (backbone, method): median over label--mode pairs of the SD of held-out probabilities; < 0.01 -> `DID_NOT_TRAIN` (currently only SPECTRE/fullft). Such pairs stay in per-cohort tables and in Full FT-Lin, and leave ranks, method averages and summary figures (`drop_did_not_train`). Not applied to VoCo (`voco_macros`).

## 7. Revision runs (`harvest_revision.py`, `revision_tables.py`)
- Seeds 43/44 of joint models for Lin, LoRA, Attn-only FT, Full FT (cls) and Dec, LoRA, Full FT (seg); full fine-tuning variants `fullft_lowlr`, `fullft_llrd`, `fullft_accum`; controls `mlpft`, `rslora_lowlr`; segmentation `pertask_cm` (compute-matched per-structure models) and `scratch`. AUROC via sklearn `roc_auc_score` here (equivalent to midrank AUROC). Status rule as above.

## 8. Discrepancies and checks in the pipeline (draft vs code)
- P1 (checked, no discrepancy): two long joint schedules exist. Classification CVPR26 `joint_lm` = 26 000 steps (`fm_adapt/generate.py` `LM_STEPS = 26000`, YAML `max_iterations: 26000`), printed as "joint (26 000 steps)" in the appendix tables; segmentation AMOS22/TotalSegmentator `joint_lm` = 24 000 steps (`fm_seg/generate.py` `LM_STEPS = 24000`), remapped to "joint" in every segmentation table. The rewrite must name both schedules explicitly.
- P2 (checked, no discrepancy): the 24 000-step two-stage segmentation child splits 12 000 + 12 000 and starts from step 12 000 of the Dec run (draft appendix l.252, l.327 state this; the main-text sentence "4000 + 4000" at l.85 covers only the 8000-step runs and should say so).
- P3: paired tests exist only against Lin and Full FT (classification) and Lin and Dec (segmentation). Any paired/significance statement about another baseline would be unsupported; none found in the current draft, but the recompute must keep this restriction or add the missing baselines (e.g. LoRA as baseline for Attn-only FT and the LoRA variants).
- P4: `harvest_metrics.SEED = 42` is written into every master row regardless of the run; seed repeats live only in `revision_metrics.csv`. The recompute script should carry the real seed per run.
- P5: the per-row case-bootstrap CIs in `master_metrics*.csv` are not used by any table or macro (label-level intervals come from `clustered_cis`); keep only as a cross-check.
- P6: ranks require every expected method in a slot; a method missing on one label removes that whole label--mode slot from the ranking (`table_methods`). The rewrite must state the number of slots and this rule.

## 9. Plan for the single recompute script (Phase 2)
Inputs: `results/raw/**/val_case_metrics/iter_<best>/{logits,dice}.csv` at the best iteration (from `best_val_metric_*.txt`), the YAML design for label names and expected runs, `revision` runs for seeds. Outputs: per-run per-label metrics, paired tests (DeLong + paired bootstrap + BH), label-level estimates with 2- and 3-stage intervals, ranks, spreads, convergence (selected iteration), and a `changes_vs_draft.csv` that diffs every macro in `paper/tables/numbers.tex` against the recomputed value.
