# Statistical procedure of `metrics/recompute.py`

Everything the paper quotes is recomputed by one script from the per-case prediction files of the finished runs. This
file states every step precisely enough for the appendix. File formats and column names are in `SCHEMA.md`; macro names
in `NAMING.md`.

## 1. Runs

- Source: `report/run_inventory.csv` (one row per run directory of the rsync mirror `midl_paper/results/raw/<site>/`).
  A run is used when `status == finished` (the best-validation record exists, the per-case file of the selected
  iteration exists with held-out rows, and the last checkpoint reached the YAML's `num_iterations`), `in_design == yes`
  (a YAML of the fm_adapt / fm_seg design, or one of the two fm_seg runs enumerated by `generate.iter_runs()` whose YAML
  was never written), `excluded` empty (the 130 quarantined directories are not used) and `aux` empty (smoke tests and
  scratch evaluations are not used). Runs finished on both sites would be deduplicated by run path (first in site order
  gaia, snellius); none is at present.
- Selected iteration: the `iteration:` line of `checkpoints/best/best_val_metric_val_auroc.txt` (classification) or
  `best_val_metric_val_dice.txt` (segmentation), i.e. the checkpoint with the best selection-split macro metric
  (strict improvement; ties keep the earlier checkpoint). Scored file: `val_case_metrics/iter_<selected>/logits.csv` or
  `dice.csv`.
- The true training seed is carried per run (42 for the main design, 43 / 44 for the seed repeats `joint_s43`,
  `joint_s44`).
- Reporting modes. Classification: `pertask`, `joint` (8000 steps), `joint_lm` (CVPR26 cohort, 26 000 steps); the seed
  repeats are reported as `joint` with seed 43 / 44. Segmentation: for AMOS22 and TotalSegmentator (15 structures) the
  24 000-step `joint_lm` run is the paper's `joint` and the 8000-step joint run is `joint_8k`; for KiTS23, LiTS and MSD
  Pancreas `joint` is the 8000-step run; `pertask_cm` are the compute-matched 1600-step per-structure runs; seed repeats
  (24 000 steps on AMOS22 / TotalSegmentator, 8000 on the others) are `joint` with seed 43 / 44.
- Exclusions: the breast MRI cohort `breastdivider` and the label `lesion_malignant` are left out of every output
  except `runs.csv` and `metrics_cls.csv` (the study is CT-only; the cohort has one or two negatives in the held-out
  split). Methods of the main design: classification Lin, Attn, LoRA, LoRA-late, LoRA+ABMIL, DoRA, rsLoRA,
  Lin->LoRA, Attn->LoRA, Attn-only FT, Full FT; segmentation Lin, Dec, LoRA, LoRA-late, DoRA, Full FT, Dec->LoRA.
  Revision methods (full fine-tuning variants `fullft_lowlr`, `fullft_llrd`, `fullft_accum`, `mlpft`, `rslora_lowlr`;
  segmentation `scratch`, `pertask_cm`) enter only Section 8.

## 2. Per-run metrics (`metrics_cls.csv`, `metrics_seg.csv`)

- Label names: the `task_names` list of the run's YAML in `task_index` order (`amos:gallstone`, `covid_ct:covid`,
  `luna25:lung_nodule_malignancy`, `NEGATIVE`, `C1`, `any_injury`, ...). Structure names: the channel order of
  `fm_seg/generate.py` `DATASETS[ds]["channels"]` by `class_index` (per-structure runs: the single structure).
  The mapping is verified for every joint classification run against the per-label selection AUROC logged in
  `logs/metrics.json` at the selected iteration (tolerance 2e-3).
- Classification: per label and split (`val` = selection, `test` = held-out), AUROC = Mann-Whitney statistic with
  midranks (tied scores count one half), AUPRC = average precision (sum over distinct score thresholds of
  recall increment x precision, as `sklearn.metrics.average_precision_score`), `n` rows, `n_pos` positives, `pred_sd` =
  standard deviation of the predicted probabilities. A label is skipped for a run when its selection or held-out split
  has fewer than two rows or a single class (listed in `out/notes.txt`). FDG-PET/CT rows are studies (a patient with
  several studies appears with repeated case ids); every analysis treats a row as a case.
- Segmentation: per structure and split, Dice = mean over the cases flagged `valid == 1` (the training code sets
  `valid` when prediction or reference is non-empty; an empty reference with an empty prediction is not counted; an
  empty reference with a non-empty prediction counts as Dice 0). `n_valid` cases.
- Verification: the recomputed selection-split macro metric (classification: mean AUROC over labels with two classes
  and >= 2 rows; segmentation: mean over structures of the mean Dice over valid cases) is compared with the logged
  best value of every run (`out/selection_check.csv`, tolerance 2e-3; the printout lists 5 random runs per task).
  All 1354 classification runs agree to 3e-8; 1033 of 1034 segmentation runs agree to 2e-3, the exception being
  `msd_pancreas/tapct/pertask/lin/tumor` (2.01e-3), whose per-case file was re-inferred from the selected checkpoint
  after training (`reinfer_dice.py`, tolerance 2e-3) and therefore differs from the in-training value by GPU
  non-determinism; the run is kept.

## 3. "Did not train" rule (`spread.csv`)

Per (backbone, method) of the main design: the median over label-mode pairs (modes per-task and joint, held-out
split, seed 42) of `pred_sd`, each label's SD taken over its own rows (the draft's `case_level.prediction_spread`
selected rows of a joint run by the `task` column, which is shared by every label of the run, so its spreads of joint
runs were those of the first label; the decision is unchanged). A pair with a median below 0.01 did not train. The rule is applied to TAP-CT, SPECTRE and
Pillar-0 only (at present SPECTRE / Full FT); VoCo spreads are reported but not applied. Pairs that did not train stay in
the per-run metrics, the per-cohort means, the paired tests and the Full FT - Lin label-level estimate; they leave the
ranks, every method-level average (`vs*`, `bestrows`, `jointdelta`, `cohortmode`, `select`, `gainbin`, `bbdiff`,
`sigcohort`, `ap`) and the other label-level estimates.

## 4. Paired tests (`paired_cls.csv`, `paired_seg.csv`)

Group = (backbone, cohort, mode in {per-task, joint}, label), seed 42, held-out split. Every method is compared with
each baseline (classification: Lin, LoRA, Attn, Full FT; segmentation: Lin, Dec, LoRA).

- Pairing requires identical case sets: positional when both runs list the same case ids in the same order, otherwise
  by sorted case id when both sets are identical without repeats; pairs whose case sets or labels differ are skipped
  and listed in `out/notes.txt`.
- Classification: two-sided DeLong test of AUROC(method) - AUROC(baseline) on the same cases, fast structural-component
  implementation with midranks (Sun & Xu, IEEE SPL 2014); columns `diff`, `se`, `p`.
- Segmentation: cases valid for both runs enter (the `note` column records when the valid sets differ). Paired case
  bootstrap of the mean Dice difference, 4000 resamples, `numpy.random.default_rng([20261031, crc32(group)])`, so every
  comparison of a group shares the same resampled case indices; `ci_lo`, `ci_hi` = 2.5 / 97.5 percentiles; p = two-sided
  bootstrap p-value, min(1, 2 min(P(mean <= 0), P(mean >= 0))).
- Multiplicity: Benjamini-Hochberg within each (task, baseline, VoCo yes/no) family -> `p_bh`; "significant" in the
  paper means `p_bh < 0.05`. A family holds the per-task and joint comparisons of the main design only (the draft's
  harvest also had the 26 000-step joint and breast MRI rows in the family, which moves a few BH-significance rates by
  one or two label-mode pairs).

## 5. Label-level estimates (`label_level_values.csv`, `label_level.csv`)

Unit: one label (or structure) on one backbone. For a comparison A - B the label's value is the mean over the per-task
and joint modes of AUROC_A - AUROC_B (Dice for segmentation); the label enters only when all four runs exist (two for a
single-mode key). For joint - per-task the value is the mean over the methods trained in both modes of
value(joint) - value(per-task). Keys: `cls_<method>_lin` for every method, `cls_<method>_attn` for every method with the attentive head (matched-head comparison with the frozen attentive probe; added 2026-11-01), `cls_lora_lin_nosc` (LoRA - Lin without the FDG-PET/CT cohort and the COVID-CTset label; added 2026-11-01), `cls_attnft_lora`, `cls_joint_pertask`,
`seg_<method>_dec` for every method, `seg_lora_dec_pertask`, `seg_lora_dec_joint`, `seg_joint_pertask`. Pairs that did
not train are dropped except for `cls_fullft_lin` (which keeps them: the paper's Full FT - Lin statement describes
full fine-tuning as run).

- Mean = mean over the labels of a backbone, or pooled over TAP-CT, SPECTRE and Pillar-0 ("all"); `n_labels`;
  `clusters_pos` / `clusters_n` = cohort-backbone clusters with a positive mean difference.
- Case-level draws per label: 1000 resamples with `default_rng([20261025, crc32(key, backbone, cohort, label)])`
  (`20261027` for VoCo). Classification: positives and negatives are resampled separately (every resample has an
  AUROC), with the same row indices for every term of the label's value, so the pairing is kept; segmentation: all
  held-out cases are resampled, each term's mean ignores its invalid cases (NaN-mean). The draws are recentred on the
  label's point value (draw - mean(draws) + value) so that the third stage adds case-level spread without bias.
- Intervals: percentile intervals from 4000 resamples with `default_rng([20261025, crc32(key, backbone)])`
  (`20261027` for VoCo). Two-stage (`lo2`, `hi2`): resample the cohort-backbone clusters with replacement, then the
  labels within each drawn cluster with replacement, take the mean of the drawn label values. Three-stage (`lo3`,
  `hi3`): as two-stage, then replace each drawn label's value by one of its 1000 case-level draws. The 2.5 and 97.5
  percentiles are reported.

## 6. Ranks (`ranks.csv`)

Slot = backbone x cohort x label x mode (per-task, joint), seed 42, with every expected method present (expected = all
methods of the task minus the pairs that did not train on that backbone; VoCo: all methods; a segmentation slot needs
all seven methods, so the TotalSegmentator per-structure slots with four methods are not ranked). Methods are ranked per
slot by held-out AUROC / Dice (1 = best; ties share the average rank). Per backbone and pooled over TAP-CT, SPECTRE and
Pillar-0 ("all"; VoCo separately): mean rank, % of slots within 0.01 of the best method, mean regret (best - method).
A method that did not train on a backbone has no rank there and its pooled values run over the other backbones' slots.
Pooled over all four backbones (`rankall_*`, `segrankall_*`; added 2026-11-02 for the rank panels of Figure 1): the mean of
the same per-slot ranks over every slot of every backbone, VoCo included (classification 156 + 32 = 188 slots,
segmentation 118 + 59 = 177). Each slot is still ranked within its own backbone, so VoCo enters only through its own
slots; no other output changes (every other file in `out/` byte-identical, `numbers.tex` gains 20 macros).

## 7. Convergence (`convergence.csv`)

Per run: `final_iter` = last logged validation iteration, validation interval = final / number of validations;
`frac_last` = share of runs whose selected iteration is the last validation, `frac_last_two` = the last or the one
before. Per (task, backbone, method, mode) and per task. The draft's macros (`conv_*`) use main backbones, seed 42,
two-stage methods (Lin->LoRA, Attn->LoRA, Dec->LoRA) excluded, per-task + joint for classification, and per
schedule group for segmentation (per-structure, joint 8000-step small datasets, joint 24 000-step, joint_8k).

## 8. Seeds and revision runs (`seeds.csv`, `seeds_summary.csv`, `revision.csv`)

- Seeds: cohort means (mean over labels of the held-out metric) of the joint models per seed (42, 43, 44) for Lin, LoRA,
  Attn-only FT, Full FT (classification) and Dec, LoRA, Full FT (segmentation). `seeds_summary.csv`: per method the
  median and maximum over (backbone, cohort) cells with >= 2 seeds of the SD across seeds; per comparison
  (LoRA > Lin, Attn-only FT > LoRA; LoRA > Dec, Full FT > Dec) the number of (cohort, seed) pairs whose mean label
  difference is positive, over all seeds and over seeds 43 / 44 only.
- Full fine-tuning variants: mean selection-split AUROC per backbone and variant (over cohort-label rows), cohorts
  whose run trained (run-level median `pred_sd` >= 0.01); the chosen variant per backbone is the one with the highest
  selection AUROC once every variant has run on every cohort of the backbone; its label-level difference to Lin and
  LoRA and the number of cohorts with a positive mean.
- Controls: MLP-only FT vs Attn-only FT and LoRA; rsLoRA at LR/4 vs LoRA and rsLoRA (label-level differences of the
  joint models, cohorts with a positive mean).
- Compute-matched per-structure models (1600 steps): per dataset, backbone and method the mean over structures of
  joint - matched and of 8000-step per-structure - matched, and the structures where the joint model is ahead.
- Scratch: from-scratch joint models minus the Full FT joint model of the same backbone and dataset.

## 9. Cohort means (`cohort_means.csv`)

Mean over labels (structures) of the held-out metric per (task, backbone, cohort, mode, method, seed); a cell is written
only when every label of the cohort (the labels scored on the main backbones) is present.

## 10. Counts and case numbers

Run counts per task and backbone, labels and structures per backbone, DeLong and paired-bootstrap comparisons, and case
counts per cohort and split from the per-case files (distinct case ids of the TAP-CT joint Lin run, pooled over the
labels of a cohort; FDG-PET/CT is counted in studies (rows: 121 held-out) with the patients given separately (98); the
CVPR26 cohort is split into AMOS, COVID-CT and LUNA25 by label source). These are the cases with at least one label in
the split, which is why FDG-PET/CT has 121 instead of the manifest's 122 held-out studies and RSNA 2022 has 201 instead
of 202 selection cases; training-set sizes are not in the per-case files and are not recomputed.

## 11. Determinism and runtime

Every random number comes from `numpy.random.default_rng` seeded with a fixed integer and the CRC-32 of the group or
key, so a second run writes byte-identical files (checked: every file in `out/` has the same checksum after two runs).
Per-case arrays are cached in `out/_cache/percase.pkl` (keyed by file path, size and mtime); the case-level bootstrap
runs in a process pool; the whole script takes about 2.5 minutes on six cores (about 6 minutes without the cache).

## 12. Differences from the draft pipeline

Point estimates agree with the draft's macros to the printed precision; intervals move by one to a few units in the
third decimal (new random streams, 4000 instead of 1000 paired-bootstrap resamples). Deliberate changes: paired tests
against LoRA and Attn (classification) and LoRA (segmentation) added; BH families restricted to the main design;
spreads per label; the true seed per run; segmentation reporting modes remapped in the metrics files themselves; case
counts from the per-case files; `segrecomp_*` keeps the draft's meaning (re-inferred segmentation runs) while
`selcheck_*` covers every run.

## 13. Per-label intervals and parameter counts (`label_ci.csv`, `params.csv`; added 2026-11-01)

Added for the per-label appendix tables and the method table in the previous draft's layouts (`tables/paper_tables.py`);
no other output changes (checked: every other file in `out/` is byte-identical, `checks.txt` gains two row counts).

- `label_ci.csv`: for every seed-42 run of a main-design method (classification: all eleven methods, modes per-task,
  joint, joint_lm; segmentation: all seven methods, modes per-task, joint, joint_8k; every backbone; breast MRI left
  out) and every label scored in `metrics_cls.csv` / `metrics_seg.csv`: the held-out AUROC or Dice (`value`, equal to
  the metrics files) and a 95% percentile interval (`ci_lo`, `ci_hi`) from 1000 bootstrap resamples of the held-out
  cases, `numpy.random.default_rng([20261101, crc32(task, backbone, cohort, mode, method, label)])`.
  Classification resamples positives and negatives separately (stratified, so every resample has an AUROC; a row is a
  case, FDG-PET/CT rows are studies); segmentation resamples the valid cases of the structure (Section 2) and takes
  the mean Dice. The draft's per-label intervals (`harvest_metrics.py`) resampled all held-out cases without
  stratification (patients as clusters on FDG-PET/CT) and dropped single-class resamples; with few positives (e.g.
  RSNA 2022 C3, 5 positives in 202) its intervals are wider.
- `params.csv`: trainable / total parameters and encoder trainable / total parameters of every classification method
  and backbone, read from the line written by the model wrappers' `_log_param_summary` at model build
  (`<Model>ClsModel [mode=...]: <trainable> / <total> trainable (...); backbone[+combiner] <b> / <B>`) in
  `logs/log_*.txt` of the first seed-42 per-task run (path order) of the pair. Counts are those of a single-label
  model (head included); `log_file` names the source.
