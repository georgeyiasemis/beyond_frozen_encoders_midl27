# Output schema of metrics/recompute.py (all files in metrics/out/; every downstream script reads only these)

Common columns: task (cls|seg), backbone (tapct|spectre|pillar0|voco), cohort (classification cohort key or segmentation dataset key),
mode, method, seed, label (classification label or segmentation structure; "joint" runs expand to one row per label).
Reporting modes: "pertask", "joint" (8000 steps; for seg amos22/totalseg "joint" = the 24000-step run and "joint_8k" the 8000-step run),
"joint_lm" (classification CVPR26 26000-step run), "pertask_cm" (seg compute-matched 1600-step per-structure), seeds 42/43/44 in the seed column.

- runs.csv: one row per finished run used: task, site, cohort, backbone, mode, method, seed, label, best_iter, n_validations, final_iter, path, percase_file
- metrics_cls.csv: task, backbone, cohort, mode, method, seed, label, split (val|test), auroc, auprc, n, n_pos, pred_sd
- metrics_seg.csv: task, backbone, cohort, mode, method, seed, label, split, dice, n_valid
- cohort_means.csv: task, backbone, cohort, mode, method, seed, metric (auroc|dice), value, n_labels   (mean over labels/structures; only complete cells)
- paired_cls.csv: backbone, cohort, mode, label, method, baseline (lin|lora|attn|fullft), value_method, value_baseline, diff, se, p, p_bh, n, n_pos
- paired_seg.csv: backbone, cohort, mode, label, method, baseline (lin|dec|lora), value_method, value_baseline, diff, ci_lo, ci_hi, p, p_bh, n
- label_level_values.csv: key, task, backbone, cohort, label, value   (one label-level difference per label; modes averaged)
- label_level.csv: key, task, backbone (tapct|spectre|pillar0|voco|all), n_labels, mean, lo2, hi2, lo3, hi3, clusters_pos, clusters_n
    keys: cls_lora_lin, cls_attnft_lora, cls_fullft_lin, cls_attn_lin, cls_<method>_lin for every method, cls_joint_pertask,
          seg_dec_lin, seg_lora_dec, seg_lora_dec_pertask, seg_lora_dec_joint, seg_fullft_dec, seg_<method>_dec for every method, seg_joint_pertask
- ranks.csv: task, backbone (per backbone and "all"), method, mean_rank, n_slots, within_001_pct, regret
- spread.csv: backbone, method, median_sd, did_not_train (bool)
- convergence.csv: task, backbone, method, mode, n_runs, frac_last, frac_last_two
- seeds.csv: task, backbone, cohort, method, mode, seed, value (cohort mean) ; plus seeds_summary.csv: task, backbone, method, sd_median, pairs_pos, pairs_n per comparison
- revision.csv: task, backbone, cohort, mode, method (fullft_lowlr|fullft_llrd|fullft_accum|mlpft|rslora_lowlr|scratch|pertask_cm...), label, split, value
- label_ci.csv: task, backbone, cohort, mode, method, seed (42), label, split (test), value, ci_lo, ci_hi, n_boot   (per-label held-out value with a 95% case-bootstrap interval; PROCEDURE.md section 13)
- params.csv: backbone, method, trainable, total, backbone_trainable, backbone_total, log_file   (classification, single-label model, from the training log)
- numbers.tex: \newcommand macros for every number the paper quotes (naming documented in NAMING.md written by the script)
- changes_vs_draft.csv: macro, old_value (from midl_paper/paper/tables/numbers.tex), new_value, abs_change (only macros whose names match)
