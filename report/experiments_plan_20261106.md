# Cheap experiments after the second review (planned 2026-11-06)

Costs come from Table 4 (training minutes per 1000 steps, validation excluded) and the wall times of the paper's
Snellius runs (H100: probes 25–60 min, LoRA about 45 min, Pillar-0 on ICH about 70 min per 8000-step run).

## E1. Learning-rate check of the frozen probes (RUNNING on Snellius)

- Question (reviewer 1, M4): do the default learning rates understate the probes, and does LoRA still beat a tuned probe?
- Runs: joint models; Lin at 1e-2 and 1e-4 (default 1e-3), Attn at 3e-5, 3e-4 and 1e-3 (default 1e-4); 10
  cohort–backbone pairs (TAP-CT x5, SPECTRE x4, Pillar-0 x1) = 50 runs. Methods `lin_lr1e2 lin_lr1e4 attn_lr3e5
  attn_lr3e4 attn_lr1e3` in `REVISION_METHODS` of fm_adapt/generate.py (written by `--revision`); configs differ from
  the default joint configs only in the learning rate.
- Submitted 2026-11-06 12:47 on Snellius gpu_a100: TAP-CT jobs 28942265–69 (lrTA_, 5 runs/job, 5 h wall), SPECTRE
  and Pillar-0 jobs 28942270–75 (lrSA_, 5 runs/job, 10 h wall). Reserved about 9600 SBU; 3472 SBU left for submit.
- Analysis: per cohort–backbone pair and probe, select the learning rate (default included) by the selection-split
  mean AUROC; recompute LoRA − tuned Lin and LoRA − tuned Attn at label level with the existing bootstrap; report
  both next to the default-setting estimates (Appendix D table + one sentence in §4.1).
- Decision rule, fixed before the results: if both intervals still exclude zero, the claim stands with "also against
  probes tuned on selection data"; otherwise the abstract and §4.1 state the gain against the default probes and the
  smaller gain against tuned ones.

## E2. Learning-rate check of LoRA (Gaia, not yet submitted)

- Question: is the Attn-only FT > LoRA ranking (difference 0.011, interval includes zero) a LoRA learning-rate effect?
- Runs: joint LoRA at 3e-5 and 3e-4 (default 1e-4), same 10 pairs = 20 runs (`lora_lr3e5`, `lora_lr3e4`; configs
  already written on both sites).
- Cost: about 20–30 GPU-hours (LoRA ~45 min per run on H100; slower on A6000).
- Analysis: as E1; the selected LoRA learning rate vs Attn-only FT.

## E3. Seed repeats of per-task models (Gaia, needs a small generator change)

- Question (reviewer 1, M1): seed variation of per-task models, which the main intervals do not include.
- Runs: per-task Lin and LoRA on FDG-PET/CT (4 labels), TAP-CT and SPECTRE, seeds 43 and 44 = 32 runs.
- Change: per-task seed modes in generate.py (today only joint models have `joint_s43` / `joint_s44`).
- Cost: about 15–25 GPU-hours.
- Analysis: per-label seed SD next to the method differences; count of label–seed pairs where LoRA > Lin.

## E4. Structure-balanced crop sampling for joint segmentation (Gaia, needs a code change)

- Question (both reviewers): is the small-structure loss of the joint models caused by crop sampling rather than budget?
- Change: in `CTSegmentationDataset`, draw the structure uniformly first, then a voxel of that structure, as the crop
  centre (today: a random foreground voxel of all structures together); config flag, default off.
- Runs: AMOS22 joint Dec and LoRA, TAP-CT and SPECTRE, 24 000 steps = 4 runs.
- Cost: a few GPU-hours on TAP-CT, more on SPECTRE; about 10–15 GPU-hours in total with validation.
- Analysis: Dice of the adrenal glands and esophagus and the mean joint − per-structure change vs the current runs.

## Analyses on the stored predictions (no GPU)

- A1. Equivalence check of joint − per-task classification with a stated margin (e.g. ±0.02 AUROC) on the existing
  bootstrap (reviewer 1, M2).
- A2. Joint − per-task for LoRA alone (the recommended configuration).
- A3. LoRA − Lin on the cohorts outside SPECTRE's pretraining data (FDG-PET/CT, RSNA22, RSNA23).
- A4. Number of positives per label in Table 22.
- A5. Median seed SD of the joint models (Table 13) quoted in §4.1.

## Not cheap: SPECTRE on RSNA19 ICH

88 runs (77 main: 66 per-task + 11 joint; 11 seed repeats and controls). ICH evaluates 6533 selection and held-out
volumes at each of the 8 validations; from the SPECTRE RSNA22 runs (about 0.5 s per volume at validation) this is
about 8 h per run on H100, about 700–750 GPU-hours in total. No record says why it was not run.

## After the runs

- Add the new methods to the public repo generator (github.com/georgeyiasemis/beyond_frozen_encoders) if they enter
  the paper.
