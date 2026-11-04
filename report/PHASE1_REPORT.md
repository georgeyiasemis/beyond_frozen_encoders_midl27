# Phase 1 audit: "Beyond Frozen Encoders" rewrite

Date: 2026-10-31. Source of truth: the code on disk in `aifofomo/` (working tree, 902 uncommitted changes included),
the run outputs mirrored under `midl_paper/results/raw/`, and the per-case prediction files. The draft
(`midl_paper/paper/main.tex`) was used only to know what the study covers. Detailed documents in this folder:

| Document | Content |
|---|---|
| `code_map_classification.md` (705 lines) | fm_adapt pipeline: data, splits, backbones, 11 methods + revision methods, optimizer, schedule, selection, outputs; 14 discrepancies/omissions |
| `code_map_segmentation.md` (426 lines) | fm_seg pipeline: manifests, labels, TotalSegmentator merge, dense backbones, Lin/Dec heads, 7 methods, modes, loss, Dice, selection; 17 discrepancies/omissions |
| `results_pipeline.md` | how the existing harvest/statistics scripts compute AUROC, Dice, DeLong, bootstrap intervals, ranks, did-not-train; plan for the Phase 2 recompute script |
| `run_inventory.md` + `run_inventory.csv` + `run_inventory_design.csv` + `scripts/build_run_inventory.py` | every run directory on disk vs the YAML design |
| `citation_audit.md` + `refs_verified.bib` | per-key verification and per-sentence fit (see section 4) |

## 1. Code map (summary; details with file:line in the two code-map documents)

### 1.1 Classification (`aifofomo/examples/fm_probes/fm_adapt`, `aifofomo/models/networks/{tapct,spectre,pillar0,voco}`)
- **Design generator**: `fm_adapt/generate.py` reproduces all 1354 YAMLs byte-for-byte (`--revision` for seed repeats and the full-FT/MLP/rsLoRA variants). VoCo seed-repeat/revision YAMLs were never written, so no such VoCo runs exist.
- **Data/splits**: AMOS/COVID/LUNA (CVPR26): official validation set split into selection/held-out halves by even/odd case rank. FDG, RSNA 2022, RSNA 2023: seeded 80/10/10 stratified splits of the released training data. RSNA 2019 ICH: external patient table with a 5k stratified training subset. Patient-level disjointness enforced in code only for FDG; verified from manifests for FDG, RSNA 2023, RSNA 2019 (0 spanning patients); RSNA 2022 split per study; CVPR26 per file. breastdivider (breast MRI) exists in the design but is excluded from the paper.
- **Backbones (as instantiated)**: TAP-CT-B-3D = plain ViT-B, 12 blocks, 768-d, 4x8x8 patches, CLS + 4 registers, fused qkv. SPECTRE-Large = plain ViT-L, 24 blocks, 1080-d, 16x16x8 patches, q and kv projections, plus a 4-block feature-combiner ViT over per-crop [CLS || mean patch] tokens. **Pillar-0 = convolutional stem + 3-stage hierarchical window transformer** (stages [2,2,8], dim 384, 11 input channels = CT windows, 1152-d pooled embedding = 3 scales concatenated and L2-normalised). **VoCo-L = hierarchical SwinUNETR-v2 encoder** (depths 2,2,2,2, window 7, fused qkv); one 2688-d token per 96^3 window = GAP of stages 2-4 with running standardisation. So two of the four backbones are hierarchical (Pillar-0 and VoCo); the draft's "three vision transformers" and "encoder family" argument are not supported by the code.
- **Methods** (11 + revision): Lin (dropout 0.2 + linear on pooled feature), Attn (two self-attention blocks over [CLS]+tokens, CLS read-out; VoCo projects 2688->768 first), LoRA r16/alpha32/dropout 0.1 on qkv (TAP-CT, VoCo) or q,kv (SPECTRE incl. combiner, Pillar-0), LoRA_late (TAP-CT blocks 8-11 0-idx, SPECTRE blocks 16-23 + combiner, Pillar-0 last stage = 8 of 12 blocks, VoCo layers4 = 2 of 8), LoRA+ABMIL (gated attention over window/crop tokens then attentive head), DoRA (magnitude (out,1), **weight-decayed**), rsLoRA (alpha/sqrt(r)), Lin->LoRA and Attn->LoRA (4000 + 4000; child requires the parent's exact `model_4000.pt`, copies probe.* [+crop_abmil.*, +VoCo token_norm buffers], fresh LoRA with B=0, fresh optimizer; head type inherited), Attn-only FT (whole attention modules incl. VoCo relative-position bias), Full FT. Revision: fullft_lowlr, fullft_llrd (0.75 per block; SPECTRE depth counted over backbone + combiner), fullft_accum (accum 8, clip 1.0, 1000 steps), mlpft, rslora_lowlr.
- **Optimisation**: AdamW (0.9, 0.999), eps 1e-8; head lr per method table, backbone lr 2e-5 for attnft/fullft; wd 0.01 except ndim<=1/bias/norm; 8000 steps, batch 1, no accumulation, no clipping, 10% linear warm-up from 1% of peak, cosine to 1e-6, bf16-mixed, seed 42 (43/44 for repeats), no EMA, gradient checkpointing only for VoCo. CVPR26 joint runs (both 8000-step `joint` and 26000-step `joint_lm`) sample cohorts 10:8:8.
- **Loss**: masked positive-weighted BCE, `(per_entry * mask).sum() / B`, `pos_weight = clip(neg/pos, 0.05, 20)` from the training manifest.
- **Selection/eval**: every 1000 steps, `val/metric/val_auroc` = macro AUROC over labels with both classes on the selection split, strict improvement; periodic trainable-only checkpoints every 500 steps; per-case `val_case_metrics/iter_<it>/logits.csv` (case_id, split, task, task_index, label, logit, probability) for val and test. Held-out numbers are recomputed from these files.

### 1.2 Segmentation (`aifofomo/examples/fm_probes/fm_seg`, `aifofomo/models/networks/seg_probe`)
- **Splits**: AMOS22 official validation set split even/odd into 50/50 (train 200). KiTS23 343/73/73 and MSD Pancreas 197/42/42 are one seed-0 70/15/15 shuffle of all labelled cases; LiTS 91/20/20 of the 131 training volumes; TotalSegmentator: official test (109) held out, 109 validation cases carved from the 1830 official training cases (1721 train). No patient ids anywhere.
- **Labels**: multi-channel sigmoid targets; overlapping parents (KiTS kidney = [1,2,3], LiTS liver = [1,2], MSD pancreas = [1,2]). TotalSegmentator MERGE_15 matches the draft's table; overlapping source masks resolved by painting larger structures first (smaller wins).
- **Dense backbones**: TAP-CT taps blocks 3/6/9/12 with the final LayerNorm, stride (4,8,8); SPECTRE blocks 6/12/18/24, stride (8,16,16), crop (64,128,128); VoCo Swin stage outputs 1-4 at strides 4-32 with **192/384/768/1536** channels, fused on the stride-4 grid; Lin on VoCo reads the 1536-d stride-32 output.
- **Heads**: Lin = 1x1x1 conv on the last tap + trilinear upsampling. Dec = 4 x Conv1x1->128, concat (512), Conv3 + InstanceNorm + LeakyReLU -> 128, then 3 (TAP-CT) / 4 (SPECTRE) / 2 (VoCo) x2 trilinear stages halving channels (min 16), Conv1x1 -> C; no input skips; derived 2.45 M / 2.62 M / 2.42 M parameters.
- **Methods**: lin (1e-2, wd 0), dec (1e-3, wd 1e-4), lora/lorag/dora (1e-4, wd 1e-4; r16 alpha32 drop 0.1; same targets/late ranges as classification), fullft (backbone 2e-5, head 1e-3), dec_then_lora (head from the parent's periodic checkpoint at the half-way step, fresh all-block LoRA, stage 2 at 1e-4). In SPECTRE, adapters/unfreezing also cover the feature combiner (no gradient, counts affected).
- **Schedule**: 8000 optimizer steps (24000 for AMOS22/TotalSegmentator `joint_lm`; 1600 for compute-matched `pertask_cm`), micro-batch 2 x accumulation 2 (= 4 crops/step), warm-up 10% from 1%, cosine to 1e-6, no clipping, bf16-mixed, seed 42 (43/44 repeats). Flip p=0.5, zoom p=0.5 in [0.85,1.15]; fg_prob 0.67.
- **Loss**: mean over (case, channel) of [1 - (2*sum(py)+1)/(sum p + sum y + 1)] + mean-voxel BCE.
- **Dice/selection**: logits > 0; valid = |p|+|t| > 0; invalid pairs stored as dice 0 / valid 0 (must filter valid==1); `val_dice` = mean over classes of mean over valid selection cases, every 1000 steps (200 for pertask_cm); held-out evaluated at the same time except TotalSegmentator (held-out via `reinfer_dice.py` after training, tol 5e-3). 109 older runs have per-case files only for the selected iteration (re-inference).
- **Modes on disk**: pertask, joint, joint_lm (AMOS22, TotalSegmentator; TAP-CT/SPECTRE/VoCo), dec_then_lora on joint_lm for TAP-CT/SPECTRE/VoCo (VoCo without YAML), pertask_cm and seed repeats for TAP-CT/SPECTRE only, scratch for TAP-CT joint only. TotalSegmentator per-task: lin/dec/lora/fullft only.

### 1.3 Results pipeline (`midl_paper/results/scripts`) — see `results_pipeline.md`
Finished run = best-checkpoint record + per-case file at that iteration + last checkpoint at the YAML's final iteration. AUROC = midrank Mann-Whitney. Paired tests: DeLong vs Lin and Full FT (classification), paired case bootstrap vs Lin and Dec (segmentation), BH within (test, metric, baseline, is_voco). Label-level estimates: a label's difference averaged over per-task and joint; 4000-resample percentile bootstrap over cohort-backbone clusters -> labels -> 1000 case-level resamples per label; pooled estimates over TAP-CT, SPECTRE, Pillar-0 only. Ranks over slots with every expected method present. Did-not-train rule: median held-out probability SD < 0.01 per (backbone, method).

## 2. Run inventory (details: `run_inventory.md`)
- 2394 run directories (cls 1356, seg 1038) + 24 auxiliary + 130 quarantined. Finished: cls 1354/1356, seg 1034/1038 (the 4 extra seg dirs are obsolete variants not in the design).
- Design (YAML trees + 2 generator-enumerated runs without YAML) = 2388 runs: cls 1354 (main 1232, revision 50, seeds 72), seg 1034 (main 789, pertask_cm 180, seeds 60, scratch 5). **Missing: 0. Unfinished: 0.**
- Finished per backbone (deduplicated): cls TAP-CT 495 + 25 revision + 40 seed; SPECTRE 418 + 20 + 24; Pillar-0 110 + 5 + 8; VoCo 209. Seg TAP-CT 263 + 90 cm + 30 seed + 5 scratch; SPECTRE 263 + 90 + 30; VoCo 263.
- Draft macros match disk: numRunsCls 990 (= 1232 main - 209 VoCo - 33 breastdivider), numSegRunsN 526 (= 789 - 263 VoCo), seeds 72 / 60, VoCo 209 / 263.
- Duplicates across sites: 2 (Gaia stubs failed on 2026-10-16; Snellius copies finished; no conflict). Failed/empty: those same 2 stubs.
- Per-case files exist for every run used by master_metrics(.csv/_seg.csv) and revision_metrics.csv; held-out case counts identical within every (cohort, mode family, label): 0 evaluation mismatches.
- Anomalies: (a) 3 Gaia revision runs (rsna2022/tapct/joint/{fullft_accum, mlpft, rslora_lowlr}) are finished on disk but absent from master_metrics.csv (present in revision_metrics.csv; tables unaffected); (b) fm_seg amos22 joint_lm/dec_then_lora YAMLs are stale (num_iterations 8000 vs the 24000 micro-iteration two-stage horizon the runs used); amos22/totalseg voco joint_lm/dec_then_lora have no YAML although the runs are finished and used; (c) fm_seg generator enumerates 120 VoCo revision runs that were never written or run; (d) no VoCo file predates 2026-10-26 (retired design fully quarantined).

## 3. Discrepancy log (draft vs code/outputs)
Severity: **H** = changes a statement the paper makes; M = detail the paper states wrongly; L = omission to add.

| # | Sev | Draft says (main.tex line) | Code / disk does | Where |
|---|---|---|---|---|
| D1 | H | "three self-supervised 3-D CT vision transformers" (l.55, 58, 170); VoCo result attributed to the "hierarchical" "encoder family" (l.164, 181) | Pillar-0 is a conv stem + 3-stage hierarchical window transformer (own description l.72 agrees); LoRA helped on it. Two of four backbones are hierarchical. | `pillar0_cls.py`, `code_map_classification.md` §2.3 |
| D2 | H | "Official validation sets are split into a selection and a held-out half ... patients never span splits" (l.83, 85) | True only for CVPR26 (cls) and AMOS22 (seg). FDG/RSNA22/RSNA23: seeded 80/10/10 of released training data; RSNA19: external table; KiTS23/MSD: 70/15/15 shuffle; LiTS 91/20/20; TotalSeg: official test held out. Patient-level disjointness verified only for FDG, RSNA23, RSNA19; unverifiable for seg datasets and CVPR26, RSNA22 split per study. | both code maps §1 |
| D3 | H | "VoCo is trained ... with every method and mode of the other backbones" (l.83) | VoCo has no joint_lm two-stage YAML (runs exist), no seed repeats, no compute-matched, no scratch, no full-FT/MLP/rsLoRA revision runs. | seg map D14, cls map D13, inventory |
| D4 | M | VoCo "96 to 768 channels"; "outputs of the four stages (strides 4 to 32) enter the decoder" (l.72) | Tapped stage outputs are 192/384/768/1536 channels; VoCo Lin reads the 1536-d stride-32 output. | seg map D5 |
| D5 | M | DoRA magnitude "without weight decay like every one-dimensional parameter" (l.237/249) | Magnitude has shape (out,1) -> decayed (wd 0.01 cls, 1e-4 seg). | both maps |
| D6 | M | Fig. armsseg caption: segmentation "counts exclude the feature combiner" (l.202) | SPECTRE lora/dora/fullft wrap/unfreeze the whole extractor incl. combiner; no gradient reaches it, results unaffected, logged counts include it. | seg map D4 |
| D7 | M | 10:8:8 cohort sampling tied to the 26k CVPR26 schedule (l.213) | Applies to every CVPR26 joint run, 8000-step included. | cls map D3 |
| D8 | M | Attn-only FT "projections and biases" (l.250) | Whole attention module incl. VoCo relative_position_bias_table. | cls map D4 |
| D9 | M | Dec "2.4-2.6 M parameters" (l.76) | 2.42 / 2.45 / 2.62 M derived; [TODO: confirm from run logs]. | seg map D12 |
| D10 | M | "RSNA 2023 abdominal trauma (4 organ injuries)" (l.83) | any_injury + 3 organ injuries. | cls map D12 |
| D11 | M | "a learned query token attends over the tokens through two transformer blocks" (l.74) | Two full self-attention blocks over [CLS]+tokens with CLS read-out (appendix wording l.237 is right). | cls map D12 |
| D12 | M | Two-stage "4000 + 4000 steps" (l.85) stated generally | 12000 + 12000 for the 24k joint segmentation models (appendix says so); 800 for pertask_cm. | pipeline P2 |
| D13 | L | Taps "after the final normalization" said for SPECTRE only (l.247) | Final LayerNorm on every TAP-CT tap; channel LayerNorm on every VoCo stage output. | seg map D6 |
| D14 | L | TotalSegmentator merge table (l.423-449) | Overlap resolution (smaller structure wins) not stated. | seg map D8 |
| D15 | L | Dice "enters when reference or prediction contains it" (l.248) | Stored as dice 0 / valid 0; must filter valid==1. | seg map D9 |
| D16 | L | VoCo attentive probe, warm start, Pillar-0 LPI reorientation, L2-normalised embedding, frozen BatchNorm under adapters (train-mode under full FT at batch 1), TAP-CT depth padding, zoom rescales depth, single-class labels skipped in harvest | Not in draft. | cls map D6-D11 |
| D17 | L | Seg augmentation probabilities, compute-matched validation every 200 steps, protected checkpoints, padded-region BCE, TotalSeg version naming (v2 structures / v3 folder / "3.0.0"), re-inference tolerance for old runs | Not or inconsistently stated. | seg map D7, D10, D13, D15, D16 |
| D18 | L | Paired tests "per label and mode" (l.87) | Only vs Lin and Full FT (cls) and Lin and Dec (seg); Attn-only FT vs LoRA and the LoRA variants have no paired test. | pipeline P3 |
| D19 | L | master_metrics.csv | 3 finished Gaia revision runs absent (in revision_metrics.csv). | inventory |

Checked and consistent with the draft: budgets, warm-up/cosine, AdamW settings, loss formulas, selection metric and interval, checkpoint cadence, LoRA/rsLoRA settings and targets, late-block ranges, head dropout, ABMIL, per-backbone preprocessing geometry, TotalSegmentator merge map, run counts, 26 000-step CVPR26 joint schedule and 24 000-step segmentation joint schedule, TotalSegmentator per-task restriction, the did-not-train rule.

## 4. Citation audit (details: `citation_audit.md`, `refs_verified.bib`)
- All 54 bib entries exist and were verified against a primary source (arXiv API, Crossref, DataCite, PMLR, CVF, OpenReview, conference sites). Nothing relies on memory; Semantic Scholar and DBLP were unreachable and were not used.
- 25 entries correct as they stand; 18 need only pages/DOI/issue/URL added.
- 11 cite a preprint or wrong venue although a published version exists: SPECTRE (CVPR 2026), VoCo-large (IEEE TPAMI 2026), VISTA3D (CVPR 2025), DINOv2 (TMLR 2024), LoRA (ICLR 2022), V-JEPA (TMLR 2024), CTAN (MICCAI 2023 Workshops), AMOS (NeurIPS 2022), RATIC (Radiology: AI 2024), LiTS (MedIA vol 84), CAE (IJCV 2024). `refs_verified.bib` has the corrected fields under the same keys.
- REMOVE: `yang2020covidct` (unused; the cohort is COVID-CTset); one of `silvarodriguez2023fseft`/`2025fseft` (same work, workshop and journal, both cited at l.53).
- MOVE/reword: l.51 `pai2025ctfm` (CT-FM fine-tunes end-to-end: belongs with "full fine-tuning", not "fine-tuned heads"); l.51 `claessens2025spectre` (SPECTRE's frozen protocol is kNN on frozen embeddings, not a linear/attentive head); l.83 `lee2024rsnacsf` is the winning-algorithms paper, the dataset paper is Lin et al. 2023 (Radiology: AI, 10.1148/ryai.230034).
- Fit CORRECT for every other cited sentence, including the Liu et al. spleen-injury statement (their Appendix C2). UNVERIFIED fit: `huang2026mosaic` (no abstract reachable). Wording to check against the sources: FM-HCT "probing approached" full fine-tuning (magnitude), HyperCT "approach single-task models", SPECTRE geometry (24 blocks, 64 crops of 512x512x256, 4-block combiner) is not in the arXiv text (cite the model card or say so), TAP-CT's TotalSegmentator protocol uses 49 classes vs 48 at l.425.
- Hygiene: 11 entries list the senior author before "and others"; 6 keys carry a year that differs from the entry; mixed title casing; one disclosed self-citation (TAP-CT).
- Verified missing citations: DeLong 1988 and Benjamini-Hochberg 1995 (statistics, l.87); Swin Transformer, Swin UNETR, SwinUNETR-V2, ViT, "Vision Transformers Need Registers" (backbone descriptions, l.72); SAM (l.53); optional Standley 2020 / Fifty 2021 (multi-task), LoRA+ (variants), ULMFiT (layer-wise LR decay, l.127/175). No KiTS23 overview paper exists; keep KiTS21 + repository footnote. 19 suggested entries are in a separate block of `refs_verified.bib`.

## 5. Open TODOs carried into the rewrite
- Co-authors and affiliations; repository URL; how the default learning rates were chosen (the generator fixes them in `BACKBONE_METHOD_OPTIM`/method tables with no recorded rationale; [TODO: ask author]).
- SPECTRE and VoCo logged parameter counts (no local log found); Dec parameter counts from logs.
- Patient-level disjointness for RSNA 2022, CVPR26 and all segmentation datasets.
- Pillar-0 drop-path activity under LoRA/full FT; InfiniteSampler seeding.
- KiTS23 label semantics (code comment only); TotalSegmentator release naming.
- Whether `make_reinfer_jobs.py` used a non-default tolerance for the pre-hook runs.

## 6. Proposed scope for Phase 2 (for approval)
1. One script `midl_paper_v2/metrics/recompute.py` reading the per-case files of all 2388 finished runs; outputs per-run metrics, paired tests (adding LoRA as a baseline so Attn-only FT and the LoRA variants get paired tests), label-level estimates (2- and 3-stage intervals), ranks, spreads, convergence, seeds, compute-matched; `changes_vs_draft.csv` diffing every macro in `numbers.tex`.
2. Decide VoCo placement from the recomputed numbers: with D1 the "hierarchical" explanation is gone; the honest reasons for an appendix placement are partial design (3/5 cohorts, no repeats/controls), a probe interface built for it, untuned settings, and a linear probe that collapses in segmentation (Dice ~0.04).
3. Report carries the "reinfer" and "stale YAML" caveats into the appendix.
