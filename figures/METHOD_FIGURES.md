# Method figures: facts drawn and where each comes from

Script: `figures/method_figures.py` (style from `figures/style.py`); outputs in `figures/out/` as PDF + PNG.
Sources: `report/code_map_classification.md` (CLS §x.y), `report/code_map_segmentation.md` (SEG §x.y), and the
`aifofomo` code where the maps were not explicit (file:line). Nothing was taken from existing figures or paper text.
Items marked **derived** are arithmetic on sourced numbers; items marked **?** could not be sourced.

Colour code on every figure: grey = frozen pretrained weights, sky blue = low-rank adapter (trained), orange = trained
in full (`style.FAMILY_COL`). Text is 5.5 pt (body) / 6.5 pt (headers); every figure is 6.0 in wide (`TEXTWIDTH_IN`).

## fig_pipeline_cls (6.00 x 6.13 in)

Row = backbone; columns input volume -> coverage -> encoder -> feature read by the head -> head.

| fact | source |
|---|---|
| TAP-CT: LPS, native depth, in-plane resize 224x224, `keep_native_depth` | CLS §1.5 table |
| TAP-CT: HU clamp [-1008, 822], z-score mean -86.8086 / std 322.6347 (in model) | CLS §1.5 table, `tapct_cls.py` l.33-34 |
| TAP-CT: depth padded with air to >= 12 and multiple of 4 | CLS §2.1 (`_ensure_min_roi` l.355-377) |
| TAP-CT: 12x224x224 windows, sw_overlap 0.75 -> stride 3, final window at last slice, <= 32 train windows, 16 per forward | CLS §1.5 summary, §2.1 (`_depth_window_starts`, `sw_batch_size 16`, `max_train_windows 32`) |
| TAP-CT: ViT-B 12 blocks, 768-d, 12 heads, patch [4,8,8], 3x28x28 = 2352 tokens + CLS + 4 registers, fused qkv + proj, fc1/fc2, SDPA | CLS §2.1 |
| TAP-CT: CLS of last block after final norm per window; Lin mean over K -> Dropout 0.2 -> Linear(768,T); Attn probe over K CLS tokens | CLS §2.1 (`_multi_crop_cls_stack`, `_probe_from_cls_bag`) |
| SPECTRE: RAS, 0.5x0.5x1.0 mm, centred 512x512x256 ROI (D,H,W 256x512x512), pad with scan min, HU [-1000,1000] -> [0,1] | CLS §1.5 table |
| SPECTRE: 128x128x64 crops, 4x4x4 = 64 at eval, 2x2x2 sub-grid (max_train_crops 8) in training, 16 crops per forward | CLS §2.2 (`_tile`, `_train_subgrid`, `max_crops_per_forward 16`) |
| SPECTRE: ViT-L 24 blocks, 1080-d, 12 heads, patch (16,16,8) -> 512 tokens + CLS, separate q/kv + proj, RoPE; combiner ViT 4 blocks 1080-d 12 heads; per-crop [CLS ‖ mean patch] 2160-d | CLS §2.2 |
| SPECTRE: Lin reads combiner CLS (1080); Attn reads N+1 combiner tokens; N = 8 train / 64 eval | CLS §2.2 (`_probe`), §6 item 5 |
| Pillar-0: LPI, 0.5x0.5x1.25 mm, centred 128x256x256, 11 channels (10 named HU windows + min-max) | CLS §1.5 table, §2.3 (`_ct_window_all`); window list `pillar0_cls.py` l.92-103 |
| Pillar-0: whole volume in one forward | CLS §2.3 |
| Pillar-0: Conv3d+BatchNorm stem, patch [8,8,4] (H,W,D), stages [2,2,8], dim 384, 6 heads, 3 scales, merge_ratio [4,4,4], no class token, separate q/kv + proj | CLS §2.3; `Pillar0-HeadCT/config.json` (`stages [2,2,8]`, `patch_size [8,8,4]`, `merge_ratio`, `atlas_args`) |
| Pillar-0: 32^3 stem grid (256/8, 256/8, 128/4); scale grids 32^3, 8^3, 2^3 | **derived** from patch/merge ratio; 32^3 and 2^3 confirmed by CLS §2.3 ("stem grid 32³ → 16³") and `pillar0_cls.py` comment ("8 tokens of a zero-padded 2³ grid") |
| Pillar-0: Lin reads pooled 1152-d L2-normalised embedding; Attn reads per-scale token sets, scale > 4096 tokens avg-pooled (32^3 -> 16^3), each scale in its own channel block of 1152 | CLS §2.3 (`_extract_pooled`, `_extract_spatial`, `max_spatial_tokens 4096`); `pillar0_cls.py` l.596-640 |
| Pillar-0: 16^3 + 8^3 + 2^3 tokens for the attentive head | **derived** (pool factor ceil((32768/4096)^(1/3)) = 2; `pillar0_cls.py` l.628) |
| VoCo: RAS, 1.5 mm, target 96^3, crop_mode pad (never crops), HU [-175,250] -> [0,1] | CLS §1.5 table |
| VoCo: 96^3 windows, sw_overlap 0, last window shifted to border, <= 16 train windows, 4 per forward | CLS §2.4 (`_windows`, `windows_per_forward 4`, `max_train_windows 16`) |
| VoCo: SwinUNETR v2 encoder, patch 2, depths (2,2,2,2), heads (3,6,12,24), window 7, feature_size 96 -> widths 96/192/384/768, outputs 192/384/768/1536 at strides 4/8/16/32, fused qkv + proj, linear1/linear2 | CLS §2.4 |
| VoCo: token = concat of spatial means of stages 2,3,4 = 2688-d, `_RunningStandardize` EMA; Lin mean over windows -> Linear(2688,T); Attn probe with input projection 2688 -> 768 | CLS §2.4, §6 item 6; `voco_cls.py` l.125 (`probe_hidden_dim 768`), l.339-345 |
| Attentive probe: 2 pre-norm self-attention blocks, 8 heads, [CLS]+tokens, LayerNorm, Dropout 0.2, Linear; hidden = embed dim for TAP-CT/SPECTRE/Pillar-0 | CLS §2.5; `attentive_probe.py` l.115-150; `tapct_cls.py` l.167, `spectre_cls.py` l.137, `pillar0_cls.py` l.283 |
| Linear head = Dropout(0.2) + Linear(embed, num_tasks) | CLS §2.5 |
| LoRA+ABMIL: gated-attention MIL pooling (hidden 128) over the tokens, then the probe | CLS §2.5, §3.1 (`babmil`) |
| LoRA targets: fused qkv (TAP-CT, VoCo); q, kv (SPECTRE incl. combiner; Pillar-0) | CLS §2.1-2.4, §3.4 |

## fig_pipeline_seg (6.00 x 4.70 in)

| fact | source |
|---|---|
| TAP-CT: LPS, in-plane 224, native depth; patch 12x224x224; HU clamp/z-score in model | SEG §1.4 table |
| SPECTRE: RAS, 1.0 mm (image linear, label nearest), HU [-1000,1000] -> [0,1], patch 64x128x128 | SEG §1.4 table |
| VoCo: RAS, 1.5 mm, HU [-175,250] -> [0,1], patch 96^3 | SEG §1.4 table |
| fg_prob 0.67 foreground-centred crops, else uniform | SEG §1.4 (`_random_crop`) |
| micro-batch 2 x accumulation 2 = 4 crops per step | SEG §4 (`MICRO_BATCH`, `ACCUM`); `fm_seg/generate.py` l.207-208 |
| Taps: TAP-CT blocks 3,6,9,12; SPECTRE 6,12,18,24; VoCo stage outputs 1-4 | SEG §2 table (`*_DEFAULT_TAPS`) |
| Final LayerNorm on each TAP-CT tap, CLS + registers dropped; SPECTRE final norm, prefix token dropped; VoCo parameter-free channel LayerNorm | SEG §2 table ("token → grid"), §7 item 6 |
| Strides (4,8,8) / (8,16,16) / taps at 4,8,16,32 with fuse grid stride 4; channels 768 x4 / 1080 x4 / 192,384,768,1536 | SEG §2 table |
| Tokens per crop 3x28x28 / 8x8x8 / stage-4 grid 3^3; VoCo grids 24^3, 12^3, 6^3 | SEG §2 table; VoCo 24/12/6 **derived** (96 / 4, 8, 16) |
| SPECTRE feature combiner not used for features | SEG §4 ("those combiner adapters never enter `features()`") |
| Lin head: Conv3d k=1 (embed -> C) on `feats[-1]`, trilinear to crop | SEG §3 (`LinearSegHead`); `heads.py` l.33-39 |
| Dec head: 4x Conv1x1x1 -> 128, resize to fuse grid, concat 512, Conv3+IN(affine)+LeakyReLU(0.01) -> 128, N stages of x2 trilinear + ConvBlock halving channels (min 16), Conv1x1x1 -> C, final trilinear | SEG §3 (`MultiTapSegDecoder`); `heads.py` l.51-100 |
| N and channel chains: TAP-CT 3 stages (2,2,2),(2,2,2),(1,2,2) 128->64->32->16; SPECTRE 4 stages 128->64->32->16->16; VoCo 2 stages 128->64->32 | SEG §3 item 3 |
| Dec parameter counts 2.45 M / 2.62 M / 2.42 M (C = 15) | SEG §3 (derived in the map, not logged) |
| No input skip connections | SEG §3 ("forward uses only `feats`") |
| Sigmoid per structure (BCE with logits), threshold logit(0.5) = 0 for Dice | SEG §3, §6 |
| Sliding window: MONAI, Gaussian, constant padding with `input_pad_value` (-1008 HU / 0 / 0), CPU stitching; overlap/batch 0.5/4, 0.25/8, 0.5/4 | SEG §2 (sliding window paragraph; `SW` dict); `fm_seg/generate.py` l.212 |

Note: the task brief said "overlap 0.5" for all backbones; the code uses 0.25 (sw_batch 8) for SPECTRE
(`generate.py` `SW`), which is what the figure shows.

## fig_methods_cls (6.00 x 4.96 in)

| fact | source |
|---|---|
| 11 methods, training_mode, head type, adapter scope, steps, warm start | CLS §3.1 table |
| frozen_backbone trains only the head (`probe.*`, `crop_abmil.*`) | CLS §3.2 |
| LoRA/DoRA replace Linears named in `lora_target_names` matching the include pattern; A, B (+ magnitude) train; original weights frozen | CLS §3.2, §3.3 |
| rsLoRA scaling alpha/sqrt(r) = 8 vs alpha/r = 2 | CLS §3.3 |
| DoRA formula, m initialised to row norms of W0 | CLS §3.3 |
| Attn-only FT: every parameter of the parent attention module (qkv or q/kv, proj, biases, VoCo relative_position_bias_table); MLP/norms/patch embed frozen | CLS §3.2, §6 item 4; `lora.py` l.254-293 |
| Full FT: every backbone parameter | CLS §3.2 |
| Late blocks: TAP-CT 9-12 of 12; SPECTRE 17-24 of 24 + combiner; Pillar-0 stage 3 (8 of 12 blocks, 3 scales each); VoCo stage 4 (2 of 8) | CLS §3.4 |
| LoRA target per backbone; r 16, alpha 32, dropout 0.1 | CLS §3.1 (settings line), §3.4 |
| Warm start: parent's `model_4000.pt`; `probe.*` (+ `crop_abmil.*`, VoCo `token_norm.*`) copied; LoRA fresh (B = 0); fresh job of 4000 steps, warm-up 400; head type inherited | CLS §3.5 |
| Steps 8000; two-stage 4000 + 4000 | CLS §3.8 |
| LR / wd: lin 1e-3/0; attn and adapters 1e-4/0.01; attnft/fullft backbone 2e-5/0.05, head 1e-4/0.01; head dropout 0.2; batch 1 | CLS §3.1, §3.8 |
| LoRA+ABMIL head: ABMIL pooling then the attentive probe | CLS §2.1-2.4 (`babmil` paths) |

## fig_methods_seg (6.00 x 4.17 in)

| fact | source |
|---|---|
| 7 methods, training_mode, head, LoRA scope, lr, wd, steps | SEG §4 table; `fm_seg/generate.py` l.167-184 |
| Head trainable only under lin/dec; LoRA/DoRA A, B (+ m); fullft every backbone parameter | SEG §4 table |
| Late blocks: TAP-CT 9-12, SPECTRE 17-24 (`backbone\.blocks\.` excludes combiner), VoCo layers4 | SEG §2 table, §4 |
| LoRA targets fused qkv / q,kv; r 16, alpha 32, dropout 0.1, scale 2 | SEG §2 table, §4 |
| SPECTRE all-block LoRA / DoRA / Full FT also wrap the combiner q/kv, no gradient | SEG §4, §7 item 4 |
| Warm start Dec -> LoRA: `head.*` from the parent's `model_<stage1*2>.pt` (= step 4000 / 12000), all-block LoRA fresh, new optimizer, stage 2 = half the steps | SEG §4 (warm start paragraph) |
| Steps 8000; joint_lm 24000 (amos22, totalseg, 15 structures); pertask_cm 1600 (dec, lora, fullft; also lin per the mode table) | SEG §4 (`_steps_for`), §5 mode table; `fm_seg/generate.py` l.160-162, 193 |
| Compute-matched and lorag/dora/dec_then_lora availability (lorag, dora, dec_then_lora have no pertask_cm run) | SEG §5 (`CM_METHODS = [dec, lora, fullft]`; totalseg pertask methods lin, dec, lora, fullft) |
| LR / wd per method; fullft backbone 2e-5, head 1e-3, wd 1e-4 | SEG §4 table |
| 2 crops x 2 accumulation per step | SEG §4 |

Note: the cell "1600" in the steps column of Lin follows SEG §5 (totalseg per-task methods lin, dec, lora, fullft);
the compute-matched `pertask_cm` revision mode itself lists `CM_METHODS = [dec, lora, fullft]` (SEG §4, `generate.py` l.193).
The note panel states the `CM_METHODS` set and the figure keeps "1600" for Lin from the per-task table; if only the
`pertask_cm` revision counts as "compute-matched", the Lin cell should read "–".

## fig_modes (6.00 x 2.62 in)

| fact | source |
|---|---|
| Per-task: num_tasks 1, one config per label; joint: one encoder, one adapter set, one head with one logit per label, same tokens feed every output | CLS §4 |
| Classification loss: `MultiLabelBCELoss`, `sum_tasks_mean_batch`, pos_weight = clip(neg/pos, 0.05, 20), masked over known labels, mean over batch | CLS §4 (formula), §1.4 |
| Batch 1 (classification) | CLS §1.7, §3.8 |
| Segmentation loss: Dice (smooth 1) + BCE per (case, structure), mean over all (b, c); separate sigmoid channels | SEG §5 (loss averaging), §6 |
| Budgets: per-task K x 8000; joint 8000 (1/K per label); 15-structure segmentation joint 24000 | CLS §3.8, §4; SEG §4 `_steps_for`, §5 |
| Selection: validation every 1000 steps (8 per run), best val macro AUROC / mean Dice with strict `>`, held-out metric at that checkpoint | CLS §5; SEG §6 (validation schedule, metric, selection) |

## Unsourced items ("?")

None of the drawn facts is marked "?". Two items are derived rather than read directly: the Pillar-0 attentive-head
token count 16^3 + 8^3 + 2^3 (from patch size, merge ratio and the 4096-token pooling rule) and the VoCo segmentation
tap grids 24^3 / 12^3 / 6^3 (96 divided by the tap strides).
