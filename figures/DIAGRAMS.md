# Diagrams in the previous draft's design: facts drawn and their sources

Script: `figures/diagrams.py` (`~/miniconda3/envs/aifofomo/bin/python diagrams.py [--refresh-data]`). Outputs
`figures/out/{fig_pipeline,fig_arms,fig_arms_seg}.{pdf,png}`; the PDFs are copied to `paper/figures/`; it also writes
`paper/tables/param_numbers.tex`. The layout, panels and colour code are those of the previous draft's
`method_figures.py`: grey = frozen, light blue = trained in full, light orange = low-rank adapter.

Figures are drawn 7.0 in wide and placed at the 6.0 in text width. Every text object is at least 7.0 pt drawn, i.e.
6.0 pt printed; `save()` asserts this. Exception: mathtext subscripts ($W_{qkv}$, $W_{proj}$, LoRA$_\mathrm{late}$)
are about 70 % of their label, so roughly 4.2 pt printed.

Sources: `METHOD_FIGURES.md` rows ("MF: ..."), `report/code_map_classification.md` (CLS §x),
`report/code_map_segmentation.md` (SEG §x), and the code. Numbers come from data files that `--refresh-data` builds:
- `data/trainable_params_cls.csv`: copy of `midl_paper/results/trainable_params.csv`. These are the logged
  `... [mode=...]: trainable / total; backbone b / B` lines of one per-task run per backbone x method.
- `data/trainable_params_seg.csv`: the logged `SegProbeModel [...]` line of the AMOS22 joint run (15 structures) per
  backbone x method, harvested from the run logs under `midl_paper/results/raw/`. It adds `no_grad_backbone`, the
  SPECTRE feature-combiner share (LoRA/DoRA counts from the module shapes; Full FT from the safetensors header).
- `data/no_grad_cls.csv`: Pillar-0 parameters that are wrapped or unfrozen but receive no gradient, counted from the
  safetensors header (see "Pillar-0 gradient check" below).
- `data/diagram_examples.npz`: CT slices used for the drawings (AMOS22 case amos_0284 on the TAP-CT grid, a head CT
  slice), cut from the previous draft's `results/examples/`.

## fig_pipeline (7.0 x 5.32 in drawn)

| element | drawn fact | source |
|---|---|---|
| (a) input | TAP-CT volume at native depth, in-plane resized to 224² | MF pipeline_cls row 1; CLS §1.5 |
| (a) windows | K windows of 12 slices, stride 3 (sw_overlap 0.75), final window ends at the last slice | MF row 4; CLS §2.1 `_depth_window_starts` |
| (a) patches | 4×8×8 patches, 3×28² grid | MF row 5; HF config `patch_size [4,8,8]` |
| (a) tokens | CLS + 4 registers + 2352 patch tokens = 2357 × 768 | MF row 5; CLS §2.1 |
| (a) encoder | ViT-B, 12 blocks, per window | CLS §2.1 |
| (a) bag | K CLS tokens (last block, after the final norm) | MF row 6; CLS §2.1 `_multi_crop_cls_stack` |
| (a) Lin | mean over K → linear (Dropout 0.2 + Linear) | MF row 6; CLS §2.5 |
| (a) Attn | learned CLS + K tokens, 2 pre-norm self-attention blocks (8 heads), CLS read out → linear | MF "Attentive probe" row; CLS §2.5; `attentive_probe.py` l.84-196 |
| (b) crop | 12 whole slices (in-plane 224 = crop size), centred on a foreground voxel with probability 0.67, else uniform | MF pipeline_seg rows 1, 4; SEG §1.4 `_random_crop`, SEG §7 item 17 |
| (b) taps | blocks 3, 6, 9, 12, after the final LayerNorm, as 768×3×28² grids; Lin reads block 12 | MF pipeline_seg rows 6-7; SEG §2 table |
| (b) projections | 4 × Conv1×1×1 → 128 | SEG §3 Dec item 1 |
| (b) decoder shapes | 128×3×28² → 64×6×56² → 32×12×112² → 16×12×224² (scales (2,2,2), (2,2,2), (1,2,2)) | MF pipeline_seg row 13; SEG §3 item 3 |
| (b) decoder parameters | 2.45 M for 15 structures (logged 2,454,239, AMOS22 joint Dec run) | `data/trainable_params_seg.csv` (tapct, dec); equals the derived 2,453,984 + 17·15 in SEG §3 |
| (b) output | one sigmoid map per structure | SEG §3 (output activation) |
| (c) frozen | Lin, Attn, Dec train no encoder weight | CLS §3.2, SEG §4 |
| (c) LoRA | rank-16 A, B on $W_{qkv}$; LoRA, rsLoRA (α/√r), DoRA (+m), LoRA+ABMIL, Dec→LoRA on all blocks; LoRA_late, Lin→LoRA, Attn→LoRA on blocks 9-12 | CLS §3.1, §3.4; SEG §4 |
| (c) Attn-only FT | $W_{qkv}$ and $W_{proj}$ with their biases (the whole attention module) | CLS §3.2 `unfreeze_attention_modules`; logged 28,348,416 = 12 × (qkv + proj with biases) |
| (c) Full FT | every encoder parameter incl. patch embedding | CLS §3.2 |
| (c) legend | $W_0x + \frac{\alpha}{r}BAx$, α = 32, r = 16 | CLS §3.3 |
| (d) modes | per-task: K runs, one label each; joint: one encoder, adapters and head, K outputs, loss masked to the known labels | MF fig_modes rows 1-2; CLS §4 |
| (d) note | segmentation: label = structure, cohort = dataset; 15-structure joint models train 3 × the steps (24 000) | SEG §4 `_steps_for`, §5 |

## fig_arms (7.0 x 9.30 in drawn)

| element | drawn fact | source |
|---|---|---|
| (a) SPECTRE input | 512²×256 at 0.5×0.5×1 mm; 4³ = 64 crops of 128²×64; training uses a random 2³ = 8 sub-grid | MF pipeline_cls rows 7-8; CLS §1.5, §2.2 |
| (a) per crop | 16×16×8 patches → 512 tokens + CLS; ViT-L, 24 blocks, 1080-d | MF row 9; CLS §2.2 |
| (a) pooling | [CLS ‖ mean(patch tokens)] = 2160-d | CLS §2.2 `_pool_crop_tokens` |
| (a) combiner | 4 blocks over the crop grid → scan CLS + N crop tokens (N = 64 eval, 8 train) | MF row 10; CLS §2.2 |
| (a) heads | Lin: scan CLS → linear; Attn: scan CLS + N crop tokens; LoRA+ABMIL: N crop tokens → ABMIL → attentive probe | CLS §2.2 `_probe` |
| (b) Pillar-0 input | 128×256×256 at 0.5×0.5×1.25 mm; 11 channels = 10 HU windows + min-max | MF row 11; CLS §1.5, §2.3 |
| (b) stem | conv stem, 8×8×4 patches (H×W×D) → 32³ grid | MF row 13-14; Pillar0-HeadCT `config.json` |
| (b) scales | 32³, 8³, 2³ = stem grid max-pooled by 4 per axis per scale; attention within a scale and across scales | remote `multimodal_atlas.py` `_build_multiscale_tokens` (MaxPool3d), `multimodal_msa.py` `forward_sequential` (`all2all_sattn`, `one2one_xattn`) |
| (b) encoder | 3 stages of 2, 2 and 8 blocks, 384-d, no CLS token; scales 32³, 8³, 2³ leave after stages 1, 2, 3 | `config.json` (`stages [2,2,8]`, `dim 384`, `class_token false`); `multimodal_atlas.py` l.243-257 (`multiscale_feats.append(x[0]); x = x[1:]`) |
| (b) Lin | max-pool over the tokens of each scale, concatenated 3 × 384 = 1152-d, L2-normalised → linear | `multimodal_atlas.py` l.295-300 (`AdaptiveMaxPool1d`, `cat`); `modeling_clip.py` l.239 (`F.normalize`); `pillar0_cls.py` `_extract_pooled` |
| (b) Attn | tokens of all scales, 32³ average-pooled to 16³ (4096-token budget) | MF rows 15-16; `pillar0_cls.py` `_extract_spatial` |
| (b) LoRA+ABMIL | ABMIL over the same spatial tokens, then the attentive probe | `pillar0_cls.py` `_full_volume_token_bag`; logged Pillar-0 `babmil` run in `trainable_params_cls.csv` |
| (c) VoCo input | RAS, 1.5 mm, HU −175…250, 96³ windows without overlap, at most 16 in training | MF rows 17-18; CLS §1.5, §2.4 |
| (c) patch embed | 2³ patches → 96 × 48³ per window | CLS §2.4 (patch 2, feature_size 96) |
| (c) stages | 2 blocks per stage at widths 96/192/384/768, shifted 7³ windows; stage outputs after merging 192/384/768/1536 at strides 4/8/16/32 | MF row 19; CLS §2.4; SEG §7 item 5 |
| (c) window token | spatial means of stage 2-4 outputs concatenated: 384 + 768 + 1536 = 2688-d | MF row 20; CLS §2.4 `_encode_windows` |
| (c) heads | Lin: mean window token → linear; Attn: all window tokens, input projection 2688 → 768; LoRA+ABMIL | MF row 20; CLS §2.4 `_probe` |
| (d) cells | which encoder components each method trains; m = DoRA magnitude; 2 = adapter added in stage 2 | MF fig_methods_cls rows 1-9; CLS §3.1-3.5 |
| (d) head column | pooled → linear (Lin, and Lin→LoRA keeps the linear head); ABMIL → attentive probe (LoRA+ABMIL); attentive probe otherwise | CLS §3.1, §3.5 |
| (d) steps | 8000; two-stage 4000 + 4000; note: CVPR26 joint models also ran 26 000 (13 000 + 13 000) | CLS §3.8 |
| (d) encoder parameters | logged `backbone_trainable` minus the parameters that receive no gradient (Pillar-0 only) | `trainable_params_cls.csv`, `no_grad_cls.csv` |
| (d) † | VoCo Lin→LoRA / Attn→LoRA: no logged line; same late pattern (`layers4\.`) as LoRA_late, so its count (98,304) | CLS §3.1, §3.4 (**derived**) |
| (d) notes | $W_{qkv}$ = fused qkv (TAP-CT, VoCo) or q, kv (SPECTRE, Pillar-0); Attn-only FT incl. biases and VoCo's relative position bias; late blocks per backbone | CLS §3.2, §3.4; logged VoCo attnft 6,476,130 = 6,278,400 (qkv + proj) + 197,730 (bias tables) |
| (d) head parameters | linear 769 / 1081 / 1153 / 2689 and attentive probe 14.2 / 28.0 / 31.9 / 16.2 M, for one label | `trainable` of the Lin and Attn rows of `trainable_params_cls.csv` (per-task runs) |

Encoder counts shown (from the logs; Pillar-0 = parameters that receive a gradient):

| method | TAP-CT | SPECTRE | Pillar-0 (logged → with gradient) | VoCo |
|---|---|---|---|---|
| LoRA, rsLoRA, LoRA+ABMIL | 589,824 | 2,419,200 | 1,105,920 → 552,960 | 184,320 |
| LoRA_late | 196,608 | 1,036,800 | 737,280 → 245,760 | 98,304 |
| DoRA | 617,472 | 2,509,920 | 1,147,392 → 573,696 | 192,960 |
| Lin→LoRA, Attn→LoRA | 196,608 | 1,036,800 | 737,280 → 245,760 | 98,304 (derived) |
| Attn-only FT | 28,348,416 | 130,757,760 | 21,288,960 → 10,644,480 | 6,476,130 |
| Full FT | 87,083,520 | 396,917,280 | 94,144,193 → 44,843,456 | 73,894,914 |

### Pillar-0 gradient check
`MultiModalAtlas.forward` runs stage 1 on scales {32³, 8³, 2³}, stage 2 on {8³, 2³} and stage 3 on {2³} only.
`MultiScaleAttentionBlock` indexes its per-scale `CrossWindowBlock` and position embedding by the scales present
(`blocks[S]`, `posemb[S]`, `multimodal_msa.py` l.558-597), but every block holds three of each. Per-scale block 2 of
stage 2 and blocks 1-2 of stage 3 therefore never run. The text projection (`model.text.proj`, 13.8 M) and
`logit_scale` never run either. `lora_target_names [q, kv]` with pattern `visual\.` wraps all 36 per-scale blocks, and
`full_finetune` unfreezes the whole `CLIPMultimodalAtlas`. A forward/backward through `extract_vision_feats`
(`figures/check_pillar0_grads.py`, `aifofomo-fm-probe` env, random weights) gives a non-zero gradient to
44,843,456 of the 94,144,193 parameters, and to 7,962,624 of the 15,925,248 q/kv weights. This matches the count
`diagrams.py` derives from the safetensors header. `_extract_spatial` (Attn and LoRA+ABMIL heads) runs the same stage
loop.

## fig_arms_seg (7.0 x 5.42 in drawn)

| element | drawn fact | source |
|---|---|---|
| (a) TAP-CT shapes | 768×3×28² → 4·128×3×28² → 128×3×28² → 64×6×56² → 32×12×112² → 16×12×224² → C×12×224² | SEG §2 (stride (4,8,8)), §3 item 3 |
| (a) SPECTRE shapes | 1080×8³ → 4·128×8³ → 128×8³ → 64×16³ → 32×32³ → 16×64³ → 16×64×128² → C×64×128² (crop 64×128×128) | SEG §2 (stride (8,16,16), patch 64×128×128), §3 item 3 |
| (a) VoCo shapes | taps 192×24³, 384×12³, 768×6³, 1536×3³ → each projected and resized to 24³ → 4·128×24³ → 128×24³ → 64×48³ → 32×96³ → C×96³ | SEG §2 (strides 4-32, fuse grid stride 4), §3 item 3; MF pipeline_seg row 9 (24/12/6 **derived** as 96/stride) |
| (a) taps | blocks 3, 6, 9, 12 / 6, 12, 18, 24 after the final norm; VoCo stage outputs with parameter-free channel LayerNorm | SEG §2 table |
| (a) blocks | 3³ conv + InstanceNorm + LeakyReLU; trilinear upsampling on axes with stride > 1; no input skips | SEG §3 |
| (a) decoder parameters | 2.45 / 2.62 / 2.42 M for C = 15, logged (2,454,239 / 2,620,927 / 2,416,047) | `trainable_params_seg.csv` (dec rows, AMOS22 joint); equal to SEG §3's derived formulas |
| (a) Lin | 1×1×1 conv on the deepest tap (block 12, block 24, VoCo stage-4 output at stride 32 = 48 mm at 1.5 mm) + trilinear upsampling; 769 / 1081 / 1537 parameters per structure | SEG §3; logged Lin `trainable` 11,535 / 16,215 / 23,055 for C = 15 |
| (a) SPECTRE | per-crop ViT-L only in the features | SEG §2, §4 |
| (b) cells | Lin, Dec frozen; LoRA, DoRA, Dec→LoRA all blocks; LoRA_late blocks 9-12 / 17-24 / stage 4; Full FT everything | MF fig_methods_seg rows 1-5; SEG §4 |
| (b) steps | 8000; Dec→LoRA 4000 + 4000; note: AMOS22 and TotalSegmentator joint models 24 000 (12 000 + 12 000) | SEG §4 `_steps_for`, warm-start paragraph |
| (b) encoder parameters updated | logged `backbone_trainable` of the AMOS22 joint runs; SPECTRE (*) minus the feature combiner share, which receives no gradient: LoRA 2,419,200 − 345,600 = 2,073,600; DoRA 2,509,920 − 358,560 = 2,151,360; Full FT 396,917,280 − 58,389,120 = 338,528,160; LoRA_late 691,200 (pattern excludes the combiner) | `trainable_params_seg.csv`; SEG §4 (combiner wrapped, no gradient); safetensors header of `cclaess/SPECTRE-Large` (combiner 58,389,120 parameters + 15 rope buffer entries) |
| (b) note | the logged SPECTRE counts including the combiner (2.4 M / 2.5 M / 397 M) | `trainable_params_seg.csv` |

## param_numbers.tex
`\numParamsbbTapctLora` 0.59\,M (589,824), `\numParamsbbTapctAttnft` 28\,M (28,348,416),
`\numParamsbbSpectreLora` 2.4\,M (2,419,200), `\numParamsbbSpectreAttnft` 131\,M (130,757,760). The values are logged
classification `backbone_trainable` counts, formatted to two significant figures as in fig_arms (d). No gradient-less
parameters exist for these four, since the SPECTRE combiner is part of the classification path.

## Corrections vs the previous version
1. fig_pipeline (a): "native spacing" → "native depth" (the in-plane grid is resized to 224², so the spacing is not native).
2. fig_pipeline (a): attentive head "query token, 2 blocks → linear" → "learned CLS + K tokens, 2 self-attention blocks; CLS → linear" (PHASE1 D11).
3. fig_pipeline (b): crop "around foreground" → "centred on foreground with probability 0.67" (otherwise uniform).
4. fig_pipeline (b): taps now say "after the final norm" (D13).
5. fig_pipeline (b): "decoder (2.5 M)" → "2.45 M parameters for 15 structures", from the run log (D9).
6. fig_pipeline (c): the panel is TAP-CT, so the title now says so. "LoRA_late: last third" → "blocks 9–12"; Lin→LoRA, Attn→LoRA and Dec→LoRA added to the method lists. Attn-only FT "attention projections" → "the attention module: projections and biases" (D8).
7. fig_arms (a): crop size 128²×64 added. "64 crop tokens" / "all 65 tokens" → "N crop tokens" (64 at evaluation, 8 in training). The ABMIL head is labelled LoRA+ABMIL.
8. fig_arms (b): the title no longer calls Pillar-0 a "three-scale Atlas encoder" alone: it is "conv stem, three-stage multi-scale encoder" (D1). The figure now states that scales 32³, 8³, 2³ leave after stages 1, 2, 3. The verified max-pool Lin feature is now marked L2-normalised. "pooled" → "avg-pooled" for the Attn tokens. A LoRA+ABMIL head box was added (Pillar-0 has LoRA+ABMIL runs).
9. fig_arms (c): VoCo stage widths (96–768) and stage outputs (192/384/768/1536 at strides 4–32) are now stated separately (D4). The window token is shown as 384 + 768 + 1536 = 2688-d. The Attn input projection to 768-d is added. "train 16" → "≤ 16". "48³ × 96" → "96 × 48³" (channels first, as in the decoder shapes).
10. fig_arms (d): **Pillar-0 encoder counts were overstated.** Half the wrapped LoRA/Attn-only FT parameters, and more than half of Full FT's, never run (see the gradient check). The counts change: LoRA 1.1 → 0.55 M, LoRA_late 0.74 → 0.25 M, DoRA 1.1 → 0.57 M, Attn-only FT 21 → 11 M, Full FT 94 → 45 M. The logged values are given in the note. The column is now "encoder parameters updated".
11. fig_arms (d): VoCo two-stage counts are marked as derived (†; no logged line). New notes: what $W_{qkv}$ means per backbone, that Attn-only FT includes biases and VoCo's relative position bias, and the CVPR26 26 000-step joint schedule.
12. fig_arms_seg (a): the VoCo tap shapes are spelled out (192×24³, 384×12³, 768×6³, 1536×3³). The decoder counts are now from the logs (unchanged values). "SPECTRE: per-crop ViT-L, no combiner" → "per-crop ViT-L only".
13. fig_arms_seg (b): the SPECTRE counts (2.1 / 0.69 / 2.2 / 2.1 / 339 M) are kept but labelled correctly. Before, the figure implied the combiner is not adapted (D6). Now the column says "encoder parameters updated", and a footnote states that LoRA, DoRA and Full FT also wrap or unfreeze the combiner, which receives no gradient, with the logged totals (2.4 / 2.5 / 397 M).
14. fig_arms_seg (b): the 24 000-step (12 000 + 12 000) budget of the AMOS22 and TotalSegmentator joint models is now stated (D12). Before, only 8000 / 4000 + 4000 appeared.

## Not verified / derived
- VoCo Lin→LoRA and Attn→LoRA encoder counts (98,304): derived from the shared late-block pattern; no log.
- TAP-CT and VoCo counts are logged trainable counts. They were not checked numerically for parameters without a
  gradient. A DINOv2-style TAP-CT `mask_token` would be 768 parameters and would not change any printed value.
- The Pillar-0 gradient check used random weights and the pooled path. The spatial path runs the same stage loop
  (read, not executed).

## Revision of 2026-11-02
fig_pipeline: panel headers without "TAP-CT" ((a) Classification, (b) Segmentation, (c) What each method trains in
every block); (c) redrawn with 10 pt (drawn) sub-boxes, an attention row without a box, residual adds in the gaps,
the same sub-box width in every card and a wider LoRA card holding A and B; the Attn head text of (a) on three lines;
canvas 5.52 in high (was 5.32) with the colour key below (c). Content unchanged.
