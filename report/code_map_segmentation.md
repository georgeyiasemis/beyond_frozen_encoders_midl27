# Code map: segmentation path (fm_seg)

Audit date 2026-10-31. Source of truth: files on disk in
`/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo` including uncommitted changes
(`git status`: `aifofomo/data/datasets/ct_segmentation.py`, `aifofomo/models/networks/seg_probe/*`,
`aifofomo/models/networks/optim_groups.py`, `examples/fm_probes/fm_seg/generate.py`, `build_manifests.py`,
`make_smoke.py` are modified and not committed). Draft used for orientation only:
`/home/g.yiasemis/projects/cvpr_ct_challenge/midl_paper/paper/main.tex` (line numbers below refer to it).

Abbreviations: `R` = `/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo`; `SEG` = `R/examples/fm_probes/fm_seg`;
`SP` = `R/aifofomo/models/networks/seg_probe`. "step" = optimizer step; "iteration" = micro-batch
(the engine counts micro-batches; every config has `gradient_accumulation_steps: 2`, so 1 step = 2 iterations).

---

## 0. Where things live

| Item | File |
|---|---|
| Config generator (all run YAMLs) | `SEG/generate.py` |
| Manifests (fixed splits) | `SEG/build_manifests.py` → `SEG/manifests/{amos22,kits23,lits,msd_pancreas,totalseg}.csv` |
| TotalSegmentator label merge | `SEG/totalseg_classes.py` (`SOURCES_117`, `TAPCT_49`, `MERGE_15`, `VALUE_15`), `SEG/totalseg_labelmaps.py` (writes `labels_117/49/15.nii.gz`) |
| Dataset | `R/aifofomo/data/datasets/ct_segmentation.py` (`CTSegmentationDataset`) |
| Resampling helpers | `R/aifofomo/data/datasets/ct_classification.py` (`_resize_volume_1dhw`, `_spacing_resample_nifti_array`), `R/aifofomo/data/datasets/nifti_io.py` (`read_nifti_dhw`) |
| Transforms | `R/aifofomo/data/transforms/modules/spatial.py` (`RandomFlip3D` l.659, `RandomZoom3D` l.769), `.../modules/intensity.py` (`ScaleIntensityRange` l.152) |
| Model | `SP/seg_probe.py` (`SegProbeModel`), `SP/seg_probe_config.py`, `SP/backbones.py`, `SP/heads.py` |
| LoRA / DoRA | `R/aifofomo/models/networks/voco/lora.py`, `.../voco/dora.py` |
| Warm start resolver | `R/aifofomo/models/networks/voco/probe_init.py` (`resolve_checkpoint_path`, `_resolve_with_max_iteration`) |
| Optimizer groups | `R/aifofomo/models/networks/optim_groups.py` |
| Loss | `R/aifofomo/losses/dice_bce_seg_loss.py` (`DiceBCESegLoss`) |
| Metric | `R/aifofomo/metrics/seg_dice_metric.py` (`SegDiceMetric`) |
| Scheduler | `R/aifofomo/lr_schedulers/lr_schedulers.py` (`_WarmupCosineLRScheduler` l.247), `R/aifofomo/utils/scheduling.py` (`cosine_schedule`) |
| Training loop | `R/aifofomo/trainers/trainer.py` (`train_iteration` l.820, `validate` l.1135) |
| Inference records | `R/aifofomo/inference/inferencer.py` (per-case Dice records l.449–478) |
| Hooks | `R/aifofomo/engine/hooks.py` (`ValidationCaseJsonHook` l.478, `CheckpointHook` l.1270) |
| Seed | `R/aifofomo/environment.py` (`--seed` default 42, l.434–441; `_seed_everything` l.189) |
| Per-case Dice recomputation / TotalSeg held-out eval | `SEG/reinfer_dice.py`, `SEG/make_reinfer_jobs.py`, `R/examples/fm_probes/fm_adapt/submit_batched.sh` (`test_eval`, l.210–225) |
| Site paths | `R/examples/fm_probes/fm_adapt/paths.env` |

---

## 1. Data

### 1.1 Datasets and manifests (`SEG/build_manifests.py`)

Every manifest is `case_id,image,label,split` with paths relative to the dataset root (`build_manifests.py` l.7–8).
Split rules are deterministic with `SEED = 0` (l.34), `random.Random(SEED).shuffle(ids)` on sorted ids
(`_shuffled_split`, l.49–60; first `n_test` → test, next `n_val` → val, rest → train).

| Dataset | Root (`paths.env`) | Source / official split used | Selection ("val") / held-out ("test") construction | Counts train/val/test (counted from CSV) |
|---|---|---|---|---|
| amos22 | `FM_DATA_AMOS22=/mnt/kronos/data_public/AMOS22/amos22` | official `imagesTr` → train, official `imagesVa` → val+test; CT only (`int(cid.split("_")[1]) >= 500: continue  # MRI`, l.72–73) | `imagesVa` sorted by name, `rank % 2 == 0 → "val"`, odd → `"test"` (l.76–78) | 200 / 50 / 50 |
| kits23 | `FM_DATA_KITS23=/mnt/kronos/data_public/KiTS23/dataset` | all `case_\d{5}` dirs with `segmentation.nii.gz` (l.83–84); no official val used | one shuffle, `n_val = n_test = round(0.15*n)` (l.86–88) | 343 / 73 / 73 (489 cases) |
| lits | `FM_DATA_MSD=/mnt/kronos/data_public/MSD`, `Task03_Liver/imagesTr` | MSD Task03 training set (= LiTS17 training volumes); MSD test images unlabeled, unused (l.22) | one shuffle, `n_val=20, n_test=20` (l.130) | 91 / 20 / 20 (131) |
| msd_pancreas | `FM_DATA_MSD`, `Task07_Pancreas/imagesTr` | MSD Task07 training set | one shuffle, `round(0.15*n)` each (l.131–133) | 197 / 42 / 42 (281) |
| totalseg | `FM_DATA_TOTALSEG=/mnt/kronos/data_public/TotalSegmentator-v3` | `meta.csv` (`;`-separated, `image_id`, `split`): official `test` → test; official `train` shuffled, first `n_val` (default = len(test)) → val, rest → train (l.99–112) | as left | 1721 / 109 / 109 (1939; `meta.csv`: 1830 train, 109 test) |

Patient-level guarantee: there is no patient column anywhere; `_read_manifest` only checks duplicate `case_id`
within a split (`ct_segmentation.py` l.60–62). KiTS23 is one case per patient by construction of that dataset;
AMOS22 split follows the official train/val partition. `meta.csv` of TotalSegmentator has columns
`image_id;age;institute;study_type;split;...;gender` and no patient id. [TODO: patient-level disjointness of
val/test halves of AMOS22 `imagesVa`, of LiTS/MSD volumes and of TotalSegmentator cannot be verified from code.]

### 1.2 Label values and class names (`SEG/generate.py` `DATASETS`, l.108–154)

| Dataset | `channels` (output order) | `class_values` (label-map values counted as foreground per channel) |
|---|---|---|
| amos22 | `spleen, right_kidney, left_kidney, gallbladder, esophagus, liver, stomach, aorta, postcava, pancreas, right_adrenal, left_adrenal, duodenum, bladder, prostate_uterus` (`AMOS_ORGANS`, l.90–106) | `{name: [i+1]}` i.e. 1..15 (l.113); matches `/mnt/kronos/data_public/AMOS22/amos22/dataset.json` labels 1–15 |
| kits23 | `kidney, tumor, cyst` | `kidney: [1,2,3]`, `tumor: [2]`, `cyst: [3]` (l.120; "the kidney channel is the whole organ") |
| lits | `liver, tumor` | `liver: [1,2]`, `tumor: [2]` (l.126); Task03 `dataset.json`: 1 liver, 2 cancer |
| msd_pancreas | `pancreas, tumor` | `pancreas: [1,2]`, `tumor: [2]` (l.132); Task07 `dataset.json`: 1 pancreas, 2 cancer |
| totalseg | `list(MERGE_15)` = `lungs_airway, heart, arteries, veins, liver, spleen, pancreas, kidneys, gi_tract, bladder_prostate, glands, brain_spinal_cord, spine, other_bones, muscles` | `{k: [VALUE_15[k]]}` = 1..15 in that order (l.147); `label_map_15.json` on disk matches (checked by `configure`, l.72–78) |

So KiTS23 kidney, LiTS liver, MSD pancreas are "whole organ incl. lesion" channels and overlap with the tumor/cyst
channel: the targets are multi-channel binary masks (`_targets`, `ct_segmentation.py` l.321–328:
`np.isin(label, self.class_values[c])` per channel), so overlapping channels are allowed; the loss is per-channel
sigmoid (Section 6).

### 1.3 TotalSegmentator merge (`SEG/totalseg_classes.py`)

117 source structures (`SOURCES_117`, l.18–33; value = 1 + alphabetical index). TAP-CT 48 classes (`TAPCT_49`, l.37–86;
left/right pairs, 24 ribs, 25 vertebrae merged). `MERGE_15` (l.94–111), exact code:

| Merged class (value) | TAP-CT classes merged | Source TotalSegmentator structures |
|---|---|---|
| lungs_airway (1) | lungs, trachea | lung_lower_lobe_left, lung_lower_lobe_right, lung_middle_lobe_right, lung_upper_lobe_left, lung_upper_lobe_right, trachea |
| heart (2) | heart, atrial_appendage | heart, atrial_appendage_left |
| arteries (3) | aorta, brachiocephalic_trunk, subclavian_artery, common_carotid_artery, iliac_artery | aorta, brachiocephalic_trunk, subclavian_artery_left/right, common_carotid_artery_left/right, iliac_artery_left/right |
| veins (4) | superior_vena_cava, inferior_vena_cava, brachiocephalic_vein, iliac_vein, portal_vein_and_splenic_vein, pulmonary_vein | superior_vena_cava, inferior_vena_cava, brachiocephalic_vein_left/right, iliac_vena_left/right, portal_vein_and_splenic_vein, pulmonary_vein |
| liver (5) | liver, gallbladder | liver, gallbladder |
| spleen (6) | spleen | spleen |
| pancreas (7) | pancreas | pancreas |
| kidneys (8) | kidneys, kidney_cysts | kidney_left/right, kidney_cyst_left/right |
| gi_tract (9) | esophagus, stomach, duodenum, small_bowel, colon | same five |
| bladder_prostate (10) | urinary_bladder, prostate | urinary_bladder, prostate |
| glands (11) | thyroid_gland, adrenal_glands | thyroid_gland, adrenal_gland_left/right |
| brain_spinal_cord (12) | brain, spinal_cord | brain, spinal_cord |
| spine (13) | vertebrae, sacrum | vertebrae_C1–C7, T1–T12, L1–L6 (25), sacrum |
| other_bones (14) | ribs, sternum, costal_cartilages, clavicula, scapula, humerus, hips, femur, skull | rib_left/right_1–12 (24), sternum, costal_cartilages, clavicula_left/right, scapula_left/right, humerus_left/right, hip_left/right, femur_left/right, skull |
| muscles (15) | gluteus_maximus, gluteus_medius, gluteus_minimus, iliopsoas, autochthon | gluteus_maximus/medius/minimus_left/right, iliopsoas_left/right, autochthon_left/right |

Label maps: `totalseg_labelmaps.py` `merge_case` (l.87–118) paints structures largest-first so that an overlapping voxel
goes to the smaller structure (`sorted(masks, key=lambda n: int(masks[n].sum()), reverse=True)`, l.100); `labels_49`
and `labels_15` are derived from `labels_117` by lookup tables (`write_49`, l.78–84). The experiments read
`<case>/labels_15.nii.gz` (`build_totalseg`, l.112). Docstring (l.23–27): overlaps ~0.1–0.7 % of foreground.

### 1.4 Geometry, intensity, crops, sampling, augmentation, caching

Per-backbone dataset entry (`generate.py` `_dataset_entry`, l.262–288; constants l.211–213):

| | tapct | spectre | voco |
|---|---|---|---|
| orientation | `LPS` | `RAS` | `RAS` |
| grid | `inplane_size: 224` (resize H/W to 224, native depth; `_resize_volume_1dhw(..., keep_native_depth=True)`, `ct_segmentation.py` l.175–180) | `target_spacing: [1.0, 1.0, 1.0]` mm (SimpleITK resample, image linear, label nearest; l.164–170, `_spacing_resample_nifti_array`) | `target_spacing: [1.5, 1.5, 1.5]` |
| intensity (YAML transform) | none in YAML; inside the model: `clamp(-1008, 822)`, `(x - (-86.8086)) / 322.6347` (`backbones.py` l.43–44, 110; `tapct_cls.py` l.31–32) | `ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=true)` | `ScaleIntensityRange(a_min=-175, a_max=250, b_min=0, b_max=1, clip=true)` |
| training `patch_size` (= model `roi_size`) | `[12, 224, 224]` | `[64, 128, 128]` | `[96, 96, 96]` |
| `fg_prob` | 0.67 | 0.67 | 0.67 |
| val/test | whole volume, `target_format: labelmap` (uint8 `(1,D,H,W)`; l.285–286) | same | same |

HU rounding: images are rounded to integers and clipped to int16 before caching (`ct_segmentation.py` l.186–188).

Foreground sampling (`_random_crop`, l.306–318 and the memory-mapped `_fast_random_crop`, l.265–295 — uncommitted,
same draws): with probability `fg_prob` a random voxel of `np.isin(label, self._fg_values)` is the crop centre, where
`_fg_values` is the union of the label values of the run's channels (l.131–133: "a per-task cyst run must centre on
cysts"); the start is clipped so the crop stays inside the volume; otherwise a uniform random crop. Volumes smaller
than the patch are padded with the image minimum / label 0 (`_pad_to`, l.299–304). One crop per `__getitem__`.

Augmentations (train only, `generate.py` `_transforms`, l.252–258):
`RandomFlip3D(keys=[data, seg_target], prob=0.5, axis_prob=1.0, axes=[2])` → left–right flip with probability 0.5
(axis 2 = W; `spatial.py` l.701–720); `RandomZoom3D(keys=[data, seg_target], prob=0.5, scale=[0.85,1.15],
mask_keys=[seg_target])` → with probability 0.5 a single scale s ~ U(0.85, 1.15): s<1 crops a random sub-volume of
relative size s and resizes back (trilinear for data, nearest for the mask); s>1 pads (image with its minimum, mask
with 0) at a random offset and resizes back (`spatial.py` l.810–878). Intensity scaling for spectre/voco is in the
same list, so it is also applied at validation. No test-time augmentation.

Caching (`_load_case`, l.195–222): `<FM_SAVE_ROOT>/seg/cache/<dataset_name>/<orientation>_<grid>_<sha1(manifest)[:8]>/<case>.npz`
with `image` int16 and `label` uint8 (+ `label_src`); the uncommitted fast path also writes `.img.npy/.lab.npy/.hist.npy`
memory maps and per-slice label histograms (l.232–263). The seg cache root in the generated YAMLs is
`/projects/direct/temp/fm_probe/v2/seg/cache`.

Dataloaders (YAML): train `batch_size: 2, num_workers: 8`; validation `batch_size: 1, num_workers: 4`
(`generate.py` l.393–395). Training uses the framework's `InfiniteSampler(shuffle=True)` (`data/samplers.py` l.55–84).

---

## 2. Backbones as dense feature extractors (`SP/backbones.py`)

Common interface (`DenseBackbone`, l.49–71): `embed_dim`, `stride (D,H,W)`, `taps`, `tap_channels`, `fuse_index`,
`input_pad_value`, `lora_target_names`, `late_pattern`, `features(x) -> list of (B,C,d,h,w)`.

| | TAP-CT (`TapCTDenseBackbone`, l.74–117) | SPECTRE (`SpectreDenseBackbone`, l.120–165) | VoCo (`VoCoDenseBackbone`, l.171–209) |
|---|---|---|---|
| weights | HF `fomofo/tap-ct-b-3d` (config: `img_size [12,224,224]`, `patch_size [4,8,8]`, 4 register tokens, 12 blocks, 768-d) | HF `cclaess/SPECTRE-Large` (24-block ViT-L, 1080-d, patch `(16,16,8)` in (X,Y,Z), `crop_size (128,128,64)`) | `FM_VOCO_WEIGHTS=/projects/cvpr2026_challenge_fomo4ct/pretrained/VoCo_L_SSL_head.pt`, MONAI `SwinUNETR(feature_size=96, use_v2=True).swinViT` (`voco_cls.py` `build_voco_swin`, l.47–84) |
| taps (0-based blocks) | `TAPCT_DEFAULT_TAPS = (2, 5, 8, 11)` → blocks 3, 6, 9, 12 (l.45) | `SPECTRE_DEFAULT_TAPS = (5, 11, 17, 23)` → blocks 6, 12, 18, 24 (l.46) | `VOCO_DEFAULT_TAPS = (1, 2, 3, 4)` → Swin stage outputs after stages 1–4 (l.168) |
| token → grid | `vit.get_intermediate_layers(x, n=list(taps), reshape=True, norm=True)` (l.116): final LayerNorm applied to every tap, CLS and 4 register tokens dropped, reshaped to `(B, 768, D/4, H/8, W/8)` (remote code `vision_transformer_base.py` l.595–607) | `vit.forward_intermediates(xyz, indices=taps, norm=True, output_fmt="NCHWD", intermediates_only=True)` after permuting (D,H,W)→(X,Y,Z); final norm on each tap; prefix token dropped; permuted back (l.161–165) | `swin(x_xyz, normalize=True)` returns 5 stage maps; `normalize=True` applies a parameter-free `F.layer_norm` over channels to each stage output (MONAI `proj_out`, `swin_unetr.py` l.1258–1270); taps 1–4 are taken and permuted back (l.208–209) |
| stride (voxels per token, D,H,W) | `(4, 8, 8)` (l.97) | `(8, 16, 16)` (l.140) | fuse grid stride `(4,4,4)` = `2**(taps[0]+1)` (l.192–193); taps are at strides 4, 8, 16, 32 |
| channels per tap | 768 ×4 | 1080 ×4 | `(192, 384, 768, 1536)` = `96 * 2**t` (l.196–197); `embed_dim` (for Lin) = 1536 (l.191) |
| tokens per crop | 3×28×28 = 2352 | 8×8×8 = 512 | stage-4 grid 3³ |
| input constraint | depth padded to a multiple of 4 with the normalised HU minimum and the extra token row cut (l.111–117) | window must be exactly `(64,128,128)` (l.159–160; config check `seg_probe_config.py` l.69–70) | every axis a multiple of 32 (l.206–207) |
| `input_pad_value` (sliding-window border fill) | `-1008.0` HU (l.77) | `0.0` (l.123) | `0.0` (l.174) |
| LoRA targets | `("qkv",)` (l.75) | `("q", "kv")` (l.121) | `("qkv",)` (l.172) |
| late pattern | `blocks\.(8|9|10|11)\.` → blocks 9–12 (l.76) | `backbone\.blocks\.(1[6-9]|2[0-3])\.` → blocks 17–24 (l.122) | `layers4\.` → stage 4 (2 blocks) (l.173) |
| module LoRA/FT acts on (`module` property) | `self.hf` (whole HF model) (l.103–105) | `self.extractor` = `SpectreImageFeatureExtractor`, which holds `backbone` AND `feature_combiner` (l.147–149; HF config `include_feature_combiner=True` by default) | `self.swin` (l.199–201) |

Sliding window at evaluation (`SP/seg_probe.py` `forward`, l.315–333): MONAI `sliding_window_inference(roi_size,
sw_batch_size, overlap, mode="gaussian", padding_mode="constant", cval=input_pad_value, sw_device=GPU, device=CPU)`,
i.e. logits are stitched on the CPU (`sw_output_device: "cpu"` default, `seg_probe_config.py` l.63–65).
Per backbone (`generate.py` `SW`, l.212): tapct overlap 0.5 / sw_batch 4; spectre 0.25 / 8; voco 0.5 / 4.
Training runs one crop through `_predict_window` (l.252–258); for `frozen_backbone` the features are computed under
`no_grad` and detached and the backbone is kept in `eval()` (`train`, l.245–249).

---

## 3. Heads (`SP/heads.py`)

**Lin** (`LinearSegHead`, l.33–39): `nn.Conv3d(in_channels=embed_dim, num_classes, kernel_size=1)` on `feats[-1]`
(the deepest tap: TAP-CT block 12, SPECTRE block 24, VoCo stage 4 at stride 32), then `F.interpolate(mode="trilinear",
align_corners=False)` to the crop size (`_up`, l.27–30). Parameters = `(embed_dim + 1) * C`: TAP-CT 769·C,
SPECTRE 1081·C, VoCo 1537·C (derived, not logged).

**Dec** (`MultiTapSegDecoder`, l.51–100; `decoder_channels` default 128, `seg_probe_config.py` l.44):
1. `proj`: one `Conv3d(c_tap, 128, k=1)` per tap (l.72); each projected tap is trilinearly resized to the grid of
   `feats[fuse_index]` (`fuse_index=-1` for ViTs = all taps share one grid; `0` for VoCo = stride-4 grid) and
   concatenated along channels → 512 channels (l.94–95).
2. `fuse`: `_ConvBlock(512, 128)` = `Conv3d(k=3, pad=1, bias=False)` + `InstanceNorm3d(affine=True)` +
   `LeakyReLU(0.01)` (l.42–48, 73).
3. Upsampling stages (l.75–88): `n_stages = max(ceil(log2(stride_axis)))`; at each stage every axis whose remaining
   stride > 1 is doubled by trilinear interpolation, followed by `_ConvBlock(cin, max(16, cin//2))`:
   TAP-CT stride (4,8,8): 3 stages, scales (2,2,2), (2,2,2), (1,2,2), channels 128→64→32→16;
   SPECTRE (8,16,16): 4 stages (2,2,2)×3 then (1,2,2), 128→64→32→16→16;
   VoCo (4,4,4): 2 stages, 128→64→32.
4. `out`: `Conv3d(cin, C, k=1)` (l.89), then a final `_up` to the crop size (identity when the grid already matches).
No skip connections from the input volume (forward uses only `feats`). Parameter counts (derived from the code,
not logged anywhere): TAP-CT 2,453,984 + 17·C (2.454 M at C=15); SPECTRE 2,620,672 + 17·C (2.621 M);
VoCo 2,415,552 + 33·C (2.416 M).

Output activation: none in the model; the loss uses `sigmoid` / `binary_cross_entropy_with_logits` and evaluation
thresholds logits at `logit(0.5) = 0` (`seg_probe.py` l.288, 293). One channel per structure (`num_classes =
len(chans)`), so overlapping classes (KiTS kidney ⊃ tumor ⊃ ...) are independent sigmoids. Training targets are
`(C,D,H,W)` float one-hot-per-channel (`_targets`, onehot); evaluation targets are uint8 label maps expanded one class
at a time with `torch.isin(lab, class_values[k])` (`_dice`, l.291–298). `seg_mask` is always all-True
(`ct_segmentation.py` l.343).

---

## 4. Methods (`SEG/generate.py` `METHODS`, l.167–196; model handling `SP/seg_probe.py`)

| key | `training_mode` | head | LoRA scope | lr | wd | steps | trainable parameters (code) |
|---|---|---|---|---|---|---|---|
| lin | frozen_backbone | linear | – | 1e-2 | 0.0 | full | head only (`_configure_backbone` freezes all backbone params, l.193–194) |
| dec | frozen_backbone | decoder | – | 1e-3 | 1e-4 | full | head only |
| lora | lora | decoder | all (`apply_lora_to_module`, targets per backbone, no path pattern) | 1e-4 | 1e-4 | full | LoRA A/B of every matched Linear + head |
| lorag | lora | decoder | late (`lora_include_pattern = LATE_PATTERN[backbone]`) | 1e-4 | 1e-4 | full | LoRA A/B in late blocks + head |
| dora | dora | decoder | all | 1e-4 | 1e-4 | full | DoRA A/B + magnitude of every matched Linear + head |
| fullft | full_finetune | decoder | – | 1e-3 (unused: groups below) | 1e-4 | full | every backbone parameter (`p.requires_grad_(True)`, l.189–192) + head; `backbone_lr 2e-5`, `head_lr 1e-3`, `backbone_weight_decay 1e-4`, `head_weight_decay 1e-4` |
| dec_then_lora | lora | decoder | all | 1e-4 | 1e-4 | stage2 (half) | as lora; head warm-started (below) |
| scratch (revision, `--revision` only; `METHODS["scratch"] = {**fullft, "random_init": True}`, l.192) | full_finetune + `random_init: true` | decoder | – | as fullft | | full | as fullft; backbone re-initialised by `reinitialize_` (trunc-normal 0.02 Linear, identity norms, `reset_parameters` for conv/patch-embed, tokens/pos-embed trunc-normal; `seg_probe.py` l.68–93). Exists only for tapct (`iter_revision_runs`, l.479). |

LoRA (`voco/lora.py`): `W0 x + (α/r)·B·A·drop(x)`, `scaling = alpha / r` (l.81), A Kaiming-uniform(a=√5), B zeros
(l.91–94), dropout before the low-rank branch (l.99). Settings for every seg run: `r=16, alpha=32.0, dropout=0.1`
(`generate.py` `LORA`, l.209) → scale 2. A Linear is replaced when its attribute name is in `target_names` and (if
given) its module path matches `include_path_pattern` (l.173–189). DoRA (`voco/dora.py` l.112–131):
`W = m ⊙ (W0 + (α/r)BA) / ||W0 + (α/r)BA||_row`, `m` initialised to row norms (l.101–103), dropout applied to the
input `x` of the whole layer (l.131), bias frozen.

Layers adapted (derived from backbone module structure): TAP-CT 12 `qkv` (late: 4); SPECTRE 24 `q` + 24 `kv` in the
ViT backbone, **plus the `q`/`kv` layers of the feature combiner** (`FeatureVisionTransformer` built from the same
`Block`/`Attention`, `spectre/models/vision_transformer_features.py` l.17, 21) because `module` is the whole extractor
and `lora`/`dora`/`fullft` use no path pattern; those combiner adapters never enter `features()` (only
`self.vit.forward_intermediates`) so they get no gradient (AdamW skips `grad is None`), but they are counted in
`_log_param_summary` and saved in trainable-only checkpoints. `lorag`'s pattern `backbone\.blocks\.` excludes the
combiner. VoCo: 8 `qkv` (2 blocks × 4 stages, MONAI `depths=(2,2,2,2)`), late: 2 (`layers4`).
LoRA parameter counts (derived): TAP-CT 589,824; SPECTRE ViT 2,073,600 (+ combiner, unverified count);
VoCo 184,320 (late 98,304).

Optimizer groups (`seg_probe.py` `get_optimizer_param_groups`, l.229–243 → `optim_groups.py`): non-fullft methods
use `build_decay_param_groups` (l.25–46): weight decay applies to parameters with `ndim >= 2` whose name does not
end in `.bias` and does not contain `norm`/`bn`/`ln`; everything else gets wd 0. Hence LoRA A/B (2-D) and conv
weights decay; InstanceNorm affine and biases do not; **the DoRA magnitude `(out, 1)` has `ndim == 2` and is named
`magnitude`, so it receives weight decay 1e-4**. `fullft` uses `build_backbone_head_param_groups` with prefixes
`backbone.` / `head.` (l.49–105); unknown trainable params fall into the head group.

Warm start (dec_then_lora): `head_init_checkpoint: <run root>/<dataset>/<backbone>/<mode>/dec[/<task>]`,
`head_init_max_iteration: stage1_steps * 2` (`generate.py` l.330–331, 353–356). `load_prefixed_weights(model,
ref, "head.", max_iteration)` (`seg_probe.py` l.35–65) → `resolve_checkpoint_path` requires the exact periodic file
`checkpoints/model_<max_iteration>.pt` of the parent (`probe_init.py` `_resolve_with_max_iteration`, l.98–117;
anything else is an error) and loads only `head.*` tensors of matching shape, non-strict. The parent `dec` run keeps
that checkpoint via `keep_iterations: [<stage1*2>]` (`generate.py` l.357; `PARENT_METHODS = {"dec"}`). Adapters start
fresh (B = 0), optimizer and schedule fresh. Budgets: 8000-step modes → parent at iteration 8000 (= step 4000),
child trains 4000 steps (`steps = stage2_steps`, l.345); joint_lm → parent at iteration 24000 (= step 12000), child
12000 steps; pertask_cm → parent at step 800, child 800 steps (`_steps_for`, l.216–223).

Schedule (YAML `lr_scheduler: warmup_cosine`): `max_iterations = steps` (optimizer steps, since the scheduler steps
once per optimizer step: `trainer.py` l.871–876 `if is_last_microbatch: ... optimizer_step(); scheduler_step()`),
`warmup_iterations = round(0.1*steps)` (`generate.py` l.351), `warmup_factor 0.01`, linear warm-up from 1 % of
peak (`lr_schedulers.py` l.298–305, 497–500), then `end + (base - end)·0.5·(1 + cos(π·t/T))` with `end_lr = 1e-6`
(`utils/scheduling.py` l.23; l.307–316). `gradient_clipping: 0.0` (none). Loss per micro-batch divided by the
accumulation factor (`trainer.py` l.866). Micro-batch 2 crops × accumulation 2 → 4 crops per step
(`MICRO_BATCH = 2, ACCUM = 2`, l.207–208). Precision `bf16-mixed` (YAML `fabric.precision` and the sbatch `--precision`).
AdamW betas/eps: framework defaults [TODO: confirm `optimizers/` defaults = (0.9, 0.999), 1e-8].

Steps per mode (`_steps_for`): pertask/joint 8000 (two-stage 4000+4000); joint_lm 24000 (12000+12000);
pertask_cm 1600 (800+800). YAML `num_iterations = steps*2`.

Seeds: `environment.py` `--seed` default 42, `seed_everything(seed + global_rank, workers=True)` (l.189–193).
Seed repeats: `generate.py` `SEED_MODES = {"joint_s43": 43, "joint_s44": 44}` write the joint (or joint_lm for
amos22/totalseg) config with a `# seed: N` header line only (l.338–340, 362); `fm_adapt/submit_batched.sh` l.237–239
parses that header and passes `--seed N`; every other run uses 42. The seed-repeat run directory is
`<mode>/joint_s4x/<method>`.

---

## 5. Modes (`generate.py` `modes_for` l.160–162, `iter_runs` l.483–494, `iter_revision_runs` l.472–480)

| mode | datasets | methods | tasks | steps | notes |
|---|---|---|---|---|---|
| pertask | all 5 | all 7 (lin, dec, lora, lorag, dora, fullft, dec_then_lora); **totalseg: `pertask_methods = ["lin","dec","lora","fullft"]`** (l.150) | one YAML per channel (`num_classes: 1`) | 8000 | per-task cm restriction confirmed on disk: `totalseg/*/pertask/{dec,fullft,lin,lora}` only |
| joint | all 5 | all 7 | one YAML, all channels | 8000 | |
| joint_lm | amos22, totalseg | all 7 | all channels | 24000 | on disk: tapct and spectre have all 7; **voco has no `joint_lm/dec_then_lora`** (`amos22/voco/joint_lm` and `totalseg/voco/joint_lm` hold dec, dora, fullft, lin, lora, lorag only) |
| joint_s43 / joint_s44 (revision) | all 5 | dec, lora, fullft | all channels | 8000, or 24000 for amos22/totalseg | on disk for tapct and spectre only (no voco seed repeats) |
| pertask_cm (revision) | amos22, totalseg | dec, lora, fullft | per channel | 1600; `validation_interval = save_interval = 1600//8*2 = 400` iterations (= 200 steps, 8 validations) (l.349–350) | on disk for tapct and spectre only |
| scratch (revision) | all 5, tapct only | fullft settings + `random_init` | joint (joint_lm for amos22/totalseg) | as the mode | `amos22/tapct/joint_lm/scratch`, `totalseg/tapct/joint_lm/scratch`, `kits23|lits|msd_pancreas/tapct/joint/scratch` |

Config tree on disk: 1356 YAMLs (324 of them `eval/` copies for totalseg), 1020 sbatch files. The totalseg YAMLs
have only the `val` split in `validation_datasets` (`include_test=not test_at_end`, l.530–531); a sibling
`eval/<stem>.yaml` has val + test and is used after training by `test_eval` (`fm_adapt/submit_batched.sh` l.210–225).

Loss averaging over structures (joint): `DiceBCESegLoss.compute` (l.115–155): Dice and BCE are computed per
`(case, channel)`, multiplied by `seg_mask` (all ones) and divided by the number of known (b, c) pairs, i.e. a plain
mean over cases and channels; `total = 1.0*dice + 1.0*bce` (YAML `dice_weight 1.0, bce_weight 1.0, multiplier 1.0`).

---

## 6. Loss, selection, Dice, outputs

**Loss formula** (`dice_bce_seg_loss.py` l.115–155), per case b and channel c with p = σ(z) over the V voxels of the crop:
`BCE_bc = mean_v BCEWithLogits(z, y)`; `Dice_bc = 1 - (2·Σ_v p·y + 1) / (Σ_v p + Σ_v y + 1)` (`smooth = 1.0`);
`L = (Σ_bc m_bc (Dice_bc + BCE_bc)) / Σ_bc m_bc`, m ≡ 1. Logged components `seg_dice`, `seg_bce`, total.

**Validation schedule**: `validation_interval = 1000 steps * 2 = 2000` iterations (`generate.py` l.347;
`trainer.py` l.795 checks `iteration % validation_interval == 0`), `compute_losses: false`. For pertask_cm: 400
iterations. `trainer.validate` (l.1135–1210) groups validation loaders by `dataset.split`: `val` → metrics logged
with prefix `val/`, `test` → prefix `test/`; per-case records get `row["split"] = prefix`. Both splits are evaluated
at every validation for every dataset except totalseg (val only during training).

**Per-case Dice** (`seg_probe.py` `_dice`, l.260–306): `p = logits > logit(0.5)`, `t = isin(labelmap,
class_values[k])` (or one-hot `> 0.5`); `inter = |p∧t|`, `denom = |p| + |t|` in float64 on the GPU one class at a time;
`valid = (denom > 0) & seg_mask`; `dice = 2·inter/denom` where valid, **0 where not valid** (empty reference and empty
prediction → dice 0, valid False). Reference empty but prediction non-empty → dice 0, valid True (counted). No NaN is
produced. Dice is on the working grid (resampled label, nearest neighbour), no post-processing.

**Metric** (`seg_dice_metric.py` l.160–177): `dice/<class>` = mean over cases with `valid`; `val_dice`
(display name) = mean over classes that have ≥1 valid case. Logged as `val/metric/val_dice`, `val/metric/dice/<class>`,
`test/metric/...` (`summarize_results.py` l.5–8, 76–84).

**Selection** (`CheckpointHook`, YAML `monitor_metric: val/metric/val_dice, mode: max, save_best: true,
best_metrics: [val/metric/val_dice]`): strict improvement `current > best` (`hooks.py` l.1615–1618) saves
`checkpoints/best/best_val_metric_val_dice.pt` and writes `best_val_metric_val_dice.txt` with `best_value` and
`iteration` (l.1405–1414). Periodic `model_<iteration>.pt` every `save_interval = 1000` iterations (= 500 steps),
`max_checkpoints: 2` rotation, protected `keep_iterations`; `trainable_only: true` stores only `requires_grad`
parameters plus buffers (`_trainable_state_dict`, l.1587–1611). Selection uses the `val` split only (metric name
prefix); the `test` metric at the same iteration is what is reported (`summarize_results.py` l.76–85).

**Per-case CSV** (`ValidationCaseJsonHook.on_validation_end`, `hooks.py` l.502–536; YAML `output_dir:
val_case_metrics, write_csv: true, write_json: false`): `<run>/val_case_metrics/iter_<iteration:06d>/dice.csv` with
columns `case_id, split, dataset, class_index, dice, valid` — one row per (case, class) for both splits, written at
every validation (records built in `inferencer.py` l.449–478).

**TotalSegmentator special case**: training validates on `val` only; after the final checkpoint exists,
`submit_batched.sh` `test_eval` runs `reinfer_dice.py --yaml eval/<stem>.yaml --exp <run> --scratch ... --force
--tol 5e-3` (l.216), which rebuilds the environment from the eval YAML, resumes from
`checkpoints/best/best_val_metric_val_dice.pt` (trainable-only, non-strict; `engine.py` l.287–295), checks that the
loaded iteration equals the recorded best iteration (`reinfer_dice.py` l.73–76), runs `trainer.validate` on val + test,
requires `|recomputed val_dice − best_value| ≤ tol` (l.87–89) and copies `dice.csv` into
`<run>/val_case_metrics/iter_<best>/dice.csv` (l.98–99); `test_eval_done.txt` marks completion.

**`reinfer_dice.py`'s other role** (docstring l.4–10; `make_reinfer_jobs.py`): for runs that finished before the
per-case hook existed, recompute `dice.csv` at the selected checkpoint with default `--tol 2e-3`; `--no-copy`
compares with an existing file.

---

## 7. DISCREPANCIES between draft (`main.tex`) and code

1. DISCREPANCY: draft l.85 "Official validation sets are split into a selection and a held-out half" vs
   `build_manifests.py`: only AMOS22 uses the official validation set (`imagesVa`, l.75–78). KiTS23 and MSD Pancreas
   are a single shuffled 70/15/15 split of all labelled cases (l.86–88, 131–133) and LiTS a 91/20/20 split of the 131
   training volumes (l.130); none of these three has an official labelled validation set in the code path.
2. DISCREPANCY: draft l.85 "patients never span splits" vs code: no patient identifiers are read anywhere
   (`build_manifests.py`; `meta.csv` has no patient column); the only check is unique `case_id` per split
   (`ct_segmentation.py` l.60–62). Unverifiable from code for AMOS22 val/test halves, LiTS, MSD and TotalSegmentator.
3. DISCREPANCY: draft l.241 (Adapters) "DoRA ... the magnitude m ... trained ... without weight decay like every
   one-dimensional parameter" vs `dora.py` l.101–103 (`magnitude` has shape `(out, 1)`, ndim 2) and
   `optim_groups.py` l.16–22 (`_is_no_decay_name` exempts only `.bias`, `ndim <= 1`, or names containing
   norm/bn/ln) → the DoRA magnitude receives weight decay 1e-4.
4. DISCREPANCY: draft l.241 "LoRA ... applied ... to the query and key/value projections of every SPECTRE block and
   of its feature combiner (classification)" and Fig. armsseg caption l.202 "segmentation runs only SPECTRE's
   per-crop ViT-L, so its counts exclude the feature combiner" vs `backbones.py` l.147–149 (`module` = whole
   `SpectreImageFeatureExtractor`, which includes `feature_combiner`) and `seg_probe.py` l.195–208 (no path pattern
   for `lora`/`dora`; `fullft` unfreezes `module.parameters()`): in segmentation the combiner's `q`/`kv` layers are
   also wrapped/unfrozen. They receive no gradient (not in `features()`), so results are unaffected, but the logged
   trainable counts and trainable-only checkpoints include them. The paper's exclusion is a post-hoc accounting
   choice not reflected in the code's own count.
5. DISCREPANCY: draft l.73 (Backbones) VoCo "four stages of two shifted-window blocks ... 96 to 768 channels" and
   "the outputs of the four stages (strides 4 to 32) enter the decoder" vs `backbones.py` l.196–197: the stage
   outputs tapped have 192, 384, 768 and 1536 channels (after each stage's patch merging; MONAI `BasicLayer`
   applies `downsample` after its blocks, `swin_unetr.py` l.1050–1052), and Lin reads the 1536-d stage-4 output
   (`embed_dim = 96 * 2**4`, l.191). "96 to 768" describes the block widths, not the tapped outputs.
6. DISCREPANCY (omission): draft l.247 says SPECTRE taps are taken "after the final normalization" (TAP-CT not
   stated); code applies the final LayerNorm to every TAP-CT tap too (`get_intermediate_layers(..., norm=True)`,
   `backbones.py` l.116) and a parameter-free channel LayerNorm to every VoCo stage output (`normalize=True`,
   l.208; MONAI `proj_out`).
7. DISCREPANCY (omission): draft l.243 states flips and zoom but not their probabilities; code: flip with probability
   0.5 (`prob 0.5 × axis_prob 1.0`), zoom with probability 0.5; zoom-out pads the image with its minimum and the mask
   with 0 (`spatial.py` l.864–866). The classification protocol paragraph (l.211) states 0.5 for both.
8. DISCREPANCY (omission): draft Table `tab:totalseg` (l.423–449) does not state how overlapping source masks are
   resolved; `totalseg_labelmaps.py` l.99–101 assigns an overlapping voxel to the smaller structure (largest painted
   first), ~0.1–0.7 % of foreground (docstring l.23–27).
9. DISCREPANCY (omission): draft l.248 "a structure enters a case's Dice when the reference or the prediction
   contains it" is correct, but the stored value for an invalid (empty/empty) pair is `dice = 0, valid = 0` in
   `dice.csv` (`seg_probe.py` l.305; `hooks.py` l.533), not NaN; any downstream mean must filter `valid == 1`
   (`reinfer_dice.py` l.100 does).
10. DISCREPANCY (omission): draft l.82 "Every 1000 steps both splits are evaluated" and l.211 "a checkpoint ... every
    500 steps" are consistent with `VAL_EVERY_STEPS = 1000`, `SAVE_EVERY_STEPS = 500`; but compute-matched runs
    validate every 200 steps (draft l.361 states this) and the two-stage parents keep an extra protected checkpoint
    (`keep_iterations`), which the draft does not mention.
11. DISCREPANCY (omission): draft l.76 says Dec→LoRA "warm-starts all-block LoRA" (correct: `lora: "all"`), while the
    classification two-stage methods warm-start late-block LoRA (l.75); fine, but Table `tab:optimseg` caption
    l.254 "Dec→LoRA warm-starts the decoder from step 4000 of the Dec run, with fresh adapters" omits that the
    warm start requires the exact periodic file `model_8000.pt` (iteration = step×2) kept by `keep_iterations`.
12. DISCREPANCY: draft l.76 "(2.4–2.6 M parameters ...)" for Dec vs derived counts 2.42 M (VoCo), 2.45 M (TAP-CT),
    2.62 M (SPECTRE). Within rounding, but SPECTRE's is 2.62 M, not ≤2.6 M. [TODO: confirm from a run log's
    `SegProbeModel [...] trainable` line.]
13. DISCREPANCY (omission): draft l.85 "TotalSegmentator (dataset version 3.0.0, 1939 CTs)" — code/paths call the
    folder `TotalSegmentator-v3` and `totalseg_classes.py` l.4 says "117 TotalSegmentator v2 structures";
    `meta.csv` has 1939 rows (1830 train, 109 test). Version naming inconsistent between code comments and draft.
14. DISCREPANCY (omission): draft l.86 "TotalSegmentator per-task models use Lin, Dec, LoRA and Full FT only" is
    correct; the draft does not mention that VoCo has no `joint_lm/dec_then_lora`, no seed repeats, no
    compute-matched and no scratch runs (config tree on disk), while l.85 says VoCo is trained "with every method and
    mode of the other backbones".
15. DISCREPANCY (omission): draft Eq. (l.244–247) matches the code, but the draft does not state that the per-crop BCE
    is the mean over voxels of the crop *including* the padded region when a volume is smaller than the crop
    (`_pad_to`), nor that `seg_mask` is identically 1 (joint loss "covers every structure in every case", l.248, is
    therefore correct).
16. DISCREPANCY (omission): draft l.248 "(TotalSegmentator) ... tolerance 0.005" matches `--tol 5e-3`; the
    recomputation for older runs (l.249) uses the script default `--tol 2e-3` (`reinfer_dice.py` l.45) unless
    overridden [TODO: check the actual `make_reinfer_jobs.py --extra` used].
17. DISCREPANCY: draft l.68 Fig. 1 caption "a training crop is 12 whole slices ... centered in depth on a foreground
    voxel" — code centres the crop on the foreground voxel in all three axes and then clips the start so the crop fits
    (`_random_crop` l.313–314); for TAP-CT with in-plane 224 = crop size, H/W start is always 0, so "in depth" is
    the effective behaviour only because the in-plane crop equals the volume.

---

## 8. Unverified items

- [TODO] Patient-level disjointness of splits (no patient ids in code or manifests).
- [TODO] AdamW betas/eps defaults in `R/aifofomo/optimizers/` (draft says (0.9, 0.999), 1e-8).
- [TODO] Exact logged trainable-parameter counts per backbone/method (derived here, not read from run logs).
- [TODO] Whether `make_reinfer_jobs.py` was run with a non-default `--tol` for the pre-hook runs.
- [TODO] Why `voco/joint_lm/dec_then_lora`, voco seed repeats and voco `pertask_cm` are absent from the config tree
  (the generator would produce them; they may have been deleted or never generated).
- [TODO] KiTS23 label semantics (1 kidney, 2 tumor, 3 cyst) are taken from the code comment (`generate.py` l.119);
  `DATASET.txt` on disk only gives the source repo/commit.
- [TODO] TotalSegmentator "v3"/"3.0.0" release identity vs the code comment "v2 structures".
- [TODO] `InfiniteSampler` seeding of the training order (seed passed from the environment seed?) — not read.
