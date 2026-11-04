# Code map: classification path of `aifofomo` (fm_adapt suite)

Audit date 2026-10-31. Source of truth: the code on disk in
`/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo` (HEAD `28fc94b8` plus uncommitted
changes; the uncommitted changes add `mlp_finetune`, `backbone_layer_decay` and the
`min(end_lr, base_lr)` scheduler guard, see §3.11). All paths below are relative to that
repository unless absolute. `generate.py` = `examples/fm_probes/fm_adapt/generate.py`.

Verification of the generated YAMLs: re-running `generate.py` and `generate.py --revision`
into a scratch directory with the site `paths.env` reproduces every one of the 1354 YAMLs in
the repository byte for byte (`diff -rq`: 0 files differ). The regenerated tree contains 39
extra YAMLs that the repository does not: VoCo revision/seed-repeat configs
(`fdg|rsna2022|rsna2023/voco/{joint_s43,joint_s44,joint/fullft_*,mlpft,rslora_lowlr}`), i.e.
the VoCo revision runs were never written to the repository (§3.11).

---

## 1. Data

### 1.1 Which dataset class each config uses

Registered names → classes (`aifofomo/data/datasets_config.py` lines 387–761;
implementations in `aifofomo/data/datasets/ct_classification.py`):

| config `name` | class | cohort |
|---|---|---|
| `amos_cls` | `AMOSClassificationDataset` (l.1416) ← `CTBinaryClassificationDataset` (l.597) | CVPR26 AMOS |
| `covid_ct_cls` | `COVIDCTClassificationDataset` (l.1523) | CVPR26 COVID-CT |
| `luna25_cls` | `LUNA25ClassificationDataset` (l.1617) | CVPR26 LUNA25 |
| `fdg_autopet_cls` | `FDGAutoPETClassificationDataset` (l.2385) | fdg |
| `rsna2022_multilabel` | `RSNA2022MultiLabelDataset` (l.1989) | rsna2022 |
| `rsna2023_multilabel` | `RSNA2023MultiLabelDataset` (l.2185) | rsna2023 |
| `rsna2019_ich` | `RSNA2019ICHMultiLabelDataset` (l.2574) | rsna2019_ich |
| `breastdivider_cls` | `BreastDividerClsDataset` (`datasets/breastdivider.py`) | breastdivider (Pillar-0 BreastMRI) |

### 1.2 Manifests and labels per cohort

Cohort → manifest, as wired by `generate.py` `DATASETS` (l.279–326) and the site file
`examples/fm_probes/fm_adapt/paths.env`:

**amos_covid_luna (CVPR26 cohort)** — `kind: amos`. Three dataset entries per split
(`generate.py::_amos_entries`, l.433–468): `amos_cls` at `FM_DATA_AMOS`, `covid_ct_cls` at
`FM_DATA_COVID`, `luna25_cls` at `FM_DATA_LUNA`. Manifest = one CSV per label,
`<root>/labels/<target>.csv` with columns `case_id,<target>,split` (`ct_classification.py`
l.901–940 `_load_entries_csv`; `case_id` must end in `.nii.gz`). Labels (`CANONICAL` in
`generate.py` l.126–144, identical to `CANONICAL_TASKS` in `ct_classification.py`), 17 in
canonical order:
`amos:adrenal_hyperplasia, amos:ascites, amos:atherosclerosis, amos:cholecystitis,
amos:colorectal_cancer, amos:fatty_liver, amos:gallstone, amos:hydronephrosis,
amos:kidney_stone, amos:liver_calcifications, amos:liver_cyst, amos:liver_lesion,
amos:lymphadenopathy, amos:renal_cyst, amos:splenomegaly, covid_ct:covid,
luna25:lung_nodule_malignancy`. The 15 AMOS CSVs exist on disk
(`/projects/cvpr2026_challenge_fomo4ct/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/labels/`),
plus `COVID-CT/labels/covid.csv` and `LUNA25/labels/lung_nodule_malignancy.csv`.
Manifest counts (from the CSVs, `split` column): AMOS `ascites.csv` train 1107 / val 68 (per-label
CSVs differ in row count; 1463 distinct AMOS case_ids over all 15 CSVs); COVID train 308 / val 64;
LUNA train 1000 / val 200. The official CSVs contain only `train`/`val` (docstring l.619).

**fdg** — `fdg_autopet_cls`, `labels_csv: cls_labels.csv` under `FM_DATA_FDG`
(`/mnt/kronos/data_public/FDG-PET-CT-Lesions/cls_labels.csv`). Columns on disk:
`case_id,study_dir,ct_path,diagnosis,NEGATIVE,MELANOMA,LUNG_CANCER,LYMPHOMA,split`. Labels
(`FDG_AUTOPET_LABEL_NAMES`, l.2383): `NEGATIVE, MELANOMA, LUNG_CANCER, LYMPHOMA`. The loaded
volume is `ct_path` (CT only; l.2470–2476). Rows are studies, `case_id` is the patient
(1014 rows, 900 distinct `case_id`); split counts train 776 / val_train 116 / val_holdout 122.

**rsna2022** — `rsna2022_multilabel`, `labels.csv` under `FM_DATA_RSNA2022`. Columns:
`StudyInstanceUID,patient_overall,C1..C7,split`. Labels (`RSNA2022_LABEL_NAMES`, l.1977–1986):
`patient_overall, C1, C2, C3, C4, C5, C6, C7`. 2019 rows = 2019 distinct UIDs; split counts
train 1615 / val_train 202 / val_holdout 202.

**rsna2023** — `rsna2023_multilabel`, `labels.csv` under `FM_DATA_RSNA2023`. Columns include
`patient_id,series_id,any_injury,bowel_injury,extravasation_injury,kidney_injury,liver_injury,
spleen_injury,incomplete_organ,aortic_hu,split`; only `RSNA2023_LABEL_NAMES` (l.2178–2183) are
read: `any_injury, kidney_injury, liver_injury, spleen_injury` ("Bowel / extravasation are
omitted", comment l.2175–2177). Case key `"{patient_id}/{series_id}"`, one series per
patient (3147 rows = 3147 patients); split counts train 2517 / val_train 315 / val_holdout 315.

**rsna2019_ich** — `rsna2019_ich`, `labels_csv: patient_table_stratified_5k.csv` under
`FM_DATA_RSNA2019_ICH`. Columns: `patient_id,study_uid,series_uid,series_id,case_id,file_path,
n_slices,any,epidural,intraparenchymal,intraventricular,subarachnoid,subdural,split`. Labels
(`RSNA2019_ICH_LABEL_NAMES`, l.2566–2573): `any, epidural, intraparenchymal, intraventricular,
subarachnoid, subdural`. Split counts train 5000 / val 3269 / test 3264 (11533 rows,
10349 patients, 0 patients span splits). The 5k train subset is drawn by
`examples/fm_probes/scripts/build_ich_stratified_table.py::build` (l.25–63): all epidural
positives kept, other labels down-sampled proportionally (seed 42), val/test kept in full.

**breastdivider (note)** — `breastdivider_cls`, `patient_table.csv`
(`/mnt/kronos/data_public/BreastDivider/patient_table.csv`), MRI not CT, backbone Pillar-0
BreastMRI only (`DATASETS["breastdivider"]["backbones"] == ["pillar0"]`), labels
`lesion_presence, lesion_malignant` read from CSV columns `y_presence`/`y_malignant`
(`BREAST_LABEL_COLUMNS`, `generate.py` l.256), `require_file_exists: true`; split counts train
291 / val 63 / test 62. It is excluded from the revision runs (`write_revision`: `if dataset ==
"breastdivider": continue`, l.882) and from seed repeats. Geometry 192×384×384 at 1 mm with
`high_contrast` p20–p99 windowing in the model (`breastdivider.py` docstring l.1–12,
`pillar0_cls.py::_mr_window`). Configs exist under `fm_adapt/breastdivider/pillar0/{joint,pertask}`.

### 1.3 Splits: train / selection ("val") / held-out ("test")

* **CVPR26 cohort** — `CTBinaryClassificationDataset._select_split_rows` (l.942–971): `train`
  = rows with `split==train`; config `val` (alias `val_train`) and `test` (alias `val_holdout`)
  carve the official `val` rows 50/50. All fm_adapt configs set `val_split_mode: per_case`
  (`generate.py` l.456), so `_official_val_case_rank` (l.973–987) ranks the sorted union of
  official-val `case_id`s over every CSV in `labels/`; even ranks → `val`, odd → `test`:
  ```python
  rank = self._official_val_case_rank(); parity = 0 if take_even else 1
  keep = df["case_id"].astype(str).map(lambda c: rank.get(c, 0) % 2 == parity)
  ```
  One assignment per case shared by every label, so per-task and joint runs see identical
  selection/held-out cases (comment l.454–455).
* **fdg / rsna2022 / rsna2023** — `resolve_table_split` (l.178–199): config `val` → stored
  `val_train`, `test` → `val_holdout`. These splits are NOT halves of an official validation
  set: they are an 80/10/10 stratified split of the released *training* data with seed 42:
  `aifofomo/benchmarks/preprocess_rsna_nifti.py` l.124–143 (RSNA2022, stratified on
  `patient_overall`, `rng = np.random.default_rng(42)`, 10 % / 10 %),
  `aifofomo/benchmarks/preprocess_rsna2023_labels.py` (docstring l.4–7: "picks one series per
  patient ... writes an 80/10/10 stratified split ... on any_injury (seed 42)"). The FDG
  `cls_labels.csv` builder is outside this repository ([TODO: original FDG split script not in
  repo]); `examples/fm_probes/scripts/build_gaia_adapters.py::build_fdg` (l.63–88) then moves
  every patient whose studies span splits to the most held-out split ("Patient-level split: a
  patient (case_id) with studies in several splits is moved entirely").
* **rsna2019_ich** — stored `train`/`val`/`test` used directly (`resolve_table_split` returns
  `requested` when present). The full `patient_table.csv` comes from
  `pillar0/rsna2019_ich/scripts/build_cls_tables.py` (external, docstring l.2578–2581) [TODO:
  not in repo].

**Patient-level guarantee.** Enforced in code only for FDG (adapter script above). Verified
from the manifests: FDG 0 patients span splits (after the fix), RSNA2023 one series per patient,
RSNA2019 0 of 10349 patients span splits. RSNA2022 is split per `StudyInstanceUID` (one row per
study; the code does not group by patient) [TODO: confirm one study per patient in the RSNA
2022 release]. AMOS/COVID/LUNA are split per `case_id` (file); no patient id exists in the
manifests [TODO: patient identity for CVPR26 cohort cannot be checked from code].

Missing volumes: `_check_missing_volumes` (l.205–220) raises if >5 % of manifest rows have no
file, else warns.

### 1.4 Label encoding, masking

* Every sample carries `label_vec` (float) and `label_mask` (bool) of equal length
  (`__getitem__`, l.1331–1414 for CVPR26; l.2136, 2344, 2526, 2742 for the table cohorts).
* **CVPR26 joint**: one `amos_cls` entry with `targets: [15 AMOS labels]` (`generate.py`
  l.452) → one row per AMOS case carrying all 15 labels; `label_vec` is the canonical 17-wide
  layout, `label_mask[t]=True` only where the case appears in that label's CSV with a 0/1 value
  (l.1356–1362: `for task_index, y in self._entry_labels.get(case_id, ...).items(): if y in (0, 1)`).
  COVID and LUNA rows set one slot each (indices 15, 16). Unknown labels are masked, not zero-filled.
* **CVPR26 per-task**: `single_task_label_vec: true` (`generate.py` l.462) → `label_vec`,
  `label_mask` of length 1 (l.1350–1355).
* **Table cohorts**: joint → `label_vec = full label row`, `label_mask = ones(len(label_names))`
  (all labels always known); per-task (`target: <label>`) → length-1 vector with
  `y = int(full[ti] > 0.5)`.
* `pos_weight` per label is precomputed by `generate.py::_pos_weight` (l.645–667) as
  `clip(neg/pos, 0.05, 20)` over the manifest `train` rows of that label (AMOS: that label's
  CSV; tables: `labels_csv`), written as `pos_weight_list` in task order
  (`_pos_weight_list`, l.670–677). Example (SPECTRE AMOS joint): `[10.000, 9.960, 12.667, 5.000,
  12.500, 5.000, 3.000, 3.000, 2.000, 5.000, 1.000, 2.000, 8.973, 1.000, 14.592, 4.133, 9.000]`.
* No minority oversampling: `oversample_minority_class` defaults False and is not set in any
  config (grep of generated YAMLs).

### 1.5 Volume loading and per-backbone geometry

Shared helpers (`ct_classification.py`): `read_nifti_dhw` (`datasets/nifti_io.py` l.31–73:
nibabel load, optional reorientation to axcodes via `nibabel.orientations`, then
`(X,Y,Z)→(D,H,W)=(Z,Y,X)` transpose); `_resize_volume_1dhw` (l.342–367, trilinear, optional
`keep_native_depth`); `_spacing_resample_nifti_array` (l.458–549, SimpleITK `DICOMOrient` +
`ResampleImageFilter`, linear, default pixel value = scan minimum; for `crop_mode: center` it
resamples straight into a centred ROI of `target_size`; nibabel/trilinear fallback for
non-orthonormal direction cosines, `warn_sitk_fallback_once`); `_crop_or_pad_dhw` (l.393–455;
`center`/`random`/`pad`, pad value = volume minimum).

Per backbone, as emitted by `generate.py::_dataset_entry` (l.391–423), `_size` (l.335–348),
`_orientation` (l.371–388), `_transforms` (l.356–368):

| backbone | orientation | spacing / size | crop | HU window (where applied) |
|---|---|---|---|---|
| TAP-CT | `LPS` | native depth, in-plane trilinear resize to 224×224 (`keep_native_depth: true`, `target_size [12,224,224]`, D ignored) | none | in model: clamp [-1008, 822], z-score mean −86.8086 / std 322.6347 (`tapct_cls.py` l.33–34, `_tapct_normalize` l.345) |
| SPECTRE | `RAS` | `target_spacing [0.5,0.5,1.0]` mm (x,y,z), `target_size [256,512,512]` (D,H,W) | `center` (resample straight into centred ROI, pad with scan min) | dataset transform `ScaleIntensityRange a_min −1000, a_max 1000 → [0,1], clip` |
| Pillar-0 (ICH) | `LPI` | `target_spacing [0.5,0.5,1.25]`, `target_size [128,256,256]` | `center` | in model: 11 channels = 10 centre/width windows + per-volume min–max (`pillar0_cls.py` l.92–132 `_ct_window_all`; `ct_window_type: all`) |
| Pillar-0 (breast) | none | 192×384×384 at 1 mm (dataset class) | centre | in model: `high_contrast` p20–p99 |
| VoCo | `RAS` | `target_spacing [1.5,1.5,1.5]`, `target_size [96,96,96]`, `crop_mode: pad` (never crops; pads short axes) | pad | dataset transform `ScaleIntensityRange −175..250 → [0,1], clip` |

`ScaleIntensityRange` (`data/transforms/modules/intensity.py` l.152–192): `(x−a_min)/(a_max−a_min)`,
then to `[b_min,b_max]`, then clamp.

The in-model tiling (windows/crops, overlap, stride) is described in §2 per backbone. Summary:
TAP-CT 12-slice windows with `sw_overlap 0.75` → stride `round(12·0.25)=3` slices plus a final
window ending at the last slice (`_depth_window_starts` l.410–426); SPECTRE 4×4×4 grid of
128×128×64 crops (no overlap; `spectre.windowing.window_scan`); Pillar-0 whole volume in one
forward; VoCo 96³ windows with `sw_overlap 0.0`, last window shifted to the border
(`voco_cls.py::window_starts` l.87–97).

### 1.6 Augmentation (train split only)

`generate.py::_transforms` (l.356–368), appended after the intensity window:
`RandomFlip3D keys [data, fg_mask], prob 0.5, axis_prob 1.0, axes [2]` (left–right flip only;
axis 2 of (z,y,x) is x; `data/transforms/modules/spatial.py` l.659–721) and
`RandomZoom3D prob 0.5, scale [0.85, 1.15], mask_keys [fg_mask]` (l.769–879: zoom-in = random
sub-crop of relative size s resized back trilinearly; zoom-out = pad with the volume minimum then
resize back). Both apply to all three axes including depth (`cd = int(D*s)`), so TAP-CT native
depth is also resampled under zoom. Validation/test entries carry no augmentation (`train=False`).
`fg_mask` is emitted empty (`fg_mask_channels: 0`, `use_fg_masks: false`; `generate.py` l.459–460).

### 1.7 Sampling

Training uses an infinite stream: `engine/engine.py::create_train_dataloader` (l.505–527) →
`setup_dataloader` l.370: `WeightedInfiniteSampler(dataset, shuffle=True, ..., weights)`.
`data/samplers.py`: per sub-dataset an `InfiniteSampler` that yields a fresh `randperm` each
pass (l.80–97, generator seeded `seed + rank`, `seed = torch.initial_seed()`), and the dataset
index is drawn per sample by `random.Random(seed+rank).choices(..., weights)` (l.227). Weights
come from `Data.resolve_infinite_sampler_weights` (`data/data.py` l.220–249): the explicit
`infinite_sampler_weights` when set, else uniform. Only the CVPR26 joint configs set them
(`generate.py` l.714–718, both `joint` and `joint_lm`): `[0.3846, 0.3077, 0.3077]` =
10/26 : 8/26 : 8/26 for AMOS : COVID : LUNA (`AMOS_COHORT_WEIGHTS`, l.252). All other runs have a
single training dataset (uniform shuffled cycling). Batch size 1, no accumulation (`BATCH`,
l.271–276; `_accum` l.328–331 returns 1 for every method since frozen probes are (1,1) too).
Dataloader: `num_workers 8` (Pillar-0: 4), `pin_memory false` for SPECTRE (comment l.707–710),
true otherwise.

---

## 2. Backbones

All four wrappers expose the same interface: `forward(data) → {"logits", "probabilities" =
sigmoid(logits), "cls_embedding", ["crop_mil_attention"]}`, the same `training_mode` literal
`frozen_backbone | lora | dora | attention_finetune | mlp_finetune | full_finetune`, the same
head types `linear | attention`, the same LoRA/DoRA helpers (`models/networks/voco/lora.py`,
`dora.py`), the same `AttentiveProbe` (`voco/attentive_probe.py`) and `GatedAttentionMILPool`
(`voco/abmil.py`), the same optimizer-group builders (`models/networks/optim_groups.py`) and the
same warm-start resolver (`voco/probe_init.py::resolve_checkpoint_path`). In
`frozen_backbone` mode the backbone is kept in `eval()` (each wrapper's `train()` override) and
run under `no_grad`; in every other mode it runs in train mode with gradients.

### 2.1 TAP-CT-B-3D — `models/networks/tapct/tapct_cls.py::TapCTClsModel`

* **Type: plain ViT** (single-scale, no hierarchy). Checkpoint `pretrained_name:
  fomofo/tap-ct-b-3d` via `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`
  (l.149). HF `config.json`: `img_size [12,224,224], patch_size [4,8,8], in_chans 1,
  num_register_tokens 4, init_values 1e-5`; remote `vision_transformer_3d.py::vit_3d_base`:
  `embed_dim=768, depth=12, num_heads=12`. Tokens per window: 3·28·28 = 2352 patches + CLS +
  4 registers (comment l.42: "2357 tokens"). Attention uses a fused `qkv` Linear (remote
  `attention.py` l.107) and `proj`; MLP `fc1`/`fc2` (remote `mlp.py` l.78–80). Every attention
  block is re-routed through `F.scaled_dot_product_attention` (`patch_tapct_attention_to_sdpa`,
  l.57–66; same function, O(N) memory).
* **Input**: `_ensure_min_roi` (l.355–377): optional z-score, in-plane resize to 224 if needed,
  depth padded with normalised air (−1008 HU) to ≥12 and to a multiple of 4.
* **Tiling** (`feature_mode: multi_crop_cls`, `cover_full_volume: true`, `roi_size [12,224,224]`,
  `sw_overlap 0.75`, `sw_batch_size 16`, `max_train_windows 32`): `_depth_window_starts`
  (l.410–426) → starts `0,3,6,…` plus `depth−12`; training keeps a random sorted subset of ≤32
  starts (l.436–439); `_multi_crop_cls_stack` (l.428–463) encodes windows 16 at a time and
  returns the per-window CLS bag `(B, K, 768)` from `pooler_output` (= class token of the last
  block after the final norm; remote `modeling_tapct.py` l.137/147, `get_intermediate_layers(norm=True)`).
* **Feature → head** (`_probe_from_cls_bag`, l.465–505): `linear` → mean over K CLS tokens →
  `Dropout(0.2)` → `Linear(768, num_tasks)`; `attention` → `AttentiveProbe` over all K CLS tokens
  (`inst.transpose(1,2)[...,None,None]`); with `multi_crop_abmil` → `GatedAttentionMILPool(768,128)`
  over the K tokens, then the probe on the single pooled token (`emb[:, :, None, None, None]`).
* **LoRA targets**: `lora_target_names: [qkv]`, no include pattern for all-block LoRA
  (12 blocks), late = `lora_include_pattern: 'blocks\.(8|9|10|11)\.'` (0-indexed → blocks 9–12
  of 12). Logged counts (run logs under `/projects/direct/temp/fm_probe/v2`): backbone
  87,083,520 params; LoRA all blocks 589,824 trainable backbone params (= 12 × 16 × (768+2304));
  late 196,608; attentive probe (17 tasks) 14,191,121; `fullft` 101,274,641 trainable of
  101,274,641.
* Param groups: `get_optimizer_param_groups` (l.187–211): backbone prefixes `("backbone.", "_vit.")`,
  head prefixes `("probe.", "crop_abmil.")`.

### 2.2 SPECTRE-Large — `models/networks/spectre/spectre_cls.py::SpectreClsModel`

* **Type: plain ViT-L per crop + a second plain ViT ("feature combiner") over the crop grid**
  (two single-scale transformers; not hierarchical). Checkpoint `pretrained_name:
  cclaess/SPECTRE-Large` (HF, `trust_remote_code`), HF config: `backbone_name
  vit_large_patch16_128`, `feature_combiner_name feat_vit_large`, `crop_size [128,128,64]`,
  `hu_range [-1000,1000]`, `orientation RAS`, rope position embeddings. From the installed
  `spectre` package (`models/vision_transformer.py` l.711–731): `img_size (128,128,64),
  patch_size (16,16,8), embed_dim 1080, depth 24, num_heads 12, mlp_ratio 4` → 8·8·8 = 512 patch
  tokens + CLS per crop; combiner (`models/vision_transformer_features.py` l.404–423):
  `embed_dim 1080, depth 4, num_heads 12`. Attention has separate `q` and `kv` Linears
  (`spectre/models/layers/attention.py` l.41–47) and `proj`; MLP is timm `Mlp` (`fc1`/`fc2`).
* **Tiling** (`_tile`, l.290–294): `spectre.windowing.window_scan(volume, crop_size,
  scale_intensity=False, pad_value=0.0)` — pad short axes, centre crop to a multiple of the crop
  size, grid patch → for 512×512×256 a (4,4,4) grid of 64 crops. Input is permuted
  `(B,1,D,H,W)→(B,1,X,Y,Z)` (`input_axes: xyz`, l.370). Training: `_train_subgrid` (l.296–313)
  draws a random contiguous sub-grid with ≤`max_train_crops: 8` crops → (2,2,2); eval uses all 64.
* **Features** (`_encode_crops` l.315–324, `_scan_tokens` l.326–337): backbone in chunks of
  `max_crops_per_forward: 16` → `_pool_crop_tokens` = `[CLS ‖ mean(patch tokens)]` (2160-d,
  `spectre/model.py` l.303–314) → `feature_combiner(pooled, grid)` → `(N+1, 1080)` tokens
  (combiner CLS + one token per crop, after the combiner's final norm; `forward_features`
  l.284–297 of `vision_transformer_features.py`).
* **Feature → head** (`_probe`, l.339–365): `linear` → combiner CLS `tokens[:,0]` → Dropout →
  Linear; `attention` → `AttentiveProbe` over the N+1 combiner tokens; `babmil` → ABMIL over the
  N crop tokens (`tokens[:,1:]`), then the probe on the pooled token (`emb[:, None, :]`).
* **LoRA targets**: `lora_target_names: [q, kv]`; all-block pattern
  `'(backbone\.blocks\.|feature_combiner\.)'` (24 backbone blocks + 4 combiner blocks); late
  `'(backbone\.blocks\.(1[6-9]|2[0-3])\.|feature_combiner\.)'` (backbone blocks 17–24, 1-indexed,
  plus the combiner). Param groups: backbone prefix `("spectre.",)`. [TODO: SPECTRE parameter
  counts — no SPECTRE run log with the `SpectreClsModel [mode=` line was found locally.]

### 2.3 Pillar-0 (HeadCT; BreastMRI for breastdivider) — `models/networks/pillar0/pillar0_cls.py::Pillar0ClsModel`

* **Type: convolutional stem + three-stage multi-scale (hierarchical) window transformer**
  (`CLIPMultimodalAtlas`; not a plain ViT). Weights: local directory
  `FM_PILLAR0_HEAD_WEIGHTS=/projects/cvpr2026_challenge_fomo4ct/pretrained/Pillar0-HeadCT`
  (`trust_remote_code`). Its `config.json`: `embed_dim 1152`; `vision_cfg`: `in_chans 11,
  image_size [256,256,128] (H,W,D), patch_size [8,8,4], merge_ratio [4,4,4], local2global
  [4,4,4], width 96, num_features 192`; `model.stages [2, 2, 8]` (three `AtlasStage`s of 2, 2 and
  8 `MultiScaleAttentionBlock`s), `atlas_args: dim 384, num_heads 6, mlp_ratio 4, num_scales 3,
  class_token False, global_pool avg, drop_path_rate 0.3`. Remote code: `BrainCTMultiConvEmbed3D`
  stem (Conv3d + BatchNorm3d, `atlas_encoders.py`), `multimodal_msa.py`: each
  `MultiScaleAttentionBlock.blocks[scale]` is a `CrossWindowBlock` whose `attn =
  CrossWindowAttention` has `q = Linear(dim,dim)`, `kv = Linear(dim, 2·dim)`, `proj` (l.30–35);
  MLP is timm `Mlp` (`fc1`/`fc2`). Module path example:
  `visual.atlas_models.<stage>.blocks.<i>.blocks.<scale>.attn.q`.
* **Input** (`_prep_input`, l.477–513): `ct_window_type: all` → `_ct_window_all` (l.115–132):
  11 channels from `_CT_ANATOMICAL_WINDOWS` (lung (−600,1500), mediastinum (50,400), abdomen
  (40,400), liver (80,150), bone (400,1800), brain (40,80), subdural (75,215), stroke (40,40),
  temporal_bone (600,2800), soft_tissue (50,350); centre/width → [0,1]) + per-volume min–max;
  `apply_ct_normalization` False by default. Whole 128×256×256 volume in one forward (no windows).
* **Features**: `linear` head uses `feature_mode: pooled` → `backbone.extract_vision_feats` =
  `encode_image(normalize=True)` (remote `modeling_clip_mmatlas.py` l.48), a 1152-d L2-normalised
  embedding (`_extract_pooled`, l.515–540). `attention` head uses `feature_mode: spatial` →
  `_extract_spatial` (l.542–656): runs the three Atlas stages, collects one token set per scale,
  average-pools any scale with more than `max_spatial_tokens: 4096` tokens on its own grid
  (stem grid 32³ → 16³), places each scale's channels in its own block of the 1152-d space and
  returns `(B, 1152, N, 1, 1)` plus the pooled `(B,1152)` = `F.normalize(cat(maxpool per scale))`.
  `babmil` → `_full_volume_token_bag` (l.669–682): ABMIL over the spatial tokens.
* **LoRA targets**: `lora_target_names: [q, kv]`, all = `'visual\.'` (all 3 stages, 36
  `CrossWindowBlock`s), late = `'visual\..*atlas_models\.2\.'` (third stage, 8 blocks). Logged
  counts (ICH joint): backbone 94,144,193; LoRA all 1,105,920 backbone trainable
  (= 12·3·(16·384+384·16 + 16·384+768·16)); `mlp_finetune` 42,536,448; attentive probe ≈ 31.9 M
  (total 126,035,015 for `fullft`, 100 % trainable).
* `train()` override (l.441–449): under `lora/dora/attention_finetune/mlp_finetune` the stem
  BatchNorm layers are kept in `eval()` ("batch size is 1, so train-mode BN would drift"); under
  `full_finetune` BN runs in train mode at batch size 1 (not guarded). [TODO: whether the Atlas
  `drop_path_rate 0.3` is active in train mode under LoRA/full FT — not checked in the remote code.]

### 2.4 VoCo-L — `models/networks/voco/voco_cls.py::VoCoClsModel`

* **Type: hierarchical Swin transformer** (MONAI `SwinUNETR(use_v2=True).swinViT`; shifted-window
  attention, patch merging between stages; no class token). Checkpoint
  `pretrained_path: FM_VOCO_WEIGHTS=/projects/cvpr2026_challenge_fomo4ct/pretrained/VoCo_L_SSL_head.pt`
  loaded by `build_voco_swin` (l.47–85): `SwinUNETR(in_channels=1, out_channels=1,
  feature_size=96, use_v2=True, use_checkpoint=grad_checkpointing).swinViT`, `swinViT.*` keys only,
  raises if any encoder weight is missing. MONAI defaults (`swin_unetr.py` l.70–73):
  `patch_size 2, depths (2,2,2,2), num_heads (3,6,12,24), window_size 7`; with `feature_size 96`
  the four stages run at 96/192/384/768 channels and emit, after merging, outputs of
  192/384/768/1536 channels at strides 4/8/16/32 (`SwinTransformer.forward`, l.1273–1293, returns
  `[x0_out … x4_out]` with `proj_out(normalize=True)`). Window attention has fused `qkv`
  (l.585) and `proj`; MLP `linear1`/`linear2`.
* **Windows** (`_windows`, l.328–337): `pad_to_roi` to ≥96³ with 0 (air after the [0,1] window),
  `window_starts` per axis with `sw_overlap 0.0` (last window shifted to the border), all
  corner combinations; training keeps a random sorted subset of ≤`max_train_windows: 16`.
  `windows_per_forward: 4`. `grad_checkpointing: true` in all VoCo configs (MONAI
  `use_checkpoint`; identical outputs, comment `generate.py` l.515–517).
* **Token** (`_encode_windows`, l.339–345): for each window, concat of the spatial means of stage
  outputs `token_stages: [2,3,4]` → 384+768+1536 = 2688-d (`self.embed_dim = sum(96·2^s)`,
  l.173), then `_RunningStandardize(2688)` (`voco_hier_attn.py` l.740–785: per-channel EMA mean
  and variance, momentum 0.01, updated on every training row, eval uses the running estimate;
  buffers are saved with the checkpoint and copied on warm start).
* **Feature → head** (`_probe`, l.355–365): `linear` → mean over window tokens → Dropout →
  `Linear(2688, num_tasks)`; `attention` → `AttentiveProbe(in=2688, hidden=probe_hidden_dim 768)`
  over all window tokens (input projection 2688→768); `babmil` → ABMIL over window tokens then the
  probe on the pooled token.
* **LoRA targets**: `[qkv]`; all blocks = no pattern (8 blocks); late = `'layers4\.'` (stage 4,
  2 blocks). [TODO: VoCo parameter counts — no VoCo run log found locally; paper's "74 M" unverified.]

### 2.5 Shared head modules

* `AttentiveProbe` (`voco/attentive_probe.py` l.84–196): `input_proj` (Identity when
  `in_channels == hidden_dim`), learnable CLS token (trunc-normal 0.02), `num_attn_layers=2`
  pre-norm blocks of `nn.MultiheadAttention(8 heads)` + GELU FFN (ratio 4), `attn_dropout 0,
  ffn_dropout 0`, final `LayerNorm`, `Dropout(head_dropout=0.2)`, `Linear(hidden, num_classes)`
  (classifier trunc-normal 0.02, zero bias). The blocks are full self-attention over
  `[CLS] + tokens`; the CLS token is read out (not a cross-attention-only query).
* `GatedAttentionMILPool` (`voco/abmil.py` l.33–105): `a = softmax(w^T(tanh(Vx) ⊙ σ(Ux)))`,
  `z = Σ a_i x_i`, `V,U: D→128` (`bag_attn_hidden: 128`), Xavier init.
* Linear head: `nn.Sequential(nn.Dropout(0.2), nn.Linear(embed_dim, num_tasks))` in every wrapper.

---

## 3. Methods

Source of every setting: `generate.py` `METHODS` (l.147–221), `REVISION_METHODS` (l.229–235),
`SEED_MODES`/`SEED_METHODS` (l.238–239), `render_config` (l.698–840). The YAML fields are
consumed by the wrappers' `_configure_backbone` and `get_optimizer_param_groups` and by the
trainer (§3.9).

### 3.1 Table of methods (identical across backbones unless noted)

| key | `training_mode` | head | adapter scope | optimizer `lr` / `wd` | steps | warm start |
|---|---|---|---|---|---|---|
| `lin` | `frozen_backbone` | `linear` | – | 1e-3 / 0.0 | 8000 | – |
| `attn` | `frozen_backbone` | `attention` | – | 1e-4 / 0.01 | 8000 | – |
| `lora` | `lora` | `attention` | all blocks (`"stage"`) | 1e-4 / 0.01 | 8000 | – |
| `lorag` | `lora` | `attention` | late blocks | 1e-4 / 0.01 | 8000 | – |
| `babmil` | `lora` | `attention` + `multi_crop_abmil: true`, `bag_attn_hidden 128` | all blocks | 1e-4 / 0.01 | 8000 | – |
| `dora` | `dora` | `attention` | all blocks | 1e-4 / 0.01 | 8000 | – |
| `rslora` | `lora` + `lora_rank_stabilized: true` | `attention` | all blocks | 1e-4 / 0.01 | 8000 | – |
| `attnft` | `attention_finetune` | `attention` | all attention modules (same match as all-block LoRA) | backbone 2e-5 / 0.05, head 1e-4 / 0.01 | 8000 | – |
| `fullft` | `full_finetune` | `attention` | everything | backbone 2e-5 / 0.05, head 1e-4 / 0.01 | 8000 | – |
| `lin_then_lorag` | `lora` | `linear` | late blocks | 1e-4 / 0.01 | 4000 (stage 2) | `lin` parent at step 4000 |
| `attn_then_lorag` | `lora` | `attention` | late blocks | 1e-4 / 0.01 | 4000 (stage 2) | `attn` parent at step 4000 |
| `fullft_lowlr` (revision) | `full_finetune` | `attention` | everything | backbone 5e-6 / 0.05, head 1e-4 / 0.01 | 8000 | – |
| `fullft_llrd` (revision) | `full_finetune` + `backbone_layer_decay 0.75` | `attention` | everything | backbone 2e-5·0.75^(L−l) / 0.05, head 1e-4 / 0.01 | 8000 | – |
| `fullft_accum` (revision) | `full_finetune` | `attention` | everything | as fullft; `gradient_accumulation_steps 8`, `gradient_clipping 1.0` | 1000 optimizer steps × 8 = 8000 cases | – |
| `mlpft` (revision) | `mlp_finetune`, targets `[fc1, fc2]` (VoCo: `[linear1, linear2]`) | `attention` | all MLP modules | backbone 2e-5 / 0.05, head 1e-4 / 0.01 | 8000 | – |
| `rslora_lowlr` (revision) | `lora` + rsLoRA | `attention` | all blocks | 2.5e-5 / 0.01 | 8000 | – |

All methods: `head_dropout: 0.2`; LoRA-type: `lora_r 16, lora_alpha 32.0, lora_dropout 0.1`.
The generated YAML's `optimizer.learning_rate/weight_decay` are the single-group values for
frozen/LoRA methods and the backbone values for `attnft/fullft/mlpft` (comment l.180).
`BACKBONE_METHOD_OPTIM` is empty (l.243) — no per-backbone overrides; the module docstring's
claim "VoCo lin: 1e-4 / 0.01" (l.15–16) is stale: VoCo `lin` uses 1e-3 / 0 like the others.

### 3.2 What is trainable

* `frozen_backbone`: backbone `requires_grad=False` and `eval()`; only `probe.*` (and
  `crop_abmil.*`, `token_norm` buffers for VoCo) train (`_configure_backbone` in each wrapper,
  e.g. `tapct_cls.py` l.264–337).
* `lora`/`dora`: backbone frozen, then `apply_lora_to_module` / `apply_dora_to_module`
  (`lora.py` l.119–186, `dora.py` l.146–180) replaces every `nn.Linear` whose attribute name is in
  `lora_target_names` and whose full path matches `lora_include_pattern` (if given) by
  `LoRALinear`/`DoRALinear`; the original weight and bias stay frozen; `lora_A`, `lora_B`
  (+ `magnitude` for DoRA) train, plus the head. Raises if no layer matched.
* `attention_finetune` / `mlp_finetune`: `unfreeze_attention_modules` (`lora.py` l.254–293) walks
  with the same matching rule and sets `requires_grad=True` on **every parameter of the parent
  module** of each matched Linear: for `qkv`/`q`/`kv` the parent is the attention module (so
  `proj` and all biases, and for VoCo the `relative_position_bias_table`, train too); for
  `fc1`/`fc2` (`linear1`/`linear2`) the parent is the MLP module.
* `full_finetune`: every backbone parameter `requires_grad=True` (for Pillar-0 this is the whole
  `CLIPMultimodalAtlas` module; for SPECTRE the HF wrapper incl. backbone and combiner).

### 3.3 LoRA / rsLoRA / DoRA definitions

`LoRALinear.forward` (`lora.py` l.96–99): `W0 x + scaling · B(A(drop(x)))` with
`scaling = alpha/r` (= 32/16 = 2.0) or, for `rank_stabilized`, `alpha/sqrt(r)` (= 8.0; l.81);
`A` Kaiming-uniform (`a=sqrt(5)`), `B` zeros (l.92–94); dropout before the low-rank branch only.
`DoRALinear.forward` (`dora.py` l.122–130):
```python
adapted = self.linear.weight + self.scaling * (self.lora_B @ self.lora_A)   # scaling = alpha/r
row_norm = adapted.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
dora_weight = (self.magnitude / row_norm) * adapted
return F.linear(self.lora_dropout(x), dora_weight, self.linear.bias)
```
`magnitude` is initialised to the row norms of `W0` (l.101–103) and has shape `(out, 1)`.

### 3.4 Which blocks "late" covers

`generate.py::_model` (l.491–640): TAP-CT `'blocks\.(8|9|10|11)\.'` (4 of 12); SPECTRE
`'(backbone\.blocks\.(1[6-9]|2[0-3])\.|feature_combiner\.)'` (8 of 24 + the 4 combiner blocks);
Pillar-0 `'visual\..*atlas_models\.2\.'` (stage 3 of 3: 8 of 12 `MultiScaleAttentionBlock`s,
each with 3 per-scale `CrossWindowBlock`s); VoCo `'layers4\.'` (stage 4 of 4: 2 of 8 blocks).

### 3.5 Warm start (two-stage methods)

`render_config` l.690–697: `probe_init_checkpoint = <SAVE_ROOT>/<dataset>/<backbone>/<mode>/<parent>[/<task>]`
and `probe_init_max_iteration = stage1_steps · accum(parent) = 4000` (13000 in `joint_lm`).
Parents (`lin`, `attn`; `PARENT_METHODS` l.245) keep that periodic save via
`keep_iterations: [4000]` (l.722). `resolve_checkpoint_path` (`probe_init.py` l.120–145) →
`_resolve_with_max_iteration` (l.96–118) requires exactly `checkpoints/model_4000.pt` and
raises otherwise ("an error rather than a fallback"). Each wrapper's `_maybe_load_probe_init`
copies, by key prefix and shape, only `probe.*` (TAP-CT l.213–262, Pillar-0), `probe.*` and
`crop_abmil.*` (SPECTRE l.232–268), or `probe.*`, `crop_abmil.*` and `token_norm.*` (VoCo
l.263–290); LoRA matrices are fresh (B = 0) and the backbone is the pretrained one. Raises if no
compatible `probe.*` key is found (prevents a silent cold start). Head type is inherited from the
parent (`lin_then_lorag` keeps a linear head; `attn_then_lorag` the attentive probe). Stage 2 is a
fresh job: `num_iterations 4000`, `max_iterations 4000`, `warmup_iterations 400`, new AdamW state.
The parent's step-4000 checkpoint is taken from its own 8000-step schedule (LR not yet decayed).

### 3.6 Learning rate groups, weight decay, layer-wise decay

`Trainer.setup_optimizer` (`trainers/trainer.py` l.538–567) calls
`model.get_optimizer_param_groups(base_lr, base_wd)` with the YAML optimizer values.
`optim_groups.py`: `_is_no_decay_name` (l.16–22) → no weight decay for names ending in `.bias`,
tensors with `ndim <= 1`, or names containing `norm`, `bn`, `ln`. `build_decay_param_groups`
(l.25–47): one LR, decay/no-decay groups (frozen and LoRA methods).
`build_backbone_head_param_groups` (l.49–108, used when `training_mode ∈ {full_finetune,
attention_finetune, mlp_finetune}` and `backbone_lr`/`head_lr` set): backbone × {decay, no-decay}
at `backbone_lr`/`backbone_weight_decay`, head (prefixes `probe.`, `crop_abmil.`, and any other
trainable param) at `head_lr`/`head_weight_decay`. With `backbone_layer_decay` (uncommitted
addition, l.80–89 and `backbone_depths` l.110–127): depth k = index of the k-th distinct
`...blocks.<i>.` prefix in registration order (SPECTRE: backbone blocks 1–24 then combiner blocks
25–28, so `top = 28`), params outside blocks take the depth of the preceding block (patch embed:
0); `lr = backbone_lr · decay^(top − depth)`.
Consequence for DoRA: `magnitude` is 2-D `(out,1)` and its name contains none of the no-decay
substrings, so it **receives weight decay 0.01** (see §6).

### 3.7 Schedule

`lr_scheduler: warmup_cosine` (`lr_schedulers/lr_schedulers.py::_WarmupCosineLRScheduler`,
l.247–315): `max_iterations = steps`, `warmup_iterations = max(1, round(0.1·steps))`
(`generate.py` l.731), `warmup_factor 0.01` linear (`_get_warmup_factor_at_iteration` l.477–499:
`lr = base·(0.01·(1−α) + α)`), then `cosine_schedule(base, it−warm, max−warm, min(end_lr, base))`
with `end_lr 1e-6` (`utils/scheduling.py` l.17–23: `end + (base−end)·0.5·(1+cos(π·it/max))`).
The scheduler steps once per optimizer step (`train_iteration` l.880–884: step only on the last
micro-batch of an accumulation cycle), so `max_iterations` is in optimizer steps while
`num_iterations`, `validation_interval`, `save_interval` are in micro-batches (`generate.py`
l.709–731; with accumulation 1 they coincide).

### 3.8 Budget, batch, precision, checkpointing, seeds, EMA

* `NUM_STEPS 8000`, `STAGE1_STEPS 4000`, `STAGE2_STEPS 4000` (`generate.py` l.248–250); `joint_lm`
  (CVPR26 cohort only): `LM_STEPS 26000`, two-stage 13000 + 13000 (`_steps_for` l.259–263).
* Batch 1, `gradient_accumulation_steps 1` (all methods; `BATCH`, `_accum`), except
  `fullft_accum`: accumulation 8, 1000 optimizer steps, `gradient_clipping 1.0`,
  `validation_interval`/`save_interval` 1000/500 micro-batches (= every 125/62.5 optimizer steps;
  l.716–723). Every other run: `gradient_clipping: 0.0` (comment l.735–737 and module docstring
  l.44–46: a clip of 1.0 at batch 1 cancelled the pos_weight).
* Accumulated loss is divided by the accumulation factor (`train_iteration` l.862–866).
* Precision `bf16-mixed` (YAML `fabric.precision` and `--precision bf16-mixed` in
  `submit_batched.sh` l.250); `torch.set_float32_matmul_precision("medium")` (`launch.py` l.69).
* Gradient checkpointing: VoCo only (`grad_checkpointing: true`); TAP-CT uses SDPA instead.
* Seeds: `--seed` default 42 (`environment.py` l.433–441), `seed_everything(seed + global_rank,
  workers=True)` (l.189–193); no deterministic-algorithm flag anywhere (grep). Seed repeats
  `joint_s43`/`joint_s44` are the `joint` configs with a `# seed: 43|44` header
  (`render_config` l.700–703, 739), passed as `--seed` by `submit_batched.sh` l.237–239, for
  `SEED_METHODS = [lin, lora, attnft, fullft]` minus `(spectre, fullft)` (`write_revision` l.886).
  The sampler's permutations and dataset choice derive from the same seed (§1.7).
* EMA: none (no EMA in `trainers/`, `train.py`, `launch.py`, `environment.py`).
* Checkpoints (`hooks.checkpoint`, YAML l.at `checkpoint_hook`): `save_interval 500`,
  `trainable_only: true` (`CheckpointHook._trainable_state_dict`, `engine/hooks.py` l.1587–1611:
  `requires_grad` params plus all buffers), `max_checkpoints 2` periodic (`model_<it>.pt`),
  `keep_iterations` protected, `save_best: true` on `monitor_metric: val/metric/val_auroc`,
  `mode: max` → `checkpoints/best/best_val_metric_val_auroc.pt` and `.txt` with
  `best_value`, `iteration` (l.1393–1420).
* Job command (`submit_batched.sh` l.245–253): `python aifofomo/train.py --config <yaml>
  --save-dir <run> --experiment-name <name> --device cuda --precision bf16-mixed --strategy
  ddp_find_unused_parameters_true --num-devices 1 --num-nodes 1 [--seed N] [--resume-training]`.

### 3.9 Training loop

`Trainer.train` (`trainers/trainer.py` l.688–818): infinite loader, iteration = one micro-batch,
OOM retry on the same iteration; validation when `iteration % validation_interval == 0`
(l.795). `train_iteration` (l.820–892): zero grad at the first micro-batch, forward
(`SupervisedTrainer._forward_pass`, `trainers/supervised/supervised.py` l.75–118: `model(batch["data"])`),
loss (`_compute_loss` l.120–141 with `arg_bindings {pred: logits, target: label_vec,
label_mask: label_mask}`), backward of `loss/accum`, then clip (if > 0) → `optimizer.step()`
→ `scheduler.step()` on the last micro-batch.

### 3.10 Parameter counts logged

`_log_param_summary` in each wrapper logs `trainable / total` and backbone trainable/total at
model build (`TapCTClsModel [mode=…]: … trainable`). Values found in run logs: §2.1 and §2.3.

### 3.11 Uncommitted changes relevant here

`git diff --stat`: `optim_groups.py` (+layer-wise decay), `tapct_cls.py`, `spectre_cls.py`,
`pillar0_cls.py` and their configs (`mlp_finetune` mode, `backbone_layer_decay` field),
`lr_schedulers.py` (`min(self.end_lr, base_lr)` so an LLRD group whose peak is below 1e-6 does not
rise), and the regenerated YAMLs (`pin_memory: false` for SPECTRE loaders). The repository holds
revision/seed configs for TAP-CT, SPECTRE and Pillar-0 only; the generator would also write them
for VoCo (fdg, rsna2022, rsna2023) but they are absent, so no VoCo seed repeats or revision
methods were run from this tree.

---

## 4. Per-task vs joint

* `mode: pertask` → `num_tasks 1` (`_num_tasks`, l.628–634), one config per label
  (`<method>/<label>.yaml`), dataset entry with `target: <label>` (tables) or the single AMOS/
  COVID/LUNA entry with `single_task_label_vec: true`; `metric task_names` = the one canonical name.
* `mode: joint` → `num_tasks 17` (CVPR26) or `len(tasks)` (4 / 8 / 4 / 6 / 2), one encoder, one
  set of adapters, one head with one logit per label (`Linear(embed, num_tasks)` or
  `AttentiveProbe(num_classes=num_tasks)`); the same window/crop tokens feed every output.
  Joint CVPR26 additionally differs in sampling (one AMOS row carries ≤15 labels; cohort weights
  10:8:8, §1.7) — `joint` and `joint_lm` both use the weights.
* **Loss** — `MultiLabelBCELoss` (`losses/multilabel_bce.py` l.40–73) with
  `reduction: sum_tasks_mean_batch` and `pos_weight_list`:
  ```python
  per_entry = F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=pos_w, reduction="none")
  ...
  loss_val = (per_entry * m).sum() / float(per_entry.shape[0])      # m = label_mask.float()
  ```
  i.e. L = (1/B) Σ_b Σ_t m_bt · BCE(z_bt, y_bt; w_t), `w_t = clip(neg_t/pos_t, 0.05, 20)` on the
  train manifest (§1.4). With `masked_mean` (not used) it would be `Σ(per_entry·m)/Σm`. The
  shape-mismatch branch (l.56–67) collapses a `(B,T)` mask onto a `(B,1)` prediction with `any`.
* Label weighting beyond `pos_weight`: none. Sampling: §1.7.

---

## 5. Model selection and evaluation

* **Validation**: `validation_interval 1000` micro-batches (= optimizer steps at accum 1; 8 per
  run, 26 for `joint_lm`; `ValidationSettings` requires a multiple of 10, `trainer.py` l.77).
  `Trainer.validate` (l.1135–1213) groups the validation loaders by `dataset.split`: `val`/
  `val_train` → logged under `val/`, `test`/`val_holdout` → under `test/`; each group is one
  pass over all its loaders (so the 17 CVPR26 labels are aggregated inside the split), and the
  per-case records of both groups are concatenated with a `split` column.
* **Metric**: `cvpr26_joint_eval` (`metrics/cvpr26_joint_eval_metric.py::CVPR26JointEvalMetric`,
  `task_names` = the run's labels, `optimize_balanced_threshold: false`,
  `arg_bindings {predictions: probabilities, targets: label_vec, label_mask: label_mask}`).
  `get_epoch_result` (l.113–185): per label, keep rows with `label_mask`; skip the label if
  `< 2` valid rows or non-finite predictions; if only one class is present, emit 0.5 and skip it
  from the macro mean; else `roc_auc_score(yt, pi)` (sklearn, ties 0.5) and balanced accuracy at
  threshold 0.5. `val_auroc` = mean over the retained labels (0.0 if none), emitted as
  `val/metric/val_auroc` and per label `val/metric/val_auroc/<dataset__target>`.
* **Selection**: `checkpoint_hook.monitor_metric: val/metric/val_auroc`, `mode: max`,
  `_is_better_than` uses strict `>` (`hooks.py` l.1613–1618): ties keep the earlier checkpoint;
  the first validation always becomes "best" (initial −inf). Edge case: if the selection split has
  one class for every label of a run, `val_auroc` is 0.0 at every validation and the step-1000
  checkpoint is the selected one. The held-out (`test/`) metrics are logged but never used for
  selection.
* **Per-case files**: `validation_case_json` hook (`engine/hooks.py::ValidationCaseJsonHook`,
  l.478–592; `write_csv: true, write_json: false`) writes
  `<run>/val_case_metrics/iter_<iteration:06d>/logits.csv` with columns
  `case_id, split, task, task_index, label, logit, probability`, one row per (case, known label)
  for both splits, from the records built in `inference/inferencer.py` l.362–445 (`probability =
  sigmoid(logit)` of output column `t`, `label = label_vec[t]`, only `label_mask[t]` columns;
  `task` = canonical `dataset:target` via `_record_task_name`, l.23–37). For FDG the `case_id`
  is the patient id, so a patient with several studies has repeated `case_id`s in `logits.csv`.
* **Reported number**: `midl_paper/results/scripts/harvest_metrics.py::harvest_run` (l.400–467):
  selection iteration = `iteration:` in `checkpoints/best/best_val_metric_val_auroc.txt`; rows
  of `val_case_metrics/iter_<it>/logits.csv` grouped by `(task, task_index)`; held-out AUROC =
  `auroc_rows` (l.196–208, Mann–Whitney with midranks, ties 0.5, NaN if one class) on the `test`
  rows; labels skipped when val or test has <2 rows or one class; checks val/test case-id
  disjointness ("leak") and that the recomputed val AUROC matches the logged
  `val/metric/val_auroc/<label>` within 2e-3 ("mapping_mismatch"). `score_run.py` (fm_adapt) does
  the same per run and writes `score.json`.
* **"Did not train" / prediction-spread rule**: `midl_paper/results/scripts/case_level.py::prediction_spread`
  (l.42–47): per (backbone, method), the median over label–mode pairs (`pertask`, `joint`, `test`
  split) of the standard deviation of the held-out probabilities;
  `make_tables.py` l.421–436: `SPREAD_MIN = 0.01`, pairs below it go into `DID_NOT_TRAIN`
  (comment: "SPECTRE Full FT 0.004, every other pair at least 0.11"); l.476–481: not applied to
  VoCo. Nothing in `aifofomo` implements this check; it lives only in the paper scripts.

---

## 6. Discrepancies and omissions (draft vs code)

Draft = `midl_paper/paper/main.tex` (Methods l.63–88, Appendix l.187–271, incl. `tab:optim`).

1. **DISCREPANCY (DoRA weight decay)**: draft l.249 "with the magnitude m ... trained alongside A
   and B, without weight decay like every one-dimensional parameter" vs code: `magnitude` has
   shape `(out, 1)` (`dora.py` l.101–103), `_is_no_decay_name` excludes only `ndim <= 1`, `.bias`
   or names containing `norm|bn|ln` (`optim_groups.py` l.16–22) → the magnitude is in the decay
   group (wd 0.01).
2. **DISCREPANCY (splits wording)**: draft l.84 "Official validation sets are split into a
   selection and a held-out half" vs code: true for AMOS/COVID/LUNA (`_select_split_rows`
   l.942–971) only; FDG, RSNA2022, RSNA2023 use a seeded 80/10/10 stratified split of the released
   training data (`preprocess_rsna_nifti.py` l.124–143, `preprocess_rsna2023_labels.py` l.4–7),
   RSNA2019 a table built outside the repo with a 5k stratified train subset
   (`build_ich_stratified_table.py`). "patients never span splits" is enforced in code for FDG only
   (`build_gaia_adapters.py` l.69–83); RSNA2022 is split per study and CVPR26 per file.
3. **DISCREPANCY (cohort sampling)**: draft l.213 attaches the 10:8:8 cohort sampling to the
   longer `joint_lm` schedule; code applies `infinite_sampler_weights [0.3846, 0.3077, 0.3077]`
   to every CVPR26 joint run, 8000-step `joint` included (`generate.py` l.714–718, `JOINT_MODES`).
4. **OMISSION (attention-only FT scope)**: draft l.250 "unfreezes every parameter of the attention
   modules ... including the output projection and the biases"; code additionally unfreezes any
   other parameter of the attention module, e.g. VoCo's `relative_position_bias_table`
   (`unfreeze_attention_modules`, `lora.py` l.279–285).
5. **OMISSION (SPECTRE all-block LoRA scope)**: draft l.248 "to the query and key/value projections
   of every SPECTRE block and of its feature combiner" — matches; note the include pattern
   `'(backbone\.blocks\.|feature_combiner\.)'` also governs `attnft`/`mlpft` and that in training
   the combiner sees a 2×2×2 grid (8 crop tokens), at evaluation 4×4×4 (draft states the crop
   count but not that the combiner's token count differs between training and evaluation).
6. **OMISSION (VoCo attentive probe)**: draft l.237 lists hidden sizes per backbone for the linear
   head only; for VoCo the attentive probe projects 2688 → `probe_hidden_dim 768` before the two
   blocks (`voco_cls.py` l.189–197), unlike the other backbones where hidden = embed dim.
7. **OMISSION (VoCo warm start)**: draft l.211 "take the head from step 4000"; for VoCo the running
   token standardiser (`token_norm.*` buffers) is also copied (`voco_cls.py` l.263–290).
8. **OMISSION (Pillar-0 preprocessing)**: draft l.258 omits the `LPI` reorientation for the ICH
   head CTs (`generate.py::_orientation` l.385–386) and that the linear head's pooled embedding is
   L2-normalised (`encode_image(normalize=True)`), as is the attentive head's pooled vector.
9. **OMISSION (Pillar-0 BatchNorm)**: under LoRA/attnft/mlpft the stem BatchNorm layers are
   frozen in eval mode, under full FT they run in train mode at batch size 1
   (`pillar0_cls.py::train`, l.441–449). Not stated in the draft; relevant to any statement about
   Pillar-0 full fine-tuning.
10. **OMISSION (TAP-CT depth padding and zoom)**: draft l.256 omits that depth is padded with air
    to ≥12 slices and to a multiple of 4 (`_ensure_min_roi` l.355–377) and that `RandomZoom3D`
    also rescales the depth axis (so "native slice spacing" holds before augmentation only).
11. **OMISSION (edge cases of selection)**: ties and single-class selection splits (§5) are not
    described; harvest skips labels with a single class in val or test (`harvest_run` l.439–441),
    which the draft does not state.
12. **WORDING**: draft l.84 "RSNA 2023 abdominal trauma (4 organ injuries)" — labels are
    `any_injury` plus three organ injuries (`RSNA2023_LABEL_NAMES`). Draft l.72 "a learned query
    token attends over the ... tokens" — the probe is two full self-attention blocks over
    `[CLS]+tokens` with CLS read-out (`attentive_probe.py` l.167–195); the appendix wording (l.237)
    is accurate.
13. **MISMATCH (revision runs on VoCo)**: the generator defines seed repeats and revision methods
    for every backbone, but the repository contains them only for TAP-CT, SPECTRE and Pillar-0
    (§0); the draft's revision appendix should not imply VoCo seed repeats.
14. **STALE DOCSTRING (not a paper issue)**: `generate.py` l.15–16 and l.25 ("VoCo lin: 1e-4 /
    0.01", "Q/V in fused attn") contradict the code (`BACKBONE_METHOD_OPTIM = {}`, fused `qkv`
    adapts Q, K and V).

Consistent with the draft (checked): 8000 steps, batch 1, 10 % linear warm-up from 1 %, cosine to
1e-6, AdamW β (0.9, 0.999) ε 1e-8, no clipping, bf16, masked positive-weighted BCE with clip
[0.05, 20], left–right flip and zoom [0.85, 1.15] on train only, validation every 1000 steps on
both splits, periodic checkpoints every 500 steps of trainable weights, selection by validation
macro AUROC, two-stage 4000 + 4000 from the parent's `model_4000.pt` with a fresh optimizer, seed
42, LoRA r 16 / α 32 / dropout 0.1 on `qkv` (TAP-CT, VoCo) or `q`,`kv` (SPECTRE, Pillar-0), rsLoRA
α/√r, late-block definitions, head dropout 0.2, probe architecture, ABMIL definition, per-backbone
preprocessing windows/spacings/sizes, TAP-CT stride 3 with a final window, SPECTRE 2×2×2 training
sub-grid, VoCo 96³ non-overlapping windows with ≤16 training windows and running standardisation,
Pillar-0 11 input channels and ≤4096 tokens per scale, the `joint_lm` 26000-step schedule.

---

## Unverified items

* [TODO] SPECTRE and VoCo parameter counts (no run log with the `ClsModel [mode=` line found
  under `/projects/direct/temp/fm_probe/v2` for these backbones; TAP-CT and Pillar-0 counts quoted).
* [TODO] Whether Pillar-0's Atlas `drop_path_rate 0.3` (and SPECTRE/TAP-CT drop-path, if any) is
  active in train mode under LoRA / full FT.
* [TODO] Patient-level disjointness for RSNA2022 (split per `StudyInstanceUID`) and for the CVPR26
  cohort (split per file; no patient id in the manifests).
* [TODO] The original FDG `cls_labels.csv` split builder and the RSNA2019 `patient_table.csv`
  builder (`pillar0/rsna2019_ich/scripts/build_cls_tables.py`) are outside this repository.
* [TODO] Draft figures of "encoder parameters trained per label (from the training logs)" were not
  recomputed here.

Addendum (main session, 2026-10-31): AdamW defaults verified in `aifofomo/optimizers/optimizers_config.py` l.85-90: `betas=(0.9, 0.999)`, `eps=1e-8`; no generated YAML sets `betas`/`eps`.
