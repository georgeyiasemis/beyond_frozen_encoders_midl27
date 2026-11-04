# Rewrite report: "Adapting 3-D CT Foundation Models Beyond Frozen Probes"

Date: 2026-10-31. Everything in `midl_paper_v2/` was produced from the code in `aifofomo/` (working tree) and the run
outputs under `midl_paper/results/raw/`; the previous draft was used only to know what the study covers.

## Deliverables

| Item | Location |
|---|---|
| LaTeX source (MIDL 2027 class), sections, bib | `paper/main.tex`, `paper/sections/*.tex`, `paper/refs.bib` (verified entries only), `paper/midl.cls` |
| Compiled PDF | `paper/main.pdf` |
| Metrics script and outputs | `metrics/recompute.py` -> `metrics/out/*.csv`, `metrics/out/numbers.tex`, `metrics/out/changes_vs_draft.csv`; procedure in `metrics/PROCEDURE.md`, macro names in `metrics/NAMING.md`, output schema in `metrics/SCHEMA.md` |
| Design counts | `tables/design_numbers.py` -> `paper/tables/design_numbers.tex` |
| Table generator | `tables/make_tables.py` -> `paper/tables/*.tex`; it calls `tables/paper_tables.py`, which writes the tables in the previous draft's layouts and file names (comparison with the old tables: `tables/PAPER_TABLES.md`) |
| Figure scripts | `figures/style.py` (palette, fonts), `figures/method_figures.py` (diagrams drawn from the code; facts in `figures/METHOD_FIGURES.md`), `figures/results_figures.py` (from the CSVs; inputs listed in `figures/RESULTS_FIGURES.md`) -> `figures/out/`, copied to `paper/figures/` |
| Audit documents | `report/PHASE1_REPORT.md` (entry point), `report/code_map_classification.md`, `report/code_map_segmentation.md`, `report/results_pipeline.md`, `report/run_inventory.md` + CSVs, `report/citation_audit.md`, `report/refs_verified.bib`, `report/style_guide.md` |

Rebuild: `cd metrics && python recompute.py` (about 3 min with the cache), `python ../tables/design_numbers.py`,
`python ../tables/make_tables.py`, `cd ../figures && python method_figures.py && python results_figures.py`,
copy `figures/out/*.pdf` to `paper/figures/` and `metrics/out/numbers.tex` to `paper/tables/` (dropping duplicate
definitions), then `tectonic -X compile paper/main.tex`. Python: `~/miniconda3/envs/aifofomo/bin/python`; TeX:
`~/miniconda3/envs/tex/bin/tectonic`.

## 1. Code map
See `report/PHASE1_REPORT.md` section 1 (summary) and the two code maps (file and function for every item).
Key facts that shaped the paper: two of the four backbones are hierarchical (Pillar-0: convolutional stem +
three-stage multi-scale window transformer; VoCo: Swin); splits are official-validation halves only for CVPR26 and
AMOS22, seeded divisions elsewhere; VoCo ran three classification cohorts and no repeats or controls; DoRA's magnitude
is weight-decayed; SPECTRE segmentation adapters also wrap the unused feature combiner (no gradient); segmentation
Dice is stored as 0/valid=0 for absent, unpredicted structures and must be filtered; TotalSegmentator held-out Dice is
re-inferred after training; AdamW defaults (0.9, 0.999), eps 1e-8 (verified in `optimizers_config.py`).

## 2. Discrepancy log (previous draft vs code/outputs; internal record, not paper content)
`report/PHASE1_REPORT.md` section 3 (19 items). The new paper does not reference the old draft; it states what the code
did. The items that would have been wrong statements are: "three vision transformers" (Pillar-0 is hierarchical),
"official validation sets split in halves" for all cohorts, "VoCo trained with every method and mode", VoCo channel
counts, DoRA without weight decay, combiner excluded from SPECTRE segmentation counts, 10:8:8 sampling tied to the long
schedule only, "4 organ injuries" (any + 3), attentive probe as a cross-attention query.

## 3. Citation audit
`report/citation_audit.md`. All 54 old entries exist; 11 updated to their published versions; `yang2020covidct` and
`silvarodriguez2023fseft` dropped; `pai2025ctfm`, `claessens2025spectre`, `lee2024rsnacsf` moved to the statements they
support (Lin et al. 2023 now cites the RSNA 2022 dataset); added and placed: DeLong 1988, Benjamini-Hochberg 1995,
Swin Transformer, Swin UNETR, SwinUNETR-V2, ViT, registers, SAM, Lin 2023 (RSNA C-spine data). `paper/refs.bib` holds
only entries verified against a primary source.

## 4. Metric recomputation and changes
One script, `metrics/recompute.py`, recomputes every number from the per-case files of all 2388 finished runs
(procedure: `metrics/PROCEDURE.md`). Verification: the recomputed selection-split metric matches the logged best value
for all 1354 classification runs (to 3e-8) and 1033/1034 segmentation runs (within 2e-3; the one exception is a
re-inferred run at 2.01e-3); the task-index -> label mapping of every joint classification run was checked against the
logged per-label AUROC; the script is byte-for-byte deterministic.

Changes relative to the previous draft's macros (`metrics/out/changes_vs_draft.csv`, 3744 shared names): every point
estimate (pooled AUROC, cohort means, label-level means, ranks, counts) agrees at printed precision. 160 macros differ:
- BH-significance rates by 1-4 percentage points (e.g. SPECTRE LoRA vs Dec significantly ahead 61 -> 57 %), because the
  BH families now contain only the per-task and joint comparisons of the main design and the segmentation bootstrap
  uses 4000 instead of 1000 resamples;
- interval endpoints by 0.001-0.007 (new random streams);
- prediction spreads per (backbone, method) (the old code measured the first label of a joint run; the did-not-train
  decision is unchanged: only SPECTRE Full FT);
- case counts per cohort now come from the per-case files (cases with at least one label): FDG-PET/CT held-out 122 -> 121
  studies, RSNA 2022 selection 202 -> 201.
Deliberate additions: paired tests against LoRA and Attn (classification) and LoRA (segmentation); the true seed per run;
label-level estimates for every method against Lin / Dec.

## 5. Conclusions drawn from the recomputed results (strength of evidence)
Consistent (interval excludes zero with the same sign on TAP-CT and SPECTRE):
- classification: LoRA on the attention projections improves on the linear probe (+0.034 pooled; +0.032 / +0.037 / +0.033 per backbone);
- segmentation: a multi-depth decoder on the frozen encoder beats a linear probe (+0.073; +0.085 / +0.061);
- segmentation: one joint model per dataset is behind per-structure models at our budgets (-0.052; -0.046 / -0.058).
Uncertain (intervals include zero): attention-only fine-tuning vs LoRA (+0.011 pooled, +0.019 TAP-CT only), the attentive
probe vs the linear probe (+0.007), LoRA on top of the decoder (+0.009), full fine-tuning vs the decoder (-0.002), the LoRA
variants vs LoRA, joint vs per-task classification (+0.004).
Conditional: full fine-tuning (did not train on SPECTRE with the default setting; layer-wise LR decay recovers it),
LoRA in segmentation (gains in some joint models, losses on TotalSegmentator), compute-matched joint vs per-structure
(sign depends on the method).
Where it fails: VoCo classification (LoRA -0.071 vs Lin, every encoder adaptation behind the probe except Lin->LoRA);
VoCo segmentation findings hold and full fine-tuning is ahead of the decoder there (+0.015).

## 6. Paper structure and length
Main text (abstract to conclusion) is 10 pages, within the MIDL 2027 full-paper limit; references start on page 11; the
appendix holds data and splits, training details with all method diagrams, the statistical procedure, repeats and
controls, VoCo, ranks, paired tests and per-label / per-structure tables. Compiles with 0 errors, 0 undefined references
or citations, one 9.9 pt overfull line in the appendix. Every number in the text is a macro from
`paper/tables/numbers.tex` (recompute output) or `paper/tables/design_numbers.tex`; style follows
`report/style_guide.md` (from the author's four papers).

Main-text figures: Figure 1 (what each method trains, drawn from the code) and Figure 2 (label-level forest plot);
Table 1 (label-level comparisons). VoCo is reported in Section 4.3 and Appendix E as a separate check, because it ran
three of five classification cohorts, has its own token interface, and was not tuned; the paper does not attribute its
classification result to the architecture.

## 7. Remaining TODOs
- Co-authors and affiliations (title block).
- (resolved 2026-11-01) Default learning rates: stated as set once by parameter type and not tuned (no pilot runs exist on disk).
- Public repository URL (Conclusion).
- Optional: VoCo logged parameter counts (not in any local log); patient-level disjointness for RSNA 2022, CVPR26 and the segmentation datasets cannot be verified from the manifests and the paper says "per case" for them.

## 8. Response to the external review of the previous draft (2026-11-01)
The review was written against the old 55-page draft. Status of each point in the rewrite:

| Review point | Status |
|---|---|
| B1 page limit | Main text is 10 pages (MIDL main track). |
| B2 TODOs | Learning-rate statement written (set once by parameter type, not tuned; no pilot runs exist). Co-authors and repository URL remain. |
| Audit 1 (VoCo exception) | Rewritten around matched-head numbers; Lin->LoRA stated as the exception. |
| Audit 2, 5, 6 | Sentences no longer in the paper. |
| Audit 3 (Attn-only FT "one backbone only") | Now "uncertain; positive mean, interval reaching zero" with per-backbone values. |
| Audit 4 (Pillar-0 "consistently") | Reworded: behind on average with an interval below zero, not for every method. |
| Audit 7 (Curia is 2-D) | Verified (arXiv text and model card: 2-D slices); moved out of the 3-D list. |
| Audit 8 ("self-supervised") | Verified: SPECTRE adds vision-language alignment with reports; Pillar-0 code is a CLIP-style multimodal model. Pretraining now stated per backbone; "self-supervised" kept only where true. |
| M1 VoCo head confound | Added matched-head comparisons (every attentive-head method vs the frozen attentive probe) to the recompute. On VoCo all are level with the attentive probe (LoRA -0.031 [-0.145, +0.043]); the attentive probe is -0.040 behind the linear probe. Text, abstract, conclusion, guide and Figure 9 updated. |
| M2 asymmetric re-tuning | Ranking claims scoped to default settings; Full FT-vs-LoRA comparison flagged as favouring Full FT; Limitations name the two methods that proved LR-sensitive. |
| M3 no segmentation reference | Not run (needs new training jobs). The TAP-CT from-scratch control stays the only reference; Limitations say so. |
| M4 head and encoder mixed | LoRA - Attn row added to Table 1: +0.027 [+0.012, +0.048], positive on 10/10 cohort-backbone pairs. |
| Shortcut question (COVID-CTset, FDG) | LoRA - Lin without them: +0.037 [+0.019, +0.057], reported in Results. |
| Convergence ("59% last checkpoint") | Reworded: the share is inflated by cosine decay; the under-training claim rests on the step-tripling gain. |
| Reference corrections | Already in the verified bib; FM-HCT DOI re-verified at Crossref (15 authors, Nature Biomedical Engineering, 2026). MoSAIC (unverified fit) removed; SEA-PEFT reworded to what its abstract supports. |

Not done, because each needs new training runs: LoRA with a linear head and a small LR sweep on VoCo; symmetric re-tuning of LoRA and Attn-only FT; nnU-Net on AMOS22 and KiTS23.

## 9. Restored structure, figures and tables (2026-11-01)

At the author's request the paper returns to the earlier title ("Beyond Frozen Encoders: Adapting 3-D CT Foundation
Models to Multi-Label Classification and Segmentation"), section structure (Introduction; Methods; Experiments;
Results with Classification / Segmentation / Per-task versus joint / VoCo; A practical guide and discussion) and its
figures, diagrams and tables, each regenerated from the verified pipeline and corrected against the code. The compact
version is kept in `paper_compact_20261101/`.

| Item | Script | Notes |
|---|---|---|
| fig_pipeline, fig_arms, fig_arms_seg | `figures/diagrams.py` | facts and corrections in `figures/DIAGRAMS.md`: attentive probe = 2 self-attention blocks over [CLS]+tokens; Attn-only FT trains the whole attention module (VoCo: + relative position bias); VoCo widths 96-768 vs stage outputs 192-1536; decoder 2.45/2.62/2.42 M from logs; Pillar-0 drawn as conv stem + 3-stage multi-scale encoder; encoder parameter counts = parameters that receive a gradient (Pillar-0 about half of the logged count; SPECTRE segmentation without the combiner) |
| fig_overview, fig_key, fig_mechanisms, fig_seg, fig_joint, fig_methods, fig_seg_methods, fig_seg_budget, fig_seg_convergence | `figures/paper_figures.py` | reads only `metrics/out` + training logs; 449 printed values checked against `numbers.tex`, 0 mismatches (`figures/out/paper_figures_checks.txt`); `figure_numbers.tex` |
| fig_examples_*, fig_cases_* | `figures/example_figures.py`, `case_figures.py`, `select_*.py` | same cases selected as before; titles use the per-case files; `example_numbers.tex`, `seg_example_numbers.tex` |
| old-layout tables | `tables/paper_tables.py` (called by `make_tables.py`) | every point estimate equals the earlier tables; differing cells (interval ends, BH rates, two counts) listed in `tables/PAPER_TABLES.md` and `tables/paper_tables_diffs.csv`; `recompute.py` gained `label_ci.csv` and `params.csv` |
| tab:overlap | hand-written in `appendix.tex` | from the backbones' papers (SPECTRE Table 4, VoCo PreCT-160K Table I, TAP-CT and Pillar-0 in-house data); LUNA25-from-NLST left as TODO |

Main text: Figures 1-4 (overview, pipeline, mechanisms, segmentation) and Table 1 (key comparisons); the per-cohort
tables (Tables cls/seg) open Appendix E. Main text = 10 pages, references from page 11.
Open TODOs: co-authors, repository URL, confirm that LUNA25 is drawn from NLST.
Rebuild adds: `python figures/diagrams.py`, `python figures/paper_figures.py`, the example/case scripts, then copy
`figures/out/*.pdf` to `paper/figures/`.

## 10. Re-review response (2026-11-01, second review)

| Point | Change |
|---|---|
| Neutrality 1 (intro evidence sentence) | replaced by the matched-head LoRA gain and the full-FT failure, no backbone named |
| Neutrality 2 (Figs 3b, 4d) | removed from main figures; new appendix `fig_bbgap` "between-backbone difference per method" |
| Neutrality 3 ("hierarchical") | Pillar-0 = "multi-scale" (fig_overview), VoCo = "Swin" everywhere; section 4.4 "A Swin backbone: VoCo" |
| N1 run counts | the 33-run gap = Pillar-0 breast-MRI runs (22 per-task + 11 joint) that are in runs.csv but excluded from every estimate; Appendix B now quotes main + revision runs (1988 + 367) |
| N2 LR rule | stated per head type, matching Tables optim/optimseg |
| N3 rsLoRA | at LR/4 level with LoRA on SPECTRE (+0.003) and Pillar-0 (+0.008), still behind on TAP-CT (-0.020) |
| N4 VoCo matched head | all eight attentive-head methods have label-level intervals including zero (label_level.csv keys cls_<m>_attn); text gives LoRA, Attn-only FT, Full FT with intervals and the range of the other five |
| N5 | "as accurate as" -> "on average level with" (abstract, intro, 4.3, conclusion) |
| N6 | scratch control in 4.2 with its learning-rate caveat |
| N7 | +0.037 without / +0.034 with FDG and COVID; unit change flagged for the collapse sentence; convergence shares are the recomputed "last two validations" shares (a different quantity from the earlier draft's "last checkpoint of 8000-step runs") |
| N8 | appendix floats [H], \clearpage per appendix section, float heights capped; no overfull boxes |
| References | LUNA25 TODO closed with the Zenodo record 10.5281/zenodo.14223624 ("acquired in participants who enrolled in the NLST"); SwinUNETR-V2 full author list (Crossref); Merlin, CT-CLIP, VoCo TPAMI, Ly & Nguyen pages and DOIs re-checked at Crossref; FM-HCT has 15 authors at Crossref (published version; the preprint has 13); TotalSegmentator v3.0.0 DOI printed as URL |
| Not done | symmetric re-tuning (needs GPU jobs); nnU-Net (declined by the author) |

## 11. Abstract and introduction motivation (2026-11-02)

- Abstract rewritten to follow the title: motivation -> "Yet ... frozen, since ..." -> "However, ..." -> "In this work, we go beyond frozen encoders and compare ..., in an effort to find out ..." -> "We consider four public 3-D CT encoders ... We use ...". The four encoders are treated together (no separate VoCo sentence); results are qualitative, with no metric values.
- Intro paragraph 1: natural-image motivation, with each claim checked against the paper text:
  - he2022mae: linear probing and fine-tuning are largely uncorrelated; tuning one block takes accuracy from 73.5% to 81.0% (Sec. 4.3, Fig. 9).
  - jia2022vpt: VPT outperforms full fine-tuning on 20 of 24 tasks (Table 1).
  - kumar2022lpft: from its abstract, fine-tuning is better than linear probing in distribution and worse out of distribution.
  - The bib entries for MAE and VPT come from Crossref.
- To keep the main text at 10 pages, prose was trimmed; no figure or table was removed.
  - The SuPreM, VISTA3D and nnU-Net citations moved to the Limitations paragraph.
  - The repository-URL TODO moved into contribution bullet 1.
- The previous abstract is saved in the scratchpad as abstract_prev.tex.

## 12. Figure revision (2026-11-02)

Fig 1 redrawn (design card fixes, categorical "what is trained" axis, VoCo in b-e), Fig 2 header and box fixes,
fig_mechanisms split into the appendix figures fig_gain_vs_probe and fig_fullft_spectre, new main-text figure
fig_joint_main, VoCo panel in fig_joint. New macros `\numRankall*` / `\numSegrankall*` (recompute.py, PROCEDURE.md
section 6; all other outputs byte-identical). Details: `figures/PAPER_FIGURES.md` and `figures/DIAGRAMS.md`
(revision sections); number check 614 comparisons, 0 different.

## 13. Multi-label framing and main-text floats (2026-11-02)

- **Framing.** The abstract (about 230 words) and the introduction now centre on datasets annotated for many targets: multi-label classification cohorts and multi-structure segmentation datasets. The two questions are what to train, and one model per dataset or one per label.
- **Main-text floats**, in order:
  - Fig 1 overview: VoCo is now in every panel. The rank bars pool all backbones, using `\numRankall*` and `\numSegrankall*`.
  - Fig 2 pipeline: TAP-CT appears in the caption only.
  - Table 1: key comparisons.
  - Fig 3: per-structure segmentation.
  - Fig 4 (new, `fig:jointmain`): joint vs per-task, classification and segmentation, all backbones.
- **Moved to the appendix.** fig_mechanisms was split into `fig:gainprobe` (app:paired) and `fig:fullftspectre` (app:revision, next to tab:fullftfix).
- **Appendix pointers.** The results now open with one sentence that points to the per-label, paired, seed and case appendices.
- **Rank text.** It now uses the all-backbone ranks (Attn-only FT 4.22, LoRA 5.14; segmentation Full FT 2.44, LoRA 3.45). The segmentation Full FT lead is attributed to TAP-CT (2.05) and VoCo (1.42).
- **Other fixes.**
  - Removed the KiTS23 GitHub footnote.
  - Added `\emergencystretch=2em`, which removes the overfull lines.
  - Placed `\vspace{-14pt}` between fig_joint_main and its caption.

## 14. VoCo integrated everywhere (2026-11-02)

- Results: Section 4.4 is removed. VoCo now appears inside 4.1 (with its own paragraph, "On VoCo, the head rather than the encoder decides"), 4.2 and 4.3.
- fig_methods and fig_seg_methods now include VoCo.
  - fig_seg_methods titles show only the method name, because VoCo's +0.68 gain over its linear probe would dominate any pooled mean.
  - In fig_methods the note moved to the empty panel so it no longer overlaps the legend.
  - The check log has 0 mismatches. The VoCo values are logged as INFO lines because VoCo has no label-mode macro.
- fig_voco.pdf (the separate VoCo-versus-main figure) is not referenced anywhere and is kept on disk only.

## 15. Results rewritten claim-first (2026-11-02)

- Subsections: "Classification: train the attention weights", "Segmentation: a decoder on the encoder", "One model per dataset or one per label". Every paragraph heading states the finding.
- New evidence surfaced in the main text:
  - LoRA and Attn-only FT beat the attentive probe in 10/10 cohort–backbone pairs.
  - Attn-only FT is ahead of LoRA in 25/30 seed pairs.
  - Lin→LoRA is the only method with a positive mean against Lin on all four backbones.
  - Joint LoRA beats per-structure LoRA at matched compute in all four pairs (+0.042 to +0.136).
- Checked against the macros: Attn-only FT ranks first only on TAP-CT (second on SPECTRE and Pillar-0), so the text says "best pooled rank, above LoRA on every backbone".
- The previous version is saved in the scratchpad as results_pre_strong.tex.

## 16. VoCo-free variant (2026-11-02)

- `paper_novoco/` is a copy of `paper/` without VoCo. Its figures and tables come from the same scripts run with the
  environment variable `NO_VOCO=1`. Defaults are unchanged: without the variable every script writes `paper/` and
  `figures/out/` exactly as before (checked: check log, figure_numbers.tex and all 60 table files byte-identical, PNGs
  pixel-identical).
- Output folders with `NO_VOCO=1`: `figures/out_novoco/` (PNG/PDF, `paper_figures_checks.txt`, `selection/` check logs)
  and `paper_novoco/{figures,tables}/`. `PAPER_DIR=<dir>` and `FIG_OUT=<dir>` override them (style.py, make_tables.py,
  paper_tables.py).
- Rebuild: `cd tables && NO_VOCO=1 python make_tables.py --check`; `cd figures && NO_VOCO=1 python paper_figures.py`,
  `NO_VOCO=1 python diagrams.py`, `NO_VOCO=1 python example_figures.py`, `NO_VOCO=1 python case_figures.py`.
- Tables only drop VoCo rows and columns. The other cells are the same CSV values (every bootstrap stream is seeded per
  comparison and backbone, VoCo has its own seed and its own BH family, and the "all" rows pool TAP-CT, SPECTRE and
  Pillar-0 only). fig_overview (c, e) use the main-backbone ranks `\numRank*` / `\numSegrank*` in this variant.
- `tab_spread.tex` is now written by `paper_tables.py` (`legacy_spread`, from spread.csv); the default output equals the
  earlier hand-made file.

## 17. No-VoCo text (2026-11-02)

- paper_novoco/sections were edited by scratchpad novoco_text.py plus caption edits:
  - The abstract and intro now say three encoders and two architecture families.
  - New macros RunsMainNoVocoN (1516) and RunsRevisionNoVocoN (367), produced by tables/design_numbers.py.
  - Pooled ranks use \numRank*/\numSegrank* (main backbones).
  - The VoCo results paragraph and guide step 1 (head choice) are removed.
  - VoCo clauses are removed from the methods, setup, stats and appendix details; the VoCo columns are removed from the overlap and cohort tables.
  - Case and example captions updated: 4 rows per figure.
- The PDF contains no "VoCo". The main text ends on page 10, with about half a page free.
- Build: `tectonic -X compile paper_novoco/main.tex`.

## 18. VoCo quarantined; review fixes (2026-11-02)

VoCo quarantine:
- The VoCo-free paper is now `paper/`. The paper with VoCo, the old `figures/out/`, `metrics/out/` and the inventory are in `quarantine_voco_20261102/`; its README says how to restore them. Nothing was deleted.
- `report/run_inventory.csv`: the 491 VoCo rows that were not yet excluded now have `excluded = voco_quarantined_20261102`. No other cell changed, and the CRLF line endings are kept.
- Generators: `NO_VOCO` is now the default (style.py, make_tables.py, paper_tables.py), and outputs go to `paper/` and `figures/out/`. `WITH_VOCO=1` gives the old variant, which needs the VoCo inventory restored. The `NO_VOCO=1` flag, `paper_novoco/` and `figures/out_novoco/` of §16 and §17 are gone.
- `metrics/recompute.py` was rerun without VoCo. 34 pooled macros had still counted VoCo runs in the old copy: the convergence shares and counts, the selection-check counts, the DeLong and paired-test counts, and rankall. All other macros are identical.
- The convergence "all" rows and the selection-check counts now leave out the 33 breast-MRI runs. Result: 1112 + 771 = 1883 = 1516 main runs + 367 repeats and controls.

Review fixes:
- New macros:
  - `jointlong_<bb>_<method>_meandiff` and `jointlong_<bb>_{min,max}diff`: CVPR26 26k minus 8k joint models.
  - `\numParams*PillarZ*` and `\numParamsratio<Bb>AttnftLora`, in diagrams.py.
- Text:
  - "a fiftieth / fifty times": now 19 to 54 times, measured on encoder parameters.
  - The MLP-only control was added to §4.1, and the section is retitled "adapt the encoder".
  - The Table 20 budget sensitivity is discussed.
  - "No variant improves on LoRA" now says "by more than +0.002".
  - The seed-pair and main-run sentence was split in two.
  - The segmentation crop sampler is stated.
  - The LoRA matched-compute result is put in context.
  - Guide: item 2 is softened, item 3 now gives the max +0.028 (DoRA on SPECTRE), and item 4 is rewritten as "one model per structure when the budget allows".
  - Limitations now cover per-label test power.
  - The abstract scopes Pillar-0 to head CT.
- Captions:
  - Figure 15: the alternative settings ran on all backbones.
  - Table 14: what bold marks.
  - Figure 13: the grid caveat.
  - Figure 17: the easy LiTS case.
- US spelling in the text and in three figure strings. "slots" became "label–mode pairs".
- Build: 57 pages, 0 errors, 0 overfull, 0 "VoCo"; the main text ends on page 10. Figure check: 611 comparisons, 0 different.

## 19. Cost table and batch size (2026-11-02)

- Time from the logs: `report/cost/parse_logs.py` reads the 1883 runs of the paper (`runs_used.csv`) and writes `steps.csv`, with the median seconds per step (validation excluded) and the GPU model of each run.
  - The runs used five GPU models: H100 (93 GB), H100 80GB HBM3, A100-SXM4-40GB, A100 80GB PCIe and RTX A6000.
  - `tables/cost_table.py` puts every time on one GPU with a fixed-effects fit per task, log(s/step) = configuration + GPU, using H100 as reference.
  - The GPU factors are in `report/cost/gpu_factors.csv`. The residual spread is 16% (classification) and 28% (segmentation).
  - The resulting wall-clock includes data loading, which dominates for SPECTRE.
- Memory and GPU time: `report/cost/measure_memory.py`, run as one job (14649, A100 80GB PCIe, `measure_memory.sbatch`), writes `memory.csv`.
  - It measures peak `max_memory_allocated` and the median time of a bf16 forward, backward and AdamW step for every backbone and method, built from the run configs.
  - Classification uses the largest of 16 training inputs at batch size 1; segmentation uses a two-crop micro-batch.
  - All 47 configurations were measured.
- Paper:
  - New Table `tab:cost` in Appendix A, with macros in `tables/cost_numbers.tex`.
  - §4.1: LoRA's memory is within 1.0 GiB of Attn-only FT and its steps are not faster, so LoRA's saving is in the stored weights.
  - Guide item 1 is reworded to "adapt the encoder's attention weights", with LoRA as the option when stored weights must be small.
  - Batch size 1 is now described as "chosen empirically" (setup and Appendix A).
- Build: 58 pages, 0 errors, 0 overfull; references start on page 11.

## 20. SPECTRE pretraining overlap resolved (2026-11-02)

The "possibly" rows of tab:overlap came from two aggregate datasets in SPECTRE's pretraining set whose members we had not checked: AbdomenAtlas1.0Mini (5,195 scans) and AbdomenCT-1K (1,062 scans).
- Sources checked: SPECTRE PDF Table 4 and A.1 (it removed only AMOS from its own evaluation); AbdomenCT-1K arXiv v2 Sec. III-A (LiTS 201, KiTS 300, MSD Pancreas 420, MSD Spleen 61, NIH Pancreas 80, Nanjing 50); AbdomenAtlas-8K appendix Table 3, whose counts sum to 5,195 (LiTS 131, KiTS 300, MSD CT 945, AMOS22 200, FLARE'23 2200, ...; TotalSegmentator has no volumes).
- Table now reads: LiTS and MSD Pancreas yes (every case of our splits); KiTS23 partly (the 300 KiTS19 cases are case ids 0-299, which covers 45 of our 73 held-out cases); TotalSegmentator and FDG-PET/CT "not confirmed", because the FLARE'23 sources are only in its supplement, which is not on arXiv.
- New bib entries: ma2022abdomenct1k (Crossref), qu2023abdomenatlas (arXiv API).
- Discussion limitation now names the overlapping sets and says SPECTRE results on them may be optimistic.
- Build: 58 pages, references start on p. 11.

## 21. More results figures in the main text (2026-11-02)

- Caption spacing: `\vspace{-8pt}` between each main-text figure and its caption, `-6pt` under the Table 1 caption; `\textfloatsep` 10pt, `\floatsep` and `\intextsep` 8pt (main.tex).
- Table 1: a rule sets off the two "Joint − per-task" rows (`tables/paper_tables.py`); caption rewritten in plain words.
- Moved into §4: new Fig. `fig:cls` (`fig_cls` in `figures/paper_figures.py`; Attn, LoRA, Attn-only FT and Full FT minus Lin per label and cohort, the classification counterpart of Fig. `fig:seg`; means checked against the paired macros), `fig:gainprobe` (from the appendix; already cited in §4.1) and `fig:examplesseg` (from the appendix; previously not cited anywhere, now cited in §4.2).
- No prose was cut yet, at the user's request: the main text now ends on p. 12, about 1.6 pages over the 10-page limit.
