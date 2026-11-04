# Results figures in the designs of the first draft (paper_figures.py and the example / case scripts)

Run (Python `~/miniconda3/envs/aifofomo/bin/python`, from `figures/`):

```
python paper_figures.py            # 10 figures, figure_numbers.tex, out/paper_figures_checks.txt
python select_seg_examples.py      # seg_example_numbers.tex, out/selection/seg_example_selection.json
python select_cases.py             # out/selection/case_selection.json
python example_figures.py          # fig_examples_seg (+ fig_examples_pred, out/ only), example_numbers.tex
python case_figures.py             # fig_examples_cls, fig_cases_cls, fig_cases_seg_<dataset>
```

Every script writes PDF + PNG to `figures/out/` and copies the PDFs to `paper/figures/` (fig_examples_pred stays in
`out/`). Nothing under `midl_paper/` is written.

## Inputs

* `metrics/out/*.csv` and `metrics/out/numbers.tex` (from `metrics/recompute.py`); `paper/tables/design_numbers.tex`
  for the design counts check.
* Training logs of the runs listed in `metrics/out/runs.csv` (paths relative to `midl_paper/results/`):
  `<path>/logs/metrics.json` (validation AUROC / Dice per evaluation; reader of the draft's `make_figures.py`) and
  `<path>/logs/log_0_0.txt` (the logged line `... : N / M trainable (...); backbone[+combiner] B / T`; classification CT cohorts only, the breast MRI cohort loads other Pillar-0 weights).
* Example / case figures only: per-case files of the runs in `runs.csv` (`percase_file`: `logits.csv`, `dice.csv`)
  for the selection rules and for the numbers printed in panel titles; the saved slices and predictions in
  `midl_paper/results/examples/` (rendered as they are, not re-inferred); CT volumes for the classification views.
* `figures/data/no_grad_cls.csv` and `trainable_params_cls.csv` (written by `diagrams.py`): parameters that are marked trainable but receive no gradient (Pillar-0), and the logged counts used to cross-check the log parser.

Conventions (metrics/PROCEDURE.md): held-out split, seed 42, modes per-task and joint; segmentation "joint" of AMOS22
and TotalSegmentator is the 24 000-step run, "joint_8k" the 8000-step run; classification `joint_lm` (CVPR26, 26 000
steps) is not used. SPECTRE Full FT (classification) did not train (`spread.csv`) and leaves every average over methods,
the ranks and the summary panels; VoCo is drawn per backbone and never pooled; Pillar-0 is classification only.

## Palette

Backbones: `style.BACKBONE_COL` / `BACKBONE_MARK` everywhere (TAP-CT blue circle, SPECTRE vermillion square, Pillar-0
green triangle, VoCo reddish-purple diamond). Methods (only where a figure colours methods: mechanisms, joint, bbgap, seg_budget, seg_convergence), Okabe-Ito hues that no backbone uses, plus greys: Lin grey `#999999`, Dec black,
LoRA sky blue, LoRA+ABMIL black (classification only, never next to Dec), Attn-only FT yellow with a dark rim,
Full FT orange, DoRA green (segmentation only; Pillar-0 never appears there), Full FT with layer decay = orange dotted,
open markers. The draft's non-Okabe-Ito method colours (blue `#2a78d6`, aqua, amber, orange-red, magenta, brown) are
replaced by these. All figures are drawn at the 6.0 in text width (fig_seg_budget at 0.62 of it, as included) and a
save hook raises any text below 6 pt to 6 pt.

## Figures (paper_figures.py)

| Figure | Panels | Data |
|---|---|---|
| fig_overview | (a) design box: backbones with architecture and logged encoder size, methods by family, label / structure counts; (b) pooled held-out AUROC minus the backbone's mean of Lin and Attn, methods grouped by encoder parameters trained (dots: methods, line: group mean), tick labels = range of encoder parameters updated (logged trainable minus those without a gradient, `figures/data/no_grad_cls.csv` from diagrams.py; the counts of fig_arms (d)) over the runs that trained; (c) mean classification rank per method (diamond: pooled over TAP-CT, SPECTRE, Pillar-0; dots: per backbone); (d) mean held-out Dice per segmentation method over the structure-mode pairs with all seven methods, TAP-CT and SPECTRE; (e) mean segmentation rank | metrics_cls, metrics_seg, ranks, training logs |
| fig_key | label-level mean difference with the three-stage 95% interval (lo3, hi3) per backbone incl. VoCo; rows as `tab_key.tex`: LoRA - Lin, LoRA - Attn, Attn-only FT - LoRA, Full FT - Lin, joint - per-task; Dec - Lin, LoRA - Dec (all, per-structure, joint), Full FT - Dec, joint - per-task. An estimate beyond the axis is a triangle at the edge with its value (VoCo Dec - Lin +0.68). | label_level |
| fig_mechanisms | one row of three: (a) LoRA - Lin held-out AUROC per label-mode pair against the probe's selection-split AUROC; bin means (bins <=0.6, 0.6-0.7, 0.7-0.8, >0.8) printed, dotted = Attn - Lin; (b) median validation AUROC over all per-task and joint runs with eight evaluations, TAP-CT solid, SPECTRE dashed; (c) SPECTRE joint models on the 4 cohorts where every curve has all evaluations, incl. Full FT with layer-wise LR decay | paired_cls, metrics_cls, runs + metrics.json |
| fig_seg | (a-c) per structure-mode pair, method - Dec held-out Dice for TAP-CT, SPECTRE, VoCo; grey: pairs (open triangles beyond -0.58), coloured: mean, printed right (a mean beyond the axis is a filled triangle at the edge: VoCo Lin -0.680) | metrics_seg |
| fig_bbgap (appendix) | (a) SPECTRE - TAP-CT held-out AUROC per cohort for Lin, LoRA, LoRA+ABMIL, Attn-only FT (Full FT did not train on SPECTRE); (b) SPECTRE - TAP-CT held-out Dice per dataset for Lin, Dec, LoRA, DoRA, Full FT, same structures and modes. Neutral titles ("between-backbone difference per method"); one legend per panel | metrics_cls, metrics_seg |
| fig_joint | joint - per-task held-out AUROC per cohort (mean over labels) for Lin, LoRA, LoRA+ABMIL, Attn-only FT, Full FT, per backbone | metrics_cls (checked against cohort_means) |
| fig_methods | one panel per classification method: per label-mode pair method - Lin (paired_cls), rows cohorts, colour backbone; large markers cohort means; SPECTRE Full FT drawn open and left out of the title mean | paired_cls |
| fig_seg_methods | one panel per segmentation method: method - Lin per structure-mode pair, TAP-CT and SPECTRE | metrics_seg |
| fig_seg_budget | validation Dice of the AMOS22 joint models (Dec, LoRA, Full FT; TAP-CT solid, SPECTRE dashed): 8000-step runs thick, 24 000-step runs thin; x = iterations / 2 (accumulation 2) | runs + metrics.json |
| fig_seg_convergence | validation Dice minus the run's last evaluation, median over runs, four schedule groups (per-structure 8000; joint 2-3 structures 8000; joint 15 structures 8000; joint 15 structures 24 000), Lin, Dec, LoRA, Full FT, TAP-CT / SPECTRE | runs + metrics.json |

`paper/tables/figure_numbers.tex` (`\providecommand`): `\numTraj<Backbone><method>{First,Last,Max,N}` for every
(backbone, method) of fig_mechanisms (b) (incl. `\numTrajSpectrefullftFirst` 0.540, `Last` 0.529, `\numTrajSpectreattnftFirst`
0.686, `Last` 0.743, N = 37 runs each) and `\numTrajJointSpectre<method>{First,Last}`, `\numTrajJointSpectreCohortsN`
for fig_mechanisms (c). The 168 names shared with the draft's `trajectory_numbers.tex` have identical values.

## Number check (out/paper_figures_checks.txt)

544 lines: 481 comparisons, all equal (0 different, 0 without a reference), and 63 INFO lines (values without a
macro, computed from the CSVs). Compared:
* fig_overview: pooled AUROC per (backbone, method) vs `\pooled<bb><method>` (31); ranks vs `\numRank<Method>Mean`
  and `\numSegrank<Method>Mean`; design counts (39 labels, 37 structures, 11 / 7 methods, 5 cohorts, 5 datasets) vs
  numbers.tex and design_numbers.tex; slots of (d) vs ranks.csv n_slots (59 per backbone); the logged trainable encoder counts behind (b) vs `figures/data/trainable_params_cls.csv` (32).
* fig_key: every mean, lo3, hi3 vs `\numCi<key><bb>{Meandiff,Lodiff,Hidiff}` (156) and the printed +0.68.
* fig_mechanisms: bin means and counts vs `\numGainbin{Low,Mid,High,Top}{Meandiff,N}` and `\numGainbinattn*`.
* fig_bbgap: (a) vs `\numBbdiff<cohort><method>Meandiff`; (b) vs `\numSeggap<method>{Meandiff,MaxMeandiff,MinMeandiff}`.
* fig_seg: (a, b) printed means and n vs `\numSegdelta<bb>Dec<method>{Meandiff,N}`; (c) VoCo: Lin - Dec vs the
  label-level Dec - Lin, LoRA / Full FT vs label-level where every structure has both modes, Full FT n vs macro (the
  three methods without TotalSegmentator per-structure runs are INFO).
* fig_joint: every plotted point vs cohort_means.csv (joint - per-task); pooled over backbones vs `\numJointcohort*`.
* fig_methods / fig_seg_methods: per-backbone parts of each title vs `\numPaired<bb>Lin<method>{Meandiff,Better,N}` /
  `\numSegdelta<bb>Lin<method>{Meandiff,Better}`; the pooled title values have no macro (INFO).

Example / case scripts: `out/selection/example_checks.txt` and `out/selection/case_checks_cohorts_cls_seg.txt`
(selection reproduced, per-case Dice and scores against the per-case files, macro values).

## Changes relative to the draft figures

* All: 6.0 in width, text >= 6 pt, style.py backbone colours/markers (VoCo marker now a diamond), Okabe-Ito method colours.
* fig_overview: Pillar-0 labelled "multi-scale" (convolutional stem + multi-scale window transformer, not a plain ViT);
  card titled "3-D CT backbones (encoder size)", sizes from the logs (87 M, 397 M incl. SPECTRE's feature combiner, 94 M,
  74 M); (b) x axis is "encoder parameters updated": late LoRA 0.2-1.0, all LoRA 0.55-2.5 (draft 0.6-2.5), attn. FT
  11-131 (draft 21-131), full FT 45-87 (draft 87-94), as fig_arms (d); (b) legend moved below the data;
  (d) method names rotated (no overlap). Values unchanged.
* fig_key: LoRA - Attn row added; three-stage intervals from label_level.csv (endpoints move by <= 0.007 vs draft);
  axis widened so that no interval is cut (classification -0.21 to 0.13).
* fig_mechanisms: the SPECTRE - TAP-CT panel moved to fig_bbgap (a); now one row (a) gain vs probe level, (b) learning
  curves (old c), (c) SPECTRE joint models (old d), 2.35 in high instead of 3.9 in; values unchanged; legend explains the (a) lines.
* fig_seg: the SPECTRE - TAP-CT panel (old d) moved to fig_bbgap (b); (a-c) unchanged; VoCo Lin mean (-0.680) drawn as a
  triangle at the axis edge.
* fig_bbgap: new appendix figure with the two between-backbone panels, neutral titles.
* fig_methods: Full FT title over the pairs that trained (TAP-CT, Pillar-0): +0.020, 71% > 0 (draft: -0.038, 50% with
  SPECTRE); SPECTRE Full FT cohort means drawn open. Other titles unchanged.
* fig_joint, fig_seg_methods, fig_seg_budget, fig_seg_convergence: values unchanged.
* fig_examples_cls: the 7-panel design described by the draft caption (the draft's PNG/PDF on disk was a stale
  5-panel version from make_example_figures.py).
* fig_examples_seg / fig_cases_seg_*: titles now print the Dice of the per-case files the metrics use (the saved
  predictions are a re-inference of the same checkpoint and differ by up to 4e-4 in the examples, up to 0.05 in the
  case figures): one example title changes (VoCo Dec in (a) 0.85 -> 0.84) and 18 case-figure titles change in the
  second decimal. Same cases selected; `\numSegEx*` unchanged.
* example_numbers.tex: written from the per-case files; 13 of 48 values differ from the draft by 0.01-0.07.

## Revision of 2026-11-02 (author feedback)

* fig_overview: (a) card title "3-D CT backbones" with an "encoder size" column (name / architecture / size), cards with
  equal gaps and identical arrows; (b) categorical axis (frozen / LoRA late blocks / LoRA all blocks / Attn-only FT /
  Full FT), parameter ranges moved to the check log as INFO lines for the caption, VoCo added as a dashed line
  (VoCo pooled minus Lin checked against `\numVocovslin<method>Meandiff`); (c, e) per-backbone markers incl. VoCo
  (checked against `\numVocorank*`, `\numVocosegrank*`), pooled mark = black bar = mean rank over the slots of all
  backbones (`\numRankall*`, `\numSegrankall*`, new in recompute.py); (d) VoCo dashed line; its Lin mean (0.058) is
  below the axis and drawn as a triangle on the lower edge with the value, not joined to the line; per-method means
  cross-checked against `label_ci.csv`. One backbone legend above (b)-(e).
* fig_gain_vs_probe (new, appendix; fig_mechanisms a): all four backbones' points; bin means unchanged (TAP-CT,
  SPECTRE, Pillar-0 pooled, `\numGainbin*`); VoCo points shown, not in the bins.
* fig_fullft_spectre (new, appendix; fig_mechanisms b, c): legend split into colour = method, line style = backbone;
  the layer-decay curve is labelled as a SPECTRE-only re-run. Writes figure_numbers.tex (values unchanged).
* fig_joint_main (new, main text): (a) per cohort and backbone, mean over labels of the label-level joint - per-task
  value (`label_level_values.csv`, key cls_joint_pertask; per label the mean over methods trained in both modes);
  grey dots: per method, mean over the cohort's labels (metrics_cls.csv); (b) the same for segmentation
  (seg_joint_pertask); grey band = AMOS22 and TotalSegmentator (24 000-step joint models). Checks: each marker equals
  the mean of its grey dots; each backbone's mean over all labels equals `\numCi{Cls,Seg}joint<bb>Meandiff`.
* fig_joint: VoCo as a fourth panel; shared cohort rows ("no runs" where a backbone has no cohort).
* fig_mechanisms is no longer in the default run (redraw with `python paper_figures.py fig_mechanisms`); its PDF stays.
