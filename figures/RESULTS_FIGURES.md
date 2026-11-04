# Results figures (figures/results_figures.py)

Every figure reads only the recomputed outputs of `metrics/recompute.py`; nothing is typed by hand. Style from `figures/style.py` (Okabe-Ito palette, one colour and marker per backbone). Explanations live in the LaTeX captions.

| Figure | Inputs |
|---|---|
| `fig_key` | `metrics/out/label_level.csv` |
| `fig_methods_cls_rank` | `metrics/out/ranks.csv`, `metrics/out/label_level.csv`, `metrics/out/spread.csv` |
| `fig_methods_seg_rank` | `metrics/out/ranks.csv`, `metrics/out/label_level.csv` |
| `fig_perlabel_cls` | `metrics/out/metrics_cls.csv`, `metrics/out/paired_cls.csv` |
| `fig_perstructure_seg` | `metrics/out/metrics_seg.csv` |
| `fig_joint` | `metrics/out/cohort_means.csv` |
| `fig_seeds` | `metrics/out/seeds.csv` |
| `fig_convergence` | `metrics/out/convergence.csv` |
| `fig_revision` | `metrics/out/cohort_means.csv`, `metrics/out/metrics_cls.csv`, `metrics/out/metrics_seg.csv`, `metrics/out/revision.csv` |
| `fig_voco` | `metrics/out/label_level.csv` |
