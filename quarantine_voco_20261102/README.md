# VoCo quarantine (2026-11-02)

VoCo was taken out of the paper and of the metric pipeline on 2026-11-02. Nothing was deleted; this folder holds
what was in use with VoCo, so the VoCo version can be restored.

| Item | Content |
|---|---|
| `run_inventory_with_voco.csv` | `report/run_inventory.csv` before the quarantine. In the live inventory, the 491 VoCo rows that were not yet excluded now have `excluded = voco_quarantined_20261102`; no other cell changed. |
| `metrics_out_with_voco/` | `metrics/out/` computed with VoCo (numbers.tex, CSVs; the per-case cache was left in `metrics/out/_cache`). |
| `paper_voco/` | The paper with VoCo (four encoders), as it was on 2026-11-02. |
| `figures_out_voco/` | `figures/out/` drawn with VoCo. |
| `paper_tables_voco/` | VoCo-only table files that the VoCo-free paper does not use. |

The raw VoCo runs stay where they were (`../../midl_paper/results/`); the older stale VoCo runs remain in
`../../midl_paper/results/raw_voco_stale_quarantine_20261027/`.

Restore: copy `run_inventory_with_voco.csv` back to `report/run_inventory.csv`, run `metrics/recompute.py`, then the
table and figure scripts with `WITH_VOCO=1`, and build `paper_voco/`.
