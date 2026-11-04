#!/usr/bin/env python3
"""Peak GPU memory and time of a training step, per backbone and method, from the run configs.

    python measure_memory.py --out memory.csv [--steps 4]

Classification: joint config of FDG-PET/CT (TAP-CT, SPECTRE) and RSNA 2019 ICH (Pillar-0); the input is the largest of
the first 16 training samples of the cohort (by voxel count), the same for every method of a backbone, at batch size 1.
Segmentation: AMOS22 joint config (15 structures), a micro-batch of two crops of the configured roi_size (zeros; memory
does not depend on the content). One step = bf16 autocast forward, a surrogate loss on the logits, backward and an
AdamW step over the trainable parameters, as in training. Two-stage methods are measured with their stage-2 model
(probe initialization cleared). Peak memory = torch.cuda.max_memory_allocated over the steps, including the weights.
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import torch

AIFO = Path("/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo")
sys.path.insert(0, str(AIFO))
from aifofomo.config.loader import build_from_config  # noqa: E402
from aifofomo.config.registry import initialize_registry  # noqa: E402
from aifofomo.config.settings import load_yaml_config_from_path  # noqa: E402
from aifofomo.engine.engine_config import DefaultConfig  # noqa: E402

FP = AIFO / "examples" / "fm_probes"
CLS = {"tapct": FP / "fm_adapt/fdg/tapct/joint", "spectre": FP / "fm_adapt/fdg/spectre/joint",
       "pillar0": FP / "fm_adapt/rsna2019_ich/pillar0/joint"}
SEG = {"tapct": FP / "fm_seg/amos22/tapct/joint", "spectre": FP / "fm_seg/amos22/spectre/joint"}
CLS_METHODS = ["lin", "attn", "lora", "lorag", "babmil", "dora", "rslora", "lin_then_lorag", "attn_then_lorag", "attnft", "fullft"]
SEG_METHODS = ["lin", "dec", "lora", "lorag", "dora", "fullft", "dec_then_lora"]


def load_cfg(path: Path):
    cfg = load_yaml_config_from_path(path, DefaultConfig)
    m = cfg.model
    if getattr(m, "probe_init_checkpoint", None):
        m.probe_init_checkpoint = None
        if hasattr(m, "probe_init_max_iteration"):
            m.probe_init_max_iteration = None
    return cfg


def measure(model, make_batch, out_key, steps, dev):
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-6)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
    times = []
    for _ in range(steps):
        batch = make_batch()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(data=batch)
        logits = out[out_key] if isinstance(out, dict) else out
        loss = logits.float().square().mean()
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    n_tr = sum(p.numel() for p in params)
    del opt, params
    return peak, sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else times[0], n_tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--only", default="cls,seg")
    a = ap.parse_args()
    initialize_registry()
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(dev)
    rows = []
    if "cls" in a.only:
        for bb, root in CLS.items():
            cfg0 = load_cfg(root / "lin" / "config.yaml")
            ds = build_from_config(cfg0.trainer.data.training_datasets[0])
            cand = [ds[i]["data"] for i in range(min(16, len(ds)))]
            x = max(cand, key=lambda t: t.numel()).unsqueeze(0)
            print(f"cls {bb}: input {tuple(x.shape)}", flush=True)
            for mt in CLS_METHODS:
                try:
                    model = build_from_config(load_cfg(root / mt / "config.yaml").model)
                    peak, sec, n_tr = measure(model, lambda: x.to(dev), "logits", a.steps, dev)
                    rows.append(dict(task="cls", backbone=bb, method=mt, input=str(tuple(x.shape)), peak_gib=round(peak, 3),
                                     s_per_step=round(sec, 4), trainable=n_tr, gpu=gpu, status="ok"))
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(task="cls", backbone=bb, method=mt, gpu=gpu, status=f"{type(e).__name__}: {e}"[:300]))
                print(rows[-1], flush=True)
                model = None; gc.collect(); torch.cuda.empty_cache()
    if "seg" in a.only:
        for bb, root in SEG.items():
            for mt in SEG_METHODS:
                try:
                    cfg = load_cfg(root / mt / "config.yaml")
                    roi = tuple(cfg.model.roi_size)
                    model = build_from_config(cfg.model)
                    peak, sec, n_tr = measure(model, lambda: torch.zeros((2, 1) + roi, device=dev), "seg_logits", a.steps, dev)
                    rows.append(dict(task="seg", backbone=bb, method=mt, input=str((2, 1) + roi), peak_gib=round(peak, 3),
                                     s_per_step=round(sec, 4), trainable=n_tr, gpu=gpu, status="ok"))
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(task="seg", backbone=bb, method=mt, gpu=gpu, status=f"{type(e).__name__}: {e}"[:300]))
                print(rows[-1], flush=True)
                model = None; gc.collect(); torch.cuda.empty_cache()
    keys = ["task", "backbone", "method", "input", "peak_gib", "s_per_step", "trainable", "gpu", "status"]
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
