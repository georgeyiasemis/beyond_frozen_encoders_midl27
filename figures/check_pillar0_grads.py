#!/usr/bin/env python3
"""Which Pillar-0 (HeadCT) parameters receive a gradient in the classification forward pass?

    ~/miniconda3/envs/aifofomo-fm-probe/bin/python check_pillar0_grads.py   (needs timm; CPU, ~2 min)

Builds the remote CLIPMultimodalAtlas from its config (random init: the gradient pattern does not depend on the
weights), unfreezes everything, runs `extract_vision_feats` on a random 11 x 128 x 256 x 256 input and backpropagates a
scalar. Prints parameters per module group and how many have a non-zero gradient. Result (2026-11-01): 44,843,456 of
94,144,193; per-scale blocks 2 of stage 2 and 1-2 of stage 3, their position embeddings, and the text projection get
none. Used by DIAGRAMS.md to justify the Pillar-0 counts in fig_arms (d); diagrams.py recomputes the same split from
the safetensors header.
"""
import collections
import re

import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

P = "/projects/cvpr2026_challenge_fomo4ct/pretrained/Pillar0-HeadCT"
m = get_class_from_dynamic_module("modeling_clip_mmatlas.CLIPMultimodalAtlas", P)(AutoConfig.from_pretrained(P, trust_remote_code=True))
m.train()
for p in m.parameters():
    p.requires_grad_(True)
out = m.extract_vision_feats({"brain_ct": torch.rand(1, 11, 128, 256, 256)})
out.pow(2).sum().backward()
c, g = collections.Counter(), collections.Counter()
for n, p in m.named_parameters():
    key = re.sub(r"(atlas_models\.\d+)\.blocks\.\d+\.(blocks\.\d+|posemb\.\d+).*", r"\1.*.\2", n)
    key = key if "atlas_models" in key else ".".join(n.split(".")[:3])
    c[key] += p.numel()
    if p.grad is not None and p.grad.abs().sum() > 0:
        g[key] += p.numel()
for k in sorted(c):
    print(f"{k:50s} {c[k]:>10d} {g[k]:>10d}")
print("total", sum(c.values()), "with gradient", sum(g.values()))
