"""Analyze model size and compute for deployment feasibility."""
import os
import sys
from pathlib import Path

import torch

pt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/micro_macro_recognition/v2_testkevin/tcn/models/ds_ms_tcn.pt")
file_size_mb = os.path.getsize(pt_path) / 1024 / 1024
print(f"Model file size: {file_size_mb:.2f} MB")

ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
if isinstance(state, dict) and any(isinstance(v, dict) for v in state.values()):
    # Flatten nested dicts
    flat = {}
    for k, v in state.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"{k}.{k2}"] = v2
        else:
            flat[k] = v
    state = flat
total_params = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
print(f"Total parameters: {total_params:,}")
print(f"Param memory (float32): {total_params * 4 / 1024 / 1024:.2f} MB")
print(f"Param memory (int8):    {total_params / 1024 / 1024:.2f} MB")
print()

layer_sizes = {}
for k, v in state.items():
    if not isinstance(v, torch.Tensor):
        continue
    prefix = k.split(".")[0]
    layer_sizes[prefix] = layer_sizes.get(prefix, 0) + v.numel()
print("Per-component parameter counts:")
for name, count in sorted(layer_sizes.items(), key=lambda x: -x[1]):
    print(f"  {name:30s} {count:>10,} params ({count * 4 / 1024:.1f} KB)")

cfg = ckpt.get("config", {})
mm = cfg.get("micro_macro", {}) if cfg else {}
print(f"\nArchitecture config:")
print(f"  num_filters:      {mm.get('num_filters', '?')}")
print(f"  num_layers:       {mm.get('num_layers', '?')}")
print(f"  kernel_size:      {mm.get('kernel_size', '?')}")
print(f"  num_macro_stages: {mm.get('num_macro_stages', '?')}")
print(f"  causal:           {mm.get('causal', '?')}")

# Estimate FLOPs for single-sample inference
# DS-MS-TCN: Stage1 (dilated TCN) + Stage2 (macro TCN) + refinement stages
# For causal conv: FLOPs per sample ~ 2 * in_ch * out_ch * kernel_size per layer
num_filters = int(mm.get("num_filters", 64))
num_layers = int(mm.get("num_layers", 6))
kernel_size = int(mm.get("kernel_size", 3))
num_macro_stages = int(mm.get("num_macro_stages", 3))
input_ch = 6  # IMU channels

# Stage 1 micro: input_conv (6->F) + L dilated layers (F->F) + output_conv (F->3)
stage1_flops = 2 * input_ch * num_filters * kernel_size  # input conv
stage1_flops += num_layers * 2 * num_filters * num_filters * kernel_size  # dilated layers
stage1_flops += 2 * num_filters * 3  # micro output
# Stage 2 macro: 3->F + L dilated (F->F) + F->5
micro_classes = 3
macro_classes = 5
stage2_flops = 2 * micro_classes * num_filters * kernel_size
stage2_flops += num_layers * 2 * num_filters * num_filters * kernel_size
stage2_flops += 2 * num_filters * macro_classes
# Refinement stages (same as stage2)
refine_flops = (num_macro_stages - 1) * stage2_flops

total_flops = stage1_flops + stage2_flops + refine_flops
print(f"\nEstimated FLOPs per sample (100 Hz):")
print(f"  Stage 1 (micro):   {stage1_flops:>10,} FLOPs")
print(f"  Stage 2 (macro):   {stage2_flops:>10,} FLOPs")
print(f"  Refinement ({num_macro_stages-1}x):  {refine_flops:>10,} FLOPs")
print(f"  Total per sample:  {total_flops:>10,} FLOPs")
print(f"  At 100 Hz:         {total_flops * 100 / 1e6:.2f} MFLOPS/s")
print(f"  At 100 Hz:         {total_flops * 100 / 1e9:.4f} GFLOPS/s")

# LuckFox Pico Zero specs
print(f"\n--- LuckFox Pico Zero Feasibility ---")
print(f"  RV1103: ARM Cortex-A7 1.2GHz + 0.5 TOPS NPU + 64 MB DDR2")
print(f"  NPU capacity: 0.5 TOPS = 500 GOPS/s (int8)")
print(f"  Model int8 size: {total_params / 1024 / 1024:.2f} MB (fits in 64 MB RAM)")
print(f"  Compute ratio: {total_flops * 100 / 500e9 * 100:.4f}% of NPU capacity at 100 Hz")
print(f"  CPU-only estimate: ~{total_flops * 100 / 1e9 / 1.0 * 1000:.1f} ms/s on Cortex-A7 @ 1 GFLOP/s")
