"""
Isolated GPU Diagnostic and Smoke-Test Suite for Deep CORAL
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Performs:
  1. PyTorch CUDA capability & device identification.
  2. Device placement verification for DeepCORALUNet, tensors, optimizer, and loss.
  3. Real Strategy C patch dimensions (B=8, C=1, H=256, W=256).
  4. End-to-end forward pass, bottleneck feature covariance alignment, and backward pass on CUDA.
  5. Peak GPU memory tracking (VRAM allocation and reservation).
  6. Direct CPU vs GPU speed benchmark across 10 identical training iterations.
  7. Zero pollution of benchmark outputs (outputs only to reports/gpu_smoke_test_results.json).
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASELINE_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b2_source_only_baseline")
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

from src.domain_adaptation.deep_coral import DeepCORALUNet, coral_loss


def run_gpu_diagnostic_and_benchmark(num_benchmark_steps=10):
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tests": {}
    }

    print("=" * 70)
    print("DEEP CORAL GPU DIAGNOSTIC & BENCHMARK SUITE")
    print(f"PyTorch Version: {results['pytorch_version']}")
    print(f"CUDA Available:  {results['cuda_available']}")
    print(f"CUDA Build:      {results['cuda_version']}")
    print(f"Device Name:     {results['device_name']}")
    print("=" * 70)

    if not torch.cuda.is_available():
        results["tests"]["cuda_available_check"] = "FAILED: torch.cuda.is_available() is False"
        print("ERROR: CUDA is not available in this environment!")
        return results

    device = torch.device("cuda:0")

    # --- TEST 1: Tensor and Model Device Placement ---
    print("\n[TEST 1] Verifying Model and Tensor CUDA Placement...")
    torch.manual_seed(42)
    model = DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64]).to(device)
    
    all_params_on_cuda = all(p.is_cuda for p in model.parameters())
    results["tests"]["model_on_cuda"] = all_params_on_cuda
    print(f"  Model parameters on CUDA: {all_params_on_cuda} (Count: {sum(p.numel() for p in model.parameters()):,})")

    # Real Strategy C batch dimensions: 8 patches, 1 channel, 256x256
    src_img = torch.randn(8, 1, 256, 256, device=device)
    src_mask = torch.randint(0, 3, (8, 256, 256), device=device)
    tgt_img = torch.randn(8, 1, 256, 256, device=device)

    results["tests"]["tensors_on_cuda"] = (src_img.is_cuda and src_mask.is_cuda and tgt_img.is_cuda)
    print(f"  Source image on CUDA: {src_img.is_cuda} | Shape: {list(src_img.shape)}")
    print(f"  Target image on CUDA: {tgt_img.is_cuda} | Shape: {list(tgt_img.shape)}")

    # --- TEST 2: Forward, Bottleneck Covariance, and Backward on CUDA ---
    print("\n[TEST 2] Verifying End-to-End Forward, CORAL Loss, and Backward on CUDA...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)

    src_logits, src_feats = model(src_img, return_features=True, feature_layer="bottleneck")
    _, tgt_feats = model(tgt_img, return_features=True, feature_layer="bottleneck")

    loss_seg = criterion(src_logits, src_mask)
    loss_coral = coral_loss(src_feats, tgt_feats)
    total_loss = loss_seg + 0.5 * loss_coral

    total_loss.backward()
    optimizer.step()

    all_grads_on_cuda = all(p.grad.is_cuda for p in model.parameters() if p.grad is not None)
    results["tests"]["forward_backward_on_cuda"] = all_grads_on_cuda
    results["tests"]["total_loss_value"] = float(total_loss.item())
    results["tests"]["seg_loss_value"] = float(loss_seg.item())
    results["tests"]["coral_loss_value"] = float(loss_coral.item())

    vram_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    vram_res_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    vram_max_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    results["vram_usage"] = {
        "allocated_mb": round(vram_alloc_mb, 2),
        "reserved_mb": round(vram_res_mb, 2),
        "peak_mb": round(vram_max_mb, 2)
    }

    print(f"  Forward-backward executed cleanly. Total Loss: {total_loss.item():.4f}")
    print(f"  Gradients on CUDA: {all_grads_on_cuda}")
    print(f"  VRAM Allocated: {vram_alloc_mb:.2f} MB | Peak: {vram_max_mb:.2f} MB | Reserved: {vram_res_mb:.2f} MB")

    # --- TEST 3: Speed Comparison (CPU vs GPU over num_benchmark_steps) ---
    print(f"\n[TEST 3] Benchmarking {num_benchmark_steps} Iterations: CPU vs GPU...")

    # GPU Benchmark
    torch.cuda.synchronize()
    t_gpu_0 = time.time()
    for _ in range(num_benchmark_steps):
        optimizer.zero_grad()
        s_logits, s_feats = model(src_img, return_features=True, feature_layer="bottleneck")
        _, t_feats = model(tgt_img, return_features=True, feature_layer="bottleneck")
        l_s = criterion(s_logits, src_mask)
        l_c = coral_loss(s_feats, t_feats)
        loss = l_s + 0.5 * l_c
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t_gpu = time.time() - t_gpu_0

    # CPU Benchmark
    cpu_model = DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64]).to("cpu")
    cpu_opt = torch.optim.AdamW(cpu_model.parameters(), lr=1e-3, weight_decay=1e-4)
    cpu_crit = nn.CrossEntropyLoss(ignore_index=255)
    src_img_cpu = src_img.cpu()
    src_mask_cpu = src_mask.cpu()
    tgt_img_cpu = tgt_img.cpu()

    t_cpu_0 = time.time()
    for _ in range(num_benchmark_steps):
        cpu_opt.zero_grad()
        s_logits, s_feats = cpu_model(src_img_cpu, return_features=True, feature_layer="bottleneck")
        _, t_feats = cpu_model(tgt_img_cpu, return_features=True, feature_layer="bottleneck")
        l_s = cpu_crit(s_logits, src_mask_cpu)
        l_c = coral_loss(s_feats, t_feats)
        loss = l_s + 0.5 * l_c
        loss.backward()
        cpu_opt.step()
    t_cpu = time.time() - t_cpu_0

    speedup = t_cpu / t_gpu if t_gpu > 0 else 0.0

    results["benchmark"] = {
        "steps": num_benchmark_steps,
        "batch_size_source": 8,
        "batch_size_target": 8,
        "cpu_duration_sec": round(t_cpu, 3),
        "cpu_per_step_sec": round(t_cpu / num_benchmark_steps, 4),
        "gpu_duration_sec": round(t_gpu, 3),
        "gpu_per_step_sec": round(t_gpu / num_benchmark_steps, 4),
        "speedup_factor": round(speedup, 2)
    }

    print(f"  CPU Duration: {t_cpu:.2f}s ({t_cpu/num_benchmark_steps:.3f}s / step)")
    print(f"  GPU Duration: {t_gpu:.2f}s ({t_gpu/num_benchmark_steps:.3f}s / step)")
    print(f"  GPU Speedup:  {speedup:.2f}x faster than CPU")
    print("=" * 70)

    # Save results safely
    out_json = os.path.join(PROJECT_ROOT, "reports", "gpu_smoke_test_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved diagnostic report to: {out_json}")

    return results


if __name__ == "__main__":
    run_gpu_diagnostic_and_benchmark(num_benchmark_steps=10)
