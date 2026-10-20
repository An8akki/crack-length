# Environment & Hardware Audit Report

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** 2026-10-20  
**Machine:** Windows 11 AMD64 (13th Gen Intel Core i7-13650HX)  
**Virtual Environment Path:** `C:\Project\ml_pipeline\crack-length\venv`  

---

## 1. Hardware Architecture

| Component | Specification | Details |
| :--- | :--- | :--- |
| **Processor (CPU)** | 13th Gen Intel(R) Core(TM) i7-13650HX | 14 Cores (6 Performance Cores, 8 Efficient Cores), 20 Logical Threads |
| **Host Memory (RAM)** | 15.71 GB Physical RAM | ~9.8 GB Available at baseline start |
| **Dedicated GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | Driver 581.86, System CUDA Support 13.0, PCIe WDDM |
| **Integrated GPU** | Intel(R) UHD Graphics | Display controller |
| **Operating System** | Windows 11 Home Single Language | Version 10.0.26100 AMD64 |

---

## 2. Software & Python Environment

The environment was established using Python 3.12 within a dedicated virtual environment (`venv`) using the pinned specification in `SDNET2021_Audit/handover/environment_requirements.txt`:

```text
torch==2.8.0
torchvision==0.23.0
numpy==2.2.6
pandas==2.3.2
pillow==11.3.0
scipy==1.16.1
scikit-learn==1.7.2
scikit-image==0.25.2
tqdm==4.67.1
```

### Verified Installed Packages:
- `python`: `3.12.0` (tags/v3.12.0:0fb18b0, Oct 2 2023, MSC v.1935 64-bit AMD64)
- `torch`: `2.8.0+cpu`
- `torchvision`: `0.23.0+cpu`
- `numpy`: `2.2.6`
- `pandas`: `2.3.2`
- `pillow`: `11.3.0`
- `scipy`: `1.16.1`
- `scikit-learn`: `1.7.2`
- `scikit-image`: `0.25.2`
- `tqdm`: `4.67.1`

---

## 3. PyTorch Execution Configuration & Reproducibility Policy

1. **Execution Backend:**
   - `torch.cuda.is_available()` returns `False` within the environment because PyTorch is pinned to the official CPU build (`2.8.0+cpu`).
   - This strictly adheres to the baseline reproducibility protocol established on the previous host (`Intel Core i7-11th Gen`, CPU execution, Intel MKL/OpenMP backend).
   - CPU intra-op thread count: `torch.get_num_threads() = 14`.
2. **Precision Policy:**
   - Mixed precision (AMP / FP16 / BF16) is **disabled**.
   - All network forward passes, loss computations, and metrics operate strictly in single-precision floating point (FP32).
3. **Seeding Policy:**
   - Deterministic seeds `[42, 123, 999]` configure:
     * `random.seed(seed)`
     * `np.random.seed(seed)`
     * `torch.manual_seed(seed)`
   - Seeding ensures identical pseudo-random initialization, batch shuffling sequences, and DataLoader sampling across runs.
   - `torch.use_deterministic_algorithms(False)` is used because PyTorch CPU convolution and adaptive pooling primitives lack bitwise deterministic implementations across differing CPU microarchitectures, but mathematical convergence and optimization behavior remain fully controlled.
