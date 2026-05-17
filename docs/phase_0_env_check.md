# Phase 0: Environment Check Report

**Date:** 2026-05-15
**Status:** COMPLETE

---

## System Overview

| Component | Value |
|-----------|-------|
| OS | Linux 5.15.0-113-generic x86_64 (container, overlay fs) |
| Python | 3.10.16 (miniconda3 env `py310`) |
| pip | 24.2 |
| Git | 2.34.1 |
| g++ | 11.4.0 |
| Disk free | ~76 GB |

## GPU & CUDA

| Component | Value |
|-----------|-------|
| GPU | **Tesla V100S-PCIE-32GB** (32 GB VRAM) |
| Compute capability | **7.0 (Volta)** |
| Driver | 580.105.08 (supports up to CUDA 13.0) |
| CUDA toolkit (nvcc) | 12.1 |
| PyTorch | 2.4.0+cu121 |
| torch.cuda.is_available() | True |

## Installed Package Versions (pinned)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.4.0+cu121 | Last major version supporting V100 (CC 7.0) |
| torchaudio | 2.4.0+cu121 | |
| torchvision | 0.19.0+cu121 | |
| mamba-ssm | 2.2.2 | Compiled from source for sm_70 |
| causal-conv1d | 1.4.0 | Compiled from source for sm_70 |
| qkan | 0.2.3dev | Editable install from qkan_upstream/ |
| transformers | 4.44.0 | Pinned for mamba-ssm 2.2.2 compat |
| triton | 3.0.0 | Bundled with PyTorch 2.4 |
| ninja | 1.13.0 | |
| einops | 0.8.2 | |

## Build Tools

| Tool | Status |
|------|--------|
| cmake | 3.22.1 |
| ninja | 1.13.0 |
| nvcc | 12.1 |
| g++ | 11.4.0 |
| packaging | 25.0 |

## Verification

```
Mamba forward pass OK: input torch.Size([2, 64, 16]) -> output torch.Size([2, 64, 16])
GPU memory used: 8 MB
```

All core dependencies installed and verified on GPU.

## Key Compatibility Notes

- V100S (CC 7.0) is the **minimum** supported by this stack. PyTorch 2.5+ drops CC 7.0.
- mamba-ssm 2.3+ requires triton >= 3.5 which pulls PyTorch 2.12 (no V100 support).
- If upgrading to 3090/4090 later, can use mamba-ssm 2.3+ with PyTorch 2.12+.
- FutureWarnings from mamba-ssm about `torch.cuda.amp` are cosmetic, not functional.

## Project Directory Structure

```
/home/project/
├── CLAUDE.md
├── RESEARCH_PROPOSAL.md
├── papers/
├── logs/
├── qkan_upstream/          # QKAN source (editable install)
├── src/
│   ├── qkan/experimental/  # Our research modules
│   └── clinical/           # Clinical-specific code
├── examples/o_qkan_mamba/  # Training/eval scripts
├── tests/                  # Unit and integration tests
├── outputs/                # Model checkpoints
├── docs/                   # Design documents
└── data/
    ├── raw/                # (gitignored)
    └── processed/          # (gitignored)
```

## Next Steps (Phase 1)

- [ ] Create `.gitignore` for data/, outputs/, etc.
- [ ] Initialize git repo
- [ ] Build baseline Spectral-QKAN-Mamba module (no clinical features)
- [ ] Write smoke test with synthetic data
- [ ] Log Phase 1 results
