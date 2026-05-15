# Phase 1 Setup Log

**Date:** 2026-05-15
**Branch:** feature/o-qkan-mamba
**Status:** COMPLETE

---

## Commands Run

```bash
# 1. Clone QKAN
git clone https://github.com/Jim137/qkan.git /home/project/qkan_upstream

# 2. Init git repo and create branch
git init
git checkout -b feature/o-qkan-mamba

# 3. Upgrade build tools
pip install --upgrade pip setuptools wheel ninja packaging

# 4. PyTorch (already installed, upgraded during mamba-ssm resolution)
# Final: torch 2.4.0+cu121

# 5. Install QKAN editable
pip install -e /home/project/qkan_upstream

# 6. Install Mamba
apt-get install -y g++
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.2 --no-build-isolation
pip install transformers==4.44.0  # pinned for mamba-ssm compat

# 7. Clinical and testing dependencies
pip install fhir.resources pandas pyarrow tqdm wandb scikit-learn
pip install networkx rdflib
pip install pytest pytest-cov

# 8. Commit scaffolding
git add ... && git commit -m "Add O-QKAN-Mamba project scaffolding"
```

## Installed Versions

| Package | Version |
|---------|---------|
| pip | 26.1.1 |
| setuptools | 82.0.1 |
| wheel | 0.47.0 |
| ninja | 1.13.0 |
| packaging | 26.2 |
| torch | 2.4.0+cu121 |
| torchaudio | 2.4.0+cu121 |
| torchvision | 0.19.0+cu121 |
| triton | 3.0.0 |
| mamba-ssm | 2.2.2 |
| causal-conv1d | 1.4.0 |
| qkan | 0.2.3dev |
| transformers | 4.44.0 |
| einops | 0.8.2 |
| fhir.resources | 8.2.0 |
| pandas | 2.3.3 |
| pyarrow | 24.0.0 |
| scikit-learn | 1.7.2 |
| networkx | 3.2.1 |
| rdflib | 7.6.0 |
| wandb | 0.27.0 |
| pytest | 9.0.3 |
| pytest-cov | 7.1.0 |

## Mamba Availability

**Available: YES**
- mamba-ssm 2.2.2 compiled from source for V100S (sm_70)
- causal-conv1d 1.4.0 compiled from source
- Forward pass verified on GPU

## Errors Encountered & Resolutions

1. **g++ missing** — `apt-get install g++` resolved
2. **mamba-ssm 2.3.2 pulled PyTorch 2.12** — PyTorch 2.12 dropped V100 (CC 7.0) support.
   Resolution: pinned mamba-ssm==2.2.2 + PyTorch 2.4.0+cu121
3. **transformers 5.8.1 incompatible with mamba-ssm 2.2.2** — removed API that mamba-ssm imports.
   Resolution: pinned transformers==4.44.0
4. **pip build isolation** pulled wrong PyTorch into build env.
   Resolution: used `--no-build-isolation` for CUDA packages

## Import Check Results

```
torch: 2.4.0+cu121 | CUDA: True
qkan: 0.2.3dev
mamba_ssm: 2.2.2
fhir.resources: 8.2.0
networkx: 3.2.1
pandas: 2.3.3
scikit-learn: 1.7.2
rdflib: 7.6.0
pytest: 9.0.3
All imports OK. Mamba available: True
```

## Git Status

- Repo initialized at `/home/project/`
- Branch: `feature/o-qkan-mamba`
- Initial commit: `2cb0997` — "Add O-QKAN-Mamba project scaffolding"
