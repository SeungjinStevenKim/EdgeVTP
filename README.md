# VT-Former

Vehicle trajectory prediction model with Original VT-Former, Bezier VT-Former, and KAN VT-Former variants.

## Installation

### Option 1: Conda (recommended)

```bash
# Create environment
conda create -n vt_former python=3.8 -y
conda activate vt_former

# Install PyTorch (CUDA 11.6 - adjust for your system: https://pytorch.org/get-started/locally/)
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install PyTorch Geometric and dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-geometric==2.0.4

# Install remaining requirements
pip install -r requirements.txt
```

### Option 2: environment.yml (from working env)

```bash
conda env create -f environment.yml
conda activate vt_former
```

### Option 3: requirements.txt only

```bash
pip install -r requirements.txt
```

Note: `torch==1.12.0+cu116` may fail with plain `pip install`. Install PyTorch first from [pytorch.org](https://pytorch.org) for your CUDA version, then run `pip install -r requirements.txt`.

## Quick Start

```bash
# Training
python main.py --config configs/ngsim/train_ngsim_30m_oneshot_bezier_80ep.yaml

# Inference
python main.py --config configs/ngsim/inference_ngsim_30m_oneshot_bezier_80ep.yaml
```

## Dataset

Download NGSIM preprocessed data and place in `datasets/ngsim/` (see project documentation for links).
