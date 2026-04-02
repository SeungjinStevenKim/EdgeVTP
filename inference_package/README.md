# VT-Former NGSIM Inference Package (5 Models)

Vehicle trajectory prediction on NGSIM. Includes 5 pre-trained models with different architectures.

## Models

| Model | Config | ADE | FDE | Params |
|-------|--------|-----|-----|--------|
| Original VT-Former (30m, MLP) | inference_ngsim_30m_80ep_5hz | 2.69 | 5.97 | 134.5K |
| Bezier VT-Former (30m) | inference_ngsim_30m_oneshot_bezier_80ep_5hz | 2.09 | 4.88 | 134.5K |
| Bezier + Residual v2 (30m) | inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_5hz | 1.85 | 4.18 | 156.7K |
| Bezier + Residual v2 + TCN (30m) | inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz | 1.86 | 4.28 | 145.9K |
| Bezier + Residual v2 + TCN (35m, K=12) | inference_ngsim_35m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k12_L2H2 | 1.87 | 4.29 | 145.9K |

## Setup

**Python 3.8+** recommended. Install PyTorch Geometric for your PyTorch/CUDA version.

### 1. Install PyTorch

```bash
# CUDA 11.6
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 2. Install PyTorch Geometric

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-geometric==2.0.4
```

### 3. Install remaining requirements

```bash
pip install -r requirements.txt
```

## Data

1. **Download** NGSIM preprocessed data:  
   [Google Drive](https://drive.google.com/file/d/16xKlIgvZQrpi0Wm6sPpKyhGjQRPIwFW0/view)

2. **Extract** under `datasets/ngsim/`:
   - `train/` — training data
   - `val/` — validation data
   - `test/` — test data

3. **Format**: Tab-delimited `.txt` files. Each line: `frame_id  ped_id  x  y`

## Inference

### Single model

```bash
python main.py --config configs/ngsim/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_5hz.yaml
```

### All 5 models

```bash
bash run_all_inference.sh
```

### Config list

- `configs/ngsim/inference_ngsim_30m_80ep_5hz.yaml`
- `configs/ngsim/inference_ngsim_30m_oneshot_bezier_80ep_5hz.yaml`
- `configs/ngsim/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_5hz.yaml`
- `configs/ngsim/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz.yaml`
- `configs/ngsim/inference_ngsim_35m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k12_L2H2.yaml`

Results are saved to `experiments/vehicle/inference/<run_name>/inference_results.txt`.

## Package Contents

`configs/ngsim/` matches the **10 YAMLs** in **`configs/ngsim/`** at the repository root (minimal five-model set).

| Path | Description |
|------|-------------|
| `main.py` | Entry point |
| `utils/` | network, loader, bezier, util, trajectories, gin_conv2 |
| `configs/ngsim/` | Minimal inference + train configs (5 + 5) |
| `experiments/vehicle/train/*/ngsim.pt` | Pre-trained weights |
| `experiments/vehicle/NGSIM_5_BENCHMARK_RESULTS.md` | Benchmark table |
| `run_all_inference.sh` | Run all 5 models |
