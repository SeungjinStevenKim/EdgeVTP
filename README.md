# VT-Former / EdgeVTP

Trajectory prediction codebase for highway surveillance datasets, including:
- original decoder (MLP / autoregressive-style baseline)
- one-shot Bezier decoder
- residual/TCN variants for better accuracy-latency trade-offs

This repository is organized around YAML configs and experiment folders under `experiments/vehicle`.

## Highlights

- One-shot Bezier inference for low-latency prediction on NGSIM/CHD-style setups
- Config-driven training/inference (`--config <yaml>`)
- End-to-end metrics and timing logs saved per run
- Includes trained checkpoints for key NGSIM operating points (see Pre-trained Models section)

## Installation

### Option 1: Conda (recommended)

```bash
conda create -n vt_former python=3.8 -y
conda activate vt_former

# Install PyTorch for your CUDA version first
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# PyG dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-geometric==2.0.4

# Remaining packages
pip install -r requirements.txt
```

### Option 2: environment.yml

```bash
conda env create -f environment.yml
conda activate vt_former
```

If `torch==...+cuXXX` installation fails, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run `pip install -r requirements.txt`.

## Dataset

Place preprocessed datasets under:

```text
datasets/
  ngsim/
    train/
    val/
    test/
```

Most vehicle experiments in this repo use `dataset: [ngsim]` and 5Hz setups (`frame_subsample: 2`).

## How to Run

Main entry point:

```bash
python ngsim_share/main.py --config <config_path>
```

### Training example

```bash
python ngsim_share/main.py --config experiments/vehicle/train/train_ngsim_35m_oneshot_bezier_80ep_residual_v2_tcn_5hz/config.yaml
```

### Inference example

```bash
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml
```

### Optional profiling output

```bash
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml --profile_inference
```

## Output Structure

Runs are saved under:

```text
experiments/vehicle/
  train/<run_name>/
  inference/<run_name>/
```

Typical outputs include:
- `config.yaml` (resolved config snapshot)
- `<dataset>.pt` (trained checkpoint, e.g., `ngsim.pt`)
- `train_log.txt` or `inference_results.txt`

## Pre-trained Models (Included)

The following key checkpoints are already present in this repository and can be used directly.

| Operating Point | Train Folder | Checkpoint | Typical Inference Config | Notes |
|---|---|---|---|---|
| Latency-focused (`r=20, K=16, Residual=N`) | `experiments/vehicle/train/train_ngsim_20m_oneshot_bezier_80ep_k16` | `ngsim.pt` | `experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_k16_5hz/config.yaml` | Fastest E2E among key variants |
| Balanced (`r=20, K=16, Residual=Y, TCN`) | `experiments/vehicle/train/train_ngsim_20m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16` | `ngsim.pt` | `experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml` | Strong accuracy-latency balance |
| Error-focused (`r=30, K=16, Residual=Y, TCN`) | `experiments/vehicle/train/train_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16` | `ngsim.pt` | `experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml` | Best ADE/FDE among key operating points |
| Original decoder baseline (`output_type=mlp`) | `experiments/vehicle/train/train_ngsim_original_obs10_decoder_raw_1ep` | `ngsim.pt` | `experiments/vehicle/inference/inference_ngsim_original_obs10_decoder_raw_5hz/config.yaml` | Baseline for speed/accuracy comparison |

You can also use packaged artifacts in repo root:
- `VT_Former_Inference_5models.zip`
- `VT_Former_NGSIM_18variants.zip`
- `VT_Former_UNCC_30m_TCN.zip`

## Repro Tips

- Keep train/inference pair configs consistent (`model_dir`, `output_type`, `relation_neighbor_limit_meter`, `relation_neighbor_limit_k`).
- For fair latency comparison, report both:
  - per-scene latency
  - benchmark latency (many runs, synchronized)
- Use the same frame rate setting (`frame_subsample`) across compared runs.

## License

See `LICENSE`.
