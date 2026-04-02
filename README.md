# VT-Former / EdgeVTP

Trajectory prediction codebase for highway surveillance datasets, including:
- one-shot Bezier decoding
- residual/TCN variants for better accuracy–latency trade-offs

This repository is organized around YAML configs and experiment folders under `experiments/vehicle`.

## Highlights

- One-shot Bezier inference for low-latency prediction on NGSIM/CHD-style setups
- Config-driven training/inference (`--config <yaml>`)
- End-to-end metrics and timing logs saved per run
- Includes trained checkpoints for key NGSIM operating points (see Pre-trained Models section)

## Representative figure (paper)

![EdgeVTP / VT-Former architecture](docs/figures/edgevtp_architecture.png)

This PNG is exported from the paper figure for GitHub rendering (PDF is not shown inline). Source PDF if you have the paper tree locally: `CVPR___EVW_2026/figures/edgevtp_architecture.pdf`.

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
python ngsim_share/main.py --config experiments/vehicle/train/train_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml
```

### Inference example

```bash
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml
```

### Optional profiling output

```bash
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml --profile_inference
```

## Config files

- **`configs/ngsim/`** — Small, stable set of YAMLs for the bundled **five NGSIM inference** models (and their train recipes). Same filenames as `inference_package/configs/ngsim/`.
- **`experiments/vehicle/.../config.yaml`** — Snapshots for **paper checkpoints** (see the pre-trained table below).
- **`configs/archive/`** — Older NGSIM sweeps, pedestrian configs, and optional Carolinas / HighD grids. Cluster scripts under `scripts/` point here when they need old paths.

See `configs/README.md` for a short index.

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

### Paper: sharing the three main checkpoints

For publication and collaborators, the three operating points we emphasize are **20 m / K=16 one-shot Bezier (fastest)**, **20 m / K=16 + residual + TCN (balanced)**, and **30 m / K=16 + residual + TCN (best error)**. Each is a single file **`ngsim.pt`** next to that run’s `config.yaml` under `experiments/vehicle/train/...` (see table above).

**Why `*.pt` is not in plain Git:** `.gitignore` excludes `*.pt` (size + GitHub limits). Ship weights via **GitHub Releases**, **Google Drive**, or similar, not the default commit stream.

**Minimal zip layout to share** (after unzip at the **repository root**, paths must match `model_dir` in the inference YAMLs):

```text
experiments/vehicle/train/train_ngsim_20m_oneshot_bezier_80ep_k16/ngsim.pt
experiments/vehicle/train/train_ngsim_20m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/ngsim.pt
experiments/vehicle/train/train_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/ngsim.pt
```

**Recipient setup:** unzip into the same VT-Former tree (or clone first, then extract so the three paths above exist). Keep the **preprocessed NGSIM** layout under `datasets/ngsim/` as in the Dataset section.

### Paper: running inference (three models)

From the repo root, with CUDA if available:

```bash
# 1) Latency-focused — one-shot Bezier, 20 m, K=16 (no residual)
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_k16_5hz/config.yaml

# 2) Balanced — same radius, residual v2 + TCN, K=16
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml

# 3) Error-focused — 30 m, residual v2 + TCN, K=16
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml
```

Optional timing breakdown:

```bash
python ngsim_share/main.py --config experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml --profile_inference
```

Run all three in sequence:

```bash
bash scripts/run_inference_paper_3models.sh
```

**Outputs:** each run writes under `experiments/vehicle/inference/<run_name>/` (e.g. `inference_results.txt` and a copy of the resolved `config.yaml`). The `run_name` is defined inside each inference `config.yaml`.

**If `FileNotFoundError` for `ngsim.pt`:** the checkpoint path is `os.path.join(model_dir, "ngsim.pt")` with `model_dir` from the YAML—verify the three `ngsim.pt` files sit in the train folders in the table at the top of this section.

## Repro Tips

- Keep train/inference pair configs consistent (`model_dir`, `output_type`, `relation_neighbor_limit_meter`, `relation_neighbor_limit_k`).
- For fair latency comparison, report both:
  - per-scene latency
  - benchmark latency (many runs, synchronized)
- Use the same frame rate setting (`frame_subsample`) across compared runs.

## License

See `LICENSE`.
