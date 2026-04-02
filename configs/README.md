# Config layout

- **`ngsim/`** — Minimal NGSIM train/inference YAMLs for the bundled five-model inference story (same files as `inference_package/configs/ngsim/`). Use from repo root:  
  `python ngsim_share/main.py --config configs/ngsim/<file>.yaml`
- **`archive/`** — Older NGSIM grids (KAN, chunked, augmentation sweeps, etc.), pedestrian ETH/UCY configs, and optional HighD / `vehicles` recipe trees. Cluster scripts under `scripts/` that still refer to those paths use `configs/archive/...`.
- **`experiments/vehicle/`** — NGSIM ablation grid (20 / 30 / 40 m × K ∈ {8,12,16} × Bezier vs residual+TCN) and Carolinas (CHD) runs used in the paper; the three headline operating points in the root README are a subset (K=16).

See `archive/README.md` for detail.
