# Config layout

- **`ngsim/`** — Minimal NGSIM train/inference YAMLs for the bundled five-model inference story (same files as `inference_package/configs/ngsim/`). Use from repo root:  
  `python ngsim_share/main.py --config configs/ngsim/<file>.yaml`
- **`archive/`** — Older NGSIM grids (KAN, chunked, augmentation sweeps, etc.), pedestrian ETH/UCY configs, and optional Carolinas / HighD / `vehicles` recipe trees. Cluster scripts under `scripts/` that still refer to those paths use `configs/archive/...`.
- **Paper checkpoints (K=16, 5 Hz)** — Only the three NGSIM operating points are versioned under `experiments/vehicle/` (six folders: three `train_*` + three `inference_*`). See the root README “Pre-trained Models” table.

See `archive/README.md` for detail.
