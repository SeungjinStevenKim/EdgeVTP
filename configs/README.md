# Config layout

- **`ngsim/`** — Minimal NGSIM train/inference YAMLs for the bundled five-model inference story (same files as `inference_package/configs/ngsim/`). Use from repo root:  
  `python ngsim_share/main.py --config configs/ngsim/<file>.yaml`
- **`archive/`** — Older NGSIM grids (KAN, chunked, augmentation sweeps, etc.), pedestrian ETH/UCY configs, and optional Carolinas / HighD / `vehicles` recipe trees. Cluster scripts under `scripts/` that still refer to those paths use `configs/archive/...`.
- **Paper checkpoints (K=16, 5 Hz, etc.)** — Canonical snapshots are the `config.yaml` files under `experiments/vehicle/train/...` and `experiments/vehicle/inference/...` (see the root README “Pre-trained Models” table).

See `archive/README.md` for detail.
