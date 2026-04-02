# Archived configs

Everything here was kept for **reproducing older sweeps, ablations, and non-NGSIM recipes**. It is not required to run the default README quickstart or the five-model `inference_package`.

| Path | Contents |
|------|-----------|
| `ngsim/` | Full historical NGSIM YAML grid (previous default `configs/ngsim/`). |
| `pedestrian/` | ETH/UCY-style pedestrian configs used by `scripts/run_all_benchmark*.slurm` and related jobs. |
| `carolinas/`, `highd/`, `vehicles/` | Optional dataset-specific grids (may be present only in your checkout). |

Slurm and shell helpers under `scripts/` that launch those jobs reference `configs/archive/...`.

The minimal, paper-facing NGSIM entry points live in **`configs/ngsim/`** at repo root (10 YAMLs) and in **`inference_package/configs/ngsim/`**.
