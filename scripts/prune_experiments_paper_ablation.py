#!/usr/bin/env python3
"""Remove experiment folders not needed for paper Table 7 NGSIM grid + CHD Carolinas.

Keeps under experiments/vehicle:
- train/inference: ngsim {20m,30m,40m} × k{8,12,16} × (bezier-only | residual_v2_tcn_5hz)
- inference: every inference_carolinas_*
- train: train_carolinas_* only if paired with an inference folder (same run stem; inference …_k16_5hz → train …_k16)

Also removes experiments/pedestrian/ (ETH/UCY etc.; not in the vehicle paper).

Deletes everything else under experiments/vehicle (highd_l40s, orphan train-only Carolinas K sweeps, etc.).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VEH = ROOT / "experiments" / "vehicle"
TRAIN = VEH / "train"
INF = VEH / "inference"


def ngsim_ablation_dirs() -> set[Path]:
    out: set[Path] = set()
    for r in ("20m", "30m", "40m"):
        for k in ("k8", "k12", "k16"):
            out.add(
                TRAIN / f"train_ngsim_{r}_oneshot_bezier_80ep_{k}"
            )
            out.add(
                TRAIN
                / f"train_ngsim_{r}_oneshot_bezier_80ep_residual_v2_tcn_5hz_{k}"
            )
            out.add(
                INF / f"inference_ngsim_{r}_oneshot_bezier_80ep_{k}_5hz"
            )
            out.add(
                INF
                / f"inference_ngsim_{r}_oneshot_bezier_80ep_residual_v2_tcn_5hz_{k}"
            )
    return out


def carolinas_train_from_inference(inference_run_name: str) -> str:
    """Map inference_carolinas_* folder name to paired train_carolinas_* (inference k16_5hz -> train k16)."""
    s = inference_run_name.replace("inference_", "train_", 1)
    if s.endswith("_5hz"):
        s = s[: -len("_5hz")]
    return s


def carolinas_dirs() -> set[Path]:
    out: set[Path] = set()
    if not INF.is_dir():
        return out
    train_names: set[str] = set()
    for p in INF.iterdir():
        if p.is_dir() and p.name.startswith("inference_carolinas_"):
            out.add(p)
            train_names.add(carolinas_train_from_inference(p.name))
    if not TRAIN.is_dir():
        return out
    for p in TRAIN.iterdir():
        if p.is_dir() and p.name.startswith("train_carolinas_") and p.name in train_names:
            out.add(p)
    return out


def main() -> None:
    keep = ngsim_ablation_dirs() | carolinas_dirs()
    dry_run = "--dry-run" in sys.argv

    removed = 0
    ped = ROOT / "experiments" / "pedestrian"
    if ped.is_dir():
        if dry_run:
            print("Would remove:", ped.relative_to(ROOT))
        else:
            shutil.rmtree(ped, ignore_errors=False)
        removed += 1
    # Top-level vehicle paths (e.g. highd_l40s/, slurm/)
    if VEH.is_dir():
        for p in VEH.iterdir():
            if p.name in ("train", "inference"):
                continue
            if dry_run:
                print("Would remove top-level:", p.relative_to(ROOT))
            else:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=False)
                else:
                    p.unlink(missing_ok=True)
            removed += 1

    for base in (TRAIN, INF):
        if not base.is_dir():
            continue
        for p in list(base.iterdir()):
            if not p.is_dir():
                continue
            if p.resolve() in keep:
                continue
            if dry_run:
                print("Would remove:", p.relative_to(ROOT))
            else:
                shutil.rmtree(p, ignore_errors=False)
            removed += 1

    print("Done." if not dry_run else "Dry run.", f"Touched {removed} paths (approx).")


if __name__ == "__main__":
    main()
