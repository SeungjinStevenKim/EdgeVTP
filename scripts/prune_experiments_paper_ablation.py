#!/usr/bin/env python3
"""Remove experiments/vehicle run folders not needed for paper Table 7 NGSIM grid + CHD Carolinas.

Keeps:
- train/inference: ngsim {20m,30m,40m} × k{8,12,16} × (bezier-only | residual_v2_tcn_5hz)
- train/inference: names starting with train_carolinas_ / inference_carolinas_

Deletes everything else under experiments/vehicle (including highd_l40s, benchmark_*, _v2, 35m, etc.).
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


def carolinas_dirs() -> set[Path]:
    out: set[Path] = set()
    for base in (TRAIN, INF):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir() and (
                p.name.startswith("train_carolinas_")
                or p.name.startswith("inference_carolinas_")
            ):
                out.add(p)
    return out


def main() -> None:
    keep = ngsim_ablation_dirs() | carolinas_dirs()
    dry_run = "--dry-run" in sys.argv

    removed = 0
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
