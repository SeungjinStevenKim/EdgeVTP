#!/usr/bin/env python3
"""Collect NGSIM benchmark results into a summary table."""
import os
import re

EXP_ROOT = "experiments/vehicle/inference"
INFERENCE_NAMES = [
    "inference_ngsim_original_20ep", "inference_ngsim_original_40ep", "inference_ngsim_original_80ep",
    "inference_ngsim_30m_20ep", "inference_ngsim_30m_40ep", "inference_ngsim_30m_80ep",
    "inference_ngsim_30m_aug_20ep", "inference_ngsim_30m_aug_40ep", "inference_ngsim_30m_aug_80ep",
    "inference_ngsim_aug_20ep", "inference_ngsim_aug_40ep", "inference_ngsim_aug_80ep",
    # KAN variants
    "inference_ngsim_kan_20ep", "inference_ngsim_kan_40ep", "inference_ngsim_kan_80ep",
    "inference_ngsim_kan_30m_20ep", "inference_ngsim_kan_30m_40ep", "inference_ngsim_kan_30m_80ep",
    "inference_ngsim_kan_30m_aug_20ep", "inference_ngsim_kan_30m_aug_40ep", "inference_ngsim_kan_30m_aug_80ep",
    "inference_ngsim_kan_aug_20ep", "inference_ngsim_kan_aug_40ep", "inference_ngsim_kan_aug_80ep",
]

def parse_results(path):
    ade = fde = None
    rmse = None  # (s1, s2, s3, s4, s5) or None
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if "Average ADE:" in line:
                ade = float(line.split(":")[1].strip())
            elif "Average FDE:" in line:
                fde = float(line.split(":")[1].strip())
            elif "|" in line and not line.strip().startswith("Summary"):
                # Data row: method | s1 | s2 | s3 | s4 | s5 | ADE | FDE
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 8:
                    try:
                        vals = [float(x) for x in parts[-7:]]
                        rmse = tuple(vals[:5])
                    except ValueError:
                        pass
    if ade is None:
        return None
    return (ade, fde, rmse)

def main():
    print("NGSIM Benchmark Results")
    print("=" * 100)
    header = f"{'Config':<18} {'Epochs':<8} {'1s':<8} {'2s':<8} {'3s':<8} {'4s':<8} {'5s':<8} {'ADE':<10} {'FDE':<10}"
    print(header)
    print("-" * 100)

    rows = []
    for name in INFERENCE_NAMES:
        path = os.path.join(EXP_ROOT, name, "inference_results.txt")
        res = parse_results(path)
        m = re.match(r"inference_ngsim_(.+)_(\d+)ep", name)
        config, ep = (m.group(1), m.group(2)) if m else (name, "-")
        suffix = " (KAN)" if "kan" in config and config != "kan" else ""
        if "original" in config:
            config_pretty = "Original" + suffix
        elif "kan_30m_aug" in config or "30m_aug" in config:
            config_pretty = "30m+aug" + suffix
        elif "kan_30m" in config or "30m" in config:
            config_pretty = "30m filter" + suffix
        elif config == "kan":
            config_pretty = "KAN only"
        elif "kan_aug" in config or "aug" in config:
            config_pretty = "Aug only" + suffix
        else:
            config_pretty = "Aug only" + suffix

        if res:
            ade, fde, rmse = res[0], res[1], res[2]
            if rmse:
                s1, s2, s3, s4, s5 = rmse
                print(f"{config_pretty:<18} {ep:<8} {s1:<8.2f} {s2:<8.2f} {s3:<8.2f} {s4:<8.2f} {s5:<8.2f} {ade:<10.4f} {fde:<10.4f}")
                rows.append((config_pretty, ep, s1, s2, s3, s4, s5, ade, fde))
            else:
                print(f"{config_pretty:<18} {ep:<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {ade:<10.4f} {fde:<10.4f}")
                rows.append((config_pretty, ep, None, None, None, None, None, ade, fde))
        else:
            print(f"{config_pretty:<18} {ep:<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<10} {'-':<10} (no results)")

    # Write summary to file
    out_path = "experiments/vehicle/NGSIM_BENCHMARK_SUMMARY.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("NGSIM Benchmark Summary\n")
        f.write("=" * 100 + "\n")
        for r in rows:
            rmse_str = f"  1s:{r[2]:.2f} 2s:{r[3]:.2f} 3s:{r[4]:.2f} 4s:{r[5]:.2f} 5s:{r[6]:.2f}" if r[2] is not None else ""
            f.write(f"{r[0]} {r[1]}ep  ADE: {r[7]:.4f}  FDE: {r[8]:.4f}" + (f"  RMSE{rmse_str}" if rmse_str else "") + "\n")
    print("-" * 80)
    print(f"\nSummary saved to: {out_path}")

if __name__ == "__main__":
    main()
