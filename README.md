# EdgeVTP / VT-Former

**EdgeVTP** (Efficient Edge Vehicle Trajectory Prediction) is a lightweight, graph-based trajectory prediction codebase designed for high-performance highway surveillance. It features one-shot Bezier decoding and TCN-based residual modules for optimal accuracy-latency trade-offs on edge devices.

![EdgeVTP / VT-Former architecture](docs/figures/edgevtp_architecture.png)

---

- [🚀 Quick Start](#-quick-start)
- [📊 Paper Results](#-paper-results-ngsim)
- [📂 Repository Structure](#-repository-structure)
- [📜 Citation](#-citation)

---

## 🚀 Quick Start

### 1. Installation
```bash
conda create -n vt_former python=3.8 -y
conda activate vt_former

# Install PyTorch & PyG
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-geometric==2.0.4 -r requirements.txt
```

### 2. Dataset Setup
Place preprocessed datasets in the following structure:
```text
datasets/
  ngsim/
    train/
    val/
    test/
```

### 3. Running Inference (Paper Repro)
Pre-trained weights are **included** in the `experiments/vehicle/train/` folders. You can reproduce the paper's three main operating points immediately:

| Model | Radius | Residual | Config Path |
| :--- | :--- | :--- | :--- |
| **Latency** | 20m | No | `experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_k16_5hz/config.yaml` |
| **Balanced** | 20m | Yes | `experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml` |
| **Error** | 30m | Yes | `experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml` |

**Command:**
```bash
python ngsim_share/main.py --config <CONFIG_PATH>
```

---

## 📊 Paper Results (NGSIM)

Verified performance on the headline operating points (K=16, 5Hz):

| Model Variant | Radius | ADE ↓ | FDE ↓ | Latency (E2E) |
| :--- | :--- | :--- | :--- | :--- |
| Latency-focused | 20m | 2.13 | 4.93 | 2.83 ms |
| Balanced | 20m | 1.89 | 4.37 | 4.36 ms |
| Error-focused | 30m | 1.85 | 4.25 | 4.60 ms |

---

## 📂 Repository Structure

*   `configs/`: Core YAML recipes for NGSIM and archive sweeps.
*   `experiments/`: Checkpoints (`ngsim.pt`) and result logs for paper runs.
*   `ngsim_share/`: Primary entry point and shared utilities.
*   `scripts/`: Automation for benchmarks and pruning.

---

## 📜 Citation

If you find this work useful, please cite our CVPR EVW 2026 paper:

```bibtex
@inproceedings{edgevtp2026,
  title={EdgeVTP: Efficient Edge Vehicle Trajectory Prediction via Graph-based Transformer},
  author={Danesh Pazho, Armin and Alinezhad Noghre, Ghazal and Katariya, Vinit and Tabkhi, Hamed},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2026}
}
```

---
**License**: See `LICENSE`.
