# NGSIM — minimal configs

These **10 YAMLs** match the **Inference Package** five-model table (plus matching train recipes). Paths assume you run from the **repository root**:

```bash
python ngsim_share/main.py --config configs/ngsim/<name>.yaml
```

**Inference (5):**

- `inference_ngsim_30m_80ep_5hz.yaml`
- `inference_ngsim_30m_oneshot_bezier_80ep_5hz.yaml`
- `inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_5hz.yaml`
- `inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz.yaml`
- `inference_ngsim_35m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k12_L2H2.yaml`

**Train (paired):** `train_ngsim_30m_80ep.yaml`, `train_ngsim_30m_oneshot_bezier_80ep.yaml`, `train_ngsim_30m_oneshot_bezier_80ep_residual_v2.yaml`, `train_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn.yaml`, `train_ngsim_35m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k12_L2H2.yaml`

For **paper operating points** (e.g. 20 m / 30 m, K=16, residual+TCN @ 5 Hz), use the `experiments/vehicle/.../config.yaml` paths in the root README “Pre-trained Models” table.

All other NGSIM grids (KAN, aug sweeps, etc.) are under **`configs/archive/ngsim/`** (see `configs/archive/README.md`).
