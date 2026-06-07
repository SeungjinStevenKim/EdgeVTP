"""Log EdgeVTP model input/output during video inference."""

import numpy as np


def _path_length(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def log_model_io(frame_idx, track_ids, obs_model, pred_model, scene_debug=None):
    """Print observed trajectories (input) and predicted trajectories (output)."""
    obs_model = np.asarray(obs_model, dtype=np.float64)
    pred_model = np.asarray(pred_model, dtype=np.float64)
    scene_debug = scene_debug or {}
    edges = scene_debug.get("num_edges", "?")

    print(f"\n[frame {frame_idx}] agents={len(track_ids)} graph_edges={edges}")
    for i, tid in enumerate(track_ids):
        obs = obs_model[i]
        pred = pred_model[i]
        obs_len = _path_length(obs)
        pred_len = _path_length(pred)
        print(
            f"track {tid}  in_len={obs_len:6.1f}px  out_len={pred_len:6.1f}px  "
            f"ratio={pred_len / max(obs_len, 1.0):.2f}"
        )
        print(f"track {tid} input ({obs.shape[0]}, 2):")
        print(np.array2string(obs, precision=1, suppress_small=True))
        print(f"track {tid} output ({pred.shape[0]}, 2):")
        print(np.array2string(pred, precision=1, suppress_small=True))
