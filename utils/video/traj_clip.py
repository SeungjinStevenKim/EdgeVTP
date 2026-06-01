"""Clip live-video predictions to plausible lengths vs observed history."""

import numpy as np


def _path_length(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _truncate_path(path, max_len):
    """Trim a polyline to max_len arc length, keeping the start fixed."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2 or max_len <= 0:
        return path

    total = _path_length(path)
    if total <= max_len:
        return path

    clipped = [path[0].copy()]
    acc = 0.0
    for i in range(1, len(path)):
        seg = path[i] - path[i - 1]
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-6:
            continue
        if acc + seg_len >= max_len:
            clipped.append(path[i - 1] + seg * ((max_len - acc) / seg_len))
            return np.asarray(clipped, dtype=np.float64)
        clipped.append(path[i].copy())
        acc += seg_len
    return np.asarray(clipped, dtype=np.float64)


def _extend_path(path, min_len):
    """Uniformly scale a short path from its start to reach min_len."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return path
    total = _path_length(path)
    if total >= min_len or total <= 1e-6:
        return path
    origin = path[0].copy()
    scale = min_len / total
    return origin + (path - origin) * scale


def _resample_path(path, n_points):
    """Arc-length resample a polyline to exactly n_points."""
    path = np.asarray(path, dtype=np.float64)
    if n_points <= 0:
        return path
    if len(path) == 0:
        return np.zeros((n_points, 2), dtype=np.float64)
    if len(path) == 1:
        return np.repeat(path, n_points, axis=0)

    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 1e-6:
        return np.repeat(path[:1], n_points, axis=0)

    samples = np.linspace(0.0, total, n_points)
    xs = np.interp(samples, cum, path[:, 0])
    ys = np.interp(samples, cum, path[:, 1])
    return np.column_stack([xs, ys])


def _recent_velocity(obs, lookback=3):
    obs = np.asarray(obs, dtype=np.float64)
    if len(obs) < 2:
        return None
    span = min(lookback, len(obs) - 1)
    return obs[-1] - obs[-1 - span]


def _align_prediction_bearing(obs, pred, max_angle_deg=35.0):
    """Rotate a mispointed prediction to match recent observed motion."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    vel = _recent_velocity(obs)
    if vel is None:
        return pred

    v_norm = float(np.linalg.norm(vel))
    start = obs[-1]
    heading = pred[-1] - start
    h_norm = float(np.linalg.norm(heading))
    if v_norm < 8.0 or h_norm < 8.0:
        return pred

    cos_a = float(np.dot(heading, vel) / (h_norm * v_norm))
    cos_a = np.clip(cos_a, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_a)))
    if angle <= max_angle_deg:
        return pred

    a_pred = float(np.arctan2(heading[1], heading[0]))
    a_vel = float(np.arctan2(vel[1], vel[0]))
    rot = a_vel - a_pred
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c, -s], [s, c]], dtype=np.float64)
    return start + (pred - start) @ rot_mat.T


def target_pred_length(obs_abs, pred_len, obs_len):
    """Expected prediction arc length from observed history speed."""
    obs = np.asarray(obs_abs, dtype=np.float64)
    hist_len = _path_length(obs)
    if hist_len <= 1e-3:
        hist_len = float(np.linalg.norm(obs[-1] - obs[0])) if len(obs) >= 2 else 1.0
    return max(
        hist_len * (float(pred_len) / max(float(obs_len) - 1.0, 1.0)),
        20.0,
    )


def clip_obs_pred_pair(obs_abs, pred_abs, pred_len, obs_len, max_ratio=1.35, min_ratio=0.60):
    """
    Clamp one agent's absolute prediction to a band around history speed.

    Matches Carolinas test-scale motion (~FDE 98 px over 25 steps) on live video.
    """
    obs = np.asarray(obs_abs, dtype=np.float64)
    pred = np.asarray(pred_abs, dtype=np.float64)
    if pred.ndim != 2 or len(pred) < 1:
        return np.zeros((pred_len, 2), dtype=np.float64)

    start = obs[-1]
    path = np.vstack([start, pred])
    target = target_pred_length(obs, pred_len, obs_len)
    path = _truncate_path(path, target * max_ratio)
    path = _extend_path(path, target * min_ratio)
    path = _resample_path(path, pred_len + 1)
    pred_out = path[1:]
    return _align_prediction_bearing(obs, pred_out)


def clip_scene_predictions(obs_abs, pred_abs, pred_len, obs_len, max_ratio=1.35, min_ratio=0.60):
    """Apply clip_obs_pred_pair for each agent in a scene batch."""
    obs_abs = np.asarray(obs_abs, dtype=np.float64)
    pred_abs = np.asarray(pred_abs, dtype=np.float64)
    out = pred_abs.copy()
    for i in range(out.shape[0]):
        out[i] = clip_obs_pred_pair(
            obs_abs[i], out[i], pred_len, obs_len, max_ratio=max_ratio, min_ratio=min_ratio
        )
    return out
