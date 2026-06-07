"""Draw predicted trajectories on video frames."""

import cv2
import numpy as np

from utils.video.tracker import MultiObjectTracker


def track_color(track_id):
    """Stable random vivid BGR color per track (seeded by id)."""
    rng = np.random.default_rng(int(track_id))
    hue = int(rng.integers(0, 180))
    sat = int(rng.integers(180, 256))
    val = int(rng.integers(200, 256))
    bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def reanchor_prediction(pred, start, prediction_anchor):
    """Shift a stale absolute prediction to the car's current position."""
    pred = np.asarray(pred, dtype=np.float64)
    if prediction_anchor is None:
        return pred
    anchor = np.asarray(prediction_anchor, dtype=np.float64)
    return pred + (start - anchor)


def build_draw_path(track):
    """Re-anchor model output and return the path to draw."""
    if track.trail_start is not None:
        start = np.asarray(track.trail_start, dtype=np.float64)
    else:
        start = np.asarray(
            MultiObjectTracker.bbox_center(track.last_bbox, track.anchor),
            dtype=np.float64,
        )

    pred = reanchor_prediction(track.prediction, start, track.prediction_anchor)
    if pred.ndim != 2 or pred.shape[0] < 1:
        return None

    return np.vstack([start, pred])


def _direction_ok(path, history_direction, max_angle_deg=45.0):
    if history_direction is None:
        return True
    pred_dir = path[-1] - path[0]
    p_norm = float(np.linalg.norm(pred_dir))
    h_norm = float(np.linalg.norm(history_direction))
    if p_norm < 3.0 or h_norm < 1e-6:
        return False
    cos_a = float(np.dot(pred_dir, history_direction) / (p_norm * h_norm))
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a))) <= max_angle_deg


def draw_tracks(frame, tracks, min_hits=3, oncoming_only=True, pred_steps=25, obs_steps=15):
    del pred_steps, obs_steps

    for track in tracks:
        if not track.confirmed(min_hits):
            continue
        if track.missed > 0 or track.prediction is None:
            continue
        if oncoming_only and not track.oncoming:
            continue

        path = build_draw_path(track)
        if path is None or len(path) < 2:
            continue
        if np.linalg.norm(path[-1] - path[0]) < 3.0:
            continue
        if oncoming_only and path[-1, 1] < path[0, 1] - 2.0:
            continue
        if not _direction_ok(path, track.history_direction):
            continue

        color = track_color(track.track_id)
        pred_pts = path.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pred_pts], False, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(frame, [pred_pts], False, color, 3, cv2.LINE_AA)
        cv2.circle(frame, tuple(pred_pts[-1, 0]), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pred_pts[-1, 0]), 4, color, -1, cv2.LINE_AA)

    return frame
