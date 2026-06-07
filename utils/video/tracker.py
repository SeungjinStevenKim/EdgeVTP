"""History buffer and scene assembly for tracked vehicles."""

from collections import deque

import numpy as np


class TrackView:
    """Lightweight track object for pipeline / visualization."""

    __slots__ = (
        "track_id",
        "last_bbox",
        "prediction",
        "prediction_anchor",
        "hits",
        "missed",
        "anchor",
        "oncoming",
        "trail_start",
        "recent_motion",
        "history_motion",
        "history_direction",
    )

    def __init__(
        self,
        track_id,
        bbox,
        hits,
        prediction=None,
        prediction_anchor=None,
        missed=0,
        oncoming=False,
        trail_start=None,
        recent_motion=0.0,
        history_motion=0.0,
        history_direction=None,
    ):
        self.track_id = track_id
        self.last_bbox = bbox
        self.prediction = prediction
        self.prediction_anchor = prediction_anchor
        self.hits = hits
        self.missed = missed
        self.anchor = "bottom"
        self.oncoming = oncoming
        self.trail_start = trail_start
        self.recent_motion = recent_motion
        self.history_motion = history_motion
        self.history_direction = history_direction

    def confirmed(self, min_hits):
        return self.hits >= min_hits and self.missed == 0


class MultiObjectTracker:
    """Maintains per-id observation history for EdgeVTP inference."""

    def __init__(
        self,
        history_len=15,
        max_age=30,
        iou_threshold=0.4,
        min_hits=3,
        max_tracks=50,
        anchor="bottom",
        history_stride=1,
        warmup_frames=None,
        frame_size=(720, 1280),
        frame_rate=30,
        oncoming_lookback=12,
        oncoming_min_dy=8.0,
    ):
        self.oncoming_lookback = oncoming_lookback
        self.oncoming_min_dy = oncoming_min_dy
        self.obs_len = history_len
        self.history_stride = max(1, int(history_stride))
        self.frame_buffer = history_len * self.history_stride
        self.warmup_frames = warmup_frames or self.frame_buffer
        self.min_hits = min_hits
        self.max_tracks = max_tracks
        self.max_age = max_age
        self.anchor = anchor
        self.history_cache = {}

    @staticmethod
    def bbox_center(bbox, anchor="bottom"):
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = y2 if anchor == "bottom" else 0.5 * (y1 + y2)
        return (cx, cy)

    @staticmethod
    def is_oncoming(entry, lookback=12, min_dy=8.0):
        """True when track moves toward the camera (down-screen in eyelevel footage)."""
        hist = list(entry["coords"])
        if len(hist) < 4:
            return False
        span = min(lookback, len(hist) - 1)
        dy = hist[-1][1] - hist[-1 - span][1]
        if dy < min_dy:
            return False
        sizes = entry.get("bbox_sizes")
        if sizes and len(sizes) > span:
            size_ratio = sizes[-1] / max(sizes[-1 - span], 1.0)
            if size_ratio < 1.02 and dy < min_dy * 1.5:
                return False
        return True

    @staticmethod
    def _sample_obs_indices(hist_len, obs_len, history_stride):
        full_span = obs_len * history_stride
        if hist_len >= full_span:
            return [
                hist_len - 1 - i * history_stride
                for i in range(obs_len - 1, -1, -1)
            ]
        return [
            int(round(i * (hist_len - 1) / (obs_len - 1)))
            for i in range(obs_len)
        ]

    @staticmethod
    def _path_length(points):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _history_motion(self, entry):
        sampled = self._sampled_history(entry)
        if sampled is None:
            return 0.0
        return self._path_length(sampled)

    def _history_direction(self, entry):
        sampled = self._sampled_history(entry)
        if sampled is None or len(sampled) < 2:
            return None
        delta = sampled[-1] - sampled[0]
        norm = float(np.linalg.norm(delta))
        if norm < 1e-3:
            return None
        return delta / norm

    def _sampled_history(self, entry):
        hist = list(entry["coords"])
        if len(hist) < 2:
            return None
        indices = self._sample_obs_indices(len(hist), self.obs_len, self.history_stride)
        return np.asarray([hist[i] for i in indices], dtype=np.float64)

    def _history_is_usable(self, entry):
        """Reject jittery or teleported histories that poison scene graphs."""
        sampled = self._sampled_history(entry)
        if sampled is None or len(sampled) < 4:
            return False

        seg = np.diff(sampled, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        if seg_len.size == 0:
            return False

        total_len = float(np.sum(seg_len))
        net_disp = float(np.linalg.norm(sampled[-1] - sampled[0]))
        if net_disp < 20.0:
            return False

        med = float(np.median(seg_len))
        if float(np.max(seg_len)) > max(35.0, med * 4.0 + 15.0):
            return False
        if total_len > net_disp * 2.0:
            return False

        net = sampled[-1] - sampled[0]
        if net_disp >= 20.0:
            direction = net / net_disp
            opposing = 0.0
            for i, step in enumerate(seg):
                if seg_len[i] <= 1e-6:
                    continue
                if float(np.dot(step, direction)) < 0.0:
                    if seg_len[i] > max(18.0, 0.18 * net_disp):
                        return False
                    opposing += float(seg_len[i])
            if opposing > 0.18 * total_len:
                return False

        if self.is_oncoming(
            entry,
            lookback=self.oncoming_lookback,
            min_dy=self.oncoming_min_dy,
        ):
            seg_dy = np.diff(sampled[:, 1])
            if np.any(seg_dy < -12.0):
                return False
            if int(np.sum(seg_dy < -6.0)) > 1:
                return False

        return True

    @staticmethod
    def _recent_motion(entry, lookback=12):
        hist = list(entry["coords"])
        if len(hist) < 2:
            return 0.0
        span = min(lookback, len(hist) - 1)
        a = np.asarray(hist[-1 - span], dtype=np.float64)
        b = np.asarray(hist[-1], dtype=np.float64)
        return float(np.linalg.norm(b - a))

    def update(self, tracked_detections):
        """
        Args:
            tracked_detections: list of (track_id, (x1, y1, x2, y2))
        """
        active_ids = set()
        for track_id, bbox in tracked_detections:
            tid = int(track_id)
            active_ids.add(tid)
            cx, cy = self.bbox_center(bbox, self.anchor)

            if tid not in self.history_cache:
                if len(self.history_cache) >= self.max_tracks:
                    self._drop_weakest()
                self.history_cache[tid] = {
                    "coords": deque(maxlen=self.frame_buffer),
                    "bbox_sizes": deque(maxlen=self.frame_buffer),
                    "prediction": None,
                    "prediction_anchor": None,
                    "hits": 0,
                    "missed": 0,
                }

            entry = self.history_cache[tid]
            entry["coords"].append((cx, cy))
            x1, y1, x2, y2 = bbox
            entry["bbox_sizes"].append(max(1.0, (x2 - x1) * (y2 - y1)))
            entry["hits"] += 1
            entry["missed"] = 0
            entry["last_bbox"] = bbox

        views = []
        for tid, entry in list(self.history_cache.items()):
            if tid in active_ids:
                hist = list(entry["coords"])
                trail_start = np.asarray(hist[-1], dtype=np.float64) if hist else None
                views.append(
                    TrackView(
                        tid,
                        entry["last_bbox"],
                        entry["hits"],
                        prediction=entry["prediction"],
                        prediction_anchor=entry.get("prediction_anchor"),
                        missed=0,
                        oncoming=self.is_oncoming(
                            entry,
                            lookback=self.oncoming_lookback,
                            min_dy=self.oncoming_min_dy,
                        ),
                        trail_start=trail_start,
                        recent_motion=self._recent_motion(
                            entry, lookback=self.oncoming_lookback
                        ),
                        history_motion=self._history_motion(entry),
                        history_direction=self._history_direction(entry),
                    )
                )
                continue
            entry["missed"] += 1
            entry["prediction"] = None
            entry["prediction_anchor"] = None
            if entry["missed"] > self.max_age:
                del self.history_cache[tid]

        return views

    def refresh_track_views(self, views):
        """Sync prediction fields on views after set_predictions."""
        for view in views:
            entry = self.history_cache.get(view.track_id)
            if entry is None:
                continue
            view.prediction = entry["prediction"]
            view.prediction_anchor = entry.get("prediction_anchor")
            view.history_motion = self._history_motion(entry)
            view.history_direction = self._history_direction(entry)
        return views

    def _drop_weakest(self):
        if not self.history_cache:
            return
        victim = min(
            self.history_cache,
            key=lambda tid: (self.history_cache[tid]["hits"], len(self.history_cache[tid]["coords"])),
        )
        del self.history_cache[victim]

    def confirmed_tracks(self):
        return [
            TrackView(
                tid,
                data["last_bbox"],
                data["hits"],
                prediction=data["prediction"],
            )
            for tid, data in self.history_cache.items()
            if data["hits"] >= self.min_hits and data["missed"] == 0
        ]

    def ready_tracks(self):
        return [
            tid
            for tid, data in self.history_cache.items()
            if len(data["coords"]) >= self.warmup_frames
            and data["hits"] >= self.min_hits
            and data["missed"] == 0
        ]

    def build_scene(self, coord_transform, oncoming_only=False):
        track_ids = []
        obs_model = []

        for tid, data in self.history_cache.items():
            if data["missed"] > 0:
                continue
            if oncoming_only and not self.is_oncoming(
                data,
                lookback=self.oncoming_lookback,
                min_dy=self.oncoming_min_dy,
            ):
                continue
            hist = list(data["coords"])
            if len(hist) < self.warmup_frames:
                continue
            if not self._history_is_usable(data):
                continue

            sampled = self._sampled_history(data)
            track_ids.append(tid)
            obs_model.append(coord_transform.pixels_to_model(sampled))

        if not track_ids:
            return [], None
        return track_ids, np.stack(obs_model, axis=0)

    def set_predictions(self, track_ids, pred_model, coord_transform):
        pred_model = pred_model.numpy() if hasattr(pred_model, "numpy") else np.asarray(pred_model)
        active = set(track_ids)
        for tid, data in self.history_cache.items():
            if data["missed"] == 0 and tid not in active:
                data["prediction"] = None
                data["prediction_anchor"] = None

        for idx, tid in enumerate(track_ids):
            if tid not in self.history_cache:
                continue
            hist = list(self.history_cache[tid]["coords"])
            anchor = hist[-1] if hist else None
            self.history_cache[tid]["prediction"] = coord_transform.model_to_pixels(pred_model[idx])
            self.history_cache[tid]["prediction_anchor"] = anchor
