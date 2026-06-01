"""Real-video detection, tracking, and EdgeVTP trajectory prediction."""

from utils.video.cli import (
    RUNTIME_DEFAULTS,
    apply_runtime_defaults,
    build_coord_transform,
    build_vehicle_detector,
    load_config,
    open_video,
    parse_args,
)
from utils.video.coords import CoordTransform
from utils.video.detector import YoloDetector, build_detector
from utils.video.pipeline import VideoPredictor
from utils.video.tracker import MultiObjectTracker, TrackView
from utils.video.viz import draw_tracks

__all__ = [
    "RUNTIME_DEFAULTS",
    "CoordTransform",
    "MultiObjectTracker",
    "TrackView",
    "VideoPredictor",
    "YoloDetector",
    "apply_runtime_defaults",
    "build_coord_transform",
    "build_detector",
    "build_vehicle_detector",
    "draw_tracks",
    "load_config",
    "open_video",
    "parse_args",
]
