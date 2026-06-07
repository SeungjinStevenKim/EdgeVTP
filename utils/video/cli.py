"""CLI parsing and runtime defaults for the video inference pipeline."""

import argparse
import os
from pathlib import Path

import cv2
import yaml

from utils.video.coords import CoordTransform
from utils.video.detector import build_detector

# Settings not exposed on the CLI (6-arg interface stays minimal).
RUNTIME_DEFAULTS = {
    "yolo_model": "yolov8n.pt",
    "conf_threshold": 0.30,
    "nms_threshold": 0.45,
    "max_detections": 50,
    "yolo_imgsz": 1280,
    "min_hits": 3,
    "min_agents": 1,
    "max_tracks": 50,
    "iou_threshold": 0.25,
    "max_age": None,
    "warmup_frames": None,
    "max_frames": None,
    "meters_per_pixel": 0.05,
    "output_dir": "runs/video_predict",
    "no_save": False,
    "oncoming_only": True,
    "oncoming_min_dy": 8.0,
    "oncoming_lookback": 12,
    "isolate_agents": False,
    "no_traj_clip": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EdgeVTP real-video trajectory prediction")
    parser.add_argument("--video", required=True, help="Input video path or camera index (0)")
    parser.add_argument("--config", required=True, help="Inference YAML config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint .pt (default: from config model_dir + dataset name)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="YOLO device: auto | cpu | mps | cuda:0 (EdgeVTP defaults to cpu when YOLO uses mps)",
    )
    parser.add_argument(
        "--edgevtp-device",
        default="auto",
        help="EdgeVTP device: auto | cpu | mps | cuda:0 (auto=cpu when --device is mps)",
    )
    parser.add_argument(
        "--history-hz",
        type=float,
        default=5.0,
        help="Model observation sample rate in Hz (training data is 5 Hz)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (default: runs/video_predict/<video_stem>_pred.mp4)",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Ultralytics weights (default: yolov8n.pt)",
    )
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument(
        "--all-trails",
        action="store_true",
        help="Draw prediction trails for all directions (default: oncoming only)",
    )
    parser.add_argument(
        "--isolate-agents",
        action="store_true",
        help="Run inference per agent independently (disables scene graph interactions)",
    )
    parser.add_argument(
        "--no-traj-clip",
        action="store_true",
        help="Disable trajectory clipping/alignment postprocess and draw raw model output",
    )
    return parser.parse_args(argv)


def apply_runtime_defaults(args, config):
    """Attach pipeline defaults that are not CLI flags."""
    obs_len = config["input_data"]["observed_steps"]
    for key, value in RUNTIME_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if getattr(args, "yolo_model", None) is None:
        args.yolo_model = RUNTIME_DEFAULTS["yolo_model"]
    args.model_path = args.yolo_model
    args.dataset_name = config["input_data"]["dataset"][0]

    stem = Path(str(args.video)).stem or "output"
    if args.output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.output = str(Path(args.output_dir) / f"{stem}_pred.mp4")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.warmup_frames is None:
        # Overridden in pipeline once fps/history_stride are known (obs_len * stride).
        args.warmup_frames = obs_len

    if args.max_age is None:
        args.max_age = max(60, args.warmup_frames * 4)

    if getattr(args, "all_trails", False):
        args.oncoming_only = False

    return args


def open_video(source):
    if str(source).isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_coord_transform(config):
    datasets = config.get("input_data", {}).get("dataset", [])
    ds_name = " ".join(str(d) for d in datasets).lower()
    if "ngsim" in ds_name or "highd" in ds_name:
        return CoordTransform(
            mode="scale",
            meters_per_pixel=RUNTIME_DEFAULTS["meters_per_pixel"],
        )
    return CoordTransform(mode="pixels")


def build_vehicle_detector(args):
    return build_detector(args)
