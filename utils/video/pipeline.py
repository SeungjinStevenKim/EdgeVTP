"""End-to-end video inference loop: detect, track, predict, visualize."""

import os

import cv2

from utils.inference_engine import EdgeVTPInference, resolve_device, resolve_edgevtp_device
from utils.video.cli import apply_runtime_defaults, build_coord_transform, build_vehicle_detector, open_video
from utils.video.debug import log_model_io
from utils.video.tracker import MultiObjectTracker
from utils.video.viz import draw_tracks


class VideoPredictor:
    """Run EdgeVTP trajectory prediction on a video source."""

    def __init__(self, args, config):
        self.args = apply_runtime_defaults(args, config)
        self.config = config
        self.obs_len = config["input_data"]["observed_steps"]
        self.history_hz = self.args.history_hz

        yolo_device = resolve_device(args.device)
        edgevtp_device = resolve_edgevtp_device(
            getattr(args, "edgevtp_device", "auto"), yolo_device
        )

        self.coord_transform = build_coord_transform(config)
        self.engine = EdgeVTPInference(
            self.args.config,
            checkpoint_path=self.args.checkpoint,
            dataset_name=self.args.dataset_name,
            device=str(edgevtp_device),
        )
        self.engine.configure_for_coord_mode(self.coord_transform.mode)
        args.device = str(yolo_device)
        self.detector = build_vehicle_detector(self.args)

        self.cap = open_video(self.args.video)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

        self.history_stride = max(1, round(self.fps / self.history_hz))
        self.history_seconds = self.obs_len / self.history_hz
        self.history_frames = self.obs_len * self.history_stride
        warmup_frames = max(self.args.warmup_frames or 0, self.history_frames)

        self.tracker = MultiObjectTracker(
            history_len=self.obs_len,
            max_age=self.args.max_age,
            iou_threshold=self.args.iou_threshold,
            min_hits=self.args.min_hits,
            max_tracks=self.args.max_tracks,
            history_stride=self.history_stride,
            warmup_frames=warmup_frames,
            frame_size=(self.height, self.width),
            frame_rate=int(round(self.fps)),
            oncoming_lookback=self.args.oncoming_lookback,
            oncoming_min_dy=self.args.oncoming_min_dy,
        )

        self.writer = None
        if not self.args.no_save:
            os.makedirs(os.path.dirname(self.args.output) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.args.output, fourcc, self.fps, (self.width, self.height))

    def run(self):
        args = self.args

        frame_idx = 0
        infer_count = 0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            detections = self.detector.track(frame)
            tracks = self.tracker.update(detections)

            if frame_idx % self.history_stride == 0:
                track_ids, obs_model = self.tracker.build_scene(
                    self.coord_transform, oncoming_only=args.oncoming_only
                )
                if obs_model is not None and obs_model.shape[0] >= args.min_agents:
                    pred_model = self.engine.predict_scene(obs_model, isolate_agents=True)
                    self.tracker.set_predictions(track_ids, pred_model, self.coord_transform)
                    tracks = self.tracker.refresh_track_views(tracks)
                    infer_count += 1
                    log_model_io(
                        frame_idx,
                        track_ids,
                        obs_model,
                        pred_model.numpy(),
                        getattr(self.engine, "last_scene_debug", {}),
                    )

            draw_tracks(
                frame,
                tracks,
                min_hits=args.min_hits,
                oncoming_only=args.oncoming_only,
                pred_steps=self.engine.pred_len,
                obs_steps=self.obs_len,
            )

            if self.writer is not None:
                self.writer.write(frame)
            if args.show:
                cv2.imshow("EdgeVTP video predict", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if args.show:
            cv2.destroyAllWindows()

        return frame_idx, infer_count
