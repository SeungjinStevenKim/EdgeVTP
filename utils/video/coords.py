"""Pixel ↔ model-space coordinate transforms for video inference."""

import json

import numpy as np


class CoordTransform:
    """
    Convert between video pixel coordinates and model input coordinates.

    Modes:
      - pixels: pass-through (Carolinas-style models)
      - scale: multiply pixels by meters_per_pixel (simple NGSIM-style approx.)
      - homography: 3x3 matrix mapping pixel (x, y, 1) -> world (x, y)
    """

    def __init__(self, mode="pixels", meters_per_pixel=None, homography_path=None):
        self.mode = mode
        self.meters_per_pixel = meters_per_pixel
        self.H = None
        self.H_inv = None

        if homography_path:
            with open(homography_path, "r") as file:
                data = json.load(file)
            self.H = np.asarray(data["pixels_to_world"], dtype=np.float64)
            if self.H.shape != (3, 3):
                raise ValueError("pixels_to_world must be a 3x3 matrix")
            self.H_inv = np.linalg.inv(self.H)
            self.mode = "homography"
        elif mode == "scale":
            if meters_per_pixel is None or meters_per_pixel <= 0:
                raise ValueError("meters_per_pixel must be > 0 for scale mode")

    def pixels_to_model(self, points):
        points = np.asarray(points, dtype=np.float64)
        single = points.ndim == 1
        if single:
            points = points.reshape(1, 2)

        if self.mode == "pixels":
            out = points.copy()
        elif self.mode == "scale":
            out = points * self.meters_per_pixel
        elif self.mode == "homography":
            ones = np.ones((points.shape[0], 1), dtype=np.float64)
            hom = np.hstack([points, ones])
            mapped = (self.H @ hom.T).T
            out = mapped[:, :2] / mapped[:, 2:3]
        else:
            raise ValueError(f"Unknown coord mode: {self.mode}")

        return out[0] if single else out

    def model_to_pixels(self, points):
        points = np.asarray(points, dtype=np.float64)
        single = points.ndim == 1
        if single:
            points = points.reshape(1, 2)

        if self.mode == "pixels":
            out = points.copy()
        elif self.mode == "scale":
            out = points / self.meters_per_pixel
        elif self.mode == "homography":
            ones = np.ones((points.shape[0], 1), dtype=np.float64)
            hom = np.hstack([points, ones])
            mapped = (self.H_inv @ hom.T).T
            out = mapped[:, :2] / mapped[:, 2:3]
        else:
            raise ValueError(f"Unknown coord mode: {self.mode}")

        return out[0] if single else out
