# Differentiable Bezier curve for End-to-End trajectory prediction.
# Cubic Bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
# P0 = start pos (fixed), P1,P2,P3 = control points from model

import numpy as np
import torch


def bezier_sample_torch(p0, p1, p2, p3, num_points):
    """
    Differentiable cubic Bezier sampling. PyTorch ops only.
    B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3

    Samples at t=1/N, 2/N, ..., 1 so output aligns with pred_traj_gt (future positions).
    Excludes t=0 (start_pos) since GT has only future positions.

    Args:
        p0: (batch, 2) start point
        p1, p2, p3: (batch, 2) control points
        num_points: int, number of samples (e.g. 25)

    Returns:
        sampled: (batch, num_points, 2)
    """
    t = torch.linspace(1.0 / num_points, 1.0, num_points, device=p0.device, dtype=p0.dtype)
    t = t.unsqueeze(0).unsqueeze(-1)  # (1, num_points, 1)
    one_minus_t = 1 - t
    b0 = one_minus_t ** 3
    b1 = 3 * (one_minus_t ** 2) * t
    b2 = 3 * one_minus_t * (t ** 2)
    b3 = t ** 3
    # p0: (B, 2) -> (B, 1, 2)
    pts = (
        b0 * p0.unsqueeze(1)
        + b1 * p1.unsqueeze(1)
        + b2 * p2.unsqueeze(1)
        + b3 * p3.unsqueeze(1)
    )
    return pts  # (batch, num_points, 2)


def bezier_sample_degree4_torch(p0, p1, p2, p3, p4, num_points):
    """
    Differentiable degree 4 Bezier sampling. PyTorch ops only.
    B(t) = (1-t)^4*P0 + 4(1-t)^3*t*P1 + 6(1-t)^2*t^2*P2 + 4(1-t)*t^3*P3 + t^4*P4

    Samples at t=1/N, 2/N, ..., 1 so output aligns with pred_traj_gt.

    Args:
        p0: (batch, 2) start point
        p1, p2, p3, p4: (batch, 2) control points
        num_points: int, number of samples (e.g. 25)

    Returns:
        sampled: (batch, num_points, 2)
    """
    t = torch.linspace(1.0 / num_points, 1.0, num_points, device=p0.device, dtype=p0.dtype)
    t = t.unsqueeze(0).unsqueeze(-1)  # (1, num_points, 1)
    one_minus_t = 1 - t
    b0 = one_minus_t ** 4
    b1 = 4 * (one_minus_t ** 3) * t
    b2 = 6 * (one_minus_t ** 2) * (t ** 2)
    b3 = 4 * one_minus_t * (t ** 3)
    b4 = t ** 4
    
    pts = (
        b0 * p0.unsqueeze(1)
        + b1 * p1.unsqueeze(1)
        + b2 * p2.unsqueeze(1)
        + b3 * p3.unsqueeze(1)
        + b4 * p4.unsqueeze(1)
    )
    return pts


def fit_bezier_control_points_np(waypoints):
    """
    Fit cubic Bezier control points to waypoints via least squares.
    P0 = waypoints[0], P3 = waypoints[-1], solve for P1, P2.

    Args:
        waypoints: (N, 2) numpy array, N >= 2

    Returns:
        control_points: (4, 2) array [P0, P1, P2, P3]
    """
    n = len(waypoints)
    if n < 2:
        return None
    p0 = waypoints[0]
    p3 = waypoints[-1]
    if n == 2:
        return np.stack([p0, p0, p3, p3], axis=0)

    t = np.linspace(0, 1, n).astype(np.float64)
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3
    # waypoints = b0*P0 + b1*P1 + b2*P2 + b3*P3
    # waypoints - b0*P0 - b3*P3 = b1*P1 + b2*P2
    rhs = waypoints - b0[:, None] * p0 - b3[:, None] * p3
    B = np.stack([b1, b2], axis=1)  # (n, 2)
    try:
        p1p2, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
        p1, p2 = p1p2[0], p1p2[1]
        return np.stack([p0, p1, p2, p3], axis=0)
    except Exception:
        return np.stack([p0, p0, p3, p3], axis=0)
