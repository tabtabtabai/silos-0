"""RGB-only stereo grounding. Calibration is camera-to-robot-root, OpenCV axes.

Only rendered RGB, camera calibration and robot proprioception cross this seam.
No object poses, segmentation, task predicates, depth or table-plane assumptions.
"""
from __future__ import annotations

import numpy as np

CAMERA_KEYS = {"left": "observation/exterior_image_1_left",
               "right": "observation/exterior_image_2_left",
               "wrist": "observation/wrist_image_left"}
CALIBRATION_VERSION = "robolab-ad45d4f-default-exteriors"


def finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


def rotation(wxyz):
    q = finite(wxyz, (4,), "quaternion")
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("zero quaternion")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def calibration(obs, side, overrides=None):
    # A simulator other than pinned RoboLab may send its own calibration per request.
    c = (overrides or obs.get("batbatbat/calibration") or {}).get(side)
    if c is None:
        # The existing RoboLab request has no calibration fields. These are the
        # fixed default exterior mounts and intrinsics from pinned RoboLab's
        # robolab/variations/camera.py; no scene objects or sim state are used.
        image = np.asarray(obs[CAMERA_KEYS[side]])
        if image.shape != (720, 1280, 3):
            raise ValueError("pinned exterior calibration requires native 1280x720 RGB")
        return default_calibration(side)
    if c.get("frame") != "robot_root" or c.get("convention") != "opencv":
        raise ValueError("camera calibration must be OpenCV camera-to-robot-root")
    K = finite(c["K"], (3, 3), "K")
    T = finite(c["T_base_camera"], (4, 4), "T_base_camera")
    R = T[:3, :3]
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-5)
            or not np.allclose(T[3], [0, 0, 0, 1])):
        raise ValueError("invalid camera calibration")
    return K, T


def default_calibration(side):
    if side not in ("left", "right"):
        raise ValueError("only exterior cameras have pinned calibration")
    position, quaternion = (([.05, .57, .66], [-.393, -.195, .399, .805]) if side == "left"
                            else ([.05, -.57, .66], [.805, .399, -.195, -.393]))
    T = np.eye(4)
    # USD/OpenGL camera: -Z forward, +Y up. Pinhole/OpenCV: +Z forward, +Y down.
    T[:3, :3] = rotation(quaternion) @ np.diag([1., -1., -1.])
    T[:3, 3] = position
    # focal_length=2.1, aperture=5.376x3.024, raster=1280x720.
    K = np.array([[500., 0, 640.], [0, 500., 360.], [0, 0, 1.]])
    return K, T


def locate(obs, left_xy, right_xy, *, calibrations=None):
    """Triangulate corresponding normalized [x,y] points (0..1000), or reject.

    A point must be the SAME visible physical feature in both views, not two
    bounding-box centres. Errors quantify geometry, not semantic correctness.
    """
    rays, cameras, pixels = [], [], []
    for side, xy in (("left", left_xy), ("right", right_xy)):
        uv = finite(xy, (2,), side + "_xy")
        if np.any(uv < 0) or np.any(uv > 1000):
            raise ValueError("point coordinates must be in 0..1000")
        image = np.asarray(obs[CAMERA_KEYS[side]])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected RGB camera image")
        h, w = image.shape[:2]
        pixel = uv * np.array([w - 1, h - 1]) / 1000
        K, T = calibration(obs, side, calibrations)
        ray = T[:3, :3] @ np.linalg.solve(K, [*pixel, 1])
        ray /= np.linalg.norm(ray)
        rays.append(ray)
        cameras.append((K, T))
        pixels.append(pixel)
    a, b = (c[1][:3, 3] for c in cameras)
    if np.linalg.norm(a - b) < 0.05:
        raise ValueError("stereo baseline below 5 cm")
    angle = float(np.arccos(np.clip(abs(rays[0] @ rays[1]), 0, 1)))
    if angle < np.deg2rad(5):
        raise ValueError("stereo rays nearly parallel")
    distances = np.linalg.lstsq(np.column_stack((rays[0], -rays[1])), b-a, rcond=None)[0]
    if np.any(distances <= 0) or np.any(distances > 3):
        raise ValueError("point behind camera or beyond 3 m")
    p, q = a + distances[0]*rays[0], b + distances[1]*rays[1]
    gap = float(np.linalg.norm(p-q))
    xyz = (p+q)/2
    errors = []
    for (K, T), pixel in zip(cameras, pixels):
        local = T[:3, :3].T @ (xyz-T[:3, 3])
        if local[2] <= 0:
            raise ValueError("point behind camera")
        projected = K @ local
        errors.append(float(np.linalg.norm(projected[:2]/projected[2]-pixel)))
    if gap > 0.025 or max(errors) > 12:
        raise ValueError(f"inconsistent points: ray gap={gap:.3f} m, reprojection={max(errors):.1f} px")
    return {"ok": True, "xyz": xyz.tolist(), "frame": "robot_root", "ray_gap_m": gap,
            "reprojection_px": errors, "ray_angle_deg": float(np.rad2deg(angle)),
            "semantic_match_verified": False}
