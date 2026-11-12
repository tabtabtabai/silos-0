"""DROID static-asset FK and bounded DLS IK, independent of Isaac and task state.

The checked-in joint frames are extracted from RoboLab's pinned USD. We control
the Robotiq base_link origin, matching RoboLab's ee_pos/ee_quat. The asset's
base_link axes differ from conventional Robotiq diagrams; no guessed fingertip
offset is used. Actual contact depends on the gripper aperture.
These trajectories are not collision plans; the planner checks contact from images and measured state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .geometry import finite, rotation

MODEL_PATH = Path(__file__).with_name("droid_kinematics.json")
FPS = 15
TCP_OFFSET = np.zeros(3)


def transform(frame):
    T = np.eye(4)
    T[:3, :3] = rotation(frame["wxyz"])
    T[:3, 3] = frame["xyz"]
    return T


def rotvec(R):
    theta = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    if theta < 1e-7:
        return v/2
    if np.pi-theta < 1e-5:
        _, axes = np.linalg.eigh((R+R.T+2*np.eye(3))/4)
        axis = axes[:, -1]
        if axis @ v < 0:
            axis = -axis
        return theta*axis
    return v*theta/(2*np.sin(theta))


def exp_rotation(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = v/theta
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


class Franka:
    def __init__(self):
        raw = MODEL_PATH.read_bytes()
        self.model = json.loads(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.chain = [(transform(j["parent_frame"]), np.linalg.inv(transform(j["child_frame"])),
                       j.get("axis")) for j in self.model["joints"]]
        self.limits = np.array([j["limits"] for j in self.model["joints"] if "axis" in j])
        if self.limits.shape != (7, 2):
            raise ValueError("expected seven Franka joints")
        extra = os.environ.get("SILOS0_JOINT_LIMITS_JSON")  # e.g. an FR3: keep only the overlap of both ranges
        if extra:
            other = np.asarray(json.loads(extra), dtype=float)
            self.limits = np.column_stack((np.maximum(self.limits[:, 0], other[:, 0]),
                                           np.minimum(self.limits[:, 1], other[:, 1])))

    def fk(self, q, *, tcp=True):
        q = finite(q, (7,), "joints")
        T, i = np.eye(4), 0
        for parent, child_inv, axis in self.chain:
            motion = np.eye(4)
            if axis:
                direction = np.eye(3)["XYZ".index(axis)]
                motion[:3, :3] = exp_rotation(direction*q[i])
                i += 1
            T = T @ parent @ motion @ child_inv
        if tcp:
            T[:3, 3] += T[:3, :3] @ TCP_OFFSET
        return T[:3, 3], T[:3, :3]

    def solve(self, xyz, R, seed):
        q = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
        for _ in range(100):
            p, r = self.fk(q)
            ep, er = xyz-p, rotvec(R @ r.T)
            if np.linalg.norm(ep) < .001 and np.linalg.norm(er) < .008:
                return q
            J = np.zeros((6, 7))
            for j in range(7):
                shifted = q.copy()
                shifted[j] += 1e-5
                p1, r1 = self.fk(shifted)
                J[:3, j] = (p1-p)/1e-5
                J[3:, j] = .2*rotvec(r1 @ r.T)/1e-5
            error = np.r_[ep, .2*er]
            delta = J.T @ np.linalg.solve(J @ J.T + 1e-5*np.eye(6), error)
            delta *= min(1., .12/max(np.linalg.norm(delta), 1e-12))
            q = np.clip(q+delta, self.limits[:, 0], self.limits[:, 1])
        p, r = self.fk(q)
        raise ValueError(f"IK unreachable: position error {np.linalg.norm(xyz-p):.4f} m; "
                         f"rotation error {np.linalg.norm(rotvec(R@r.T)):.3f} rad")

    def path(self, seed, xyz, quaternion=None, duration_s=3., horizon=32, max_move=.35, stretch=False,
             max_joint_step=.06, duration_scale=1., enforce_speed_limit=True):
        """Joint rows for a smooth straight-line move. stretch lengthens duration (up to 16 s)
        to respect max_joint_step. Motor tools disable that guard and verify achieved
        pose within their original time budget; grasp routines retain their tracking margin."""
        if not np.isfinite(max_joint_step) or not 0 < max_joint_step <= .06:
            raise ValueError("max_joint_step must be in (0,0.06] radians per control step")
        seed = finite(seed, (7,), "joints")
        xyz = finite(xyz, (3,), "xyz")
        p0, R0 = self.fk(seed)
        if (np.any(xyz < [-.2, -.8, -.05]) or np.any(xyz > [.9, .8, 1.2])
                or np.linalg.norm(xyz-p0) > max_move):
            raise ValueError(f"target outside workspace or move exceeds {max_move:g} m; use smaller moves")
        if not np.isfinite(duration_s) or not 1 <= duration_s <= 16:
            raise ValueError("duration_s must be in [1,16]")
        target_R = R0 if quaternion is None else rotation(quaternion)
        turn = rotvec(target_R @ R0.T)
        if np.linalg.norm(turn) > np.pi/2+.001:
            raise ValueError("rotation exceeds 90 degrees; use smaller moves")
        if not np.isfinite(duration_scale) or duration_scale < 1:
            raise ValueError("duration_scale must be finite and >= 1")
        ticks = max(1, int(np.ceil(float(duration_s)/duration_scale*FPS)))
        while True:
            q = seed.copy()
            rows, fastest = [], 0.
            for i in range(1, ticks+1):
                f = i/ticks
                # Smooth endpoint velocities; preserve orientation for move_to.
                f = f*f*(3-2*f)
                next_q = self.solve(p0+f*(xyz-p0), exp_rotation(f*turn) @ R0, q)
                fastest = max(fastest, float(np.max(abs(next_q-q))))
                rows.append(next_q)
                q = next_q
            if not enforce_speed_limit or fastest <= max_joint_step:
                break
            if not stretch or ticks >= 16*FPS:
                raise ValueError(f"joint path exceeds {max_joint_step*FPS:g} rad/s; increase duration_s or use a smaller move")
            ticks = min(16*FPS, int(np.ceil(ticks*fastest/max_joint_step*1.1)))
        while len(rows) % horizon:
            rows.append(q.copy())
        return np.asarray(rows, dtype=np.float32)
