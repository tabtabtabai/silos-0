"""Turn one MolmoSpaces observation into a policy-server request, and a server action row into a MolmoSpaces action.

Kept free of MolmoSpaces imports so it is unit-tested without the simulator.

Only what a real DROID setup provides is sent: RGB from the chosen cameras, their calibration, joint/gripper state
and the task text. MolmoSpaces also hands every policy privileged fields (object poses, masks, task_info, success);
none of them is read here.
"""
from __future__ import annotations

import numpy as np

HORIZON = 8           # the server's chunk length; batbatbat/step must advance by exactly this
GRIP_FULL = 0.824033  # MuJoCo 2F-85 driver joint at full close (as in MolmoSpaces' own pi client)


def camera_to_root(sensor_param: dict, T_root_world: np.ndarray) -> dict:
    """MolmoSpaces sensor_param_<cam> (world->camera, OpenCV) as the server's camera-to-robot-root calibration."""
    E = np.vstack([np.asarray(sensor_param["extrinsic_cv"], float).reshape(3, 4), [0, 0, 0, 1]])
    return {"frame": "robot_root", "convention": "opencv",
            "K": np.asarray(sensor_param["intrinsic_cv"], float).reshape(3, 3).tolist(),
            "T_base_camera": (np.asarray(T_root_world, float) @ np.linalg.inv(E)).tolist()}


def build_request(obs: dict, *, cams: tuple[str, str, str], T_world_root, prompt: str, session_id: str, step: int,
                  benchmark: str, slot: int | None, motion_limit_s: float) -> dict:
    """cams = (left, right, wrist): the server locates by two-view triangulation between left and right.
    On sets with one outside camera, right is the wrist camera (its calibration moves with the hand)."""
    left, right, wrist = cams
    T_root_world = np.linalg.inv(np.asarray(T_world_root, float))
    q = np.asarray(obs["qpos"]["arm"][:7], np.float32)
    grip = np.float32(np.clip(obs["qpos"]["gripper"][0] / GRIP_FULL, 0, 1))
    img = {k: np.ascontiguousarray(obs[c], dtype=np.uint8) for k, c in (("left", left), ("right", right), ("wrist", wrist))}
    # The composite the request contract requires (wrist on top, both outside views at half size). The LLM-only
    # server never reads it. All MolmoSpaces DROID cameras render at the same size, so widths match.
    composite = np.ascontiguousarray(np.concatenate(
        [img["wrist"], np.concatenate([img["left"], img["right"]], 1)[::2, ::2]], 0))
    return {
        "observation/image": composite,
        "observation/exterior_image_1_left": img["left"],
        "observation/exterior_image_2_left": img["right"],
        "observation/wrist_image_left": img["wrist"],
        "observation/joint_position": q,
        "observation/gripper_position": np.array([grip], np.float32),
        "prompt": prompt,
        "session_id": session_id,
        "batbatbat/benchmark": benchmark, "batbatbat/slot": slot, "batbatbat/env_id": 0, "batbatbat/episode": 0,
        "batbatbat/step": step,
        "batbatbat/motion_time_limit_s": float(motion_limit_s), "batbatbat/control_hz": 15.0,
        "batbatbat/calibration": {"left": camera_to_root(obs[f"sensor_param_{left}"], T_root_world),
                                  "right": camera_to_root(obs[f"sensor_param_{right}"], T_root_world)},
        "batbatbat/camera_notes": {"right": "this is the WRIST camera again; it moves with the hand"} if right == wrist else {},
    }


def to_action(row) -> dict:
    """One server row [7 joint targets, gripper 0/1] as a MolmoSpaces action (gripper 0..255, 255 closed)."""
    row = np.asarray(row, np.float32)
    return {"arm": row[:7].astype(np.float64), "gripper": np.array([255.0 if row[7] > .5 else 0.0])}
