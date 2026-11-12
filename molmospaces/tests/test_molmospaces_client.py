"""The MolmoSpaces client (request building, calibration, actions), the robot options it relies on, and the slot list."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # client/ lives next to src/, outside the installed package
    sys.path.insert(0, str(ROOT))

from silos0.harness.geometry import CAMERA_KEYS, calibration, locate  # noqa: E402
from silos0.harness.kinematics import Franka  # noqa: E402
from silos0.submission import MOLMOSPACES_ROBOT  # noqa: E402
from client.request import GRIP_FULL, HORIZON, build_request, camera_to_root, to_action  # noqa: E402

BENCHMARK = ROOT / "benchmark/molmospaces-hundred.json"
H, W = 352, 624


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def look_at(eye, target):
    """World->camera extrinsic (OpenCV: +z forward, +y down) for a camera at eye looking at target."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return np.column_stack([R, -R @ eye])


def sensor(eye, target, f=450.):
    return {"intrinsic_cv": [[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]],
            "extrinsic_cv": look_at(np.asarray(eye, float), target).tolist()}


def root_pose(xyz=(2.0, -1.0, 0.3), yaw=0.7):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = xyz
    return T


def ms_obs():
    T_world_root = root_pose()
    target_world = (T_world_root @ [0.6, 0.1, 0.15, 1])[:3]
    img = lambda: np.zeros((H, W, 3), np.uint8)
    return {"exo_camera_1": img(), "wrist_camera": img(),
            "sensor_param_exo_camera_1": sensor(T_world_root[:3, 3] + [0.1, 0.6, 0.7], target_world),
            "sensor_param_wrist_camera": sensor(T_world_root[:3, 3] + [0.45, -0.1, 0.6], target_world),
            "qpos": {"arm": [0.0, -0.8, 0.0, -2.3, 0.0, 1.6, 0.0, 9.9], "gripper": [GRIP_FULL / 2, GRIP_FULL / 2]},
            "object_image_points": "privileged", "task_info": "privileged"}, T_world_root


def request(obs, T_world_root, cams=("exo_camera_1", "wrist_camera", "wrist_camera"), slot=7, step=16):
    return build_request(obs, cams=cams, T_world_root=T_world_root, prompt="pick up the kettle.", session_id="s",
                         step=step, benchmark="molmospaces-hundred/pick-v1.5", slot=slot, motion_limit_s=20)


def test_request_carries_state_calibration_and_no_privileged_fields():
    obs, T = ms_obs()
    req = request(obs, T)
    assert req["observation/joint_position"].shape == (7,)
    assert req["observation/gripper_position"][0] == pytest.approx(0.5, abs=1e-6)
    assert req["silos0/step"] == 16 and req["silos0/slot"] == 7 and req["silos0/motion_time_limit_s"] == 20
    assert set(req["silos0/calibration"]) == {"left", "right"}
    assert req["silos0/camera_notes"]["right"].startswith("this is the WRIST camera")
    assert req["observation/image"].shape == (H + H // 2, W, 3)
    assert not any("privileged" in str(v) for v in req.values() if isinstance(v, str))
    distinct = request(obs, T, cams=("exo_camera_1", "wrist_camera", "exo_camera_1"))
    assert distinct["silos0/camera_notes"] == {}


def test_camera_to_root_is_the_camera_pose_in_the_arm_frame():
    obs, T_world_root = ms_obs()
    cal = camera_to_root(obs["sensor_param_exo_camera_1"], np.linalg.inv(T_world_root))
    T = np.asarray(cal["T_base_camera"])
    assert np.allclose((T_world_root @ T)[:3, 3], T_world_root[:3, 3] + [0.1, 0.6, 0.7])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9)
    assert cal["frame"] == "robot_root" and cal["convention"] == "opencv"


def test_server_triangulates_with_calibration_from_the_request():
    obs, T_world_root = ms_obs()
    req = request(obs, T_world_root)
    target = np.array([0.6, 0.1, 0.15])
    xy = {}
    for side in ("left", "right"):
        K, T = calibration(req, side)
        local = T[:3, :3].T @ (target - T[:3, 3])
        pixel = (K @ local)[:2] / (K @ local)[2]
        xy[side] = pixel * 1000 / np.array([W - 1, H - 1])
    assert np.allclose(locate(req, xy["left"], xy["right"])["xyz"], target, atol=1e-6)


def test_without_request_calibration_the_default_calibration_needs_its_native_size():
    obs, T = ms_obs()
    req = request(obs, T)
    del req["silos0/calibration"]
    with pytest.raises(ValueError, match="1280x720"):
        calibration(req, "left")
    assert CAMERA_KEYS["left"] in req


def test_action_row_maps_to_molmospaces_gripper_scale():
    act = to_action([0.1] * 7 + [0.9])
    assert act["arm"].shape == (7,) and act["gripper"][0] == 255
    assert to_action([0.0] * 7 + [0.2])["gripper"][0] == 0
    assert HORIZON == 8


def test_joint_limits_keep_only_the_overlap(monkeypatch):
    base = Franka().limits.copy()
    fr3 = json.loads(MOLMOSPACES_ROBOT["SILOS0_JOINT_LIMITS_JSON"])
    monkeypatch.setenv("SILOS0_JOINT_LIMITS_JSON", json.dumps(fr3))
    limits = Franka().limits
    assert np.allclose(limits[:, 0], np.maximum(base[:, 0], np.asarray(fr3)[:, 0]))
    assert np.allclose(limits[:, 1], np.minimum(base[:, 1], np.asarray(fr3)[:, 1]))
    assert limits[5, 0] == pytest.approx(0.54)


def test_fingertip_offset_follows_the_environment():
    code = "import silos0.harness.policy as p; print(p.FINGERTIP_OFFSET_M)"
    env = {**os.environ, "SILOS0_FINGERTIP_OFFSET_M": MOLMOSPACES_ROBOT["SILOS0_FINGERTIP_OFFSET_M"]}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert float(out.stdout) == pytest.approx(0.141)


def test_benchmark_slots_make_one_job_each():
    jobs_mod = script("jobs")
    benchmark = json.loads(BENCHMARK.read_text())
    jobs = jobs_mod.lines(benchmark, Path("/assets"))
    assert len(jobs) == 100
    job, set_name, bench, slot, bdir, idx, horizon, cams = jobs[0].split()
    assert (job, set_name, slot) == ("slot-001", "molmospaces-hundred", "1")
    assert bdir.startswith("/assets/benchmarks/") and len(cams.split(",")) == 3
    counts = {}
    for s in benchmark["slots"]:
        counts[s["benchmark"]] = counts.get(s["benchmark"], 0) + 1
    assert sorted(counts.values()) == [12, 12, 12, 12, 13, 13, 13, 13]
    assert len(jobs_mod.lines(benchmark, Path("/a"), jobs_mod.parse_slots("1-3,40"))) == 4
