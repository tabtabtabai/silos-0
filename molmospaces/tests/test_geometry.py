import numpy as np
import pytest
from silos0.harness.geometry import CAMERA_KEYS, locate
from silos0.harness.kinematics import Franka, rotvec, exp_rotation
Q = np.array([0, -np.pi/5, 0, -4*np.pi/5, 0, 3*np.pi/5, 0])
def obs(sid="a", step=0):
    image = np.zeros((720, 1280, 3), np.uint8)
    cameras = {}
    for side, x in (("left", -.3), ("right", .3)):
        T = np.eye(4)
        T[0, 3] = x
        cameras[side] = {"K": np.array([[500, 0, 640], [0, 500, 360], [0, 0, 1]]),
                         "T_base_camera": T, "frame": "robot_root", "convention": "opencv"}
    return {"session_id": sid, "prompt": "put the cube in the bowl", "silos0/step": step,
            "silos0/slot": 1, "silos0/env_id": 0, "silos0/episode": 0, "silos0/benchmark": "test.v1",
            "observation/joint_position": Q.copy(), "observation/gripper_position": np.array([0.]),
            "observation/image": image, **{key: image for key in CAMERA_KEYS.values()},
            "observation/camera_calibration": cameras}

def project(o, point):
    result = []
    for side in ("left", "right"):
        c = o["observation/camera_calibration"][side]
        T = c["T_base_camera"]
        local = T[:3, :3].T @ (point-T[:3, 3])
        pixel = c["K"] @ local
        result.append(pixel[:2]/pixel[2]/[1279, 719]*1000)
    return result

def test_stereo_reconstructs_and_checks_geometry():
    o = obs()
    point = np.array([.05, .02, 1.])
    left, right = project(o, point)
    cameras = o["observation/camera_calibration"]
    result = locate(o, left, right, calibrations=cameras)
    np.testing.assert_allclose(result["xyz"], point, atol=1e-10)
    assert max(result["reprojection_px"]) < 1e-9
    with pytest.raises(ValueError, match="inconsistent"):
        locate(o, left, right+[0, 100], calibrations=cameras)
    with pytest.raises(ValueError, match="parallel"):
        locate(o, [500, 500], [500, 500], calibrations=cameras)
    with pytest.raises(ValueError, match="0..1000"):
        locate(o, [-1, 200], right, calibrations=cameras)
    o["observation/camera_calibration"]["left"]["convention"] = "opengl"
    with pytest.raises(ValueError, match="OpenCV"):
        locate(o, left, right, calibrations=cameras)

def test_ik_path_reaches_pose_with_bounded_speed():
    robot = Franka()
    initial, R = robot.fk(Q)
    path = robot.path(Q, initial+[0, 0, .05])
    p, actual_R = robot.fk(path[-1])
    assert len(path) % 32 == 0
    np.testing.assert_allclose(p, initial+[0, 0, .05], atol=.0015)
    assert np.linalg.norm(rotvec(actual_R@R.T)) < .01
    assert abs(np.diff(np.vstack((Q, path)), axis=0)).max() <= .06
    assert np.all(path >= robot.limits[:, 0]) and np.all(path <= robot.limits[:, 1])
    with pytest.raises(ValueError, match="workspace"):
        robot.path(Q, [4, 0, 0])
    # Near-pi orientation errors must remain finite and nonzero.
    assert np.linalg.norm(rotvec(exp_rotation(np.array([np.pi, 0, 0])))) == pytest.approx(np.pi)
