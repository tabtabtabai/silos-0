"""Send a synthetic RGB/proprioception request; this is not a task evaluation."""
import argparse
import uuid

import numpy as np
from websockets.sync.client import connect

from .harness.geometry import CAMERA_KEYS
from .harness.wire import pack, unpack


def fake_observation():
    obs = {key: np.zeros((720, 1280, 3), dtype=np.uint8) for key in CAMERA_KEYS.values()}
    obs.update({
        "observation/image": np.zeros((540, 640, 3), dtype=np.uint8),
        "observation/joint_position": np.array([0, -0.6, 0, -2.2, 0, 1.6, 0.8], dtype=np.float32),
        "observation/gripper_position": np.array([0], dtype=np.float32),
        "prompt": "Synthetic connection test: hold the current pose. The images are blank.",
        "session_id": "probe-" + uuid.uuid4().hex,
        "silos0/env_id": 0, "silos0/episode": 0, "silos0/step": 0,
        "silos0/control_hz": 15, "silos0/motion_time_limit_s": 90,
    })
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8001")
    args = parser.parse_args()
    with connect(args.url, compression=None, max_size=64*1024*1024, ping_interval=None) as ws:
        metadata = unpack(ws.recv(timeout=30))
        ws.send(pack(fake_observation()))
        result = unpack(ws.recv(timeout=600))
    if result.get("type") == "error":
        raise RuntimeError(result["message"])
    actions = np.asarray(result["action"])
    if actions.shape != (8, 8) or not np.isfinite(actions).all():
        raise ValueError("Expected a finite 8 x 8 action chunk")
    print(f"Connected to {metadata['policy']}; received {actions.shape} finite action chunk")


if __name__ == "__main__":
    main()
