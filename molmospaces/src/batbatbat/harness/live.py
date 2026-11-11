"""Optional live view of a rollout while it runs.

With BATBATBAT_LIVE_DIR set, the server writes the latest frame of each camera it receives (one per 32-step chunk)
and a small state file there. Writes are atomic renames, so a reader never sees half a frame. Off by default.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .geometry import CAMERA_KEYS


class LiveTap:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls):
        directory = os.environ.get("BATBATBAT_LIVE_DIR", "").strip()
        return cls(directory) if directory else None

    def _replace(self, name, data):
        tmp = self.dir / f".{name}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, self.dir / name)

    def frame(self, obs, session_id):
        from io import BytesIO
        from PIL import Image
        for side, key in CAMERA_KEYS.items():
            if key in obs:
                buf = BytesIO()
                Image.fromarray(np.asarray(obs[key], dtype=np.uint8)).save(buf, format="JPEG", quality=80)
                self._replace(f"{side}.jpg", buf.getvalue())
        self._replace("state.json", json.dumps({"time": time.time(), "session_id": session_id,
                                                "step": obs.get("batbatbat/step"), "prompt": obs.get("prompt")}).encode())
