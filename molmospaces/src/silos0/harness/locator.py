"""A second model that picks locate's image points (config `locator`), so the planner only describes them.

The planner's locate / locate_many then carry descriptions alone; this asks the locator model, with the two exterior
images, for the same feature's [x, y] in each view (0..1000, top left origin), and the harness triangulates them as
before. One request serves every point of a turn. The reply is plain JSON; points are given as {"x", "y"} objects so
a model's [y, x] habit cannot swap them.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time

import numpy as np

from .geometry import CAMERA_KEYS

PROMPT = """You find points in two photographs of the same table, taken at the same moment by two fixed cameras:
image 1 is LEFT, image 2 is RIGHT. For each description, find that ONE small physical feature (a corner, an edge
point, a rim point, a marking) and give its position in BOTH images. It must be the same 3D point in both views,
not the centre of each silhouette. Positions are {"x": .., "y": ..} with x across and y down, each 0..1000 of the
image width and height, origin at the top left. If the feature is hidden or not in view in either image, give
"error": "not visible" for that point instead of guessing.
Reply with JSON only: {"points": [{"object": "<the description, verbatim>", "left": {"x": .., "y": ..},
"right": {"x": .., "y": ..}} or {"object": "...", "error": "..."}, ...]} in the order given.

Descriptions:
"""


def jpeg_url(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Locator:
    """config locator: {provider: gemini|openai-compat, base_url, model, effort?, key_env?, timeout_s?}."""

    def __init__(self, cfg, record=lambda **row: None, post=None):
        if not isinstance(cfg, dict) or not cfg.get("base_url") or not cfg.get("model"):
            raise ValueError("locator needs base_url and model")
        if cfg.get("provider", "gemini") not in ("gemini", "openai-compat"):
            raise ValueError("locator provider must be gemini or openai-compat (an OpenAI-compatible chat endpoint)")
        self.cfg, self.record = cfg, record
        self.key = os.environ.get(cfg.get("key_env", "SILOS0_LOCATOR_API_KEY"), "")
        if not self.key and post is None:
            raise ValueError(f"locator key environment variable {cfg.get('key_env', 'SILOS0_LOCATOR_API_KEY')} is unset")
        self.post = post or self._post

    def provenance(self):
        return {k: self.cfg[k] for k in ("provider", "model", "effort") if k in self.cfg}

    def _post(self, body):
        import requests
        r = requests.post(self.cfg["base_url"].rstrip("/") + "/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {self.key}"}, timeout=self.cfg.get("timeout_s", 90))
        r.raise_for_status()
        return r.json()

    def pick(self, obs, descriptions):
        """[(left_xy, right_xy) or error string] for each description, from one locator request."""
        content = [{"type": "text", "text": PROMPT + "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))}]
        for side in ("left", "right"):
            content.append({"type": "image_url", "image_url": {"url": jpeg_url(obs[CAMERA_KEYS[side]])}})
        body = {"model": self.cfg["model"], "messages": [{"role": "user", "content": content}]}
        if self.cfg.get("effort"):
            body["reasoning_effort"] = self.cfg["effort"]
        started, reply, error = time.perf_counter(), None, None
        try:
            reply = self.post(body)
            text = reply["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
            found = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))["points"]
        except Exception as exc:  # noqa: BLE001 -- a locator failure is a per-call error for the planner
            error = f"{type(exc).__name__}: {exc}"[:300]
            found = []
        finally:
            self.record(event="locator_request", latency_s=time.perf_counter()-started, model=self.cfg["model"],
                        descriptions=descriptions, usage=(reply or {}).get("usage") if isinstance(reply, dict) else None,
                        error=error, picks=found if error is None else None)
        out = []
        for i, description in enumerate(descriptions):
            p = found[i] if i < len(found) and isinstance(found[i], dict) else None
            if error:
                out.append(f"locator failed: {error}")
            elif p is None:
                out.append("locator returned no point")
            elif p.get("error"):
                out.append(f"locator: {p['error']}")
            else:
                try:
                    out.append(([float(p["left"]["x"]), float(p["left"]["y"])],
                                [float(p["right"]["x"]), float(p["right"]["y"])]))
                except (KeyError, TypeError, ValueError):
                    out.append("locator returned a malformed point")
        return out
