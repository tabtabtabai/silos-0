"""MolmoSpaces policy that drives the silos0 policy server (silos-0-serve) over its websocket.

Pass `client.policy:HarnessEvalConfig` to MolmoSpaces' eval_main.py; scripts/run_slot.sh does this for one slot.

Runs inside a MolmoSpaces v0.2.9 environment with this directory and its `src` on PYTHONPATH. Settings come
from the environment, one episode per eval_main call:
  HARNESS_URL            policy server websocket (default ws://127.0.0.1:8001)
  HARNESS_CAMS           "left,right,wrist" MolmoSpaces camera names (see request.build_request)
  HARNESS_MOTION_LIMIT_S the episode's robot-time limit, for the server's motion-budget note
  HARNESS_BENCH          benchmark label recorded by the server
  HARNESS_SLOT           benchmark slot, recorded in the session id
"""
from __future__ import annotations

import os
import uuid

import numpy as np
from websockets.sync.client import connect

import client.fixes  # noqa: F401  (MolmoSpaces v0.2.9 articulated-object loader fix)
from silos0.harness.wire import pack, unpack
from client.request import HORIZON, build_request, to_action
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.policy.base_policy import BasePolicy


class HarnessPolicy(BasePolicy):
    @property
    def type(self):
        return "learned"

    def reset(self):
        self.queue, self.step = [], 0
        slot = os.environ.get("HARNESS_SLOT")
        self.session = (f"molmo-s{int(slot):03d}-" if slot else "molmo-") + uuid.uuid4().hex
        if getattr(self, "ws", None) is None:
            self.ws = connect(os.environ.get("HARNESS_URL", "ws://127.0.0.1:8001"), max_size=64 * 1024 * 1024,
                              open_timeout=60, close_timeout=5)
            meta = unpack(self.ws.recv())
            if meta.get("open_loop_horizon") != HORIZON:
                raise RuntimeError(f"server open_loop_horizon {meta.get('open_loop_horizon')}, client expects {HORIZON}")

    def get_action(self, observation):
        obs = observation[0] if isinstance(observation, list) else observation
        if not hasattr(self, "queue"):
            self.reset()
        if not self.queue:
            cams = tuple(os.environ.get("HARNESS_CAMS", "exo_camera_1,wrist_camera,wrist_camera").split(","))
            arm = self.task.env.current_robot.robot_view.get_move_group("arm")  # robot kinematics, not scene state
            slot = os.environ.get("HARNESS_SLOT")
            req = build_request(obs, cams=cams, T_world_root=arm.root_frame_to_world,
                                prompt=self.task.get_task_description(), session_id=self.session, step=self.step,
                                benchmark=os.environ.get("HARNESS_BENCH", "molmospaces"),
                                slot=int(slot) if slot else None,
                                motion_limit_s=float(os.environ.get("HARNESS_MOTION_LIMIT_S", "30")))
            self.ws.send(pack(req))
            reply = unpack(self.ws.recv())
            if not isinstance(reply, dict) or "action" not in reply:
                raise RuntimeError(f"policy server error: {reply}")
            self.queue = list(np.asarray(reply["action"], np.float32)[:HORIZON])
            self.step += HORIZON
        return to_action(self.queue.pop(0))


class HarnessPolicyConfig(BasePolicyConfig):
    policy_type: str = "learned"
    policy_cls: type = None
    policy_factory: object = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.policy_cls = HarnessPolicy
        self.policy_factory = HarnessPolicy


class HarnessEvalConfig(JsonBenchmarkEvalConfig):
    """Each episode's own task_horizon_sec applies (eval_main's default); 66 ms steps; stop at first success."""
    policy_dt_ms: float = 66.0
    end_on_success: bool = True
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: HarnessPolicyConfig = HarnessPolicyConfig()

    @property
    def tag(self):
        return "harness"

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
