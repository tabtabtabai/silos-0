"""Serve the tool harness behind the RoboLab-style policy contract. Start it with batbatbat-serve (submission.py)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import resolve_config
from .kinematics import Franka, TCP_OFFSET
from .live import LiveTap
from .locator import Locator
from .geometry import CALIBRATION_VERSION
from .planner import make_planner, tools
from .policy import Episode, GRASP_AIDS, HORIZON, validate_obs, execution_settings, MOVE_POSITION_TOLERANCE_M, MOVE_ROTATION_TOLERANCE_RAD, MOVE_STALL_CHUNKS, MOVE_MIN_PROGRESS, MOVE_NEAR_STALL_CHUNKS, MOVE_NEAR_POSITION_M, MOVE_NEAR_ROTATION_RAD
from .wire import pack, unpack

log = logging.getLogger(__name__)


class DecisionLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", buffering=1)
        self.lock = threading.Lock()

    def __call__(self, **row):
        with self.lock:
            self.file.write(json.dumps({"time": time.time(), **row}, allow_nan=False) + "\n")

    def close(self):
        self.file.close()


@dataclass
class Session:
    episode: Episode
    identity: tuple
    prompt: str
    last_step: int = -HORIZON
    last_reply: dict | None = None
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    failed: bool = False


class Service:
    def __init__(self, config, record, *, planner_factory=make_planner,
                 max_sessions=128, idle_seconds=3600, warmup=True):
        config = resolve_config(config)
        if config.get("mode") != "llm-only" or config.get("locator"):
            raise ValueError("This server runs the submitted LLM-only policy without auxiliary models")
        self.horizon, self.move_speed_multiplier, self.gripper_settle_steps = execution_settings(config)
        if config.get("locator") is not None:
            Locator(config["locator"], post=lambda body: {})  # validates the config without a key or request
        for flag in ("pace_note", "move_path", "no_done", "placement_note", "carry_note", "motion_urgency"):
            if not isinstance(config.get(flag, False), bool):
                raise ValueError(f"{flag} is true/false")
        if not isinstance(config.get("multi_locate", False), bool):
            raise ValueError("multi_locate is true/false")
        aids = config.get("grasp_aids")
        if aids is not None:
            if (not isinstance(aids, dict) or set(aids) - set(GRASP_AIDS)
                    or not all(isinstance(v, bool) for v in aids.values())):
                raise ValueError(f"grasp_aids takes true/false for {list(GRASP_AIDS)}")
            if aids.get("closeup") and not aids.get("fingertips"):
                raise ValueError("grasp_aids.closeup needs fingertips: the crops are centred on them")
        self.config, self.record = config, record
        self.live = LiveTap.from_env()
        self.planner_factory = planner_factory
        self.robot = Franka()
        self.sessions = {}
        self.registry_lock = threading.Lock()
        self.max_sessions, self.idle_seconds = max_sessions, idle_seconds
        planner = planner_factory(config)
        if warmup:
            planner.ping()  # actual configured provider and tool-bearing wire; fail before listen
        revision = os.environ.get("BATBATBAT_GIT_SHA")
        if not revision:
            try:
                revision = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[3]),
                                                    "rev-parse", "HEAD"], text=True,
                                                   stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = "unknown"
        source_digest = hashlib.sha256()
        package = Path(__file__).parent
        sources = [*package.glob("*.py"), package/"droid_kinematics.json",
                   *sorted(p for p in (package.parent/"harness_knowledge").rglob("*.*") if "__pycache__" not in p.parts)]
        for path in sorted(sources):
            source_digest.update(str(path.relative_to(package.parent)).encode())
            source_digest.update(path.read_bytes())
        self.metadata = {
            "policy": "batbatbat-harness-llm", "control_space": "joint_position", "action_dim": 8,
            "orchestrator_kind": "harness",
            "policy_checkpoint": config.get("policy_checkpoint"),
            "open_loop_horizon": self.horizon, "control_hz": 15,
            "move_speed_multiplier": self.move_speed_multiplier,
            "gripper_settle_steps": self.gripper_settle_steps,
            "move_completion": {"deadline": "requested_duration_chunk_rounded",
                                "position_tolerance_m": MOVE_POSITION_TOLERANCE_M,
                                "orientation_tolerance_rad": MOVE_ROTATION_TOLERANCE_RAD,
                                "stall_chunks": MOVE_STALL_CHUNKS,
                                "near_target_stall_chunks": MOVE_NEAR_STALL_CHUNKS,
                                "near_target_position_m": MOVE_NEAR_POSITION_M,
                                "near_target_orientation_rad": MOVE_NEAR_ROTATION_RAD,
                                "min_progress_tolerance_units": MOVE_MIN_PROGRESS,
                                "progress_metric": "sum_of_normalized_excess_pose_errors",
                                "joint_speed_limit_enforced": False},
            "mode": "llm-only", "task": config.get("task"), "planner": planner.provenance(),
            **({"grasp_aids": {k: bool(aids.get(k)) for k in GRASP_AIDS}} if aids else {}),
            **({"multi_locate": True} if config.get("multi_locate") else {}),
            **({k: True for k in ("pace_note", "move_path", "no_done", "placement_note", "carry_note", "motion_urgency") if config.get(k)}),
            **({"locator": Locator(config["locator"], post=lambda body: {}).provenance()} if config.get("locator") else {}),
            **({"knowledge": planner.knowledge} if getattr(planner, "knowledge", None) else {}),
            **({"few_calls": True} if config.get("few_calls") else {}),
            "planner_prompt_sha256": hashlib.sha256(planner.system_prompt().encode()).hexdigest(),
            "tools": [t["name"] for t in tools(multi_locate=bool(config.get("multi_locate")),
                                                     locator=config.get("locator") is not None,
                                                     move_path=bool(config.get("move_path")),
                                                     no_done=bool(config.get("no_done")))],
            "perception": "rgb-stereo-correspondence",
            "camera_calibration": CALIBRATION_VERSION,
            "kinematics_sha256": self.robot.sha256, "robot_asset_sha256": self.robot.model["asset_sha256"],
            "tcp_offset_base_link_m": TCP_OFFSET.tolist(), "gripper": "0=open,1=closed",
            "source_revision": revision, "source_content_sha256": source_digest.hexdigest(),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        }
        record(event="server_started", metadata=self.metadata)

    def _session(self, obs):
        sid = obs.get("session_id")
        if not isinstance(sid, str) or not sid or len(sid) > 200:
            raise ValueError("session_id must be a nonempty string of at most 200 characters")
        identity = tuple(obs.get(k) for k in ("batbatbat/benchmark", "batbatbat/slot", "batbatbat/env_id", "batbatbat/episode"))
        key = (sid, identity)
        # Hand-runs have no benchmark slot and reuse run/env IDs across tasks.
        # Keep each instruction's history separate; numbered benchmark slots
        # retain strict task identity checks. A new task must still start at 0.
        if obs.get("batbatbat/slot") is None:
            key = (sid, identity, obs["prompt"])
        now = time.monotonic()
        with self.registry_lock:
            for old_key, session in list(self.sessions.items()):
                if now-session.last_used > self.idle_seconds and session.lock.acquire(blocking=False):
                    try:
                        del self.sessions[old_key]
                    finally:
                        session.lock.release()
            session = self.sessions.get(key)
            if session is None:
                if obs["batbatbat/step"] != 0:
                    raise ValueError("unknown/expired session must start at batbatbat/step=0 with a new id")
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError("session capacity reached; active episodes cannot be evicted")
                planner = self.planner_factory(self.config)
                episode = Episode(planner, self.robot,
                                  record=lambda **row: self.record(session_id=sid, **row),
                                  options={"open_loop_horizon": self.horizon,
                                           "move_speed_multiplier": self.move_speed_multiplier,
                                           "multi_locate": bool(self.config.get("multi_locate")),
                                           "move_path": bool(self.config.get("move_path")),
                                           "no_done": bool(self.config.get("no_done")),
                                           "motion_urgency": self.config.get("motion_urgency", False)},
                                  opening="batbatbat/task_context" if self.config.get("task") == "grasp" else None,
                                  aids=self.config.get("grasp_aids"),
                                  locator=(Locator(self.config["locator"],
                                                   record=lambda **row: self.record(session_id=sid, **row))
                                           if self.config.get("locator") else None))
                episode.record = lambda **row: self.record(session_id=sid, identity=identity,
                                                           episode_id=episode.episode_id, **row)
                planner.record = episode.record
                bind_trace = getattr(planner, "set_trace_context", None)
                if callable(bind_trace):
                    bind_trace(session_id=episode.episode_id, client_session_id=sid,
                               benchmark=identity[0], slot=identity[1], env_id=identity[2], episode=identity[3],
                               mode="llm-only", source_revision=self.metadata["source_revision"],
                               source_content_sha256=self.metadata["source_content_sha256"],
                               config_sha256=self.metadata["config_sha256"])
                session = self.sessions[key] = Session(episode, identity, obs["prompt"], last_step=-self.horizon)
                self.record(event="session_started", session_id=sid, identity=identity, prompt=obs["prompt"],
                            episode_id=episode.episode_id)
            if identity != session.identity or obs["prompt"] != session.prompt:
                raise ValueError("session identity/task changed: use a new session_id")
            # Reserve before releasing the registry lock so idle cleanup cannot
            # detach a session whose request has already been admitted.
            if not session.lock.acquire(blocking=False):
                raise ValueError("concurrent request for the same session")
            session.last_used = now
            return session

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("observation must be a dictionary")
        validate_obs(obs)
        step = obs.get("batbatbat/step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("batbatbat/step must be a nonnegative integer")
        session = self._session(obs)
        if self.live is not None:
            try:
                self.live.frame(obs, obs.get("session_id"))
            except Exception:
                log.exception("live view frame write failed")  # a viewer never stops an episode
        try:
            if session.failed:
                raise ValueError("session failed; restart the episode with a new session_id")
            if step == session.last_step:
                return session.last_reply  # retransmit after a lost reply without rerunning tools
            if step != session.last_step + self.horizon:
                raise ValueError(f"batbatbat/step must advance by exactly open_loop_horizon={self.horizon}")
            try:
                bind_trace = getattr(session.episode.planner, "set_trace_context", None)
                if callable(bind_trace):
                    bind_trace(control_step=step)
                started = time.perf_counter()
                reply = session.episode.infer(obs)
                session.episode.record(event="policy_request", step=step,
                                       latency_s=time.perf_counter()-started)
            except Exception:
                session.failed = True  # never retry an uncertain primitive execution
                raise
            session.last_step, session.last_reply = step, reply
            return reply
        finally:
            session.last_used = time.monotonic()
            session.lock.release()

    def close(self):
        self.sessions.clear()


async def handle_connection(ws, service):
    await ws.send(pack(service.metadata))
    async for raw in ws:
        try:
            obs = unpack(raw)
            reply = await asyncio.to_thread(service.infer, obs)
        except Exception as exc:
            log.exception("policy request failed")
            service.record(event="request_failed", error=f"{type(exc).__name__}: {exc}")
            reply = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        await ws.send(pack(reply))


async def serve(service, host, port):
    from functools import partial
    from websockets.asyncio.server import serve as websocket_serve

    async with websocket_serve(partial(handle_connection, service=service), host, port,
                               compression=None, max_size=64*1024*1024,
                               ping_interval=None, close_timeout=5):
        log.info("Policy server ready on %s:%s", host, port)
        await asyncio.Future()

