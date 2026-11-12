"""One episode's planner and primitive state, with completion evidence from the next observation."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import uuid

import numpy as np
from PIL import Image

from .geometry import CAMERA_KEYS, calibration, finite, locate, rotation
from .planner import tools
from .kinematics import rotvec

HORIZON = 32  # the reference chunk the tool budgets are written in; the executed chunk is open_loop_horizon
FPS = 15
MOVE_POSITION_TOLERANCE_M = .005
MOVE_ROTATION_TOLERANCE_RAD = .05
MOVE_STALL_CHUNKS = 2
MOVE_NEAR_STALL_CHUNKS = 1
MOVE_NEAR_POSITION_M = .01
MOVE_NEAR_ROTATION_RAD = math.radians(5)
MOVE_MIN_PROGRESS = .2  # tolerance units: 1 mm or 0.01 rad beyond arrival bounds
# Grasp aids (config grasp_aids). Robot constants only: the fingertips are a fixed
# offset from the mount the kinematics already give, and the close-ups are crops of the cameras the planner sees.
READ_TOOLS = ("locate", "locate_many")
GRIPPER_SETTLE_STEPS = 20  # a close takes about 19 control steps in free air  # tools that observe only: no motion, no robot time
GRASP_AIDS = ("fingertips", "closeup")
# Mount to the fingertip ends along base_link x; Sol's lowest close on RoboLab left the
# mount 0.151 m over the table. Another simulator's gripper model may differ (MolmoSpaces' MuJoCo 2F-85: 0.141).
FINGERTIP_OFFSET_M = float(os.environ.get("SILOS0_FINGERTIP_OFFSET_M", .15))
MAX_OPENING_M = .085       # Robotiq 2F-85 stroke
CLOSEUP_RADIUS_M = .15     # a move ending with the fingertips this near the last located point brings close-ups
CLOSEUP_PX = 320           # exterior crop side in native pixels, centred where the fingertip midpoint projects
WRIST_CLOSEUP = (.30, .30, .95, 1.)  # normalized box of the wrist image that holds the fingertips and the gap
                                     # between them (the camera moves with the gripper)


def motion_budget(obs):
    """Use the simulator's motion clock; older clients may omit the budget."""
    keys = ("silos0/motion_time_limit_s", "silos0/control_hz")
    if not any(key in obs for key in keys):
        return None
    limit, hz, step = (float(obs[key]) for key in (*keys, "silos0/step"))
    if (not all(math.isfinite(v) for v in (limit, hz, step))
            or limit <= 0 or hz <= 0 or step < 0 or not step.is_integer()):
        raise ValueError("Invalid simulator motion budget")
    remaining = max(0., limit - step / hz)
    if remaining >= 30:
        return None
    text = (f"URGENT: Only {remaining:.1f} seconds of motion remain. "
            "Minimize movements made only to explore or inspect; prioritize actions that advance the task. "
            "Retry when needed. Think carefully—thinking does not consume motion time. "
            "Urgency does not mean moving faster.")
    return text, dict(limit_s=limit, elapsed_s=step/hz, remaining_s=remaining,
                      urgent=remaining < 30, control_step=int(step))


def image_block(rgb, scale=1):
    image = Image.fromarray(rgb)
    if scale != 1:
        image = image.resize((image.width*scale, image.height*scale), Image.LANCZOS)
    # Equal scaling preserves normalized pointing coordinates. Keep enough detail for correspondence.
    if image.width > 1280:
        image.thumbnail((1280, 1280))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    return ("image", base64.b64encode(out.getvalue()).decode("ascii"))


def state(obs):
    q = finite(obs["observation/joint_position"], (7,), "joint_position")
    grip = finite(obs["observation/gripper_position"], (1,), "gripper_position")[0]
    if not 0 <= grip <= 1:
        raise ValueError("gripper_position outside [0,1]")
    return q, float(grip)


def validate_obs(obs):
    state(obs)
    if not isinstance(obs.get("prompt"), str) or not obs["prompt"].strip():
        raise ValueError("prompt is required")
    for key in ["observation/image", *CAMERA_KEYS.values()]:
        rgb = np.asarray(obs[key])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError(f"{key} must be a uint8 RGB image")
    for side in ("left", "right"):
        calibration(obs, side)


def execution_settings(config):
    horizon = config.get("open_loop_horizon", 8)
    speed = config.get("move_speed_multiplier", 4.)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (1, 2, 4, 8, 16, 32):
        raise ValueError("open_loop_horizon must be one of 1,2,4,8,16,32")
    if isinstance(speed, bool) or not isinstance(speed, (float, int)) or not np.isfinite(speed) or speed < 1:
        raise ValueError("move_speed_multiplier must be finite and >= 1")
    if "gripper_settle_steps" in config:
        raise ValueError("Configure gripper timing with move_speed_multiplier, not gripper_settle_steps")
    settle = max(1, math.ceil((HORIZON / speed) / horizon)) * horizon
    return horizon, float(speed), settle


class Episode:
    def __init__(self, planner, robot, record=lambda **row: None, options=None, opening=None, aids=None, locator=None):
        self.planner, self.robot = planner, robot
        # The request key the planner's opening message comes from; None means the task prompt itself.
        self.opening = opening
        self.aids = {k: bool((aids or {}).get(k)) for k in GRASP_AIDS}
        self.last_located = None
        self.closeup_due = False
        self.record = record
        self.options = {"multi_locate": False, "move_path": False, "no_done": False, **(options or {})}
        self.horizon, self.move_speed_multiplier, self.gripper_settle_steps = execution_settings(self.options)
        self.locator = locator
        self.allowed = {t["name"] for t in tools(multi_locate=self.options["multi_locate"], locator=locator is not None,
                                                 move_path=self.options["move_path"], no_done=self.options["no_done"])}
        self.started = False
        self.stopped = None
        self.active = None
        self.pending = []
        self.gripper_command = None
        # Several servers may log to one sink and receive identical client IDs; this id is unique to the episode.
        self.episode_id = "episode-" + uuid.uuid4().hex

    def evidence(self, obs):
        q, gripper = state(obs)
        xyz, R = self.robot.fk(q)
        out = {"joints": q.tolist(), "gripper_measured_closed_fraction": gripper,
               "gripper_base_xyz": xyz.tolist(), "base_link_rotation_matrix": R.tolist()}
        if self.aids["fingertips"]:
            out["fingertips"] = self.fingertips(q, gripper)
        return {**out, "grasp_verified": False}

    def fingertips(self, q, grip):
        """Where the fingers are, from the joints: the mount pose plus the fixed finger offset."""
        xyz, R = self.robot.fk(q)
        closing = R[:2, 1]  # the fingers close along base_link y
        return {"midpoint_xyz": np.round(xyz+FINGERTIP_OFFSET_M*R[:, 0], 4).tolist(),
                "closing_direction_deg": round(float(np.degrees(np.arctan2(closing[1], closing[0]))) % 180, 1),
                "tilt_deg": round(float(np.degrees(np.arccos(np.clip(-R[2, 0], -1, 1)))), 1),
                "opening_m": round(MAX_OPENING_M*(1-grip), 4)}

    def closeups(self, obs):
        """Enlarged full-resolution crops around the fingertips: each exterior camera, then the wrist camera."""
        q, grip = state(obs)
        tip = np.asarray(self.fingertips(q, grip)["midpoint_xyz"])
        blocks = []
        for side in ("left", "right"):
            rgb = np.asarray(obs[CAMERA_KEYS[side]])
            h, w = rgb.shape[:2]
            K, T = calibration(obs, side)
            local = T[:3, :3].T @ (tip-T[:3, 3])
            u, v = (K @ local)[:2]/local[2] if local[2] > 0 else (-1., -1.)
            if not (0 <= u < w and 0 <= v < h):
                blocks.append(("text", f"{side.upper()} close-up omitted: the fingertips are outside this view."))
                continue
            x0 = int(np.clip(round(u)-CLOSEUP_PX//2, 0, max(w-CLOSEUP_PX, 0)))
            y0 = int(np.clip(round(v)-CLOSEUP_PX//2, 0, max(h-CLOSEUP_PX, 0)))
            crop = rgb[y0:y0+CLOSEUP_PX, x0:x0+CLOSEUP_PX]
            n = lambda x, y: [round(x/(w-1)*1000), round(y/(h-1)*1000)]
            blocks += [("text", f"{side.upper()} close-up, enlarged 2x: region {n(x0, y0)} to "
                                f"{n(x0+crop.shape[1]-1, y0+crop.shape[0]-1)} of the {side.upper()} image "
                                f"(normalized 0..1000); the fingertip midpoint projects to {n(u, v)}."),
                       image_block(np.ascontiguousarray(crop), 2)]
        rgb = np.asarray(obs[CAMERA_KEYS["wrist"]])
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = (round(f*s) for f, s in zip(WRIST_CLOSEUP, (w, h, w, h)))
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        scale = 2 if max(crop.shape[:2]) < 640 else 1
        blocks += [("text", f"WRIST close-up{', enlarged 2x' if scale == 2 else ''}: region "
                            f"{[round(1000*f) for f in WRIST_CLOSEUP[:2]]} to {[round(1000*f) for f in WRIST_CLOSEUP[2:]]} "
                            "of the WRIST image, where the fingertips are."),
                   image_block(crop, scale)]
        return blocks

    def blocks(self, obs):
        blocks = []
        for side, key in CAMERA_KEYS.items():
            rgb = np.asarray(obs[key])
            h, w = rgb.shape[:2]
            note = (obs.get("silos0/camera_notes") or {}).get(side)
            blocks += [("text", f"{side.upper()} camera{' (' + note + ')' if note else ''}, native {w}x{h}; normalized [x,y] 0..1000:"),
                       image_block(rgb)]
        if self.closeup_due:
            self.closeup_due = False
            blocks += self.closeups(obs)
        blocks.append(("text", json.dumps({**self.evidence(obs), "control_step": obs["silos0/step"]})))
        budget = motion_budget(obs) if self.options.get("motion_urgency", False) else None
        if budget is not None:
            text, timing = budget
            blocks.append(("text", text))
            self.record(event="motion_budget", **timing, text=text)
        return blocks

    def hold(self, obs):
        q, grip = state(obs)
        # A grasp can block closure. Keep the commanded value, not an intermediate
        # measured width that binarization would turn into an unintended release.
        if self.gripper_command is None:
            self.gripper_command = float(grip > .5)
        return np.tile(np.r_[q, self.gripper_command], (self.horizon, 1)).astype(np.float32)

    def complete(self, obs):
        if self.active is None:
            return
        a = self.active
        if "deadline_steps" in a and a["executed"]:
            q, _ = state(obs)
            position, orientation = self.robot.fk(q)
            position_error = float(np.linalg.norm(position-a["target_xyz"]))
            rotation_error = float(np.linalg.norm(rotvec(a["target_rotation"] @ orientation.T)))
            reached = (position_error <= MOVE_POSITION_TOLERANCE_M
                       and rotation_error <= MOVE_ROTATION_TOLERANCE_RAD)
            elapsed = a["executed"] * self.horizon
            # Ignore errors already within tolerance. Require net improvement
            # against the best meaningful residual, so jitter cannot keep a
            # stationary arm alive indefinitely. Start only after the trajectory.
            residual = (max(position_error / MOVE_POSITION_TOLERANCE_M - 1, 0)
                        + max(rotation_error / MOVE_ROTATION_TOLERANCE_RAD - 1, 0))
            near_target = (position_error <= MOVE_NEAR_POSITION_M
                           and rotation_error <= MOVE_NEAR_ROTATION_RAD)
            stall_limit = MOVE_NEAR_STALL_CHUNKS if near_target else MOVE_STALL_CHUNKS
            stalled = False
            if elapsed >= a["trajectory_steps"]:
                if "progress_reference" not in a:
                    a.update(progress_reference=residual, stalled_chunks=0)
                elif a["progress_reference"] - residual >= MOVE_MIN_PROGRESS:
                    a.update(progress_reference=residual, stalled_chunks=0)
                else:
                    a["stalled_chunks"] += 1
                stalled = a["stalled_chunks"] >= stall_limit
            a["arrival"] = dict(target_reached=reached, position_error_m=position_error,
                                orientation_error_rad=rotation_error, elapsed_sim_s=elapsed/FPS,
                                requested_duration_s=a["requested_duration_s"],
                                deadline_sim_s=a["deadline_steps"]/FPS,
                                target_xyz=a["target_xyz"].tolist(), actual_xyz=position.tolist(),
                                stalled_chunks=a.get("stalled_chunks", 0),
                                near_target=near_target, stall_chunk_limit=stall_limit)
            self.record(event="move_progress", call_id=a["id"], step=obs["silos0/step"],
                        residual_tolerance_units=residual,
                        trajectory_finished=elapsed >= a["trajectory_steps"], **a["arrival"])
            if reached or elapsed >= a["deadline_steps"] or stalled:
                a["remaining"] = 0
                a["move_stop_reason"] = ("target_reached" if reached else
                                         "duration_exhausted" if elapsed >= a["deadline_steps"] else "stalled")
            elif not a["remaining"]:
                # Keep the FINAL target, never the current measured position, while catching up.
                a["rows"] = np.tile(a["rows"][-1], (self.horizon, 1))
                a["offset"] = 0
                a["remaining"] = 1
        if a["remaining"]:
            return
        active, self.active = self.active, None
        result = {"ok": True, "action": active["name"], "chunks_executed": active["executed"],
                  "evidence": self.evidence(obs)}
        if "arrival" in active:
            result.update(active["arrival"], ok=active["arrival"]["target_reached"],
                          stop_reason=active["move_stop_reason"])
            if not result["ok"]:
                result["error"] = ("Target not reached: progress stalled after the trajectory finished. "
                                   "Inspect the remaining pose error and choose a revised move."
                                   if active["move_stop_reason"] == "stalled" else
                                   "Target not reached within requested duration; allow more time or choose an easier move.")
        if self.aids["closeup"] and active["name"] in ("move_to", "move_pose", "move_path") and self.last_located is not None:
            gap = float(np.linalg.norm(np.asarray(self.fingertips(*state(obs))["midpoint_xyz"])-self.last_located))
            if gap <= CLOSEUP_RADIUS_M:
                self.closeup_due = True
                result["closeup"] = {"included": True, "fingertips_to_last_located_m": round(gap, 3)}
        self.pending.append((active["id"], json.dumps(result)))
        self.record(event="primitive_completed", call_id=active["id"], result=result)

    @staticmethod
    def count(args):
        n = args.get("max_chunks", 1)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 4:
            raise ValueError("max_chunks must be an integer in [1,4]")
        return n

    def execute(self, cid, name, args, obs):
        if name not in self.allowed:
            raise ValueError(f"tool {name!r} is not available")
        q, grip = state(obs)
        if name == "done":
            self.stopped = "done"
            return {"ok": True, "summary": str(args.get("summary", "")), "simulator_success": "not asserted"}
        if name == "locate":
            if self.locator is not None:
                picked = self.locator.pick(obs, [str(args["object"])])[0]
                if isinstance(picked, str):
                    raise ValueError(picked)
                args = {**args, "left_xy": picked[0], "right_xy": picked[1]}
            result = {**locate(obs, args["left_xy"], args["right_xy"]), "object": str(args["object"])}
            if self.locator is not None:
                result.update(left_xy=args["left_xy"], right_xy=args["right_xy"])
            self.last_located = np.asarray(result["xyz"])
            return result
        if name == "locate_many":
            points = args.get("points")
            if not isinstance(points, list) or not 1 <= len(points) <= 12:
                raise ValueError("locate_many takes 1 to 12 points")
            if self.locator is not None:  # one locator request picks every point of the call
                picks = self.locator.pick(obs, [str(p.get("object") if isinstance(p, dict) else p) for p in points])
                points = [{"object": str(p.get("object") if isinstance(p, dict) else p), "error": pick}
                          if isinstance(pick, str) else
                          {"object": str(p.get("object") if isinstance(p, dict) else p), "left_xy": pick[0], "right_xy": pick[1]}
                          for p, pick in zip(points, picks)]
            out = []
            for point in points:
                try:
                    if "error" in point:
                        raise ValueError(point["error"])
                    found = {**locate(obs, point["left_xy"], point["right_xy"]), "object": str(point["object"])}
                    if self.locator is not None:
                        found.update(left_xy=point["left_xy"], right_xy=point["right_xy"])
                    self.last_located = np.asarray(found["xyz"])
                except (ValueError, TypeError, KeyError) as exc:
                    found = {"object": str(point.get("object") if isinstance(point, dict) else point), "error": str(exc)}
                out.append(found)
            return {"ok": any("xyz" in p for p in out), "points": out}
        rows, prompt, chunks, extra = None, None, 1, {}
        if name == "hold":
            chunks = self.count(args) * (HORIZON // self.horizon)
        elif name in ("set_gripper", "release"):
            closed = 0 if name == "release" else args["closed"]
            if isinstance(closed, bool) or closed not in (0, 1):
                raise ValueError("closed must be 0 (open) or 1 (closed)")
            self.gripper_command = float(closed)
            chunks = self.gripper_settle_steps // self.horizon
            rows = np.tile(self.hold(obs), (chunks, 1))
            self.record(event="gripper_schedule", call_id=cid, closed=closed,
                        settle_steps=self.gripper_settle_steps, control_hz=FPS,
                        move_speed_multiplier=self.move_speed_multiplier,
                        open_loop_horizon=self.horizon)
        elif name == "move_path":
            waypoints = args.get("waypoints")
            if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 6:
                raise ValueError("move_path takes 1 to 6 waypoints")
            if sum(float(w.get("duration_s", 1.5)) for w in waypoints) > 16:
                raise ValueError("move_path is limited to 16 robot seconds in all")
            seed, grip, rows = q.copy(), float(self.hold(obs)[0, 7]), []
            for i, w in enumerate(waypoints):
                segment = self.robot.path(seed, w["xyz"], w.get("quaternion_wxyz"), float(w.get("duration_s", 1.5)), 1,
                                          duration_scale=self.move_speed_multiplier, enforce_speed_limit=False)
                rows += [np.r_[joints, grip] for joints in segment]
                seed = segment[-1]
                if "gripper" in w:
                    if isinstance(w["gripper"], bool) or w["gripper"] not in (0, 1):
                        raise ValueError(f"waypoint {i+1}: gripper must be 0 (open) or 1 (closed)")
                    if float(w["gripper"]) != grip:  # let the fingers finish before the next segment
                        grip = float(w["gripper"])
                        # the fingers physically take ~19 steps to close, whatever the move speed
                        rows += [np.r_[seed, grip]] * max(GRIPPER_SETTLE_STEPS, self.gripper_settle_steps)
            while len(rows) % self.horizon:
                rows.append(rows[-1].copy())
            self.gripper_command = grip
            rows = np.asarray(rows, dtype=np.float32)
            chunks = len(rows)//self.horizon
        elif name in ("move_to", "move_pose"):
            quaternion = args["quaternion_wxyz"] if name == "move_pose" else None
            requested = args.get("duration_s", 3.)
            path = self.robot.path(q, args["xyz"], quaternion, requested, 1,
                                   max_move=.35,
                                   duration_scale=self.move_speed_multiplier, enforce_speed_limit=False)
            deadline_steps = math.ceil(math.ceil(requested*FPS)/self.horizon)*self.horizon
            extra = dict(deadline_steps=deadline_steps, requested_duration_s=requested,
                         target_xyz=np.asarray(args["xyz"], dtype=float),
                         target_rotation=self.robot.fk(q)[1] if quaternion is None else rotation(quaternion))
            motion_steps = len(path)
            padding = (-motion_steps) % self.horizon
            if padding:
                path = np.vstack((path, np.tile(path[-1], (padding, 1))))
            extra["trajectory_steps"] = len(path)
            self.record(event="move_schedule", call_id=cid, requested_duration_s=requested,
                        move_speed_multiplier=self.move_speed_multiplier,
                        scaled_duration_s=requested/self.move_speed_multiplier,
                        motion_steps=motion_steps, padding_steps=padding, control_hz=FPS,
                        open_loop_horizon=self.horizon, deadline_steps=deadline_steps,
                        joint_speed_limit_enforced=False,
                        position_tolerance_m=MOVE_POSITION_TOLERANCE_M,
                        orientation_tolerance_rad=MOVE_ROTATION_TOLERANCE_RAD)
            gripper = self.hold(obs)[0, 7]
            rows = np.column_stack((path, np.full(len(path), gripper))).astype(np.float32)
            chunks = len(rows)//self.horizon
        self.active = {"id": cid, "name": name, "remaining": chunks, "chunks": chunks, "executed": 0,
                       "rows": rows, "prompt": prompt, "offset": 0, "trace": [], "events": [], "held": False,
                       **extra}
        return None

    def infer(self, obs):
        self.complete(obs)  # a newly observed state, after the final chunk ran
        if self.active is None and self.stopped is None:
            blocks = self.blocks(obs)
            if not self.started:
                text = obs["prompt"] if self.opening is None else obs.get(self.opening)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{self.opening} is required for this task")
                self.planner.start(text, blocks)
                self.started = True
            for _ in range(4):
                calls, stop = self.planner.step(blocks, self.pending)
                self.pending = []
                self.record(event="planner", calls=calls, stop=stop, planner_calls=self.planner.calls)
                if stop not in (None, "retry"):
                    self.stopped = stop
                    break
                if not calls:
                    break
                writes = 0
                for index, (cid, name, args) in enumerate(calls):
                    # multi_locate: any number of reads (no motion, no robot time) plus one other tool per turn
                    reading = self.options["multi_locate"] and name in READ_TOOLS
                    if not reading:
                        writes += 1
                    if (index and not self.options["multi_locate"]) or (not reading and writes > 1):
                        result = {"ok": False, "error": ("not executed: one action per turn; locates may be batched"
                                                         if self.options["multi_locate"] else
                                                         "not executed: one tool per turn; reissue after observation")}
                    else:
                        try:
                            result = self.execute(cid, name, args, obs)
                        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                            result = {"ok": False, "error": str(exc)}
                    self.record(event="primitive", call_id=cid, name=name, args=args, result=result)
                    if result is not None:
                        self.pending.append((cid, json.dumps(result)))
                if self.active is not None or self.stopped is not None:
                    break
        action = self.hold(obs)
        instruction = "hold"
        if self.active is not None:
            a = self.active
            instruction = a["prompt"] or a["name"]
            if a["rows"] is not None:
                action = a["rows"][a["offset"]:a["offset"]+self.horizon].copy()
                a["offset"] += self.horizon
            a["remaining"] -= 1
            a["executed"] += 1
            self.gripper_command = float(action[-1, 7] > .5)
        self.record(event="action", step=obs["silos0/step"], instruction=instruction,
                    stopped=self.stopped, action=action.tolist())
        return {"action": action, "meta.instruction": instruction,
                "harness": {"stopped": self.stopped, "planner_calls": self.planner.calls}}
