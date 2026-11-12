"""Franka prompt and tool library on the LLMPlanner: an LLM plans, analytical tools move the arm."""
from __future__ import annotations

import os
import time

from .conversation import LLMPlanner
from .tracing import reasoning_summary


def said(response, provider):
    """The reasoning summary and any text the model wrote beside its tool call, for the decision log."""
    text = []
    if provider == "openai-responses":
        for item in response.get("output") or []:
            if item.get("type") == "message":
                text += [c.get("text", "") for c in item.get("content") or [] if c.get("type") == "output_text"]
    else:
        for choice in response.get("choices") or []:
            content = (choice.get("message") or {}).get("content")
            if isinstance(content, str):
                text.append(content)
    out = {"thought": reasoning_summary(response, provider), "text": "\n\n".join(t.strip() for t in text if t.strip())}
    return {k: v for k, v in out.items() if v}


class MeasuredPlanner(LLMPlanner):
    """Keep the existing wire; record each HTTP attempt, including retries.

    Preserve provider usage verbatim so billing can be calculated with the
    endpoint's actual rates afterwards. Missing usage is unknown, never zero.
    The callback is bound to the episode by Service before the first request.
    """

    record = staticmethod(lambda **row: None)

    def _make_backend(self):
        backend = super()._make_backend()
        if not hasattr(backend, "_post"):
            raise ValueError("Planner telemetry requires a Responses or chat-compatible backend")
        post = backend._post

        def measured(body):
            started = time.perf_counter()
            response, error = None, None
            try:
                response = post(body)
                return response
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                self.record(event="planner_request", latency_s=time.perf_counter()-started,
                            provider=self.provider, model=self.model,
                            response_id=response.get("id") if isinstance(response, dict) else None,
                            usage=response.get("usage") if isinstance(response, dict) else None,
                            error=error, **self.tracing.last_observation,
                            **(said(response, self.provider) if isinstance(response, dict) else {}))

        backend._post = measured
        return backend

PROMPT_VERSION = "harness-llm-only-v1"
# The planner's system prompt. The chunk length is substituted for the configured open_loop_horizon.
SYSTEM_PROMPT = """You control a single DROID Franka arm with a Robotiq gripper, by calling ONE tool per turn.
The task instruction and three RGB cameras are your evidence. There is no object-state oracle or depth.
The simulator pauses while you think. Control is 15 Hz; each 32-step chunk lasts about 2.13 robot seconds.

FRAME AND TOOLS
Positions are metres in the robot-root frame: +x forward, +y left, +z up. The controlled point is the
Robotiq gripper MOUNT, base_link, NOT the fingertips or contact point. Its orientation is base_link's
quaternion (w,x,y,z), NOT Euler angles. Gripper fingers extend below the mount when facing down; maintain
ample clearance (at least 0.20 m above a grounded surface when staging for manipulation). Gripper: 0 OPEN, 1 CLOSED.
move_to moves the mount while retaining its current orientation. move_pose also changes orientation.
They use bounded IK with joint limits, but do NOT check collisions. A completed command is not proof of
contact or grasp. Check measured state and fresh images. Avoid moving through objects or the table.

locate triangulates two image points. In the labeled LEFT and RIGHT exterior images, identify the SAME
visible physical feature on the requested object/target. Send [x,y] for each, normalized to [0,1000],
where [0,0] is top left and [1000,1000] is bottom right. Do NOT use [y,x] or raw pixel coordinates.
Image dimensions are provided. Prefer a shared corner or surface marking; the silhouette/bounding-box
centres in different views need not be the same 3D point. If occluded in either view, do not invent a point.
The result is a surface feature, not the object's centre or a ready-made grasp pose. Use the returned
geometric errors; good geometry still does not prove you picked the correct object. Re-ground after motion.

WORKFLOW
Use locate to ground the scene. Stage above the target with clearance, using small analytical moves.
Use move_to/move_pose and set_gripper/release for every manipulation. There is no action-model handoff. Inspect fresh images and measured state after each motion; a closed gripper is not proof of holding an object.
set_gripper changes only the gripper; release opens it. hold observes after a fixed number of chunks.
If a tool rejects a request, read the error and correct it. Do not repeatedly issue an unreachable target.
Only call done when the FULL task is visibly complete. The simulator, not your statement, judges success.
"""


def tool(name, description, properties, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}


def vector(n, description):
    return {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n,
            "description": description}


def tools(*, multi_locate=False, locator=False, move_path=False, no_done=False):
    duration = {"type": "number", "minimum": 1, "maximum": 16, "description": "Robot seconds; default 3."}
    chunks = {"type": "integer", "minimum": 1, "maximum": 4}
    basic = [
        tool("hold", "Hold measured joints and the last commanded gripper, then observe.", {"max_chunks": chunks}),
        tool("done", "Declare visually complete; hold position. This does not set simulator success.",
             {"summary": {"type": "string"}}, ("summary",)),
    ]
    specs = basic + [
        (LOCATE_DESCRIBED if locator else
         tool("locate", "Triangulate the same visible feature in both RGB exterior views; no robot motion.",
              {"object": {"type": "string"}, "left_xy": vector(2, "LEFT [x,y], 0..1000"),
               "right_xy": vector(2, "RIGHT [x,y], 0..1000")}, ("object", "left_xy", "right_xy"))),
        tool("move_to", "Move gripper MOUNT in robot-root metres, retaining orientation; no collision planning.",
             {"xyz": vector(3, "Gripper mount in robot-root metres"), "duration_s": duration}, ("xyz",)),
        tool("move_pose", "Move and rotate Robotiq base_link (MOUNT); bounded IK, no collision planning.",
             {"xyz": vector(3, "Gripper mount in robot-root metres"), "quaternion_wxyz": vector(4, "base_link orientation"),
              "duration_s": duration}, ("xyz", "quaternion_wxyz")),
        tool("set_gripper", "Command gripper without arm motion: 0 open, 1 closed.",
             {"closed": {"type": "integer", "enum": [0, 1]}}, ("closed",)),
        tool("release", "Open the gripper without arm motion.", {}),
    ] + ([LOCATE_MANY_DESCRIBED if locator else LOCATE_MANY] if multi_locate else []) + ([MOVE_PATH] if move_path else [])
    if no_done:  # the simulator ends an episode on success itself, so done can only end one early
        specs = [t for t in specs if t["name"] != "done"]
    return specs


# config move_path: consecutive moves (and a gripper change on arrival) as one continuous motion, sharing chunks.
MOVE_PATH = tool("move_path", "Move the gripper MOUNT through up to 6 waypoints in one continuous motion, without "
                 "stopping between them. Each waypoint may set an orientation (else it keeps the current one) and a "
                 "gripper command on arrival; the path then waits for the fingers to finish before going on. Each "
                 "segment has move_to/move_pose's IK and limits; no collision planning.",
                 {"waypoints": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
                     "type": "object", "additionalProperties": False, "required": ["xyz"],
                     "properties": {"xyz": vector(3, "Gripper mount in robot-root metres"),
                                    "quaternion_wxyz": vector(4, "base_link orientation at this waypoint"),
                                    "duration_s": {"type": "number", "minimum": 1, "maximum": 8,
                                                   "description": "Robot seconds for this segment; default 1.5."},
                                    "gripper": {"type": "integer", "enum": [0, 1],
                                                "description": "On arrival: 0 open, 1 closed."}}}}}, ("waypoints",))
# config no_done: the simulator ends the episode the moment the task is complete, so there is nothing to declare.
NO_DONE_RULE = ("There is no done tool. The simulator ends the episode the moment the FULL task is complete, so while\n"
                "you are still being asked, it is not complete: compare the scene with the task, find what is still\n"
                "wrong (an object off target, still held, or tipped) and fix it.\n")

# config pace_note: how motion uses robot time, which the episode limit counts.
TIMING = """TIMING
Motion runs in chunks of 32 control steps, 2.13 robot seconds each, and every move rounds up to whole chunks:
duration_s 3 costs 4.3 s, while duration_s 2 or 1 costs 2.1 s. The simulator pauses while you think, so thinking
and locating cost no robot time; motion does, and each episode has a fixed robot-time limit. With an empty gripper
use duration_s 1-2; carry a held object at about 2 s per 20 cm so it does not swing loose.
"""


def timing_note(horizon, speed):
    """TIMING for the configured execution (open_loop_horizon steps per chunk, moves run move_speed_multiplier
    times faster than requested). At 32 steps and 1x this is TIMING as the white-mug runs used it."""
    if (horizon, speed) == (32, 1):
        return TIMING
    import math

    def cost(duration_s):
        return math.ceil(math.ceil(duration_s * 15 / speed) / horizon) * horizon / 15
    return (f"TIMING\nMotion runs in chunks of {horizon} control steps, {horizon / 15:.2f} robot seconds each; a move "
            f"runs {speed:g}x faster than its duration_s and rounds up to whole\nchunks: duration_s 3 costs "
            f"{cost(3):.2f} s, duration_s 1 costs {cost(1):.2f} s. The simulator pauses while you think, so thinking\n"
            "and locating cost no robot time; motion does, and each episode has a fixed robot-time limit. With an "
            "empty gripper\nuse short moves; carry a held object more gently so it does not swing loose.\n")


CHAIN = ("Chain consecutive moves (approach, descend, close, lift, carry) in one move_path, so they share chunks "
         "instead of each rounding up on its own.\n")

# Carry guidance is independent of orientation and candidate notes.
CARRYING = """CARRYING
A held item reaches below the fingertips by the part of it under the grip, and further once it hangs or tips in the
fingers, so clear walls, rims and other objects with the item's lowest point, not the fingertips or the mount. Plan
the carry height from where the fingertips closed on the item and how far it extends below that, plus a margin; lift
straight up to that height before moving sideways, and come down only over the goal. A hanging item swings with the
arm: move it slowly and without sharp reversals.
"""

# config placement_note: where a target region is, and placing the object rather than the gripper on it.
PLACING = """PLACING
A target region (the middle of a surface, a spot beside something, the inside of a container) is rarely one visible
feature: work it out from features you can see in both views. The robot often hides part of a surface, so measure
the edges and corners you can see, say which ones you cannot, and do not guess them; a rectangular surface's
opposite edges are parallel and equal in length, so its visible corners and side edges fix its size and centre.
Place the OBJECT, not the gripper. A grip off the object's centre (an edge, a rim, a handle, one end) holds the
object off the gripper by that amount, so with the object held just above where it will go, locate points on the
object itself and move until the object, not the mount or the fingertips, is over the target; then lower it and
release. After releasing, locate the object again; if it is off target, pick it up and correct it.
"""

# config locator: a second model picks the image points, so locate and locate_many carry descriptions only.
DESCRIBE = ("One small visible physical feature, precisely: the part and side, e.g. 'mug rim on the side away from "
            "the handle'. A vision model finds it in both exterior views.")
LOCATE_DESCRIBED = tool("locate", "Find one described feature in both exterior views and triangulate it; no robot "
                        "motion.", {"object": {"type": "string", "description": DESCRIBE}}, ("object",))
LOCATE_MANY_DESCRIBED = tool("locate_many", "Find several described features in both exterior views and triangulate "
                             "each; no robot motion. Each point is reported on its own.",
                             {"points": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
                                 "type": "object", "additionalProperties": False, "required": ["object"],
                                 "properties": {"object": {"type": "string", "description": DESCRIBE}}}}}, ("points",))
LOCATE_PICKS = """locate triangulates two image points. In the labeled LEFT and RIGHT exterior images, identify the SAME
visible physical feature on the requested object/target. Send [x,y] for each, normalized to [0,1000],
where [0,0] is top left and [1000,1000] is bottom right. Do NOT use [y,x] or raw pixel coordinates.
Image dimensions are provided. Prefer a shared corner or surface marking; the silhouette/bounding-box
centres in different views need not be the same 3D point. If occluded in either view, do not invent a point.
"""
LOCATE_DESCRIBES = """locate finds a feature you describe: a separate vision model picks it in the LEFT and RIGHT exterior images and
the harness triangulates it. Describe ONE small, distinct physical feature per point (a corner, a rim point on a
named side, a marking), not a whole object, and say which object it is on. If it is hidden in either view the
point comes back with an error; describe another feature.
"""

# multi_locate's batch form of locate: several labelled points in one call, each triangulated on its own.
LOCATE_MANY = tool("locate_many", "Triangulate several features at once, each as with locate; no robot motion. "
                   "Each point is reported on its own, so one bad pair does not spoil the rest.",
                   {"points": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
                       "type": "object", "additionalProperties": False, "required": ["object", "left_xy", "right_xy"],
                       "properties": {"object": {"type": "string", "description": "Your label for this point"},
                                      "left_xy": vector(2, "LEFT [x,y], 0..1000"),
                                      "right_xy": vector(2, "RIGHT [x,y], 0..1000")}}}}, ("points",))


# Grasp-only protocol (not used by the MolmoSpaces run): one named object, one attempt, LLM-only tools.
PROMPT_VERSION_GRASP = "harness-grasp-v1"
GRASP_TRIAL = """GRASP TRIAL
The opening message names one object and describes the restored scene. Pick that object up, lift it clear of
the table and hold it, then call done. Do not carry it anywhere or set it down. You get ONE attempt: once the
gripper closes, the trial ends as soon as it is clear whether the object is held, so position the fingers
before closing. You have at most {max_calls} tool calls. The simulator, not your statement, judges success.
"""
FEW_CALLS = "Use as few tool calls as you think possible.\n"  # config few_calls
DONE_RULE = "Only call done when the FULL task is visibly complete. The simulator, not your statement, judges success.\n"
TASKS = (None, "grasp")

# The same trial with grasp aids (config grasp_aids): the gripper's geometry as facts, what held in earlier
# runs, and the fingertip readout and close-ups when those aids are on. The text omits the lines of an aid that
# is off; planner_prompt_sha256 and the server's grasp_aids say which text ran.
PROMPT_VERSION_GRASP_AIDS = "harness-grasp-aids-v1"
# The grasp trial with the general knowledge packet (config knowledge) in place of the aids' advice text.
PROMPT_VERSION_KNOWLEDGE = "harness-grasp-knowledge-v1"
VAGUE_FINGERS = ("Gripper fingers extend below the mount when facing down; maintain\nample clearance (at least "
                 "0.20 m above a grounded surface when staging for manipulation). ")
GRIPPER = """GRIPPER
The gripper is a Robotiq 2F-85 parallel gripper on the mount (base_link). Its two fingers point along base_link's
x-axis and end about 0.15 m from the mount: pointing straight down, the fingertip ends are about 0.15 m below
gripper_base_xyz. Each finger pad is about 2 cm wide and runs about 4 cm up from the fingertip end. The fingers
close along base_link's y-axis. Open, the pads are 85 mm apart, so the part you grip must be narrower than that
where the pads meet it. Closing, the fingers swing inward on an arc, so the pads' height shifts by about 1 cm.
In the robot's home pose base_link's x-axis points down, its y-axis along robot +y and its z-axis along robot +x.
While the gripper points straight down, turning base_link about the vertical turns the closing direction by the
same angle.
"""
FINGERTIPS = """Every observation and result reports fingertips, computed from the joints like gripper_base_xyz:
midpoint_xyz (between the fingertip ends, robot-root metres), closing_direction_deg (the direction the fingers
close along, on the table: 0 = robot +x, 90 = robot +y; 0 and 180 are the same), tilt_deg (lean from straight
down) and opening_m (about how far apart the pads are now). Compare midpoint_xyz with points you located.
"""
CLOSEUP = """After a move_to or move_pose that ends with the fingertip midpoint within 0.15 m of the last point you
located, the next observation also has enlarged full-resolution close-ups: a crop of each exterior camera centred
where the fingertip midpoint projects, and the part of the wrist image holding the fingertips. Each says which
region of its full image it shows, in the same 0..1000 coordinates.
"""
WHAT_HELD = """WHAT HELD IN EARLIER RUNS
In earlier runs of this robot on these tasks, a grasp held 72% of the time with the fingertip midpoint within
1 cm of the object's centre, against 26% when it was more than 4 cm off. Fingertip ends 1-2 cm above the table
held 85% of the time, more than 8 cm up 26%; fingertips level with an object's top often closed over it. Fingers
closing across an elongated object's narrow side held far more often than along its length. Grasps within 5
degrees of vertical held about half the time; few tilted more than 25 degrees did. A neighbour within a finger's
width of the object can block a finger. A handle, neck or rim can be a narrower part to grip than the whole
object. The grasp has to survive a lift and a carry, and should suit where the task needs the object next.
"""
GRASP_PLAN = """In the same reply as your first motion, write a short grasp plan: where the object is in both
exterior views, its narrow side and about how wide it is, the closing direction across it, the fingertip height
you will close at, and any neighbour within reach of the fingers.
"""
# multi_locate: locates (observation only) may be batched in a turn beside at most one other tool.
ONE_TOOL = "by calling ONE tool per turn.\n"
MULTI_LOCATE = ("by calling tools.\n"
                "In one turn you may call locate or locate_many as often as you need, plus at most ONE other tool,\n"
                "which ends the turn. Each turn counts once against your call budget, however many locates it has.\n"
                "When you need more than one point before moving, use ONE locate_many call with all of them; each\n"
                "point keeps your label.\n")

# The grasp aids on a whole task (no task set): the facts and what held, a plan before each grasp.
PROMPT_VERSION_TASK_AIDS = "harness-llm-only-aids-v1"
TASK_GRASP_PLAN = GRASP_PLAN.replace("In the same reply as your first motion", "Before each grasp, in the same reply as its first motion")



def make_planner(config, transport=None):
    from .config import resolve_config
    config = resolve_config(config)
    knowledge = config.get("knowledge", True)
    if knowledge is True:
        knowledge = {}
    elif knowledge is False or knowledge is None:
        knowledge = None
    elif not isinstance(knowledge, dict):
        raise ValueError("knowledge must be true, false, null, or an options mapping")
    config["knowledge"] = knowledge
    cfg = dict(config["planner"])
    # Existing compat backends read effort at construction. One server hosts one
    # planner configuration; reject a mismatch rather than stamp fictitious effort.
    actual = os.environ.get("SILOS0_LLM_REASONING_EFFORT", "").strip()
    if actual != cfg["effort"]:
        raise ValueError("SILOS0_LLM_REASONING_EFFORT must equal planner.effort")
    if config.get("mode", "llm-only") != "llm-only":
        raise ValueError("mode must be llm-only")
    horizon = config.get("open_loop_horizon", 8)   # policy.execution_settings' default, which validates it
    prompt = SYSTEM_PROMPT.replace("each 32-step chunk lasts about 2.13 robot seconds",
                                   f"each {horizon}-step chunk lasts about {horizon / 15:.2f} robot seconds")
    task = config.get("task")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    aids = config.get("grasp_aids") or {}
    if aids:
        if VAGUE_FINGERS not in prompt or "WORKFLOW\n" not in prompt or DONE_RULE not in prompt:
            raise ValueError("llm-only prompt changed; update the grasp-aids prompt")
        prompt = prompt.replace(VAGUE_FINGERS, "Stage with clearance above objects and the table. ")
        facts = ("" if config.get("knowledge") is not None else GRIPPER) + (FINGERTIPS if aids.get("fingertips") else "") + (CLOSEUP if aids.get("closeup") else "")
        prompt = prompt.replace("WORKFLOW\n", facts + "\nWORKFLOW\n")
    if task == "grasp":
        if DONE_RULE not in prompt:
            raise ValueError("llm-only prompt changed; update the grasp prompt")
        trial = GRASP_TRIAL.format(max_calls=cfg["max_calls"]) + (FEW_CALLS if config.get("few_calls") else "")
        if aids and config.get("knowledge") is None:
            trial += "\n" + WHAT_HELD + GRASP_PLAN
        prompt = prompt.replace(DONE_RULE, trial)
    if config.get("few_calls") and task != "grasp":
        raise ValueError("few_calls is defined for task grasp")
    if task != "grasp" and aids and config.get("knowledge") is None:
        # A whole task: the same facts, without the one-attempt trial rules; a plan before every grasp.
        prompt = prompt.replace(DONE_RULE, WHAT_HELD + TASK_GRASP_PLAN + DONE_RULE)
    if config.get("locator") is not None:
        if LOCATE_PICKS not in prompt:
            raise ValueError("prompt changed; update the locator text")
        prompt = prompt.replace(LOCATE_PICKS, LOCATE_DESCRIBES)
    if config.get("no_done"):
        if task == "grasp" or DONE_RULE not in prompt:
            raise ValueError("no_done is defined for whole tasks (the grasp trial ends on done)")
        prompt = prompt.replace(DONE_RULE, NO_DONE_RULE)
    if any(config.get(k) for k in ("pace_note", "move_path", "placement_note", "carry_note")):
        if "WORKFLOW\n" not in prompt:
            raise ValueError("prompt changed; update the pace_note, move_path, placement_note and carry_note text")
        execution = (config.get("open_loop_horizon", 8), config.get("move_speed_multiplier", 4.))
        note = (timing_note(*execution) if config.get("pace_note") else "") + (CHAIN if config.get("move_path") else "")
        note += PLACING if config.get("placement_note") else ""
        note += CARRYING if config.get("carry_note") else ""
        prompt = prompt.replace("WORKFLOW\n", note + "\nWORKFLOW\n", 1)
    if config.get("multi_locate"):
        if ONE_TOOL not in prompt:
            raise ValueError("prompt changed; update the multi_locate text")
        prompt = prompt.replace(ONE_TOOL, MULTI_LOCATE)
    tool_specs = tools(multi_locate=bool(config.get("multi_locate")),
                       locator=config.get("locator") is not None, move_path=bool(config.get("move_path")),
                       no_done=bool(config.get("no_done")))
    packet = knowledge_packet(config, tool_specs) if config.get("knowledge") is not None else None
    if packet is not None:
        prompt = prompt.rstrip("\n") + "\n\n" + packet.text
    multi = bool(config.get("multi_locate"))
    version = (("harness-grasp-aids-knowledge-v1" if aids else PROMPT_VERSION_KNOWLEDGE)
               if packet is not None and task == "grasp" else
               PROMPT_VERSION_GRASP_AIDS if aids and task == "grasp" else
               PROMPT_VERSION_TASK_AIDS if aids else
               PROMPT_VERSION_GRASP if task == "grasp" else PROMPT_VERSION)
    located = config.get("locator") is not None
    version += (("+multi-locate" if multi else "")
                + ("+locator" if located else "") + ("+pace" if config.get("pace_note") else "")
                + ("+move-path" if config.get("move_path") else "") + ("+no-done" if config.get("no_done") else "")
                + ("+placement" if config.get("placement_note") else "")
                + ("+carry" if config.get("carry_note") else ""))
    if packet is not None and task != "grasp":
        version += "+knowledge"
    planner = MeasuredPlanner(cfg, transport=transport, system_prompt=prompt,
                              tool_specs=tool_specs,
                              prompt_version=version)
    planner.knowledge = packet.provenance() if packet is not None else None
    return planner


def knowledge_packet(config, tool_specs=None):
    """The general knowledge packet (silos0.harness_knowledge) for this config's own tools.

    config `knowledge:` takes optional `layers` (default: the package's defaults), `features` (harness features
    the packet may rely on, e.g. budget_meter) and `candidates` (true also renders candidate notes, for testing
    them). The robot is this harness's DROID Franka; max_calls comes from the planner config."""
    from silos0.harness_knowledge import packet as kp
    spec = config["knowledge"] or {}
    if not isinstance(spec, dict) or set(spec) - {"layers", "features", "candidates"}:
        raise ValueError("knowledge takes optional layers, features and candidates")
    if tool_specs is None:
        tool_specs = tools(multi_locate=bool(config.get("multi_locate")),
                           locator=config.get("locator") is not None, move_path=bool(config.get("move_path")),
                           no_done=bool(config.get("no_done")))
    features = set(spec.get("features", ()))
    # Runtime-owned capabilities must reflect the actual schema, not a knowledge override.
    features.discard("locator")
    if config.get("locator") is not None:
        features.add("locator")
    built = kp.build([t["name"] for t in tool_specs], "harness", "llm-only", "droid-franka-2f85",
                     config["planner"]["model"], features,
                     {"max_calls": config["planner"]["max_calls"]}, tuple(spec.get("layers", kp.DEFAULT_LAYERS)),
                     ("active", "candidate") if spec.get("candidates") else ("active",))
    hits = kp.leaks(built.text)
    if hits:
        raise ValueError(f"knowledge packet names benchmark content: {hits}")
    return built
