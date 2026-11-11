"""Resolve a whole-task config to the approved Sol-200 configuration.

Explicit values win. Grasp-only protocols do not inherit whole-task tool/prompt
switches. Resolution never mutates the caller's mapping.
"""
from copy import deepcopy

DEFAULT_PLANNER = {
    "provider": "openai-responses",
    "base_url": "",
    "model": "gpt-6.1-sol",
    "effort": "high",
    "max_calls": 200,
}
WHOLE_TASK_DEFAULTS = {
    "multi_locate": True,
    "move_path": True,
    "pace_note": True,
    "no_done": True,
    "placement_note": True,
    "grasp_aids": {"fingertips": True, "closeup": True},
}


def resolve_config(config=None):
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    result = deepcopy(config or {})
    result.setdefault("mode", "llm-only")
    if result["mode"] == "llm-only" and result.get("task") is None:
        for key, value in WHOLE_TASK_DEFAULTS.items():
            result.setdefault(key, deepcopy(value))
    planner = result.get("planner", {})
    if not isinstance(planner, dict):
        raise ValueError("planner must be a mapping")
    defaults = dict(DEFAULT_PLANNER)
    # A different provider must never inherit Sol's model or endpoint.
    if planner.get("provider", defaults["provider"]) != defaults["provider"]:
        defaults.pop("base_url")
        defaults.pop("model")
        if not planner.get("model"):
            raise ValueError("planner.model is required for a non-default provider")
    result["planner"] = {**defaults, **planner}
    return result
