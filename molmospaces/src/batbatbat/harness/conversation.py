"""One episode's conversation with the planner model: send the task and images, return its tool calls, send the
results back, enforce the call budget. planner.py supplies the prompt and tools; responses_api.py makes the calls."""
import json
import logging
import os
from .tracing import LLMRequestTrace
log = logging.getLogger(__name__)
PROMPT_VERSION = "v1"
SYSTEM_PROMPT = ""
def tools_spec():
    return []

class LLMPlanner:
    name = "llm"

    def __init__(self, planner_cfg: dict, memory=None, transport=None, *,
                 system_prompt=None, tool_specs=None, prompt_version=None):
        from . import responses_api as lp
        self.cfg = dict(planner_cfg)
        self.provider = self.cfg.get("provider", "anthropic")
        self.model = self.cfg.get("model") or ("claude-opus-5" if self.provider == "anthropic" else "")
        self.effort = self.cfg.get("effort", "low")
        self.include_thoughts = self.cfg.get("include_thoughts", True)
        if not isinstance(self.include_thoughts, bool):
            raise ValueError("planner.include_thoughts must be a boolean")
        self.max_calls = int(self.cfg.get("max_calls", 80))
        self.max_output_tokens = self.cfg.get("max_output_tokens", lp.DEFAULT_MAX_OUTPUT_TOKENS)
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int) \
                or self.max_output_tokens <= 0:
            raise ValueError("planner.max_output_tokens must be a positive integer")
        self.memory = memory
        self._injected = transport
        self._lp = lp
        self._system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
        self._tool_specs = tools_spec() if tool_specs is None else tool_specs
        self._prompt_version = PROMPT_VERSION if prompt_version is None else prompt_version
        self.calls = 0
        self._backend = None
        self._fails = 0
        self._retry = False
        self._nudges = 0
        self._opened = False        # the opening turn is pending: the first step() must not push another
        self._issued = []           # tool_use ids of the last reply; every one needs a result on the next turn
        self.tracing = LLMRequestTrace(name="harness.planner", provider=self.provider, model=self.model,
                                       metadata=self.provenance())

    def set_trace_context(self, *, session_id=None, **metadata):
        self.tracing.bind(session_id=session_id, **metadata)

    def system_prompt(self):
        rules = self.memory.rules_text() if self.memory is not None else ""
        return self._system_prompt + (f"\nRULES LEARNED ON EARLIER RUNS\n{rules}\n" if rules else "")

    def _make_backend(self):
        lp, system, tools = self._lp, self.system_prompt(), self._tool_specs
        if self.provider != "openai-responses":
            raise ValueError("This submission requires the Responses API")
        key = os.environ.get("BATBATBAT_LLM_API_KEY", "")
        if not key and self._injected is None:
            raise ValueError(f"BATBATBAT_LLM_API_KEY is required for provider {self.provider!r}")
        base = self.cfg.get("base_url") or lp.PROVIDER_BASE_URLS.get(self.provider) or ""
        if not base:
            raise ValueError(f"planner.base_url is required for provider {self.provider!r}")
        if self.provider == "openai-responses":
            return lp._ResponsesBackend(self.model, base, key, post=self._injected, system=system,
                                        tools=tools, observe=self.tracing.request,
                                        reasoning_summary="auto" if self.include_thoughts else None,
                                        max_output_tokens=self.max_output_tokens)

    def backend(self):
        if self._backend is None:
            self._backend = self._make_backend()
        return self._backend

    def ping(self):
        self.tracing.bind(phase="warmup")
        try:
            return self.backend().ping()
        finally:
            self.tracing.reset()  # warmup is not part of the first measured episode

    def provenance(self):
        return {"kind": "llm", "provider": self.provider, "model": self.model, "effort": self.effort,
                "max_calls": self.max_calls, "prompt_version": self._prompt_version,
                # Only a raised cap is stamped, so runs at the default keep their earlier provenance.
                **({"max_output_tokens": self.max_output_tokens}
                   if self.max_output_tokens != self._lp.DEFAULT_MAX_OUTPUT_TOKENS else {}),
                **({"include_thoughts": self.include_thoughts}
                   if self.provider in ("gemini", "openai-responses") else {})}

    def reset(self):
        self._backend = None
        self.calls = 0
        self._fails = self._nudges = 0
        self._retry = False
        self._opened = False
        self._issued = []
        self.tracing.reset()

    def start(self, task_text, blocks):
        self.tracing.bind(task=task_text, phase="episode")
        card = self.memory.card_for(task_text) if self.memory is not None else None
        opening = [("text", f"TASK: {task_text}\n")]
        if card:
            opening.append(("text", "A primitive sequence that solved this task on an earlier layout (spatial "
                                    "arguments are perception queries; re-ground them with locate):\n"
                                    + json.dumps(card.get("steps", []))[:4000] + "\n"))
        opening.append(("text", "Initial observation:"))
        self.backend().push_opening(opening + blocks)
        self._opened = True

    def step(self, blocks, results):
        """results: [(call_id, text)] for every call executed since the last turn."""
        b = self.backend()
        if self.calls >= self.max_calls:
            return [], "budget"
        if self._opened:
            self._opened = False            # the opening already carries this observation
        elif not self._retry:
            # Every id the model issued last turn must be answered in this one user message, executed or not;
            # a missing id makes the Anthropic wire refuse the whole transcript from here on.
            answered = {cid for cid, _ in results}
            results = list(results) + [(cid, json.dumps({"ok": False, "error": "not executed this turn"}))
                                       for cid in self._issued if cid not in answered]
            if results:
                b.push_results([cid for cid, _ in results],
                               [("text", "\n".join(f"{cid}: {text}" for cid, text in results))] + blocks)
            else:
                self._nudges += 1
                if self._nudges > 2:
                    return [], "no_tool_calls"
                b.push_nudge(blocks + [("text", "Reply with exactly one tool call.")])
        try:
            self.tracing.bind(turn=self.calls + 1, consecutive_failures=self._fails)
            calls, refusal = b.request()
        except Exception as e:  # noqa: BLE001 -- the arm must hold, not crash
            self._fails += 1
            self._retry = True
            log.warning("planner call failed (%s: %s) [%d consecutive]", type(e).__name__, e, self._fails)
            return [], ("failed" if self._fails >= 3 else "retry")
        self._retry = False
        self._fails = 0
        self.calls += 1
        if refusal:
            return [], "refusal"
        if calls:
            self._nudges = 0
        self._issued = [cid for cid, _name, _args in calls if cid]
        return [(cid, name, args if isinstance(args, dict) else {}) for cid, name, args in calls], None
