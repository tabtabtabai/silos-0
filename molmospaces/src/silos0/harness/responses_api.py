"""The planner's OpenAI Responses API client (POST /responses): builds each request, retries with backoff and
trims old images from the conversation."""
import json
import logging
import os
import time
log = logging.getLogger(__name__)
SYSTEM_PROMPT = ""
TOOLS_SPEC = []
PROVIDER_BASE_URLS = {}
RETRY_STATUSES = (429, 502, 503)
RETRY_WAIT_S = 120.0

KEEP_IMAGES_LAST_N = 2

DEFAULT_MAX_OUTPUT_TOKENS = 4000

def _post_with_backoff(send, sleep=None):
    """send() -> requests.Response; retried with exponential backoff (2 s doubling to 30 s) on RETRY_STATUSES."""
    sleep = sleep or time.sleep
    delay, waited = 2.0, 0.0
    while True:
        r = send()
        if r.status_code not in RETRY_STATUSES or waited >= RETRY_WAIT_S:
            return r
        sleep(delay)
        waited += delay
        delay = min(delay * 2, 30.0)

class _ResponsesBackend:
    """OpenAI Responses wire (POST {base}/responses) over plain HTTP.

    Stateless (store=false): the whole item list is resent each call, so the
    newest-frames-only pruning works exactly as on the other backends.
    Output items (reasoning, function_call, message) go back verbatim;
    reasoning items carry their encrypted content for that purpose.
    Tool results are function_call_output TEXT items and the fresh frames
    ride a follow-up user message, as on the compat backend.

    The warmup ping sends the real tool list: on chat-completions an
    endpoint could answer a toolless ping and then 400 every tool-bearing
    call, which would park every episode
    while the server looked healthy."""

    def __init__(self, model, base_url, api_key, post=None, system=None, tools=None, observe=None,
                 reasoning_summary=None, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._reasoning_summary = reasoning_summary
        self._post = post or self._http_post
        if observe is not None:
            send = self._post
            self._post = lambda body: observe(body, send)
        self._effort = os.environ.get("SILOS0_LLM_REASONING_EFFORT", "").strip()
        self._system = SYSTEM_PROMPT if system is None else system
        self._tool_specs = TOOLS_SPEC if tools is None else tools   # `_tools()` below renders them
        self._items = []

    def _http_post(self, body):
        import requests  # lazy; ships with the serving image
        r = _post_with_backoff(lambda: requests.post(
            f"{self._base}/responses",
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json"},
            json=body, timeout=(10, 180)))
        if r.status_code >= 400:
            raise RuntimeError(f"LLM API returned HTTP {r.status_code}; check endpoint access and model availability")
        return r.json()

    def _tools(self):
        return [{"type": "function", "name": t["name"],
                 "description": t["description"], "parameters": t["parameters"]}
                for t in self._tool_specs]

    def _call(self, items, max_out):
        body = {"model": self._model, "instructions": self._system,
                "input": items, "tools": self._tools(),
                "max_output_tokens": max_out, "store": False,
                "include": ["reasoning.encrypted_content"]}
        if self._effort:
            body["reasoning"] = {"effort": self._effort}
        if self._reasoning_summary:
            body.setdefault("reasoning", {})["summary"] = self._reasoning_summary
        out = self._post(body)
        # Validate before storing: an error body or a missing output list
        # must be a retryable failure, not a poisoned conversation.
        if out.get("error") or not isinstance(out.get("output"), list):
            raise RuntimeError(
                f"provider returned no output: {json.dumps(out)[:300]}")
        return out

    def ping(self):
        out = self._call([{"role": "user", "content": [
            {"type": "input_text", "text": "Reply with the single word: ready"}]}],
            max_out=256)
        return out.get("status")

    def _content(self, blocks):
        out = []
        for kind, value in blocks:
            if kind == "text":
                out.append({"type": "input_text", "text": value})
            else:
                out.append({"type": "input_image",
                            "image_url": "data:image/jpeg;base64," + value})
        return out

    def push_opening(self, blocks):
        self._items.append({"role": "user", "content": self._content(blocks)})

    def push_results(self, pending_ids, blocks):
        text = "\n".join(v for k, v in blocks if k == "text")
        for tid in pending_ids:
            self._items.append({"type": "function_call_output", "call_id": tid,
                                "output": text or "executed"})
        images = [b for b in blocks if b[0] == "image"]
        if images:
            self._items.append({
                "role": "user",
                "content": self._content([("text", "Current camera views:")] + images),
            })

    def push_nudge(self, blocks):
        self._items.append({"role": "user", "content": self._content(blocks)})

    def request(self):
        self._prune()
        out = self._call(self._items, max_out=self._max_output_tokens)
        output = out["output"]
        self._items.extend(output)
        for item in output:
            if item.get("type") == "message" and any(
                    isinstance(c, dict) and c.get("type") == "refusal"
                    for c in item.get("content") or []):
                log.error("LLM refused (refusal content)")
                return [], True
        if (out.get("incomplete_details") or {}).get("reason") == "content_filter":
            log.error("LLM refused (incomplete: content_filter)")
            return [], True
        calls = []
        for item in output:
            if item.get("type") != "function_call":
                continue
            try:
                args = json.loads(item.get("arguments") or "{}")
            except (TypeError, ValueError):
                log.warning("unparseable tool arguments: %.200s", item.get("arguments"))
                args = {}
            calls.append((item.get("call_id"), item.get("name"), args))
        return calls, False

    def _prune(self):
        user_idx = [i for i, m in enumerate(self._items)
                    if isinstance(m, dict) and m.get("role") == "user"]
        for i in user_idx[:-KEEP_IMAGES_LAST_N]:
            content = self._items[i].get("content")
            if not isinstance(content, list):
                continue
            self._items[i]["content"] = [
                {"type": "input_text", "text": "[earlier camera frame removed to save context]"}
                if isinstance(b, dict) and b.get("type") == "input_image" else b
                for b in content
            ]
