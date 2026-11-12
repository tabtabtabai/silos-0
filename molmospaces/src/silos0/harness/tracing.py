"""Local-only tracing adapter. No external telemetry is sent."""
import re

def reasoning_summary(response, provider):
    """Provider-exposed text only; never infer it from tokens or tool arguments."""
    parts = []
    if provider == "openai-responses":
        for item in response.get("output") or []:
            if item.get("type") == "reasoning":
                parts.extend(part.get("text", "") for part in item.get("summary") or []
                             if part.get("type") == "summary_text")
    elif provider == "anthropic":
        parts.extend(part.get("thinking", "") for part in response.get("content") or []
                     if part.get("type") == "thinking")
    else:
        for choice in response.get("choices") or []:
            message = choice.get("message") or {}
            value = message.get("reasoning_content")
            if isinstance(value, str):
                parts.append(value)
            # Gemini's compatibility endpoint may return summaries inline.
            content = message.get("content")
            if isinstance(content, str):
                parts.extend(re.findall(r"<thought>(.*?)</thought>", content, flags=re.DOTALL))
    return "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())

class LLMRequestTrace:
    def __init__(self, **kwargs):
        self.reset()
    def reset(self):
        self.last_observation = {}
    def bind(self, **kwargs):
        pass
    def request(self, body, send):
        return send(body)

def flush():
    pass
