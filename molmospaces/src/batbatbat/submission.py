"""Launch the policy server with explicit, user-supplied LLM API access."""
import argparse
import asyncio
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from .harness.config import resolve_config

# MolmoSpaces' DROID robot differs from RoboLab's in two places: its MuJoCo Robotiq 2F-85 pads end 0.141 m from the
# gripper mount, and the arm is an FR3, whose joint ranges are intersected with the Panda model's.
MOLMOSPACES_ROBOT = {
    "BATBATBAT_FINGERTIP_OFFSET_M": "0.141",
    "BATBATBAT_JOINT_LIMITS_JSON": "[[-2.74,2.74],[-1.78,1.78],[-2.9,2.9],[-3.04,-0.15],[-2.81,2.81],[0.54,4.52],[-3.02,3.02]]",
}


def load_config(path, environ=None, *, base_url=None, api_key=None, model=None):
    """Returns (config, api_key). Explicit arguments win over BATBATBAT_LLM_BASE_URL / _API_KEY / _MODEL."""
    env = os.environ if environ is None else environ
    config = resolve_config(yaml.safe_load(Path(path).read_text()))
    base = (base_url or env.get("BATBATBAT_LLM_BASE_URL", "")).strip().rstrip("/")
    url = urlsplit(base)
    local = url.hostname in ("localhost", "127.0.0.1", "::1")
    if not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Pass --llm-base-url (or set BATBATBAT_LLM_BASE_URL): an API base URL without embedded credentials")
    if url.scheme != "https" and not (local and url.scheme == "http"):
        raise ValueError("Use HTTPS for the LLM API (HTTP is allowed only for local testing)")
    key = (api_key or env.get("BATBATBAT_LLM_API_KEY", "")).strip()
    if not key:
        raise ValueError("Pass --llm-api-key-file (or set BATBATBAT_LLM_API_KEY); never commit it")
    config["planner"]["base_url"] = base
    config["planner"]["model"] = (model or env.get("BATBATBAT_LLM_MODEL", "")).strip() or config["planner"]["model"]
    return config, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/submission.yaml"))
    parser.add_argument("--llm-base-url", help="API base serving POST /responses, including its version prefix "
                                               "(default: $BATBATBAT_LLM_BASE_URL)")
    parser.add_argument("--llm-api-key-file", type=Path, help="file holding the API key (default: $BATBATBAT_LLM_API_KEY)")
    parser.add_argument("--llm-api-key", help="the API key itself; visible in the process list, prefer the file or env")
    parser.add_argument("--llm-model", help="deployment alias serving the same model (default: the config's model)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log", type=Path, default=Path("outputs/decisions.jsonl"))
    args = parser.parse_args()
    try:
        key = args.llm_api_key or (args.llm_api_key_file.read_text() if args.llm_api_key_file else None)
        config, key = load_config(args.config, base_url=args.llm_base_url, api_key=key, model=args.llm_model)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # The planner reads these from the environment; the robot constants must be set before the server modules load.
    os.environ["BATBATBAT_LLM_API_KEY"] = key
    os.environ["BATBATBAT_LLM_REASONING_EFFORT"] = config["planner"]["effort"]
    os.environ.update(MOLMOSPACES_ROBOT)
    from .harness.server import DecisionLog, Service, serve

    logging.basicConfig(level=logging.INFO)
    record = DecisionLog(args.log)
    service = None
    try:
        service = Service(config, record)
        asyncio.run(serve(service, args.host, args.port))
    finally:
        if service is not None:
            service.close()
        record.close()


if __name__ == "__main__":
    main()
