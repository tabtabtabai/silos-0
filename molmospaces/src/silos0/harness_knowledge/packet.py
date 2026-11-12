"""Assemble the general knowledge packet for an LLM robot harness, and check it.

The packet is plain text appended to a harness planner's system prompt. It is built from three kinds of file
in this package, so each can change on its own:

- notes/<layer>/*.md   one rule per file (YAML front matter + the rule text + an optional "Why:" line).
                       layers: core (how to work), manipulation (general physics), model (one model family).
- tools/<harness>/*.yaml  one card per tool: purpose, when to use and not, what the result does and does not
                       show, limits (each with the code location that defines it), and what to do on errors.
- robots/*.yaml        physical facts about one robot.

Only the tools a harness actually exposes are rendered, so adding a tool means adding one card. Nothing here may
name a benchmark, task, scene or object: `leaks()` checks the rendered text against names taken from the
benchmark manifests in this repository.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VERSION = "0.2.0-draft"
LAYERS = ("core", "tools", "robot", "manipulation", "model")
DEFAULT_LAYERS = ("core", "tools", "robot", "manipulation", "model")
NOTE_FIELDS = {"id", "title", "layer", "status", "priority", "evidence", "confidence", "falsify"}
CARD_FIELDS = {"tool", "capability", "purpose", "use_when", "avoid_when", "proves", "not_proof", "how",
               "limits", "errors"}
STATUSES = {"candidate", "active", "retired"}
TOKEN_BUDGET = 4000          # rough tokens (words x 1.35) for the whole rendered packet, all layers


def token_budget(tools):
    """Base packet cap plus room for the optional batch/path tool cards."""
    return TOKEN_BUDGET + 200 * len(set(tools) & {"locate_many", "move_path"})


@dataclass
class Note:
    meta: dict
    rule: str
    why: str
    path: Path

    @property
    def id(self):
        return self.meta["id"]


@dataclass
class Packet:
    text: str
    note_ids: list
    tools: list
    sha: str
    words: int
    missing_cards: list = field(default_factory=list)

    @property
    def tokens(self):
        return int(self.words * 1.35)

    def provenance(self):
        return {"knowledge_version": VERSION, "knowledge_sha256": self.sha, "knowledge_notes": self.note_ids,
                "knowledge_tools": self.tools, "knowledge_tokens_est": self.tokens}


def load_notes(root: Path = HERE / "notes") -> list[Note]:
    notes = []
    for path in sorted(root.glob("*/*.md")):
        raw = path.read_text()
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
        if not m:
            raise ValueError(f"{path}: missing YAML front matter")
        meta, body = yaml.safe_load(m.group(1)), m.group(2).strip()
        rule, _, why = body.partition("\nWhy:")
        notes.append(Note(meta, " ".join(rule.split()), " ".join(why.split()), path))
    return notes


def load_cards(harness: str, root: Path = HERE / "tools") -> dict[str, dict]:
    cards = {}
    for path in sorted((root / harness).glob("*.yaml")):
        card = yaml.safe_load(path.read_text())
        card["_path"] = path
        cards[card["tool"]] = card
    return cards


def load_robot(name: str, root: Path = HERE / "robots") -> dict:
    return yaml.safe_load((root / f"{name}.yaml").read_text())


def fill(text: str, item: dict) -> str:
    return text.format(v=item.get("v")) if "{v" in text else text


def applies(item: dict, mode: str | None) -> bool:
    modes = item.get("modes")
    return not modes or mode in modes


def model_matches(note: Note, model: str | None) -> bool:
    pats = note.meta.get("models")
    return not pats or (model is not None and any(fnmatch.fnmatch(model, p) for p in pats))


def select_notes(notes, layer, features, model, statuses=("active",)):
    out = []
    for n in notes:
        if n.meta["layer"] != layer or n.meta["status"] not in statuses or not model_matches(n, model):
            continue
        if any(f not in features for f in n.meta.get("requires", [])):
            continue
        if any(f in features for f in n.meta.get("excludes", [])):
            continue
        out.append(n)
    return sorted(out, key=lambda n: (n.meta.get("priority", 99), n.id))


def render_note(n: Note, params: dict) -> str:
    text = f"[{n.id}] {n.rule}" + (f" Why: {n.why}" if n.why else "")
    return text.format(**params) if "{" in text else text


def render_card(card: dict, mode: str | None, features=()) -> str:
    for feature, variant in card.get("variants", {}).items():
        if feature in features:
            card = {**card, **variant}
    parts = [f"{card['tool']}: {card['purpose']}", f"Use it when {card['use_when']}; not when {card['avoid_when']}",
             f"Its result shows {card['proves']}, not {card['not_proof']}"]
    if card.get("how"):
        parts.append(card["how"])
    limits = [fill(item["text"], item) for item in card.get("limits") or [] if applies(item, mode)]
    if limits:
        parts.append("Limits: " + "; ".join(limits) + ".")
    errors = card.get("errors") or {}
    if errors:
        parts.append("If rejected: " + " ".join(f'"{k}": {v}' for k, v in errors.items()))
    return " ".join(p if p.endswith((".", ":")) else p + "." for p in parts)


def build(tools: list[str], harness: str = "harness", mode: str | None = None, robot: str | None = None,
          model: str | None = None, features=(), params: dict | None = None, layers=DEFAULT_LAYERS,
          statuses=("active",)) -> Packet:
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise ValueError(f"unknown layers {sorted(unknown)}; choose from {LAYERS}")
    params = {"max_calls": "a limited number of", **(params or {})}
    features = set(features)
    notes = load_notes()
    sections, ids = [], []

    def add_notes(layer, heading):
        chosen = select_notes(notes, layer, features, model, statuses)
        if chosen:
            sections.append(heading + "\n" + "\n".join(render_note(n, params) for n in chosen))
            ids.extend(n.id for n in chosen)

    if "core" in layers:
        add_notes("core", "HOW TO WORK")
    missing = []
    if "tools" in layers:
        cards = load_cards(harness)
        missing = [t for t in tools if t not in cards]
        if missing:
            raise ValueError(f"no knowledge card for tools {missing} in tools/{harness}/; add one per tool")
        sections.append("TOOLS\n" + "\n".join(render_card(cards[t], mode, features) for t in tools))
    if "robot" in layers and robot:
        r = load_robot(robot)
        sections.append(f"ROBOT: {r['title']}\n" + "\n".join("- " + fill(f["text"], f) for f in r["facts"]))
    if "manipulation" in layers:
        add_notes("manipulation", "HANDLING OBJECTS")
    if "model" in layers:
        add_notes("model", "NOTES FOR THIS MODEL")
    body = "\n\n".join(sections)
    sha = hashlib.sha256(body.encode()).hexdigest()[:16]
    head = (f"GENERAL OPERATING GUIDE (knowledge {VERSION}, {sha}). These notes hold for any task; "
            "the bracketed ids let you say which note you are following.")
    text = head + "\n\n" + body + "\n"
    return Packet(text, ids, list(tools), sha, len(text.split()), missing)


# ---------------------------------------------------------------------------------------------------- checks

# Words that occur in benchmark task or asset names but are part of any manipulation vocabulary. Adding one is a
# reviewed decision: it means the packet may use the word.
GENERIC = set("""a above agnostic all and around away behind big both by center clean cleaning clear container
control front grab in it items larger left measuring more move moving non object objects of off on one or order out
pick picking products put right sideways small smaller specified square stack stacking table take then three throw
to tool tools top total two unstack up vertical with""".split())
PHRASES = ("robolab", "hundred", "leaderboard", "benchmark", "earlier runs", "these tasks", "this task suite",
           "in earlier episodes", "success rate", "% of the time")
# Files whose task, scene and object names define what the packet must not mention. The original task and
# scene lists are not part of this package; the audit skips paths that do not exist.
TASK_LISTS = ()
ASSET_LISTS = ()


def denylist() -> set[str]:
    """Lower-case words naming benchmark tasks, scenes and objects, from the manifests in this repo."""
    words = set()
    for rel in TASK_LISTS:
        path = REPO / rel
        if not path.is_file():
            continue
        for t in json.loads(path.read_text())["tasks"]:
            words.add(t["name"].lower())
            words.update(w.lower() for w in re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])", t["name"].replace("Task", "")))
            words.update((t.get("scene") or "").lower().split("_"))
    for rel in ASSET_LISTS:
        path = REPO / rel
        if path.is_file():
            for name in re.findall(r'"(?:target|object|target_group|name|asset)"\s*:\s*"([a-z][a-z0-9_]+)"',
                                   path.read_text()):
                words.update(name.split("_"))
    return {w for w in words if len(w) > 2 and w.isalpha()} - GENERIC


def leaks(text: str, deny: set[str] | None = None) -> list[str]:
    deny = denylist() if deny is None else deny
    low = text.lower()
    hits = sorted({w for w in re.findall(r"[a-z][a-z']+", low) if w in deny})
    hits += [p for p in PHRASES if p in low]
    return hits


def lint() -> list[str]:
    """Problems in the note and card files (empty when clean)."""
    problems, seen = [], {}
    for n in load_notes():
        missing = NOTE_FIELDS - set(n.meta)
        if missing:
            problems.append(f"{n.path.name}: missing {sorted(missing)}")
        if n.meta.get("status") not in STATUSES:
            problems.append(f"{n.path.name}: status must be one of {sorted(STATUSES)}")
        if n.meta.get("layer") not in ("core", "manipulation", "model"):
            problems.append(f"{n.path.name}: layer must be core, manipulation or model")
        if n.id in seen:
            problems.append(f"{n.path.name}: id {n.id} also used by {seen[n.id]}")
        seen[n.id] = n.path.name
        if not n.meta.get("evidence"):
            problems.append(f"{n.path.name}: a rule needs at least one evidence line")
        if re.search(r"\b(MUST|NEVER|ALWAYS|CRITICAL|IMPORTANT)\b", n.rule):
            problems.append(f"{n.path.name}: write calmly, without capitalised emphasis")
        if len(n.rule.split()) > 130:
            problems.append(f"{n.path.name}: rule is {len(n.rule.split())} words; keep it under 130")
    for harness_dir in sorted((HERE / "tools").iterdir()):
        for tool, card in load_cards(harness_dir.name).items():
            missing = CARD_FIELDS - set(card)
            if missing:
                problems.append(f"{harness_dir.name}/{tool}: missing {sorted(missing)}")
    deny = denylist()
    for n in load_notes():
        hit = leaks(n.rule + " " + n.why, deny)
        if hit:
            problems.append(f"{n.path.name}: names benchmark content {hit}")
    for harness_dir in sorted((HERE / "tools").iterdir()):
        for tool, card in load_cards(harness_dir.name).items():
            for features in [(), *[(key,) for key in card.get("variants", {})]]:
                hit = leaks(render_card(card, None, features), deny)
                if hit:
                    problems.append(f"{harness_dir.name}/{tool}: names benchmark content {hit}")
    return problems


def harness_tools(mode: str) -> list[str]:
    from silos0.harness.planner import tools
    return [t["name"] for t in tools()]  # one tool set; mode is kept for the card filters


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m silos0.harness_knowledge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="print the packet for one harness configuration")
    b.add_argument("--harness", default="harness")
    b.add_argument("--mode", default="v3")
    b.add_argument("--tools", help="comma list; default: the harness's own tool list for --mode")
    b.add_argument("--robot", default="droid-franka-2f85")
    b.add_argument("--model")
    b.add_argument("--features", default="", help="comma list, e.g. budget_meter")
    b.add_argument("--param", action="append", default=[], help="key=value, e.g. max_calls=20")
    b.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    b.add_argument("--include-candidates", action="store_true")
    b.add_argument("--json", action="store_true", help="print provenance instead of the text")
    sub.add_parser("lint", help="check every note and card")
    args = ap.parse_args(argv)
    if args.cmd == "lint":
        problems = lint()
        print("\n".join(problems) or "clean")
        return 1 if problems else 0
    tools = args.tools.split(",") if args.tools else harness_tools(args.mode)
    params = dict(p.split("=", 1) for p in args.param)
    packet = build(tools, args.harness, args.mode, args.robot, args.model,
                   [f for f in args.features.split(",") if f], params, tuple(args.layers.split(",")),
                   ("active", "candidate") if args.include_candidates else ("active",))
    hit = leaks(packet.text)
    if hit:
        print(f"packet names benchmark content: {hit}", file=sys.stderr)
        return 1
    print(json.dumps(packet.provenance(), indent=1) if args.json else packet.text)
    print(f"[{packet.tokens} tokens est., budget {token_budget(tools)}]", file=sys.stderr)
    return 0 if packet.tokens <= token_budget(tools) else 2


if __name__ == "__main__":
    sys.exit(main())
