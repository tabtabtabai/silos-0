"""Score a finished MolmoSpaces Hundred run with MolmoSpaces' own scorer, then report the All Combined estimate.

    "$MOLMOSPACES_PYTHON" scripts/score.py --run-out "$RUN_OUT" [--molmospaces "$MOLMOSPACES_SRC"] [--json results.json]

Per-episode success comes from MolmoSpaces' scripts/benchmarks/eval_to_csv.py with --success-condition both, as
MolmoSpaces' benchmark docs run it, called on each slot's output. The numbers reported here are its oracle columns
(success at any step), which the leaderboard ranks; the at-end count is reported beside them. Its per-category CSV
for each benchmark, in the format a leaderboard entry attaches, is written to $RUN_OUT/scores/<benchmark>.csv.

This script adds only what the subset needs: which slots are missing, the per-benchmark counts, and All Combined,
the mean of the nine benchmark rates with pick-v2-filament taken from the pick-v2-classic slots. All Combined is
printed only when every benchmark has a result.

It also records what a submission must state, so a run folder is self-describing:
- each slot's rows of the run's decisions.jsonl (the policy server's log: reasoning summaries, tool calls and
  results) go to slot-NNN/decisions.jsonl, headed by the startup record of the server that ran that episode, so one
  episode folder holds its videos, trajectory, reasoning and the policy that produced them;
- $RUN_OUT/policy.json: that startup record (policy, model and effort, call budget, prompt hash, config, source
  revision), one entry per distinct server configuration seen in the run;
- $RUN_OUT/environment.json: the MolmoSpaces commit, the simulator package versions, the renderer and data versions
  the episodes logged, and the platform.

Run it with the MolmoSpaces environment's Python (it needs h5py, pandas and scipy).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark/molmospaces-hundred.json"
ALL_COMBINED = ["open-v1", "close-v1", "pick-v1.5", "pick-v2-classic", "pick-v2-filament", "pick-v2-rand-cam",
                "pnp-v2", "pnp-next-to-v2", "pnp-color-v2"]
SCORED_BY = {"pick-v2-filament": "pick-v2-classic"}
POLICY = "sol200-flagshipv3b"


def load_scorer(molmospaces: Path):
    path = molmospaces / "scripts/benchmarks/eval_to_csv.py"
    if not path.is_file():
        raise SystemExit(f"{path} not found: pass --molmospaces (or set MOLMOSPACES_SRC) to the v0.2.9 checkout")
    spec = importlib.util.spec_from_file_location("molmospaces_eval_to_csv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_to_csv


def overall(scorer, folder: Path, csv: Path) -> tuple[int, int, int]:
    """MolmoSpaces' (oracle successes, at-end successes, episodes) for every trajectory file under folder.
    With success_condition "both", eval_to_csv's plain `successes` column is the at-end count."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = scorer(run_path=str(folder), policy_name=POLICY, success_condition="both", output_csv=str(csv),
                    dt=0.066)
    row = df[df["category"] == "OVERALL"].iloc[0]
    return int(row["oracle_successes"]), int(row["successes"]), int(row["total"])


def split_decisions(run_out: Path, slots: set[int]) -> tuple[int, list[dict]]:
    """Write each slot's rows of $RUN_OUT/decisions.jsonl to slot-NNN/decisions.jsonl, after the startup record of
    the server that was running when the slot's first row was logged. Returns (rows written, distinct server
    metadata). The run-level file is left as is."""
    log = run_out / "decisions.jsonl"
    if not log.is_file():
        return 0, []
    rows, servers, current = defaultdict(list), [], None
    for line in log.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "server_started":
            current = line
            if row["metadata"] not in servers:
                servers.append(row["metadata"])
            continue
        identity = row.get("identity") or [None, None]
        if identity[1] in slots:
            if not rows[identity[1]] and current is not None:
                rows[identity[1]].append(current)
            rows[identity[1]].append(line)
    for slot, lines in rows.items():
        (run_out / f"slot-{slot:03d}" / "decisions.jsonl").write_text("\n".join(lines) + "\n")
    return sum(len(v) for v in rows.values()), servers


def version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment(molmospaces: Path, run_out: Path) -> dict:
    """The software a submission must name, read from this interpreter, the checkout and the episodes' own logs."""
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(molmospaces), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    renderers, data_versions = set(), None
    for log in sorted(run_out.glob("slot-*/eval/*/*/running_log.log")):
        text = log.read_text(errors="replace")
        renderers.update(re.findall(r"Using MuJoCo renderer: (\w+)", text))
        if data_versions is None:
            found = re.search(r"Data versions in use: (\{.*\})", text)
            if found:
                data_versions = ast.literal_eval(found.group(1))
    return {
        "molmospaces": {"commit": git("rev-parse", "HEAD"), "describe": git("describe", "--tags", "--always"),
                        "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))},
        "packages": {name: version(name) for name in ("molmo-spaces", "mujoco", "mujoco-warp", "warp-lang", "numpy")},
        "renderers": sorted(renderers),
        "data_versions": data_versions,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--run-out", type=Path, required=True)
    ap.add_argument("--molmospaces", type=Path, default=os.environ.get("MOLMOSPACES_SRC"),
                    help="MolmoSpaces v0.2.9 checkout (default: $MOLMOSPACES_SRC)")
    ap.add_argument("--json", type=Path, help="also write per-slot and per-benchmark results here")
    a = ap.parse_args()
    if a.molmospaces is None:
        raise SystemExit("pass --molmospaces or set MOLMOSPACES_SRC")
    scorer = load_scorer(a.molmospaces)
    benchmark = json.loads(a.benchmark.read_text())
    scores = a.run_out / "scores"
    scores.mkdir(parents=True, exist_ok=True)

    per_slot, by_benchmark = [], defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        for s in benchmark["slots"]:
            out = a.run_out / f"slot-{s['slot']:03d}" / "eval"
            h5s = sorted(out.rglob("*.h5")) if out.is_dir() else []
            if len(h5s) > 1:
                raise SystemExit(f"{out}: {len(h5s)} trajectory files; rerun this slot into a clean folder")
            ok = None
            if h5s:
                wins, _, n = overall(scorer, out, Path(tmp) / "slot.csv")
                if n != 1:
                    raise SystemExit(f"{out}: expected one episode, MolmoSpaces' scorer found {n}")
                ok = bool(wins)
                link = Path(tmp) / s["benchmark"] / f"slot-{s['slot']:03d}.h5"
                link.parent.mkdir(exist_ok=True)
                link.symlink_to(h5s[0].resolve())
                by_benchmark[s["benchmark"]].append(s["slot"])
            per_slot.append({"slot": s["slot"], "benchmark": s["benchmark"], "success": ok})
        # One MolmoSpaces scoring pass per benchmark over its finished slots, for the per-category CSVs.
        tally = {name: overall(scorer, Path(tmp) / name, scores / f"{name}.csv") for name in by_benchmark}
    for name in tally:  # the CSV names the temporary folder it was scored from; name the run's slots instead
        csv = scores / f"{name}.csv"
        csv.write_text("".join(f"# run_path: {a.run_out.name}/slot-* ({name})\n" if line.startswith("# run_path:")
                               else line for line in csv.read_text().splitlines(keepends=True)))

    missing = [r["slot"] for r in per_slot if r["success"] is None]
    split, servers = split_decisions(a.run_out, {r["slot"] for r in per_slot if r["success"] is not None})
    if servers:
        (a.run_out / "policy.json").write_text(json.dumps(servers[0] if len(servers) == 1 else servers, indent=1) + "\n")
    env = environment(a.molmospaces, a.run_out)
    (a.run_out / "environment.json").write_text(json.dumps(env, indent=1) + "\n")
    print(f"{'benchmark':18s} oracle         (at-end)")
    for name in sorted(tally, key=lambda n: -tally[n][0] / tally[n][2]):
        wins, at_end, n = tally[name]
        print(f"{name:18s} {wins:3d}/{n:<3d} {100 * wins / n:5.1f}%   ({at_end}/{n})")
    wins, at_end, n = (sum(t[i] for t in tally.values()) for i in range(3))
    print(f"{'all':18s} {wins:3d}/{n:<3d}" + (f" {100 * wins / n:5.1f}%   ({at_end}/{n})" if n else ""))
    rates = {b: tally.get(SCORED_BY.get(b, b)) for b in ALL_COMBINED}
    combined = (100 * sum(w / k for w, _, k in rates.values()) / len(rates)) if all(rates.values()) else None
    print("All Combined estimate: " + (f"{combined:.1f}" if combined is not None else "needs every benchmark"))
    if missing:
        print(f"missing ({len(missing)}): {','.join(map(str, missing))}")
    print(f"per-category CSVs from MolmoSpaces' scorer: {scores}")
    print(f"decision log rows split into slot folders: {split}")
    if len(servers) > 1:
        print(f"WARNING: {len(servers)} different server configurations ran in this run; see policy.json")
    print(f"policy: {a.run_out / 'policy.json'}" if servers else "policy: no server startup record in decisions.jsonl")
    print(f"environment: MolmoSpaces {env['molmospaces']['describe']}, mujoco {env['packages']['mujoco']}, "
          f"mujoco-warp {env['packages']['mujoco-warp']}, renderer {','.join(env['renderers']) or 'unknown'}")
    if a.json:
        a.json.write_text(json.dumps({"slots": per_slot,
                                      "benchmarks": {k: {"oracle_successes": v[0], "at_end_successes": v[1],
                                                         "episodes": v[2]} for k, v in tally.items()},
                                      "all_combined": combined, "missing": missing}, indent=1) + "\n")


if __name__ == "__main__":
    main()
