"""Score a finished run (the MolmoSpaces Hundred or full benchmarks) with MolmoSpaces' own scorer.

    "$MOLMOSPACES_PYTHON" scripts/score.py --run-out "$RUN_OUT" [--molmospaces "$MOLMOSPACES_SRC"]
        [--policy-name NAME] [--json results.json]

The run's jobs come from $RUN_OUT/jobs.txt (the output of scripts/jobs.py); a run folder without one is read as the
Hundred laid out as slot-NNN. Scoring uses MolmoSpaces' scripts/benchmarks/eval_to_csv.py:
- each episode's success is its get_success_any (oracle: success at any step, which the leaderboard ranks) and
  get_success_last_frame (at-end), applied to the episode's trajectory;
- each benchmark is then scored with eval_to_csv itself, --success-condition both, as MolmoSpaces' benchmark docs run
  it. Its per-category CSV, the file a leaderboard entry attaches, goes to $RUN_OUT/scores/<benchmark>.csv, and the
  per-benchmark counts printed here are its OVERALL row. --policy-name sets the name in those CSVs; the name is not
  part of the policy, so a run can be rescored under another name at any time.

On top it adds the missing episodes, the per-benchmark counts and All Combined, the mean of the nine benchmark
rates with pick-v2-filament taken from pick-v2-classic, printed only when every benchmark has a result.

It also records what a submission must state, so a run folder is self-describing:
- each episode's rows of the run's decisions.jsonl (the policy server's log: reasoning summaries, tool calls and
  results) go to <job>/decisions.jsonl, headed by the startup record of the server that ran that episode;
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
POLICY = "silos-0"


def load_scorer(molmospaces: Path):
    path = molmospaces / "scripts/benchmarks/eval_to_csv.py"
    if not path.is_file():
        raise SystemExit(f"{path} not found: pass --molmospaces (or set MOLMOSPACES_SRC) to the v0.2.9 checkout")
    spec = importlib.util.spec_from_file_location("molmospaces_eval_to_csv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jobs(run_out: Path, benchmark_path: Path) -> list[dict]:
    """The run's jobs from jobs.txt, or the Hundred's slots for a run folder without one."""
    path = run_out / "jobs.txt"
    if path.is_file():
        jobs = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 8:
                jobs.append({"job": parts[0], "set": parts[1], "benchmark": parts[2], "slot": int(parts[3])})
        return jobs
    slots = json.loads(benchmark_path.read_text())["slots"]
    return [{"job": f"slot-{s['slot']:03d}", "set": "molmospaces-hundred", "benchmark": s["benchmark"],
             "slot": s["slot"]} for s in slots]


def episode_success(scorer, eval_dir: Path) -> tuple[Path, bool, bool] | None:
    """(trajectory, oracle, at-end) with MolmoSpaces' success functions, or None when the episode has no result."""
    import h5py
    files = sorted(eval_dir.rglob("*.h5")) if eval_dir.is_dir() else []
    if not files:
        return None
    if len(files) > 1:
        raise SystemExit(f"{eval_dir}: {len(files)} trajectory files; rerun this episode into a clean folder")
    with h5py.File(files[0], "r") as f:
        trajs = [k for k in f if k.startswith("traj_")]
        if trajs != ["traj_0"]:
            raise SystemExit(f"{files[0]}: expected one episode (traj_0), found {trajs}")
        success = f["traj_0/success"][:]
    return files[0], scorer.get_success_any(success), scorer.get_success_last_frame(success)


def overall(scorer, folder: Path, csv: Path, policy: str) -> tuple[int, int, int]:
    """eval_to_csv's (oracle successes, at-end successes, episodes) for every trajectory under folder.
    With success_condition "both", its plain `successes` column is the at-end count."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = scorer.eval_to_csv(run_path=str(folder), policy_name=policy, success_condition="both",
                                output_csv=str(csv), dt=0.066)
    row = df[df["category"] == "OVERALL"].iloc[0]
    return int(row["oracle_successes"]), int(row["successes"]), int(row["total"])


def split_decisions(run_out: Path, folders: dict[tuple[str, int], str]) -> tuple[int, list[dict]]:
    """Write each episode's rows of $RUN_OUT/decisions.jsonl to <job>/decisions.jsonl, after the startup record of
    the server that was running when the episode's first row was logged. `folders` maps the server's episode identity
    (set/benchmark, slot) to the job folder. Returns (rows written, distinct server metadata)."""
    log = run_out / "decisions.jsonl"
    if not log.is_file():
        return 0, []
    rows, servers, current = defaultdict(list), [], None
    with log.open() as f:
        for line in f:
            line = line.rstrip("\n")
            row = json.loads(line)
            if row.get("event") == "server_started":
                current = line
                if row["metadata"] not in servers:
                    servers.append(row["metadata"])
                continue
            identity = row.get("identity") or [None, None]
            job = folders.get((identity[0], identity[1]))
            if job is not None:
                if not rows[job] and current is not None:
                    rows[job].append(current)
                rows[job].append(line)
    for job, lines in rows.items():
        (run_out / job / "decisions.jsonl").write_text("\n".join(lines) + "\n")
    return sum(len(v) for v in rows.values()), servers


def version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment(molmospaces: Path, logs: list[Path]) -> dict:
    """The software a submission must name, read from this interpreter, the checkout and the episodes' own logs."""
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(molmospaces), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    renderers, data_versions = set(), None
    for log in logs:
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
    ap.add_argument("--policy-name", default=POLICY, help=f"name written into the CSVs (default: {POLICY})")
    ap.add_argument("--json", type=Path, help="also write per-episode and per-benchmark results here")
    a = ap.parse_args()
    if a.molmospaces is None:
        raise SystemExit("pass --molmospaces or set MOLMOSPACES_SRC")
    scorer = load_scorer(a.molmospaces)
    jobs = read_jobs(a.run_out, a.benchmark)
    scores = a.run_out / "scores"
    scores.mkdir(parents=True, exist_ok=True)

    per_episode, by_benchmark, logs = [], defaultdict(list), []
    with tempfile.TemporaryDirectory() as tmp:
        for j in jobs:
            folder = a.run_out / j["job"]
            found = episode_success(scorer, folder / "eval")
            entry = {"job": j["job"], "benchmark": j["benchmark"], "slot": j["slot"], "success": None,
                     "success_at_end": None}
            if found is not None:
                trajectory, oracle, at_end = found
                entry.update(success=oracle, success_at_end=at_end)
                link = Path(tmp) / j["benchmark"] / (j["job"].replace("/", "__") + ".h5")
                link.parent.mkdir(exist_ok=True)
                link.symlink_to(trajectory.resolve())
                by_benchmark[j["benchmark"]].append(j["job"])
                logs += sorted((folder / "eval").glob("*/*/running_log.log"))
            per_episode.append(entry)
        # One eval_to_csv pass per benchmark over its finished episodes: the CSV a submission attaches, and the counts.
        tally = {name: overall(scorer, Path(tmp) / name, scores / f"{name}.csv", a.policy_name) for name in by_benchmark}
    for name in tally:  # the CSV names the temporary folder it was scored from; name the run instead
        csv = scores / f"{name}.csv"
        csv.write_text("".join(f"# run_path: {a.run_out.name} ({name})\n" if line.startswith("# run_path:")
                               else line for line in csv.read_text().splitlines(keepends=True)))
    for name, (wins, at_end, n) in tally.items():  # the per-episode reading must agree with eval_to_csv's totals
        mine = [e for e in per_episode if e["benchmark"] == name and e["success"] is not None]
        if (sum(e["success"] for e in mine), sum(e["success_at_end"] for e in mine), len(mine)) != (wins, at_end, n):
            raise SystemExit(f"{name}: per-episode success disagrees with eval_to_csv's totals")

    missing = [e["job"] for e in per_episode if e["success"] is None]
    finished = {(f"{j['set']}/{j['benchmark']}", j["slot"]): j["job"] for j, e in zip(jobs, per_episode)
                if e["success"] is not None}
    split, servers = split_decisions(a.run_out, finished)
    if servers:
        (a.run_out / "policy.json").write_text(json.dumps(servers[0] if len(servers) == 1 else servers, indent=1) + "\n")
    env = environment(a.molmospaces, logs)
    (a.run_out / "environment.json").write_text(json.dumps(env, indent=1) + "\n")
    print(f"{'benchmark':18s} oracle         (at-end)")
    for name in sorted(tally, key=lambda n: -tally[n][0] / tally[n][2]):
        wins, at_end, n = tally[name]
        print(f"{name:18s} {wins:4d}/{n:<4d} {100 * wins / n:5.1f}%   ({at_end}/{n})")
    wins, at_end, n = (sum(t[i] for t in tally.values()) for i in range(3))
    print(f"{'all':18s} {wins:4d}/{n:<4d}" + (f" {100 * wins / n:5.1f}%   ({at_end}/{n})" if n else ""))
    rates = {b: tally.get(SCORED_BY.get(b, b)) for b in ALL_COMBINED}
    combined = (100 * sum(w / k for w, _, k in rates.values()) / len(rates)) if all(rates.values()) else None
    print("All Combined estimate: " + (f"{combined:.1f}" if combined is not None else "needs every benchmark"))
    if missing:
        print(f"missing ({len(missing)} of {len(jobs)}): {', '.join(missing[:20])}" + (" ..." if len(missing) > 20 else ""))
    print(f"per-category CSVs from MolmoSpaces' scorer: {scores}")
    print(f"decision log rows split into episode folders: {split}")
    if len(servers) > 1:
        print(f"WARNING: {len(servers)} different server configurations ran in this run; see policy.json")
    print(f"policy: {a.run_out / 'policy.json'}" if servers else "policy: no server startup record in decisions.jsonl")
    print(f"environment: MolmoSpaces {env['molmospaces']['describe']}, mujoco {env['packages']['mujoco']}, "
          f"mujoco-warp {env['packages']['mujoco-warp']}, renderer {','.join(env['renderers']) or 'unknown'}")
    if a.json:
        a.json.write_text(json.dumps({"episodes": per_episode,
                                      "benchmarks": {k: {"oracle_successes": v[0], "at_end_successes": v[1],
                                                         "episodes": v[2]} for k, v in tally.items()},
                                      "all_combined": combined, "missing": missing}, indent=1) + "\n")


if __name__ == "__main__":
    main()
