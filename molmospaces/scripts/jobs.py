"""Print one run_slot.sh job line per episode, for the MolmoSpaces Hundred or for full benchmarks.

    python scripts/jobs.py --assets "$MLSPACES_ASSETS_DIR" [--slots 1-52]                 # the Hundred
    python scripts/jobs.py --assets "$MLSPACES_ASSETS_DIR" --full all [--episodes 0-99]   # full benchmarks
    ... | tee "$RUN_OUT/jobs.txt" | xargs -P 10 -L 1 scripts/run_slot.sh

A job line is `job set benchmark slot benchmark_dir episode_index horizon_s left,right,wrist`:
- job: the episode's folder under $RUN_OUT (`slot-001` for the Hundred, `open-v1/ep-0033` for a full benchmark);
- set: `molmospaces-hundred` or `molmospaces`, recorded by the server with the benchmark name;
- slot: the episode's number within the set (the Hundred slot, or the episode index of a full benchmark).

Full benchmarks take each episode's time limit from MolmoSpaces' own benchmark.json, and the cameras and benchmark
folder from benchmark/molmospaces-hundred.json, which records them per benchmark. Keep the printed list as
$RUN_OUT/jobs.txt: scripts/progress.py and scripts/score.py read it to know what the run is meant to contain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark/molmospaces-hundred.json"
HUNDRED, FULL = "molmospaces-hundred", "molmospaces"


def job_line(job, set_name, benchmark, slot, benchmark_dir, index, horizon_s, cams) -> str:
    if len(cams) != 3:
        raise ValueError(f"{job}: policy_cameras must be [left, right, wrist]")
    for value in (job, set_name, benchmark, str(benchmark_dir)):
        if not value or any(c.isspace() for c in value):
            raise ValueError(f"{job}: job fields must be non-empty and contain no spaces: {value!r}")
    return f"{job} {set_name} {benchmark} {slot} {benchmark_dir} {index} {horizon_s:g} {','.join(cams)}"


def lines(benchmark: dict, assets: Path, wanted: set[int] | None = None) -> list[str]:
    """The Hundred's slots."""
    return [job_line(f"slot-{s['slot']:03d}", HUNDRED, s["benchmark"], s["slot"],
                     assets / "benchmarks" / s["benchmark_dir"], s["episode_index"], float(s["episode_seconds"]),
                     s["policy_cameras"])
            for s in benchmark["slots"] if wanted is None or s["slot"] in wanted]


def benchmarks(benchmark: dict) -> dict[str, dict]:
    """Per benchmark: its folder and cameras, which every Hundred slot of that benchmark shares."""
    out = {}
    for s in benchmark["slots"]:
        entry = {"benchmark_dir": s["benchmark_dir"], "policy_cameras": s["policy_cameras"]}
        if out.setdefault(s["benchmark"], entry) != entry:
            raise ValueError(f"{s['benchmark']}: slots disagree on benchmark folder or cameras")
    return out


def full_lines(benchmark: dict, assets: Path, names: list[str], wanted: set[int] | None = None) -> list[str]:
    """Every episode (or the wanted episode indices) of each named full benchmark."""
    known = benchmarks(benchmark)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown benchmarks {unknown}; known: {', '.join(known)}")
    out = []
    for name in names:
        folder = assets / "benchmarks" / known[name]["benchmark_dir"]
        episodes = json.loads((folder / "benchmark.json").read_text())
        for index, episode in enumerate(episodes):
            if wanted is not None and index not in wanted:
                continue
            horizon = (episode.get("task") or {}).get("task_horizon_sec")
            if horizon is None:
                raise ValueError(f"{folder}/benchmark.json episode {index}: no task.task_horizon_sec")
            out.append(job_line(f"{name}/ep-{index:04d}", FULL, name, index, folder, index, float(horizon),
                                known[name]["policy_cameras"]))
    return out


def parse_slots(text: str) -> set[int]:
    wanted = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        wanted.update(range(int(a), int(b or a) + 1))
    return wanted


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--assets", type=Path, required=True, help="MLSPACES_ASSETS_DIR (holds benchmarks/)")
    ap.add_argument("--slots", help="Hundred slots, e.g. 1-52 or 3,7,10-12 (default: all)")
    ap.add_argument("--full", help="full benchmarks instead of the Hundred: comma-separated names, or 'all'")
    ap.add_argument("--episodes", help="with --full: episode indices, e.g. 0-99 (default: every episode)")
    a = ap.parse_args()
    benchmark = json.loads(a.benchmark.read_text())
    if a.full:
        if a.slots:
            ap.error("--slots selects Hundred slots; use --episodes with --full")
        names = list(benchmarks(benchmark)) if a.full == "all" else a.full.split(",")
        out = full_lines(benchmark, a.assets, names, parse_slots(a.episodes) if a.episodes else None)
    else:
        if a.episodes:
            ap.error("--episodes needs --full")
        out = lines(benchmark, a.assets, parse_slots(a.slots) if a.slots else None)
    print("\n".join(out))


if __name__ == "__main__":
    main()
