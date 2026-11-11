"""Summarise a run in progress: what is done, running, stalled or failed, success so far, speed and ETA.

    python scripts/progress.py --run-out "$RUN_OUT" [--run-out "$OTHER_RUN_OUT" ...] [--json progress.json]

Reads, for each run folder: jobs.txt (the jobs the run is meant to contain, from scripts/jobs.py), every
<job>/status.json (from scripts/run_slot.sh) and decisions.jsonl (the policy server's log). Several folders, for
example one per machine, are summarised together. Standard library only, and cheap enough to run every minute.

Writes progress.json (default $RUN_OUT/progress.json for a single folder) for a dashboard to read, and prints the same
summary. Success here is the judge's success read by run_slot.sh, oracle (any step) and at-end, as a progress signal;
scripts/score.py, which uses MolmoSpaces' own scorer, is the score.

A job is "stalled" when it is running and the server has logged nothing for its episode for --stall-minutes, and
the endpoint is flagged when no planner response has arrived for --stall-minutes while jobs are running. Both are
the usual signs of a hung simulator or LLM endpoint, as opposed to slow episodes.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def read_jobs(run_out: Path) -> list[dict]:
    path = run_out / "jobs.txt"
    if not path.is_file():
        return []
    jobs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 8:
            jobs.append({"job": parts[0], "set": parts[1], "benchmark": parts[2], "slot": int(parts[3])})
    return jobs


def read_status(run_out: Path) -> dict[str, dict]:
    out = {}
    for path in run_out.rglob("status.json"):
        if "attempts" in path.relative_to(run_out).parts:
            continue
        try:
            status = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # being replaced right now; the next run of this script sees it
        status["attempts"] = len(list((path.parent / "attempts").glob("*"))) + 1
        out[status["job"]] = status
    return out


def read_decisions(run_out: Path) -> dict:
    """Planner totals and the last time the server logged anything for each episode (by set/benchmark and slot)."""
    totals = Counter()
    last_activity, last_response, servers = {}, None, 0
    path = run_out / "decisions.jsonl"
    if not path.is_file():
        return {"totals": totals, "last_activity": last_activity, "last_response": None, "servers": 0}
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written
            event = row.get("event")
            if event == "server_started":
                servers += 1
            identity = row.get("identity")
            if identity:
                last_activity[(identity[0], identity[1])] = row["time"]
            if event == "planner_request":
                usage = row.get("usage") or {}
                totals["calls"] += 1
                totals["errors"] += bool(row.get("error"))
                totals["input_tokens"] += usage.get("input_tokens", 0)
                totals["cached_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                totals["output_tokens"] += usage.get("output_tokens", 0)
                if not row.get("error"):
                    last_response = max(last_response or 0, row["time"])
            elif event == "request_failed":
                totals["failed_requests"] += 1
    return {"totals": totals, "last_activity": last_activity, "last_response": last_response, "servers": servers}


def summarise(run_outs: list[Path], stall_minutes: float, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    jobs, statuses, totals = [], {}, Counter()
    last_activity, last_response, servers = {}, None, 0
    for run_out in run_outs:
        jobs += read_jobs(run_out)
        statuses.update(read_status(run_out))
        d = read_decisions(run_out)
        totals.update(d["totals"])
        last_activity.update(d["last_activity"])
        servers += d["servers"]
        if d["last_response"]:
            last_response = max(last_response or 0, d["last_response"])
    expected = {j["job"]: j for j in jobs}
    for job, status in statuses.items():  # a status with no jobs.txt entry still counts
        expected.setdefault(job, {"job": job, "set": status.get("set"), "benchmark": status.get("benchmark"),
                                  "slot": status.get("slot")})

    per = defaultdict(Counter)
    failures, stalled, durations, finished_times = [], [], [], []
    for job, info in sorted(expected.items()):
        status = statuses.get(job)
        state = status["state"] if status else "pending"
        b = per[info["benchmark"]]
        b["expected"] += 1
        b[state] += 1
        if state == "done":
            b["success_any"] += bool(status.get("success_any"))
            b["success_at_end"] += bool(status.get("success_at_end"))
            durations.append(status.get("duration_s") or 0)
            finished_times.append(status.get("finished_unix") or now)
        elif state == "failed":
            finished_times.append(status.get("finished_unix") or now)
            failures.append({"job": job, "benchmark": info["benchmark"], "exit_code": status.get("exit_code"),
                             "error": status.get("error"), "attempts": status.get("attempts"),
                             "host": status.get("host")})
        elif state == "running":
            started = status.get("started_unix") or now
            seen = last_activity.get((f"{info['set']}/{info['benchmark']}", info["slot"]))
            quiet = now - max(started, seen or started)
            if quiet > stall_minutes * 60:
                stalled.append({"job": job, "benchmark": info["benchmark"], "host": status.get("host"),
                                "running_min": round((now - started) / 60, 1), "quiet_min": round(quiet / 60, 1)})

    total = Counter()
    for b in per.values():
        total.update(b)
    hour_ago = now - 3600
    recent = sum(t >= hour_ago for t in finished_times)
    first_start = min((s.get("started_unix") for s in statuses.values() if s.get("started_unix")), default=None)
    elapsed_h = (now - first_start) / 3600 if first_start else 0
    rate = recent if elapsed_h >= 1 else (len(finished_times) / elapsed_h if elapsed_h > 0 else 0)
    remaining = total["pending"] + total["running"]
    running = total["running"]
    endpoint_quiet = (now - last_response) / 60 if last_response else None

    def rates(c):
        done = c["done"]
        return {"expected": c["expected"], "done": done, "failed": c["failed"], "running": c["running"],
                "pending": c["pending"], "success_any": c["success_any"], "success_at_end": c["success_at_end"],
                "success_any_pct": round(100 * c["success_any"] / done, 1) if done else None}

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "run_outs": [str(p) for p in run_outs],
        "totals": rates(total),
        "benchmarks": {name: rates(c) for name, c in sorted(per.items())},
        "speed": {"episodes_last_hour": recent, "episodes_per_hour": round(rate, 1),
                  "mean_episode_min": round(sum(durations) / len(durations) / 60, 1) if durations else None,
                  "elapsed_h": round(elapsed_h, 2),
                  "eta_h": round(remaining / rate, 1) if rate and remaining else (0 if not remaining else None)},
        "planner": {**{k: totals[k] for k in ("calls", "errors", "failed_requests", "input_tokens",
                                              "cached_input_tokens", "output_tokens")},
                    "last_response_age_min": round(endpoint_quiet, 1) if endpoint_quiet is not None else None,
                    "endpoint_quiet": bool(running and endpoint_quiet is not None and endpoint_quiet > stall_minutes),
                    "server_starts": servers},
        "failures": failures,
        "stalled": stalled,
    }


def show(p: dict) -> str:
    t, s, pl = p["totals"], p["speed"], p["planner"]
    out = [f"{p['generated_at']}  done {t['done']}/{t['expected']}  running {t['running']}  failed {t['failed']}  "
           f"pending {t['pending']}  success {t['success_any']}/{t['done']}"
           + (f" ({t['success_any_pct']}%)" if t["success_any_pct"] is not None else ""),
           f"speed {s['episodes_per_hour']}/h  mean episode {s['mean_episode_min']} min  ETA "
           + (f"{s['eta_h']} h" if s["eta_h"] is not None else "unknown"),
           f"planner {pl['calls']} calls, {pl['errors']} errors, {pl['input_tokens'] / 1e6:.1f}M in "
           f"({pl['cached_input_tokens'] / 1e6:.1f}M cached), {pl['output_tokens'] / 1e3:.0f}k out; last response "
           + (f"{pl['last_response_age_min']} min ago" if pl["last_response_age_min"] is not None else "none yet"),
           f"{'benchmark':18s} {'done':>9s} {'run':>4s} {'fail':>4s} {'success':>12s}"]
    for name, b in p["benchmarks"].items():
        pct = f"{b['success_any']}/{b['done']}" + (f" {b['success_any_pct']:.0f}%" if b["success_any_pct"] is not None else "")
        out.append(f"{name:18s} {b['done']:>4d}/{b['expected']:<4d} {b['running']:>4d} {b['failed']:>4d} {pct:>12s}")
    if pl["endpoint_quiet"]:
        out.append(f"WARNING: no planner response for {pl['last_response_age_min']} min while episodes are running")
    for st in p["stalled"]:
        out.append(f"STALLED {st['job']} on {st['host']}: running {st['running_min']} min, quiet {st['quiet_min']} min")
    for f in p["failures"]:
        out.append(f"FAILED {f['job']} (exit {f['exit_code']}, attempt {f['attempts']}): {f['error']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-out", type=Path, action="append", required=True, help="a run folder; repeat for several")
    ap.add_argument("--json", type=Path, help="where to write progress.json (default: <run-out>/progress.json for one)")
    ap.add_argument("--stall-minutes", type=float, default=15)
    a = ap.parse_args()
    p = summarise(a.run_out, a.stall_minutes)
    target = a.json or (a.run_out[0] / "progress.json" if len(a.run_out) == 1 else None)
    if target:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(p, indent=1) + "\n")
        tmp.replace(target)
    print(show(p))


if __name__ == "__main__":
    main()
