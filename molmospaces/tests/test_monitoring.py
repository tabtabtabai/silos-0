"""The run tooling: full-benchmark job lists, per-episode status, the progress summary and the scorer's job list."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark/molmospaces-hundred.json"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_assets(tmp_path, episodes=3):
    """A benchmarks/ tree holding a small benchmark.json for every benchmark the Hundred names."""
    for name, entry in script("jobs").benchmarks(json.loads(BENCHMARK.read_text())).items():
        folder = tmp_path / "benchmarks" / entry["benchmark_dir"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "benchmark.json").write_text(json.dumps(
            [{"task": {"task_horizon_sec": 20 + i}, "language": {"task_description": f"task {i}"}}
             for i in range(episodes)]))
    return tmp_path


def test_full_benchmark_jobs_cover_every_episode_with_its_own_time_limit(tmp_path):
    jobs = script("jobs")
    benchmark = json.loads(BENCHMARK.read_text())
    assets = fake_assets(tmp_path)
    names = list(jobs.benchmarks(benchmark))
    assert len(names) == 8 and "pick-v2-filament" not in names
    lines = jobs.full_lines(benchmark, assets, names)
    assert len(lines) == 8 * 3
    job, set_name, bench, slot, bdir, idx, horizon, cams = lines[2].split()
    assert (job, set_name, bench, slot, idx, horizon) == ("open-v1/ep-0002", "molmospaces", "open-v1", "2", "2", "22")
    assert Path(bdir, "benchmark.json").is_file() and cams.count(",") == 2
    rand_cam = [line for line in lines if line.startswith("pick-v2-rand-cam/")][0].split()[7]
    assert rand_cam == "randomized_zed2_analogue_1,wrist_camera_zed_mini,wrist_camera_zed_mini"
    assert len(jobs.full_lines(benchmark, assets, ["close-v1"], {0, 2})) == 2
    with pytest.raises(ValueError, match="unknown benchmarks"):
        jobs.full_lines(benchmark, assets, ["nope"])


def test_episode_status_marks_a_crash_failed_with_its_error(tmp_path):
    status = ROOT / "scripts/episode_status.py"
    episode = tmp_path / "open-v1" / "ep-0003"
    episode.mkdir(parents=True)
    subprocess.run([sys.executable, str(status), "start", str(episode), "open-v1/ep-0003", "molmospaces",
                    "open-v1", "3", "3"], check=True)
    started = json.loads((episode / "status.json").read_text())
    assert started["state"] == "running" and started["slot"] == 3 and started["host"]
    (episode / "eval.log").write_text("loading\nTraceback (most recent call last):\nValueError: house failed\n")
    subprocess.run([sys.executable, str(status), "finish", str(episode), "1"], check=True)
    finished = json.loads((episode / "status.json").read_text())
    assert finished["state"] == "failed" and finished["exit_code"] == 1
    assert finished["error"] == "ValueError: house failed" and finished["duration_s"] >= 0
    # Exit 0 without a trajectory is not a scorable episode either.
    subprocess.run([sys.executable, str(status), "finish", str(episode), "0"], check=True)
    assert json.loads((episode / "status.json").read_text())["state"] == "failed"


def test_progress_counts_states_success_failures_stalls_and_tokens(tmp_path):
    progress = script("progress")
    now = 1_000_000.0
    run = tmp_path / "run"
    jobs = [f"open-v1/ep-000{i} molmospaces open-v1 {i} /b {i} 30 a,b,b" for i in range(4)]
    run.mkdir()
    (run / "jobs.txt").write_text("\n".join(jobs) + "\n")

    def status(i, **fields):
        folder = run / "open-v1" / f"ep-000{i}"
        folder.mkdir(parents=True)
        (folder / "status.json").write_text(json.dumps({"job": f"open-v1/ep-000{i}", "set": "molmospaces",
                                                        "benchmark": "open-v1", "slot": i, "host": "vm-1", **fields}))
        return folder
    status(0, state="done", success_any=True, success_at_end=True, started_unix=now - 900, finished_unix=now - 600,
           duration_s=300)
    status(1, state="failed", exit_code=1, error="ValueError: x", started_unix=now - 900, finished_unix=now - 800)
    (status(2, state="running", started_unix=now - 3000) / "attempts" / "1").mkdir(parents=True)
    rows = [{"time": now - 2000, "event": "server_started", "metadata": {}},
            {"time": now - 1990, "event": "planner_request", "identity": ["molmospaces/open-v1", 2, 0, 0],
             "usage": {"input_tokens": 1000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 400}}}]
    (run / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    p = progress.summarise([run], stall_minutes=15, now=now)
    t = p["totals"]
    assert t["expected"] == 4
    assert (t["done"], t["failed"], t["running"], t["pending"], t["success_any"]) == (1, 1, 1, 1, 1)
    assert p["failures"][0]["error"] == "ValueError: x"
    assert p["stalled"][0]["job"] == "open-v1/ep-0002" and p["stalled"][0]["quiet_min"] > 15
    assert [r["job"] for r in p["failures"]] == ["open-v1/ep-0001"]
    assert p["planner"]["calls"] == 1 and p["planner"]["input_tokens"] == 1000
    assert p["planner"]["endpoint_quiet"] is True and p["planner"]["server_starts"] == 1
    assert p["speed"]["eta_h"] is not None
    assert "STALLED open-v1/ep-0002" in progress.show(p) and "FAILED open-v1/ep-0001" in progress.show(p)


def test_scorer_reads_jobs_txt_and_falls_back_to_the_hundred(tmp_path):
    score = script("score")
    hundred = score.read_jobs(tmp_path, BENCHMARK)
    assert len(hundred) == 100 and hundred[0] == {"job": "slot-001", "set": "molmospaces-hundred",
                                                  "benchmark": "open-v1", "slot": 1}
    (tmp_path / "jobs.txt").write_text("close-v1/ep-0007 molmospaces close-v1 7 /b 7 25 a,b,b\n")
    assert score.read_jobs(tmp_path, BENCHMARK) == [{"job": "close-v1/ep-0007", "set": "molmospaces",
                                                     "benchmark": "close-v1", "slot": 7}]
