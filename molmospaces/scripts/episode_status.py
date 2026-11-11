"""Write an episode's status.json; run_slot.sh calls it when the episode starts and when it ends.

    python scripts/episode_status.py start  <episode dir> <job> <set> <benchmark> <slot> <episode_index>
    python scripts/episode_status.py finish <episode dir> <exit code>

status.json is what scripts/progress.py reads: the job, its benchmark, state (running, done or failed), host, start
and end times, duration, the eval's exit code and, once it ends, the judge's success read from MolmoSpaces'
trajectory file (any step, as the leaderboard ranks, and the last step) or the last error line of eval.log. It is a
progress signal; scripts/score.py remains the score. Writes are atomic, so a reader never sees half a file.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path


def write(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def trajectory(episode: Path) -> dict:
    """Judge success from the episode's trajectory file, or why it could not be read."""
    files = sorted((episode / "eval").rglob("trajectories_*.h5"))
    if len(files) != 1:
        return {"trajectory": None, "note": f"{len(files)} trajectory files"}
    try:
        import h5py
        import numpy as np
        with h5py.File(files[0], "r") as f:
            success = np.asarray(f["traj_0/success"][:], dtype=bool)
    except Exception as exc:  # a missing h5py or a damaged file must not lose the rest of the status
        return {"trajectory": str(files[0].relative_to(episode)), "note": f"unreadable: {type(exc).__name__}: {exc}"}
    return {"trajectory": str(files[0].relative_to(episode)), "steps": int(len(success)),
            "success_any": bool(success.any()), "success_at_end": bool(success[-1]) if len(success) else False}


def last_error(log: Path) -> str | None:
    """The last line that names an exception, or else the last non-empty line, of eval.log."""
    if not log.is_file():
        return None
    lines = [line.strip() for line in log.read_text(errors="replace").splitlines() if line.strip()]
    errors = [line for line in lines if "Error" in line or "Exception" in line]
    return (errors or lines or [None])[-1]


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("start", "finish"):
        print(__doc__, file=sys.stderr)
        return 2
    episode = Path(argv[2])
    path = episode / "status.json"
    if argv[1] == "start":
        job, set_name, benchmark, slot, index = argv[3:8]
        write(path, {"job": job, "set": set_name, "benchmark": benchmark, "slot": int(slot),
                     "episode_index": int(index), "state": "running", "host": socket.gethostname(),
                     "started_at": now(), "started_unix": time.time()})
        return 0
    status = json.loads(path.read_text()) if path.is_file() else {}
    code = int(argv[3])
    status.update(finished_at=now(), finished_unix=time.time(), exit_code=code,
                  duration_s=round(time.time() - status.get("started_unix", time.time()), 1))
    status.update(trajectory(episode))
    # An eval that exits 0 without a readable trajectory has not produced a scorable episode.
    status["state"] = "done" if code == 0 and "success_any" in status else "failed"
    if status["state"] == "failed":
        status["error"] = last_error(episode / "eval.log")
    write(path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
