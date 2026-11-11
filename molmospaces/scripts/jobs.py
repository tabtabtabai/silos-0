"""Print one run_slot.sh job line per benchmark slot: `slot benchmark benchmark_dir idx horizon_s left,right,wrist`.

    python scripts/jobs.py --assets "$MLSPACES_ASSETS_DIR" [--slots 1-52] | xargs -P 10 -L 1 scripts/run_slot.sh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark/molmospaces-hundred.json"


def lines(benchmark: dict, assets: Path, wanted: set[int] | None = None) -> list[str]:
    out = []
    for s in benchmark["slots"]:
        if wanted is not None and s["slot"] not in wanted:
            continue
        cams = s["policy_cameras"]
        if len(cams) != 3:
            raise ValueError(f"slot {s['slot']}: policy_cameras must be [left, right, wrist]")
        out.append(f"{s['slot']:03d} {s['benchmark']} {assets / 'benchmarks' / s['benchmark_dir']} "
                   f"{s['episode_index']} {s['episode_seconds']} {','.join(cams)}")
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
    ap.add_argument("--slots", help="e.g. 1-52 or 3,7,10-12 (default: all)")
    a = ap.parse_args()
    benchmark = json.loads(a.benchmark.read_text())
    print("\n".join(lines(benchmark, a.assets, parse_slots(a.slots) if a.slots else None)))


if __name__ == "__main__":
    main()
