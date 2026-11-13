# silos-0

Code to run the **silos-0** policy (an API-hosted LLM planner driving analytical motion
tools, with no VLA or model checkpoint) on public robot-manipulation benchmarks. Each benchmark has its own
self-contained directory with the policy server, the simulator client and run instructions.

| Directory | Benchmark | Status |
|---|---|---|
| [`molmospaces/`](molmospaces/) | [MolmoSpaces](https://github.com/allenai/molmospaces) (Ai2, MuJoCo, DROID Franka) | code and run instructions here |
| `robolab/` | [RoboLab](https://github.com/NVlabs/RoboLab) (NVIDIA, Isaac Lab) | not yet moved here |

## Results

The same planner prompt and tools produced both scores.

| Benchmark | Episodes | Score |
|---|---|---|
| MolmoSpaces, all nine benchmarks | 7,915 | **85.6 All Combined** (mean of the nine oracle success rates) |
| RoboLab, 120 tasks × 10 episodes (benchmark version 1.1) | 1,200 | **89.2% success** (1,070/1,200), 90.4% mean progress |

RoboLab recordings, videos and logs for every episode are at
[vasinwr/robolab-sol200-flagshipv3b-120x10](https://huggingface.co/datasets/vasinwr/robolab-sol200-flagshipv3b-120x10)
(RoboLab revision `ad45d4f`, official environments 0–9, native task scoring). That run applies a runtime gripper
configuration override (driven-joint armature, effort limit and pad contact stiffness and damping), described on
the dataset card; take it into account when comparing with other RoboLab results.

MolmoSpaces by benchmark, from the entry at
[allenai/molmospaces#243](https://github.com/allenai/molmospaces/issues/243); trajectories, camera videos and logs
for every episode are at
[TabTabTab/molmospace-silos-0](https://huggingface.co/datasets/TabTabTab/molmospace-silos-0):

| Benchmark | Oracle successes | Rate |
|---|---|---|
| Open-v1 | 715/1000 | 71.5% |
| Close-v1 | 891/915 | 97.4% |
| Pick-v1.5 | 802/1000 | 80.2% |
| Pick-v2 (classic) | 860/1000 | 86.0% |
| Pick-v2 (Filament column) | 860/1000 | 86.0%, the classic-renderer run counted again |
| Pick-v2 randomized camera | 751/1000 | 75.1% |
| PnP-v2 | 916/1000 | 91.6% |
| PnP-next-to-v2 | 909/1000 | 90.9% |
| PnP-color-v2 | 921/1000 | 92.1% |
| **All Combined** | | **85.6** |

Every MolmoSpaces benchmark ran with the classic renderer, so the Filament column repeats the classic Pick-v2 run
rather than a separate pass. A 100-episode subset, MolmoSpaces Hundred, scores 86/100 (All Combined estimate 86.8);
see [`molmospaces/`](molmospaces/README.md#result).

You supply the LLM API endpoint and key when you start a policy server. No credentials, private endpoints or
cloud setup are included in this repository.

Before committing, run `python scripts/check_public_files.py` to check that no credentials or private
references are in the tracked files.

## License

Apache-2.0; see [LICENSE](LICENSE).
