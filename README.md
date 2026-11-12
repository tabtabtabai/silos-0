# silos-0

Code to run the **silos-0** policy (an API-hosted LLM planner driving analytical motion
tools, with no VLA or model checkpoint) on public robot-manipulation benchmarks. Each benchmark has its own
self-contained directory with the policy server, the simulator client and run instructions.

| Directory | Benchmark | Status |
|---|---|---|
| [`molmospaces/`](molmospaces/) | [MolmoSpaces](https://github.com/allenai/molmospaces) (Ai2, MuJoCo, DROID Franka) | 86/100 on MolmoSpaces Hundred, All Combined estimate 86.8 |
| `robolab/` | [RoboLab](https://github.com/NVlabs/RoboLab) (NVIDIA, Isaac Lab) | not yet moved here |

You supply the LLM API endpoint and key when you start a policy server. No credentials, private endpoints or
cloud setup are included in this repository.

Before committing, run `python scripts/check_public_files.py` to check that no credentials or private
references are in the tracked files.

## License

Apache-2.0; see [LICENSE](LICENSE).
