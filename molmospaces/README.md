# BatBatBat on MolmoSpaces

The **sol200-flagshipv3b** policy run on Ai2's [MolmoSpaces](https://github.com/allenai/molmospaces) benchmarks
(MuJoCo, DROID Franka FR3 + Robotiq 2F-85). The policy is an API-hosted LLM planner (`gpt-6.1-sol`, high
effort, at most 200 planner calls per episode) driving analytical motion tools: two-view stereo localization,
inverse kinematics and gripper control. It has no VLA and no model checkpoint. The policy prompt and tools are
the ones used for the RoboLab submission, unchanged.

This directory has three parts:

1. **Policy server** (`src/batbatbat`, `batbatbat-serve`): receives observations over a WebSocket and returns
   8-step joint-position chunks. It calls the LLM API you give it.
2. **MolmoSpaces client** (`client/`): the policy class, policy config and eval config MolmoSpaces asks an
   external policy to provide. It sends each observation to the server and plays back its chunks. This is the
   whole integration: MolmoSpaces' own `eval_main.py` runs the episodes and its own judge scores them.
3. **MolmoSpaces Hundred** (`benchmark/molmospaces-hundred.json`, `scripts/`): our own quick check, not part of
   MolmoSpaces' protocol. 100 fixed episodes chosen to track the leaderboard's All Combined score, with scripts that
   call `eval_main.py` once per episode in parallel and score with MolmoSpaces' `eval_to_csv.py`. A run takes
   hours instead of the days a full sweep takes.

For a leaderboard number, run MolmoSpaces' own command for each full benchmark instead (see
[Leaderboard runs](#leaderboard-runs)).

## Result

One run of the 100 episodes on a Mac (MuJoCo CPU, classic renderer):

| Benchmark | Success |
|---|---|
| Close-v1 | 13/13 |
| Pick-v2-classic (Objaverse) | 12/13 |
| PnP-color-v2 | 11/12 |
| PnP-next-to-v2 | 11/12 |
| Pick-v1.5 | 11/13 |
| Pick-v2-rand-cam | 10/12 |
| PnP-v2 | 10/12 |
| Open-v1 | 8/13 |
| **All 100** | **86/100; All Combined estimate 86.8** |

For scale, the best [leaderboard](https://molmospaces.allen.ai/leaderboard) entry with results on all nine
benchmarks scores 54.1 All Combined. One run of 100 episodes is about ±7 points at 95%. Three of the 14 failures
were not attempts: the model's first reply was a refusal, and the server ends an episode on a refusal. The other 11
ran out of robot time, mostly on Open. API models are nondeterministic, so repeated runs will vary.

## LLM API access

You supply the LLM endpoint and key when you start the server. The endpoint must serve the OpenAI Responses
protocol (`POST /responses`) with vision inputs, function tools, high reasoning effort and the model
**`gpt-6.1-sol`**. Chat completions alone is not enough. A different model is a different policy, not a
reproduction of this result.

There are two endpoints, and only the server needs the LLM credential:

- **Policy server:** runs locally from this directory, normally `ws://127.0.0.1:8001`. The MolmoSpaces client talks to it.
- **LLM API:** the base URL and key you pass to `batbatbat-serve`. The server appends `/responses` to the base URL.

No credentials, endpoints, cloud setup or telemetry are included. Decision logs are written locally to
`outputs/decisions.jsonl`. They contain task text and model responses, so review them before sharing.

## 1. Start the policy server

Python 3.12 or 3.13. No GPU needed.

```bash
cd molmospaces
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

# Keep the key in a file outside the repository (or in BATBATBAT_LLM_API_KEY).
batbatbat-serve \
  --llm-base-url https://<your-endpoint>/v1 \
  --llm-api-key-file ~/.config/batbatbat/llm-key \
  --port 8001
```

| Option | Default | Meaning |
|---|---|---|
| `--llm-base-url` | `$BATBATBAT_LLM_BASE_URL` | API base including its version prefix. HTTPS, no credentials in the URL. |
| `--llm-api-key-file` | `$BATBATBAT_LLM_API_KEY` | File holding the API key. |
| `--llm-api-key` | | The key itself. It is visible in the process list, so prefer the file or the environment. |
| `--llm-model` | `gpt-6.1-sol` | A deployment alias, only when it serves the same model. |
| `--host`, `--port` | `127.0.0.1`, `8001` | WebSocket bind address. |
| `--log` | `outputs/decisions.jsonl` | Local decision log. |

You can also put the values in `.env` (copy `.env.example`) and run `set -a; source .env; set +a` first.

On startup the server makes one small LLM request with the full tool list to check access before it opens the
port. That request is billed. The WebSocket has no authentication or TLS: bind it to loopback or a private
network, and use an SSH tunnel if the simulator runs on another machine.

Docker alternative:

```bash
docker build -t batbatbat-molmospaces .
docker run --rm -e BATBATBAT_LLM_BASE_URL -e BATBATBAT_LLM_API_KEY -p 127.0.0.1:8001:8001 batbatbat-molmospaces
```

Check the server with a synthetic request (blank images, one real LLM call, no simulator):

```bash
batbatbat-probe --url ws://127.0.0.1:8001
```

## 2. Install MolmoSpaces

Use MolmoSpaces **v0.2.9**, the version the leaderboard names, in its own environment (Python 3.11), following
[its README](https://github.com/allenai/molmospaces/tree/v0.2.9). Then download the benchmark assets and add the
client's dependencies:

```bash
git clone --branch v0.2.9 https://github.com/allenai/molmospaces "$MOLMOSPACES_SRC"
# ... install MolmoSpaces into its own venv as its README says ...
"$MOLMOSPACES_PYTHON" -m pip install 'mujoco~=3.5.0' 'mujoco-warp~=3.5.0'   # see below
"$MOLMOSPACES_PYTHON" -m pip uninstall -y mujoco-core                       # if present
export MLSPACES_ASSETS_DIR=/path/to/molmospaces-assets
"$MOLMOSPACES_PYTHON" -m molmo_spaces.molmo_spaces_constants   # downloads benchmarks/ and assets (about 20 GB)
"$MOLMOSPACES_PYTHON" -m pip install 'websockets>=13' msgpack h5py
```

**Pin MuJoCo to 3.5.** MolmoSpaces v0.2.9 asks for `mujoco~=3.5.0` but leaves `mujoco-warp` unpinned, so a fresh
install pulls `mujoco-warp` 4.x, which drags in MuJoCo 4 files next to 3.5. Every episode then fails at setup with
`AttributeError: ... no attribute 'mjDSBL_MULTICCD'`. The two pins above restore the 3.5 set the leaderboard uses;
check with `python -c "import mujoco, mujoco_warp; print(mujoco.__version__)"`, which should print `3.5.0`.

On a headless Linux machine, set `MUJOCO_GL=egl` (install `libegl1 libegl-mesa0`); no GPU is needed.

On Apple Silicon `toppra` does not build. Only MolmoSpaces' data generation uses it, so install the other
dependencies and then the package with `--no-deps`.

## 3. Run the 100 episodes

From this directory, with the server running:

```bash
export MOLMOSPACES_SRC=/path/to/molmospaces MOLMOSPACES_PYTHON=/path/to/molmospaces/.venv/bin/python
export MLSPACES_ASSETS_DIR=/path/to/molmospaces-assets RUN_OUT=$PWD/runs/first
# Optional, if the server is not on ws://127.0.0.1:8001:
# export HARNESS_URL=ws://127.0.0.1:8001

python scripts/jobs.py --assets "$MLSPACES_ASSETS_DIR" | xargs -P 8 -L 1 scripts/run_slot.sh
"$MOLMOSPACES_PYTHON" scripts/score.py --run-out "$RUN_OUT" --json "$RUN_OUT/results.json"
```

Each slot writes `$RUN_OUT/slot-NNN/` with MolmoSpaces' own output: trajectories, videos and `eval.log`. A
finished slot leaves a `DONE` marker, so running the same command again resumes and skips finished slots.
`--slots 1-13` or `--slots 3,7,40-45` picks a subset. One episode takes a few minutes of wall time, most of it
waiting on the LLM, and about 10 planner calls. 100 episodes took about 2.5 hours with 6–10 running in parallel.
Every planner call is billed by your endpoint.

`score.py` gets each episode's success from MolmoSpaces' `scripts/benchmarks/eval_to_csv.py` with
`--success-condition both`, as MolmoSpaces' benchmark docs run it. That records two numbers per episode: **oracle**
(the judge said success at any step), which the leaderboard ranks and which is the number reported here, and
**at-end** (success on the last step), printed in brackets. The client stops an episode at its first success, so
the two rarely differ. `score.py` writes that script's per-category CSV for each benchmark to `$RUN_OUT/scores/`,
and adds only the per-benchmark counts, the missing slots and the All Combined average.

It also records what produced the run, at the top of `$RUN_OUT`:

| File | Contents |
|---|---|
| `policy.json` | the policy server's startup record: policy name, model, reasoning effort, call budget, prompt hash, full config and its hash, source revision. One entry per distinct server configuration if the server changed mid-run, with a warning. |
| `environment.json` | the MolmoSpaces commit and tag, the `molmo-spaces`, `mujoco`, `mujoco-warp`, `warp-lang` and `numpy` versions, the renderer and asset data versions the episodes logged, Python, platform and `MUJOCO_GL` |

and splits the server's decision log into each slot's folder, so `$RUN_OUT/slot-NNN/` holds everything about one
episode:

| File | From | Contents |
|---|---|---|
| `decisions.jsonl` | policy server | first, the startup record of the server that ran the episode; then per planner call: reasoning summary (`thought`), visible text, latency, token usage; per tool call: arguments and result; the robot-time budget as it runs down |
| `eval/.../trajectories_*.h5` | MolmoSpaces | per step: joint state, actions, judge success |
| `eval/.../episode_*_<camera>.mp4` | MolmoSpaces | one video per policy camera |
| `eval/.../running_log.log`, `experiment_config_*.pkl` | MolmoSpaces | the episode's log and full eval config |
| `eval.log` | `run_slot.sh` | `eval_main.py`'s console output |

The reasoning is the summary the API returns, not the model's full hidden reasoning, and the exact frames the
model was shown are not stored (the videos show the same cameras). The decision logs contain task text and model
responses: review them before sharing.

## Leaderboard runs

A leaderboard entry uses MolmoSpaces' full benchmarks (about 1,000 episodes each) and its own commands, listed
in its `docs/ms-bench.md` and `docs/mb-bench.md`, with this client as the eval config. For one benchmark:

```bash
cd "$MOLMOSPACES_SRC"
export PYTHONPATH="/path/to/batbatbat/molmospaces/src:/path/to/batbatbat/molmospaces"
HARNESS_CAMS=exo_camera_1,wrist_camera,wrist_camera \
  python molmo_spaces/evaluation/eval_main.py client.policy:HarnessEvalConfig \
  --benchmark_dir "$MLSPACES_ASSETS_DIR/benchmarks/<benchmark folder>" --output_dir <run dir>
python scripts/benchmarks/eval_to_csv.py <run dir>/<date folder> sol200-flagshipv3b --success-condition both \
  --output-csv <task>/sol200-flagshipv3b.csv
```

Add the other arguments from MolmoSpaces' command for that benchmark. The client chooses its cameras from
`HARNESS_CAMS` (`left,right,wrist`), not from `--camera_names`. Use the cameras listed for that benchmark in
`benchmark/molmospaces-hundred.json`: for Pick-v2-rand-cam that is the randomized room camera and the wrist
camera, the same restriction Ai2's command sets with `--camera_names`. Five of the nine leaderboard benchmarks use
the Filament renderer (`--use-filament`), which needs Linux x86; the Hundred uses classic everywhere.

## Submitting

Results go to Ai2 as a GitHub issue from their
[Benchmark Entry template](https://github.com/allenai/molmospaces/issues/new?template=benchmark-entry-template.md),
as [allenai/molmospaces#8](https://github.com/allenai/molmospaces/issues/8) describes. A MolmoSpaces Hundred
result is not an entry: it is a subset, and entries report full benchmarks (for example 517/1000, or 744/915 on
Close-v1).

**Attach** the `eval_to_csv.py` CSV for each benchmark (`--success-condition both`), as files or one zip.
Accepted entries state the MolmoSpaces version, the benchmark set and the success condition in the issue, and give
a per-benchmark table of oracle successes. Reviewers have asked for evaluation logs before listing an entry, so
keep each benchmark's whole output folder (trajectories, videos, `running_log.log`, experiment configs) and the
server's `decisions.jsonl`, and be ready to share a link to them.

**State in the issue** (`policy.json` and `environment.json` from the run hold the values):

- MolmoSpaces v0.2.9 (`7351388`), MuJoCo 3.5.0 with `mujoco-warp` 3.5, benchmark data `molmospaces-bench-v1`
  20260408 and `molmospaces-bench-v2` 20260415.
- The policy's model, effort, call budget, prompt hash and source revision.
- The renderer for each benchmark. If the Filament benchmarks were run with the classic renderer, say so: Ai2's
  commands use Filament there, and a maintainer said in
  [allenai/molmospaces#158](https://github.com/allenai/molmospaces/issues/158) that renderer use is not checked.
- The policy reads only RGB, camera calibration, joint and gripper state and the task text, never MolmoSpaces'
  privileged observation fields, and the run used no benchmark success signal during execution.
- The setup-only loader fix in `client/fixes.py` (equivalent to allenai/molmospaces#225).

**Questionnaire draft** (the template's fields):

```text
policy name: sol200-flagshipv3b
authors/institution name: <authors> / <institution>
MolmoBot or other in-domain data used: No (no training; the policy is an API LLM with a fixed prompt and tools)
multi-task: Yes (one unchanged configuration for every benchmark)
data type (real/sim/mix): N/A (no training data)
robot generalist: Yes (the same prompt and tools ran on RoboLab)
data opensourced: N/A
model opensourced: No (the planner is the API model gpt-6.1-sol); the harness code is <link if published>
source link: <link>
model type, e.g. vla, wam, bm, vlm: VLM planner with analytical tools (stereo localization, IK); no VLA
embodiment: Franka (DROID: FR3 + Robotiq 2F-85)
action space: Joint position
no. of params: N/A (API model)
data released: N/A
chunk size (if applicable): 8 (15 Hz control)
context length (if applicable): whole episode in one conversation, up to 200 planner calls
```

## How the client works

- **Only robot-obtainable inputs.** The client sends RGB from three cameras, their calibration
  (`sensor_param_*` plus the arm's root frame), joint and gripper state, and the task text. MolmoSpaces also
  gives every policy object poses, masks, `task_info` and success. None of these are read, and using them would
  disqualify a run.
- **Cameras.** Each slot names its `[left, right, wrist]` cameras. The server locates objects by triangulating
  between left and right. Benchmarks with one outside camera use the wrist camera as the second view, and the
  request tells the model so. Pick-v2-rand-cam uses only the randomized room camera and the wrist camera, as Ai2's
  own command for it does.
- **Robot.** `batbatbat-serve` sets the MolmoSpaces robot: fingertips 0.141 m from the gripper mount (MuJoCo
  2F-85) and the FR3's joint limits intersected with the Panda model's. The planner prompt is the same as on
  RoboLab; these values only change motion.
- **Actions.** The server returns 8 steps of 7 joint targets plus a binary gripper. The client maps the gripper to
  MolmoSpaces' 0–255 scale. Control runs at MolmoSpaces' 66 ms step, and each episode has its own time limit
  (`task_horizon_sec`, 20–40 s of robot time) and ends at first success.
- **Loader fix.** MolmoSpaces v0.2.9 cannot look up articulated objects built into the house, so about half of
  the Open/Close episodes fail to load (allenai/molmospaces#218). `client/fixes.py` retries the lookup with the
  builtin `droid` grasp library, as that function's own comment prescribes. Upstream fixed this after v0.2.9
  (allenai/molmospaces#225). It touches episode setup only, never scoring.

## MolmoSpaces Hundred

Ai2's headline score, **All Combined**, averages nine benchmarks of about 1,000 episodes each, scored by oracle
success (success at any step). A full sweep is about 9,000 episodes. MolmoSpaces Hundred keeps MolmoSpaces' own
episode specs, simulator, judge and time limits at v0.2.9, and fixes which 100 episodes run, so runs compare slot
by slot.

| | |
|---|---|
| Shape | 8 benchmarks × 12–13 episodes = 100 slots |
| Benchmarks | Open-v1, Close-v1, Pick-v1.5, Pick-v2-classic, Pick-v2-rand-cam, PnP-v2, PnP-next-to-v2, PnP-color-v2 |
| Not run | Pick-v2-filament: its renderer runs only on Linux x86. It is scored from the Pick-v2-classic slots. |
| Renderer | classic everywhere |
| Score | MolmoSpaces' `eval_to_csv.py` per episode (oracle, from `--success-condition both`); All Combined = mean of the nine benchmark rates (`scripts/score.py`) |

**How the 100 were chosen.** The leaderboard publishes each policy's oracle rate per object category, so a
subset's expected score for a policy is the mean of its category rates over the subset's episodes. Open, Close and
Pick-v1.5 have 9–18 categories and are stratified by category. The Objaverse sets have about one episode per
category. For each benchmark we drew 1,500 candidate subsets and kept the one with the smallest RMS gap over half
of the policies that have results on all nine benchmarks, then checked it on the other half.

All Combined, full leaderboard against this subset's expectation (oracle %):

| Policy | Full | Hundred | |
|---|---|---|---|
| Cosmic | 54.1 | 53.0 | held out |
| ZoeInside | 49.5 | 50.2 | fit |
| Phoenix | 48.0 | 48.4 | fit |
| Cosmos3-Nano | 39.0 | 36.4 | held out |
| WALL-OSS | 38.5 | 39.4 | fit |
| PRTS | 27.8 | 27.7 | fit |
| MolmoAct | 25.9 | 26.5 | fit |
| Psi-R2 | 22.8 | 20.4 | held out |
| π0.5 | 17.8 | 18.9 | held out |

The held-out mean gap is 1.8 points and the worst is 2.6. The sampling noise of 100 episodes is larger (about ±10
points at 95% around 60%), so compare runs slot by slot.

## Tests

```bash
pip install -e '.[test]'
python -m pytest -q
```

The tests run on CPU with a synthetic local API, so they need no key and cost nothing. They check the rendered
prompt hash, the request settings, the WebSocket exchange with the server started through `--llm-base-url` and
`--llm-api-key-file`, the client's request and calibration, triangulation from request calibration, the robot
options and the slot list. They do not run MolmoSpaces or reproduce the score.

The rendered planner prompt's SHA-256, the same as the RoboLab submission's:

```text
f8cf317b38bf94ac786a06a213b051a68e94f50d2199c758ea909e6e5d041229
```

## License

Apache-2.0; see [LICENSE](../LICENSE). MolmoSpaces, its assets and the hosted LLM remain under their own licenses
and access terms and are not redistributed here.
