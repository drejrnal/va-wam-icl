# Robot demonstration experiment protocol

This protocol measures reset-time demonstration conditioning without changing
the held-out task targets, rollout seeds, or evaluation budget between arms. It
does not create simulator outcomes: the only reported values are recorded
RoboTwin rollouts.

## Implementation scope

This is robot-only post-training. `next-forcing-base` is a Wan2.2-derived
causal video/action base; editing the code does not create a trained
checkpoint, and it needs robot paired post-training before demonstration
conditioning is useful. The frozen WanVAE encodes synchronized robot video;
each demonstration is reduced to at most 17 latent frames. Patch features and
spatiotemporal positions are projected to 512-dimensional context during both
training and reset-cache construction. Zero-initialized demonstration attention
is added to the main and MCP blocks. The target video, action, and MCP loss
weights remain unchanged, with no alignment loss; training uses demonstration
dropout `0.2` and text dropout `0.1`. The reset-time K/V cache stays fixed for
the rollout. A is the original backbone trained on the same filtered data; B
is the modified checkpoint without context; C is that same modified checkpoint
with context.

## Episode registry and split

Author the versioned JSON object from the experiment's task registry, then use
`script/prepare_robot_demonstrations.py` to create each referenced tensor
payload:

```json
{
  "schema_version": 1,
  "episodes": [
    {
      "demo_id": "globally-unique-client-selector",
      "repo_id": "../datasets/robotwin/task",
      "episode_id": "source-episode-id",
      "episode_index": 12,
      "task": "pick_cube",
      "family": "pick-and-place",
      "goal": "place the blue cube in the bowl",
      "embodiment": "robotwin-bimanual",
      "success": true,
      "split": "unassigned",
      "cameras": {
        "observation.images.cam_high": "episodes/12/high.mp4",
        "observation.images.cam_left_wrist": "episodes/12/left.mp4",
        "observation.images.cam_right_wrist": "episodes/12/right.mp4"
      },
      "prepared_tensor_path": "prepared/12.pt",
      "source_seed": 17
    }
  ]
}
```

`task`, `family`, `goal`, and `embodiment` are supplied by the experimenter;
the tools never infer or invent task families. `demo_id` is globally unique,
and `prepared_tensor_path` and every camera path are manifest-relative. Only
successful records can become targets or supports.
`repo_id` is an absolute LeRobot root or a path relative to the manifest, such
as `../datasets/robotwin/task` when `NEXT_FORCING_DATASET_PATH` is its parent.
Physical episode uniqueness is the resolved repository path plus
`episode_index`. `source_seed` is non-negative and may be reused across
distinct source datasets/tasks.

Prepare a full synchronized camera-video demonstration with the frozen WanVAE:

```bash
export NEXT_FORCING_PRETRAINED_MODEL_PATH=$PWD/checkpoints/next-forcing-base
uv run script/prepare_robot_demonstrations.py \
  --camera-video observation.images.cam_high=manifests/episodes/12/high.mp4 \
  --camera-video observation.images.cam_left_wrist=manifests/episodes/12/left.mp4 \
  --camera-video observation.images.cam_right_wrist=manifests/episodes/12/right.mp4 \
  --vae-path "$NEXT_FORCING_PRETRAINED_MODEL_PATH/vae" \
  --layout robotwin_tshape --output manifests/prepared/12.pt
```

For pre-encoded camera tensors, use `--camera-latent CAMERA=PATH` for each
camera and explicitly pass `--camera-latents-normalized`; that flag confirms
the tensors already use Wan latent normalization.

Split before preparing a training view or selecting evaluation supports:

```bash
uv run script/demo_experiment.py split \
  --registry manifests/robotwin_episodes.json \
  --output manifests/robotwin_episodes_split.json \
  --split-seed robot-demo-v1
```

Keep the split output in the same manifest directory as its input; the command
rejects another directory so manifest-relative repository, camera, and prepared
tensor paths cannot be silently rebased.

The command ranks explicit family names with the supplied seed and assigns the
ranked families to 70% train, 10% validation, and 20% test buckets. It rejects
registries with fewer than 10 families, ensuring every bucket is nonempty, and
writes `robotwin_episodes_split.split-summary.json` beside the output with the
achieved family and episode counts. A family is assigned once, so close variants
never straddle a split. The planner rejects a registry whose family appears in
multiple splits.

## Post-training arms

Set the original base checkpoint and dataset root first. The dataset root must
contain `empty_emb.pt`; the manifest-relative repository paths above resolve
under that root.

```bash
export NEXT_FORCING_PRETRAINED_MODEL_PATH=$PWD/checkpoints/next-forcing-base
export NEXT_FORCING_DATASET_PATH=$PWD/datasets
test -f "$NEXT_FORCING_DATASET_PATH/empty_emb.pt"
```

Run three independent training seeds with the same split, optimization budget,
and original base checkpoint. Save each original-model checkpoint as arm A.
Run the same three seeds with demonstration conditioning enabled; each resulting
modified checkpoint supplies both B and C. Do not adapt on test episodes,
including their demonstrations; at test time supports are reset-time context
only.

```bash
NGPU=8 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --seed 41 --save-root runs/original-seed-41 \
  --demonstration-manifest-path manifests/robotwin_episodes_split.json \
  --demonstration-split train
NGPU=8 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --seed 42 --save-root runs/original-seed-42 \
  --demonstration-manifest-path manifests/robotwin_episodes_split.json \
  --demonstration-split train
NGPU=8 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --seed 43 --save-root runs/original-seed-43 \
  --demonstration-manifest-path manifests/robotwin_episodes_split.json \
  --demonstration-split train
```

Repeat the same templates with `CONFIG_NAME=robotwin_demo_train` and
`runs/demo-seed-{41,42,43}`. The paired comparison is:

| Condition | Checkpoint | Reset context |
| --- | --- | --- |
| A `a_original` | Original-model run | none |
| B `b_none` | Modified checkpoint | none |
| C `c_correct` | Same modified checkpoint as B | one selected video from a fixed three-video correct pool |
| `wrong_task` | Same modified checkpoint as B/C | one selected video from a fixed three-video wrong-task pool |
| `temporal_shuffled` | Same modified checkpoint as B/C | C's selected video with content shuffled and positions retained |

For a newly saved trainer checkpoint, stage the original base model's `vae`,
`tokenizer`, and `text_encoder` directories next to its saved `transformer`
directory before starting the server. `Trainer.save_checkpoint` saves only the
transformer; a server path aimed directly at that incomplete directory cannot
load the companion components. For example:

```bash
RUN=$PWD/runs/demo-seed-41/checkpoints/checkpoint_step_N
BASE=$PWD/checkpoints/next-forcing-base
ln -s "$BASE/vae" "$RUN/vae"
ln -s "$BASE/tokenizer" "$RUN/tokenizer"
ln -s "$BASE/text_encoder" "$RUN/text_encoder"
```

Use an empty destination directory or remove no existing links; `ln -s` fails
if an output already exists.

## Deterministic evaluation manifest

Generate the experiment manifest from the held-out split. The default rollout
seeds are `100` through `199`. Each selected support carries its user-provided
simulator `source_seed`; the command rejects a support pool whose source seed
overlaps any rollout seed, fewer than three exact task/family/goal/embodiment
supports, or fewer than three wrong-task supports. It writes 100 paired
rollouts for each eligible task/goal and condition. The three supports are a
deterministic candidate pool; a target registry record may also be a support
because evaluation rolls out a distinct simulator seed rather than replaying
that recorded episode. Every individual rollout selects and sends one `demo_id`
from the pool. The plan's `coverage.excluded_task_goals` lists every task/goal
that could not meet the support or seed-isolation contract.

```bash
export ROBOTWIN_ROOT=/path/to/your/RoboTwin
export ROBOTWIN_RANDOM_TASK_CONFIG=/path/to/your/randomized-task-config.yml
uv run script/demo_experiment.py plan \
  --registry manifests/robotwin_episodes_split.json \
  --output experiments/demo-test-plan.json \
  --a-checkpoint "$PWD/runs/original-seed-41/checkpoints/checkpoint_step_N" \
  --modified-checkpoint "$PWD/runs/demo-seed-41/checkpoints/checkpoint_step_N" \
  --evaluation-split test \
  --training-seed 41 \
  --clean-task-config "$ROBOTWIN_ROOT/task_config/demo_clean.yml" \
  --random-task-config "$ROBOTWIN_RANDOM_TASK_CONFIG"
```

Set `ROBOTWIN_RANDOM_TASK_CONFIG` to the installed RoboTwin randomized task
configuration. The plan records absolute config paths, SHA-256 content digests,
and complete `domain_randomization` mappings for Clean and Random; a file whose
randomization sentinels disagree with its label is rejected.

Each plan row is a complete assignment: target episode, fixed prompt, condition,
one selected support `demo_id`, the three-support provenance pool, and the exact
rollout seed. Arm B sends no demo id. C and `wrong_task` send their selected
demo id; `temporal_shuffled` additionally sends the selected support's source
seed as `demo_shuffle_seed`.

```bash
export NEXT_FORCING_ROOT=$PWD
export ROBOTWIN_ROOT=/path/to/your/RoboTwin
python -m evaluation.robotwin.eval_policy_client_openpi \
  --config "$ROBOTWIN_ROOT/policy/ACT/deploy_policy.yml" \
  --experiment-plan "$NEXT_FORCING_ROOT/experiments/demo-test-plan.json" \
  --trial-id pick_cube:100:c_correct \
  --experiment-setting Clean \
  --result-jsonl "$NEXT_FORCING_ROOT/experiments/observed_rollouts.jsonl" \
  --save_root "$NEXT_FORCING_ROOT/experiments/client-artifacts" \
  --overrides --task_name pick_cube --task_config demo_clean \
  --train_config_name 0 --model_name 0 --ckpt_setting 0 \
  --seed 0 --policy_name ACT --video_guidance_scale 5 \
  --action_guidance_scale 1 --port 29056
```

Start a server for the selected arm. The following command is for B, C, and
the two modified-checkpoint controls; set `NEXT_FORCING_MODEL_PATH` to the
corresponding `original-seed-41` checkpoint for A.

```bash
NEXT_FORCING_MODEL_PATH=$NEXT_FORCING_ROOT/runs/demo-seed-41/checkpoints/checkpoint_step_N \
NEXT_FORCING_DEMONSTRATION_MANIFEST=$NEXT_FORCING_ROOT/manifests/robotwin_episodes_split.json \
NEXT_FORCING_DEMONSTRATION_SPLIT=test \
NEXT_FORCING_DEMONSTRATION_EMBODIMENT=robotwin-bimanual \
bash evaluation/robotwin/launch_server.sh
```

Experiment-plan mode passes the row's exact seed without remapping or fallback,
uses its fixed prompt for every condition, and appends the observed JSONL row.
`--experiment-setting` is checked against the selected task configuration's
randomization fields: use a Clean task config for `Clean` and a configuration
with RoboTwin randomization enabled for `Random`; a mismatch fails before a
rollout. Plan mode also verifies the server checkpoint against the row's
`expected_checkpoint` before reset. The reset binds selected support identity,
source seed, registry SHA, and prepared-payload SHA-256; the server returns
independently computed provenance for aggregation to compare. The client sends
`{reset: true, prompt, demo_id?, demo_shuffle_seed?, expected_demo_provenance?}`.
The server resolves only manifest records; it does not accept arbitrary client
file paths.

Before collecting a condition's outcomes, check the reset result captured in
the JSONL: A must report the original checkpoint and null demonstration
provenance; B must report the modified checkpoint and null provenance; C must
report the same modified checkpoint plus the row's exact demonstration
provenance. The client rejects a reset that differs from the planned checkpoint
or support, so this check binds the target environment to the arm assignment.

## Recorded-results aggregation

Write one JSON object per observed rollout. `success` must be a boolean;
latency and peak memory are optional measured values rather than estimates:

```json
{"trial_id":"target:100:a_original","setting":"Clean","success":true,"actual_rollout_seed":100,"actual_demo_id":null,"actual_demo_provenance":null,"actual_checkpoint":"/checkpoints/original","actual_task_config":{"path":"/RoboTwin/task_config/demo_clean.yml","sha256":"...","domain_randomization":{"cluttered_table":false}},"fixed_prompt":"place cube","latency_ms":42.1,"peak_memory_mb":17321}
```

Aggregate only those rows:

```bash
uv run script/demo_experiment.py aggregate \
  --plan experiments/demo-test-plan.json \
  --results experiments/observed_rollouts.jsonl \
  --output experiments/demo-report.json \
  --bootstrap-samples 2000
```

The report carries every held-out task/goal, the support-eligible subset, and
every exclusion with its reason. Its primary estimand is equal-weight task
success and C(correct) minus B(none) among **support-eligible held-out
task/goals only**; it does not claim the full held-out population when an
exclusion exists. It keeps Clean and Random separate, reports per-task and
equal-weight task-mean success, and makes C(correct) minus B(none) the primary
comparison.
Its bootstrap resamples within each task before equal-weighting fixed evaluated
tasks. Every comparison reports expected, observed, and missing pair counts;
missing outcomes are never synthesized. A-versus-condition comparisons remain
diagnostic controls.
