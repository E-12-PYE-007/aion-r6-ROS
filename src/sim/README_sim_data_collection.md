# Sim Data Collection Pipeline Notes

This document summarizes the simulation data collection work added on the `sim-data-collection` branch. The goal of this branch is to support Isaac-generated simulation scenes, automatically generate valid language/task specs, generate expert action chunks with obstacle-aware planning, track those chunks with pure pursuit, and collect stream-style datasets for action-head training.

## High-Level Pipeline

The intended pipeline is:

```text
Isaac base scene YAML
  -> Isaac generated layout YAML with exact obstacles/clutter
  -> Aion task spec generator
  -> semantic task validation
  -> optional Hybrid A* planner validation/filtering
  -> Isaac rollout
  -> stream-style sim dataset folder
```

At runtime, the ROS data/control flow is:

```text
Isaac Sim
  publishes camera + sim_odom

expert trajectory node
  subscribes sim_odom
  resolves task into a reference path
  optionally plans with Hybrid A*
  publishes /vla/action_chunk
  optionally publishes /expert/cmd_vel for debugging/reference

sim_waypoint_tracking
  subscribes sim_odom + /vla/action_chunk
  tracks the action chunk with PurePursuitController
  publishes cmd_vel

Isaac Sim
  consumes cmd_vel

sim_dataset_collector
  records camera, odom, cmd_vel, action_chunk, language, task metadata, variant metadata
```

The main driving command for Isaac should be `cmd_vel` from `sim_waypoint_tracking`, not `/expert/cmd_vel`. The `/expert/cmd_vel` topic is useful for debugging the expert profile, but the pure pursuit tracker is the control path intended for collection.

## Rollout Orchestration

`src/sim/sim/collect_task_spec_rollouts.py`

This script is the automation layer for collecting a batch of task/variant rollouts from a validated task spec. It reads the task spec, selects every requested `task_id` and `variant_id`, launches the matching expert trajectory node, launches the pure-pursuit sim waypoint tracker, launches `sim_dataset_collector`, waits for the configured collection duration, then stops the nodes and moves to the next combination.

The basic dry-run command is:

```bash
ros2 run sim collect_task_spec_rollouts \
  src/sim/config/generated_task_specs/fence_gap_01_seed43_roverstart_right_base_task_spec.yaml \
  --task-id follow_fence_01_left_from_scene_rover_pose \
  --variant-id nominal \
  --dry-run
```

The basic collection command is:

```bash
ros2 run sim collect_task_spec_rollouts \
  src/sim/config/generated_task_specs/fence_gap_01_seed43_roverstart_right_base_task_spec.yaml \
  --task-id follow_fence_01_left_from_scene_rover_pose \
  --variant-id nominal
```

By default it uses the task spec's `collection.duration_s`, `collection.base_dir`, camera topic, odom topic, command topic, and action chunk topic. Each rollout folder gets a deterministic trajectory name containing the suite, task, and variant. The collector records the selected task metadata, selected variant metadata, planner settings, speed profile, language instruction, odom, `cmd_vel`, and `/vla/action_chunk`.

Useful options:

```text
--dry-run
  Print the commands without starting ROS nodes.

--task-id <id>
  Collect only a specific task. Can be repeated.

--variant-id <id>
  Collect only a specific variant. Can be repeated.

--limit N
  Collect only the first N selected rollouts.

--duration-s N
  Override the spec's collection duration.

--base-dir <path>
  Override the output dataset directory.

--no-tracker
  Start the expert and collector without the pure-pursuit tracker.

--prepare-scene-command "<command>"
  Optional hook to reload/reset Isaac before each rollout. The command can use placeholders:
  {scene_yaml}, {generated_layout_yaml}, {generated_usd}, {task_spec}, {task_id}, {variant_id}, {trajectory_name}.

--use-isaac-bridge
  Use the file-based Isaac rollout bridge before each rollout. This writes a request to the bridge, waits for Isaac to load/reset the requested scene, then waits for the camera and odom topics before starting the tracker/expert/collector.
```

Important limitation: Isaac still needs to be running the bridge process. The rollout script can request a scene through `--use-isaac-bridge`, but it does not start Isaac itself.

### Isaac rollout bridge

The bridge is split across the two repos:

- Isaac side: `scripts/isaac_rollout_bridge.py`
- ROS side: `src/sim/sim/prepare_isaac_rollout.py`

Start the Isaac-side bridge first from the Isaac repo, using Isaac Sim's Python. Replace `$ISAAC_SIM_PYTHON` with the workstation's Isaac Python launcher, for example the `python.sh` inside the Isaac Sim installation:

```bash
cd ~/isaac_files

$ISAAC_SIM_PYTHON scripts/isaac_rollout_bridge.py \
  --headless \
  --isaac-root ~/isaac_files \
  --command-file /tmp/isaac_rollout_request.json \
  --status-file /tmp/isaac_rollout_status.json
```

Then run collection with bridge preparation enabled:

```bash
cd ~/aion-r6-ROS
source install/setup.bash

ros2 run sim collect_task_spec_rollouts \
  src/sim/config/generated_task_specs/fence_gap_01_seed43_roverstart_right_base_task_spec.yaml \
  --use-isaac-bridge \
  --isaac-root ~/isaac_files \
  --task-id follow_fence_01_left_from_scene_rover_pose \
  --variant-id nominal \
  --limit 1
```

For each selected task/variant, the ROS prepare command writes a JSON request containing the task spec, task id, variant id, generated USD path, and layout YAML path. The Isaac bridge opens the requested USD, applies the layout YAML's `rover_pose` to `/World/RoverSystem` in memory, starts simulation playback, and writes a ready/error status JSON. The ROS prepare command then waits for `/sim_odom` and `/vla/cam` before the rollout nodes start.

Pose-perturbed recovery variants still require a layout YAML/USD generated for the perturbed start pose. The bridge applies whatever `rover_pose` is present in the requested layout YAML; it does not invent recovery poses by itself.

## Main Files Added Or Changed

### Task generation

`src/sim/sim/generate_scene_task_specs.py`

Generates task spec YAMLs from Isaac scene YAMLs. It supports Isaac base configs and generated layout YAMLs. It infers `config_type` when needed:

- `fenceline` if the YAML contains `fences`
- `road` if the YAML contains `roads`
- `shedline` if the YAML contains `shed`

It writes specs containing:

- scene metadata
- collection defaults
- expert defaults
- language instruction(s)
- structured task fields
- success conditions
- validation requirements
- trajectory variants

Current generated task families include:

- `follow_fence`
- `follow_and_turn`
- `follow_fence_sequence`
- `follow_corridor`
- `pass_through_gap`
- `stop_at_gap`
- `switch_sides`
- `stop_at_landmark`
- `follow_road`
- `approach_target`
- `follow_shed_side`
- `hold_position`

Explicit obstacle-language tasks are intentionally disabled for now. Generated layout YAMLs can still contain obstacles, and Hybrid A* still plans around them, but the task language does not currently say things like "go around the boulder" or "pass the log on your left." This avoids generating instructions that the expert policy does not semantically guarantee yet.

The generator was updated to be geometry-aware instead of applying every task to every start pose. It now:

- skips fence tasks when the start pose is too far from the relevant fence segment
- computes `left`/`right` from actual segment geometry instead of only guessing from the start name
- generates connected multi-segment fence tasks for loops, perimeters, and U-shaped fences
- generates corridor-centerline tasks when two parallel fence lines form a corridor
- improves gate/gap wording for inside/outside starts
- still deduplicates task IDs before writing the final spec

### Task validation

`src/sim/sim/validate_scene_task_specs.py`

Validates generated task specs before collection. It has two levels:

1. Semantic/geometry validation.
2. Optional planner validation with `--check-planner`.

Semantic validation checks things like:

- required task fields exist
- `start_pose` exists in the scene
- target fences/roads exist
- start pose is close enough to the target segment
- requested turn direction matches geometry
- connected segment tasks are actually connected
- gap tasks refer to real collinear gaps
- switch-side gaps are wide enough
- corridor fences are parallel
- expert support exists for the task type

Planner validation checks each trajectory variant using Hybrid A*. Variants that fail planning can be filtered out when writing collection-ready specs. If all variants for a task fail, that task becomes invalid.

Useful validation flags:

```bash
ros2 run sim validate_scene_task_specs <spec_or_dir>
ros2 run sim validate_scene_task_specs <spec_or_dir> --check-planner
ros2 run sim validate_scene_task_specs <spec_or_dir> --write-valid-output-dir <output_dir>
ros2 run sim validate_scene_task_specs <spec_or_dir> --allow-invalid --verbose
```

### Expert trajectory node base

`src/sim/sim/expert_policy_node.py`

Shared ROS node base for task-driven expert policies. It:

- loads a task spec YAML
- selects a `task_id`
- selects a `variant_id`
- loads the source Isaac scene YAML
- resolves the semantic task into a reference path through a subclass
- optionally runs Hybrid A* from the live robot pose to the task goal
- smooths/resamples the path
- builds a velocity/time profile
- publishes timed future poses as `/vla/action_chunk`
- publishes optional `/expert/cmd_vel`

Important parameters:

```text
task_spec
task_id
variant_id
odom_topic
action_chunk_topic
expert_cmd_vel_topic
waypoint_spacing_m
future_time_offsets_s
publish_rate_hz
flip_isaac_y
use_hybrid_astar
robot_radius_m
obstacle_padding_m
grid_resolution_m
yaw_resolution_deg
step_size_m
min_turn_radius_m
goal_tolerance_m
hybrid_astar_max_iterations
allow_reverse
max_speed_mps
max_yaw_rate_radps
max_accel_mps2
max_decel_mps2
max_angular_accel_radps2
min_profile_speed_mps
stop_at_end
```

Variant-specific settings override node defaults. Older task specs that do not define variants still work with `variant_id:=nominal`.

### Fenceline expert policy

`src/sim/sim/fenceline_expert_trajectory.py`

Resolves fenceline task specs into geometric reference paths. It supports:

- single-fence following
- two-segment turn following
- multi-segment connected fence sequences
- corridor centerline following
- gap pass-through
- stop at gap
- switch sides through a gap
- stop at fence landmark
- hold position

For fence-following tasks, it offsets the path from the fence by `preferred_offset_m`, which can come from the selected trajectory variant.

For corridor tasks, it creates a centerline between the two parallel fences.

### Road expert policy

`src/sim/sim/road_expert_trajectory.py`

Resolves road task specs into reference paths. It supports:

- road following
- road turn following
- approach target point
- stop at road landmark
- hold position

Road paths use the road segments directly rather than fence offsets.

### Shed expert policy

`src/sim/sim/shed_expert_trajectory.py`

Resolves shed task specs into reference paths. It supports:

- following a shed side
- approaching the shed
- stopping beside the shed
- holding position

It uses shed bounding box dimensions from the Isaac asset metadata to build a path at a safe clearance around the shed side. The selected trajectory variant can override `preferred_offset_m`.

## Hybrid A* Planner

`src/sim/sim/hybrid_astar.py`

Hybrid A* is used for obstacle-aware expert path generation. The geometric task resolver still defines the semantic intent and rough target path, and Hybrid A* creates a physically plausible path while respecting turning limits and collision constraints.

The planner now follows reference subgoals sampled along the semantic path instead of planning only to the final goal. This is important for fence/road/shed following: it prevents the planner from taking a valid but semantically wrong shortcut across a rectangle, corridor, or perimeter scene.

This is important because Isaac generated layouts can include obstacles such as plants, logs, and boulders. The planner helps route around them instead of blindly following a straight offset path.

Key planner settings:

```text
grid_resolution_m
yaw_resolution_deg
step_size_m
min_turn_radius_m
goal_tolerance_m
hybrid_astar_max_iterations
allow_reverse
```

Current defaults are conservative:

```text
grid_resolution_m: 0.25
yaw_resolution_deg: 15.0
step_size_m: 0.35
min_turn_radius_m: 0.75
goal_tolerance_m: 0.35
planner_subgoal_spacing_m: 2.0
hybrid_astar_max_iterations: 20000
allow_reverse: false
```

If Hybrid A* fails at runtime, the expert node falls back to the geometric reference path and logs a warning. During offline validation with `--check-planner`, variants that fail planning can be filtered before collection.

## Obstacle Handling

The current approach is:

```text
task language = route/goal instruction
planner = obstacle avoidance
```

For example, the instruction can remain:

```text
Follow the fence.
Drive through the gate.
Follow the road.
Drive between the fences.
```

If the generated Isaac layout contains plants, logs, or boulders, Hybrid A* uses the collision map to route around them while still completing the route/goal task.

Explicit obstacle-language tasks are not generated at this stage because they require stronger semantic guarantees. For example, "pass the boulder on your left" requires the expert to identify that exact obstacle and enforce a pass side. That support can be added later, but it is not needed for initial data collection where the desired behavior is general safe navigation around clutter.

## Collision Map

`src/sim/sim/collision_map.py`

Builds collision geometry from Isaac generated layout YAMLs. It reads obstacles from the scene and uses asset bounding boxes to create inflated obstacle boxes.

The collision map accounts for:

- obstacle position
- obstacle yaw
- asset bounding box size
- robot radius
- obstacle padding
- optional Isaac Y-axis flipping

Default limits:

```text
robot_radius_m: 0.35
obstacle_padding_m: 0.25
```

Trajectory variants can adjust these values. For example, the cautious variant uses a larger obstacle padding.

## Smoothing And Velocity Profile

`src/sim/sim/trajectory_profile.py`

After path generation, the path is smoothed/resampled and converted into a timed trajectory. This is what makes the expert output more physically realistic than raw waypoints.

The profile enforces limits for:

- maximum linear speed
- maximum yaw rate
- maximum acceleration
- maximum deceleration
- maximum angular acceleration
- minimum crawl speed
- stop at end behavior

Current default speed profile:

```text
max_speed_mps: 0.35
max_yaw_rate_radps: 0.45
max_accel_mps2: 0.25
max_decel_mps2: 0.35
max_angular_accel_radps2: 0.6
min_profile_speed_mps: 0.03
stop_at_end: true
```

The expert node samples this timed trajectory at future time offsets and publishes those relative future poses as an `ActionChunk`.

Default future offsets:

```text
[0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
```

## Action Chunk Publishing

The expert policy builds `/vla/action_chunk` messages from the timed trajectory. Each action chunk contains future relative poses in the robot frame.

The generated future poses represent where the robot should be at future time offsets, not simply evenly spaced distances. This is better aligned with action-head training because speed, yaw rate, acceleration, and stopping behavior are reflected in the action target.

## Pure Pursuit Tracking

`src/hw_interface/src/sim_waypoint_tracking.cpp`

The sim waypoint tracker was wired to use:

```cpp
hw_interface::PurePursuitController
```

instead of the previous feedforward-style action chunk tracking. It subscribes to:

```text
sim_odom
/vla/action_chunk
```

and publishes:

```text
cmd_vel
```

This node is the intended low-level controller in the sim collection pipeline. The expert node publishes the action chunk; pure pursuit converts that action chunk into the command Isaac should execute.

## Stream-Style Dataset Collector

`src/sim/sim/sim_dataset_collector.py`

The sim collector was expanded to behave like the existing stream collector pattern. It records a continuous stream of image frames and synchronized metadata.

It saves:

```text
trajectory_dir/
  img/
    <timestamp>.jpg
  poses.jsonl
  metadata.json
```

Each JSONL record includes:

- image filename
- image timestamp
- robot pose
- robot velocity
- latest `cmd_vel`
- latest `/vla/action_chunk`
- language instruction
- dataset name
- task ID
- variant ID
- variant type
- recovery case
- planner settings
- speed profile

Important collector parameters:

```text
base_dir
dataset_name
trajectory_name
task_id
variant_id
variant_type
recovery_case
language_instruction
structured_task_json
planner_settings_json
speed_profile_json
camera_topic
odom_topic
cmd_vel_topic
action_chunk_topic
sample_frequency_hz
jpeg_quality
flip_isaac_y
```

The collector does not decide the task or drive the robot. It only records the stream and metadata needed for training and later debugging.

## Trajectory Variants

Every newly generated task spec now includes `trajectory_variants`. These represent multiple valid ways to collect the same semantic task.

Default variant families:

### Nominal

```text
variant_id: nominal
variant_type: nominal
preferred_offset_m: 0.8
```

Uses the default planner and speed profile.

### Cautious Wide Clearance

```text
variant_id: cautious_wide_clearance
variant_type: clearance
preferred_offset_m: 1.0
obstacle_padding_m: 0.35
min_turn_radius_m: 0.85
max_speed_mps: 0.25
max_yaw_rate_radps: 0.35
```

This produces slower, wider, more conservative behavior.

### Normal Tight Clearance

```text
variant_id: normal_tight_clearance
variant_type: clearance
preferred_offset_m: 0.65
obstacle_padding_m: 0.2
```

This keeps the robot closer to the target structure.

## Recovery Variants

Recovery variants create controlled off-reference starts for the same semantic task. These are intended to help collect data for cases where the VLA or robot has drifted away from the ideal path or temporarily lost the target.

Current recovery cases:

```text
recovery_left_offset
  recovery_case: lost_target_left
  start_pose_delta: x=0.0, y=0.8, yaw=0.35 rad

recovery_right_offset
  recovery_case: lost_target_right
  start_pose_delta: x=0.0, y=-0.8, yaw=-0.35 rad

recovery_wrong_heading
  recovery_case: wrong_heading
  start_pose_delta: x=0.0, y=0.0, yaw=0.85 rad
  max_speed_mps: 0.25
  max_yaw_rate_radps: 0.35
```

The `start_pose_delta` can be applied at the Isaac layout YAML level with `expand_pose_variants`. This avoids needing a runtime reset service in Isaac. The expert node still reads planner and speed settings from the selected variant.

Hold-position tasks only get the nominal variant.

## Pose Variant Expansion

`src/sim/sim/expand_pose_variants.py`

This script creates perturbed Isaac generated-layout YAMLs for recovery variants. It reads:

- a generated Isaac layout YAML containing `rover_pose`
- a generated task spec containing `trajectory_variants`
- one or more `task_id` values
- optional `variant_id` filters

It writes new Isaac-compatible layout YAMLs where only the rover start pose and metadata are changed. Obstacles, fences, roads, sheds, assets, and output directories are preserved from the source layout.

The delta is applied in the rover's local frame:

```text
world_dx = cos(yaw) * local_dx - sin(yaw) * local_dy
world_dy = sin(yaw) * local_dx + cos(yaw) * local_dy
world_yaw = yaw + delta_yaw
```

Example:

```bash
ros2 run sim expand_pose_variants \
  --layout-yaml /path/to/generated_layout.yaml \
  --task-spec /path/to/task_spec.yaml \
  --task-id follow_example_task \
  --variant-id recovery_left_offset \
  --output-dir /path/to/perturbed_layouts
```

If `--variant-id` is omitted, the script expands every variant for that task that has a nonzero `start_pose_delta`.

Each output YAML gets:

```yaml
pose_variant:
  source_layout_yaml: ...
  source_task_spec: ...
  task_id: ...
  variant_id: ...
  variant_type: ...
  recovery_case: ...
  start_pose_delta: ...
  base_rover_pose: ...
  perturbed_rover_pose: ...
```

Nominal testing does not require this step. Use it later when collecting recovery variants.

## Multiple Valid Trajectories Per Task

The task generator now creates a set of trajectory variants per task. The validator can then run Hybrid A* against each variant.

The intended workflow is:

```text
generate task specs
  -> each task has multiple trajectory_variants

validate with --check-planner
  -> variants that fail Hybrid A* can be removed

write collection-ready specs
  -> only valid task/variant pairs remain
```

This lets us collect multiple valid trajectories for one language instruction/task without hand-authoring each rollout.

## Rules And Limits Currently Encoded

### Task generation rules

- Only generate fenceline tasks for starts near the target fence.
- Use geometric side detection for `left` and `right`.
- Only generate turn tasks for connected segments that actually turn.
- Only generate gap tasks for collinear fence sections with a real gap.
- Only generate switch-side tasks when the gap is wide enough.
- Only generate sequence tasks for connected chains with at least three fences.
- Only generate corridor tasks when exactly two fences are parallel and separated.
- Gate wording is based on start labels such as `inside`, `outside`, and `gate`.

### Validation rules

- Start pose must exist and have a position.
- Target objects/segments must exist.
- Starts must be close enough to the relevant target.
- Follow side must match actual geometry.
- Turn direction must match actual geometry.
- Connected sequence fences must have matching endpoints.
- Corridor fences must be parallel.
- Gap center must match the detected gap.
- Planner validation checks Hybrid A* through reference subgoals and can filter impossible variants.

### Planner/vehicle limits

- Minimum turn radius defaults to `0.75 m`.
- Maximum speed defaults to `0.35 m/s`.
- Maximum yaw rate defaults to `0.45 rad/s`.
- Acceleration/deceleration and angular acceleration are limited in the timed profile.
- Reverse is disabled by default.
- Robot radius defaults to `0.35 m`.
- Obstacle padding defaults to `0.25 m`.

## New Task Types Added For New Isaac Fence Scenes

### `follow_fence_sequence`

Used for connected multi-fence chains such as:

- closed rectangles
- large perimeters
- U-shaped fences
- multi-turn connected fence paths

Important fields:

```yaml
task_type: follow_fence_sequence
target_fences:
  - south_fence
  - east_fence
  - north_fence
follow_side: left
sequence_type: perimeter
```

The fenceline expert resolves this by concatenating all target fences and offsetting the resulting path.

### `follow_corridor`

Used for two parallel fence lines forming a passage.

Important fields:

```yaml
task_type: follow_corridor
corridor_fences:
  - left_fence
  - right_fence
```

The fenceline expert resolves this as the centerline between the two fences.

### Gate/gap tasks

Existing gap task types are reused:

- `pass_through_gap`
- `stop_at_gap`
- `switch_sides`

For gate-like scenes, the generated language now includes enter/exit/pass-through wording when the start pose name indicates inside/outside/gate context.

## Deleted Old Generated Task Files

The old generated task specs were deleted from:

```text
src/sim/config/generated_task_specs
src/sim/config/scene_valid_task_specs
src/sim/config/collection_ready_task_specs
```

This was done so the next generated batch will be consistent with the new variant-aware task generator. No new permanent task specs were regenerated during this cleanup.

## Console Scripts Registered

`src/sim/setup.py` registers:

```text
sim_dataset_collector
fenceline_action_chunk_publisher
fenceline_expert_trajectory
road_expert_trajectory
shed_expert_trajectory
generate_scene_task_specs
validate_scene_task_specs
expand_pose_variants
generate_collection_manifest
run_collection_manifest
validate_collected_rollout
prepare_isaac_rollout
collect_task_spec_rollouts
```

The older `fenceline_action_chunk_publisher` remains in the repo, but the task-spec-driven expert trajectory nodes are the intended path for the new sim data generation pipeline.

## Typical Commands

Generate task specs:

```bash
ros2 run sim generate_scene_task_specs <isaac_yaml_or_dir> --output-dir src/sim/config/generated_task_specs --summary
```

Validate scene semantics:

```bash
ros2 run sim validate_scene_task_specs src/sim/config/generated_task_specs --write-valid-output-dir src/sim/config/scene_valid_task_specs
```

Validate planner support and write collection-ready specs:

```bash
ros2 run sim validate_scene_task_specs src/sim/config/scene_valid_task_specs --check-planner --write-valid-output-dir src/sim/config/collection_ready_task_specs
```

Generate a collection manifest:

```bash
ros2 run sim generate_collection_manifest \
  src/sim/config/collection_ready_task_specs \
  --output src/sim/config/collection_manifest.yaml \
  --isaac-root ~/isaac_files \
  --summary
```

Include existing sky/ground USD variations as separate rollout rows:

```bash
ros2 run sim generate_collection_manifest \
  src/sim/config/collection_ready_task_specs \
  --output src/sim/config/collection_manifest.yaml \
  --isaac-root ~/isaac_files \
  --include-visual-variations \
  --summary
```

If recovery pose variants have been expanded with `expand_pose_variants`, pass their layout directory so the manifest can point those rows at the perturbed layout/USD files:

```bash
ros2 run sim generate_collection_manifest \
  src/sim/config/collection_ready_task_specs \
  --output src/sim/config/collection_manifest.yaml \
  --isaac-root ~/isaac_files \
  --pose-variant-layout-dir ~/isaac_files/configs/generated_layouts/pose_variants \
  --summary
```

Dry-run the first pending manifest row:

```bash
ros2 run sim run_collection_manifest \
  src/sim/config/collection_manifest.yaml \
  --limit 1 \
  --use-isaac-bridge \
  --isaac-root ~/isaac_files \
  --dry-run
```

Collect the first pending manifest row:

```bash
ros2 run sim run_collection_manifest \
  src/sim/config/collection_manifest.yaml \
  --limit 1 \
  --use-isaac-bridge \
  --isaac-root ~/isaac_files
```

The manifest runner selects rows with `status: pending` and `pose_variant_ready: true`. It writes each selected row to `running`, creates a temporary per-rollout task spec with the row's `layout_yaml` and `visual_usd`, runs the Isaac bridge plus tracker/expert/collector, then updates the row to `complete` or `failed`. Use `--retry-failed` to retry failed rows.

After a real rollout, the manifest runner validates the collected folder before marking the row complete. It checks for `metadata.json`, `poses.jsonl`, saved JPEGs, image references, minimum sample count, action chunks, `cmd_vel`, task/variant metadata, and required motion. Use `--skip-validation` only for debugging.

Validate one collected rollout folder manually:

```bash
ros2 run sim validate_collected_rollout \
  ~/sim_datasets/generated/<trajectory_name> \
  --expected-task-id follow_example_task \
  --expected-variant-id nominal
```

### Manual parallel worker test

Only run this after the single-worker test is passing. Each worker needs its own Isaac process and ROS domain. The manifest runner uses a lock file beside the manifest so workers claim pending rows one at a time.

Worker 00 Isaac terminal:

```bash
cd ~/isaac_files
export ROS_DOMAIN_ID=31

$ISAAC_SIM_PYTHON scripts/isaac_rollout_bridge.py \
  --headless \
  --isaac-root ~/isaac_files \
  --command-file /tmp/isaac_rollout_worker_00_request.json \
  --status-file /tmp/isaac_rollout_worker_00_status.json
```

Worker 00 ROS terminal:

```bash
cd ~/aion-r6-ROS
source install/setup.bash
export ROS_DOMAIN_ID=31

ros2 run sim run_collection_manifest \
  src/sim/config/collection_manifest.yaml \
  --worker-id worker_00 \
  --limit 1 \
  --use-isaac-bridge \
  --isaac-root ~/isaac_files
```

Worker 01 Isaac terminal:

```bash
cd ~/isaac_files
export ROS_DOMAIN_ID=32

$ISAAC_SIM_PYTHON scripts/isaac_rollout_bridge.py \
  --headless \
  --isaac-root ~/isaac_files \
  --command-file /tmp/isaac_rollout_worker_01_request.json \
  --status-file /tmp/isaac_rollout_worker_01_status.json
```

Worker 01 ROS terminal:

```bash
cd ~/aion-r6-ROS
source install/setup.bash
export ROS_DOMAIN_ID=32

ros2 run sim run_collection_manifest \
  src/sim/config/collection_manifest.yaml \
  --worker-id worker_01 \
  --limit 1 \
  --use-isaac-bridge \
  --isaac-root ~/isaac_files
```

When `--worker-id` is set, the runner automatically uses worker-specific defaults:

```text
bridge request: /tmp/isaac_rollout_<worker_id>_request.json
bridge status:  /tmp/isaac_rollout_<worker_id>_status.json
logs:           logs/sim_rollouts/<worker_id>
runtime specs:  logs/sim_rollouts/<worker_id>/runtime_task_specs
dataset dir:    <collection.base_dir>/<worker_id>
```

The worker claims one pending row at a time by marking it `running` with `worker_id`, releases the manifest lock while Isaac collects, then reacquires the lock to write `complete` or `failed`.

Run a fenceline expert:

```bash
ros2 run sim fenceline_expert_trajectory --ros-args \
  -p task_spec:=/path/to/task_spec.yaml \
  -p task_id:=follow_example_task \
  -p variant_id:=nominal
```

Run pure pursuit tracker:

```bash
ros2 run hw_interface sim_waypoint_tracking
```

Run sim collector:

```bash
ros2 run sim sim_dataset_collector --ros-args \
  -p base_dir:=/path/to/sim_dataset_output \
  -p task_id:=follow_example_task \
  -p variant_id:=nominal \
  -p cmd_vel_topic:=cmd_vel \
  -p action_chunk_topic:=/vla/action_chunk
```

## Recommended First Test

Before collecting a large dataset, run one tiny end-to-end test:

1. Generate one Isaac layout from a known-good base config.
2. Generate task specs for that one layout.
3. Validate the task spec normally.
4. Validate it with `--check-planner`.
5. Run one expert policy with `variant_id:=nominal`.
6. Run `sim_waypoint_tracking`.
7. Run the collector.
8. Inspect the output folder manually:
   - images are present
   - `poses.jsonl` has records
   - `metadata.json` has the correct task/variant fields
   - `cmd_vel` is present
   - `action_chunk` is present

Once one rollout works, scale up by sampling tasks and variants from the collection-ready specs.

## Notes For Teammates

- This branch is about generating simulation data for action-head training.
- Robot state was not added as an action-head input.
- The model target is still future action chunks/poses, but the expert pipeline can now condition those targets on task semantics, obstacles, and physical limits.
- Existing generated task specs without `trajectory_variants` still run in nominal fallback mode, but new generated specs should be preferred.
- The Isaac repo branch `miah_isaac` contains additional base fenceline scenes. Those are scene templates, not generated layouts.
- The Aion repo branch `sim-data-collection` contains the task generation, validation, expert planning, pure pursuit tracking, and data collection pipeline.
