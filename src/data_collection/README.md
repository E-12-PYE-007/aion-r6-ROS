# data_collection

Nodes for logging teleop trajectories (image + pose + velocity) for VLA training.

## Nodes

- **`stream_data_collector`** — logs every frame continuously for the node's whole lifetime. No start/stop control. Kept for reference; prefer `episode_data_collector` for real runs.
- **`episode_data_collector`** — same logging, but gated into named episodes via services instead of node lifetime. Idle (no subscriptions dropped, no files written) until told to record.
- **`collection_interface`** — one-terminal operator console for the above. Arrow keys publish `/cmd_vel`; `x`, `s`, and `q` manage recording episodes.

## Running

This is a Jetson-side setup. SSH into the rover, start the robot/data-collection launch in one terminal, then run the combined drive/record console in another attached SSH terminal.

**Jetson terminal 1** — `cmd_vel_to_roboclaw`, the RGB-only camera stream (`camera_launch.py`), `local_localisation_launch.py` (which owns `roboclaw_for_motors`, mavros, wheel-encoder localisation, and the EKF), plus `episode_data_collector` logging alongside. Uses `local_localisation_launch.py` (no GPS) hardcoded, not `global_localisation_launch.py` — swap the include in the launch file itself if you need the GPS-fused setup, there's no launch arg for it yet.

```
ros2 launch bringup teleop_data_collection_launch.py base_dir:=/path/to/trajectories
```

| arg | default | notes |
|---|---|---|
| `base_dir` | *(required)* | where episodes get written |
| `cam_topic` | `/camera/color/image_raw` | |
| `odom_topic` | `/odometry/filtered` | assumes `local_localisation_launch.py` (single EKF, no GPS). Pass the global EKF output if using the GPS-fused setup instead |

**Jetson terminal 2** — combined arrow-key teleop and episode orchestration:

```
ros2 run data_collection collection_interface
```

It is kept out of the launch file deliberately — it needs a real attached terminal for raw keypress capture and `input()` prompts. It also needs to run on the same machine as `episode_data_collector`: discarding an episode deletes its directory straight off local disk, so client and collector must share a filesystem.

`collection_interface` publishes plain `geometry_msgs/Twist` on `/cmd_vel`, using the same simple arrow-key velocity model as `key_teleop`:

| control | behavior |
|---|---|
| Up arrow | drive forward |
| Down arrow | drive backward |
| Left arrow | yaw left |
| Right arrow | yaw right |
| `x` | start an episode |
| `s` | stop the episode, then ask whether to save it |
| `q` | quit, stopping + asking to save first if recording |

Drive rate params:

| param | default | notes |
|---|---|---|
| `cmd_vel_topic` | `/cmd_vel` | topic to publish teleop commands on |
| `forward_rate` | `0.4` (m/s) | up-arrow speed |
| `backward_rate` | `0.4` (m/s) | down-arrow speed |
| `rotation_rate` | `0.4` (rad/s) | left/right-arrow yaw rate |
| `hz` | `10.0` | command publish rate |
| `key_timeout` | `0.5` (s) | command returns to zero when arrow key repeats stop |

Example with slower driving:

```
ros2 run data_collection collection_interface --ros-args -p forward_rate:=0.2 -p backward_rate:=0.2 -p rotation_rate:=0.2
```

When starting an episode, enter a compact descriptor. Descriptors must start with `fl` or `fr`:

| prefix | saved language prompt |
|---|---|
| `fl...` | `follow the fence on your left` |
| `fr...` | `follow the fence on your right` |

Extra descriptor letters can describe the run, for example `s` straight, `o` obstacle, `c` clear, `tr` turn right, or `tl` turn left, but only the leading `fl` or `fr` affects the language prompt. Example names: `fls`, `fro`, `fltr`, `frtl`.

For drive-only testing without episode control, `bringup` also provides a `key_teleop` wrapper:

```
ros2 launch bringup teleop_launch.py forward_rate:=0.4 backward_rate:=0.4 rotation_rate:=0.4
```

## Service interface (`episode_data_collector`)

- `~/start_episode` (`aion_msgs/srv/StartEpisode`, req: `name`, `prompt`) — creates `<base_dir>/<name>/`. Refuses if already recording, name has invalid chars (`[A-Za-z0-9_-]+` only), or the dir already exists. Empty name → timestamp. `collection_interface` handles the "already exists" case by retrying with an incrementing suffix (`_02`, `_03`, ...); calling the service directly, you're on your own for that.
- `~/stop_episode` (`std_srvs/srv/Trigger`) — refuses if nothing's recording.

## Output layout

```
<base_dir>/<episode_name>/
├── img/
│   └── <img_time_ms>.jpg
└── poses.jsonl
```

Each `poses.jsonl` line: `{episode, prompt, image, img_time, pose, velocity}`.
`pose = (t, x, y, yaw)`, `velocity = (linear_x, angular_z)`, both from the most recent odometry message.

Image filenames are only unique within an episode — join on `(episode, image)` if pooling files across episodes.

## Topics

`episode_data_collector` takes these as ROS params (`cam_topic`, `odom_topic`), settable via the launch args above or `--ros-args -p`. `stream_data_collector` has them hardcoded to the same defaults.

- `/camera/color/image_raw` (`sensor_msgs/Image`) — Orbbec Gemini 330 driver (`camera_launch.py`, `camera_name:=camera`), RGB only — depth/IR/point cloud/TF are disabled since nothing here needs them
- `/odometry/filtered` (`nav_msgs/Odometry`) — `robot_localization` EKF output, single-EKF/no-GPS (`local_localisation_launch.py`). The GPS-fused setup also exposes a global EKF output; pass that as `odom_topic` instead if using it.
- `/cmd_vel` (`geometry_msgs/Twist`) — published by `collection_interface`, consumed by `cmd_vel_to_roboclaw`.
