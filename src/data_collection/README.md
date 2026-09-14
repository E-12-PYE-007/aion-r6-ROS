# data_collection

Nodes for logging teleop trajectories (image + pose + velocity) for VLA training.

## Nodes

- **`stream_data_collector`** — logs every frame continuously for the node's whole lifetime. No start/stop control. Kept for reference; prefer `episode_data_collector` for real runs.
- **`episode_data_collector`** — same logging, but gated into named episodes via services instead of node lifetime. Idle (no subscriptions dropped, no files written) until told to record.
- **`episode_recorder`** — keyboard client for the above. Run it alongside teleop.

## Running

`bringup` has a launch file for teleop + data collection: `joy_node` → `teleop_twist_joy` → `cmd_vel_to_roboclaw` → `roboclaw_for_motors`, plus `episode_data_collector` logging alongside. It does **not** start the camera driver or localisation/EKF chain — run those separately until they're merged into `main`. It also does not start `episode_recorder` — see below for why.

```
ros2 launch bringup teleop_data_collection_launch.py base_dir:=/path/to/trajectories
```

Launch args (all optional except `base_dir`):

| arg | default | notes |
|---|---|---|
| `base_dir` | *(required)* | where episodes get written |
| `cam_topic` | `/camera/color/image_raw` | |
| `odom_topic` | `/odometry/filtered` | assumes `local_localisation_launch.py` (single EKF, no GPS). Pass `/odometry/global` if using the GPS-fused setup instead |
| `max_linear_speed` | `0.3` (m/s) | |
| `max_yaw_rate` | `0.3` (rad/s) | |

Controller: Xbox One (`teleop_twist_joy`'s `xbox.config.yaml` layout — left stick drives, left trigger is the deadman, turbo disabled). Verify axis/button numbers with `ros2 topic echo /joy` on first connect if using Bluetooth, since enumeration can differ from wired.

In a **separate terminal**, once the above is running:

```
ros2 run data_collection episode_recorder
```

It's kept out of the launch file deliberately — it needs a real attached terminal for raw keypress capture and `input()` prompts, which `ros2 launch` doesn't reliably provide to spawned processes (same reason `teleop_twist_keyboard` is normally run by hand).

In `episode_recorder`: `x` start (prompts for a name), `s` stop, `q` quit (stops first if recording).

Suggested name format: `<target>_<follow_side>_turn_<turns>`. Not enforced — just a hint at the prompt.

## Service interface (`episode_data_collector`)

- `~/start_episode` (`aion_msgs/srv/StartEpisode`, req: `name`) — creates `<base_dir>/<name>/`. Refuses if already recording, name has invalid chars (`[A-Za-z0-9_-]+` only), or the dir already exists. Empty name → timestamp. `episode_recorder` handles the "already exists" case by retrying with an incrementing suffix (`_02`, `_03`, ...); calling the service directly, you're on your own for that.
- `~/stop_episode` (`std_srvs/srv/Trigger`) — refuses if nothing's recording.

## Output layout

```
<base_dir>/<episode_name>/
├── img/
│   └── <img_time_ms>.jpg
└── poses.jsonl
```

Each `poses.jsonl` line: `{episode, image, img_time, pose, velocity}`.
`pose = (t, x, y, yaw)`, `velocity = (linear_x, angular_z)`, both from the most recent odometry message.

Image filenames are only unique within an episode — join on `(episode, image)` if pooling files across episodes.

## Topics

`episode_data_collector` takes these as ROS params (`cam_topic`, `odom_topic`), settable via the launch args above or `--ros-args -p`. `stream_data_collector` has them hardcoded to the same defaults.

- `/camera/color/image_raw` (`sensor_msgs/Image`) — Orbbec Gemini 330 driver, `camera_name:=camera` (see `visual_odom` branch, not yet merged)
- `/odometry/filtered` (`nav_msgs/Odometry`) — `robot_localization` EKF output. This is the single-EKF, no-GPS topic name (`local_localisation_launch.py`); the GPS-fused setup (`global_localisation_launch.py`) publishes `/odometry/local` and `/odometry/global` instead (see `robot-localization-ekf` branch, not yet merged)
