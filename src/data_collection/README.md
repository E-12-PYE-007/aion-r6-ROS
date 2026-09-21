# data_collection

Nodes for logging teleop trajectories (image + pose + velocity) for VLA training.

## Nodes

- **`stream_data_collector`** — logs every frame continuously for the node's whole lifetime. No start/stop control. Kept for reference; prefer `episode_data_collector` for real runs.
- **`episode_data_collector`** — same logging, but gated into named episodes via services instead of node lifetime. Idle (no subscriptions dropped, no files written) until told to record.
- **`collection_interface`** — keyboard client for the above. Run it alongside teleop.

## Running

This is a two-machine setup: the controller/joystick lives on the workstation, everything else (motor control, data collector) runs on the Jetson.

**Workstation** — `joy_node` → `teleop_twist_joy`, publishing `/cmd_vel` over the network:

```
ros2 launch bringup teleop_launch.py
```

| arg | default | notes |
|---|---|---|
| `max_linear_speed` | `0.3` (m/s) | |
| `max_yaw_rate` | `0.3` (rad/s) | |

Controller: Xbox One (`teleop_twist_joy`'s `xbox.config.yaml` layout — left stick drives, left trigger is the deadman, turbo disabled). Verify axis/button numbers with `ros2 topic echo /joy` on first connect if using Bluetooth, since enumeration can differ from wired.

**Jetson** — `cmd_vel_to_roboclaw` → `roboclaw_for_motors`, plus `episode_data_collector` logging alongside. Does **not** start the camera driver or localisation/EKF chain — run those separately until they're merged into `main`. Does not start `collection_interface` — see below for why.

```
ros2 launch bringup teleop_data_collection_launch.py base_dir:=/path/to/trajectories
```

| arg | default | notes |
|---|---|---|
| `base_dir` | *(required)* | where episodes get written |
| `cam_topic` | `/camera/color/image_raw` | |
| `odom_topic` | `/odometry/local` | assumes `local_localisation_launch.py` (single EKF, no GPS). Pass `/odometry/global` if using the GPS-fused setup instead |

In a **separate terminal on the Jetson** (e.g. its own SSH session from the workstation), once the above is running:

```
ros2 run data_collection collection_interface
```

It's kept out of the launch file deliberately — it needs a real attached terminal for raw keypress capture and `input()` prompts, which `ros2 launch` doesn't reliably provide to spawned processes (same reason `teleop_twist_keyboard` is normally run by hand). It also needs to run on the same machine as `episode_data_collector`: discarding an episode deletes its directory straight off local disk, so client and collector must share a filesystem.

In `collection_interface`: `x` start (prompts for a name, then a prompt describing the episode), `s` stop (then asks `Save episode? [y/n]`, deleting the episode directory on `n`), `q` quit (stops + asks to save first if recording).

Suggested name format: `<target>_<follow_side>_turn_<turns>`. Not enforced — just a hint at the prompt.

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

- `/camera/color/image_raw` (`sensor_msgs/Image`) — Orbbec Gemini 330 driver, `camera_name:=camera` (see `visual_odom` branch, not yet merged)
- `/odometry/local` (`nav_msgs/Odometry`) — `robot_localization` EKF output, single-EKF/no-GPS (`local_localisation_launch.py`). The GPS-fused setup (`global_localisation_launch.py`) also publishes `/odometry/global`; pass that as `odom_topic` instead if using it.
