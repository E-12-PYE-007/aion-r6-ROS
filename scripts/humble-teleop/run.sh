#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"

if [[ ! -d /dev/input ]]; then
    echo 'No /dev/input directory found. Connect the controller first.' >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- ros2 launch /launch/teleop_launch.py max_linear_speed:=0.2 max_yaw_rate:=0.2
fi

# Host networking lets ROS discover the Jetson. Input access supports reconnects.
# Mount only the launch file; host Jazzy build/install directories stay separate.
exec sudo docker run --rm -it \
    --network host \
    --ipc host \
    --device-cgroup-rule 'c 13:* rmw' \
    --mount type=bind,src=/dev/input,dst=/dev/input,readonly \
    --mount type=bind,src=/run/udev,dst=/run/udev,readonly \
    --mount "type=bind,src=$repo_dir/src/bringup/launch/teleop_launch.py,dst=/launch/teleop_launch.py,readonly" \
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" \
    -e ROS_LOCALHOST_ONLY=0 \
    aion-humble-teleop "$@"
