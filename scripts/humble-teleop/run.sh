#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
launch_file="$repo_dir/src/bringup/launch/teleop_launch.py"

if [[ $# -eq 0 ]]; then
    if [[ -f "$launch_file" ]]; then
        set -- ros2 launch /launch/teleop_launch.py max_linear_speed:=0.2 max_yaw_rate:=0.2
    else
        echo 'Teleop launch file is absent; opening a Humble shell instead.' >&2
        set -- bash
    fi
fi

mount_args=()
if [[ -f "$launch_file" ]]; then
    mount_args+=(--mount "type=bind,src=$launch_file,dst=/launch/teleop_launch.py,readonly")
fi
if [[ -d /dev/input ]]; then
    mount_args+=(--device-cgroup-rule 'c 13:* rmw'
        --mount type=bind,src=/dev/input,dst=/dev/input,readonly)
fi
if [[ -d /run/udev ]]; then
    mount_args+=(--mount type=bind,src=/run/udev,dst=/run/udev,readonly)
fi

# Host networking lets ROS discover the Jetson. Source Humble explicitly
# inside the container; do not inherit the host's Jazzy workspace setup.
exec sudo docker run --rm -it \
    --network host \
    --ipc host \
    --mount "type=bind,src=$repo_dir,dst=/workspace/aion-r6-ROS" \
    --workdir /workspace/aion-r6-ROS \
    "${mount_args[@]}" \
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" \
    -e ROS_LOCALHOST_ONLY=0 \
    --entrypoint /bin/bash \
    aion-humble-teleop -c \
    'source /opt/ros/humble/setup.bash; exec "$@"' -- "$@"
