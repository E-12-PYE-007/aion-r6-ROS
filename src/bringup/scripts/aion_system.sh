#!/usr/bin/env bash
# Start, stop and query the rover stack from a remote shell. The Endeavour display runs this over ssh.
#
#   aion_system.sh start [--no-nvblox] [launch args]
#                                        ros2 launch bringup system.launch.py [launch args], detached, then
#                                        the nvblox container (skipped with --no-nvblox or controller:=pure_pursuit)
#   aion_system.sh stop                  the stack first (SIGINT, then SIGTERM and SIGKILL if it does not exit),
#                                        then nvblox
#   aion_system.sh status                "running <pid>" or "stopped" (exit 0 or 3), then the nvblox state
#
# The stack runs in its own session, so it survives the ssh connection that started it. Its pid and logs
# are kept in ~/.local/state/aion-system. stop only signals the process group that start created, and
# only after checking that the pid file still points at it.
#
# nvblox lives in the Isaac ROS docker image. It is started with the same docker run as
# isaac_ros_common/scripts/run_dev.sh, detached, in a container named aion-nvblox (not the interactive dev
# container). If nvblox does not come up, start stops the stack again so nothing is half started.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${AION_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STATE_DIR="${AION_STATE_DIR:-$HOME/.local/state/aion-system}"
PID_FILE="$STATE_DIR/pid"
STOP_TIMEOUT="${AION_STOP_TIMEOUT:-20}"   # seconds to wait after SIGINT
TERM_TIMEOUT="${AION_TERM_TIMEOUT:-5}"    # seconds to wait after SIGTERM
KEEP_LOGS=10

ROS_DOMAIN="${ROS_DOMAIN_ID:-42}"
DISCOVERY_SERVER="${ROS_DISCOVERY_SERVER:-10.43.254.23:11811}"

NVBLOX_ENABLED="${AION_NVBLOX:-1}"
ISAAC_ROS_WS="${AION_ISAAC_ROS_WS:-/home/vla-cap/vla-cap-ws/isaac-ros-ws}"
NVBLOX_CONTAINER="${AION_NVBLOX_CONTAINER:-aion-nvblox}"
NVBLOX_IMAGE="${AION_NVBLOX_IMAGE:-isaac_ros_dev-aarch64}"
NVBLOX_LAUNCH="${AION_NVBLOX_LAUNCH:-ros2 launch aion_vslam nvblox_camera.launch.py}"
NVBLOX_SETTLE="${AION_NVBLOX_SETTLE:-3}"                      # seconds the container must stay up
DOCKER_STOP_TIMEOUT="${AION_DOCKER_STOP_TIMEOUT:-15}"          # seconds after SIGINT before docker kills it

# ---------------------------------------------------------------- the ROS stack

# When a process started, in clock ticks since boot (field 22 of /proc/<pid>/stat). Unlike the wall-clock
# time from ps, it cannot change when the system clock is adjusted.
proc_start_ticks() {
    local stat
    stat="$(< "/proc/$1/stat")" 2>/dev/null || return 1
    stat="${stat##*) }"      # drop "pid (comm) ", leaving the fields from the third on
    set -- $stat
    echo "${20}"
}

# Prints the pid of a live stack started by this script, or nothing. The pid file holds the pid and the
# process start time, so a recycled pid is not mistaken for the stack.
running_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid started
    { read -r pid; read -r started; } < "$PID_FILE"
    [[ "$pid" =~ ^[0-9]+$ && -n "$started" ]] || return 1
    [[ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$pid" ]] || return 1
    [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == "$(id -un)" ]] || return 1
    [[ "$(proc_start_ticks "$pid")" == "$started" ]] || return 1
    echo "$pid"
}

# Serialises start and stop, so two presses at once cannot launch two stacks. The stack is started
# without this descriptor, otherwise it would hold the lock for as long as it runs.
take_lock() {
    mkdir -p "$STATE_DIR"
    exec 9> "$STATE_DIR/lock"
    flock -w 40 9 || { echo "another start or stop is still in progress" >&2; exit 1; }
}

group_gone() { ! kill -0 -- "-$1" 2>/dev/null; }

wait_for_exit() {   # wait_for_exit <pgid> <seconds>
    local ticks=$(( $2 * 5 ))
    while (( ticks-- > 0 )); do
        group_gone "$1" && return 0
        sleep 0.2
    done
    group_gone "$1"
}

stack_start() {   # stack_start <launch args...>
    local pid
    if pid="$(running_pid)"; then
        echo "already running (pid $pid)"
        return 0
    fi
    rm -f "$PID_FILE"

    local launch
    if [[ -n "${AION_LAUNCH_CMD:-}" ]]; then
        launch="$AION_LAUNCH_CMD"   # used by the tests
    else
        if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
            echo "no built workspace at $WORKSPACE (install/setup.bash is missing)" >&2
            return 1
        fi
        launch="source /opt/ros/humble/setup.bash; source '$WORKSPACE/install/setup.bash'; "
        launch+="export ROS_DOMAIN_ID=$ROS_DOMAIN ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER ROS_SUPER_CLIENT=TRUE; "
        launch+="exec ros2 launch bringup system.launch.py $(printf '%q ' "$@")"
    fi

    local log="$STATE_DIR/system_$(date +%Y%m%d_%H%M%S).log"
    ln -sfn "$log" "$STATE_DIR/latest.log"
    # The inner shell becomes the session leader, records its own pid and start time, then runs the stack.
    # --fork, not a shell "&": a backgrounded command in a non-interactive shell starts with SIGINT ignored,
    # and the stack would inherit that, so stop could never interrupt it.
    PID_FILE="$PID_FILE" setsid --fork bash -c \
        'cmd=$1; s=$(< /proc/$$/stat); s=${s##*) }; set -- $s; { echo $$; echo ${20}; } > "$PID_FILE"; eval "$cmd"' \
        _ "$launch" 9>&- > "$log" 2>&1 < /dev/null

    local waited=0
    until pid="$(running_pid)"; do
        if (( waited++ >= 50 )); then
            echo "the stack did not start or exited at once, last log lines:" >&2
            tail -n 15 "$log" >&2
            return 1
        fi
        sleep 0.1
    done
    sleep 1
    if ! pid="$(running_pid)"; then
        echo "the stack exited right after starting, last log lines:" >&2
        tail -n 15 "$log" >&2
        return 1
    fi
    echo "started (pid $pid), log $log"
}

stack_stop() {
    local pid
    if ! pid="$(running_pid)"; then
        rm -f "$PID_FILE"
        echo "not running"
        return 0
    fi
    kill -INT -- "-$pid" 2>/dev/null
    if wait_for_exit "$pid" "$STOP_TIMEOUT"; then
        rm -f "$PID_FILE"; echo "stopped"; return 0
    fi
    echo "still running ${STOP_TIMEOUT}s after SIGINT, sending SIGTERM" >&2
    kill -TERM -- "-$pid" 2>/dev/null
    if wait_for_exit "$pid" "$TERM_TIMEOUT"; then
        rm -f "$PID_FILE"; echo "stopped (needed SIGTERM)"; return 0
    fi
    echo "still running ${TERM_TIMEOUT}s after SIGTERM, sending SIGKILL" >&2
    kill -KILL -- "-$pid" 2>/dev/null
    if wait_for_exit "$pid" 3; then
        rm -f "$PID_FILE"; echo "stopped (needed SIGKILL)"; return 0
    fi
    echo "could not stop pid $pid" >&2
    return 1
}

# ---------------------------------------------------------------- nvblox in docker

nvblox_managed() { [[ "$NVBLOX_ENABLED" == "1" ]] && command -v docker > /dev/null; }

nvblox_exists() { [[ -n "$(docker ps -aq --filter "name=^/${NVBLOX_CONTAINER}\$" 2>/dev/null)" ]]; }

nvblox_running() {
    [[ -n "$(docker ps -aq --filter "name=^/${NVBLOX_CONTAINER}\$" --filter status=running 2>/dev/null)" ]]
}

nvblox_start() {
    if [[ ! -d "$ISAAC_ROS_WS" ]]; then
        echo "no Isaac ROS workspace at $ISAAC_ROS_WS" >&2
        return 1
    fi
    # Only ever removes the container this script owns.
    docker rm -f "$NVBLOX_CONTAINER" > /dev/null 2>&1

    # The same flags and mounts as isaac_ros_common/scripts/run_dev.sh on aarch64, detached and with a command.
    local args=(
        run -d --init --name "$NVBLOX_CONTAINER"
        --privileged --network host --ipc=host --pid=host --runtime nvidia
        -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all,nvidia.com/pva=all
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e ROS_DOMAIN_ID="$ROS_DOMAIN" -e ROS_DISCOVERY_SERVER="$DISCOVERY_SERVER"
        -e USER="$(id -un)" -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev
        -e HOST_USER_UID="$(id -u)" -e HOST_USER_GID="$(id -g)"
        -v "$ISAAC_ROS_WS":/workspaces/isaac_ros-dev
        -v /etc/localtime:/etc/localtime:ro
        -v /tmp/:/tmp/
        -v /usr/bin/tegrastats:/usr/bin/tegrastats
        -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra
        -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api
        -v /usr/share/vpi3:/usr/share/vpi3
        -v /dev/input:/dev/input
        --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh
        --workdir /workspaces/isaac_ros-dev
        "$NVBLOX_IMAGE"
        bash -c "source /opt/ros/humble/setup.bash && source /workspaces/isaac_ros-dev/install/setup.bash && exec $NVBLOX_LAUNCH"
    )
    local out
    if ! out="$(docker "${args[@]}" 2>&1)"; then
        echo "docker could not start nvblox: $out" >&2
        return 1
    fi

    local waited=0
    until nvblox_running; do
        if (( waited++ >= 50 )); then
            echo "nvblox is not running, last log lines:" >&2
            docker logs --tail 15 "$NVBLOX_CONTAINER" >&2 2>&1
            docker rm -f "$NVBLOX_CONTAINER" > /dev/null 2>&1
            return 1
        fi
        sleep 0.1
    done
    sleep "$NVBLOX_SETTLE"
    if ! nvblox_running; then
        echo "nvblox exited right after starting, last log lines:" >&2
        docker logs --tail 15 "$NVBLOX_CONTAINER" >&2 2>&1
        docker rm -f "$NVBLOX_CONTAINER" > /dev/null 2>&1
        return 1
    fi
    echo "nvblox started (container $NVBLOX_CONTAINER)"
}

nvblox_stop() {
    nvblox_managed || return 0
    nvblox_exists || return 0
    mkdir -p "$STATE_DIR"
    docker logs --tail 3000 "$NVBLOX_CONTAINER" > "$STATE_DIR/nvblox_$(date +%Y%m%d_%H%M%S).log" 2>&1
    docker stop --signal SIGINT --time "$DOCKER_STOP_TIMEOUT" "$NVBLOX_CONTAINER" > /dev/null 2>&1
    docker rm -f "$NVBLOX_CONTAINER" > /dev/null 2>&1
    if nvblox_exists; then
        echo "could not stop the nvblox container $NVBLOX_CONTAINER" >&2
        return 1
    fi
    echo "nvblox stopped"
}

# ---------------------------------------------------------------- commands

prune_logs() {
    ls -t "$STATE_DIR"/system_*.log 2>/dev/null | tail -n +"$(( KEEP_LOGS + 1 ))" | while read -r old; do rm -f -- "$old"; done
    ls -t "$STATE_DIR"/nvblox_*.log 2>/dev/null | tail -n +"$(( KEEP_LOGS + 1 ))" | while read -r old; do rm -f -- "$old"; done
}

cmd_start() {
    take_lock
    local launch_args=() want_nvblox=1 arg
    for arg in "$@"; do
        case "$arg" in
            --no-nvblox) want_nvblox=0 ;;
            controller:=pure_pursuit) want_nvblox=0; launch_args+=("$arg") ;;   # only the CBF reads the nvblox map
            *) launch_args+=("$arg") ;;
        esac
    done
    nvblox_managed || want_nvblox=0

    if ! stack_start "${launch_args[@]}"; then
        return 1
    fi
    if (( want_nvblox )); then
        if ! nvblox_start; then
            echo "nvblox did not start, stopping the stack again" >&2
            stack_stop > /dev/null
            return 1
        fi
    fi
    prune_logs
}

cmd_stop() {
    take_lock
    trap '' HUP   # a dropped ssh connection must not cut the escalation short
    local rc=0
    stack_stop || rc=1      # the robot first, then the map
    nvblox_stop || rc=1
    return "$rc"
}

cmd_status() {
    local pid rc=3
    if pid="$(running_pid)"; then
        echo "running $pid"
        rc=0
    else
        echo "stopped"
    fi
    if nvblox_managed; then
        if nvblox_running; then echo "nvblox running"; else echo "nvblox stopped"; fi
    fi
    return "$rc"
}

case "${1:-}" in
    start) shift; cmd_start "$@" ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    *) echo "usage: $0 start [--no-nvblox] [launch args] | stop | status" >&2; exit 2 ;;
esac
