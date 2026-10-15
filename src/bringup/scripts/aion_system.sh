#!/usr/bin/env bash
# Start, stop and query the rover stack from a remote shell. The Endeavour display runs this over ssh.
#
#   aion_system.sh start [launch args]   ros2 launch bringup system.launch.py [launch args], detached
#   aion_system.sh stop                  SIGINT the stack, then SIGTERM and SIGKILL if it does not exit
#   aion_system.sh status                prints "running <pid>" or "stopped" (exit 0 or 3)
#
# The stack runs in its own session, so it survives the ssh connection that started it. Its pid and logs
# are kept in ~/.local/state/aion-system. stop only signals the process group that start created, and
# only after checking that the pid file still points at it.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${AION_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STATE_DIR="${AION_STATE_DIR:-$HOME/.local/state/aion-system}"
PID_FILE="$STATE_DIR/pid"
STOP_TIMEOUT="${AION_STOP_TIMEOUT:-20}"   # seconds to wait after SIGINT
TERM_TIMEOUT="${AION_TERM_TIMEOUT:-5}"    # seconds to wait after SIGTERM
KEEP_LOGS=10

# Prints the pid of a live stack started by this script, or nothing. The pid file holds the pid and the
# process start time, so a recycled pid is not mistaken for the stack.
running_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid started
    { read -r pid; read -r started; } < "$PID_FILE"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$pid" ]] || return 1
    [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == "$(id -un)" ]] || return 1
    [[ "$(ps -o lstart= -p "$pid" 2>/dev/null)" == "$started" ]] || return 1
    echo "$pid"
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

cmd_start() {
    local pid
    if pid="$(running_pid)"; then
        echo "already running (pid $pid)"
        return 0
    fi
    mkdir -p "$STATE_DIR"
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
        launch+="export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42} ROS_DISCOVERY_SERVER=${ROS_DISCOVERY_SERVER:-10.43.254.23:11811} ROS_SUPER_CLIENT=TRUE; "
        launch+="exec ros2 launch bringup system.launch.py $(printf '%q ' "$@")"
    fi

    local log="$STATE_DIR/system_$(date +%Y%m%d_%H%M%S).log"
    ln -sfn "$log" "$STATE_DIR/latest.log"
    # The inner shell becomes the session leader, records its own pid and start time, then runs the stack.
    # --fork, not a shell "&": a backgrounded command in a non-interactive shell starts with SIGINT ignored,
    # and the stack would inherit that, so stop could never interrupt it.
    PID_FILE="$PID_FILE" setsid --fork bash -c '{ echo $$; ps -o lstart= -p $$; } > "$PID_FILE"; eval "$1"' _ "$launch" \
        > "$log" 2>&1 < /dev/null

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
    # Keep only the newest logs.
    ls -t "$STATE_DIR"/system_*.log 2>/dev/null | tail -n +"$(( KEEP_LOGS + 1 ))" | while read -r old; do rm -f -- "$old"; done
}

cmd_stop() {
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

cmd_status() {
    local pid
    if pid="$(running_pid)"; then
        echo "running $pid"
        return 0
    fi
    echo "stopped"
    return 3
}

case "${1:-}" in
    start) shift; cmd_start "$@" ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    *) echo "usage: $0 start [launch args] | stop | status" >&2; exit 2 ;;
esac
