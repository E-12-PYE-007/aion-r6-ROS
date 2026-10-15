import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'aion_system.sh'

# A stand-in for the real stack: a leader and a child that each report the SIGINT they receive.
CHILD = ("import signal, sys, time; "
         "signal.signal(signal.SIGINT, lambda *a: (print('child got SIGINT', flush=True), sys.exit(0))); "
         "[time.sleep(0.1) for _ in iter(int, 1)]")
CLEAN_STACK = f'''
python3 -c "{CHILD}" &
trap 'echo leader got SIGINT; wait; echo clean exit; exit 0' INT
echo stack running
while :; do sleep 0.1; done
'''
STUBBORN_STACK = "trap '' INT TERM; while :; do sleep 0.1; done"

# A stand-in for docker that records every call and remembers whether the container exists or runs.
FAKE_DOCKER = r'''#!/usr/bin/env bash
echo "docker $*" >> "$FAKE_DOCKER_DIR/calls"
case "$1" in
  run)
    [[ -n "${FAKE_DOCKER_RUN_FAILS:-}" ]] && { echo "boom: no such image" >&2; exit 125; }
    touch "$FAKE_DOCKER_DIR/exists"
    [[ -n "${FAKE_DOCKER_EXITS_AT_ONCE:-}" ]] || touch "$FAKE_DOCKER_DIR/running"
    echo fakeid ;;
  ps)
    if [[ "$*" == *status=running* ]]; then [[ -f "$FAKE_DOCKER_DIR/running" ]] && echo fakeid
    else [[ -f "$FAKE_DOCKER_DIR/exists" ]] && echo fakeid; fi ;;
  stop) rm -f "$FAKE_DOCKER_DIR/running" ;;
  rm) rm -f "$FAKE_DOCKER_DIR/running" "$FAKE_DOCKER_DIR/exists" ;;
  logs) echo "fake nvblox log line" ;;
esac
exit 0
'''


def script_env(state_dir, launch=None, extra=None):
    env = {**os.environ, 'AION_STATE_DIR': str(state_dir), 'AION_STOP_TIMEOUT': '3', 'AION_TERM_TIMEOUT': '2',
           'AION_NVBLOX': '0'}
    if launch is not None:
        env['AION_LAUNCH_CMD'] = launch
    env.update(extra or {})
    return env


def run(state_dir, *args, launch=None, extra=None):
    return subprocess.run([str(SCRIPT), *args], env=script_env(state_dir, launch, extra),
                          capture_output=True, text=True, timeout=60)


def start_in_background(state_dir, *args, launch=None, extra=None):
    return subprocess.Popen([str(SCRIPT), *args], env=script_env(state_dir, launch, extra),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.fixture
def state_dir(tmp_path):
    yield tmp_path
    run(tmp_path, 'stop')


def test_status_is_stopped_when_nothing_was_started(state_dir):
    result = run(state_dir, 'status')
    assert result.returncode == 3 and result.stdout.strip() == 'stopped'


def test_start_status_and_a_clean_stop(state_dir):
    started = run(state_dir, 'start', launch=CLEAN_STACK)
    assert started.returncode == 0 and started.stdout.startswith('started (pid ')
    assert run(state_dir, 'status').stdout.startswith('running ')
    assert 'already running' in run(state_dir, 'start', launch=CLEAN_STACK).stdout

    stopped = run(state_dir, 'stop')
    assert stopped.returncode == 0 and stopped.stdout.strip() == 'stopped'
    log = (state_dir / 'latest.log').read_text()
    assert 'leader got SIGINT' in log and 'child got SIGINT' in log and 'clean exit' in log
    assert run(state_dir, 'status').returncode == 3


def test_stop_escalates_when_the_stack_ignores_sigint_and_sigterm(state_dir):
    assert run(state_dir, 'start', launch=STUBBORN_STACK).returncode == 0
    stopped = run(state_dir, 'stop')
    assert stopped.returncode == 0 and 'SIGKILL' in stopped.stdout
    assert run(state_dir, 'status').returncode == 3


def test_a_stack_that_exits_straight_away_is_reported(state_dir):
    result = run(state_dir, 'start', launch='echo boom; exit 1')
    assert result.returncode == 1 and 'boom' in result.stderr


def test_a_stale_pid_file_is_ignored_and_cleaned(state_dir):
    (state_dir / 'pid').write_text('999999\nFri Jan  1 00:00:00 2100\n')
    assert run(state_dir, 'status').returncode == 3
    assert 'not running' in run(state_dir, 'stop').stdout
    assert not (state_dir / 'pid').exists()


def test_a_pid_the_script_does_not_own_is_never_signalled(state_dir):
    # pid 1 is its own group leader but belongs to root, so it must be refused.
    (state_dir / 'pid').write_text('1\nanything\n')
    assert run(state_dir, 'status').returncode == 3
    assert 'not running' in run(state_dir, 'stop').stdout


def test_stop_finishes_escalating_when_the_connection_that_ran_it_drops(state_dir):
    assert run(state_dir, 'start', launch=STUBBORN_STACK).returncode == 0
    stopper = start_in_background(state_dir, 'stop')
    time.sleep(0.8)                      # mid-wait, after SIGINT was ignored by the stack
    stopper.send_signal(signal.SIGHUP)   # what sshd sends when the connection drops
    out, _ = stopper.communicate(timeout=20)
    assert stopper.returncode == 0 and 'SIGKILL' in out
    assert run(state_dir, 'status').returncode == 3


def test_two_starts_at_once_launch_one_stack(state_dir):
    counter = '$AION_STATE_DIR/launches'
    launch = f'echo launched >> "{counter}"; while :; do sleep 0.1; done'
    first = start_in_background(state_dir, 'start', launch=launch)
    second = start_in_background(state_dir, 'start', launch=launch)
    outputs = [first.communicate(timeout=30)[0], second.communicate(timeout=30)[0]]
    assert first.returncode == 0 and second.returncode == 0
    assert sum('started (pid' in o for o in outputs) == 1 and sum('already running' in o for o in outputs) == 1
    assert (state_dir / 'launches').read_text().count('launched') == 1


# ------------------------------------------------------------------ nvblox in docker

@pytest.fixture
def docker(tmp_path):
    shim_dir = tmp_path / 'bin'
    shim_dir.mkdir()
    shim = shim_dir / 'docker'
    shim.write_text(FAKE_DOCKER)
    shim.chmod(0o755)
    fake_dir = tmp_path / 'fake_docker'
    fake_dir.mkdir()
    workspace = tmp_path / 'isaac-ros-ws'
    workspace.mkdir()
    extra = {'AION_NVBLOX': '1', 'AION_NVBLOX_SETTLE': '0.3', 'AION_ISAAC_ROS_WS': str(workspace),
             'FAKE_DOCKER_DIR': str(fake_dir), 'PATH': f'{shim_dir}:{os.environ["PATH"]}'}
    return extra, fake_dir


def calls(fake_dir):
    path = fake_dir / 'calls'
    return path.read_text().splitlines() if path.exists() else []


def test_nvblox_is_started_after_the_stack_with_the_run_dev_flags(state_dir, docker):
    extra, fake_dir = docker
    result = run(state_dir, 'start', launch=CLEAN_STACK, extra=extra)
    assert result.returncode == 0 and 'nvblox started' in result.stdout
    run_call = next(c for c in calls(fake_dir) if c.startswith('docker run '))
    for expected in ('-d', '--init', '--name aion-nvblox', '--network host', '--ipc=host', '--runtime nvidia',
                     '--privileged', '-e ROS_DOMAIN_ID=42', '-e ROS_DISCOVERY_SERVER=10.43.254.23:11811',
                     f'-v {extra["AION_ISAAC_ROS_WS"]}:/workspaces/isaac_ros-dev',
                     '--entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh', 'isaac_ros_dev-aarch64',
                     'ros2 launch aion_vslam nvblox_camera.launch.py'):
        assert expected in run_call, expected
    status = run(state_dir, 'status', extra=extra).stdout.splitlines()
    assert status[0].startswith('running ') and status[1] == 'nvblox running'
    run(state_dir, 'stop', extra=extra)


def test_stop_halts_the_stack_before_nvblox(state_dir, docker):
    extra, fake_dir = docker
    log = fake_dir / 'calls'
    launch = f'trap \'echo "STACK GOT SIGINT" >> "{log}"; exit 0\' INT; while :; do sleep 0.1; done'
    assert run(state_dir, 'start', launch=launch, extra=extra).returncode == 0
    stopped = run(state_dir, 'stop', extra=extra)
    assert stopped.returncode == 0 and stopped.stdout.splitlines() == ['stopped', 'nvblox stopped']
    order = calls(fake_dir)
    stack_stopped = order.index('STACK GOT SIGINT')
    nvblox_stopped = next(i for i, c in enumerate(order) if c.startswith('docker stop --signal SIGINT'))
    assert stack_stopped < nvblox_stopped
    assert any(c.startswith('docker logs') for c in order)
    assert list(state_dir.glob('nvblox_*.log'))
    assert run(state_dir, 'status', extra=extra).stdout.splitlines() == ['stopped', 'nvblox stopped']


@pytest.mark.parametrize('failure', ['FAKE_DOCKER_RUN_FAILS', 'FAKE_DOCKER_EXITS_AT_ONCE'])
def test_if_nvblox_does_not_come_up_the_stack_is_stopped_again(state_dir, docker, failure):
    extra, fake_dir = docker
    result = run(state_dir, 'start', launch=CLEAN_STACK, extra={**extra, failure: '1'})
    assert result.returncode == 1 and 'nvblox' in result.stderr
    assert run(state_dir, 'status', extra=extra).stdout.splitlines()[0] == 'stopped'


@pytest.mark.parametrize('start_args', [['--no-nvblox'], ['controller:=pure_pursuit']])
def test_nvblox_is_skipped_when_it_is_not_needed(state_dir, docker, start_args):
    extra, fake_dir = docker
    assert run(state_dir, 'start', *start_args, launch=CLEAN_STACK, extra=extra).returncode == 0
    assert not any(c.startswith('docker run') for c in calls(fake_dir))
    run(state_dir, 'stop', extra=extra)


def test_docker_is_never_touched_when_nvblox_is_disabled(state_dir, docker):
    extra, fake_dir = docker
    extra = {**extra, 'AION_NVBLOX': '0'}
    assert run(state_dir, 'start', launch=CLEAN_STACK, extra=extra).returncode == 0
    assert run(state_dir, 'stop', extra=extra).returncode == 0
    assert calls(fake_dir) == []


def test_a_system_clock_adjustment_does_not_make_a_live_stack_look_stopped(state_dir, tmp_path):
    # ps derives its start time from the wall clock, which can shift by a second when the clock is adjusted.
    # The shim makes every "ps -o lstart" answer differ, as if the clock moved between calls.
    real_ps = shutil.which('ps')
    shim_dir = tmp_path / 'psbin'
    shim_dir.mkdir()
    shim = shim_dir / 'ps'
    shim.write_text(f'#!/usr/bin/env bash\nif [[ "$*" == *lstart* ]]; then echo "Jan $RANDOM $RANDOM"; exit 0; fi\n'
                    f'exec {real_ps} "$@"\n')
    shim.chmod(0o755)
    extra = {'PATH': f'{shim_dir}:{os.environ["PATH"]}'}

    assert run(state_dir, 'start', launch=CLEAN_STACK, extra=extra).returncode == 0
    for _ in range(3):
        assert run(state_dir, 'status', extra=extra).stdout.startswith('running ')
    assert run(state_dir, 'stop', extra=extra).stdout.strip() == 'stopped'
    assert run(state_dir, 'status', extra=extra).returncode == 3
