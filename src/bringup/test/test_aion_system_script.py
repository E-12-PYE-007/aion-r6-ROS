import os
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


def run(state_dir, *args, launch=None):
    env = {**os.environ, 'AION_STATE_DIR': str(state_dir), 'AION_STOP_TIMEOUT': '3', 'AION_TERM_TIMEOUT': '2'}
    if launch is not None:
        env['AION_LAUNCH_CMD'] = launch
    return subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30)


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


def start_in_background(state_dir, *args, launch=None):
    env = {**os.environ, 'AION_STATE_DIR': str(state_dir), 'AION_STOP_TIMEOUT': '3', 'AION_TERM_TIMEOUT': '2'}
    if launch is not None:
        env['AION_LAUNCH_CMD'] = launch
    return subprocess.Popen([str(SCRIPT), *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


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
