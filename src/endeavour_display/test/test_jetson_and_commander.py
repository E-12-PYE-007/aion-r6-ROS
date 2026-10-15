import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from endeavour_display.commander import Commander
from endeavour_display.jetson import (SSH_OPTIONS, JetsonControl, Result, parse_status, start_arguments)
from endeavour_display.state import DisplayState

SCRIPT = '~/ws/aion_system.sh'


def completed(code=0, out='', err=''):
    return SimpleNamespace(returncode=code, stdout=out, stderr=err)


class FakeRunner:
    """Answers by host, and records every command it was given."""

    def __init__(self, answers):
        self.answers = answers      # host -> list of results, consumed in order (the last one repeats)
        self.calls = []

    def __call__(self, argv, **kwargs):
        host = argv[-2].split('@')[1] if '@' in argv[-2] else None
        self.calls.append((argv, kwargs))
        queue = self.answers[host]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def control(answers, hosts=('10.0.0.1',)):
    runner = FakeRunner(answers)
    return JetsonControl(list(hosts), script=SCRIPT, runner=runner), runner


def test_every_command_is_a_fresh_ssh_with_timeouts_and_no_multiplexing():
    jetson, runner = control({'10.0.0.1': [completed(0, 'running 12\nnvblox running\n')]})
    jetson.status()
    argv, kwargs = runner.calls[0]
    assert argv[0] == 'ssh' and argv[1:1 + len(SSH_OPTIONS)] == SSH_OPTIONS
    assert argv[-2:] == ['vla-cap@10.0.0.1', '~/ws/aion_system.sh status']   # ~ left for the remote shell to expand
    for option in ('BatchMode=yes', 'ConnectTimeout=5', 'ControlMaster=no', 'ControlPath=none'):
        assert option in argv
    assert kwargs['timeout'] > 0


def test_status_exit_code_3_just_means_stopped():
    jetson, _ = control({'10.0.0.1': [completed(3, 'stopped\nnvblox stopped\n')]})
    result, stack, nvblox = jetson.status()
    assert result.ok and (stack, nvblox) == ('stopped', 'stopped')


def test_parse_status_and_start_arguments():
    assert parse_status('running 4242\nnvblox running\n') == ('running', 'running')
    assert parse_status('stopped\n') == ('stopped', None)
    assert start_arguments(['plant1', 'plant3']) == ['target_classes:=plant1,plant3']
    assert start_arguments([]) == ['weed:=false']


def test_ssh_failure_is_unreachable_and_the_hotspot_address_is_tried_next():
    jetson, runner = control({'10.0.0.1': [completed(255, '', 'Connection timed out')],
                              '10.42.0.1': [completed(0, 'stopped\n')]}, hosts=('10.0.0.1', '10.42.0.1'))
    result, stack, _ = jetson.status()
    assert result.ok and stack == 'stopped' and jetson.last_host == '10.42.0.1'
    jetson.status()
    assert runner.calls[-1][0][-2] == 'vla-cap@10.42.0.1'      # the host that worked is tried first from now on


def test_when_no_host_answers_it_is_unreachable_or_a_timeout():
    jetson, _ = control({'10.0.0.1': [completed(255)]})
    assert jetson.status()[0].kind == 'unreachable'
    jetson, _ = control({'10.0.0.1': [subprocess.TimeoutExpired('ssh', 12)]})
    result = jetson.status()[0]
    assert result.kind == 'timeout' and not result.ok


def test_a_missing_script_is_reported_in_words():
    jetson, _ = control({'10.0.0.1': [completed(127, '', 'No such file or directory')]})
    result = jetson.status()[0]
    assert result.kind == 'failed' and 'aion_system.sh is not on the Jetson' in result.message


def test_start_is_confirmed_by_asking_for_the_status_afterwards():
    jetson, runner = control({'10.0.0.1': [completed(0, 'started (pid 5)\n'), completed(0, 'running 5\nnvblox running\n')]})
    assert jetson.start(['plant2']).ok
    assert runner.calls[0][0][-1] == "~/ws/aion_system.sh start 'target_classes:=plant2'" or \
        runner.calls[0][0][-1] == '~/ws/aion_system.sh start target_classes:=plant2'
    assert runner.calls[1][0][-1].endswith(' status')


def test_a_start_that_did_not_take_effect_reports_the_scripts_last_line():
    jetson, _ = control({'10.0.0.1': [completed(1, '', 'nvblox did not start, stopping the stack again\n'),
                                       completed(3, 'stopped\n')]})
    result = jetson.start(['plant1'])
    assert not result.ok and result.kind == 'failed' and 'nvblox did not start' in result.message


def test_stop_is_confirmed_too():
    jetson, _ = control({'10.0.0.1': [completed(0, 'stopped\n'), completed(3, 'stopped\n')]})
    assert jetson.stop().ok
    jetson, _ = control({'10.0.0.1': [completed(0, 'stopped\n'), completed(0, 'running 5\n')]})
    assert not jetson.stop().ok


def test_local_mode_runs_a_shell_not_ssh():
    jetson = JetsonControl(['x'], script='true', runner=FakeRunner({None: [completed(0)]}), local=True)
    jetson.status()
    assert jetson._run.calls[0][0][:2] == ['bash', '-c']


# ------------------------------------------------------------------ the commander

class FakeControl:
    def __init__(self, stack='stopped', start=None, stop=None, gate=None):
        self.stack, self._start, self._stop, self.gate = stack, start or Result(True, 'ok'), stop or Result(True, 'ok'), gate
        self.started_with = None

    def status(self):
        return Result(True, 'ok'), self.stack, None

    def start(self, labels):
        self.started_with = labels
        if self.gate:
            self.gate.wait(5)
        return self._start

    def stop(self):
        if self.gate:
            self.gate.wait(5)
        return self._stop


PLANTS = [{'label': 'plant1', 'name': 'Plant 1'}, {'label': 'plant2', 'name': 'Plant 2'}]


def wait_idle(state, timeout=3):
    end = time.monotonic() + timeout
    while state.command_in_flight and time.monotonic() < end:
        time.sleep(0.01)


def test_start_is_refused_when_the_robot_cannot_be_reached_or_is_already_running():
    state = DisplayState(PLANTS)
    commander = Commander(FakeControl(), state)
    assert commander.request_start() == (False, "can't reach the robot")
    state.on_link('running')
    accepted, message = commander.request_start()
    assert not accepted and 'already running' in message


def test_start_passes_the_selected_plants_and_ends_running():
    state = DisplayState(PLANTS)
    control = FakeControl()
    commander = Commander(control, state)
    state.on_link('stopped')
    state.set_selected(['plant2'])
    assert commander.request_start() == (True, 'starting')
    wait_idle(state)
    assert control.started_with == ['plant2'] and state.stack == 'running'


def test_a_failed_start_shows_the_reason_and_re_reads_the_real_state():
    state = DisplayState(PLANTS)
    commander = Commander(FakeControl(start=Result(False, 'failed', 'nvblox did not start')), state)
    state.on_link('stopped')
    commander.request_start()
    wait_idle(state)
    snap = state.snapshot()
    assert snap['message'] == 'nvblox did not start' and snap['stack'] == 'stopped'


def test_only_one_command_runs_at_a_time():
    gate = threading.Event()
    state = DisplayState(PLANTS)
    commander = Commander(FakeControl(stack='running', gate=gate), state)
    state.on_link('running')
    assert commander.request_stop() == (True, 'stopping')
    assert commander.request_stop() == (False, 'another command is still running')
    gate.set()
    wait_idle(state)
    assert state.stack == 'stopped'


def test_stop_is_allowed_even_when_the_state_is_unknown():
    state = DisplayState(PLANTS)
    commander = Commander(FakeControl(), state)
    assert commander.request_stop() == (True, 'stopping')
    wait_idle(state)


def test_an_exception_in_a_command_never_leaves_the_page_stuck():
    class Boom(FakeControl):
        def stop(self):
            raise RuntimeError('boom')

    state = DisplayState(PLANTS)
    commander = Commander(Boom(), state)
    commander.request_stop()
    wait_idle(state)
    assert state.command_in_flight is None and 'boom' in state.snapshot()['message']
    assert commander.request_stop()[0] is True      # and the lock was released


def test_the_poll_updates_the_link_and_skips_while_a_command_runs():
    state = DisplayState(PLANTS)
    commander = Commander(FakeControl(stack='running'), state)
    commander.poll_once()
    assert state.stack == 'running'
    state.begin_command('stopping')
    state.on_link('stopped')
    commander.poll_once()
    assert state.stack == 'stopped'                  # the poll did not overwrite it
