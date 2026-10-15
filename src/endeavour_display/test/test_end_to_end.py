"""The whole display against a stand-in rover.

A real bridge node, the real bringup/scripts/aion_system.sh (running a fake robot that publishes the topics the
display reads), and a real Chrome tab driving the page. Needs Chrome and a sourced ROS environment.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cdp_client import Chrome

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
SCRIPT = PACKAGE_ROOT.parent / 'bringup' / 'scripts' / 'aion_system.sh'
FAKE_ROBOT = HERE / 'fake_robot.py'

pytestmark = pytest.mark.skipif(not (shutil.which('google-chrome') and shutil.which('ros2')),
                                reason='needs Chrome and a sourced ROS environment')

# Cross-process discovery on this setup only works on the default domain, so that is what the rig uses.
# Set DISPLAY_TEST_DOMAIN to use another.
DOMAIN = os.environ.get('DISPLAY_TEST_DOMAIN', '0')

UI = """(function () {
  var $ = function (id) { return document.getElementById(id); };
  return {
    conn: $('conntext').textContent, light: $('connlight').className,
    go: $('go').textContent, goDisabled: $('go').disabled, status: $('status').textContent,
    ind: $('indtext').textContent, weeds: $('nweeds').textContent,
    camLive: $('camwrap').classList.contains('live'), camWidth: $('camimg').naturalWidth,
    ticks: [].map.call(document.querySelectorAll('#ticks input'), function (i) { return [i.dataset.label, i.checked, i.disabled]; }),
    trail: [].filter.call(document.querySelectorAll('#trail line'), function (l) { return l.style.display !== 'none'; }).length,
    pins: [].filter.call(document.querySelectorAll('#pins g'), function (g) { return g.style.display !== 'none'; }).length
  };
})()"""
HOLD_START = "document.getElementById('go').dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, cancelable: true}))"
CLICK_STOP = "document.getElementById('go').click()"
COLLECT_ERRORS = ("window.__errors = []; window.addEventListener('error', function (e) { __errors.push(String(e.message)); });"
                  "window.addEventListener('unhandledrejection', function (e) { __errors.push(String(e.reason)); });")


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Rig:
    def __init__(self, tmp_path):
        self.state_dir = tmp_path / 'state'
        self.state_dir.mkdir()
        self.port = free_port()
        self.env = {
            **os.environ, 'ROS_DOMAIN_ID': DOMAIN, 'PYTHONPATH': f'{PACKAGE_ROOT}:{os.environ.get("PYTHONPATH", "")}',
            'AION_STATE_DIR': str(self.state_dir), 'AION_NVBLOX': '0', 'AION_STOP_TIMEOUT': '10', 'AION_TERM_TIMEOUT': '3',
            'AION_LAUNCH_CMD': f'echo "$AION_LAUNCH_ARGS" > "{self.state_dir}/args"; exec {sys.executable} {FAKE_ROBOT}',
        }
        self.bridge_log = open(tmp_path / 'bridge.log', 'w')
        self.bridge = subprocess.Popen(
            [sys.executable, '-m', 'endeavour_display.bridge_node', '--ros-args', '-p', f'port:={self.port}',
             '-p', 'local_test:=true', '-p', f'script_path:={SCRIPT}', '-p', 'poll_period_s:=1.0'],
            env=self.env, stdout=self.bridge_log, stderr=subprocess.STDOUT)
        self.chrome = Chrome(f'http://127.0.0.1:{self.port}/', tmp_path / 'chrome_profile', free_port())
        self.chrome.call('Page.enable')
        self.chrome.call('Page.addScriptToEvaluateOnNewDocument', source=COLLECT_ERRORS)
        self.chrome.call('Page.navigate', url=f'http://127.0.0.1:{self.port}/')

    def ui(self):
        return self.chrome.js(UI)

    def wait_for(self, check, timeout, what):
        end = time.monotonic() + timeout
        ui = None
        while time.monotonic() < end:
            try:
                ui = self.ui()
                if ui and check(ui):
                    return ui
            except Exception:
                pass
            time.sleep(0.2)
        pytest.fail(f'timed out waiting for {what}; last UI state: {ui}')

    def args_given_to_the_stack(self):
        return (self.state_dir / 'args').read_text().strip()

    def script(self, action):
        return subprocess.run([str(SCRIPT), action], env=self.env, capture_output=True, text=True, timeout=60)

    def close(self):
        self.script('stop')
        self.chrome.close()
        if self.bridge.poll() is None:
            self.bridge.terminate()
            try:
                self.bridge.wait(5)
            except subprocess.TimeoutExpired:
                self.bridge.kill()
        self.bridge_log.close()


@pytest.fixture
def rig(tmp_path):
    env = {**os.environ, 'ROS_DOMAIN_ID': DOMAIN}
    subprocess.run(['ros2', 'daemon', 'stop'], env=env, capture_output=True)
    topics = subprocess.run(['ros2', 'topic', 'list'], env=env, capture_output=True, text=True).stdout
    if '/odometry/local' in topics:
        pytest.skip(f'something is already publishing /odometry/local on ROS domain {DOMAIN}')
    rig = Rig(tmp_path)
    yield rig
    rig.close()


def test_the_display_from_connection_to_start_data_stop_and_loss(rig):
    # 1. It connects to the (local) "Jetson" and offers the plants, all ticked, ready to start.
    ui = rig.wait_for(lambda u: u['conn'] == 'Connected' and u['go'] == 'Start', 15, 'a connected, stopped display')
    assert ui['light'] == 'light g' and ui['status'] == 'Hold Start to launch' and ui['ind'] == 'Stack Off'
    assert ui['ticks'] == [['plant1', True, False], ['plant2', True, False], ['plant3', True, False], ['plant4', True, False]]

    # 2. Unticking a plant is remembered by the bridge, not just the page.
    rig.chrome.js("document.getElementById('plant-plant2').click()")
    rig.wait_for(lambda u: not dict((t[0], t[1]) for t in u['ticks'])['plant2'], 5, 'plant2 unticked')
    rig.chrome.call('Page.reload')
    ui = rig.wait_for(lambda u: u['conn'] == 'Connected' and u['go'] == 'Start', 15, 'the reloaded page')
    assert [t[1] for t in ui['ticks']] == [True, False, True, True]

    # 3. A short press does nothing; holding Start launches the stack with the chosen classes.
    rig.chrome.js(HOLD_START)
    time.sleep(0.5)
    rig.chrome.js("document.getElementById('go').dispatchEvent(new PointerEvent('pointerup', {bubbles: true}))")
    time.sleep(0.5)
    assert rig.script('status').returncode == 3
    rig.chrome.js(HOLD_START)
    rig.wait_for(lambda u: u['go'] in ('Starting', 'Stop'), 10, 'the start to begin')
    ui = rig.wait_for(lambda u: u['go'] == 'Stop' and u['camLive'] and u['camWidth'] > 0 and u['pins'] >= 6, 30,
                      'a running stack with data and a decoded camera frame')
    assert rig.args_given_to_the_stack() == 'target_classes:=plant1,plant3,plant4'
    assert ui['weeds'] == '6' and ui['camWidth'] > 0 and all(t[2] for t in ui['ticks'])   # locked while running
    assert ui['light'] == 'light g'

    # 4. The trail builds, and the state light follows the CBF flags through all three modes.
    seen, end = set(), time.monotonic() + 14
    while time.monotonic() < end and len(seen) < 3:
        seen.add(rig.ui()['ind'])
        time.sleep(0.25)
    assert {'VLA Active', 'CBF Active', 'Recovery Mode'} <= seen, seen
    assert rig.ui()['trail'] >= 3
    if os.environ.get('DISPLAY_TEST_SCREENSHOT'):
        rig.chrome.screenshot(os.environ['DISPLAY_TEST_SCREENSHOT'])

    # 5. The stack goes silent while ssh still works: the Connected light drops, then returns.
    pid = int((rig.state_dir / 'pid').read_text().splitlines()[0])
    os.killpg(pid, signal.SIGSTOP)
    try:
        ui = rig.wait_for(lambda u: u['conn'] == 'Connection Lost', 15, 'the light to drop when the data stops')
        assert ui['light'] == 'light r'
    finally:
        os.killpg(pid, signal.SIGCONT)
    rig.wait_for(lambda u: u['conn'] == 'Connected', 15, 'the light to recover')

    # 6. Stop returns to the start state and unlocks the plants.
    rig.chrome.js(CLICK_STOP)
    ui = rig.wait_for(lambda u: u['go'] == 'Start' and u['ind'] == 'Stack Off', 40, 'the stop to finish')
    assert rig.script('status').returncode == 3 and not any(t[2] for t in ui['ticks'])

    # 7. With no plant ticked nothing is detected: the stack starts without the weed pipeline.
    for label in ('plant1', 'plant3', 'plant4'):
        rig.chrome.js(f"document.getElementById('plant-{label}').click()")
    rig.wait_for(lambda u: not any(t[1] for t in u['ticks']), 5, 'all plants unticked')
    rig.chrome.js(HOLD_START)
    ui = rig.wait_for(lambda u: u['go'] == 'Stop', 30, 'the second start')
    assert rig.args_given_to_the_stack() == 'weed:=false'
    time.sleep(2)
    ui = rig.ui()
    assert ui['weeds'] == '0' and ui['camLive'] is False
    rig.chrome.js(CLICK_STOP)
    rig.wait_for(lambda u: u['go'] == 'Start', 40, 'the second stop')

    # 8. If the bridge itself goes away, the page says so.
    rig.bridge.terminate()
    ui = rig.wait_for(lambda u: u['conn'] == 'Connection Lost', 10, 'the page to notice the bridge is gone')
    assert "display bridge" in ui['status'] and ui['goDisabled'] is True

    assert rig.chrome.js('window.__errors') == []
