import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from endeavour_display import jetson
from endeavour_display.jetson import JetsonControl, ssh_reachable
from endeavour_display.profiles import DEFAULT_SERVER, choose_profile

LAUNCH = Path(__file__).resolve().parents[1] / 'launch' / 'display.launch.py'
LAN, HOTSPOT = '10.43.254.23', '10.42.0.1'


def probe_answering(*hosts):
    return lambda host: host in hosts


def test_the_lan_is_used_when_the_jetson_answers_there():
    profile = choose_profile('auto', LAN, HOTSPOT, DEFAULT_SERVER, probe_answering(LAN, HOTSPOT), {})
    assert (profile.name, profile.discovery, profile.hosts) == ('lan', 'server', [LAN, HOTSPOT])
    assert profile.environment == {'ROS_DISCOVERY_SERVER': DEFAULT_SERVER, 'ROS_SUPER_CLIENT': 'TRUE'}


def test_the_hotspot_is_used_when_only_it_answers_and_the_discovery_settings_are_unset():
    profile = choose_profile('auto', LAN, HOTSPOT, DEFAULT_SERVER, probe_answering(HOTSPOT), {})
    assert (profile.name, profile.discovery, profile.hosts) == ('hotspot', 'multicast', [HOTSPOT, LAN])
    assert profile.environment == {'ROS_DISCOVERY_SERVER': None, 'ROS_SUPER_CLIENT': None}


@pytest.mark.parametrize('fallback', ['', HOTSPOT])
def test_when_nothing_answers_the_shells_own_environment_is_left_alone(fallback):
    profile = choose_profile('auto', LAN, fallback, DEFAULT_SERVER, probe_answering(), {})
    assert (profile.name, profile.discovery, profile.environment) == ('as-is', 'auto', {})


def test_the_mode_can_be_forced_and_a_bad_mode_is_rejected():
    never = probe_answering()
    assert choose_profile('server', LAN, '', DEFAULT_SERVER, never, {}).discovery == 'server'
    assert choose_profile('multicast', LAN, '', DEFAULT_SERVER, never, {}).discovery == 'multicast'
    with pytest.raises(ValueError):
        choose_profile('wifi', LAN, '', DEFAULT_SERVER, never, {})


def test_ssh_reachable_is_true_only_for_a_clean_exit():
    ran = []
    ok = lambda argv, **kw: ran.append(argv) or SimpleNamespace(returncode=0)
    assert ssh_reachable(LAN, runner=ok) is True and ran[0][-2:] == ['vla-cap@10.43.254.23', 'true']
    assert ssh_reachable(LAN, runner=lambda argv, **kw: SimpleNamespace(returncode=255)) is False

    def times_out(argv, **kw):
        raise subprocess.TimeoutExpired('ssh', 8)
    assert ssh_reachable(LAN, runner=times_out) is False


def test_start_tells_the_jetson_script_to_use_multicast_only_in_multicast_mode():
    sent = []

    def runner(argv, **kw):
        sent.append(argv[-1])
        return SimpleNamespace(returncode=0, stdout='running 5\n', stderr='')

    JetsonControl([LAN], script='~/s.sh', runner=runner, multicast=True).start(['plant1'])
    assert sent[0] == '~/s.sh start --multicast target_classes:=plant1'
    sent.clear()
    JetsonControl([LAN], script='~/s.sh', runner=runner).start(['plant1'])
    assert sent[0] == '~/s.sh start target_classes:=plant1'


# ------------------------------------------------------------------ the launch file, with a fake probe

def resolve(context, value):
    if isinstance(value, ParameterValue):
        value = value.value
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (tuple, list)) and value and not isinstance(value[0], str):
        return perform_substitutions(context, list(value))
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def run_launch(monkeypatch, answering, launch_args=(), shell_env=None):
    for name in ('ROS_DISCOVERY_SERVER', 'ROS_SUPER_CLIENT'):
        monkeypatch.delenv(name, raising=False)
    for name, value in (shell_env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(jetson, 'ssh_reachable', probe_answering(*answering))
    seen = {}

    def fake_node(self, context):
        seen['env'] = {name: context.environment.get(name, '<unset>') for name in ('ROS_DISCOVERY_SERVER', 'ROS_SUPER_CLIENT')}
        for group in getattr(self, '_Node__parameters', []):
            seen['params'] = {resolve(context, k): yaml.safe_load(resolve(context, v)) if isinstance(resolve(context, v), str) else resolve(context, v)
                              for k, v in group.items()}

    monkeypatch.setattr(Node, 'execute', fake_node)
    monkeypatch.setattr(ExecuteProcess, 'execute', lambda self, context: None)
    description = LaunchDescription([IncludeLaunchDescription(PythonLaunchDescriptionSource(str(LAUNCH)),
                                                              launch_arguments=list(launch_args))])
    service = LaunchService()
    service.include_launch_description(description)
    assert service.run() == 0
    return seen


def test_launch_on_the_lan(monkeypatch):
    seen = run_launch(monkeypatch, [LAN], [('jetson_fallback_host', HOTSPOT)])
    assert seen['env'] == {'ROS_DISCOVERY_SERVER': DEFAULT_SERVER, 'ROS_SUPER_CLIENT': 'TRUE'}
    assert seen['params']['discovery'] == 'server' and seen['params']['jetson_host'] == LAN


def test_launch_on_the_hotspot_unsets_the_discovery_settings_the_shell_had(monkeypatch):
    seen = run_launch(monkeypatch, [HOTSPOT], [('jetson_fallback_host', HOTSPOT)],
                      shell_env={'ROS_DISCOVERY_SERVER': DEFAULT_SERVER, 'ROS_SUPER_CLIENT': 'TRUE'})
    assert seen['env'] == {'ROS_DISCOVERY_SERVER': '<unset>', 'ROS_SUPER_CLIENT': '<unset>'}
    assert seen['params']['discovery'] == 'multicast' and seen['params']['jetson_host'] == HOTSPOT
    assert seen['params']['jetson_fallback_host'] == LAN


def test_launch_with_nothing_reachable_keeps_the_shell_as_it_was(monkeypatch):
    seen = run_launch(monkeypatch, [], [('jetson_fallback_host', HOTSPOT)])
    assert seen['env'] == {'ROS_DISCOVERY_SERVER': '<unset>', 'ROS_SUPER_CLIENT': '<unset>'}
    assert seen['params']['discovery'] == 'auto'
    seen = run_launch(monkeypatch, [], shell_env={'ROS_DISCOVERY_SERVER': 'x:1'})
    assert seen['env']['ROS_DISCOVERY_SERVER'] == 'x:1'


def test_launch_can_be_forced_into_multicast(monkeypatch):
    seen = run_launch(monkeypatch, [LAN], [('discovery_mode', 'multicast')])
    assert seen['params']['discovery'] == 'multicast' and seen['env']['ROS_DISCOVERY_SERVER'] == '<unset>'
