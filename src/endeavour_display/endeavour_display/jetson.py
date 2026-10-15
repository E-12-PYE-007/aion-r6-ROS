"""Runs bringup's aion_system.sh on the Jetson, over a fresh ssh connection for every command.

Nothing is held open between commands: a long-lived connection goes stale when the link flaps or the Jetson
changes address, and then hangs. Every command is idempotent, so a failed one can simply be retried.
"""
import shlex
import subprocess
from dataclasses import dataclass

DEFAULT_SCRIPT = '~/vla-cap-ws/aion-r6-vla-master/aion-r6-ROS/src/bringup/scripts/aion_system.sh'
SSH_OPTIONS = [
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=5',
    '-o', 'ServerAliveInterval=2',
    '-o', 'ServerAliveCountMax=3',
    '-o', 'ControlMaster=no',
    '-o', 'ControlPath=none',
    '-o', 'StrictHostKeyChecking=accept-new',
]
STATUS_TIMEOUT_S = 12
COMMAND_TIMEOUT_S = 90        # a stop can take the stack's 20 s, plus escalation and the nvblox container
SSH_FAILED = 255
SCRIPT_NOT_FOUND = 127


@dataclass
class Result:
    ok: bool
    kind: str          # ok, unreachable, timeout, failed
    message: str = ''
    output: str = ''


def start_arguments(selected_labels):
    """The launch arguments for the chosen weed classes. With none chosen there is nothing to detect."""
    if not selected_labels:
        return ['weed:=false']
    return ['target_classes:=' + ','.join(selected_labels)]


def parse_status(output):
    """('running'|'stopped', 'running'|'stopped'|None) from aion_system.sh status."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    stack = 'running' if lines and lines[0].startswith('running') else 'stopped'
    nvblox = None
    for line in lines[1:]:
        if line.startswith('nvblox '):
            nvblox = line.split(' ', 1)[1]
    return stack, nvblox


class JetsonControl:
    def __init__(self, hosts, user='vla-cap', script=DEFAULT_SCRIPT, runner=subprocess.run, local=False, env=None):
        """hosts: addresses to try in order (the usual one, then the hotspot). local runs the script in a
        local shell instead of over ssh, for tests."""
        self.hosts = [h for h in hosts if h]
        self.user = user
        self.script = script
        self._run = runner
        self.local = local
        self.env = env
        self.last_host = self.hosts[0] if self.hosts else None

    def _argv(self, host, remote_command):
        if self.local:
            return ['bash', '-c', remote_command]
        return ['ssh', *SSH_OPTIONS, f'{self.user}@{host}', remote_command]

    def _remote(self, action, args=()):
        # The script path is left unquoted so the remote shell expands the ~.
        return ' '.join([self.script, action, *[shlex.quote(a) for a in args]])

    def _execute(self, action, args, timeout):
        """(Result, stdout). Tries the host that last worked first, then the others, only on connection failures."""
        ordered = ([self.last_host] + [h for h in self.hosts if h != self.last_host]) if self.hosts else [None]
        result, stdout = Result(False, 'unreachable', 'no Jetson address configured'), ''
        for host in ordered:
            try:
                completed = self._run(self._argv(host, self._remote(action, args)), capture_output=True,
                                      text=True, timeout=timeout, env=self.env)
            except subprocess.TimeoutExpired:
                result, stdout = Result(False, 'timeout', f'the Jetson did not answer within {timeout} s'), ''
                continue
            except OSError as exc:
                return Result(False, 'failed', f'could not run ssh: {exc}'), ''
            output = (completed.stdout or '') + (completed.stderr or '')
            if not self.local and completed.returncode == SSH_FAILED:
                result, stdout = Result(False, 'unreachable', "can't reach the Jetson", output), ''
                continue
            self.last_host = host
            if completed.returncode == SCRIPT_NOT_FOUND:
                return Result(False, 'failed', 'aion_system.sh is not on the Jetson (is its workspace updated?)',
                              output), completed.stdout or ''
            return Result(True, 'ok', output=output), completed.stdout or ''
        return result, stdout

    def status(self):
        """(Result, 'running'|'stopped'|None, nvblox state or None). Exit code 3 only means the stack is stopped."""
        result, stdout = self._execute('status', (), STATUS_TIMEOUT_S)
        if not result.ok:
            return result, None, None
        stack, nvblox = parse_status(stdout)
        return result, stack, nvblox

    def start(self, selected_labels):
        return self._command('start', start_arguments(selected_labels))

    def stop(self):
        return self._command('stop', ())

    def _command(self, action, args):
        result, stdout = self._execute(action, args, COMMAND_TIMEOUT_S)
        if not result.ok:
            return result
        # The script's own exit status is not visible here, so ask it.
        check, stack, _ = self.status()
        if not check.ok:
            return check
        wanted = 'running' if action == 'start' else 'stopped'
        if stack != wanted:
            return Result(False, 'failed', (result.output.strip().splitlines() or [f'{action} did not take effect'])[-1],
                          result.output)
        return Result(True, 'ok', output=result.output)
