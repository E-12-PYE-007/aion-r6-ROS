"""Polls the Jetson's status and runs Start and Stop, one command at a time."""
import threading

from endeavour_display.state import RUNNING, STOPPED, UNKNOWN


class Commander:
    def __init__(self, control, state, poll_s=5.0):
        self.control = control
        self.state = state
        self.poll_s = poll_s
        self._command_lock = threading.Lock()
        self._quit = threading.Event()

    # ---- the status poll

    def poll_once(self):
        """Asks the Jetson whether the stack is running. A command that is in flight reports for itself."""
        if self.state.command_in_flight:
            return
        result, stack, nvblox = self.control.status()
        if result.ok:
            self.state.on_link(stack, nvblox)

    def run_forever(self):
        while not self._quit.is_set():
            self.poll_once()
            self._quit.wait(self.poll_s)

    def close(self):
        self._quit.set()

    # ---- commands from the page: (accepted, message)

    def request_start(self):
        stack = self.state.snapshot()['stack']
        if stack == UNKNOWN:
            return False, "can't reach the robot"
        if stack != STOPPED:
            return False, 'the stack is already running' if stack == RUNNING else f'the stack is {stack}'
        return self._launch('starting', lambda: self.control.start(self.state.selected()), RUNNING)

    def request_stop(self):
        return self._launch('stopping', self.control.stop, STOPPED)

    def _launch(self, name, action, expected_stack):
        if not self._command_lock.acquire(blocking=False):
            return False, 'another command is still running'
        self.state.begin_command(name)
        threading.Thread(target=self._execute, args=(name, action, expected_stack), daemon=True).start()
        return True, name

    def _execute(self, name, action, expected_stack):
        try:
            result = action()
            if result.ok:
                self.state.on_link(expected_stack, None)
            self.state.end_command(None if result.ok else result.message)
            if not result.ok:
                self.poll_once()    # find out what state it really is in
        except Exception as exc:    # never leave the page stuck on "starting"
            self.state.end_command(f'{name} failed: {exc}')
        finally:
            self._command_lock.release()
