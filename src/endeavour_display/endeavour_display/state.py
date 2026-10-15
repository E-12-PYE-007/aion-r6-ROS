"""What the display shows, and the rules that turn robot signals into it."""
import math
import threading
import time

LINK_TIMEOUT_S = 10.0    # the last successful contact with the Jetson must be at least this recent
DATA_TIMEOUT_S = 3.0     # while the stack runs, ROS data from it must be at least this fresh
FLAG_TIMEOUT_S = 1.5     # the CBF flags are published every tick, so an older one no longer counts
CAMERA_TIMEOUT_S = 2.0
MESSAGE_SECONDS = 8.0
MOVE_EPS_M = 0.01        # a weed that moves less than this has not been seen again

STOPPED, RUNNING, UNKNOWN = 'stopped', 'running', 'unknown'
STARTING, STOPPING = 'starting', 'stopping'


class DisplayState:
    """Thread-safe: the ROS callbacks, the operator thread and the HTTP handlers all touch it."""

    def __init__(self, plants, clock=time.monotonic):
        self._lock = threading.Lock()
        self._clock = clock
        self.plants = [dict(p) for p in plants]
        self._selected = [p['label'] for p in self.plants]
        self._pose = None
        self._weeds = {}
        self._filtering = (False, None)
        self._reversing = (False, None)
        self._heartbeat_t = None
        self._odom_t = None
        self._camera_t = None
        self._link_ok_t = None
        self._stack = UNKNOWN
        self._nvblox = None
        self._command = None
        self._message = (None, None)

    # ---- inputs from the robot

    def on_odometry(self, x, y, yaw):
        with self._lock:
            self._pose = (x, y, yaw)
            self._odom_t = self._clock()

    def on_weeds(self, weeds):
        now = self._clock()
        with self._lock:
            kept = {}
            for weed in weeds:
                previous = self._weeds.get(weed.track_id)
                moved = (previous is None or previous['label'] != weed.label
                         or math.hypot(previous['x'] - weed.x, previous['y'] - weed.y) > MOVE_EPS_M)
                kept[weed.track_id] = {'label': weed.label, 'x': weed.x, 'y': weed.y,
                                       'changed_at': now if moved else previous['changed_at']}
            self._weeds = kept

    def on_filtering(self, active):
        with self._lock:
            self._filtering = (bool(active), self._clock())

    def on_reversing(self, active):
        with self._lock:
            self._reversing = (bool(active), self._clock())

    def on_heartbeat(self):
        with self._lock:
            self._heartbeat_t = self._clock()

    def on_camera_frame(self):
        with self._lock:
            self._camera_t = self._clock()

    # ---- inputs from the ssh side

    def on_link(self, stack, nvblox=None):
        """A status poll or a command reached the Jetson and told us the stack is 'running' or 'stopped'."""
        with self._lock:
            self._link_ok_t = self._clock()
            self._stack = stack
            self._nvblox = nvblox

    def begin_command(self, name):
        with self._lock:
            self._command = name
            self._message = (None, None)

    def end_command(self, message=None):
        with self._lock:
            self._command = None
            if message:
                self._message = (message, self._clock() + MESSAGE_SECONDS)

    @property
    def command_in_flight(self):
        with self._lock:
            return self._command

    @property
    def stack(self):
        with self._lock:
            return self._stack

    # ---- the plant selection

    def selected(self):
        with self._lock:
            return list(self._selected)

    def can_edit_selection(self):
        with self._lock:
            return self._command is None and self._stack == STOPPED and self._link_ok(self._clock())

    def set_selected(self, labels):
        """False when the stack is not stopped: the classes are fixed when it starts."""
        with self._lock:
            if not (self._command is None and self._stack == STOPPED and self._link_ok(self._clock())):
                return False
            wanted = set(labels)
            self._selected = [p['label'] for p in self.plants if p['label'] in wanted]
            return True

    # ---- derived

    def _link_ok(self, now):
        return self._command is not None or (self._link_ok_t is not None and now - self._link_ok_t <= LINK_TIMEOUT_S)

    @staticmethod
    def _fresh(t, now, limit):
        return t is not None and now - t <= limit

    def snapshot(self):
        now = self._clock()
        with self._lock:
            link_ok = self._link_ok(now)
            stack = self._command or (self._stack if link_ok else UNKNOWN)
            data_ok = (self._fresh(self._heartbeat_t, now, DATA_TIMEOUT_S)
                       or self._fresh(self._odom_t, now, DATA_TIMEOUT_S))
            connected = link_ok and (self._stack != RUNNING or self._command is not None or data_ok)

            if stack != RUNNING:
                mode = 'off'
            elif self._reversing[0] and self._fresh(self._reversing[1], now, FLAG_TIMEOUT_S):
                mode = 'recovery'
            elif self._filtering[0] and self._fresh(self._filtering[1], now, FLAG_TIMEOUT_S):
                mode = 'cbf'
            else:
                mode = 'vla'

            message, until = self._message
            pose = None
            if self._pose is not None:
                pose = {'x': self._pose[0], 'y': self._pose[1], 'yaw': self._pose[2]}
            return {
                'connected': connected,
                'stack': stack,
                'nvblox': self._nvblox,
                'mode': mode,
                'pose': pose,
                'weeds': [{'id': i, 'label': w['label'], 'x': w['x'], 'y': w['y'], 'age': now - w['changed_at']}
                          for i, w in sorted(self._weeds.items())],
                'camera_alive': self._fresh(self._camera_t, now, CAMERA_TIMEOUT_S),
                'plants': self.plants,
                'selected': list(self._selected),
                'can_edit_selection': self._command is None and self._stack == STOPPED and link_ok,
                'message': message if (message and until is not None and now < until) else None,
            }
