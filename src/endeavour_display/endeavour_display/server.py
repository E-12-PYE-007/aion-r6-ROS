"""The HTTP side: the page, a server-sent-event stream of the display state, the camera as MJPEG, and the API."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVENT_PERIOD_S = 0.1
MAX_BODY_BYTES = 4096
FONT_TYPES = {'.ttf': 'font/ttf'}
BOUNDARY = 'frame'


class CameraBuffer:
    """The latest JPEG, with a way for streaming clients to wait for the next one."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._count = 0

    def push(self, jpeg):
        with self._cond:
            self._frame = bytes(jpeg)
            self._count += 1
            self._cond.notify_all()

    def wait_for_frame(self, last_count, timeout):
        """(frame, count) newer than last_count, or (None, last_count) if none arrives within timeout."""
        with self._cond:
            if self._count == last_count:
                self._cond.wait(timeout)
            if self._count == last_count:
                return None, last_count
            return self._frame, self._count


def sse_message(snapshot):
    return ('data: ' + json.dumps(snapshot, separators=(',', ':')) + '\n\n').encode()


def make_handler(state, commander, camera, web_dir):
    web_dir = Path(web_dir)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):   # the bridge's own log is enough
            pass

        # ---- helpers

        def _send(self, status, body, content_type, extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload).encode(), 'application/json')

        def _read_json(self):
            length = int(self.headers.get('Content-Length') or 0)
            if length > MAX_BODY_BYTES:
                return None
            try:
                return json.loads(self.rfile.read(length) or b'{}')
            except ValueError:
                return None

        # ---- routes

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                return self._file(web_dir / 'index.html', 'text/html; charset=utf-8')
            if path.startswith('/fonts/'):
                font = web_dir / 'fonts' / Path(path).name
                return self._file(font, FONT_TYPES.get(font.suffix, 'application/octet-stream'))
            if path == '/events':
                return self._events()
            if path == '/camera.mjpg':
                return self._camera()
            self._json(404, {'error': 'not found'})

        def _file(self, path, content_type):
            if not path.is_file():
                return self._json(404, {'error': 'not found'})
            self._send(200, path.read_bytes(), content_type)

        def _events(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            try:
                while True:
                    self.wfile.write(sse_message(state.snapshot()))
                    self.wfile.flush()
                    time.sleep(EVENT_PERIOD_S)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _camera(self):
            self.send_response(200)
            self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            count = 0
            try:
                while True:
                    frame, count = camera.wait_for_frame(count, timeout=1.0)
                    if frame is None:
                        self.wfile.write(b'\r\n')       # keeps a dead client from lingering
                        self.wfile.flush()
                        continue
                    self.wfile.write(f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n'.encode())
                    self.wfile.write(frame + b'\r\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_POST(self):
            body = self._read_json()
            if body is None:
                return self._json(400, {'ok': False, 'message': 'bad request'})
            if self.path == '/api/start':
                accepted, message = commander.request_start()
            elif self.path == '/api/stop':
                accepted, message = commander.request_stop()
            elif self.path == '/api/select':
                labels = body.get('classes')
                if not isinstance(labels, list):
                    return self._json(400, {'ok': False, 'message': 'classes must be a list'})
                accepted = state.set_selected([str(label) for label in labels])
                message = 'ok' if accepted else 'the plants can only be changed while the stack is stopped'
            else:
                return self._json(404, {'ok': False, 'message': 'not found'})
            self._json(200 if accepted else 409, {'ok': accepted, 'message': message})

    return Handler


class DisplayServer:
    def __init__(self, state, commander, camera, web_dir, host='127.0.0.1', port=8080):
        self.httpd = ThreadingHTTPServer((host, port), make_handler(state, commander, camera, web_dir))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
