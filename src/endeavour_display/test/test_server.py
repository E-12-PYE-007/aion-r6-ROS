import json
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from endeavour_display.server import CameraBuffer, DisplayServer
from endeavour_display.state import DisplayState

PLANTS = [{'label': 'plant1', 'name': 'Plant 1'}, {'label': 'plant2', 'name': 'Plant 2'}]


class FakeCommander:
    def __init__(self):
        self.calls = []

    def request_start(self):
        self.calls.append('start')
        return True, 'starting'

    def request_stop(self):
        self.calls.append('stop')
        return False, 'another command is still running'


@pytest.fixture
def rig(tmp_path):
    (tmp_path / 'fonts').mkdir()
    (tmp_path / 'index.html').write_text('<title>page</title>')
    (tmp_path / 'fonts' / 'a.ttf').write_bytes(b'FONT')
    state, camera, commander = DisplayState(PLANTS), CameraBuffer(), FakeCommander()
    server = DisplayServer(state, commander, camera, tmp_path, host='127.0.0.1', port=0)
    server.start()
    yield SimpleNamespace(base=f'http://127.0.0.1:{server.port}', state=state, camera=camera, commander=commander,
                           port=server.port)
    server.close()



def post(rig, path, payload=None):
    request = urllib.request.Request(rig.base + path, data=json.dumps(payload or {}).encode(), method='POST',
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def test_the_page_and_fonts_are_served_and_unknown_paths_are_404(rig):
    assert b'<title>page</title>' in urllib.request.urlopen(rig.base + '/', timeout=5).read()
    assert urllib.request.urlopen(rig.base + '/fonts/a.ttf', timeout=5).read() == b'FONT'
    for path in ('/nothing', '/fonts/../index.html', '/fonts/missing.ttf'):
        try:
            urllib.request.urlopen(rig.base + path, timeout=5)
            assert path == '/fonts/../index.html'      # the browser-side client normalises this one to /index.html
        except urllib.error.HTTPError as err:
            assert err.code == 404


def test_the_event_stream_sends_the_snapshot_repeatedly(rig):
    rig.state.on_link('stopped')
    with urllib.request.urlopen(rig.base + '/events', timeout=5) as stream:
        first = json.loads(stream.readline().decode().removeprefix('data: '))
        stream.readline()
        second = json.loads(stream.readline().decode().removeprefix('data: '))
    assert first['stack'] == 'stopped' and first['connected'] is True
    assert second['plants'] == PLANTS and 'weeds' in second


def test_the_camera_is_streamed_as_mjpeg_frames_in_order(rig):
    frames = [b'\xff\xd8first\xff\xd9', b'\xff\xd8second\xff\xd9']
    with socket.create_connection(('127.0.0.1', rig.port), timeout=5) as sock:
        sock.sendall(b'GET /camera.mjpg HTTP/1.1\r\nHost: x\r\n\r\n')
        time.sleep(0.2)
        for frame in frames:
            rig.camera.push(frame)
            time.sleep(0.15)
        data = b''
        sock.settimeout(1)
        try:
            while data.count(b'--frame') < 2:
                data += sock.recv(4096)
        except socket.timeout:
            pass
    assert b'multipart/x-mixed-replace; boundary=frame' in data
    assert data.index(b'first') < data.index(b'second')


def test_the_camera_buffer_only_hands_out_newer_frames():
    camera = CameraBuffer()
    assert camera.wait_for_frame(0, timeout=0.05) == (None, 0)
    camera.push(b'a')
    frame, count = camera.wait_for_frame(0, timeout=0.05)
    assert frame == b'a' and count == 1
    assert camera.wait_for_frame(count, timeout=0.05) == (None, 1)


def test_the_api_routes_to_the_commander_and_the_state(rig):
    assert post(rig, '/api/start') == (200, {'ok': True, 'message': 'starting'})
    assert post(rig, '/api/stop') == (409, {'ok': False, 'message': 'another command is still running'})
    assert rig.commander.calls == ['start', 'stop']

    assert post(rig, '/api/select', {'classes': ['plant1']})[0] == 409      # the link is not up yet
    rig.state.on_link('stopped')
    assert post(rig, '/api/select', {'classes': ['plant2']}) == (200, {'ok': True, 'message': 'ok'})
    assert rig.state.selected() == ['plant2']
    assert post(rig, '/api/select', {'classes': 'plant1'})[0] == 400
    assert post(rig, '/api/nothing')[0] == 404


def test_a_malformed_or_oversized_body_is_rejected(rig):
    request = urllib.request.Request(rig.base + '/api/start', data=b'{not json', method='POST')
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(request, timeout=5)
    assert err.value.code == 400
    request = urllib.request.Request(rig.base + '/api/start', data=b'{"x": "' + b'a' * 5000 + b'"}', method='POST')
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(request, timeout=5)
    assert err.value.code == 400
