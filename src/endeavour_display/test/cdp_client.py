"""A minimal Chrome DevTools Protocol client over a raw websocket, standard library only."""
import base64
import json
import os
import socket
import struct
import subprocess
import time
import urllib.request


class Chrome:
    def __init__(self, url, profile_dir, port, size='1920,1080'):
        self.proc = subprocess.Popen(
            ['google-chrome', '--headless=new', '--no-sandbox', '--disable-gpu', f'--window-size={size}',
             f'--remote-debugging-port={port}', f'--user-data-dir={profile_dir}', url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tab = None
        for _ in range(80):
            try:
                tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json', timeout=2))
                tab = next(t for t in tabs if t['type'] == 'page')
                break
            except Exception:
                time.sleep(0.25)
        if tab is None:
            self.proc.terminate()
            raise RuntimeError('chrome did not start')
        path = tab['webSocketDebuggerUrl'].split(f':{port}', 1)[1]
        self.sock = socket.create_connection(('127.0.0.1', port))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f'GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
                           f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        reply = b''
        while b'\r\n\r\n' not in reply:
            reply += self.sock.recv(1)
        assert b' 101 ' in reply.split(b'\r\n')[0], reply
        self._id = 0

    def _send(self, text):
        data = text.encode()
        mask = os.urandom(4)
        n = len(data)
        head = b'\x81' + (bytes([0x80 | n]) if n < 126 else b'\xfe' + struct.pack('>H', n) if n < 65536
                          else b'\xff' + struct.pack('>Q', n))
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _read(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return buf

    def _recv(self):
        message = b''
        while True:
            b1, b2 = self._read(2)
            n = b2 & 0x7f
            if n == 126:
                n = struct.unpack('>H', self._read(2))[0]
            elif n == 127:
                n = struct.unpack('>Q', self._read(8))[0]
            message += self._read(n)
            if b1 & 0x80:
                return message.decode()

    def call(self, method, **params):
        self._id += 1
        wanted = self._id
        self._send(json.dumps({'id': wanted, 'method': method, 'params': params}))
        while True:
            reply = json.loads(self._recv())
            if reply.get('id') == wanted:
                if 'error' in reply:
                    raise RuntimeError(reply['error'])
                return reply['result']

    def js(self, expression):
        return self.call('Runtime.evaluate', expression=expression, returnByValue=True)['result'].get('value')

    def screenshot(self, path):
        with open(path, 'wb') as out:
            out.write(base64.b64decode(self.call('Page.captureScreenshot', format='png')['data']))

    def close(self):
        try:
            self.call('Browser.close')
        except Exception:
            pass
        self.proc.terminate()
