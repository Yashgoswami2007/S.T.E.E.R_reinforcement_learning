# env_client.py
# Simple client to communicate with Blender bridge
import socket
import json
import time
import _lsprof
import _hashlib

API= "https://datasets-server.huggingface.co/splits?dataset=allenai%2FolmOCR-bench"

class BlenderEnvClient:
    def __init__(self, host="172.16.5.93", port=5001):
        self.host = host
        self.port = port
        self.sock = None
        self._connect()

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Connecting to Blender bridge...")
        self.sock.connect((self.host, self.port))
        print("Connected")

    def recv_state(self):
        # DATa reading
        data = b""
        while not data.endswith(b"\n"):
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Disconnected")
            data += chunk
        j = json.loads(data.decode("utf-8").strip())
        if j.get("type") == "state":
            return j["state"]
        return None

    def send_action(self, action):
        msg = {"type":"action", "action": action}
        self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))

    def send_reset(self):
        msg = {"type":"reset"}
        self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except:
            pass


