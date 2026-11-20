"""OpenSSH ProxyCommand: a WebSocket<->stdio bridge to the exact managed runtime.

Each new connection fetches fresh connection info from the official Colab API, so
the hour-long runtime proxy token never needs to be stored. The token travels only
in a request header to the runtime's own ``*.colab.dev`` host.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import urlparse

import websocket

SSH_PATH = "/colab/ssh"
BUSY = ("Colab accepts one SSH session per runtime and another client holds it (HTTP 429): an interactive "
        "SSH, a native `colab ssh`, or another tool's connection. Close it and retry; a multiplexed "
        "`ssh colab-cuda` connection closes with `ssh -O exit colab-cuda`.")


def public_key(identity: Path) -> str:
    pub = Path(str(identity) + ".pub")
    if pub.is_file():
        return pub.read_text().strip()
    return subprocess.run(["ssh-keygen", "-y", "-f", str(identity)], capture_output=True,
                          text=True, check=True).stdout.strip()


def connect(runtime: dict, pubkey: str):
    info = runtime["connectionInfo"]
    parsed = urlparse(info["url"])
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".colab.dev"):
        raise RuntimeError("Unexpected runtime address; refusing to send the runtime token.")
    ws = websocket.WebSocket()
    try:
        ws.connect(f"wss://{parsed.netloc}{SSH_PATH}", timeout=60,
                   header=[f"X-Colab-Ssh-Pubkey: {pubkey}", f"X-Colab-Runtime-Proxy-Token: {info['token']}"])
    except websocket.WebSocketBadStatusException as error:
        if error.status_code == 429:
            raise RuntimeError(BUSY) from None
        raise RuntimeError(f"The runtime refused SSH (HTTP {error.status_code}). Check `colab-persist status`.") from None
    ws.settimeout(None)
    return ws


def _write_all(descriptor, data):
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def bridge(ws, stdin=0, stdout=1):
    def upstream():
        try:
            while True:
                data = os.read(stdin, 65536)
                if not data:
                    break
                ws.send_binary(data)
        except (OSError, websocket.WebSocketException):
            pass

    threading.Thread(target=upstream, daemon=True).start()
    while True:
        try:
            opcode, frame = ws.recv_data(control_frame=True)
        except (websocket.WebSocketConnectionClosedException, OSError):
            break
        if opcode == websocket.ABNF.OPCODE_CLOSE:
            break
        if opcode in (websocket.ABNF.OPCODE_BINARY, websocket.ABNF.OPCODE_TEXT) and frame:
            _write_all(stdout, frame)
    try:
        ws.close()
    except websocket.WebSocketException:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", help="Managed runtime ID this connection is pinned to")
    parser.add_argument("--endpoint", help=argparse.SUPPRESS)  # v0.3 alias fragments; refreshed by `start`.
    args = parser.parse_args()
    try:
        from . import backend, colab_api
        item = backend.require_runtime()
        pinned = args.runtime or args.endpoint
        if colab_api.runtime_id_of(item) != pinned:
            raise RuntimeError("Runtime was replaced. Run `colab-persist start` to refresh the SSH alias.")
        cfg = backend.config()
        key = Path(cfg.get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser()
        bridge(connect(item, public_key(key)))
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
