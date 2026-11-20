"""WebSocket<->stdio SSH bridge: the runtime token only goes to the runtime's own colab.dev host."""
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import websocket

from colab_persist import proxy

import fakes


class ConnectTests(unittest.TestCase):
    def runtime(self, url):
        runtime = fakes.make_runtime("cp-cuda-aaa111")
        runtime["connectionInfo"]["url"] = url
        return runtime

    def test_refuses_any_host_but_https_colab_dev(self):
        for url in ("http://8080-m-s-x-c.us.prod.r.colab.dev", "https://evil.example.com",
                    "https://colab.dev.evil.example.com", "https://8080-x-c.colab.dev@evil.example.com",
                    "https://colab.dev", "wss://8080-m-s-x-c.us.prod.r.colab.dev", "not a url"):
            with self.subTest(url=url), patch.object(proxy.websocket, "WebSocket") as socket:
                with self.assertRaisesRegex(RuntimeError, "refusing to send the runtime token"):
                    proxy.connect(self.runtime(url), "ssh-ed25519 AAAA test")
                socket.assert_not_called()

    def test_connects_to_the_runtime_ssh_endpoint_with_headers(self):
        runtime = fakes.make_runtime("cp-cuda-aaa111")
        with patch.object(proxy.websocket, "WebSocket") as socket:
            ws = proxy.connect(runtime, "ssh-ed25519 AAAA test")
        host = runtime["connectionInfo"]["url"].split("://", 1)[1]
        ws.connect.assert_called_once()
        self.assertEqual(ws.connect.call_args.args[0], f"wss://{host}/colab/ssh")
        self.assertEqual(ws.connect.call_args.kwargs["header"],
                         ["X-Colab-Ssh-Pubkey: ssh-ed25519 AAAA test",
                          "X-Colab-Runtime-Proxy-Token: fake-runtime-proxy-token"])
        ws.settimeout.assert_called_once_with(None)
        self.assertIs(ws, socket.return_value)

    def test_busy_slot_and_other_refusals(self):
        runtime = fakes.make_runtime("cp-cuda-aaa111")
        for status, message in ((429, "one SSH session per runtime"), (403, r"refused SSH \(HTTP 403\)")):
            with self.subTest(status=status), patch.object(proxy.websocket, "WebSocket") as socket:
                socket.return_value.connect.side_effect = websocket.WebSocketBadStatusException(
                    "Handshake status %d", status)
                with self.assertRaisesRegex(RuntimeError, message):
                    proxy.connect(runtime, "key")

    def test_public_key_prefers_the_pub_file(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "id_test"
            Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA from-file\n")
            self.assertEqual(proxy.public_key(key), "ssh-ed25519 AAAA from-file")
            Path(str(key) + ".pub").unlink()
            with patch.object(proxy.subprocess, "run", return_value=MagicMock(stdout="ssh-ed25519 derived\n")) as run:
                self.assertEqual(proxy.public_key(key), "ssh-ed25519 derived")
            self.assertEqual(run.call_args.args[0], ["ssh-keygen", "-y", "-f", str(key)])


class FakeWebSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []
        self.closed = False
        self.upstream_done = threading.Event()

    def send_binary(self, data):
        self.sent.append(data)

    def recv_data(self, control_frame=False):
        if not self.frames:
            raise websocket.WebSocketConnectionClosedException("closed")
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class BridgeTests(unittest.TestCase):
    def test_bridge_copies_both_directions_until_close(self):
        stdin_read, stdin_write = os.pipe()
        stdout_read, stdout_write = os.pipe()
        os.write(stdin_write, b"client hello")
        os.close(stdin_write)
        ws = FakeWebSocket([(websocket.ABNF.OPCODE_BINARY, b"server "), (websocket.ABNF.OPCODE_PING, b"ignored"),
                            (websocket.ABNF.OPCODE_TEXT, b"banner"), (websocket.ABNF.OPCODE_CLOSE, b"")])
        sent_before_close = threading.Event()
        original = ws.recv_data

        def recv_data(control_frame=False):
            if ws.frames and ws.frames[0][0] == websocket.ABNF.OPCODE_CLOSE:
                sent_before_close.wait(5)  # Let the upstream thread forward stdin first.
            return original(control_frame)
        ws.recv_data = recv_data
        original_send = ws.send_binary

        def send_binary(data):
            original_send(data)
            sent_before_close.set()
        ws.send_binary = send_binary
        proxy.bridge(ws, stdin=stdin_read, stdout=stdout_write)
        os.close(stdout_write)
        with os.fdopen(stdout_read, "rb") as received:
            self.assertEqual(received.read(), b"server banner")
        os.close(stdin_read)
        self.assertEqual(b"".join(ws.sent), b"client hello")
        self.assertTrue(ws.closed)

    def test_main_refuses_a_replaced_runtime(self):
        with patch.object(proxy.sys, "argv", ["proxy", "--runtime", "cp-cuda-old000"]), \
             patch("colab_persist.backend.require_runtime", return_value=fakes.make_runtime("cp-cuda-new111")), \
             patch.object(proxy, "connect") as connect, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as stopped:
                proxy.main()
        self.assertEqual(stopped.exception.code, 1)
        connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
