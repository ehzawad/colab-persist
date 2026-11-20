"""Offline fakes shared by the unit tests.

* ``FakeDrive``: the DriveClient surface used by ``remote.Store``, ``datasets.ShardCache``
  and ``backend.snapshots``, with real SHA-256/MD5 and fault-injection knobs. It is
  in-memory by default; with ``state_dir`` it keeps its state on disk so that a
  subprocess (the detached job worker) can share it.
* ``FakeTokens``: a token source whose ``get()`` can be made to fail.
* ``ScriptedOpener`` / ``FakeResponse``: an HTTP-level ``opener=`` for ``drive_api.DriveClient``.
* ``FakeColabSession``: an HTTP-level session for ``colab_api.ColabAPI``.
* ``FakeColabAPI``: a method-level stand-in for ``colab_api.ColabAPI``.

Nothing here touches the network. Only placeholder identities are used.
"""
from __future__ import annotations

import base64
import contextlib
from datetime import datetime, timedelta, timezone
import fcntl
import functools
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import urllib.error

from colab_persist import drive_api

# Offline tests never start the real background Drive helper; its own tests opt back in.
os.environ["COLAB_PERSIST_NO_KEEPER"] = "1"

FOLDER = drive_api.FOLDER_MIME
EMAIL = "first@example.com"
QUOTA_PROJECT = "demo-quota-project"
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_VERIFY = drive_api.DriveClient._verify  # Bound now: install_file_drive replaces DriveClient.


# -- tokens -------------------------------------------------------------------------------
class FakeTokens:
    """Stands in for drive_api.TokenSource.

    ``expire(times)`` makes the next ``times`` calls to ``get()`` raise TokenExpired;
    ``revoked = True`` makes every call raise until it is reset.
    """

    def __init__(self, token="fake-access-token", quota_project=QUOTA_PROJECT, folder="Colab-Persist"):
        self.token, self.quota_project, self._folder = token, quota_project, folder
        self.failures_left = 0
        self.revoked = False
        self.calls = 0

    def expire(self, times=1):
        self.failures_left = times

    def get(self):
        self.calls += 1
        if self.revoked:
            raise drive_api.TokenExpired("Fake Drive token was revoked.")
        if self.failures_left:
            self.failures_left -= 1
            raise drive_api.TokenExpired("The Drive access token expired (fake).")
        return self.token, self.quota_project

    def folder(self):
        return self._folder


# -- Drive --------------------------------------------------------------------------------
def _transaction(method):
    """Load/save the on-disk state around a public call when the fake is file-backed."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.state_dir is None or self._depth:
            self._depth += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._depth -= 1
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / "state.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._load()
                self._depth += 1
                try:
                    return method(self, *args, **kwargs)
                finally:
                    self._depth -= 1
                    self._save()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return wrapper


class FakeDrive:
    """In-memory Drive with the DriveClient surface used by the package.

    Fault knobs:
      * ``fail_upload(suffix, error, times=1)``: the next upload(s) whose name ends with
        ``suffix`` raise ``error`` before anything is stored (an upload that never completed).
      * ``corrupt_uploads.add(suffix)``: Drive stores mangled bytes, so the Drive-side
        checksum differs and DriveClient-style verification fails (the bad file remains).
      * ``lag = True``: files created from now on are hidden from ``children()`` until
        ``settle()`` (list-after-write lag); ``get()`` still finds them.
      * ``add_duplicate_folder(parts)``: create a second folder with the same name.
      * ``corrupt(file_id, data, keep_metadata=False)``: replace stored bytes, optionally
        keeping the old size/checksums in metadata.
      * ``after_upload``: callable(name, metadata) run after each stored upload.
      * ``report_sha256 = False``: omit sha256Checksum from metadata (MD5 only).
    """

    def __init__(self, email=EMAIL, tokens=None, state_dir=None):
        self.email = email
        self.tokens = tokens if tokens is not None else FakeTokens()
        self.state_dir = Path(state_dir) if state_dir else None
        self.files: dict = {}
        self.counter = 0
        self.hidden: set = set()
        self.calls: list = []
        self.upload_failures: list = []
        self.corrupt_uploads: set = set()
        self.lag = False
        self.after_upload = None
        self.report_sha256 = True
        self._depth = 0

    # -- persistence (file-backed mode) ----------------------------------------------------
    def _load(self):
        path = self.state_dir / "state.json"
        if not path.exists():
            return
        state = json.loads(path.read_text())
        self.counter = state["counter"]
        self.files = {}
        for file_id, record in state["files"].items():
            record = dict(record)
            if record.get("data") is not None:
                record["data"] = base64.b64decode(record["data"])
            self.files[file_id] = record

    def _save(self):
        files = {}
        for file_id, record in self.files.items():
            record = dict(record)
            if record.get("data") is not None:
                record["data"] = base64.b64encode(record["data"]).decode()
            files[file_id] = record
        path = self.state_dir / "state.json"
        temporary = path.with_name(f"state.{os.getpid()}.partial")
        temporary.write_text(json.dumps({"counter": self.counter, "files": files}))
        os.replace(temporary, path)

    # -- internals ------------------------------------------------------------------------
    def _new_id(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter:05d}"

    def _create(self, name, parent, mime, data=None, app_properties=None):
        file_id = self._new_id("folder-" if mime == FOLDER else "file-")
        self.files[file_id] = {
            "id": file_id, "name": name, "parents": [parent], "mimeType": mime,
            "appProperties": dict(app_properties or {}), "trashed": False, "data": data,
            "createdTime": (_EPOCH + timedelta(seconds=self.counter)).isoformat().replace("+00:00", "Z"),
            "order": self.counter, "reported": None}
        if self.lag:
            self.hidden.add(file_id)
        return self.files[file_id]

    def _meta(self, record):
        meta = {"id": record["id"], "name": record["name"], "mimeType": record["mimeType"],
                "createdTime": record["createdTime"], "parents": list(record["parents"])}
        if record["appProperties"]:
            meta["appProperties"] = dict(record["appProperties"])
        if record["mimeType"] != FOLDER:
            data = record["data"] or b""
            size, md5, sha = record["reported"] or (len(data), hashlib.md5(data).hexdigest(),
                                                    hashlib.sha256(data).hexdigest())
            meta.update(size=str(size), md5Checksum=md5)
            if self.report_sha256:
                meta["sha256Checksum"] = sha
        return meta

    def _maybe_fail(self, name):
        for rule in self.upload_failures:
            suffix, error, remaining = rule
            if name.endswith(suffix) and (remaining is None or remaining > 0):
                if remaining is not None:
                    rule[2] = remaining - 1
                raise error

    def _store_upload(self, operation, data, parent, name, app_properties, mime):
        self.calls.append((operation, name))
        self._maybe_fail(name)
        if parent not in self.files and parent != "root":
            raise drive_api.DriveError(f"Parent {parent} not found (fake).", 404, "notFound")
        stored = data
        if any(name.endswith(suffix) for suffix in self.corrupt_uploads):
            stored = bytes([data[0] ^ 0xFF]) + data[1:] if data else b"x"  # Same size, other bytes.
        record = self._create(name, parent, mime, stored, app_properties)
        meta = self._meta(record)
        if self.after_upload:
            self.after_upload(name, meta)
        # The same verification DriveClient applies to what Drive reports.
        return _VERIFY(meta, len(data), hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest())

    # -- DriveClient surface ---------------------------------------------------------------
    @_transaction
    def about(self):
        self.calls.append(("about", None))
        return {"user": {"emailAddress": self.email},
                "storageQuota": {"limit": str(15 * 1024 ** 3), "usage": "0"}}

    @_transaction
    def children(self, parent, *, name=None, folders=None):
        self.calls.append(("children", name))
        result = []
        for record in sorted(self.files.values(), key=lambda r: r["order"]):
            if record["trashed"] or parent not in record["parents"] or record["id"] in self.hidden:
                continue
            if name is not None and record["name"] != name:
                continue
            is_folder = record["mimeType"] == FOLDER
            if (folders is True and not is_folder) or (folders is False and is_folder):
                continue
            result.append(self._meta(record))
        return result

    @_transaction
    def get(self, file_id, fields=None):
        self.calls.append(("get", file_id))
        record = self.files.get(file_id)
        if record is None:
            return None
        return {**self._meta(record), "trashed": record["trashed"]}

    @_transaction
    def find_folder(self, parent, name):
        matches = self.children(parent, name=name, folders=True)
        if len(matches) > 1:
            listing = ", ".join(f"{m['id']} (created {m.get('createdTime')})" for m in matches)
            raise drive_api.DuplicateFolder(f"Drive has {len(matches)} folders named {name!r} in one place: "
                                            f"{listing}. Merge or rename them in Drive, then retry; "
                                            "nothing was changed.")
        return matches[0]["id"] if matches else None

    @_transaction
    def ensure_folder(self, parent, name):
        existing = self.find_folder(parent, name)
        if existing:
            return existing
        self.calls.append(("create_folder", name))
        return self._create(name, parent, FOLDER)["id"]

    @_transaction
    def resolve(self, parts, *, create=False, root="root"):
        current = root
        for part in parts:
            found = self.ensure_folder(current, part) if create else self.find_folder(current, part)
            if found is None:
                return None
            current = found
        return current

    @_transaction
    def upload_file(self, path, parent, name, *, app_properties=None, sha256=None,
                    mime="application/octet-stream", progress=None):
        path = Path(path)
        data = path.read_bytes()
        if sha256 and hashlib.sha256(data).hexdigest() != sha256:
            raise drive_api.DriveError(f"{path.name} changed before upload; refusing to publish it.")
        meta = self._store_upload("upload_file", data, parent, name, app_properties, mime)
        if progress:
            progress(len(data), len(data))
        return meta

    @_transaction
    def upload_bytes(self, data, parent, name, *, app_properties=None, mime="application/json"):
        return self._store_upload("upload_bytes", bytes(data), parent, name, app_properties, mime)

    @_transaction
    def download(self, file_id, destination, *, expected_size=None, expected_sha256=None,
                 reserve_check=None):
        self.calls.append(("download", file_id))
        record = self.files.get(file_id)
        if record is None or record["trashed"]:
            raise drive_api.DriveError(f"File not found: {file_id}.", 404, "notFound")
        data = record["data"] or b""
        if expected_size is not None and len(data) > expected_size:
            raise drive_api.DriveError("Drive returned more bytes than expected.")
        if reserve_check and data:
            reserve_check(len(data))
        with Path(destination).open("wb") as target:
            target.write(data)
        if expected_size is not None and len(data) != expected_size:
            raise drive_api.DriveError(f"Downloaded {len(data)} bytes, expected {expected_size}.")
        result = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and result != expected_sha256:
            raise drive_api.DriveError("Downloaded file failed its SHA-256 check.")
        return result

    @_transaction
    def read_bytes(self, file_id, max_bytes=64 * 1024 * 1024):
        self.calls.append(("read_bytes", file_id))
        record = self.files.get(file_id)
        if record is None or record["trashed"]:
            raise drive_api.DriveError(f"File not found: {file_id}.", 404, "notFound")
        if len(record["data"] or b"") > max_bytes:
            raise drive_api.DriveError("Drive file exceeds the allowed size for an in-memory read.")
        return record["data"] or b""

    # -- test helpers ----------------------------------------------------------------------
    def fail_upload(self, suffix, error, times=1):
        self.upload_failures.append([suffix, error, times])

    def settle(self):
        self.hidden.clear()

    @_transaction
    def put(self, parts, name, data, app_properties=None, mime="application/octet-stream"):
        """Place a file directly (as another writer would), creating folders as needed."""
        parent = self.resolve(list(parts), create=True)
        return self._meta(self._create(name, parent, mime, bytes(data), app_properties))

    @_transaction
    def add_duplicate_folder(self, parts):
        parent = self.resolve(list(parts[:-1]), create=True)
        return self._create(parts[-1], parent, FOLDER)["id"]

    @_transaction
    def corrupt(self, file_id, data, keep_metadata=False):
        record = self.files[file_id]
        if keep_metadata:
            old = record["data"] or b""
            record["reported"] = (len(old), hashlib.md5(old).hexdigest(), hashlib.sha256(old).hexdigest())
        record["data"] = bytes(data)

    @_transaction
    def trash(self, file_id):
        self.files[file_id]["trashed"] = True

    @_transaction
    def listing(self, parts):
        """Names of the visible, untrashed files in a folder path (empty if it is missing)."""
        folder = self.resolve(list(parts))
        return [] if folder is None else [item["name"] for item in self.children(folder, folders=False)]

    @_transaction
    def by_name(self, name):
        matches = [r for r in self.files.values() if r["name"] == name and not r["trashed"]]
        if len(matches) != 1:
            raise AssertionError(f"Expected one file named {name}, found {len(matches)}")
        return {**self._meta(matches[0]), "data": matches[0]["data"]}

    @_transaction
    def data_of(self, file_id):
        return self.files[file_id]["data"]

    def upload_names(self):
        return [name for operation, name in self.calls if operation in ("upload_file", "upload_bytes")]


def install_file_drive(state_dir, email=EMAIL):
    """Make drive_api.DriveClient build a file-backed FakeDrive (used inside the worker subprocess)."""
    def factory(tokens, **_ignored):
        return FakeDrive(email=email, tokens=tokens, state_dir=state_dir)
    drive_api.DriveClient = factory


SITECUSTOMIZE = '''\
import importlib.util, os
_state = os.environ.get("COLAB_PERSIST_FAKE_DRIVE")
if _state:
    _spec = importlib.util.spec_from_file_location("colab_persist_test_fakes", os.environ["COLAB_PERSIST_FAKES_FILE"])
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.install_file_drive(_state)
'''


# -- HTTP-level fakes for drive_api.DriveClient ---------------------------------------------
def json_body(value) -> bytes:
    return json.dumps(value).encode()


def drive_error_body(status, reason, message="fake error") -> bytes:
    return json_body({"error": {"code": status, "message": message, "errors": [{"reason": reason}]}})


def http_headers(values=None):
    message = http.client.HTTPMessage()
    for key, value in (values or {}).items():
        message[key] = value
    return message


class FakeResponse:
    """A urllib-style response. ``chunks`` controls what each read() returns; an exception
    instance in ``chunks`` is raised at that point (a dropped connection)."""

    def __init__(self, status=200, headers=None, body=b"", chunks=None):
        self.status = status
        self.headers = http_headers(headers)
        self._chunks = list(chunks) if chunks is not None else [body]
        self.closed = False

    def read(self, size=-1):
        while self._chunks:
            item = self._chunks.pop(0)
            if isinstance(item, BaseException):
                raise item
            if size is None or size < 0:
                return item + b"".join(c for c in self._chunks if isinstance(c, bytes))
            if len(item) > size:
                self._chunks.insert(0, item[size:])
                return item[:size]
            return item
        return b""

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


class Recorded:
    def __init__(self, request):
        self.method = request.get_method()
        self.url = request.full_url
        self.headers = {key.lower(): value for key, value in request.header_items()}
        self.body = request.data


class ScriptedOpener:
    """``opener=`` for DriveClient. Each scripted item is one of:
    (status, headers, body); a FakeResponse; an exception instance to raise; or a
    callable(Recorded) returning one of those. Non-2xx statuses raise HTTPError like urllib."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[Recorded] = []
        self.errors: list[urllib.error.HTTPError] = []

    def close(self):
        """Close the HTTPErrors handed out (DriveClient reads but does not close them)."""
        for error in self.errors:
            error.close()

    def add(self, *responses):
        self.responses.extend(responses)

    def open(self, request, timeout=None):
        recorded = Recorded(request)
        self.requests.append(recorded)
        if not self.responses:
            raise AssertionError(f"Unexpected request: {recorded.method} {recorded.url}")
        item = self.responses.pop(0)
        if callable(item) and not isinstance(item, (FakeResponse, BaseException)):
            item = item(recorded)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, FakeResponse):
            return item
        status, headers, body = item
        if not 200 <= status < 300:
            error = urllib.error.HTTPError(recorded.url, status, "fake", http_headers(headers), io.BytesIO(body))
            self.errors.append(error)
            raise error
        return FakeResponse(status, headers, body)


# -- Colab API fakes ------------------------------------------------------------------------
GPU_L4 = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
GPU_T4 = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
GPU_A100 = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}
CPU = {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": "SHAPE_STANDARD"}
TPU_V5E1 = {"variant": "VARIANT_TPU", "accelerator": "V5E1", "shape": "SHAPE_STANDARD"}


def default_specs():
    return [{"key": GPU_L4, "eligible": True}, {"key": GPU_T4, "eligible": True},
            {"key": CPU, "eligible": True}, {"key": GPU_A100, "eligible": False},
            {"key": TPU_V5E1, "eligible": True}]


def make_runtime(runtime_id, spec=None, url=None):
    return {"name": "runtimes/" + runtime_id, "runtimeSpec": dict(spec or GPU_L4),
            "connectionInfo": {"url": url or f"https://8080-m-s-{runtime_id}-c.us-central1.prod.r.colab.dev",
                               "token": "fake-runtime-proxy-token",
                               "expireTime": "2026-11-20T12:00:00.123456789Z"}}


class FakeColabAPI:
    """Method-level stand-in for colab_api.ColabAPI."""

    def __init__(self, runtimes=None, specs=None, tier="SUBSCRIPTION_TIER_NONE"):
        self.runtimes = list(runtimes or [])
        self.specs = specs if specs is not None else default_specs()
        self.tier = tier
        self.created, self.deleted, self.enabled = [], [], []
        self.list_error = None

    def subscription(self):
        return self.tier

    def runtime_specs(self):
        return self.specs

    def list_runtimes(self):
        if self.list_error:
            raise self.list_error
        return [dict(item) for item in self.runtimes]

    def get_runtime(self, runtime_id):
        return next((dict(r) for r in self.runtimes if r["name"] == "runtimes/" + runtime_id), None)

    def create_runtime(self, runtime_id, request_id, spec, **_ignored):
        self.created.append((runtime_id, request_id, spec))
        runtime = make_runtime(runtime_id, spec)
        self.runtimes.append(runtime)
        return runtime

    def delete_runtime(self, runtime_id, *, allow_missing=False):
        self.deleted.append(runtime_id)
        self.runtimes = [r for r in self.runtimes if r["name"] != "runtimes/" + runtime_id]

    def enable_services(self, project, services=("colaboratory.googleapis.com", "drive.googleapis.com")):
        self.enabled.append(project)
        return {service: "ENABLED" for service in services}


class FakeColabResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.content = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())

    def json(self):
        return json.loads(self.content)


class FakeColabSession:
    """HTTP-level session for ColabAPI. Scripted items: (status, body) or callable(call)->(status, body)."""

    def __init__(self, *responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, params=None, json=None, timeout=None, headers=None):
        call = {"method": method, "url": url, "params": params, "json": json, "headers": headers,
                "timeout": timeout}
        self.calls.append(call)
        if not self.responses:
            raise AssertionError(f"Unexpected Colab API request: {method} {url}")
        item = self.responses.pop(0)
        if callable(item):
            item = item(call)
        status, body = item
        return FakeColabResponse(status, body)


def colab_error(status, code, message="fake error", reason=None):
    error = {"code": status, "status": code, "message": message}
    if reason:
        error["details"] = [{"reason": reason}]
    return status, {"error": error}


def fixed_umask(case, mask=0o022):
    """Checkpoint modes depend on the umask; pin it for one test (restored afterwards)."""
    previous = os.umask(mask)
    case.addCleanup(os.umask, previous)


@contextlib.contextmanager
def preserved_sigterm():
    """Store._work installs a SIGTERM handler; restore the test process's handler afterwards."""
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
