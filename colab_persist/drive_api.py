"""Stdlib-only Google Drive v3 client, used on this computer and on the Colab VM.

Credentials never reach the VM in long-lived form. Callers supply a short-lived
OAuth access token, either directly or through a JSON token file that the local
client refreshes (``COLAB_DRIVE_TOKEN_FILE``). Every request carries the quota
project, which the Drive API requires for user credentials.

Files are verified by the size and SHA-256 that Drive reports after an upload.
A resumable upload that never completes leaves no file in Drive.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.googleapis.com/drive/v3"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"
CHUNK_BYTES = 16 * 1024 * 1024  # A multiple of Drive's required 256 KiB.
BLOCK_BYTES = 4 * 1024 * 1024
TOKEN_ENV = "COLAB_DRIVE_TOKEN_FILE"
FOLDER_ENV = "COLAB_DRIVE_FOLDER"
DEFAULT_TOKEN_FILE = "/dev/shm/colab-persist/drive-token.json"
FILE_FIELDS = "id,name,size,md5Checksum,sha256Checksum,mimeType,createdTime,appProperties"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
RATE_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "backendError"}
QUOTA_REASONS = {"storageQuotaExceeded", "dailyLimitExceeded", "uploadLimitExceeded",
                 "teamDriveFileLimitExceeded", "quotaExceeded"}


class DriveError(RuntimeError):
    def __init__(self, message, status=None, reason=None):
        super().__init__(message)
        self.status, self.reason = status, reason


class TokenExpired(DriveError):
    """The access token was rejected and no fresher one is available yet."""


class QuotaExceeded(DriveError):
    """Drive storage or the daily upload limit is exhausted. Retrying will not help."""


class DuplicateFolder(DriveError):
    """More than one folder has the same name under one parent."""


def escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def write_token_file(path, token: dict) -> Path:
    """Atomically write a short-lived token document readable only by its owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as target:
        json.dump(token, target)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    return path


class TokenSource:
    """Supplies (access_token, quota_project); a file source is re-read on demand."""

    def __init__(self, token=None, quota_project=None, path=None):
        self._token, self._quota, self.path = token, quota_project, Path(path) if path else None

    @classmethod
    def from_environment(cls):
        return cls(path=os.environ.get(TOKEN_ENV, DEFAULT_TOKEN_FILE))

    def get(self):
        if self.path is None:
            if not self._token or not self._quota:
                raise TokenExpired("No Drive access token is available.")
            return self._token, self._quota
        try:
            document = json.loads(self.path.read_text())
        except FileNotFoundError:
            raise TokenExpired("Drive is not connected on this VM yet. Run `colab-persist mount` "
                               "on your computer; no browser is needed.") from None
        token, quota = document.get("token"), document.get("quota_project")
        if not token or not quota:
            raise TokenExpired("The Drive token file is incomplete.")
        expires = document.get("expires_at")
        if isinstance(expires, (int, float)) and expires < time.time() + 30:
            raise TokenExpired("The Drive access token expired. Reconnect with `colab-persist attach`, "
                               "`colab-persist mount` or any other command from your computer.")
        return token, quota

    def folder(self):
        if self.path is not None:
            try:
                value = json.loads(self.path.read_text()).get("folder")
                if value:
                    return value
            except (FileNotFoundError, ValueError):
                pass
        return os.environ.get(FOLDER_ENV, "Colab-Persist")


class DriveClient:
    def __init__(self, tokens: TokenSource, *, opener=None, sleep=time.sleep, retries=6, timeout=120):
        self.tokens = tokens
        self.opener = opener or urllib.request.build_opener()
        self.sleep, self.retries, self.timeout = sleep, retries, timeout

    # -- transport -----------------------------------------------------------------
    def _send(self, method, url, body=None, headers=None):
        token, quota = self.tokens.get()
        merged = {"Authorization": "Bearer " + token, "x-goog-user-project": quota}
        merged.update(headers or {})
        request = urllib.request.Request(url, data=body, method=method, headers=merged)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            with error:
                return error.code, dict(error.headers or {}), error.read(), None
        return response.status, dict(response.headers or {}), None, response

    @staticmethod
    def _reason(raw):
        try:
            error = json.loads(raw or b"{}").get("error", {})
        except ValueError:
            return None, (raw or b"")[:300].decode(errors="replace")
        reasons = [item.get("reason") for item in error.get("errors", []) if item.get("reason")]
        for detail in error.get("details", []):
            if detail.get("reason"):
                reasons.append(detail["reason"])
        return (reasons[0] if reasons else error.get("status")), error.get("message", "")

    def request(self, method, url, *, params=None, body=None, headers=None, ok=(200,), stream=False):
        """Send with bounded retries. Returns (status, headers, payload or response)."""
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        refreshed = False
        for attempt in range(self.retries + 1):
            try:
                status, response_headers, raw, response = self._send(method, url, body, headers)
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout,
                    ConnectionError, TimeoutError) as error:
                if attempt == self.retries:
                    raise DriveError(f"Drive request failed: {error}") from None
                self.sleep(min(60, 2 ** attempt))
                continue
            if status in ok or (status == 308 and 308 in ok):
                if stream:
                    return status, response_headers, response
                payload = response.read() if response is not None else raw
                return status, response_headers, payload
            reason, message = self._reason(raw)
            if status == 401:
                if not refreshed:
                    refreshed = True  # A file token source may now hold a fresher token.
                    continue
                raise TokenExpired("Drive rejected the access token (401). Reconnect from your computer.",
                                   status, reason)
            if status == 403 and reason in QUOTA_REASONS:
                raise QuotaExceeded(f"Drive quota exhausted ({reason}): {message}", status, reason)
            if status in RETRY_STATUS or (status == 403 and reason in RATE_REASONS):
                if attempt < self.retries:
                    self.sleep(min(60, 2 ** attempt))
                    continue
            raise DriveError(f"Drive {method} failed ({status} {reason}): {message}", status, reason)
        raise DriveError("Drive request retries exhausted.")

    def json(self, method, url, **kwargs):
        _, _, payload = self.request(method, url, **kwargs)
        return json.loads(payload or b"{}")

    # -- metadata ------------------------------------------------------------------
    def about(self):
        return self.json("GET", API + "/about", params={"fields": "user(emailAddress),storageQuota"})

    def list(self, query, fields=FILE_FIELDS, order="createdTime"):
        items, token = [], None
        while True:
            params = {"q": query, "fields": f"nextPageToken,files({fields})", "pageSize": 1000,
                      "orderBy": order, "spaces": "drive"}
            if token:
                params["pageToken"] = token
            page = self.json("GET", API + "/files", params=params)
            items.extend(page.get("files", []))
            token = page.get("nextPageToken")
            if not token:
                return items

    def children(self, parent, *, name=None, folders=None):
        terms = [f"'{escape(parent)}' in parents", "trashed = false"]
        if name is not None:
            terms.append(f"name = '{escape(name)}'")
        if folders is True:
            terms.append(f"mimeType = '{FOLDER_MIME}'")
        elif folders is False:
            terms.append(f"mimeType != '{FOLDER_MIME}'")
        return self.list(" and ".join(terms))

    def get(self, file_id, fields=FILE_FIELDS):
        try:
            return self.json("GET", f"{API}/files/{urllib.parse.quote(file_id)}",
                             params={"fields": fields + ",trashed"})
        except DriveError as error:
            if error.status == 404:
                return None
            raise

    def find_folder(self, parent, name):
        matches = self.children(parent, name=name, folders=True)
        if len(matches) > 1:
            listing = ", ".join(f"{m['id']} (created {m.get('createdTime')})" for m in matches)
            raise DuplicateFolder(f"Drive has {len(matches)} folders named {name!r} in one place: {listing}. "
                                  "Merge or rename them in Drive, then retry; nothing was changed.")
        return matches[0]["id"] if matches else None

    def ensure_folder(self, parent, name):
        existing = self.find_folder(parent, name)
        if existing:
            return existing
        created = self.json("POST", API + "/files", params={"fields": "id"},
                            body=json.dumps({"name": name, "mimeType": FOLDER_MIME,
                                             "parents": [parent]}).encode(),
                            headers={"Content-Type": "application/json; charset=UTF-8"})
        return created["id"]

    def resolve(self, parts, *, create=False, root="root"):
        """Resolve a folder path such as ["Colab-Persist", "projects", "demo"]."""
        current = root
        for part in parts:
            found = self.ensure_folder(current, part) if create else self.find_folder(current, part)
            if found is None:
                return None
            current = found
        return current

    # -- transfer ------------------------------------------------------------------
    @staticmethod
    def _verify(metadata, size, sha256, md5):
        if int(metadata.get("size", -1)) != size:
            raise DriveError(f"Drive stored {metadata.get('size')} bytes for {metadata.get('name')}, expected {size}.")
        stored = metadata.get("sha256Checksum")
        if stored:
            if stored != sha256:
                raise DriveError(f"Drive SHA-256 mismatch for {metadata.get('name')}.")
        elif metadata.get("md5Checksum") != md5:
            raise DriveError(f"Drive checksum mismatch for {metadata.get('name')}.")
        return metadata

    def upload_bytes(self, data: bytes, parent, name, *, app_properties=None, mime="application/json"):
        boundary = "colab-persist-" + hashlib.sha256(data).hexdigest()[:24]
        metadata = {"name": name, "parents": [parent], "appProperties": app_properties or {}}
        body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
                + json.dumps(metadata) + f"\r\n--{boundary}\r\nContent-Type: {mime}\r\n\r\n").encode()
        body += data + f"\r\n--{boundary}--".encode()
        result = self.json("POST", UPLOAD, params={"uploadType": "multipart", "fields": FILE_FIELDS},
                           body=body, headers={"Content-Type": f"multipart/related; boundary={boundary}"})
        return self._verify(result, len(data), hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest())

    def upload_file(self, path, parent, name, *, app_properties=None, sha256=None,
                    mime="application/octet-stream", progress=None):
        """Resumable upload of an immutable local file; returns verified metadata."""
        path = Path(path)
        size = path.stat().st_size
        local_sha, local_md5 = hashlib.sha256(), hashlib.md5()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(BLOCK_BYTES), b""):
                local_sha.update(block)
                local_md5.update(block)
        if sha256 and local_sha.hexdigest() != sha256:
            raise DriveError(f"{path.name} changed before upload; refusing to publish it.")
        for _ in range(3):  # A rare expired session restarts the whole upload.
            result = self._resumable(path, size, parent, name, app_properties, mime, progress)
            if result is not None:
                return self._verify(result, size, local_sha.hexdigest(), local_md5.hexdigest())
        raise DriveError(f"Drive kept expiring the upload session for {name}.")

    def _resumable(self, path, size, parent, name, app_properties, mime, progress):
        metadata = json.dumps({"name": name, "parents": [parent],
                               "appProperties": app_properties or {}}).encode()
        _, headers, _ = self.request("POST", UPLOAD, params={"uploadType": "resumable", "fields": FILE_FIELDS},
                                     body=metadata, headers={"Content-Type": "application/json; charset=UTF-8",
                                                             "X-Upload-Content-Type": mime,
                                                             "X-Upload-Content-Length": str(size)})
        session = headers.get("Location") or headers.get("location")
        if not session:
            raise DriveError("Drive did not return a resumable upload session.")
        offset, stalls = 0, 0
        with path.open("rb") as source:
            while True:
                if size == 0:
                    chunk, end = b"", -1
                    content_range = "bytes */0"
                else:
                    source.seek(offset)
                    chunk = source.read(min(CHUNK_BYTES, size - offset))
                    end = offset + len(chunk) - 1
                    content_range = f"bytes {offset}-{end}/{size}"
                try:
                    status, headers, payload = self.request(
                        "PUT", session, body=chunk, headers={"Content-Range": content_range},
                        ok=(200, 201, 308))
                except DriveError as error:
                    if error.status == 404:
                        return None  # Session expired before completion; restart.
                    if isinstance(error, (TokenExpired, QuotaExceeded)) or error.status not in (None, *RETRY_STATUS):
                        raise
                    status, headers, payload = self._upload_status(session, size)
                    if status == 404:
                        return None
                if status in (200, 201):
                    return json.loads(payload or b"{}")
                received = headers.get("Range") or headers.get("range")
                new_offset = int(received.rsplit("-", 1)[1]) + 1 if received else 0
                stalls = stalls + 1 if new_offset <= offset else 0
                if stalls > 5:
                    raise DriveError(f"Drive stopped accepting bytes for {name}.")
                offset = new_offset
                if progress:
                    progress(offset, size)

    def _upload_status(self, session, size):
        try:
            return self.request("PUT", session, body=b"", headers={"Content-Range": f"bytes */{size}"},
                                ok=(200, 201, 308))
        except DriveError as error:
            if error.status == 404:
                return 404, {}, b""
            raise

    def download(self, file_id, destination, *, expected_size=None, expected_sha256=None,
                 reserve_check=None):
        """Stream a file to ``destination`` with Range resume; verify size and SHA-256."""
        destination = Path(destination)
        digest, written = hashlib.sha256(), 0
        url = f"{API}/files/{urllib.parse.quote(file_id)}?alt=media"
        with destination.open("wb") as target:
            for _ in range(self.retries + 1):
                headers = {"Range": f"bytes={written}-"} if written else None
                try:
                    status, _, response = self.request("GET", url, headers=headers, ok=(200, 206), stream=True)
                    if written and status == 200:
                        # The server ignored the Range header and is sending everything again.
                        target.seek(0)
                        target.truncate()
                        digest, written = hashlib.sha256(), 0
                    with response:
                        for block in iter(lambda: response.read(BLOCK_BYTES), b""):
                            written += len(block)
                            if expected_size is not None and written > expected_size:
                                raise DriveError("Drive returned more bytes than expected.")
                            if reserve_check:
                                reserve_check(len(block))
                            target.write(block)
                            digest.update(block)
                    break
                except (urllib.error.URLError, http.client.HTTPException, socket.timeout,
                        ConnectionError, TimeoutError):
                    self.sleep(2)
            target.flush()
            os.fsync(target.fileno())
        if expected_size is not None and written != expected_size:
            raise DriveError(f"Downloaded {written} bytes, expected {expected_size}.")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise DriveError("Downloaded file failed its SHA-256 check.")
        return digest.hexdigest()

    def read_bytes(self, file_id, max_bytes=64 * 1024 * 1024):
        _, _, payload = self.request("GET", f"{API}/files/{urllib.parse.quote(file_id)}?alt=media")
        if len(payload) > max_bytes:
            raise DriveError("Drive file exceeds the allowed size for an in-memory read.")
        return payload


def relative_parts(path: str):
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError("Drive paths must be relative, without '.', '..' or empty parts.")
    return list(pure.parts)


def main(argv=None):
    """Small helper for scripts on the VM: ls, put and get under the configured Drive folder."""
    import argparse
    parser = argparse.ArgumentParser(prog="python3 -m colab_persist.drive_api",
                                     description="List, upload or download files under My Drive/<folder>/.")
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("ls")
    ls.add_argument("path", nargs="?", default="")
    put = sub.add_parser("put")
    put.add_argument("local")
    put.add_argument("remote", help="Relative path, for example datasets/corpus/v1/shard-0001.parquet")
    get = sub.add_parser("get")
    get.add_argument("remote")
    get.add_argument("local")
    args = parser.parse_args(argv)
    tokens = TokenSource.from_environment()
    drive = DriveClient(tokens)
    base = [tokens.folder()]
    if args.command == "ls":
        folder = drive.resolve(base + (relative_parts(args.path) if args.path else []))
        if folder is None:
            raise SystemExit("Not found.")
        for item in drive.children(folder):
            kind = "dir " if item["mimeType"] == FOLDER_MIME else "file"
            print(kind, item.get("size", "-"), item["name"])
    elif args.command == "put":
        parts = relative_parts(args.remote)
        parent = drive.resolve(base + parts[:-1], create=True)
        if drive.children(parent, name=parts[-1], folders=False):
            raise SystemExit(f"{args.remote} already exists in Drive; files are never overwritten.")
        result = drive.upload_file(args.local, parent, parts[-1])
        print(json.dumps({"id": result["id"], "size": int(result["size"]),
                          "sha256": result.get("sha256Checksum")}))
    else:
        parts = relative_parts(args.remote)
        parent = drive.resolve(base + parts[:-1])
        matches = drive.children(parent, name=parts[-1], folders=False) if parent else []
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one file at {args.remote}; found {len(matches)}.")
        item = matches[0]
        drive.download(item["id"], args.local, expected_size=int(item["size"]),
                       expected_sha256=item.get("sha256Checksum"))
        print(args.local)


if __name__ == "__main__":
    sys.exit(main())
