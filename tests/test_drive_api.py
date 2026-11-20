"""Stdlib Drive v3 client against a scripted HTTP opener (no network)."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse

from colab_persist import drive_api

import fakes
from fakes import FakeResponse, ScriptedOpener, drive_error_body, json_body

SESSION = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=fake-session"


def query(url):
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


def metadata(data, name="file.bin", **overrides):
    value = {"id": "file-1", "name": name, "size": str(len(data)), "md5Checksum": hashlib.md5(data).hexdigest(),
             "sha256Checksum": hashlib.sha256(data).hexdigest()}
    value.update(overrides)
    return {k: v for k, v in value.items() if v is not None}


class ClientCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.sleeps = []

    def client(self, *responses, tokens=None, retries=6):
        self.opener = ScriptedOpener(*responses)
        self.addCleanup(self.opener.close)
        return drive_api.DriveClient(tokens or drive_api.TokenSource("fake-token", fakes.QUOTA_PROJECT),
                                     opener=self.opener, sleep=self.sleeps.append, retries=retries)


class TransportTests(ClientCase):
    def test_requests_carry_the_token_and_quota_project(self):
        drive = self.client((200, {}, json_body({"user": {"emailAddress": fakes.EMAIL}})))
        self.assertEqual(drive.about()["user"]["emailAddress"], fakes.EMAIL)
        sent = self.opener.requests[0]
        self.assertEqual(sent.headers["authorization"], "Bearer fake-token")
        self.assertEqual(sent.headers["x-goog-user-project"], fakes.QUOTA_PROJECT)
        self.assertTrue(sent.url.startswith(drive_api.API + "/about?"))

    def test_server_errors_are_retried_with_backoff(self):
        drive = self.client((503, {}, drive_error_body(503, "backendError")),
                            (500, {}, b"<html>oops</html>"),
                            (200, {}, json_body({"ok": True})))
        self.assertEqual(drive.json("GET", drive_api.API + "/about"), {"ok": True})
        self.assertEqual(self.sleeps, [1, 2])

    def test_retries_are_bounded(self):
        drive = self.client(*[(503, {}, drive_error_body(503, "backendError"))] * 3, retries=2)
        with self.assertRaises(drive_api.DriveError) as caught:
            drive.about()
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(len(self.opener.requests), 3)
        self.assertEqual(self.sleeps, [1, 2])

    def test_rate_limits_are_retried(self):
        drive = self.client((429, {}, drive_error_body(429, "rateLimitExceeded")),
                            (403, {}, drive_error_body(403, "userRateLimitExceeded")),
                            (200, {}, json_body({})))
        drive.about()
        self.assertEqual(len(self.sleeps), 2)

    def test_storage_quota_is_not_retried(self):
        drive = self.client((403, {}, drive_error_body(403, "storageQuotaExceeded", "The user's Drive is full.")))
        with self.assertRaises(drive_api.QuotaExceeded) as caught:
            drive.about()
        self.assertEqual((caught.exception.status, caught.exception.reason), (403, "storageQuotaExceeded"))
        self.assertIn("Drive is full", str(caught.exception))
        self.assertEqual(self.sleeps, [])
        self.assertEqual(len(self.opener.requests), 1)

    def test_other_client_errors_are_not_retried(self):
        drive = self.client((400, {}, drive_error_body(400, "invalid", "Bad query")))
        with self.assertRaisesRegex(drive_api.DriveError, "400 invalid.*Bad query"):
            drive.about()
        self.assertEqual(self.sleeps, [])

    def test_401_reloads_the_token_file_once(self):
        path = drive_api.write_token_file(self.base / "token.json", {
            "token": "old-token", "quota_project": fakes.QUOTA_PROJECT, "expires_at": time.time() + 3600})

        def rejected(_request):
            drive_api.write_token_file(path, {"token": "new-token", "quota_project": fakes.QUOTA_PROJECT,
                                              "expires_at": time.time() + 3600})
            return 401, {}, drive_error_body(401, "authError")
        drive = self.client(rejected, (200, {}, json_body({"user": {}})),
                            tokens=drive_api.TokenSource(path=path))
        drive.about()
        self.assertEqual([r.headers["authorization"] for r in self.opener.requests],
                         ["Bearer old-token", "Bearer new-token"])
        self.assertEqual(self.sleeps, [])

    def test_a_second_401_raises_token_expired(self):
        drive = self.client((401, {}, b""), (401, {}, b""))
        with self.assertRaisesRegex(drive_api.TokenExpired, "Reconnect"):
            drive.about()

    def test_network_errors_are_retried(self):
        drive = self.client(urllib.error.URLError("connection refused"), ConnectionResetError("reset"),
                            (200, {}, json_body({})))
        drive.about()
        self.assertEqual(self.sleeps, [1, 2])

    def test_network_errors_exhausted(self):
        drive = self.client(urllib.error.URLError("down"), urllib.error.URLError("down"), retries=1)
        with self.assertRaisesRegex(drive_api.DriveError, "Drive request failed"):
            drive.about()

    def test_get_returns_none_for_missing_files(self):
        drive = self.client((404, {}, drive_error_body(404, "notFound")),
                            (200, {}, json_body({"id": "x", "trashed": True})))
        self.assertIsNone(drive.get("missing"))
        self.assertTrue(drive.get("x")["trashed"])
        self.assertIn("trashed", query(self.opener.requests[1].url)["fields"])


class MetadataTests(ClientCase):
    def test_children_query_escapes_names(self):
        drive = self.client((200, {}, json_body({"files": []})), (200, {}, json_body({"files": []})))
        drive.children("parent'id", name="it's \\ here", folders=True)
        drive.children("p", folders=False)
        first, second = (query(r.url)["q"] for r in self.opener.requests)
        self.assertEqual(first, "'parent\\'id' in parents and trashed = false and name = 'it\\'s \\\\ here' "
                                f"and mimeType = '{drive_api.FOLDER_MIME}'")
        self.assertEqual(second, f"'p' in parents and trashed = false and mimeType != '{drive_api.FOLDER_MIME}'")

    def test_list_follows_pages(self):
        drive = self.client((200, {}, json_body({"files": [{"id": "a"}], "nextPageToken": "page-2"})),
                            (200, {}, json_body({"files": [{"id": "b"}]})))
        self.assertEqual([f["id"] for f in drive.list("trashed = false")], ["a", "b"])
        self.assertNotIn("pageToken", query(self.opener.requests[0].url))
        self.assertEqual(query(self.opener.requests[1].url)["pageToken"], "page-2")
        self.assertEqual(query(self.opener.requests[0].url)["orderBy"], "createdTime")

    def test_duplicate_folders_are_refused(self):
        drive = self.client((200, {}, json_body({"files": [{"id": "f1", "createdTime": "t1"},
                                                           {"id": "f2", "createdTime": "t2"}]})))
        with self.assertRaisesRegex(drive_api.DuplicateFolder, "2 folders named 'Colab-Persist'.*f1.*f2.*Merge"):
            drive.ensure_folder("root", "Colab-Persist")
        self.assertEqual([r.method for r in self.opener.requests], ["GET"])

    def test_resolve_creates_missing_folders_only_when_asked(self):
        drive = self.client((200, {}, json_body({"files": []})))
        self.assertIsNone(drive.resolve(["Colab-Persist", "projects"]))
        self.assertEqual(len(self.opener.requests), 1)
        drive = self.client((200, {}, json_body({"files": []})),
                            (200, {}, json_body({"ids": ["created-1"]})),
                            (200, {}, json_body({"id": "created-1"})),
                            (200, {}, json_body({"files": [{"id": "existing-2"}]})))
        self.assertEqual(drive.resolve(["Colab-Persist", "projects"], create=True), "existing-2")
        self.assertIn("/files/generateIds", self.opener.requests[1].url)
        post = self.opener.requests[2]
        self.assertEqual(post.method, "POST")
        # The pre-generated ID makes a retried create land on the same folder.
        self.assertEqual(json.loads(post.body), {"id": "created-1", "name": "Colab-Persist",
                                                 "mimeType": drive_api.FOLDER_MIME, "parents": ["root"]})
        self.assertIn("'created-1' in parents", query(self.opener.requests[3].url)["q"])

    def test_a_create_that_already_happened_is_not_repeated(self):
        drive = self.client((200, {}, json_body({"files": []})),
                            (200, {}, json_body({"ids": ["created-1"]})),
                            (409, {}, drive_error_body(409, "alreadyExists")),
                            (200, {}, json_body({"id": "created-1", "name": "Colab-Persist",
                                                 "mimeType": drive_api.FOLDER_MIME, "trashed": False})))
        self.assertEqual(drive.ensure_folder("root", "Colab-Persist"), "created-1")


class UploadBytesTests(ClientCase):
    def test_multipart_upload_is_verified(self):
        data = b'{"schema": 1}\n'
        drive = self.client((200, {}, json_body({"ids": ["file-1"]})), (200, {}, json_body(metadata(data, "x.json"))))
        result = drive.upload_bytes(data, "folder-1", "x.json", app_properties={"cp_kind": "manifest"})
        self.assertEqual(result["id"], "file-1")
        sent = self.opener.requests[1]
        self.assertIn(b'"id": "file-1"', sent.body)
        self.assertEqual(query(sent.url)["uploadType"], "multipart")
        self.assertTrue(sent.headers["content-type"].startswith("multipart/related; boundary="))
        self.assertIn(b'"parents": ["folder-1"]', sent.body)
        self.assertIn(b'"appProperties": {"cp_kind": "manifest"}', sent.body)
        self.assertIn(data, sent.body)

    def test_checksum_mismatch_is_an_error(self):
        data = b"payload"
        drive = self.client((200, {}, json_body({"ids": ["file-1"]})),
                            (200, {}, json_body(metadata(b"other!!", "x.json"))))
        with self.assertRaisesRegex(drive_api.DriveError, "SHA-256 mismatch"):
            drive.upload_bytes(data, "folder-1", "x.json")


class ResumableUploadTests(ClientCase):
    def setUp(self):
        super().setUp()
        chunk = patch.object(drive_api, "CHUNK_BYTES", 4)
        chunk.start()
        self.addCleanup(chunk.stop)
        self.data = b"0123456789"
        self.path = self.base / "archive.tar.gz"
        self.path.write_bytes(self.data)

    def start(self, location=SESSION):
        return 200, {"Location": location}, b""

    def done(self, **overrides):
        return 200, {}, json_body(metadata(self.data, "archive.tar.gz", **overrides))

    def ranges(self):
        return [r.headers.get("content-range") for r in self.opener.requests if r.method == "PUT"]

    def test_follows_308_ranges_including_partial_acceptance(self):
        progress = []
        drive = self.client(self.start(),
                            (308, {"Range": "bytes=0-3"}, b""),
                            (308, {"Range": "bytes=0-5"}, b""),  # Drive kept only two of four bytes.
                            self.done())
        result = drive.upload_file(self.path, "folder-1", "archive.tar.gz", app_properties={"cp_kind": "archive"},
                                   sha256=hashlib.sha256(self.data).hexdigest(), mime="application/gzip",
                                   progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(result["sha256Checksum"], hashlib.sha256(self.data).hexdigest())
        self.assertEqual(self.ranges(), ["bytes 0-3/10", "bytes 4-7/10", "bytes 6-9/10"])
        puts = [r for r in self.opener.requests if r.method == "PUT"]
        self.assertEqual([p.body for p in puts], [b"0123", b"4567", b"6789"])
        self.assertTrue(all(p.url == SESSION for p in puts))
        self.assertEqual(progress, [(4, 10), (6, 10)])
        initiation = self.opener.requests[0]
        self.assertEqual(initiation.headers["x-upload-content-length"], "10")
        self.assertEqual(initiation.headers["x-upload-content-type"], "application/gzip")
        self.assertEqual(json.loads(initiation.body)["appProperties"], {"cp_kind": "archive"})

    def test_308_without_range_restarts_from_the_beginning(self):
        drive = self.client(self.start(), (308, {}, b""), (308, {"Range": "bytes=0-3"}, b""),
                            (308, {"Range": "bytes=0-7"}, b""), self.done())
        drive.upload_file(self.path, "folder-1", "archive.tar.gz")
        self.assertEqual(self.ranges(), ["bytes 0-3/10", "bytes 0-3/10", "bytes 4-7/10", "bytes 8-9/10"])

    def test_interrupted_chunk_resumes_from_the_reported_offset(self):
        drive = self.client(self.start(),
                            (503, {}, drive_error_body(503, "backendError")),
                            (308, {"Range": "bytes=0-1"}, b""),  # Upload status query.
                            (308, {"Range": "bytes=0-5"}, b""),
                            self.done(), retries=0)
        drive.upload_file(self.path, "folder-1", "archive.tar.gz")
        self.assertEqual(self.ranges(), ["bytes 0-3/10", "bytes */10", "bytes 2-5/10", "bytes 6-9/10"])

    def test_expired_session_restarts_the_whole_upload(self):
        second = SESSION.replace("fake-session", "second-session")
        drive = self.client(self.start(), (404, {}, drive_error_body(404, "notFound")),
                            self.start(second), (308, {"Range": "bytes=0-3"}, b""),
                            (308, {"Range": "bytes=0-7"}, b""), self.done())
        drive.upload_file(self.path, "folder-1", "archive.tar.gz")
        posts = [r for r in self.opener.requests if r.method == "POST"]
        self.assertEqual(len(posts), 2)
        self.assertEqual(self.opener.requests[-1].url, second)

    def test_a_file_changed_before_upload_is_refused_without_requests(self):
        drive = self.client()
        with self.assertRaisesRegex(drive_api.DriveError, "changed before upload"):
            drive.upload_file(self.path, "folder-1", "archive.tar.gz", sha256="0" * 64)
        self.assertEqual(self.opener.requests, [])

    def test_size_reported_by_drive_must_match(self):
        drive = self.client(self.start(), (308, {"Range": "bytes=0-3"}, b""), (308, {"Range": "bytes=0-7"}, b""),
                            self.done(size="9"))
        with self.assertRaisesRegex(drive_api.DriveError, "stored 9 bytes.*expected 10"):
            drive.upload_file(self.path, "folder-1", "archive.tar.gz")

    def test_md5_is_used_when_drive_reports_no_sha256(self):
        responses = [self.start(), (308, {"Range": "bytes=0-3"}, b""), (308, {"Range": "bytes=0-7"}, b"")]
        drive = self.client(*responses, self.done(sha256Checksum=None))
        drive.upload_file(self.path, "folder-1", "archive.tar.gz")
        drive = self.client(*responses, self.done(sha256Checksum=None, md5Checksum="0" * 32))
        with self.assertRaisesRegex(drive_api.DriveError, "checksum mismatch"):
            drive.upload_file(self.path, "folder-1", "archive.tar.gz")

    def test_empty_file(self):
        empty = self.base / "empty"
        empty.write_bytes(b"")
        drive = self.client(self.start(), (200, {}, json_body(metadata(b"", "empty"))))
        drive.upload_file(empty, "folder-1", "empty")
        self.assertEqual(self.ranges(), ["bytes */0"])

    def test_quota_and_token_errors_mid_upload_are_raised(self):
        drive = self.client(self.start(), (403, {}, drive_error_body(403, "storageQuotaExceeded")))
        with self.assertRaises(drive_api.QuotaExceeded):
            drive.upload_file(self.path, "folder-1", "archive.tar.gz")
        drive = self.client(self.start(), (401, {}, b""), (401, {}, b""))
        with self.assertRaises(drive_api.TokenExpired):
            drive.upload_file(self.path, "folder-1", "archive.tar.gz")

    def test_missing_session_location(self):
        drive = self.client((200, {}, b""))
        with self.assertRaisesRegex(drive_api.DriveError, "resumable upload session"):
            drive.upload_file(self.path, "folder-1", "archive.tar.gz")

    def test_drive_that_stops_accepting_bytes(self):
        drive = self.client(self.start(), *[(308, {"Range": "bytes=0-3"}, b"")] * 8)
        with self.assertRaisesRegex(drive_api.DriveError, "stopped accepting bytes"):
            drive.upload_file(self.path, "folder-1", "archive.tar.gz")


class DownloadTests(ClientCase):
    def setUp(self):
        super().setUp()
        self.data = b"0123456789"
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.target = self.base / "download"

    def test_complete_download_is_verified(self):
        drive = self.client(FakeResponse(200, body=self.data))
        self.assertEqual(drive.download("id-1", self.target, expected_size=10, expected_sha256=self.sha), self.sha)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertNotIn("range", self.opener.requests[0].headers)
        self.assertTrue(self.opener.requests[0].url.endswith("/files/id-1?alt=media"))

    def test_dropped_connection_resumes_with_a_range_request(self):
        drive = self.client(FakeResponse(200, chunks=[self.data[:4], ConnectionResetError("reset")]),
                            FakeResponse(206, body=self.data[4:]))
        drive.download("id-1", self.target, expected_size=10, expected_sha256=self.sha)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(self.opener.requests[1].headers["range"], "bytes=4-")
        self.assertEqual(self.sleeps, [2])

    def test_full_reply_to_a_range_resume_restarts_cleanly(self):
        drive = self.client(FakeResponse(200, chunks=[self.data[:4], ConnectionResetError("reset")]),
                            FakeResponse(200, body=self.data))
        drive.download("id-1", self.target, expected_size=10, expected_sha256=self.sha)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_checksum_mismatch(self):
        drive = self.client(FakeResponse(200, body=self.data))
        with self.assertRaisesRegex(drive_api.DriveError, "SHA-256"):
            drive.download("id-1", self.target, expected_size=10, expected_sha256="0" * 64)

    def test_short_and_long_downloads(self):
        drive = self.client(FakeResponse(200, body=self.data[:3]), FakeResponse(206, body=self.data[3:]))
        self.assertEqual(drive.download("id-1", self.target, expected_size=10, expected_sha256=self.sha), self.sha)
        self.assertEqual(self.opener.requests[1].headers.get("range"), "bytes=3-")
        drive = self.client(*[FakeResponse(206 if i else 200, body=self.data[:3] if i == 0 else b"")
                              for i in range(drive.retries + 1)])
        with self.assertRaisesRegex(drive_api.DriveError, "Downloaded 3 bytes, expected 10"):
            drive.download("id-1", self.target, expected_size=10)
        drive = self.client(FakeResponse(200, body=self.data + b"extra"))
        with self.assertRaisesRegex(drive_api.DriveError, "more bytes than expected"):
            drive.download("id-1", self.target, expected_size=10)

    def test_reserve_check_sees_each_block_and_can_abort(self):
        seen = []
        drive = self.client(FakeResponse(200, body=self.data))
        drive.download("id-1", self.target, reserve_check=seen.append)
        self.assertEqual(seen, [10])

        def full(_size):
            raise RuntimeError("Insufficient local disk")
        drive = self.client(FakeResponse(200, body=self.data))
        with self.assertRaisesRegex(RuntimeError, "Insufficient local disk"):
            drive.download("id-1", self.target, reserve_check=full)

    def test_read_bytes_is_bounded(self):
        drive = self.client((200, {}, b"x" * 11))
        with self.assertRaisesRegex(drive_api.DriveError, "allowed size"):
            drive.read_bytes("id-1", max_bytes=10)


class TokenTests(ClientCase):
    def test_static_token_source(self):
        self.assertEqual(drive_api.TokenSource("t", "p").get(), ("t", "p"))
        for token, quota in ((None, "p"), ("t", None)):
            with self.assertRaises(drive_api.TokenExpired):
                drive_api.TokenSource(token, quota).get()

    def test_file_token_source_is_reread_and_checked(self):
        path = self.base / "token.json"
        source = drive_api.TokenSource(path=path)
        with self.assertRaisesRegex(drive_api.TokenExpired, "not connected on this VM"):
            source.get()
        drive_api.write_token_file(path, {"token": "a", "quota_project": "p", "expires_at": time.time() + 600,
                                          "folder": "Custom"})
        self.assertEqual(source.get(), ("a", "p"))
        self.assertEqual(source.folder(), "Custom")
        drive_api.write_token_file(path, {"token": "b", "quota_project": "p"})
        self.assertEqual(source.get(), ("b", "p"))
        drive_api.write_token_file(path, {"token": "c", "quota_project": "p", "expires_at": time.time() + 10})
        with self.assertRaisesRegex(drive_api.TokenExpired, "expired"):
            source.get()
        drive_api.write_token_file(path, {"token": "d"})
        with self.assertRaisesRegex(drive_api.TokenExpired, "incomplete"):
            source.get()

    def test_folder_defaults(self):
        with patch.dict(os.environ, {drive_api.FOLDER_ENV: "From-Env"}):
            self.assertEqual(drive_api.TokenSource(path=self.base / "missing.json").folder(), "From-Env")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(drive_api.TokenSource("t", "p").folder(), "Colab-Persist")
            self.assertEqual(drive_api.TokenSource.from_environment().path, Path(drive_api.DEFAULT_TOKEN_FILE))

    def test_token_file_is_private_and_atomic(self):
        path = drive_api.write_token_file(self.base / "shm" / "colab-persist" / "drive-token.json",
                                          {"token": "t", "quota_project": "p"})
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        self.assertEqual([p.name for p in path.parent.iterdir()], ["drive-token.json"])

    def test_relative_parts(self):
        self.assertEqual(drive_api.relative_parts("datasets/corpus/v1/a.parquet"),
                         ["datasets", "corpus", "v1", "a.parquet"])
        for bad in ("/abs", "../up", "a/../b", "a//b", "./a", "a/"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                drive_api.relative_parts(bad)


class HelperCommandTests(unittest.TestCase):
    """python -m colab_persist.drive_api ls/put/get on the VM."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.drive = fakes.FakeDrive()
        for item in (patch.object(drive_api, "DriveClient", return_value=self.drive),
                     patch.object(drive_api.TokenSource, "from_environment", return_value=fakes.FakeTokens())):
            item.start()
            self.addCleanup(item.stop)

    def run_main(self, *argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            drive_api.main(list(argv))
        return output.getvalue()

    def test_put_get_and_ls_never_overwrite(self):
        local = self.base / "shard.bin"
        local.write_bytes(b"shard-data")
        result = json.loads(self.run_main("put", str(local), "datasets/corpus/shard.bin"))
        self.assertEqual(result["size"], 10)
        self.assertEqual(result["sha256"], hashlib.sha256(b"shard-data").hexdigest())
        with self.assertRaisesRegex(SystemExit, "already exists"):
            self.run_main("put", str(local), "datasets/corpus/shard.bin")
        self.assertIn("file 10 shard.bin", self.run_main("ls", "datasets/corpus"))
        self.assertIn("dir  - corpus", self.run_main("ls", "datasets"))
        copy = self.base / "copy.bin"
        self.run_main("get", "datasets/corpus/shard.bin", str(copy))
        self.assertEqual(copy.read_bytes(), b"shard-data")
        with self.assertRaisesRegex(SystemExit, "found 0"):
            self.run_main("get", "datasets/corpus/missing.bin", str(copy))
        with self.assertRaisesRegex(SystemExit, "Not found"):
            self.run_main("ls", "nothing-here")
        with self.assertRaises(ValueError):
            self.run_main("put", str(local), "../escape.bin")


if __name__ == "__main__":
    unittest.main()
