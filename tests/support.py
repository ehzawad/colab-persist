"""Backend test fixture: a private config directory and offline Colab/Drive fakes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from colab_persist import accounts, backend, colab_api, runtimes

import fakes

EMAIL = fakes.EMAIL


class BackendFixture:
    """Mixin for unittest.TestCase. Call ``use_backend()`` in setUp."""

    def use_backend(self, *, runtimes_list=None, quota_project=fakes.QUOTA_PROJECT, **config):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_dir = self.root / "config"
        self.config_dir.mkdir(mode=0o700)
        self.start_patch(patch.object(backend, "CONFIG_DIR", self.config_dir))
        credentials = self.root / "credentials.json"
        credentials.write_text(json.dumps({"email": EMAIL}))
        self.cfg = {"expected_email": EMAIL, "credentials_file": str(credentials), "session": "cuda",
                    "gpu": "L4", "ssh_identity": str(self.root / "id_test"), "drive_folder": "Colab-Persist",
                    "legacy_drive_folder": "Colab-CUDA", "storage": 2, **config}
        backend.write_config(self.cfg)
        if quota_project:
            accounts.save_meta(self.config_dir, EMAIL, quota_project=quota_project)
        self.api = fakes.FakeColabAPI(runtimes_list)
        self.drive = fakes.FakeDrive(EMAIL)
        self.api_projects = []

        def make_api(_credentials, project, **_ignored):
            self.api_projects.append(project)
            return self.api
        self.start_patch(patch.object(colab_api, "ColabAPI", side_effect=make_api))
        self.start_patch(patch.object(accounts, "verified_credentials", return_value=SimpleNamespace(email=EMAIL)))
        self.start_patch(patch.object(backend, "local_drive", side_effect=lambda _cfg: self.drive))
        backend._TOKENS.clear()
        self.addCleanup(backend._TOKENS.clear)

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def managed(self, runtime_id="cp-cuda-aaa111", spec=fakes.GPU_L4, session="cuda"):
        """Record a ready managed runtime in the ledger and on the fake account."""
        runtime = fakes.make_runtime(runtime_id, spec)
        self.api.runtimes.append(runtime)
        data = runtimes.load(self.config_dir, EMAIL)
        data[session] = {"runtime_id": runtime_id, "request_id": "req-" + runtime_id, "spec": spec,
                         "state": "ready", "created_at": "2026-11-20T00:00:00+00:00",
                         "endpoint": colab_api.endpoint_from_url(runtime["connectionInfo"]["url"])}
        runtimes._save(self.config_dir, EMAIL, data)
        return runtime

    def ledger(self, session="cuda"):
        return runtimes.entry(self.config_dir, EMAIL, session)
