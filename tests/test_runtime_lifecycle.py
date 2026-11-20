"""Managed runtime resolution through the ledger and the official API list."""
import json
import unittest
from unittest.mock import patch

from colab_persist import backend, colab_api, runtimes

import fakes
from support import EMAIL, BackendFixture


class StartRuntimeTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.install = self.start_patch(patch.object(backend, "install_engine"))
        self.alias = self.start_patch(patch.object(backend, "write_ssh_alias"))
        self.remote = self.start_patch(patch.object(backend, "remote_call",
                                                    return_value={"drive_connected": True}))

    def test_new_runtime_is_recorded_before_creation_and_marked_ready(self):
        recorded = []
        original = self.api.create_runtime

        def create(runtime_id, request_id, spec, **kwargs):
            recorded.append(runtimes.entry(self.config_dir, EMAIL, "cuda"))
            return original(runtime_id, request_id, spec, **kwargs)
        self.api.create_runtime = create
        result = backend.start_runtime()
        self.assertEqual(recorded[0]["state"], "pending")
        runtime_id, request_id, spec = self.api.created[0]
        self.assertEqual((runtime_id, request_id), (recorded[0]["runtime_id"], recorded[0]["request_id"]))
        self.assertEqual(spec, fakes.GPU_L4)
        ledger = self.ledger()
        self.assertEqual((ledger["state"], ledger["endpoint"]), ("ready", f"m-s-{runtime_id}"))
        self.assertEqual(result["runtime"], runtime_id)
        self.assertEqual(result["gpu"], "L4")
        self.assertTrue(result["created"])
        self.assertTrue(result["drive_connected"])
        self.install.assert_called_once()
        self.alias.assert_called_once()
        self.remote.assert_called_once_with("status", item=self.api.runtimes[-1])
        self.assertEqual(self.api_projects[0], fakes.QUOTA_PROJECT)

    def test_hardware_choices_select_the_matching_spec(self):
        for kwargs, spec in ((dict(gpu="T4"), fakes.GPU_T4), (dict(cpu=True), fakes.CPU),
                             (dict(tpu="v5e-1"), fakes.TPU_V5E1)):
            with self.subTest(kwargs=kwargs):
                runtimes.forget(self.config_dir, EMAIL, "cuda", (self.ledger() or {}).get("runtime_id"))
                self.api.runtimes = []
                backend.start_runtime(**kwargs)
                self.assertEqual(self.api.created[-1][2], spec)

    def test_configured_default_gpu_is_used(self):
        cfg = json.loads((self.config_dir / "config.json").read_text())
        backend.write_config({**cfg, "gpu": "T4"})
        backend.start_runtime()
        self.assertEqual(self.api.created[0][2], fakes.GPU_T4)

    def test_ineligible_hardware_allocates_nothing(self):
        with self.assertRaisesRegex(ValueError, "not eligible for A100"):
            backend.start_runtime(gpu="A100")
        self.assertEqual(self.api.created, [])
        self.assertIsNone(self.ledger())

    def test_drive_is_confirmed_before_spending_compute_units(self):
        self.drive.email = "someone-else@example.com"
        with self.assertRaisesRegex(RuntimeError, "Drive account mismatch"):
            backend.start_runtime()
        self.assertEqual(self.api.created, [])
        self.assertIsNone(self.ledger())

    def test_existing_managed_runtime_is_reused(self):
        runtime = self.managed()
        result = backend.start_runtime()
        self.assertFalse(result["created"])
        self.assertEqual(result["runtime"], "cp-cuda-aaa111")
        self.assertEqual(self.api.created, [])
        self.install.assert_called_once_with(runtime)

    def test_existing_runtime_with_other_hardware_is_not_replaced(self):
        self.managed(spec=fakes.GPU_T4)
        with self.assertRaisesRegex(RuntimeError, "Runtime already uses T4. Save and stop it before selecting L4"):
            backend.start_runtime()
        self.assertEqual(self.api.created, [])
        self.install.assert_not_called()

    def test_interrupted_creation_resumes_with_the_same_ids(self):
        pending = runtimes.begin(self.config_dir, EMAIL, "cuda", fakes.GPU_L4)
        backend.start_runtime()
        self.assertEqual(self.api.created, [(pending["runtime_id"], pending["request_id"], fakes.GPU_L4)])

    def test_failed_creation_is_abandoned_and_adopted_if_it_appears(self):
        def refuse(runtime_id, request_id, spec, **_):
            self.api.created.append((runtime_id, request_id, spec))
            raise colab_api.ColabAPIError("rate limited", 429, "RESOURCE_EXHAUSTED")
        with patch.object(self.api, "create_runtime", side_effect=refuse):
            with self.assertRaisesRegex(colab_api.ColabAPIError, "rate limited"):
                backend.start_runtime()
        runtime_id = self.api.created[-1][0]
        self.assertIsNone(runtimes.entry(self.config_dir, EMAIL, "cuda"))
        self.assertIn(runtime_id, runtimes.abandoned(self.config_dir, EMAIL))
        # The VM turns up after all: it is ours by its exact ID, so it becomes the managed runtime.
        self.api.runtimes.append(fakes.make_runtime(runtime_id, fakes.GPU_L4))
        runtime, _, _ = backend.managed_runtime(backend.config(), self.api)
        self.assertEqual(colab_api.runtime_id_of(runtime), runtime_id)
        self.assertEqual(runtimes.entry(self.config_dir, EMAIL, "cuda")["state"], "ready")

    def test_a_listing_miss_is_confirmed_by_id_before_forgetting(self):
        runtime = self.managed()
        self.api.list_error = None
        with patch.object(self.api, "list_runtimes", return_value=[]):
            found, _, _ = backend.managed_runtime(backend.config(), self.api)
        self.assertEqual(colab_api.runtime_id_of(found), colab_api.runtime_id_of(runtime))
        self.assertEqual(runtimes.entry(self.config_dir, EMAIL, "cuda")["state"], "ready")

    def test_a_runtime_without_connection_details_stays_pending(self):
        record = runtimes.begin(self.config_dir, EMAIL, "cuda", fakes.GPU_L4)
        starting = fakes.make_runtime(record["runtime_id"], fakes.GPU_L4)
        starting.pop("connectionInfo", None)
        self.api.runtimes.append(starting)
        found, _, pending = backend.managed_runtime(backend.config(), self.api)
        self.assertIsNone(found)
        self.assertEqual(pending["runtime_id"], record["runtime_id"])
        self.assertEqual(runtimes.entry(self.config_dir, EMAIL, "cuda")["state"], "pending")

    def test_unmanaged_runtimes_are_never_adopted(self):
        self.api.runtimes.append(fakes.make_runtime("browser-notebook-1", fakes.GPU_L4))
        result = backend.start_runtime()
        self.assertTrue(result["created"])
        self.assertNotEqual(result["runtime"], "browser-notebook-1")

    def test_conflicting_hardware_choices_allocate_nothing(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            backend.start_runtime(gpu="T4", cpu=True)
        self.assertEqual(self.api.created, [])


class StatusTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.remote = self.start_patch(patch.object(backend, "remote_call",
                                                    return_value={"drive_connected": True, "workspaces": {}}))

    def test_external_running_runtime_is_reported_without_adoption(self):
        self.api.runtimes.append(fakes.make_runtime("browser-notebook-1"))
        result = backend.status()
        self.assertFalse(result["active"])
        self.assertEqual(result["active_runtimes"], 1)
        self.assertEqual(result["next"], "colab-persist sessions")
        self.assertEqual(result["account"], EMAIL)
        self.assertEqual(result["drive_folder"], "My Drive/Colab-Persist")
        self.remote.assert_not_called()
        self.assertNotIn("fake-runtime-proxy-token", json.dumps(result))

    def test_no_runtime_and_pending_creation_suggest_start(self):
        self.assertEqual(backend.status()["next"], "colab-persist start")
        runtimes.begin(self.config_dir, EMAIL, "cuda", fakes.GPU_L4)
        self.api.runtimes.append(fakes.make_runtime("browser-notebook-1"))
        result = backend.status()
        self.assertTrue(result["pending_creation"])
        self.assertEqual(result["next"], "colab-persist start")

    def test_managed_runtime_reports_vm_status(self):
        self.managed(spec=fakes.GPU_T4)
        result = backend.status()
        self.assertTrue(result["active"])
        self.assertEqual((result["runtime"], result["gpu"], result["session"]), ("cp-cuda-aaa111", "T4", "cuda"))
        self.assertTrue(result["drive_connected"])
        self.assertNotIn("fake-runtime-proxy-token", json.dumps(result))

    def test_sessions_distinguish_managed_and_external_runtimes(self):
        self.managed()
        self.api.runtimes.append(fakes.make_runtime("browser-notebook-1", fakes.CPU))
        result = backend.list_sessions()
        self.assertEqual(result["active_runtimes"], 2)
        self.assertEqual(result["sessions"], [
            {"runtime": "cp-cuda-aaa111", "spec": "L4", "managed": True, "name": "cuda"},
            {"runtime": "browser-notebook-1", "spec": "CPU", "managed": False, "name": None}])
        self.assertNotIn("fake-runtime-proxy-token", json.dumps(result))

    def test_specs_list_eligibility_and_tier(self):
        self.api.tier = "SUBSCRIPTION_TIER_PRO_PLUS"
        result = backend.specs()
        self.assertEqual(result["tier"], "pro_plus")
        self.assertEqual(result["eligible"], ["CPU", "L4", "T4", "V5E1"])
        self.assertEqual(result["not_eligible"], ["A100"])
        self.assertEqual(result["default_gpu"], "L4")

    def test_other_sessions_ledgers_are_independent(self):
        self.managed(runtime_id="cp-other-bbb222", session="other")
        self.assertIsNone(backend.runtime_info())
        self.assertEqual(runtimes.entry(self.config_dir, EMAIL, "other")["runtime_id"], "cp-other-bbb222")


if __name__ == "__main__":
    unittest.main()
