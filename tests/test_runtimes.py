"""Private per-account ledger of the runtime generations this tool created."""
import json
from pathlib import Path
import stat
import tempfile
import unittest

from colab_persist import accounts, colab_api, runtimes

import fakes

EMAIL = "first@example.com"


class LedgerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Path(temporary.name)

    def test_runtime_ids_are_valid_colab_ids(self):
        for session in ("cuda", "Training_Run-2", "9lives", "-", "x" * 64):
            with self.subTest(session=session):
                runtime_id = runtimes.new_runtime_id(session)
                self.assertTrue(colab_api.RUNTIME_ID.fullmatch(runtime_id), runtime_id)
                self.assertTrue(runtime_id.startswith("cp-"))
                self.assertLessEqual(len(runtime_id), 63)
        self.assertRegex(runtimes.new_runtime_id("Training_Run"), r"^cp-training-run-[0-9a-f]{6}$")
        self.assertRegex(runtimes.new_runtime_id("9lives"), r"^cp-s9lives-[0-9a-f]{6}$")
        self.assertRegex(runtimes.new_runtime_id("___"), r"^cp-session-[0-9a-f]{6}$")
        self.assertNotEqual(runtimes.new_runtime_id("cuda"), runtimes.new_runtime_id("cuda"))

    def test_begin_records_a_pending_generation_and_resumes_it(self):
        self.assertEqual(runtimes.load(self.config, EMAIL), {})
        first = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        self.assertEqual(first["state"], "pending")
        self.assertEqual(first["spec"], fakes.GPU_L4)
        again = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        self.assertEqual((again["runtime_id"], again["request_id"]), (first["runtime_id"], first["request_id"]))
        # A pending creation may still complete server-side, so it is never silently replaced.
        with self.assertRaisesRegex(RuntimeError, "still pending"):
            runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_T4)
        self.assertEqual(runtimes.entry(self.config, EMAIL, "cuda")["runtime_id"], first["runtime_id"])
        runtimes.forget(self.config, EMAIL, "cuda", first["runtime_id"])
        other = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_T4)
        self.assertNotEqual(other["runtime_id"], first["runtime_id"])
        self.assertEqual(runtimes.entry(self.config, EMAIL, "cuda")["spec"], fakes.GPU_T4)

    def test_ready_generation_blocks_a_second_begin(self):
        record = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        ready = runtimes.mark_ready(self.config, EMAIL, "cuda", record["runtime_id"], "m-s-endpoint")
        self.assertEqual((ready["state"], ready["endpoint"]), ("ready", "m-s-endpoint"))
        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        # Other sessions are independent.
        self.assertEqual(runtimes.begin(self.config, EMAIL, "other", fakes.GPU_L4)["state"], "pending")

    def test_mark_ready_refuses_another_generation(self):
        runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        with self.assertRaisesRegex(RuntimeError, "refusing to adopt another runtime"):
            runtimes.mark_ready(self.config, EMAIL, "cuda", "cp-cuda-other0", "m-s-x")
        with self.assertRaisesRegex(RuntimeError, "refusing to adopt"):
            runtimes.mark_ready(self.config, EMAIL, "never-started", "cp-x-000000", "m-s-x")
        self.assertEqual(runtimes.entry(self.config, EMAIL, "cuda")["state"], "pending")

    def test_forget_removes_only_the_exact_generation(self):
        record = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        runtimes.forget(self.config, EMAIL, "cuda", "cp-cuda-replacement")
        self.assertEqual(runtimes.entry(self.config, EMAIL, "cuda")["runtime_id"], record["runtime_id"])
        runtimes.forget(self.config, EMAIL, "cuda", record["runtime_id"])
        self.assertIsNone(runtimes.entry(self.config, EMAIL, "cuda"))
        runtimes.forget(self.config, EMAIL, "cuda", record["runtime_id"])  # Idempotent.

    def test_ledger_is_private_per_account_and_names_no_email(self):
        runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        path = runtimes.ledger_path(self.config, EMAIL)
        self.assertEqual(path.parent, accounts.account_dir(self.config, EMAIL))
        self.assertNotIn("example.com", str(path))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["runtimes.json", "runtimes.lock"])
        self.assertEqual(list(json.loads(path.read_text())), ["cuda"])
        self.assertEqual(runtimes.load(self.config, "second@example.com"), {})
        self.assertEqual(runtimes.ledger_path(self.config, "FIRST@example.com"), path)


if __name__ == "__main__":
    unittest.main()


class AbandonedCreationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Path(temporary.name)

    def test_abandoned_creation_keeps_its_id_and_can_be_adopted(self):
        record = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        runtimes.abandon(self.config, EMAIL, "cuda", record["runtime_id"])
        self.assertIsNone(runtimes.entry(self.config, EMAIL, "cuda"))
        self.assertIn(record["runtime_id"], runtimes.abandoned(self.config, EMAIL))
        adopted = runtimes.adopt(self.config, EMAIL, "cuda", record["runtime_id"], "m-s-x")
        self.assertEqual((adopted["state"], adopted["runtime_id"]), ("ready", record["runtime_id"]))
        self.assertEqual(runtimes.abandoned(self.config, EMAIL), {})

    def test_only_recorded_ids_can_be_adopted(self):
        with self.assertRaisesRegex(RuntimeError, "Only runtimes this tool created"):
            runtimes.adopt(self.config, EMAIL, "cuda", "cp-cuda-zzz999", "m-s-x")

    def test_stale_abandoned_ids_are_pruned_but_present_ones_kept(self):
        first = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        runtimes.abandon(self.config, EMAIL, "cuda", first["runtime_id"])
        second = runtimes.begin(self.config, EMAIL, "cuda", fakes.GPU_L4)
        runtimes.abandon(self.config, EMAIL, "cuda", second["runtime_id"])
        runtimes.prune_abandoned(self.config, EMAIL, {second["runtime_id"]}, max_age_hours=-1)
        self.assertEqual(set(runtimes.abandoned(self.config, EMAIL)), {second["runtime_id"]})
