"""Official Colab REST v1beta client: specs, idempotent creation, error mapping, Service Usage."""
import json
import unittest

import requests

from colab_persist import colab_api

import fakes
from fakes import FakeColabSession, colab_error

QUOTA = fakes.QUOTA_PROJECT


def api_with(*responses):
    session = FakeColabSession(*responses)
    sleeps = []
    return colab_api.ColabAPI(None, QUOTA, session=session, sleep=sleeps.append), session, sleeps


class SpecTests(unittest.TestCase):
    def test_wanted_spec(self):
        self.assertEqual(colab_api.wanted_spec(), fakes.GPU_L4)
        self.assertEqual(colab_api.wanted_spec(gpu="t4"), fakes.GPU_T4)
        self.assertEqual(colab_api.wanted_spec(gpu="A100", high_mem=True)["shape"], "SHAPE_HIGHMEM")
        self.assertEqual(colab_api.wanted_spec(cpu=True), fakes.CPU)
        self.assertEqual(colab_api.wanted_spec(tpu="v5e-1"), fakes.TPU_V5E1)
        for choice in (dict(gpu="T4", cpu=True), dict(gpu="T4", tpu="V5E1"), dict(tpu="V5E1", cpu=True)):
            with self.subTest(choice=choice), self.assertRaisesRegex(ValueError, "only one"):
                colab_api.wanted_spec(**choice)

    def test_spec_labels(self):
        self.assertEqual(colab_api.spec_label(fakes.GPU_L4), "L4")
        self.assertEqual(colab_api.spec_label({**fakes.GPU_A100, "shape": "SHAPE_HIGHMEM"}), "A100 high-mem")
        self.assertEqual(colab_api.spec_label(fakes.CPU), "CPU")
        self.assertEqual(colab_api.spec_label(fakes.TPU_V5E1), "V5E1")

    def test_choose_spec_returns_the_eligible_key(self):
        self.assertEqual(colab_api.choose_spec(fakes.default_specs(), fakes.GPU_T4), fakes.GPU_T4)
        reordered = {"shape": "SHAPE_STANDARD", "accelerator": "L4", "variant": "VARIANT_GPU"}
        self.assertEqual(colab_api.choose_spec(fakes.default_specs(), reordered), fakes.GPU_L4)

    def test_ineligible_or_unknown_specs_name_the_eligible_ones(self):
        with self.assertRaisesRegex(ValueError, "not eligible for A100. Eligible: CPU, L4, T4, V5E1. "
                                                "Nothing was allocated"):
            colab_api.choose_spec(fakes.default_specs(), fakes.GPU_A100)
        with self.assertRaisesRegex(ValueError, "does not offer H100. Eligible for this account: CPU, L4"):
            colab_api.choose_spec(fakes.default_specs(), colab_api.wanted_spec(gpu="H100"))
        with self.assertRaisesRegex(ValueError, "Eligible for this account: none"):
            colab_api.choose_spec([], fakes.GPU_L4)

    def test_endpoint_and_runtime_ids(self):
        self.assertEqual(colab_api.endpoint_from_url("https://8080-m-s-xyz-c.region.prod.r.colab.dev"), "m-s-xyz")
        self.assertEqual(colab_api.endpoint_from_url("https://8080-m-s-1a2b-3c-c.us-east1.prod.r.colab.dev/path"),
                         "m-s-1a2b-3c")
        self.assertEqual(colab_api.endpoint_from_url("https://plain-host.colab.dev"), "plain-host")
        self.assertEqual(colab_api.runtime_id_of({"name": "runtimes/cp-cuda-abc123"}), "cp-cuda-abc123")


class ClientTests(unittest.TestCase):
    def test_quota_project_is_required_and_sent(self):
        with self.assertRaisesRegex(colab_api.ColabAPIError, "No quota project.*setup --project"):
            colab_api.ColabAPI(None, None, session=FakeColabSession())
        api, session, _ = api_with((200, {"runtimes": [fakes.make_runtime("cp-cuda-abc123")]}))
        self.assertEqual(session.headers["x-goog-user-project"], QUOTA)
        self.assertEqual(len(api.list_runtimes()), 1)
        call = session.calls[0]
        self.assertEqual((call["method"], call["url"]), ("GET", colab_api.BASE + "runtimes"))
        self.assertIsNone(call["headers"])

    def test_simple_reads_and_defaults(self):
        api, session, _ = api_with((200, {"tier": "SUBSCRIPTION_TIER_PRO"}), (200, {}), (200, None),
                                   (200, {"runtimeSpecs": fakes.default_specs()}))
        self.assertEqual(api.subscription(), "SUBSCRIPTION_TIER_PRO")
        self.assertEqual(api.list_runtimes(), [])
        self.assertEqual(api.subscription(), "SUBSCRIPTION_TIER_UNSPECIFIED")
        self.assertEqual(len(api.runtime_specs()), 5)

    def test_get_runtime_maps_404_to_none(self):
        api, _, _ = api_with(colab_error(404, "NOT_FOUND"), colab_error(500, "INTERNAL", "boom"))
        self.assertIsNone(api.get_runtime("cp-gone-000000"))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "boom"):
            api.get_runtime("cp-error-000000")

    def test_delete_runtime(self):
        api, session, _ = api_with((200, {}), (200, {}))
        api.delete_runtime("cp-cuda-abc123")
        api.delete_runtime("cp-cuda-abc123", allow_missing=True)
        self.assertEqual(session.calls[0]["method"], "DELETE")
        self.assertEqual(session.calls[0]["url"], colab_api.BASE + "runtimes/cp-cuda-abc123")
        self.assertIsNone(session.calls[0]["params"])
        self.assertEqual(session.calls[1]["params"], {"allowMissing": "true"})

    def test_quota_project_errors_explain_setup(self):
        for response in (colab_error(403, "PERMISSION_DENIED", "API has not been used", reason="SERVICE_DISABLED"),
                         colab_error(403, "PERMISSION_DENIED", "This API requires a quota project")):
            api, _, _ = api_with(response)
            with self.subTest(response=response):
                with self.assertRaisesRegex(colab_api.ColabAPIError, "colab-persist setup --project") as caught:
                    api.list_runtimes()
                self.assertEqual(caught.exception.status, 403)

    def test_non_json_error_body(self):
        api, _, _ = api_with((502, b"<html>bad gateway</html>"))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "HTTP 502"):
            api.list_runtimes()


class CreateRuntimeTests(unittest.TestCase):
    RUNTIME = "cp-cuda-abc123"
    REQUEST = "11111111-2222-3333-4444-555555555555"

    NOT_FOUND = colab_error(404, "NOT_FOUND")

    def test_waits_by_reading_the_runtime_and_reposts_rarely(self):
        runtime = fakes.make_runtime(self.RUNTIME, fakes.GPU_L4)
        api, session, sleeps = api_with((200, {"name": "operations/x", "done": False}), self.NOT_FOUND,
                                        self.NOT_FOUND, (200, {"done": True, "response": runtime}))
        self.assertEqual(api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4, poll_seconds=7,
                                            repost_every=2), runtime)
        self.assertEqual(sleeps, [7, 7])
        self.assertEqual([c["method"] for c in session.calls], ["POST", "GET", "GET", "POST"])
        for call in (session.calls[0], session.calls[3]):
            self.assertEqual(call["params"], {"runtimeId": self.RUNTIME, "requestId": self.REQUEST})
            self.assertEqual(call["json"], {"runtimeSpec": fakes.GPU_L4})

    def test_done_without_connection_info_reads_the_runtime(self):
        runtime = fakes.make_runtime(self.RUNTIME)
        pending = {"name": "runtimes/" + self.RUNTIME, "runtimeSpec": fakes.GPU_L4}
        api, session, sleeps = api_with((200, {"done": True}), (200, pending), (200, runtime))
        self.assertEqual(api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4), runtime)
        self.assertEqual([c["method"] for c in session.calls], ["POST", "GET", "GET"])
        self.assertEqual(len(sleeps), 1)

    def test_transient_errors_keep_waiting(self):
        runtime = fakes.make_runtime(self.RUNTIME)
        api, _, sleeps = api_with(colab_error(503, "UNAVAILABLE"), colab_error(503, "UNAVAILABLE"),
                                  colab_error(504, "DEADLINE_EXCEEDED"), self.NOT_FOUND,
                                  colab_error(400, "DEADLINE_EXCEEDED"), self.NOT_FOUND,
                                  (200, {"done": True, "response": runtime}))
        self.assertEqual(api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4, repost_every=1), runtime)
        self.assertEqual(len(sleeps), 3)

    def test_a_refused_repeat_never_means_nothing_was_created(self):
        # Live regression: the first POST started a VM; a rate-limited repeat must not abandon it.
        runtime = fakes.make_runtime(self.RUNTIME)
        api, session, _ = api_with((200, {"done": False}), self.NOT_FOUND,
                                   colab_error(429, "RESOURCE_EXHAUSTED"), (200, runtime))
        self.assertEqual(api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4, repost_every=1), runtime)
        self.assertEqual([c["method"] for c in session.calls], ["POST", "GET", "POST", "GET"])

    def test_resume_reads_the_runtime_before_posting(self):
        runtime = fakes.make_runtime(self.RUNTIME)
        api, session, _ = api_with((200, runtime))
        self.assertEqual(api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4, resume=True), runtime)
        self.assertEqual([c["method"] for c in session.calls], ["GET"])

    def test_already_exists_with_the_same_spec_is_adopted(self):
        runtime = fakes.make_runtime(self.RUNTIME, fakes.GPU_L4)
        api, session, _ = api_with(colab_error(409, "ALREADY_EXISTS"), (200, runtime))
        self.assertEqual(api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4), runtime)
        self.assertEqual(session.calls[1]["url"], colab_api.BASE + "runtimes/" + self.RUNTIME)

    def test_already_exists_with_another_spec_is_refused(self):
        api, _, _ = api_with(colab_error(409, "ALREADY_EXISTS", "exists"),
                             (200, fakes.make_runtime(self.RUNTIME, fakes.GPU_T4)))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "exists"):
            api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4)

    def test_failed_precondition_and_rate_limits_explain_that_nothing_was_created(self):
        api, _, _ = api_with(colab_error(400, "FAILED_PRECONDITION", "Insufficient compute units"))
        with self.assertRaisesRegex(colab_api.ColabAPIError,
                                    "could not allocate this runtime now.*Insufficient compute units") as caught:
            api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4)
        self.assertEqual(caught.exception.code, "FAILED_PRECONDITION")
        api, _, _ = api_with(colab_error(429, "RESOURCE_EXHAUSTED"))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "rate-limiting runtime creation; nothing was created"):
            api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4)

    def test_operation_error_and_other_errors(self):
        api, _, _ = api_with((200, {"done": True, "error": {"code": 13, "message": "VM boot failed"}}))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "Runtime creation failed: VM boot failed"):
            api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4)
        api, _, _ = api_with(colab_error(403, "PERMISSION_DENIED", "denied"))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "denied"):
            api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4)

    def test_deadline_tells_the_user_to_resume(self):
        api, _, sleeps = api_with((200, {"done": False}), self.NOT_FOUND)
        with self.assertRaisesRegex(colab_api.ColabAPIError, "still pending.*nothing extra will be allocated") as caught:
            api.create_runtime(self.RUNTIME, self.REQUEST, fakes.GPU_L4, deadline_seconds=-1)
        self.assertEqual(caught.exception.code, "PENDING")
        self.assertEqual(sleeps, [])

    def test_invalid_runtime_id_sends_nothing(self):
        api, session, _ = api_with()
        for bad in ("Upper", "1starts-with-digit", "ends-", "x" * 64, "under_score"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                api.create_runtime(bad, self.REQUEST, fakes.GPU_L4)
        self.assertEqual(session.calls, [])


class RecordingAdapter(requests.adapters.BaseAdapter):
    """A requests transport adapter that answers from a script and records what was sent."""

    def __init__(self, responses):
        super().__init__()
        self.responses, self.sent = list(responses), []

    def send(self, request, **_kwargs):
        self.sent.append(request)
        status, body = self.responses.pop(0)
        response = requests.Response()
        response.status_code, response._content = status, json.dumps(body).encode()
        response.headers["Content-Type"] = "application/json"
        response.request, response.url = request, request.url
        return response

    def close(self):
        pass


class ServiceUsageTests(unittest.TestCase):
    def test_enable_services_polls_the_operation_and_reports_states(self):
        api, session, sleeps = api_with((200, {"name": "operations/enable-1", "done": False}),
                                        (200, {"name": "operations/enable-1", "done": True}),
                                        (200, {"state": "ENABLED"}), (200, {"state": "ENABLED"}))
        states = api.enable_services("demo-project")
        self.assertEqual(states, {"colaboratory.googleapis.com": "ENABLED", "drive.googleapis.com": "ENABLED"})
        self.assertEqual(sleeps, [3])
        first = session.calls[0]
        self.assertEqual(first["url"], colab_api.SERVICE_USAGE + "projects/demo-project/services:batchEnable")
        self.assertEqual(first["json"], {"serviceIds": list(colab_api.REQUIRED_SERVICES)})
        self.assertEqual(session.calls[1]["url"], colab_api.SERVICE_USAGE + "operations/enable-1")
        for call in session.calls:
            self.assertEqual(call["headers"], {"x-goog-user-project": None})

    def test_enable_services_errors(self):
        api, _, _ = api_with((200, {"done": True, "error": {"message": "Billing is not required but denied"}}))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "Could not enable APIs"):
            api.enable_services("demo-project")
        api, _, _ = api_with((200, {"name": "operations/slow", "done": False}))
        with self.assertRaisesRegex(colab_api.ColabAPIError, "taking longer"):
            api.enable_services("demo-project", deadline_seconds=-1)

    def test_service_usage_requests_do_not_send_the_quota_project_header(self):
        """With a real requests.Session (AuthorizedSession's base), None removes the session header."""
        adapter = RecordingAdapter([(200, {"done": True}), (200, {"state": "ENABLED"}),
                                    (200, {"state": "ENABLED"}), (200, {"runtimes": []})])
        session = requests.Session()
        session.mount("https://", adapter)
        api = colab_api.ColabAPI(None, QUOTA, session=session, sleep=lambda _s: None)
        api.enable_services("demo-project")
        self.assertEqual(len(adapter.sent), 3)
        for request in adapter.sent:
            self.assertNotIn("x-goog-user-project", {k.lower() for k in request.headers})
        api.list_runtimes()
        self.assertEqual(adapter.sent[-1].headers["x-goog-user-project"], QUOTA)


if __name__ == "__main__":
    unittest.main()
