"""Client for Google's official Colab REST API (v1beta).

Runtimes are Jupyter servers on Google infrastructure. User credentials need a
quota project (a free Cloud project with colaboratory.googleapis.com enabled).
Creation returns a long-running Operation without an operations.get method:
repeating the identical create (same runtimeId and requestId) waits for it.
"""
from __future__ import annotations

import re
import time
from urllib.parse import urlparse

from google.auth.transport.requests import AuthorizedSession

BASE = "https://colaboratory.googleapis.com/v1beta/"
SERVICE_USAGE = "https://serviceusage.googleapis.com/v1/"
REQUIRED_SERVICES = ("colaboratory.googleapis.com", "drive.googleapis.com")
GPUS = ("T4", "L4", "G4", "A100", "H100")
RUNTIME_ID = re.compile(r"[a-z]([-a-z0-9]{0,61}[a-z0-9])?")


class ColabAPIError(RuntimeError):
    def __init__(self, message, status=None, code=None, reason=None):
        super().__init__(message)
        self.status, self.code, self.reason = status, code, reason


def _error(response):
    try:
        error = response.json().get("error", {})
    except ValueError:
        return ColabAPIError(f"Colab API returned HTTP {response.status_code}.", response.status_code)
    reason = next((d.get("reason") for d in error.get("details", []) if d.get("reason")), None)
    message = error.get("message", "")
    if reason == "SERVICE_DISABLED" or "requires a quota project" in message:
        message = ("The Colab API needs a quota project with colaboratory.googleapis.com enabled. "
                   "Run `colab-persist setup --project YOUR_PROJECT_ID` (free; no billing needed).")
    return ColabAPIError(message or f"Colab API returned HTTP {response.status_code}.",
                         response.status_code, error.get("status"), reason)


def endpoint_from_url(url: str) -> str:
    """'https://8080-m-s-xyz-b.region.prod.r.colab.dev' -> 'm-s-xyz' (Colab's internal endpoint).

    The host label is '<port>-<endpoint>-<zone letter>'."""
    label = urlparse(url).netloc.split(".", 1)[0]
    match = re.fullmatch(r"\d+-(.+)-[a-z]", label)
    return match.group(1) if match else label


def runtime_id_of(runtime: dict) -> str:
    return runtime["name"].split("/", 1)[1]


def spec_label(key: dict) -> str:
    variant = key.get("variant", "").replace("VARIANT_", "")
    accelerator = key.get("accelerator", "NONE")
    shape = "high-mem" if key.get("shape") == "SHAPE_HIGHMEM" else "standard"
    return (accelerator if accelerator != "NONE" else variant) + ("" if shape == "standard" else " " + shape)


def wanted_spec(gpu=None, tpu=None, cpu=False, high_mem=False) -> dict:
    chosen = [bool(gpu), bool(tpu), bool(cpu)]
    if sum(chosen) > 1:
        raise ValueError("Choose only one of --gpu, --tpu or --cpu.")
    shape = "SHAPE_HIGHMEM" if high_mem else "SHAPE_STANDARD"
    if cpu:
        return {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": shape}
    if tpu:
        return {"variant": "VARIANT_TPU", "accelerator": tpu.upper().replace("-", ""), "shape": shape}
    return {"variant": "VARIANT_GPU", "accelerator": (gpu or "L4").upper(), "shape": shape}


def choose_spec(specs: list, want: dict) -> dict:
    keyed = {tuple(sorted(s["key"].items())): s for s in specs}
    match = keyed.get(tuple(sorted(want.items())))
    eligible = sorted(spec_label(s["key"]) for s in specs if s.get("eligible"))
    if match is None:
        raise ValueError(f"Colab does not offer {spec_label(want)}. Eligible for this account: "
                         + (", ".join(eligible) or "none"))
    if not match.get("eligible"):
        raise ValueError(f"This account is not eligible for {spec_label(want)}. Eligible: "
                         + (", ".join(eligible) or "none") + ". Nothing was allocated.")
    return match["key"]


class ColabAPI:
    def __init__(self, credentials, quota_project, *, session=None, sleep=time.sleep):
        if not quota_project:
            raise ColabAPIError("No quota project is set for this account. Run "
                                "`colab-persist setup --project YOUR_PROJECT_ID` (free; no billing needed).")
        self.quota_project = quota_project
        self.session = session or AuthorizedSession(credentials)
        self.session.headers["x-goog-user-project"] = quota_project
        self.sleep = sleep

    def _call(self, method, path, *, params=None, body=None, timeout=60, base=BASE, ok=(200,)):
        # Service Usage must work before the quota project itself is ready.
        headers = {"x-goog-user-project": None} if base == SERVICE_USAGE else None
        response = self.session.request(method, base + path, params=params, json=body, timeout=timeout,
                                        headers=headers)
        if response.status_code not in ok:
            raise _error(response)
        return response.json() if response.content else {}

    def subscription(self):
        return self._call("GET", "subscription").get("tier", "SUBSCRIPTION_TIER_UNSPECIFIED")

    def runtime_specs(self):
        return self._call("GET", "runtimespecs").get("runtimeSpecs", [])

    def list_runtimes(self):
        return self._call("GET", "runtimes").get("runtimes", [])

    def get_runtime(self, runtime_id):
        try:
            return self._call("GET", "runtimes/" + runtime_id)
        except ColabAPIError as error:
            if error.status == 404:
                return None
            raise

    def create_runtime(self, runtime_id, request_id, spec, *, deadline_seconds=900, poll_seconds=5,
                       resume=False, repost_every=6):
        """Create one runtime, or keep waiting for it. Safe to call again with the same IDs.

        The first POST starts creation. After that, waiting reads the runtime; the identical
        POST is repeated only occasionally to learn the operation's outcome. A refusal of a
        repeated POST never means that nothing was created."""
        if not RUNTIME_ID.fullmatch(runtime_id):
            raise ValueError(f"Invalid runtime ID {runtime_id!r}.")
        deadline = time.monotonic() + deadline_seconds
        started, polls = resume, 0
        if resume:
            existing = self.get_runtime(runtime_id)
            if existing and existing.get("connectionInfo"):
                return existing
        while True:
            if not started or polls % repost_every == 0:
                try:
                    operation = self._call("POST", "runtimes",
                                           params={"runtimeId": runtime_id, "requestId": request_id},
                                           body={"runtimeSpec": spec}, timeout=120)
                except ColabAPIError as error:
                    if error.code == "ALREADY_EXISTS":
                        existing = self.get_runtime(runtime_id)
                        if not existing or existing.get("runtimeSpec") != spec:
                            raise
                        if existing.get("connectionInfo"):
                            return existing
                        # Ours and still starting; keep waiting below.
                    elif error.code == "FAILED_PRECONDITION":
                        raise ColabAPIError("Colab could not allocate this runtime now: no capacity, quota or "
                                            "compute units, or too many active runtimes. " + str(error),
                                            error.status, error.code, error.reason) from None
                    elif error.code == "RESOURCE_EXHAUSTED" and not started:
                        raise ColabAPIError("Colab is rate-limiting runtime creation; nothing was created. "
                                            "Wait a few minutes and retry `colab-persist start`.",
                                            error.status, error.code, error.reason) from None
                    elif not (error.code in ("DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "UNAVAILABLE")
                              or error.status in (429, 500, 502, 503, 504)):
                        raise
                    # Otherwise the request may have been accepted; keep waiting by reading the runtime.
                else:
                    if operation.get("done"):
                        if operation.get("error"):
                            error = operation["error"]
                            raise ColabAPIError("Runtime creation failed: " + error.get("message", "unknown error"),
                                                code="CREATION_FAILED")
                        runtime = operation.get("response")
                        if runtime and runtime.get("connectionInfo"):
                            return runtime
                started = True
            try:
                runtime = self.get_runtime(runtime_id)
            except ColabAPIError as error:
                if error.status not in (429, 500, 502, 503, 504):
                    raise
                runtime = None  # A transient read error is not an answer; keep waiting.
            if runtime and runtime.get("connectionInfo"):
                return runtime
            if time.monotonic() > deadline:
                raise ColabAPIError("Runtime creation is still pending. Run `colab-persist start` again to keep "
                                    "waiting for the same runtime; nothing extra will be allocated.", code="PENDING")
            polls += 1
            self.sleep(poll_seconds)

    def delete_runtime(self, runtime_id, *, allow_missing=False):
        self._call("DELETE", "runtimes/" + runtime_id,
                   params={"allowMissing": "true"} if allow_missing else None, timeout=120)

    def enable_services(self, project, services=REQUIRED_SERVICES, *, deadline_seconds=180):
        """Enable free APIs on the user's project. Never touches billing."""
        operation = self._call("POST", f"projects/{project}/services:batchEnable",
                               body={"serviceIds": list(services)}, base=SERVICE_USAGE, timeout=120)
        deadline = time.monotonic() + deadline_seconds
        while not operation.get("done"):
            if time.monotonic() > deadline:
                raise ColabAPIError("Enabling the APIs is taking longer than expected; retry setup shortly.")
            self.sleep(3)
            operation = self._call("GET", operation["name"], base=SERVICE_USAGE)
        if operation.get("error"):
            raise ColabAPIError("Could not enable APIs: " + operation["error"].get("message", ""))
        states = {}
        for service in services:
            states[service] = self._call("GET", f"projects/{project}/services/{service}",
                                         base=SERVICE_USAGE).get("state")
        return states
