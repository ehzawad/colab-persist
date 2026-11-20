"""One-time Drive mount used only by ``colab-persist import-legacy``.

colab-persist 0.3 wrote checkpoints through Colab's own Drive mount, so the
narrower Drive permission used from 0.4 on cannot see them. This helper runs in a
temporary environment that has Google's colab CLI (``uv run --with``), registers
the managed runtime from the official API's connection info, and reuses the CLI's
``drivemount`` flow. Colab asks for browser consent once for this VM.

Instead of waiting for Enter on the terminal, the helper polls Colab's own
authorization check, so the mount continues by itself once consent is granted.
This also works from non-interactive agents. It is not imported by the rest of the
package and is never installed as a dependency.
"""
import builtins
import json
import re
import sys
import time
from datetime import datetime

CONSENT_WAIT_SECONDS = 600


def main():
    with open(sys.argv[1]) as handle:
        args = json.load(handle)
    from colab_cli.auth import AuthProvider, get_credentials
    from colab_cli.commands import automation
    from colab_cli.common import state
    from colab_cli.state import SessionState

    state.config_path = args["config_path"]
    state.auth_provider = AuthProvider.ADC
    expires = None
    if args.get("expires_at"):
        # Trim nanoseconds, which datetime.fromisoformat does not accept on every Python.
        text = re.sub(r"(\.\d{6})\d+", r"\1", args["expires_at"]).replace("Z", "+00:00")
        expires = datetime.fromisoformat(text)
    host = args["url"].split("//", 1)[-1].split("/", 1)[0]
    for assignment in state.client.list_assignments():
        # Prefer Colab's own record of this runtime over parsing its address.
        if assignment.runtime_proxy_info.url.split("//", 1)[-1].split("/", 1)[0] == host:
            args["endpoint"] = assignment.endpoint
    state.store.add(SessionState(name=args["name"], token=args["token"], url=args["url"],
                                 endpoint=args["endpoint"], token_expires_at=expires))

    def authorized():
        # A dry run keeps reporting "unauthorized" even after consent; the real propagation
        # call (which the CLI makes after Enter) succeeds once the user has approved.
        url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{args['endpoint']}"
        params = {"authuser": "0", "authtype": "dfs_ephemeral", "version": "2", "dryrun": "false",
                  "propagate": "true", "record": "false"}
        session = get_credentials(state.client_oauth_config, provider=state.auth_provider)
        response = session.request("GET", url, params=params)
        token = json.loads(response.text.split("\n", 1)[-1]).get("token")
        response = session.request("POST", url, params=params, headers={"x-goog-colab-token": token},
                                   files={"file_id": (None, "empty.ipynb")})
        return bool(json.loads(response.text.split("\n", 1)[-1]).get("success"))

    class ConsentWaiter:
        """Stands in for the CLI's "Press Enter" prompt on /dev/tty."""

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def readline(self):
            print("\nWaiting for you to approve Drive access in the browser; this continues by itself.",
                  flush=True)
            deadline = time.monotonic() + CONSENT_WAIT_SECONDS
            while time.monotonic() < deadline:
                time.sleep(3)
                try:
                    if authorized():
                        return "\n"
                except Exception:
                    pass
            return "\n"  # The CLI then reports whether propagation succeeded.

    def patched_open(path, *rest, **options):
        if path == "/dev/tty":
            return ConsentWaiter()
        return builtins.open(path, *rest, **options)

    automation.open = patched_open  # Module globals shadow the builtin inside drivefs_hook.
    automation.drivemount(session=args["name"], path="/content/drive")


if __name__ == "__main__":
    main()
