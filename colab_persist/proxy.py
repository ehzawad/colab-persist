"""SSH bridge pinned to the exact runtime selected by the control plane."""
import argparse
import os
from pathlib import Path
import sys
from . import backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    try:
        item = backend.require_session()
        if item.endpoint != args.endpoint:
            raise RuntimeError("Runtime was replaced. Run `colab-persist start` to refresh the SSH alias.")
        cfg = backend.config()
        command = [*backend.colab_command(), "ssh", "--proxy-mode", "-s", item.name, "-i",
                   str(Path(cfg.get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser())]
        os.execv(command[0], command)
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
