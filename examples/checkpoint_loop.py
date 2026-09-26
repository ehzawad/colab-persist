"""An application checkpoint: resume completed steps, rather than in-memory state."""
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
args = parser.parse_args()
output = Path(os.environ.get("COLAB_OUTPUT_DIR", "outputs"))
output.mkdir(parents=True, exist_ok=True)
checkpoint = output / "progress.json"
completed = json.loads(checkpoint.read_text())["completed"] if checkpoint.exists() else 0
for step in range(completed, args.steps):
    # Replace with useful computation and a full application state checkpoint.
    value = sum(i * i for i in range(100_000))
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps({"completed": step + 1, "last_value": value}) + "\n")
    temporary.replace(checkpoint)
    print(f"Completed {step + 1}/{args.steps}", flush=True)
    time.sleep(1)
