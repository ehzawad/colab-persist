# Colab Persist

Use a disposable Colab GPU from your terminal. Keep the work in Google Drive.

Colab Persist adds verified workspace checkpoints and a local MCP server/client to
[Google's official Colab CLI](https://github.com/googlecolab/google-colab-cli).
**L4 is the default GPU. No notebook editing or tmux is required.** Google may still
ask you to authorize a Drive mount in your browser when a new VM starts.

```text
Mac: source code + coding agents + credentials
                    │ SSH / official Colab CLI
                    ▼
Colab L4: temporary workspace + CUDA execution
                    │ versioned archives + SHA-256 manifests
                    ▼
Google Drive: Colab-CUDA/projects/<project>/
```

For datasets larger than VM disk, use the separate manifest-driven shard cache:
`colab-persist dataset-plan corpus.json`. It stages a bounded working set outside
workspace snapshots. See [large datasets and LoRA](LARGE_DATA.md) for the 1 TB
storage layout, checkpoint requirements, and current limits. Do not place a
training corpus or base model cache inside the managed workspace.

## A complete run

```sh
colab-persist run examples/cuda_demo.py --project cuda-learning
```

This starts or reuses an L4, mounts Drive, restores the project's last checkpoint,
uploads the script, and runs it on the GPU. During execution it attempts a checkpoint
every 60 seconds. After the script exits—even with an error—it checkpoints again,
asks Drive to flush outstanding writes, and stops the VM **only if saving succeeds**.
The example compiles real CUDA, validates 256 results on the GPU, and increments a
counter restored from the previous run.

## Install and authenticate

Requires macOS or Linux, Python 3.12+, OpenSSH, `uv`, and Google Cloud CLI.

```sh
git clone https://github.com/ehzawad/colab-persist.git
cd colab-persist
uv tool install .
colab-persist configure --email YOUR_GOOGLE_EMAIL --gpu L4

gcloud auth application-default login YOUR_GOOGLE_EMAIL --no-launch-browser \
  --disable-quota-project \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Use the authorization link from **that exact terminal attempt** and paste its code
back into the same terminal. Authorize Drive with the same Google account later.
The helper checks the Google identity before connecting or allocating a VM.

Configuration lives in `~/.config/colab-persist/config.json`; Google credentials use
the existing local ADC store. A dedicated Ed25519 key is created only if the selected
key does not exist. None of these files are uploaded by this tool.

If Colab's initial Drive mount fails immediately after consent, the helper retries
once and verifies the actual mount. A failed mount never becomes an ordinary local
directory masquerading as persistent storage.

## Terminal and SSH workflow

```sh
colab-persist start                         # L4 + Drive + default workspace
colab-persist ssh                           # ordinary SSH shell
# On the VM:
cd /content/colab-persist/default/workspace
nvcc my_kernel.cu -o my_kernel
./my_kernel
exit
# Back on the Mac:
colab-persist stop                          # checkpoint + flush + shutdown
```

On the next `start`, the default workspace is restored. Work inside the displayed
managed workspace; files elsewhere on the VM are outside the backup scope.
`colab-persist ssh -- nvidia-smi` also works for single commands.

To get a conventional SSH alias, configure a **dedicated** fragment:

```sh
colab-persist configure --email YOUR_GOOGLE_EMAIL --gpu L4 \
  --ssh-config ~/.ssh/config.d/colab-cuda.conf
```

Include `~/.ssh/config.d/*` from your existing SSH config, then run `start`.
`ssh colab-cuda` and `rsync ... colab-cuda:/content/colab-persist/default/workspace/`
work normally. Host keys are pinned separately for each runtime endpoint. Starting
a replacement refreshes this alias; connecting to an old endpoint fails rather
than quietly connecting to another VM.

Select another GPU explicitly when starting a fresh runtime:

```sh
colab-persist start --gpu T4
colab-persist start --gpu A100
```

L4 remains the configured default. Availability and compute-unit cost depend on
your account. The helper never silently substitutes another GPU or changes a live
runtime's hardware; save and stop it first.

## Projects and script arguments

Only the named script is uploaded by default. To include a whole project:

```sh
colab-persist run ./train.py --source . --project experiment -- --epochs 10
colab-persist run examples/checkpoint_loop.py --project resumable -- --steps 30
```

The script runs with its project workspace as the current directory:

| Variable | Meaning |
|---|---|
| `COLAB_WORKSPACE` | Temporary restored project directory |
| `COLAB_OUTPUT_DIR` | Its `outputs/` directory, included in every checkpoint |
| `COLAB_DATASET_ROOT` | Read-only source shards under Drive `Colab-CUDA/datasets/` |
| `COLAB_DATASET_CACHE` | Disposable local shard cache, outside snapshots |
| `HF_HOME` | Defaults to the external local Hugging Face cache if not already set |

Scripts may invoke `nvcc`, `make`, profilers, or other commands. Store a dependency
file and bootstrap script in the project and run them as needed after replacement;
installed system packages and Python environments are not VM images and are not
restored automatically. CUDA comes from the Colab runtime image.

Source overlays update matching project files without deleting other restored
files. Caches, virtual environments, `.git`, common credential directories,
`.env*`, private-key extensions, symbolic links, and temporary files are excluded.
These exclusions are not a secret scanner: do not put credentials in ordinary
project files or command-line arguments.

## Checkpoints and shutdown

```sh
colab-persist status
colab-persist snapshots --project experiment   # requires mounted Drive
colab-persist save                            # flush/unmount; VM stays allocated
colab-persist mount                           # remount to continue saving
colab-persist restore --project experiment    # empty destination required
colab-persist stop
```

Each snapshot is an immutable `.tar.gz` plus a JSON manifest containing SHA-256
hashes for every saved file and the archive. A manifest is written only after
the copied archive verifies. Incomplete uploads are ignored; unchanged snapshots
are reused. Restore validates both the archive and extracted files, rejects links
and path traversal, and refuses to overwrite a nonempty workspace. Checkpoints are
retained until you choose to remove them from Drive; there is no automatic pruning.
Workspace and source uploads default to a 5 GiB size cap checked before hashing;
oversized saves refuse shutdown. Use the separate shard cache for large corpora.

Periodic snapshots report `pending_drive_flush`. A successful final save reports
`drive_flush_confirmed`, using Colab's `drive.flush_and_unmount()`. Save/stop refuses
while a managed script is running. If Drive is unavailable, full, or cannot flush,
the operation fails and leaves the VM allocated for recovery. Fix the problem and
retry `stop`; do not bypass it unless you accept losing unsaved work.

`run --keep-runtime` flushes Drive but leaves the GPU allocated. Remount Drive before
another save. Direct `colab stop`, the Colab browser's Delete Runtime action, and
Google's own reclamation bypass the save guard.

## MCP: local agents, remote CUDA

Keep Codex, Claude Code, or your preferred coding agent on your Mac. Their login
and configuration survive runtime deletion; the MCP tools operate the GPU worker.

```json
{
  "mcpServers": {
    "colab-persist": {
      "command": "/absolute/path/to/colab-persist-mcp"
    }
  }
}
```

For Codex, find the installed executable with `command -v colab-persist-mcp`, then:

```sh
codex mcp add colab-persist -- /absolute/path/to/colab-persist-mcp
```

Tools: `runtime_status`, `start_runtime`, `prepare_workspace`, `run_script`,
`list_checkpoints`, `restore_workspace`, `save_workspaces`, `safe_stop`, and `plan_dataset`.
The terminal client uses the same tools over real MCP stdio. `colab-persist tools`
checks the connection. No HTTP port is exposed. Drive consent runs in a terminal,
because OAuth codes should not be passed through a model conversation. Long jobs
are best submitted through the supplied client: other MCP hosts may impose their
own tool timeouts. An interrupted client does not confirm a job stopped; inspect
runtime status before starting another job or shutting down.

## What survives—and what cannot

Saved source files, binaries, outputs, application checkpoints, and setup recipes
survive in Drive. RAM, VRAM, running processes, root filesystem changes, and writes
that never reached Drive do not. An application must write restartable state
(for example model/optimizer state and the current step) to resume computation.
Use atomic replacement for application checkpoint files; copying a live multi-file
database is not a transactional backup. Background child processes must finish
before your script exits.

Colab Pro does not guarantee a ten-hour session. Paid runtimes still have variable
idle and maximum-lifetime limits. This tool does not simulate activity or prevent
Google from reclaiming a VM. After an unexpected termination, start a replacement
and restore the last checkpoint that reached Drive. Recovery is explicit, so a
replayed workload cannot silently allocate repeated GPUs or duplicate side effects.

## Existing tools

| Project | What it already provides | Where this project fits |
|---|---|---|
| [Official Colab CLI](https://github.com/googlecolab/google-colab-cli) | GPU selection, SSH, execution, file transfer and Drive mounting | Used directly as the transport |
| [colabctl](https://github.com/mandipadk/colabctl) | Broader runtime/job APIs, MCP, and direct Drive checkpoint helpers | A fuller alternative; its direct Drive transfers require a Cloud quota project and Drive API |
| [Colab MCP](https://github.com/googlecolab/colab-mcp) | Local-agent bridge to a Colab session in the browser | Useful for notebook interaction |

This is a small, independent workflow layer for a single account and named runtime,
not an official Google product. It deliberately uses the native Drive mount already
supported by Colab; no separate Google Cloud project is needed for that mount.

## Development and verification

```sh
uv sync --no-editable
uv run --no-editable python -m unittest discover -s tests -v
uv run --no-editable colab-persist tools
```

Tests cover recovery into a new workspace, corruption, interrupted publication,
credential/link exclusions, malicious archives, overwrite protection, script errors,
active-job locks, and refusal to stop after a failed flush. Live GPU/Drive checks
consume compute units and require your own account; see `VALIDATION.md` for the
maintainer's recorded verification.

Sources: [Colab FAQ and runtime limits](https://research.google.com/colaboratory/faq.html),
[Drive flush implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py),
[CLI Drive integration](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md),
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
