# Colab Persist

Use a disposable Colab GPU from your terminal. Keep the work in Google Drive.

Colab Persist adds verified workspace checkpoints and a local MCP server/client to
[Google's official Colab CLI](https://github.com/googlecolab/google-colab-cli).
**L4 is the default GPU. No notebook editing or tmux is required.** Google may still
ask you to authorize a Drive mount in your browser when a new VM starts.

## Quick command reference

New here? [Install and authenticate](#install-and-authenticate) first. These
commands use the Google account selected by `colab-persist login`.

| What you want to do | Command |
|---|---|
| Check the selected account and whether a GPU is running | `colab-persist status` |
| List every active runtime on that account | `colab-persist sessions` |
| Start a GPU, mount Drive, and restore your workspace | `colab-persist start` |
| Start a T4 instead of the default L4 | `colab-persist start --gpu T4` |
| Open an SSH shell on the running VM | `colab-persist ssh` |
| Check the GPU from your Mac | `colab-persist ssh -- nvidia-smi` |
| Run a local script, save its workspace, then stop the VM | `colab-persist run ./train.py --project experiment` |
| Upload the current project folder and pass script arguments | `colab-persist run ./train.py --source . --project experiment -- --epochs 10` |
| Save your work to Drive and shut down the VM | `colab-persist stop` |
| Select another saved Google account, or sign in to a new one | `colab-persist login --email OTHER_GOOGLE_EMAIL` |
| See all commands or a command's options | `colab-persist --help` / `colab-persist run --help` |

`start` uses your configured GPU, initially **L4**. A different `--gpu` does not
resize a running VM: save and stop it first. Account switching also requires the
old account's runtimes to be stopped. `status` and `sessions` never allocate a VM;
`ssh` requires one to be running already. Scripts and outputs must stay inside the
managed workspace to be saved.

More commands: [save and restore](#checkpoints-and-shutdown) ·
[account switching](#accounts-and-the-official-cli) ·
[built-in `colab` commands](#useful-built-in-colab-commands) ·
[large datasets](LARGE_DATA.md) · [Codex and MCP setup](#mcp-local-agents-remote-cuda).

## How it works

```text
Mac: source code + coding agents + credentials
                    │ SSH / official Colab CLI
                    ▼
Colab L4: temporary workspace + CUDA execution
                    │ versioned archives + SHA-256 manifests
                    ▼
Google Drive: Colab-CUDA/projects/<project>/
```

For datasets larger than VM disk, use the separate manifest-driven shard cache.
`colab-persist dataset-plan corpus.json` validates the manifest and reports storage
budgets without allocating a GPU or copying data. Your training script uses
`ShardCache` to stage a bounded working set outside workspace snapshots.
See [large datasets and LoRA](LARGE_DATA.md) for the 1 TB
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
Local script paths, source size, project names and checkpoint intervals are checked
before allocating a GPU. Run the example from the cloned repository directory.
The example compiles real CUDA, validates 256 results on the GPU, and increments a
counter restored from the previous run.

## Install and authenticate

Requires macOS or Linux, Git, Python 3.12+, OpenSSH, `uv`, and Google Cloud CLI.

```sh
git clone https://github.com/ehzawad/colab-persist.git
cd colab-persist
uv tool install .
colab-persist configure --email YOUR_GOOGLE_EMAIL --gpu L4
colab-persist login --email YOUR_GOOGLE_EMAIL --no-launch-browser
```

The official Colab CLI is installed inside the tool's isolated Python environment;
you do not need to install it separately to use `colab-persist`. A standalone
`colab` command is optional; see the [built-in command reference](#useful-built-in-colab-commands).
If `colab-persist` is not found after
installation, run `uv tool update-shell` and open a new terminal. Keep your own
Google credentials on your computer; no repository credentials are supplied.

Use the authorization link from **that exact terminal attempt** and paste its code
back into the same terminal. Authorize Drive with the same Google account later.
The helper checks the Google identity before connecting or allocating a VM.

Check the installation and configured Google identity before allocating a GPU:

```sh
colab-persist tools                         # nine MCP tools
colab-persist status                        # checks account and runtime status
```

Then, from the cloned repository directory, run the CUDA example twice:

```sh
colab-persist run examples/cuda_demo.py --project first-run
colab-persist run examples/cuda_demo.py --project first-run
colab-persist status
```

With a new project name, the first run reports count 1; the second restores it and
reports count 2 with previous count 1. Each successful run saves, flushes Drive,
and stops its VM, so the final status should be inactive. Expect a browser consent
step for each new VM's Drive mount. These runs consume Colab compute units.
This flow was [verified from a fresh public clone on macOS](VALIDATION.md#version-021-fresh-installation-and-recovery).

Configuration lives in `~/.config/colab-persist/config.json`. Login stores private
Google ADC credentials and session mappings under `~/.config/colab-persist/accounts/`,
separated by account. Existing installations without private credentials continue
to use their current ADC login until `login` is explicitly run. A dedicated Ed25519
key is created only if the selected key does not exist. None of these files are
uploaded by this tool.

If Colab's initial Drive mount fails immediately after consent, the helper retries
once and verifies the actual mount. A failed mount never becomes an ordinary local
directory masquerading as persistent storage.

## Accounts and the official CLI

`colab` and `colab-persist` are separate executables. With the pinned CLI version
0.7.4, bare `colab` defaults to OAuth2, while this wrapper explicitly uses ADC.
They can therefore be signed in as different Google accounts. The wrapper uses a
private session file for each account and does not edit the official CLI's shared
session file. During upgrade, an existing mapping is copied only after the server
confirms that its runtime belongs to the verified account.
Upstream CLI history, logs and settings remain shared; only credentials and session
mappings have separate account storage.

Use these commands to check the account and its runtimes without allocating one:

```sh
colab-persist status       # selected account, managed runtime, total active count
colab-persist sessions     # every runtime on that account, including unmanaged ones
```

`active: false` means the configured managed session is absent. Check
`active_runtimes: 0` to confirm that the selected account has no runtimes at all.
These commands do not inspect other Google accounts.

To verify and save another account's login while keeping the current account selected:

```sh
colab-persist login --email OTHER_GOOGLE_EMAIL --no-switch
```

This verifies the target email and Colab access, then saves its private credentials.
It leaves the selected-account configuration unchanged and does not require the
current account's runtimes to stop. Select the saved account later using `login`
without `--no-switch`; the normal checks for running VMs apply then.

To switch later, save and stop the current account's runtimes, then log in:

```sh
colab-persist stop         # if the managed VM is active
colab-persist login --email OTHER_GOOGLE_EMAIL
colab-persist status
colab-persist start --gpu T4
```

The switch refuses while the old account has any running Colab VM. Save and stop
unmanaged runtimes through their owning workflow. New browser credentials are
staged separately; the expected email and Colab access are verified before changing
the selection. Cancelled or wrong-account login leaves the selection unchanged.
Previously saved accounts can be selected again with the same `login --email`
command; `--reauth` requests fresh browser consent if credentials expired or were
revoked. The current account must still authenticate so its running VMs can be
checked; reauthenticate it first if needed. Sign in privately once per account,
including an account previously used only through global ADC, before cached
switching is available. Account switching never transfers files between Google Drives. Authorize
each VM's Drive mount using the selected Colab account; the browser consent step
is responsible for choosing the matching Drive identity.

`configure --email` sets the account expectation on first setup; it cannot change
an existing account. `login` does not overwrite global gcloud credentials, the
official CLI's OAuth2 token, or coding-agent logins. An explicit private login also
takes precedence over a shell's `GOOGLE_APPLICATION_CREDENTIALS` setting. Merely
running `gcloud config set account` does not switch ADC credentials; see
[Google's ADC documentation](https://docs.cloud.google.com/docs/authentication/application-default-credentials).

Use the wrapper's SSH and persistence commands for its managed VM. Raw commands
such as `colab stop` or `colab restart-kernel` can still affect that same server if
authenticated to its account, and bypass the wrapper's save/locking checks.
Account isolation cannot prevent changes made directly through Google or another
CLI. Avoid simultaneous raw CLI mutations of a managed runtime.

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

## Useful built-in `colab` commands

These are commands from **Google's official CLI 0.7.4**, the version this project
pins. They operate independently of `colab-persist`. The wrapper installation
includes the CLI internally; if you also want a standalone `colab` executable:

```sh
uv tool install google-colab-cli==0.7.4
colab --help
colab version
colab update                 # checks for updates; does not install them
```

**Account and session selection are separate.** Bare `colab` defaults to OAuth2.
The examples below explicitly use `--auth adc`, which uses your shell's
`GOOGLE_APPLICATION_CREDENTIALS` file or global ADC login. Neither automatically
follows the private account selected by `colab-persist login`. The native CLI also
uses its own session file, so its session names do not automatically identify
wrapper-managed VMs. See [Google's local ADC setup guide](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)
to set up ADC for the native CLI if needed.

The name `native-demo` below is a **separate, independently managed session**.
For your persistent workspace, use the `colab-persist` commands above.

| What you want to do | Official CLI command |
|---|---|
| List active runtimes on the native CLI's account | `colab --auth adc sessions` |
| Check a named session | `colab --auth adc status -s native-demo` |
| Show compute-unit balance and usage rate | `colab --auth adc usage` |
| Create an L4 session | `colab --auth adc new -s native-demo --gpu L4` |
| Open SSH | `colab --auth adc ssh -s native-demo` |
| Open an interactive Python prompt | `colab --auth adc repl -s native-demo` |
| Execute a local Python file on the existing session | `colab --auth adc exec -s native-demo -f ./train.py --timeout 600` |
| Install a Python package on the VM | `colab --auth adc install -s native-demo numpy` |
| List remote files | `colab --auth adc ls -s native-demo /content` |
| Upload one file | `colab --auth adc upload -s native-demo ./input.json /content/input.json` |
| Download one file | `colab --auth adc download -s native-demo /content/output.json ./output.json` |
| View recent session events | `colab --auth adc log -s native-demo -n 20` |
| Open the session's notebook URL in a browser | `colab --auth adc url -s native-demo --open` |
| Stop this native session without a wrapper checkpoint | `colab --auth adc stop -s native-demo` |

Replace `L4` with `T4` when creating a T4 session. Native `new` defaults to CPU if
you omit `--gpu` and `--tpu`. Native `ssh` can create a VM if the named session is
missing; use `sessions` or `status` to check first. If needed, pass
`-i /path/to/private_ssh_key` to `ssh`. Upload and download transfer individual
files, not whole project folders. Installed packages and files on these native
VMs disappear when the VM is deleted unless you save them elsewhere.

For a disposable script run, the native CLI also offers:

```sh
colab --auth adc run --gpu T4 --timeout 600 ./train.py --epochs 10
```

This creates a fresh VM and stops it afterward. Put CLI options **before** the
script path; everything after it is passed to the script. Native `run` does not
perform Colab Persist's Drive checkpoints. Native `stop`, `restart-kernel`, and
`drivemount` also bypass its save and locking checks. Always use
`colab-persist stop` to save and shut down a wrapper-managed VM.

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

### Claude Code: talk to your GPU from your Mac

After [installing and authenticating](#install-and-authenticate), register the
server once. If you already have a custom `colab` skill, merge its instructions
instead of overwriting it. Run these commands from the cloned repository:

```sh
claude mcp add --scope user --transport stdio colab-persist -- "$(command -v colab-persist-mcp)"
mkdir -p ~/.claude/skills/colab
cp integrations/claude-code/colab/SKILL.md ~/.claude/skills/colab/SKILL.md
claude mcp get colab-persist
```

The connection and `/colab` skill are available across your local projects.
Start a new Claude Code session in your project folder:

```sh
claude
```

Then say, for example:

```text
/colab Check my account and whether any GPU is running.
/colab Run train.py on an L4, save checkpoints to Drive, and stop when finished.
/colab Download this dataset on Colab, keep durable shards in Drive, and use a bounded local cache.
```

You can also ask in plain language; the skill is discoverable automatically.
Claude stays on the Mac and uses MCP/SSH for remote work. Dataset downloads
started on the VM use Colab's network, without routing the dataset through your
Mac. You do not need another Claude login for each GPU VM.

Google may still require you to run `colab-persist mount` in a local interactive
terminal and authorize Drive in your browser. Claude then continues with the
mounted runtime. It never needs your OAuth code in chat. For long jobs, the skill
uses the CLI in a monitored background Bash task to avoid short MCP tool deadlines;
keep the Mac awake and connected. Normal Claude tool permissions still apply.

The skill covers remote command routing, bounded datasets, application resume,
account switching, and stopping only after a confirmed save. It does not turn
Colab into an always-on service or make an arbitrary trainer resumable.

### Codex and other MCP clients

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

This is a small, independent workflow layer for one selected account and named
runtime at a time, not an official Google product. It uses the native Drive mount already
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
consume compute units and require your own account; see [VALIDATION.md](VALIDATION.md) for the
maintainer's recorded verification.

Sources: [Colab FAQ and runtime limits](https://research.google.com/colaboratory/faq.html),
[Drive flush implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py),
[CLI Drive integration](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md),
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
