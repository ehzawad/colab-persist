# Colab Persist

Use a disposable Colab GPU from your terminal. Keep the work in Google Drive.

Colab Persist runs verified workspace checkpoints, a terminal client and a local MCP
server on top of [Google's official Colab API](https://developers.google.com/colab/api/reference/rest).
**L4 is the default GPU. Drive connects from the command line: one sign-in per
account, and no browser consent for each new VM.** No notebook editing or tmux is required.

## Quick command reference

New here? [Install and sign in](#install-and-sign-in) first. These commands use the
Google account selected by `colab-persist login`.

| What you want to do | Command |
|---|---|
| Check the selected account and whether a runtime is running | `colab-persist status` |
| List every runtime on that account | `colab-persist sessions` |
| See which GPUs/TPUs your account may use | `colab-persist specs` |
| Start a GPU, connect Drive, and restore your workspace | `colab-persist start` |
| Start a T4, a TPU or a CPU runtime instead of the default L4 | `colab-persist start --gpu T4` / `--tpu V5E1` / `--cpu` |
| Open an SSH shell on the running VM | `colab-persist ssh` |
| Check the GPU from your Mac | `colab-persist ssh -- nvidia-smi` |
| Run a local script, save its workspace, then stop the VM | `colab-persist run ./train.py --project experiment` |
| Upload the current project folder and pass script arguments | `colab-persist run ./train.py --source . --project experiment -- --epochs 10` |
| Start a long job and get your terminal back | `colab-persist run ./train.py --project experiment --detach` |
| Follow that job again, then save and stop | `colab-persist attach --project experiment` |
| Show a job's state and latest log lines | `colab-persist job --project experiment` |
| List checkpoints in Drive (no VM needed) | `colab-persist snapshots --project experiment` |
| Save your work to Drive and shut down the VM | `colab-persist stop` |
| Select another saved Google account, or sign in to a new one | `colab-persist login --email OTHER_GOOGLE_EMAIL --project PROJECT_ID` |
| Bring checkpoints over from colab-persist 0.3, once | `colab-persist import-legacy` |
| See all commands or a command's options | `colab-persist --help` / `colab-persist run --help` |

`start` uses your configured GPU, initially **L4**. A different `--gpu` does not
resize a running VM: save and stop it first. Account switching also requires the
old account's runtimes to be stopped. `status`, `sessions`, `specs` and `snapshots`
never allocate a VM; `ssh` requires one to be running already. Scripts and outputs
must stay inside the managed workspace to be saved.

More: [Drive](#how-drive-works) · [accounts](#accounts) · [long jobs](#long-jobs) ·
[save and restore](#checkpoints-and-shutdown) · [large datasets](LARGE_DATA.md) ·
[Claude Code, Codex and MCP](#mcp-local-agents-remote-cuda) · [upgrading from 0.3](#upgrading-from-03).

## How it works

```text
Mac: source code + coding agents + Google credentials
        │ official Colab API (create, list, delete runtimes)
        │ SSH through the runtime's /colab/ssh endpoint
        ▼
Colab L4: temporary workspace + CUDA execution
        │ Drive API, short-lived token limited to colab-persist's own files
        │ versioned archives + SHA-256 manifests, verified by Drive
        ▼
Google Drive: My Drive/Colab-Persist/projects/<project>/
```

For datasets larger than VM disk, use the separate manifest-driven shard cache.
`colab-persist dataset-plan corpus.json` validates the manifest and reports storage
budgets without allocating a GPU or copying data. Your training script uses
`ShardCache` to stage a bounded working set outside workspace snapshots.
See [large datasets and LoRA](LARGE_DATA.md). Do not place a training corpus or
base model cache inside the managed workspace.

## A complete run

```sh
colab-persist run examples/cuda_demo.py --project cuda-learning
```

This starts or reuses an L4, connects Drive, restores the project's last checkpoint,
uploads the script, and runs it on the GPU as a managed job. While it runs, the job
saves a checkpoint every 60 seconds. After the script exits, even with an error, it
saves again, Drive confirms the upload's size and SHA-256, and the VM is stopped
**only if that verification succeeds**. Local script paths, source size, project
names and checkpoint intervals are checked before allocating a GPU. Run the
example from the cloned repository directory. It compiles real CUDA, validates 256
results on the GPU, and increments a counter restored from the previous run.

## Install and sign in

Requires macOS or Linux, Git, Python 3.12+, OpenSSH, `uv`, Google Cloud CLI, and a
Google Cloud project you own. Any project works; **no billing account is needed**.
The Colab API and the Drive API are free; Colab compute is billed to your Colab
plan as usual.

```sh
git clone https://github.com/ehzawad/colab-persist.git
cd colab-persist
uv tool install .
colab-persist configure --email YOUR_GOOGLE_EMAIL --gpu L4
colab-persist login --email YOUR_GOOGLE_EMAIL --project YOUR_PROJECT_ID
```

Don't have a project? `gcloud projects create YOUR_PROJECT_ID` creates one; any
existing project, including one Google AI Studio created for you, also works.

`login` opens Google's consent page once for this account. Approve it in the
browser; the terminal continues by itself. It then enables the Colab and Drive APIs
on your project if needed, and checks that Colab and Drive both answer for the
same Google account before saving anything. If `colab-persist` is not found after
installation, run `uv tool update-shell` and open a new terminal.

```sh
colab-persist tools        # twelve MCP tools
colab-persist status       # account and runtime status, no allocation
colab-persist specs        # runtime types this account may use
```

Then, from the cloned repository directory, run the CUDA example twice:

```sh
colab-persist run examples/cuda_demo.py --project first-run
colab-persist run examples/cuda_demo.py --project first-run
colab-persist status
```

The first run reports count 1; the second restores it and reports count 2 with
previous count 1. Each successful run saves, verifies, and stops its VM, so the
final status should be inactive. No browser step is needed for either VM. These
runs consume Colab compute units. See [VALIDATION.md](VALIDATION.md).

Configuration lives in `~/.config/colab-persist/config.json`. Each account's
private credentials, Cloud project and runtime ledger live under
`~/.config/colab-persist/accounts/`, in a directory named by a hash, not the email.
A dedicated Ed25519 SSH key is created only if the selected key does not exist.
None of these files are uploaded by this tool.

## How Drive works

Colab Persist talks to Drive through the Drive API, not a Drive mount on the VM.

- **One permission, granted once.** `login` asks Google for `drive.file`: access to
  the files and folders colab-persist itself creates, and nothing else in your Drive.
- **Nothing long-lived on the VM.** Your sign-in stays on your computer. For each
  operation the VM receives an access token that expires within an hour and is
  limited to Drive. The local client renews it while you work.
- **Verified, not just flushed.** Every archive is uploaded with a resumable upload.
  Drive reports the stored size and SHA-256, and colab-persist compares them with the
  local archive before writing the JSON manifest, which is the commit marker.
  An interrupted upload leaves no file; an archive without a manifest is ignored.
- **Same account, always.** Every Drive token is checked against the selected Colab
  account on your computer before it is used or sent to the VM; a mismatch stops
  before anything is written.

Checkpoints live in `My Drive/Colab-Persist/projects/<project>/` as
`<id>.tar.gz` plus `<id>.json`, the same format as earlier versions. Because
`drive.file` only sees files colab-persist created, put dataset shards in Drive
through colab-persist too (see [LARGE_DATA.md](LARGE_DATA.md)), not through the
Drive website.

Want colab-persist to read anything in your Drive instead? Create a Desktop OAuth
client in your Cloud project and sign in with
`colab-persist login --email YOUR_GOOGLE_EMAIL --project YOUR_PROJECT_ID --client-id-file client_secret.json`.
That requests full Drive access through your own client. Google shows an
"unverified app" warning on that consent screen, and refresh tokens from apps left
in Testing mode expire after seven days. Repeat `--client-id-file` whenever you
select that account again; without it, `login` signs in again with `drive.file`.

## Accounts

Each Google account has its own private sign-in, Cloud project and runtime ledger.
`login` never changes your global gcloud login, gcloud's application default
credentials, or coding-agent logins.

```sh
colab-persist status       # selected account, managed runtime, total runtime count
colab-persist sessions     # every runtime on that account, including browser ones
```

`active: false` means no managed runtime. Check `active_runtimes: 0` to confirm the
account has no runtimes at all. Runtimes you opened in the browser appear in
`sessions` as unmanaged; colab-persist never adopts or deletes them.

To save another account's sign-in while keeping the current one selected:

```sh
colab-persist login --email OTHER_GOOGLE_EMAIL --project OTHER_PROJECT_ID --no-switch
```

To switch, save and stop the current account's runtimes, then log in:

```sh
colab-persist stop
colab-persist login --email OTHER_GOOGLE_EMAIL
colab-persist status
```

The switch refuses while the old account has any running Colab VM, or when that
cannot be verified. Cancelled or wrong-account sign-in leaves the selection
unchanged. `--reauth` requests fresh consent if credentials expired or were revoked.
Account switching never transfers files between Google Drives.

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
colab-persist stop                          # checkpoint + verify + shutdown
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
work normally. Host keys are pinned separately for each runtime. Starting a
replacement refreshes this alias; connecting to an old runtime fails rather than
quietly connecting to another VM.

Colab accepts one SSH session per runtime. colab-persist and the `colab-cuda` alias
therefore share one connection per runtime: an SSH control master at
`~/.ssh/cm-cp-<hash>`.
- **Running side by side:** `ssh colab-cuda`, `rsync` and colab-persist's own commands can run at the same time.
- **Lingering connection:** an interactive connection stays open for 5 minutes after its last session, and one that colab-persist opened stays open for 30 seconds.
- **The busy-session error:** if another client holds the runtime's SSH session, close it; `ssh -O exit colab-cuda` closes the shared one.

Select other hardware explicitly when starting a fresh runtime:

```sh
colab-persist specs
colab-persist start --gpu T4
colab-persist start --gpu A100 --high-mem
colab-persist start --tpu V6E1
colab-persist start --cpu
```

L4 remains the configured default. `specs` lists what your account is eligible for;
a request outside that list is refused before anything is allocated. The helper
never silently substitutes other hardware or changes a live runtime; save and stop it first.

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
| `COLAB_DATASET_ROOT` | `gdrive://Colab-Persist/datasets`, read-only shard source for `ShardCache` |
| `COLAB_DATASET_CACHE` | Disposable local shard cache, outside snapshots |
| `COLAB_DRIVE_TOKEN_FILE` | The VM's short-lived Drive token, for `colab_persist.drive_api` |
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

## Long jobs

Every `run` is a managed job that runs detached on the VM. The terminal follows its
log; pressing Ctrl+C only detaches your terminal, and the job keeps running.

```sh
colab-persist run ./train.py --source . --project experiment --checkpoint-seconds 600 --detach
colab-persist job --project experiment       # state, last checkpoint, latest log lines
colab-persist attach --project experiment    # follow again; save and stop when it ends
```

While a job runs, `status`, `sessions`, `ssh` and `job` work normally. `save`,
`stop` and `restore` refuse until the job ends, so a job is never saved half-written.
A file rewritten during a checkpoint is retried shortly and never archived torn.

The VM's Drive token lasts about an hour and is renewed whenever your computer
talks to the VM. If your computer is away longer, the job keeps running; periodic
checkpoints pause, and when the script exits the job waits up to 24 hours for you
to reconnect (`attach`, `job`, `status` or `mount`) before it finishes saving.
The VM is never stopped with unsaved results. Google can still reclaim a VM at any
time; the last verified checkpoint is then your recovery point.

## Checkpoints and shutdown

```sh
colab-persist status
colab-persist snapshots --project experiment   # straight from Drive, no VM
colab-persist save                            # checkpoint + verify; VM stays allocated
colab-persist restore --project experiment    # empty destination required
colab-persist stop
```

Each snapshot is an immutable `.tar.gz` plus a JSON manifest containing SHA-256
hashes for every saved file and the archive. The manifest is uploaded only after
Drive verifies the archive. Unchanged snapshots are reused without uploading again.
Restore downloads with resume, verifies the archive and every extracted file,
reapplies recorded file modes, rejects links and path traversal, and refuses to
overwrite a nonempty workspace. If the newest checkpoint is incomplete, `restore`
says so instead of silently picking an older one; choose one explicitly with
`--snapshot`. Checkpoints are retained until you remove them from Drive; there is
no automatic pruning. Workspace and source uploads default to a 5 GiB size cap
checked before hashing; oversized saves refuse shutdown.

Every checkpoint reports `durability: drive_upload_verified`. A successful save
also reports `drive_upload_verified: true` (and `drive_flushed: true` for scripts
written against 0.3). If Drive is unavailable, full, or the daily upload limit is
reached, the operation fails and leaves the VM allocated for recovery.

A runtime with no managed workspace has nothing to save. `stop` deletes it and
reports `nothing_to_save: true`. As with every stop, files outside managed
workspaces are not captured. `stop` deletes only the exact runtime it started,
never a replacement or a browser runtime. `run --keep-runtime` saves and verifies
but leaves the runtime allocated. Deleting a runtime in the Colab browser, or
Google's own reclamation, bypasses the save guard.

## MCP: local agents, remote CUDA

Keep Codex, Claude Code, or your preferred coding agent on your Mac. Their login
and configuration survive runtime deletion; the MCP tools operate the GPU worker.

### Claude Code: talk to your GPU from your Mac

After [installing and signing in](#install-and-sign-in), register the server once.
If you already have a custom `colab` skill, merge its instructions instead of
overwriting it. Run these commands from the cloned repository:

```sh
claude mcp add --scope user --transport stdio colab-persist -- "$(command -v colab-persist-mcp)"
mkdir -p ~/.claude/skills/colab
cp integrations/claude-code/colab/SKILL.md ~/.claude/skills/colab/SKILL.md
claude mcp get colab-persist
```

Then start Claude Code in your project folder and say, for example:

```text
/colab Check my account and whether any GPU is running.
/colab Run train.py on an L4, save checkpoints to Drive, and stop when finished.
/colab Download this dataset on Colab, keep durable shards in Drive, and use a bounded local cache.
```

Claude stays on the Mac and uses MCP and SSH for remote work. Downloads started on
the VM use Colab's network. No browser step is needed per VM, so Claude can carry a
whole run from start to stop. Long jobs run detached: Claude can start one with
`run_script(detach=true)` and check on it with `job_status`.

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

Tools: `runtime_status`, `runtime_specs`, `start_runtime`, `prepare_workspace`,
`run_script`, `job_status`, `attach_job`, `list_checkpoints`, `restore_workspace`,
`save_workspaces`, `safe_stop`, and `plan_dataset`. The terminal client uses the
same tools over real MCP stdio; `colab-persist tools` checks the connection. No
HTTP port is exposed and no credential appears in tool results.

Google also hosts its own Colab MCP endpoint (`https://colaboratory.googleapis.com/mcp`)
for creating runtimes and running code. Colab Persist is a local layer on the same
official API that adds Drive checkpoints, verified saves and safe shutdown.

## Upgrading from 0.3

```sh
colab-persist stop                                   # with 0.3, if a VM is running
uv tool install --force .                            # from the updated clone
colab-persist login --email YOUR_GOOGLE_EMAIL --reauth --project YOUR_PROJECT_ID
colab-persist import-legacy                          # once, needs a running VM
```

0.3 wrote checkpoints through Colab's own Drive mount into `My Drive/Colab-CUDA/`.
The `drive.file` permission cannot see those files, so 0.4 keeps its checkpoints in
`My Drive/Colab-Persist/` and `import-legacy` copies the newest checkpoint of each
old project across (`--all-snapshots` for every one, `--project NAME` to pick).
It mounts Drive the old way one last time: Colab shows its familiar Drive consent
in the browser once, and the import continues by itself after you approve. Any
running VM works; a workspace that holds only empty folders still restores the
imported checkpoint, and an empty workspace never replaces it. Every
archive is checked against its recorded SHA-256 before it is published. Nothing in
`Colab-CUDA/` is modified or deleted. Your configuration is migrated automatically,
and the old folder name is kept as `legacy_drive_folder`.

What changed: the unofficial transport (`google-colab-cli`, its internal API and
FUSE Drive mount) is gone, along with its large dependency tree. Runtimes are
managed through the official Colab REST API, SSH goes straight to the runtime's
`/colab/ssh` endpoint, and Drive is reached through the Drive API. `mount` now just
checks the Drive connection. Every v0.3 command and MCP tool keeps its name.

## What survives, and what cannot

Saved source files, binaries, outputs, application checkpoints, and setup recipes
survive in Drive. RAM, VRAM, running processes, root filesystem changes, and writes
that never reached Drive do not. An application must write restartable state
(for example model/optimizer state and the current step) to resume computation.
Use atomic replacement for application checkpoint files; copying a live multi-file
database is not a transactional backup.

Colab Pro does not guarantee a ten-hour session. Paid runtimes still have variable
idle and maximum-lifetime limits. This tool does not simulate activity or prevent
Google from reclaiming a VM. After an unexpected termination, start a replacement
and restore the last checkpoint that reached Drive. Recovery is explicit, so a
replayed workload cannot silently allocate repeated GPUs or duplicate side effects.

## Existing tools

| Project | What it provides | Where this project fits |
|---|---|---|
| [Colab REST API](https://developers.google.com/colab/api/reference/rest) | Official runtime management and Jupyter connection info | Used directly as the control plane |
| [Colab MCP endpoint](https://developers.google.com/colab/api/reference/mcp) | Google-hosted MCP tools for runtimes and code execution | A complementary agent interface without persistence |
| [Colab extension for VS Code](https://github.com/googlecolab/colab-vscode) | Editor connection to Colab runtimes, on the same API | Editor workflow; this project is terminal and agent workflow |
| [Official Colab CLI](https://github.com/googlecolab/google-colab-cli) | GPU selection, SSH, execution, file transfer and Drive mounting | Used by 0.3; now only for the one-time `import-legacy` mount |

This is a small, independent workflow layer for one selected account and named
runtime at a time, not an official Google product.

## Development and verification

```sh
uv sync --no-editable
uv run --no-editable python -m unittest discover -s tests -v
uv run --no-editable colab-persist tools
```

The suite runs offline against in-memory fakes of the Colab and Drive APIs. It covers
recovery into a new workspace, corruption, interrupted and unverified uploads,
list-after-write lag, credential and link exclusions, malicious archives, overwrite
protection, script errors, detached jobs and token expiry, active-job locks,
runtime-ledger safety, and refusal to stop after a failed save. Live GPU and Drive
checks consume compute units and require your own account; see
[VALIDATION.md](VALIDATION.md) for the maintainer's recorded verification.

Sources: [Colab REST API](https://developers.google.com/colab/api/reference/rest),
[Colab MCP endpoint](https://developers.google.com/colab/api/reference/mcp),
[Drive API uploads](https://developers.google.com/workspace/drive/api/guides/manage-uploads),
[Drive API scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth),
[Colab FAQ and runtime limits](https://research.google.com/colaboratory/faq.html),
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
