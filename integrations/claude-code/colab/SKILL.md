---
name: colab
description: Use colab-persist to run CUDA, dataset downloads, and training on Google Colab from local Claude Code, with Drive checkpoints, GPU selection, and account switching. Use for Colab work, not unrelated local development.
---

# Colab from local Claude Code

Keep Claude Code and its login on this computer. Use the `colab-persist` MCP
server and CLI to operate the remote GPU. An SSH shell is on the Colab VM;
downloads launched there use Colab's network. A local Bash download uses this
computer's network. Do not install Claude on the VM or transfer its credentials
as part of this workflow.

## Choose the operation

Start with `runtime_status` when no managed job is running. It verifies the
selected Google account. `active` describes the managed session;
`active_runtimes` counts all runtimes on that account. Use
`colab-persist sessions` to list them. A status request never calls
`start_runtime`. Start compute when the user's requested work needs it.
The Claude subscription identity and Google account are independent and may
differ; compare the Google identity only with the user's intended Colab account.

| Task | Interface |
|---|---|
| Start or reuse compute | MCP `start_runtime(gpu=...)`; omit GPU to use the configured default, initially L4 |
| Restore/create a project | MCP `prepare_workspace(project=...)`, after Drive is mounted |
| Execute a short Python script with persistence | MCP `run_script` with a local `script_path` and project |
| Run a long training/download job | Local CLI `colab-persist run` in a monitored Claude background Bash task |
| Inspect GPU/disk/processes | CLI `colab-persist ssh -- nvidia-smi`, `... -- df -h /content`, or `... -- ps aux` |
| Plan a large corpus | MCP `plan_dataset` with a local manifest path; no data transfer or GPU allocation |
| List/restore snapshots | MCP `list_checkpoints` / `restore_workspace`; Drive must be mounted |
| Save and stop | MCP `safe_stop`; success requires confirmed Drive flush |

All `script_path`, `source_directory`, and `manifest_path` MCP arguments refer
to files on this computer. Write source locally, then upload/run it through the
wrapper. The script executes in `/content/colab-persist/<project>/workspace` on
the VM. Only pass `source_directory` when the job actually needs the project
folder. Use absolute local paths and one stable project name for resume.

`start_runtime` does not authorize Drive or prepare a project. When
`drive_mounted` is false, have the user run `colab-persist mount` in an interactive
local terminal and complete Google's consent for the selected account. OAuth
codes belong in that terminal, never in the conversation or model tool input.
Then recheck status and call `prepare_workspace`. Do not run the interactive
mount command in a background/noninteractive Bash task. The same handoff applies
if first-time account login needs browser consent.

## Long jobs and remote commands

Once Drive is mounted, a long job can use:

```sh
colab-persist run ./train.py --source . --project experiment --checkpoint-seconds 600 -- --epochs 10
```

Use Claude's background Bash task support to keep the local client running and
monitor its task output. Keep the Mac awake and connected for this supervised
workflow. The CLI supports long calls; MCP hosts may impose shorter deadlines.
Neither mode guarantees Colab uptime or that a job survives local interruption.
Do not use tmux or simulated activity to keep a VM alive.

Choose the checkpoint interval from measured save size/time and the acceptable
loss window; 600 seconds above is an example. The default is 60 seconds. Snapshots
are full archives retained without automatic pruning, and the default workspace
cap is 5 GiB. Size the trainer's saved state, Drive growth, and VM archive space
before a long run. If necessary, adjust the cap deliberately with
`colab-persist configure --email SELECTED_EMAIL --snapshot-limit-gib N` after
checking capacity; do not increase it just to hide an oversized data cache.

The managed run holds the local operation lock. `runtime_status`, save, stop,
and other persistence operations can report that another operation is active.
While it runs, inspect through SSH instead:

```sh
colab-persist ssh -- nvidia-smi
colab-persist ssh -- tail -n 40 /content/colab-persist/experiment/workspace/outputs/run.log
```

For noninteractive SSH commands that need the CUDA/helper environment, use:

```sh
colab-persist ssh -- bash -lc 'source /etc/profile.d/colab-cuda.sh; nvcc --version'
```

Raw SSH commands do not get periodic workspace checkpoints. Use managed scripts
for substantive training/download work, and save their outputs appropriately.
An interrupted or timed-out client does not prove the remote job stopped. Inspect
the existing VM before retrying; never start duplicate training blindly.

`run_script` and CLI `run` save, flush/unmount Drive, and stop by default, even
after script failure if saving succeeds. `stop_after=false` / `--keep-runtime`
leaves compute allocated but still unmounts Drive: remount before another save.
On a save failure, preserve the running VM for recovery. Use the wrapper's
`safe_stop` or `colab-persist stop`, not raw `colab stop`.
Periodic snapshots may report `pending_drive_flush`; hashes and manifests alone
do not confirm Drive flush. Require the final `drive_flush_confirmed` receipt.
Unexpected VM loss can lose recent writes, including downloaded shards on a
mounted Drive that had not finished flushing.

## Large data and recovery

Download data from inside the VM, in independently resumable shards. Durable
corpus files belong under the mounted Drive's `Colab-CUDA/datasets/<name>/<version>/`.
Stage only a bounded working set in `/content/colab-persist-cache/datasets/`;
base-model caches belong outside the workspace too. Do not put a 1 TB dataset
into a workspace snapshot or assume the VM has the same capacity as Drive.

`plan_dataset` validates metadata only. Training code must use
`colab_persist.datasets.ShardCache` explicitly; it is not an automatic downloader.
Check VM disk, Drive capacity and real shard sizes before large transfers.
See the project's [large-data integration guide](https://github.com/ehzawad/colab-persist/blob/main/LARGE_DATA.md)
for the manifest schema, cache API, and checkpoint publication API.

Write restartable training state under `COLAB_OUTPUT_DIR` using atomic completed
generations: model/adapter, optimizer/scheduler, RNG and data-loader position as
required by the trainer. File restore does not resume RAM, processes, packages,
or training automatically. Keep a dependency/bootstrap recipe in the workspace.
Check exit status and the save receipt before reporting success. No 1 TB transfer
or Qwen training benchmark has been validated by this project.

## Accounts and GPU changes

Use `colab-persist login --email EMAIL` only when the user requests that account.
Cached logins switch without another browser flow; first login/reauthentication
may require the terminal handoff. Switching refuses while the old account has
any active runtimes. It does not move data between Google Drives.

The wrapper cannot resize a live GPU. Save/stop before changing GPU type, and
do not silently substitute hardware. Native `colab` uses separate credentials
and session mappings; its SSH command can allocate a VM. Use the wrapper for
this workflow. Google can still reclaim a paid runtime; recover from the last
verified checkpoint without promising a fixed session length.
