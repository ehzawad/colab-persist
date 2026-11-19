---
name: colab
description: Use colab-persist to run CUDA, dataset downloads, and training on Google Colab from local Claude Code, with verified Drive checkpoints, GPU/TPU selection, detached jobs, and account switching. Use for Colab work, not unrelated local development.
---

# Colab from local Claude Code

Keep Claude Code and its login on this computer. Use the `colab-persist` MCP
server and CLI to operate the remote runtime. An SSH shell is on the Colab VM;
downloads launched there use Colab's network. A local Bash download uses this
computer's network. Do not install Claude on the VM or transfer its credentials
as part of this workflow.

colab-persist 0.4 runs on Google's official Colab API. Drive is reached through the
Drive API with a short-lived token: there is **no per-VM browser consent** and no
Drive mount to authorize. Checkpoints live in `My Drive/Colab-Persist/projects/`.

## Choose the operation

Start with `runtime_status` when no managed job is running. It verifies the
selected Google account. `active` describes the managed runtime;
`active_runtimes` counts all runtimes on that account, including ones opened in the
browser. Use `colab-persist sessions` to list them. A status request never calls
`start_runtime`. Start compute only when the user's requested work needs it.
The Claude subscription identity and Google account are independent and may
differ; compare the Google identity only with the user's intended Colab account.

| Task | Interface |
|---|---|
| See eligible hardware and the Colab tier | MCP `runtime_specs` |
| Start or reuse compute | MCP `start_runtime(gpu=...)`, or `tpu=...` / `cpu=true`; omit to use the configured default, initially L4 |
| Restore/create a project | MCP `prepare_workspace(project=...)` |
| Execute a short Python script with persistence | MCP `run_script` with a local `script_path` and project |
| Run a long training/download job | MCP `run_script(..., detach=true)`, then `job_status`; or local CLI `colab-persist run ... --detach` |
| Follow a job to completion, then save/stop | MCP `attach_job(project=...)` or CLI `colab-persist attach --project ...` in a background Bash task |
| Inspect GPU/disk/processes | CLI `colab-persist ssh -- nvidia-smi`, `... -- df -h /content`, or `... -- ps aux` |
| Plan a large corpus | MCP `plan_dataset` with a local manifest path; no data transfer or GPU allocation |
| List/restore snapshots | MCP `list_checkpoints` (no VM needed) / `restore_workspace` |
| Save and stop | MCP `safe_stop`; success requires `drive_upload_verified: true` |

All `script_path`, `source_directory`, and `manifest_path` MCP arguments refer
to files on this computer. Write source locally, then upload/run it through the
wrapper. The script executes in `/content/colab-persist/<project>/workspace` on
the VM. Only pass `source_directory` when the job actually needs the project
folder. Use absolute local paths and one stable project name for resume.

`start_runtime` connects Drive automatically (`drive_connected: true`). If it
reports a missing Cloud project, the user runs `colab-persist setup --project
PROJECT_ID` once. If credentials predate 0.4 or expired, the user runs
`colab-persist login --email EMAIL --reauth` in an interactive local terminal and
approves Google's consent page; OAuth codes never belong in the conversation.
Requests for hardware outside `runtime_specs` eligibility are refused before
anything is allocated; never substitute hardware silently.

## Long jobs and remote commands

Every `run_script` is a detached managed job on the VM. With `detach=true` the
call returns right after launch; follow with `job_status` (state, last checkpoint,
recent log lines) at sensible intervals, or `attach_job` to wait for completion.
An interrupted call or Ctrl+C never stops the job. MCP hosts may impose short tool
deadlines, so prefer `detach=true` plus `job_status` for anything long, or run
`colab-persist attach --project NAME` as a monitored background Bash task.

```sh
colab-persist run ./train.py --source . --project experiment --checkpoint-seconds 600 --detach -- --epochs 10
colab-persist job --project experiment
colab-persist attach --project experiment
```

Choose the checkpoint interval from measured save size/time and the acceptable
loss window; 600 seconds above is an example. The default is 60 seconds. Snapshots
are full archives retained without automatic pruning, and the default workspace
cap is 5 GiB. Size the trainer's saved state and Drive growth before a long run.
If necessary, adjust the cap deliberately with
`colab-persist configure --email SELECTED_EMAIL --snapshot-limit-gib N` after
checking capacity; do not increase it just to hide an oversized data cache.

While a job runs, `runtime_status`, `job_status`, `sessions` and SSH work; save,
stop and restore refuse until the job ends. Inspect through SSH as needed:

```sh
colab-persist ssh -- nvidia-smi
colab-persist ssh -- tail -n 40 /content/colab-persist/experiment/workspace/outputs/run.log
```

For noninteractive SSH commands that need the CUDA/helper environment, use:

```sh
colab-persist ssh -- bash -lc 'source /etc/profile.d/colab-cuda.sh; nvcc --version'
```

Raw SSH commands do not get periodic workspace checkpoints. Use managed scripts
for substantive training/download work. A timed-out client does not prove the
remote job stopped: check `job_status` before retrying; never start duplicate
training blindly.

The VM's Drive token lasts about an hour and is renewed by any call from this
computer (`job_status` included). If the computer is away longer, periodic
checkpoints pause; when the script exits, the job waits up to 24 hours in state
`awaiting_drive` for a reconnect, then saves. `exited_unsaved` or `lost` means the
final save did not happen: the VM was left running, so fix the cause and run
`safe_stop`. Report success only with the final save receipt
(`drive_upload_verified: true`) and the script's exit code.

## Large data and recovery

Download data from inside the VM, in independently resumable shards. Durable
corpus files belong under `My Drive/Colab-Persist/datasets/<name>/<version>/` and
must be uploaded through colab-persist (`python3 -m colab_persist.drive_api put
LOCAL datasets/NAME/VERSION/FILE` on the VM); files added through the Drive website
are invisible to it. Stage only a bounded working set in
`/content/colab-persist-cache/datasets/`; base-model caches belong outside the
workspace too. Do not put a 1 TB dataset into a workspace snapshot or assume the VM
has the same capacity as Drive. Drive accepts at most 750 GB of uploads per day.

`plan_dataset` validates metadata only. Training code must use
`colab_persist.datasets.ShardCache` explicitly (its default source is
`COLAB_DATASET_ROOT`, `gdrive://Colab-Persist/datasets`); it is not an automatic
downloader. Check VM disk, Drive capacity and real shard sizes before large
transfers. See the project's [large-data integration guide](https://github.com/ehzawad/colab-persist/blob/main/LARGE_DATA.md).

Write restartable training state under `COLAB_OUTPUT_DIR` using atomic completed
generations: model/adapter, optimizer/scheduler, RNG and data-loader position as
required by the trainer. File restore does not resume RAM, processes, packages,
or training automatically. Keep a dependency/bootstrap recipe in the workspace.
No 1 TB transfer or Qwen training benchmark has been validated by this project.

## Accounts, hardware and upgrades

Use `colab-persist login --email EMAIL --project PROJECT_ID` only when the user
requests that account. Cached logins switch without another browser flow; first
sign-in needs the terminal handoff. Switching refuses while the old account has
any active runtimes. It does not move data between Google Drives.

The wrapper cannot resize a live runtime. Save/stop before changing hardware, and
do not silently substitute it. Browser-opened runtimes are listed but never adopted
or deleted. Google can still reclaim a paid runtime; recover from the last
verified checkpoint without promising a fixed session length.

Checkpoints from colab-persist 0.3 (in `My Drive/Colab-CUDA/`) are copied once with
`colab-persist import-legacy` on a running VM. It shows Colab's Drive consent page
one final time; the user approves it in the browser and the import continues by
itself. Run it in a background Bash task and tell the user to approve the page.
