# Verification record

Tested on macOS with Python 3.14.7, Google Colab CLI 0.7.4, MCP Python SDK 2.2.0,
and NVIDIA L4 Colab runtimes on 2026-09-27 (Asia/Dhaka).

## Version 0.2.1 fresh installation and recovery

The installed tool, both command launchers, and its Python environment were removed.
The public HTTPS repository was cloned to a new directory and installed with
`uv tool install --no-cache .` on macOS 27.0, Apple silicon. The original tool
configuration was moved aside; setup created a fresh configuration and a new
mode-0600 SSH key. The fresh installation discovered all nine MCP tools and
provisioned an L4. Existing uv/Python/gcloud/OpenSSH and Google login credentials
were retained, so this is a clean application installation, not a factory-reset OS
or a newly authenticated Google account.

The public 0.2.0 checkout passed all 54 tests. Review of that onboarding flow found
that invalid local inputs could allocate a GPU before failing. Version 0.2.1 moves
those checks ahead of allocation and adds eight regression tests (62 total),
covering CLI and MCP paths without cloud calls.

The tool was then uninstalled again and reinstalled with caching disabled from a
second fresh public clone, at commit
`4e90e281e94f449731695dc889f41bb5f161577a` (0.2.1). All eight installed Python modules
matched that clone. Fresh configuration and SSH key creation succeeded again;
all **62 tests** passed. An actual invocation with a missing script failed before
adding any Colab runtime events. The public commit's
[test workflow](https://github.com/ehzawad/colab-persist/actions/runs/36266222596)
and dependency workflow passed.

### Live recovery from the fresh installation

Using project `fresh-install-20260927` and the freshly installed command:

1. Ran `colab-persist run examples/cuda_demo.py --project fresh-install-20260927`.
   The L4 compiled CUDA and verified all 256 GPU results. The counter became 1.
2. The command saved seven files, confirmed Drive's flush and stopped the VM.
   A separate status check confirmed the runtime was inactive.
3. Ran `colab-persist start --project fresh-install-20260927`. A different L4
   endpoint restored all seven files with the same snapshot ID and archive hash.
   An SSH command verified NVIDIA L4 (23,034 MiB) and the restored counter of 1.
4. Ran the small live storage checks described below on the replacement VM.
5. Repeated the CUDA example. All 256 results passed again, and the counter became
   2 with previous count 1. The final snapshot contained ten files, including the
   storage test result and completed checkpoint.
6. The second save confirmed Drive's flush and stopped the replacement VM.
   Status again confirmed no active test runtime. Both runs exited successfully
   with no periodic checkpoint errors.

| Receipt | First VM | Replacement VM |
|---|---|---|
| Snapshot | `20260926T200023455759Z-d772c537` | `20260926T200735758360Z-092ae3a1` |
| Files | 7 | 10 |
| Archive bytes | 356,567 | 357,238 |
| Final durability | `drive_flush_confirmed` | `drive_flush_confirmed` |
| Drive flush time (UTC) | 2026-09-26 20:00:32 | 2026-09-26 20:07:45 |

First archive SHA-256:
`02a9ea0f97f831822d115bed619169acc16d1cb49c0abc385af64b77d51bb969`.
Second archive SHA-256:
`14f4b8de330df3537da4875a83c6fbbac16934e85ab3406ac0a8ec8374724669`.

### Small live Drive storage checks

The replacement VM read three 64-byte fixture shards from the real Drive mount
using a 128-byte cache with a 20 GiB free-space reserve. Assertions verified:

- Hash-verified reads, same-process active-lease protection, least-recently-used
  eviction, the cache size bound and unchanged Drive source files.
- An incomplete checkpoint generation was excluded from the workspace inventory.
- A completed generation with its required file was included.

The smoke result reported `pending_drive_flush` when written; the subsequent
successful final save above confirmed the Drive flush for the workspace and
fixture writes. This is a 192-byte storage correctness check, not a throughput or
scale test. Cross-process reader protection is covered by local tests. No live
1 TB transfer, Qwen training, multi-gigabyte model checkpoint, optimizer/data-loader
recovery or abrupt VM termination was tested.

Original tool settings and dedicated known-hosts were restored after the test.
The default remained L4; local coding-agent login and MCP registration were retained.

## Version 0.2.0 storage checks

On 2026-09-27, all **54 tests** passed in an isolated Python 3.14.7 environment.
The wheel and source distribution built successfully. This adds verification of:

- Bounded shard staging, eviction and cross-process active-reader protection.
- Interrupted copy cleanup, source mutation, corruption and low-disk failures.
- Manifest path validation, symlink/hardlink rejection and source/cache separation.
- A synthetic 1 TB manifest through the real MCP tool. Only metadata was processed;
  no terabyte dataset was created or transferred.
- Atomic publication of required checkpoint files, excluding incomplete generations.
- Workspace/source size caps, restore headroom and deployed runtime module imports.
- Preserving existing GPU/session/key configuration when changing only the size cap.

These new storage paths have not been exercised against a live terabyte-scale
Drive corpus, a Qwen trainer, or an abrupt Colab termination. The runtime deployment
archive was imported in an isolated local process; the original 0.2.0 storage
validation did not allocate a GPU. The later 0.2.1 test above exercised small live
fixtures. LoRA fit, throughput and complete optimizer/data-loader recovery remain
application-level tests to perform with an actual dataset and training recipe.

## Version 0.1.0 automated checks

- 17 local tests passed, including a real stdio MCP client/server handshake.
- Wheel and source distribution built successfully.
- Failure tests covered corrupt archives, incomplete checkpoint publication,
  archive path traversal, credential and symlink exclusions, overwrite refusal,
  a script exiting with an error, active-job locking, failed Drive flushes, and
  workspace changes during or after a flush.
- The mount test checks the filesystem even when the upstream CLI returns zero
  after a failed kernel operation.

## Original version 0.1.0 live GPU and recovery test

1. Allocated an L4 through the official CLI and connected over SSH.
2. Mounted Google Drive using the same Google account as Colab.
3. Ran `examples/cuda_demo.py`: `nvcc` compiled the kernel and all 256 GPU results
   passed verification. The persisted run counter became 1.
4. Saved seven workspace files into a checksummed archive. Colab's Drive flush
   completed, and the first VM was stopped.
5. Allocated a different L4 VM and authorized its Drive mount.
6. Restored the seven files, verified the archive and file hashes, compiled and
   ran CUDA again. The counter became 2 and reported that it restored count 1.
7. The second save completed with `drive_flush_confirmed`.

The first archive SHA-256 was
`14bf5f8a57a0fc33e99011d1e86c80d8ec9b86c4c884ab8a6b7b253d2569f390`.
The second was
`d1bfb4df2f4289ed40500bc9dbc20c52748c5071dc1bd548346fc0a5a124fe39`.
These are test artifacts, not credentials or account identifiers.

## Observed limits

The first Drive mount on both new runtimes failed after browser consent; retrying
with the propagated credentials succeeded. The helper now retries once and checks
that a real Drive mount exists. It does not claim an upstream mount is reliable
merely because its CLI process exited successfully.

This verifies saved-file recovery across actual VM deletion. It does not verify
ten-hour runtime availability, abrupt-termination loss bounds, every GPU type,
multi-gigabyte model checkpoints, or application-level consistency of live databases.
The final flush provides stronger evidence than a read through Drive's local cache.
