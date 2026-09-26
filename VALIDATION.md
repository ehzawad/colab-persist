# Verification record

Tested on macOS with Python 3.14.7, Google Colab CLI 0.7.4, MCP Python SDK 2.2.0,
and NVIDIA L4 Colab runtimes on 2026-09-27 (Asia/Dhaka).

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
archive was imported in an isolated local process; no GPU was allocated for this
update. LoRA fit, throughput and complete optimizer/data-loader recovery remain
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

## Live GPU and recovery test

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
