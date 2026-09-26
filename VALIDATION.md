# Verification record

Tested on macOS with Python 3.14.7, Google Colab CLI 0.7.4, MCP Python SDK 2.2.0,
and NVIDIA L4 Colab runtimes on 2026-09-27 (Asia/Dhaka).

## Automated checks

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
