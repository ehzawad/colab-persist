# Large datasets and LoRA training

The dataset layer supports a corpus larger than the VM disk by staging individual
verified shards. **This is storage infrastructure, not a validated 1 TB training
pipeline.** No terabyte transfer or Qwen training benchmark has been performed.

## Separate the three kinds of files

| Files | Location | Recovery |
|---|---|---|
| Immutable corpus shards | Drive `Colab-CUDA/datasets/<name>/<version>/` | Read again from Drive as needed |
| Disposable dataset working set | VM `/content/colab-persist-cache/datasets/` | Rebuild from the manifest; never included in workspace archives |
| Base model/download cache | VM `/content/colab-persist-cache/huggingface/` | Download the pinned model revision again |
| Source, manifest, small training checkpoints | Managed project `workspace/` | Existing Drive workspace snapshots |

Default dataset cache: **40 GiB**, with **20 GiB free disk reserved** at transfer
time. These are starting budgets, not an assertion about available Colab disk.
The reserve must also accommodate model downloads, temporary files, activations
offloaded to disk and checkpoint archives. Other processes can consume disk; the
cache checks space before and during each copy, but cannot reserve it system-wide.

Google recommends avoiding many small reads from mounted Drive and using local
VM storage for active data. More Drive capacity does not enlarge VM disk. See the
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Prepare immutable shards once

Start with approximately 256 MiB–1 GiB shards as a tuning choice. Use independently
readable Parquet files for tabular/trajectory data, or another format your loader
can read in bounded batches. A single 1 TB archive cannot fit in a 40 GiB cache.
Partition very large file counts into subdirectories. Do not rewrite a shard
under an existing version; publish a new version and manifest.

Manifest paths are relative to `Colab-CUDA/datasets/`, **not the manifest file**:

```json
{
  "schema": 1,
  "name": "agent-trajectories",
  "version": "v1",
  "shards": [
    {
      "path": "agent-trajectories/v1/part-00000.parquet",
      "size": 536870912,
      "sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_SHA256"
    }
  ]
}
```

The placeholder is deliberately invalid. Record real sizes and SHA-256 values
while generating shards; otherwise computing them requires one full read of the
corpus. The planner never silently scans or hashes 1 TB of Drive data.

```sh
colab-persist dataset-plan ./corpus.json --cache-gib 40 --reserve-gib 20
```

This uses the `plan_dataset` MCP tool and reads only the small local manifest. It
reports logical/unique bytes, largest shard, budgets and a canonical manifest
fingerprint. It rejects malformed manifests, unsafe paths, duplicate paths and a
shard larger than the cache. It does not allocate a GPU or verify Drive files,
Drive capacity, current VM free space, bandwidth or model memory requirements.

## Read a bounded working set on the VM

Managed scripts and new SSH login shells can import `colab_persist.datasets`.
The module uses only the Python standard library; install a reader such as
PyArrow separately through your pinned training environment when using Parquet.

```python
import os
from colab_persist.datasets import ShardCache, load_manifest, plan

manifest = load_manifest("corpus.json")
cache = ShardCache(
    manifest,
    dataset_root=os.environ.get("COLAB_DATASET_ROOT", "/content/drive/MyDrive/Colab-CUDA/datasets"),
    cache_root=os.environ.get("COLAB_DATASET_CACHE", "/content/colab-persist-cache/datasets"),
    cache_bytes=40 * 1024**3,
    reserve_bytes=20 * 1024**3,
)
dataset_id = plan(manifest)["manifest_sha256"]

for shard_index, shard in enumerate(manifest.shards):
    with cache.lease(shard) as local_file:
        # Example reader; the training application handles batching and progress.
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(local_file).iter_batches(batch_size=128):
            consume_batch(batch)  # your training/data-loader integration
```

Keep the lease open until all readers, workers and memory maps finish. A path
returned by a closed lease may later be evicted. Never modify a leased file.
Cached blob names are hashes without filename extensions; configure readers
explicitly with the format. Batching is still the reader's responsibility.

Copies are sequential and on demand. A copy goes to `.partial`, is checked against
the manifest's size and SHA-256, and becomes visible only after successful
verification. Interrupted copies restart from the beginning of that shard. This
release does not implement range resume, asynchronous prefetch, automatic retries
for Drive quota failures or sharding an existing large file.

Least-recently-used cached blobs are evicted when space is needed. Active leases
are protected with process locks. All manifests sharing one cache root share the
same byte budget; a conflicting budget is refused. Cache hits verify the local
blob without reading the original Drive file. The cache never deletes or writes
source data. Its byte budget covers blobs; small manifest/lock metadata is extra.

## Keep training state complete

The workspace/source cap is **5 GiB by default**, checked before hashing files.
Snapshots and restore also check scratch-space headroom. An oversized save fails
and prevents automatic shutdown. This prevents accidental full-corpus backups;
it is not a substitute for capacity planning.

You can deliberately raise the cap after measuring checkpoint size:

```sh
colab-persist configure --email YOUR_GOOGLE_EMAIL --snapshot-limit-gib 8
```

Workspace archives are still full snapshots, and old archives are retained. A
log change can cause an unchanged adapter to be archived again. Measure storage
growth and choose a suitable checkpoint interval; multi-gigabyte or frequent
checkpoints need an incremental uploader and retention policy before scale-up.
The tool does not automatically prune your only recovery copy.

For actual training resume, save a consistent generation containing adapter and
configuration, optimizer, scheduler, RNG/scaler state as applicable, training step,
dataset fingerprint and loader cursor, plus model/tokenizer revisions and training
configuration. Saving only an adapter supports inference; it is not a full
training restart. See [Transformers Trainer](https://huggingface.co/docs/transformers/main/main_classes/trainer).

Write every generation into a directory ending in `.partial` and publish it only
after all writers finish. `atomic_checkpoint` provides that publication boundary:

```python
from pathlib import Path
from colab_persist.checkpoints import atomic_checkpoint

with atomic_checkpoint(
    Path("outputs/checkpoints"), "step-1000",
    required_files=["adapter_model.safetensors", "optimizer.pt", "resume.json"],
    metadata={"dataset_manifest_sha256": dataset_id, "step": 1000},
) as staging:
    save_complete_training_state(staging)  # application-specific; finish all writes here
```

This checks publication and required files, not the semantic completeness of your
trainer state. It is not a drop-in Trainer callback. Do not leave a second copy of
live, half-written checkpoints elsewhere in the workspace. Restore should select
a completed generation and verify its dataset/model revision before training.

Streaming resume must include the data loader. Hugging Face documents that a
streaming shuffle buffer loses buffered examples on restart; a shard number alone
does not reproduce the same training sequence. Use deterministic ordering or a
loader that saves the full required state. See [streaming state](https://huggingface.co/docs/datasets/stream).
Online agent rollouts also need task/rollout IDs and explicit handling of partial
episodes and external tool side effects; a filesystem snapshot cannot recreate them.

Periodic snapshots through the native Drive mount remain `pending_drive_flush`.
Only the final flush-and-unmount is acknowledged by this tool. Unexpected VM loss
can lose recent writes. This release does not promise a maximum loss interval or
exact recovery after forced termination; direct acknowledged cloud uploads remain
future work.

## Qwen3.5-9B on an L4

The [official Qwen3.5-9B model](https://huggingface.co/Qwen/Qwen3.5-9B) includes a
vision encoder. Its large advertised context is not a training-memory guarantee.
[NVIDIA lists 24 GB on L4](https://www.nvidia.com/en-us/data-center/l4/).
[Unsloth's Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) reports
about 22 GB for optimized 9B BF16 LoRA and advises against QLoRA for this family.
That is a maintainer result, not a measurement from this project. The same guide
requires Transformers v5; pin and validate a compatible training stack.

Keep LoRA explicit. Start with a small representative sample, batch size 1 and a
modest context, then measure peak VRAM and tokens/sec. Long agent trajectories,
vision batches and online RL may require a larger GPU. L4 stays the default;
the helper never silently changes hardware or quantization.

One terabyte is a storage quantity, not a ten-hour training plan. Estimate tokens
after filtering/packing, benchmark throughput, and choose a token/step budget.
There is no measured full-corpus training-time estimate yet.

Before relying on this for a long run, validate a real multi-shard sample, an
optimizer/data-loader resume across a replacement VM, transfer throughput and
checkpoint storage growth. Those need your dataset format/location and trainer.
