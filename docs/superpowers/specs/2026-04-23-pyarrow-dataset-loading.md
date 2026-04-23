# Switch Dataset Loading from HuggingFace Streaming to Direct PyArrow

## Problem

BlueVela compute nodes cannot reach HuggingFace, and the `datasets` library's streaming mode for local parquet files reads at only 31 rows/s. Non-streaming mode tries to build a disk cache and exceeds the disk quota. Direct pyarrow reads achieve 543,000 rows/s (17,000x faster).

## Benchmarks (BlueVela login node)

| Approach | Speed | Notes |
|----------|-------|-------|
| `datasets` streaming (local parquet) | 31 rows/s | Current implementation, unusably slow |
| `datasets` non-streaming | ~90K rows/s | Requires disk cache, exceeded quota |
| Direct `pyarrow.parquet.read_table` | 543K rows/s | No cache needed, 17,000x faster than streaming |

## Dataset

- **Path:** `/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT`
- **Files:** 140 parquet files (`000_00000.parquet` through `013_00009.parquet`)
- **Schema:** Single `text` column, ~726K rows per file
- **Total size:** 267GB on disk, ~100B tokens of text

## Design

### File-level sharding

Instead of the current approach where `ds.shard(N, i)` iterates ALL rows but yields only every Nth (wasting (N-1)/N of I/O), assign whole parquet files to workers using round-robin:

```
file index % total_shards == shard_index
```

With 140 files and 4 shards (e.g., 2 ranks x 2 workers), each shard gets 35 files with zero wasted I/O.

### Two code paths in FineWebEduDataset

1. **Local pyarrow path** (when `DATASET_PATH` is set): Read parquet files directly with `pyarrow.parquet.read_table()`, one file at a time. Tokenize text, feed into existing rolling buffer. Loop infinitely for training.

2. **HuggingFace streaming fallback** (when `DATASET_PATH` is empty): Existing `datasets.load_dataset` code, preserved unchanged for environments with internet access.

### Memory profile

- One parquet file's text column: ~500MB-1.5GB in Arrow format
- Explicit `del table` / `del text_column` before loading next file
- With `num_workers=4`: ~6GB peak across all workers, well within node budget

## Files Changed

| File | Change |
|------|--------|
| `training/1b_poc_fineweb.py` | Rewrite `FineWebEduDataset`: add `_get_parquet_files()`, `_iter_parquet()`, `_iter_streaming()`, rewrite `__iter__()`, remove `_load_dataset()` |
| `training/requirements.txt` | Add `pyarrow>=15.0.0` (explicit, already transitive dep of `datasets`) |

## What stays unchanged

- `__init__` signature and env var interface (`DATASET_PATH`, `DATASET_SUBSET`)
- Output contract: yields `(input_ids, target_ids)` tensors of shape `[seq_len]`
- Rolling buffer tokenization logic
- DataLoader construction (`num_workers=4`, `pin_memory=True`)
- Training loop, checkpointing, ClearML, generation test
- `training/3b_fine_web_edu.py` (untouched)
