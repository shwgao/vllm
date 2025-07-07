# Debug Log: vLLM Issues

## 1. Safetensor Loading Performance Issue ⏳
- **Problem:** Safetensor checkpoint loading shows exponential slowdown - from 1.8s per shard initially to 342s per shard by the third shard (7 total shards).
- **Solution:** Need to investigate memory pressure, disk I/O, and network storage latency on HPC system.
- **Status:** investigating.
