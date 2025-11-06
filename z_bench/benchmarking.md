## Benchmarking throughput

```bash
vllm bench throughput --model openai/gpt-oss-20b --input-len 2000 --output-len 1 --enforce-eager --dataset-name random --num-prompts 500 --random-range-ratio 0 --max-num-batched-tokens 8192 --data-parallel-size 2 --tensor-parallel-size 1
```

## Benchmarking latency

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm bench latency --model openai/gpt-oss-20b --input-len 128000 --output-len 1 --enforce-eager --batch-size 16 --num-iters-warmup 10 --num-iters 100 --max-num-batched-tokens 8192 --data-parallel-size 1 --tensor-parallel-size 2 --enforce-eager --max-model-len 220000 --load-format dummy
```

## Benchmarking serving

```bash
vllm serve openai/gpt-oss-20b --disable-log-requests --data-parallel-size 2 --tensor-parallel-size 1 --pipeline-parallel-size 1 --max-model-len 5000 --enforce-eager --gpu-memory-utilization 0.8 --max-num-batched-tokens 8192 --no-enable-prefix-caching --max_num_seqs 10 --block-size 16 --port 8009

vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path /nfs/stak/users/gaosho/hpc-share/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --model openai/gpt-oss-20b --num-prompts 200 --test-real-world-workload --port 8009 --request-rate 1
```
