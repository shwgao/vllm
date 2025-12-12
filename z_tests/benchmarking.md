## Benchmarking throughput

```bash
HF_HUB_OFFLINE=1 vllm bench throughput --model meta-llama/Meta-Llama-3-70B-Instruct --input-len 2000 --output-len 256 --enforce-eager --dataset-name random --num-prompts 1000 --random-range-ratio 0 --max-num-batched-tokens 8192 --data-parallel-size 4 --tensor-parallel-size 2
```

## Benchmarking latency

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm bench latency --model openai/gpt-oss-20b --input-len 128000 --output-len 1 --enforce-eager --batch-size 16 --num-iters-warmup 10 --num-iters 100 --max-num-batched-tokens 8192 --data-parallel-size 1 --tensor-parallel-size 2 --enforce-eager --max-model-len 220000 --load-format dummy
```

## Benchmarking serving


### Launching the server

```bash

# when run on defiant-compute
export HF_HUB_OFFLINE=1

# launch Meta-Llama-3-70B-Instruct TP mode
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 HF_HUB_OFFLINE=1 vllm serve meta-llama/Meta-Llama-3-70B-Instruct --disable-log-requests --data-parallel-size 1 --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-model-len 128000 --enforce-eager --gpu-memory-utilization 0.8 --max-num-batched-tokens 2048 --no-enable-prefix-caching --max_num_seqs 1000 --block-size 16 --port 8007

# launch Meta-Llama-3-70B-Instruct DP mode
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 HF_HUB_OFFLINE=1 vllm serve meta-llama/Meta-Llama-3-70B-Instruct --disable-log-requests --data-parallel-size 4 --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-model-len 128000 --enforce-eager --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096 --no-enable-prefix-caching --max_num_seqs 1000 --block-size 16 --port 8007

# launch gpt-oss-120b TP mode
TIKTOKEN_RS_CACHE_DIR=/ccsopen/home/shouwei/model/hub/models--openai--gpt-oss-120b HF_HUB_OFFLINE=1 vllm serve openai/gpt-oss-120b --disable-log-requests --data-parallel-size 1 --tensor-parallel-size 8 --max-model-len 32000 --enforce-eager --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096 --no-enable-prefix-caching --max_num_seqs 1000 --block-size 16 --port 8007 --async-scheduling

# launch gpt-oss-120b DP mode
TIKTOKEN_RS_CACHE_DIR=/ccsopen/home/shouwei/model/hub/models--openai--gpt-oss-120b HF_HUB_OFFLINE=1 vllm serve openai/gpt-oss-120b --disable-log-requests --data-parallel-size 8 --tensor-parallel-size 1 --max-model-len 65000 --enforce-eager --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096 --no-enable-prefix-caching --max_num_seqs 1000 --block-size 16 --port 8007

# shift parallelism for Meta-Llama-3-70B-Instruct
ARCTIC_INFERENCE_ENABLED=1 HF_HUB_OFFLINE=1 vllm serve /ccsopen/home/shouwei/model/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f --tensor-parallel-size 2 --ulysses-sequence-parallel-size 4 --enable-shift-parallel --shift-parallel-threshold 512 --enforce-eager --gpu-memory-utilization 0.7 --port 8007 --max-num-batched-tokens 1024 --no-enable-prefix-caching

# shift parallelism for gpt-oss-120b
ARCTIC_INFERENCE_ENABLED=1 HF_HUB_OFFLINE=1 vllm serve /ccsopen/home/shouwei/model/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a --tensor-parallel-size 1 --ulysses-sequence-parallel-size 4 --enable-shift-parallel --shift-parallel-threshold 512 --enforce-eager --gpu-memory-utilization 0.7 --port 8007 --max-num-batched-tokens 1024 --no-enable-prefix-caching

HF_HUB_OFFLINE=1 vllm serve /ccsopen/home/shouwei/model/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a  --disable-log-requests --data-parallel-size 8 --tensor-parallel-size 1 --max-model-len 65000 --enforce-eager --gpu-memory-utilization 0.8 --max-num-batched-tokens 1024 --no-enable-prefix-caching --max_num_seqs 1000 --block-size 16 --port 8007
```

### Benchmarking the serving system

```bash
# benchmark the serving system

# benchmark Meta-Llama-3-70B-Instruct
# random worklaod
HF_HUB_OFFLINE=1 vllm bench serve --backend vllm --dataset-name random --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 1000 --random-input-len 2000 --random-output-len 256 --random-range-ratio 0.2 --test-real-world-workload --port 8007 --request-rate 1

# bench arctic inference
HF_HUB_OFFLINE=1 vllm bench serve --backend vllm --dataset-name random --model /ccsopen/home/shouwei/model/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f --num-prompts 1000 --random-input-len 2000 --random-output-len 256 --random-range-ratio 0.5 --test-real-world-workload --port 8007 --request-rate 1

# sharegpt workload
HF_HUB_OFFLINE=1 vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --model /ccsopen/home/shouwei/model/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f --num-prompts 2000 --test-real-world-workload --port 8007 --request-rate 1

# benchmark gpt-oss-120b
# random worklaod
HF_HUB_OFFLINE=1 vllm bench serve --backend vllm --dataset-name random --model openai/gpt-oss-120b --num-prompts 2000 --random-input-len 2000 --random-output-len 256 --random-range-ratio 0.5 --test-real-world-workload --port 8007 --request-rate 1

# sharegpt workload
HF_HUB_OFFLINE=1 vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --model openai/gpt-oss-120b --num-prompts 2000 --test-real-world-workload --port 8007 --request-rate 1

# retrive the metrics
# /ccsopen/home/shouwei/model/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
# /ccsopen/home/shouwei/model/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f

```


### benchmarking results:

#### gpt-oss-120b
dp mode: 
input :HF_HUB_OFFLINE=1 vllm bench serve --backend vllm --dataset-name random --model openai/gpt-oss-120b --num-prompts 1000 --random-input-len 8000 --random-output-len 256 --random-range-ratio 0.2 --test-real-world-workload --port 8007 --request-rate 1
{"from":"2025-12-09T21:33:14.035Z","to":"2025-12-09T21:45:40.200Z"}



```bash
vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --model openai/gpt-oss-20b --num-prompts 200 --test-real-world-workload --port 8007 --request-rate 1

vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 1000 --test-real-world-workload --port 8007 --request-rate 1

vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path /nfs/stak/users/gaosho/hpc-share/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --model openai/gpt-oss-20b --num-prompts 200 --test-real-world-workload --port 8009 --request-rate 1

vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path /nfs/stak/users/gaosho/hpc-share/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --num-prompts 1000 --test-real-world-workload --port 8000 --request-rate 1

vllm bench serve --backend vllm --dataset-name hf --dataset-path THUDM/LongBench-v2 --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 1000  --port 8006 --request-rate 1000 --longbench-length-filter short

vllm bench serve --backend vllm --dataset-name random --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 1000  --port 8007 --request-rate 1000 --random-input-len 2000 --random-output-len 256 --random-range-ratio 0.5

vllm bench serve --backend vllm --dataset-name random --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 1000 --random-input-len 2000 --random-output-len 256 --random-range-ratio 0.5 --test-real-world-workload --port 8007 --request-rate 1


# bech shift parallelism
HF_HUB_OFFLINE=1 vllm bench serve --backend vllm --dataset-name random --model /ccsopen/home/shouwei/model/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f --num-prompts 2000 --random-input-len 2000 --random-output-len 256 --random-range-ratio 0.5 --test-real-world-workload --port 8007 --request-rate 1

# /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json
# /nfs/stak/users/gaosho/hpc-share/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json
```
```bash
Use `module load gnuplot` to plot the metrics.

# salloc to get a GPU on defiant
salloc -A gen150 -J gpu_job -N 1 -t 02:00:00 -p batch-gpu
```