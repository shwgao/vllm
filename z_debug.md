# Debug Log: vLLM Issues

## 1. Safetensor Loading Performance Issue ⏳
- **Problem:** Safetensor checkpoint loading shows exponential slowdown - from 1.8s per shard initially to 342s per shard by the third shard (7 total shards).
- **Solution:** Need to investigate memory pressure, disk I/O, and network storage latency on HPC system.
- **Status:** investigating.

## 2. Can't see cuda: ✅
- **Problem:** ```RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":52, please report a bug to PyTorch. device=1, num_gpus=1```
- **Solution:** `export CUDA_VISIBLE_DEVICES=0` when using H100 cluster.
- **Status:** solved.

## 3. Environment Issue ✅
- **Problem:** some how mix up with the OmniServe environment.
- **Solution:** `source vllm-env/bin/activate`
- **Status:** solved.

## 4. 
- **Problem:** 
- **Solution:** 
- **Status:** investigating.


# Using command line
### model saving path
```bash
echo $HF_HOME
# /nfs/stak/users/gaosho/hpc-share/projects/llm/models
```

```bash
# 1. benchmarks serving
# server
vllm serve gradientai/Llama-3-8B-Instruct-Gradient-1048k --swap-space 16 --disable-log-requests --tensor_parallel_size 1 --max_model_len 65000
# client
python benchmarks/benchmark_serving.py     --backend vllm     --model gradientai/Llama-3-8B-Instruct-Gradient-1048k     --dataset-name random     --dataset-path None     --random_input_len 32000 --random_output_len 128    --request-rate 1     --num-prompts 20

```

