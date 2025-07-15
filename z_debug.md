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

## 4. Environment setup on Polaris ✅
- **Problem:** when I just build vllm from source on Polaris login node, it encounters an error `exit code -9`, which may mean out of memory.
- **Solution:** I request a node, and build vllm from source on the node.
- **Status:** solved.

## 5. Ray Socket Path Length Issue ⏳
- **Problem:** ```OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes: /var/tmp/pbs.5513403.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/ray/session_2025-07-14_05-22-38_664188_503911/sockets/plasma_store```
- **Solution:** Set `export TMPDIR=/tmp` before running vLLM to use shorter socket paths.
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

# 2. benchmark throughput
vllm bench throughput --model nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1 --dataset-name random --max-model-len 65000 --tensor-parallel-size 1 --pipeline-parallel-size 4 --num-prompts 20 --input-len 32000 --output-len 128 --trust-remote-code

# 3. Fixed command for Nemotron Ultra (with TMPDIR fix for Ray socket path issue)
export TMPDIR=/tmp
python3 -m vllm.entrypoints.openai.api_server --model "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1" --trust-remote-code --seed=1 --host="0.0.0.0" --port=5000 --served-model-name "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1" --tensor-parallel-size=8 --max-model-len=32768 --gpu-memory-utilization 0.95 --enforce-eager

```

```bash
# 2. request debug node on Polaris
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A Picom_AI_Accelerator
```
