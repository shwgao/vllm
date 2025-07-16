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

## 5. Ray Socket Path Length Issue ✅
- **Problem:** ```OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes: /var/tmp/pbs.5513403.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/ray/session_2025-07-14_05-22-38_664188_503911/sockets/plasma_store```
- **Solution:** Set `export TMPDIR=/tmp` before running vLLM to use shorter socket paths.
- **Status:** investigating.

## 6. Docker container is not supported on Polaris ✅
- **Problem:** Polaris does not support docker container. Instead, it uses Apptainer to run container.
- **Solution:** I need to transfer Docker image of vLLM to Apptainer image, and then use it to run the cluster.
- **Status:** solved.

## 7. Apptainer build issue ✅
- **Problem:** ```ERROR  : Installation issue: starter-suid doesn't have setuid bit set```
- **Solution:** Always use the --fakeroot flag on Polaris compute nodes.
- **Status:** solved.

## 8. Ray didn't start while using run_cluster_apptainer.sh ⏳
- **Problem:** Ray didn't start while using run_cluster_apptainer.sh.
- **Solution:** I start the apptainer shell, and then start the Ray cluster.
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

### set up the cluster
```bash
# 1. get the node info
cat $PBS_NODEFILE

# 2. get the ip address of the nodes
getent hosts `hostname`

# 3. ssh the other nodes
ssh x3006c0s19b0n0

# 4. start the head node
bash examples/online_serving/run_cluster_apptainer.sh  vllm/vllm-openai 10.201.2.10 --head /home/shouwei/projects/hf_models --env VLLM_HOST_IP=10.201.2.10

# 5. start the worker nodes
bash examples/online_serving/run_cluster_apptainer.sh  vllm/vllm-openai 10.201.2.10 --worker /home/shouwei/projects/hf_models --env VLLM_HOST_IP=10.201.1.237

# 6. check the instance list
ml use /soft/modulefiles && ml spack-pe-base/0.8.1 && ml use /soft/spack/testing/0.8.1/modulefiles && ml apptainer/main && apptainer instance list

# 7. enter the instance
apptainer shell "instance://${INSTANCE_NAME}"

# 8. check the Ray cluster
ray status # check the problem 8
    # if the Ray cluster is not started, you can start it by:
    ray start --block --disable-usage-stats --temp-dir=/tmp/ray --head --port=6379 --node-ip-address=${HEAD_NODE_ADDRESS}
    ray start --block --disable-usage-stats --temp-dir=/tmp/ray --address=${HEAD_NODE_ADDRESS}:6379


# 9. start to serve the model
vllm serve gradientai/Llama-3-8B-Instruct-Gradient-1048k --swap-space 16 --disable-log-requests --tensor_parallel_size 4 --max_model_len 65000 --pipeline-parallel-size 2

# 10. stop the instance
apptainer instance stop vllm-openai-instance

# 11. start the instance
apptainer instance start vllm-openai-instance

# 12. check the Ray cluster
ray status
```