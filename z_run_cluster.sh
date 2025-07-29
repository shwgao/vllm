#!/bin/bash
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A Picom_AI_Accelerator
#PBS -k oed

# --- 1. Environment Setup ---
cd $PBS_O_WORKDIR
source ~/projects/vllm/.venv/bin/activate
export HF_HOME=/lus/eagle/projects/Picom_AI_Accelerator/projects/hf_models

node_num=$(sort -u "$PBS_NODEFILE" | wc -l)

# FIX: Limit CPU threads for stability in HPC environments
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GLOO_NUM_THREADS=1


# --- 2. Network & Ray Configuration ---
# **FINAL FIX**: Configure networking for PyTorch Distributed and NCCL
# This is critical on HPC systems with specialized high-speed networks.
# Find the high-speed network interface (e.g., hsn0, hsn1...).
# We assume 'hsn0' here; you may need to verify this on the node.
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=INFO


# Set Ray temp directory on the node-local /tmp filesystem
export NUMERIC_JOB_ID=${PBS_JOBID%%.*}
export RAY_TMPDIR="/tmp/ray-session-$USER-$NUMERIC_JOB_ID"


# --- 3. Node Information ---
nodes=$(cat $PBS_NODEFILE)
head_node_name=$(hostname)
head_node_ip=$(hostname -i | cut -d' ' -f1)
# Each node should have its own unique IP for vLLM
export VLLM_HOST_IP="$head_node_ip"

echo "--- Setup Information ---"
echo "Job ID: $PBS_JOBID (Numeric: $NUMERIC_JOB_ID)"
echo "Head Node: $head_node_name (IP: $head_node_ip)"
echo "Ray Temp Dir: $RAY_TMPDIR"
echo "NCCL Interface: $NCCL_SOCKET_IFNAME"
echo "-------------------------"

export CUDA_VISIBLE_DEVICES=0,1

# --- 4. Launch Ray Cluster ---
mkdir -p $RAY_TMPDIR
echo "Starting Ray head node..."
ray start --head --node-ip-address="$head_node_ip" --port=6379 --disable-usage-stats --temp-dir="$RAY_TMPDIR" &
sleep 20

# Launch worker nodes
i=0
for node in $nodes; do
    # i=$((i+1))
    # if [ $i -gt 1 ]; then
    #     break
    # fi
    if [ "$node" != "$head_node_name" ]; then
        echo "Starting Ray worker on $node..."
        # Pass all critical environment variables to the remote worker
        ssh $node 'export OMP_NUM_THREADS=1; \
                   export OPENBLAS_NUM_THREADS=1; \
                   export MKL_NUM_THREADS=1; \
                   export GLOO_NUM_THREADS=1; \
                   export NCCL_SOCKET_IFNAME=hsn0; \
                   export NCCL_DEBUG=INFO; \
                   export CUDA_VISIBLE_DEVICES=0,1; \
                   export VLLM_HOST_IP=$(hostname -i | cut -d" " -f1); \
                   source ~/projects/vllm/.venv/bin/activate; \
                   mkdir -p '"$RAY_TMPDIR"'; \
                   ray start --address='"$head_node_ip:6379"' --disable-usage-stats' &
    fi
done
sleep 20


# --- 5. Launch vLLM Service ---
echo "Verifying Ray cluster status..."
ray status

export RAY_ADDRESS="$head_node_ip:6379"
echo "RAY_ADDRESS set to: $RAY_ADDRESS"

echo "Starting vLLM service..."
# # python -m vllm.entrypoints.openai.api_server \
# vllm serve \
#     --model gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#     --tensor-parallel-size 4 \
#     --pipeline-parallel-size 2 \
#     --max-model-len 65000 \
#     --gpu-memory-utilization 0.95 \
#     --request-rate 1 \
#     --enforce-eager \
#     --host 0.0.0.0 \
#     --port 8000

# python3 -m vllm.entrypoints.openai.api_server \
#   --model "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1" \
#   --trust-remote-code \
#   --seed=1 \
#   --host="0.0.0.0" \
#   --port=5000 \
#   --served-model-name "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1" \
#   --tensor-parallel-size=4 \
#   --pipeline-parallel-size=$node_num \
#   --max-model-len=32768 \
#   --gpu-memory-utilization 0.95 \
#   --enforce-eager

# vllm bench throughput \
#     --model nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 \
#     --tensor-parallel-size 4 \
#     --max-model-len 2000 \
#     --gpu-memory-utilization 0.9 \
#     --num-prompts 10 \
#     --input-len 1200 \
#     --output-len 256 \
#     --trust-remote-code \
#     --pipeline-parallel-size $node_num \
#     # --enforce-eager \

# vllm bench throughput \
#     --model gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#     --tensor-parallel-size 1 \
#     --max-model-len 65000 \
#     --gpu-memory-utilization 0.95 \
#     --num-prompts 1 \
#     --input-len 1000 \
#     --output-len 256 \
#     --trust-remote-code \
#     --pipeline-parallel-size $node_num \


MODEL="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
TP=2
PP=2
MAX_MODEL_LEN=92000  # 取最大input-len+output-len的安全值
OUTPUT_LEN=128
RESULTS_FILE="benchmark_results_2tp_2pp.csv"

# 参数列表
num_prompts_list=(1 2 4 6 8)
input_len_list=(4000 8000 16000 32000 64000 75000 86000)
# num_prompts_list=(1)
# input_len_list=(4000)

# 写表头
echo "num_prompts,input_len,throughput_requests,throughput_total_tokens,throughput_output_tokens" > $RESULTS_FILE


for num_prompts in "${num_prompts_list[@]}"; do
  for input_len in "${input_len_list[@]}"; do
    echo "Running: num_prompts=$num_prompts, input_len=$input_len"
    output=$(vllm bench throughput \
      --model $MODEL \
      --tensor-parallel-size $TP \
      --pipeline-parallel-size $PP \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization 0.9 \
      --num-prompts $num_prompts \
      --input-len $input_len \
      --output-len $OUTPUT_LEN \
      --trust-remote-code 2>&1)

    # 打印输出调试（取消注释以查看完整输出）
    echo "$output"

    # 提取 throughput 行
    line=$(echo "$output" | grep -i "Throughput")
    if [ -z "$line" ]; then
      echo "$num_prompts,$input_len,N/A,N/A,N/A" >> $RESULTS_FILE
      continue
    fi

    # 更精确的解析方法
    # 示例: Throughput: 0.26 requests/s, 16951.36 total tokens/s, 33.84 output tokens/s
    reqs=$(echo "$line" | sed -n 's/.*Throughput: \([0-9.]*\) requests\/s.*/\1/p')
    toks=$(echo "$line" | sed -n 's/.*requests\/s, \([0-9.]*\) total tokens\/s.*/\1/p')
    out_toks=$(echo "$line" | sed -n 's/.*total tokens\/s, \([0-9.]*\) output tokens\/s.*/\1/p')

    # 如果 sed 失败，尝试 awk 方法
    if [ -z "$reqs" ] || [ -z "$toks" ] || [ -z "$out_toks" ]; then
      echo "Warning: sed parsing failed, trying awk method for line: $line"
      reqs=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="Throughput:") print $(i+1)}')
      toks=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="requests/s,") print $(i+1)}')
      out_toks=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="tokens/s,") print $(i+1)}' | tail -1)
    fi

    echo "$num_prompts,$input_len,$reqs,$toks,$out_toks" >> $RESULTS_FILE
  done
done

echo "All benchmarks done. Results saved to $RESULTS_FILE"


# sleep 1000000

# --- 6. Clean up ---
ray stop

echo "Job finished."