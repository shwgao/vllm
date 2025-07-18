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


# --- 4. Launch Ray Cluster ---
mkdir -p $RAY_TMPDIR
echo "Starting Ray head node..."
ray start --head --node-ip-address="$head_node_ip" --port=6379 --disable-usage-stats --temp-dir="$RAY_TMPDIR" &
sleep 10

# Launch worker nodes
for node in $nodes; do
    if [ "$node" != "$head_node_name" ]; then
        echo "Starting Ray worker on $node..."
        # Pass all critical environment variables to the remote worker
        ssh $node 'export OMP_NUM_THREADS=1; \
                   export OPENBLAS_NUM_THREADS=1; \
                   export MKL_NUM_THREADS=1; \
                   export GLOO_NUM_THREADS=1; \
                   export NCCL_SOCKET_IFNAME=hsn0; \
                   export NCCL_DEBUG=INFO; \
                   export VLLM_HOST_IP=$(hostname -i | cut -d" " -f1); \
                   source ~/projects/vllm/.venv/bin/activate; \
                   mkdir -p '"$RAY_TMPDIR"'; \
                   ray start --address='"$head_node_ip:6379"' --disable-usage-stats' &
    fi
done
sleep 15


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

vllm bench throughput \
    --model gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --max-model-len 65000 \
    --gpu-memory-utilization 0.9 \
    --num-prompts 100 \
    --input-len 32000 \
    --output-len 256 \
    --trust-remote-code

# --- 6. Clean up ---
ray stop

echo "Job finished."