model="llama-3-8b"
model="llama-3-70b"
model="gpt-oss-20b"
# mode="ours-DP"
mode="vanilla-DP"
# mode="vanilla-TP"
TP=1
DP=1
MAX_MODEL_LEN=520000 
OUTPUT_LEN=2

# parameters list
chunk_prefill_list=(8192)
input_len_list=(1000 2000 4000 8000 16000 32000 64000 128000)
batch_size_list=(1 4 16)

export CUDA_VISIBLE_DEVICES=0,1,2,3


if [ "$model" == "llama-3-8b" ]; then
    MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
elif [ "$model" == "llama-3-70b" ]; then
    MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
elif [ "$model" == "gpt-oss-20b" ]; then
    MODEL="openai/gpt-oss-20b"
fi

RESULTS_FILE="bench_latency-${mode}_${TP}tp_${DP}dp_random_4gpus_${model}.csv" 
OUTPUT_FILE="bench_latency-${mode}_${TP}tp_${DP}dp_random_4gpus_${model}.txt" 

# set vllm path
# VLLM_PATH="/nfs/hpc/share/gaosho/conda_envs/arctic-inference/bin/vllm"

# server related variables
SERVER_PORT=8000
SERVER_HOST="localhost"
SERVER_PID=""

# write table header
echo "chunk_prefill,input_len,batch_size,avg_latency,p10_latency,p25_latency,p50_latency,p75_latency,p90_latency,p99_latency" > $RESULTS_FILE


for chunk_prefill in "${chunk_prefill_list[@]}"; do
  for input_len in "${input_len_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
      echo "Running: chunk_prefill=$chunk_prefill, input_len=$input_len, batch_size=$batch_size"
      
      output=$(VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm bench latency \
        --model $MODEL \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size $TP \
        --data-parallel-size $DP \
        --input-len $input_len \
        --output-len $OUTPUT_LEN \
        --trust-remote-code \
        --enforce-eager \
        --load-format dummy \
        --batch-size $batch_size \
        --num-iters-warmup 10\
        --num-iters 100 2>&1)

      # print output for debugging (uncomment to see full output)
      echo "$output"

      # extract latency metrics
      avg_latency=$(echo "$output" | grep "Avg latency:" | sed 's/.*Avg latency: \([0-9.]*\) seconds.*/\1/')
      p10_latency=$(echo "$output" | grep "10% percentile latency:" | sed 's/.*10% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p25_latency=$(echo "$output" | grep "25% percentile latency:" | sed 's/.*25% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p50_latency=$(echo "$output" | grep "50% percentile latency:" | sed 's/.*50% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p75_latency=$(echo "$output" | grep "75% percentile latency:" | sed 's/.*75% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p90_latency=$(echo "$output" | grep "90% percentile latency:" | sed 's/.*90% percentile latency: \([0-9.]*\) seconds.*/\1/')
      p99_latency=$(echo "$output" | grep "99% percentile latency:" | sed 's/.*99% percentile latency: \([0-9.]*\) seconds.*/\1/')

      # check if successful to extract latency data
      if [ -z "$avg_latency" ] || [ -z "$p10_latency" ] || [ -z "$p25_latency" ] || [ -z "$p50_latency" ] || [ -z "$p75_latency" ] || [ -z "$p90_latency" ] || [ -z "$p99_latency" ]; then
        echo "Warning: Failed to extract latency data, setting to N/A"
        avg_latency="N/A"
        p10_latency="N/A"
        p25_latency="N/A"
        p50_latency="N/A"
        p75_latency="N/A"
        p90_latency="N/A"
        p99_latency="N/A"
      fi

      echo "$chunk_prefill,$input_len,$batch_size,$avg_latency,$p10_latency,$p25_latency,$p50_latency,$p75_latency,$p90_latency,$p99_latency" >> $RESULTS_FILE
    done
  done
done

echo "All benchmarks done. Results saved to $RESULTS_FILE" 
