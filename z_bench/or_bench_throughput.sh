# model="llama-3-8b"
# model="llama-3-70b"
model="gpt-oss-20b"
# mode="ours-DP"
# mode="vanilla-DP"
mode="vanilla-TP"
TP=2
DP=1
MAX_MODEL_LEN=520000 
OUTPUT_LEN=1

# parameters list
chunk_prefill_list=(8192)
input_len_list=(1000 2000 4000 8000 16000 32000 64000 128000)
request_rate_list=(300)

export CUDA_VISIBLE_DEVICES=0,1,2,3


if [ "$model" == "llama-3-8b" ]; then
    MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
elif [ "$model" == "llama-3-70b" ]; then
    MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
elif [ "$model" == "gpt-oss-20b" ]; then
    MODEL="openai/gpt-oss-20b"
fi

RESULTS_FILE="bench_throughput-${mode}_${TP}tp_${DP}dp_random_4gpus_${model}.csv" 
OUTPUT_FILE="bench_throughput-${mode}_${TP}tp_${DP}dp_random_4gpus_${model}.txt" 

# set vllm path
# VLLM_PATH="/nfs/hpc/share/gaosho/conda_envs/arctic-inference/bin/vllm"

# server related variables
SERVER_PORT=8000
SERVER_HOST="localhost"
SERVER_PID=""

# write table header
echo "chunk_prefill,input_len,request_rate,successful_requests,benchmark_duration,request_throughput,output_token_throughput,total_token_throughput,mean_ttft,median_ttft,p99_ttft,mean_tpot,median_tpot,p99_tpot,mean_itl,median_itl,p99_itl" > $RESULTS_FILE

# start server function
start_server() {
    echo "Starting vLLM server..."
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve $MODEL \
        --disable-log-requests \
        --tensor-parallel-size $TP \
        --data-parallel-size $DP \
        --max-model-len $MAX_MODEL_LEN \
        --port $SERVER_PORT \
        --host $SERVER_HOST \
        --trust-remote-code \
        --enforce-eager \
        --max-num-seqs 100 \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --max-num-batched-tokens $1 > server.log 2>&1 &
    
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    
    # wait for server to start
    echo "Waiting for server to start..."
    sleep 120
    
    # check if server is running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Error: Server failed to start"
        cat server.log
        exit 1
    fi
    
    echo "Server is ready"
}

# stop server function
stop_server() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
        echo "Server stopped"
    fi
}

# cleanup function
cleanup() {
    stop_server
    echo "Cleanup completed"
}

# set exit to cleanup
trap cleanup EXIT

# parse benchmark result function
parse_benchmark_output() {
    local output="$1"
    local chunk_prefill="$2"
    local input_len="$3"
    local request_rate="$4"
    
    # print output for debugging (uncomment to see full output)
    echo "$output"
    
    # extract each metric - adjust according to actual output format
    successful_requests=$(echo "$output" | grep "Successful requests:" | awk '{print $3}')
    benchmark_duration=$(echo "$output" | grep "Benchmark duration (s):" | awk '{print $4}')
    request_throughput=$(echo "$output" | grep "Request throughput (req/s):" | awk '{print $4}')
    output_token_throughput=$(echo "$output" | grep "Output token throughput (tok/s):" | awk '{print $5}')
    total_token_throughput=$(echo "$output" | grep "Total Token throughput (tok/s):" | awk '{print $5}')
    
    # TTFT metrics
    mean_ttft=$(echo "$output" | grep "Mean TTFT (ms):" | awk '{print $4}')
    median_ttft=$(echo "$output" | grep "Median TTFT (ms):" | awk '{print $4}')
    p99_ttft=$(echo "$output" | grep "P99 TTFT (ms):" | awk '{print $4}')
    
    # TPOT metrics
    mean_tpot=$(echo "$output" | grep "Mean TPOT (ms):" | awk '{print $4}')
    median_tpot=$(echo "$output" | grep "Median TPOT (ms):" | awk '{print $4}')
    p99_tpot=$(echo "$output" | grep "P99 TPOT (ms):" | awk '{print $4}')
    
    # ITL metrics
    mean_itl=$(echo "$output" | grep "Mean ITL (ms):" | awk '{print $4}')
    median_itl=$(echo "$output" | grep "Median ITL (ms):" | awk '{print $4}')
    p99_itl=$(echo "$output" | grep "P99 ITL (ms):" | awk '{print $4}')
    
    # debug output
    echo "Parsed values:"
    echo "  successful_requests: $successful_requests"
    echo "  benchmark_duration: $benchmark_duration"
    echo "  request_throughput: $request_throughput"
    echo "  output_token_throughput: $output_token_throughput"
    echo "  total_token_throughput: $total_token_throughput"
    echo "  mean_ttft: $mean_ttft"
    echo "  median_ttft: $median_ttft"
    echo "  p99_ttft: $p99_ttft"
    echo "  mean_tpot: $mean_tpot"
    echo "  median_tpot: $median_tpot"
    echo "  p99_tpot: $p99_tpot"
    echo "  mean_itl: $mean_itl"
    echo "  median_itl: $median_itl"
    echo "  p99_itl: $p99_itl"
    
    # check if successful parsing
    if [ -z "$successful_requests" ]; then
        echo "Warning: Failed to parse benchmark output"
        echo "$chunk_prefill,$input_len,$request_rate,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" >> $RESULTS_FILE
        return 1
    fi
    
    # write to CSV
    echo "$chunk_prefill,$input_len,$request_rate,$successful_requests,$benchmark_duration,$request_throughput,$output_token_throughput,$total_token_throughput,$mean_ttft,$median_ttft,$p99_ttft,$mean_tpot,$median_tpot,$p99_tpot,$mean_itl,$median_itl,$p99_itl" >> $RESULTS_FILE
    return 0
}

# main loop
for chunk_prefill in "${chunk_prefill_list[@]}"; do
    echo "Starting server with chunk_prefill=$chunk_prefill"
    start_server $chunk_prefill
    
    for input_len in "${input_len_list[@]}"; do
        for request_rate in "${request_rate_list[@]}"; do
            echo "Running benchmark: input_len=$input_len, request_rate=$request_rate"
            
            # run benchmark
            output=$(vllm bench serve \
                --backend vllm \
                --model $MODEL \
                --dataset-name random \
                --random-input-len $input_len \
                --random-output-len $OUTPUT_LEN \
                --random-range-ratio 0 \
                --request-rate $request_rate \
                --num-prompts 500 \
                --host $SERVER_HOST \
                --port $SERVER_PORT 2>&1)
            
            # parse result
            parse_benchmark_output "$output" $chunk_prefill $input_len $request_rate
            
            # short break
            sleep 5
        done
    done
    
    # stop current server
    stop_server
    sleep 10
done

echo "All benchmarks done. Results saved to $RESULTS_FILE"