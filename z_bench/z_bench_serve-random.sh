MODEL="gradientai/Llama-3-8B-Instruct-Gradient-1048k" 
TP=1
DP=2
MAX_MODEL_LEN=122000 
OUTPUT_LEN=128 
RESULTS_FILE="benchmark_serve_1tp_1sp_2dp_random_2gpus.csv" 
OUTPUT_FILE="benchmark_serve_1tp_1sp_2dp_random_2gpus.txt" 

# # 设置vllm路径
# VLLM_PATH="/nfs/hpc/share/gaosho/conda_envs/arctic-inference/bin/vllm"

# 参数列表
chunk_prefill_list=(8192)
input_len_list=(60000)
request_rate_list=(1 2)

# 服务器相关变量
SERVER_PORT=8000
SERVER_HOST="localhost"
SERVER_PID=""

# 写表头
echo "chunk_prefill,input_len,request_rate,successful_requests,benchmark_duration,request_throughput,output_token_throughput,total_token_throughput,mean_ttft,median_ttft,p99_ttft,mean_tpot,median_tpot,p99_tpot,mean_itl,median_itl,p99_itl" > $RESULTS_FILE

export CUDA_VISIBLE_DEVICES=0,1

# 启动服务器的函数
start_server() {
    echo "Starting vLLM server..."
    vllm serve $MODEL \
        --disable-log-requests \
        --tensor-parallel-size $TP \
        --data-parallel-size $DP \
        --max-model-len $MAX_MODEL_LEN \
        --port $SERVER_PORT \
        --host $SERVER_HOST \
        --trust-remote-code \
        --enforce-eager \
        --max-num-batched-tokens $1 > server.log 2>&1 &
    
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    
    # 等待服务器启动
    echo "Waiting for server to start..."
    sleep 120
    
    # 检查服务器是否正在运行
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Error: Server failed to start"
        cat server.log
        exit 1
    fi
    
    echo "Server is ready"
}

# 停止服务器的函数
stop_server() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
        echo "Server stopped"
    fi
}

# 清理函数
cleanup() {
    stop_server
    echo "Cleanup completed"
}

# 设置退出时清理
trap cleanup EXIT

# 解析benchmark结果的函数
parse_benchmark_output() {
    local output="$1"
    local chunk_prefill="$2"
    local input_len="$3"
    local request_rate="$4"
    
    # 打印输出调试（取消注释以查看完整输出）
    echo "$output"
    
    # 提取各个指标 - 根据实际输出格式调整
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
    
    # 调试输出
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
    
    # 检查是否成功解析
    if [ -z "$successful_requests" ]; then
        echo "Warning: Failed to parse benchmark output"
        echo "$chunk_prefill,$input_len,$request_rate,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" >> $RESULTS_FILE
        return 1
    fi
    
    # 写入CSV
    echo "$chunk_prefill,$input_len,$request_rate,$successful_requests,$benchmark_duration,$request_throughput,$output_token_throughput,$total_token_throughput,$mean_ttft,$median_ttft,$p99_ttft,$mean_tpot,$median_tpot,$p99_tpot,$mean_itl,$median_itl,$p99_itl" >> $RESULTS_FILE
    return 0
}

# 主循环
for chunk_prefill in "${chunk_prefill_list[@]}"; do
    echo "Starting server with chunk_prefill=$chunk_prefill"
    start_server $chunk_prefill
    
    for input_len in "${input_len_list[@]}"; do
        for request_rate in "${request_rate_list[@]}"; do
            echo "Running benchmark: input_len=$input_len, request_rate=$request_rate"
            
            # 运行benchmark
            output=$(vllm bench serve \
                --backend vllm \
                --model $MODEL \
                --dataset-name random \
                --dataset-path None \
                --random-input-len $input_len \
                --random-output-len $OUTPUT_LEN \
                --random-range-ratio 0.8 \
                --request-rate $request_rate \
                --num-prompts 300 \
                --host $SERVER_HOST \
                --port $SERVER_PORT 2>&1)
            
            # 解析结果
            parse_benchmark_output "$output" $chunk_prefill $input_len $request_rate
            
            # 短暂休息
            sleep 5
        done
    done
    
    # 停止当前服务器
    stop_server
    sleep 10
done

echo "All benchmarks done. Results saved to $RESULTS_FILE"