#!/bin/bash

MODEL="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
TP=1
PP=4
MAX_MODEL_LEN=92000  # 取最大input-len+output-len的安全值
OUTPUT_LEN=128
RESULTS_FILE="benchmark_results_1tp_4pp.csv"

# 参数列表
num_prompts_list=(1 2 4 6 8)
input_len_list=(4000 8000 16000 32000 64000 75000 86000)
# num_prompts_list=(1)
# input_len_list=(4000)

# 写表头
echo "num_prompts,input_len,throughput_requests,throughput_total_tokens,throughput_output_tokens" > $RESULTS_FILE

export CUDA_VISIBLE_DEVICES=0,1

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
    # echo "$output"

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