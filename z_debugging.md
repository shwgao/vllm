# vLLM Benchmark 代理问题调试

## 问题描述
在使用 `vllm bench serve` 命令时，服务一直显示 "Waiting for endpoint to become up" 并无限等待，无法正常进行基准测试。

## 问题原因
系统环境设置了代理（`http_proxy`, `https_proxy`, `ftp_proxy`），导致 vLLM benchmark 工具对 localhost 的请求也被代理服务器处理，造成连接失败。

## 解决方案（不修改源码）

### 方法1：使用 no_proxy 环境变量（推荐）

```bash
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
vllm bench serve --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

**优点：**
- ✅ 不需要修改源码
- ✅ 保持代理设置，其他网络请求仍使用代理
- ✅ 只对 localhost 禁用代理，精确解决问题
- ✅ 简单易用

### 方法2：创建包装脚本

创建 `run_benchmark.sh`：
```bash
#!/bin/bash
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
cd /ccsopen/home/shouwei/projects/vllm-newest/vllm
source .venv/bin/activate
vllm bench serve "$@"
```

使用方法：
```bash
chmod +x run_benchmark.sh
./run_benchmark.sh --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

### 方法3：使用 alias

在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
alias vllm-bench='no_proxy="localhost,127.0.0.1,0.0.0.0" NO_PROXY="localhost,127.0.0.1,0.0.0.0" vllm bench'
```

使用方法：
```bash
vllm-bench serve --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

### 方法4：使用环境变量文件

创建 `.env` 文件：
```bash
no_proxy=localhost,127.0.0.1,0.0.0.0
NO_PROXY=localhost,127.0.0.1,0.0.0.0
```

使用方法：
```bash
source .env && vllm bench serve --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

## 推荐使用方法1

**方法1（no_proxy 环境变量）**是最优雅的解决方案，因为：
- 不需要修改源码
- 保持代理设置
- 只对 localhost 禁用代理
- 简单易用
- 兼容性好

## 网络连接问题：无法访问 HuggingFace

### 问题描述
当系统无法连接到 `huggingface.co` 时（例如 HPC 系统网络限制），会出现以下错误：
```
Failed to establish a new connection: [Errno 101] Network is unreachable
```

### 解决方案：使用本地模型和 Tokenizer

#### 方法1：设置离线模式（推荐）

如果模型已经下载到本地（通常在 `~/.cache/huggingface/hub/` 或指定的缓存目录），可以设置 `HF_HUB_OFFLINE` 环境变量：

```bash
export HF_HUB_OFFLINE=1
vllm bench serve --model /path/to/local/model --tokenizer /path/to/local/tokenizer --dataset-name sharegpt --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 200 --test-real-world-workload --port 8000
```

或者如果模型在 HuggingFace 缓存目录中，可以使用模型ID（vLLM会自动查找本地缓存）：

```bash
export HF_HUB_OFFLINE=1
vllm bench serve --model openai/gpt-oss-20b --dataset-name sharegpt --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 200 --test-real-world-workload --port 8000
```

#### 方法2：直接指定本地路径

如果模型已经下载到本地目录，可以直接使用本地路径：

```bash
# 使用本地模型路径
vllm bench serve \
  --model /path/to/local/model \
  --tokenizer /path/to/local/tokenizer \
  --dataset-name sharegpt \
  --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 \
  --test-real-world-workload \
  --port 8000
```

#### 方法3：指定 HuggingFace 缓存目录

如果模型在自定义缓存目录中：

```bash
export HF_HOME=/path/to/huggingface/cache
export HF_HUB_OFFLINE=1
vllm bench serve --model openai/gpt-oss-20b --dataset-name sharegpt --dataset-path /ccsopen/home/shouwei/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 200 --test-real-world-workload --port 8000
```

### 查找本地模型路径

如果模型已经下载过，可以通过以下方式查找：

```bash
# 查找 HuggingFace 缓存目录
ls -la ~/.cache/huggingface/hub/models--*/snapshots/*/

# 或者使用 find 命令
find ~/.cache/huggingface -name "tokenizer_config.json" -path "*/openai/gpt-oss-20b/*" 2>/dev/null
```

## 注意事项

1. 确保 vLLM API 服务器正在运行（端口 8000）
2. 如果使用不同的主机或端口，需要相应调整 `no_proxy` 设置
3. 某些网络环境可能需要额外的代理配置
4. 使用本地模型时，确保模型文件完整（包括 `tokenizer_config.json`, `tokenizer.json` 等）
5. 如果模型路径中包含特殊字符（如 `--`），vLLM 会自动处理模型ID到本地路径的转换
